"""Post-recall relevance gate: one System One pass that drops memories below a keep threshold.

Runs client-side, after the server's 4096-token cut and after the session journal. The
daemon's ranking is untouched — this step only filters, it never reorders. A memory is kept
when the decision model says showing it helps answer the query; a verdict below the absolute
floor, or below ``min_noul_of_best`` of the pool's best, drops the item. The floor is
absolute: when nothing clears it the block empties and the turn runs without memories.

Every failure path fails OPEN — a timeout, a refused answer, a malformed response leaves the
recall exactly as the journal passed it. The gate must never cost memory it cannot grade, and
it never costs the injection itself: run inside the prefetch thread, it stays off the reply
path, with a socket timeout well inside the join window that consumes the prefetched block.

This gate is the plugin's own, independent of Hindsight's server-side recall prune: it calls
its own System One endpoint and model, configured by the plugin's ``recall_prune_*`` settings,
and may run a different model than the server does. The question wording and default floors
started from what the server's prune measured (``decision_rerank_interface.py``), but they are
this plugin's defaults now; neither side has to follow the other's changes.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

#: The best-calibrated of four wordings measured for the server's prune (lifts the wanted
#: candidate to 0.74 vs 0.59 for "contains information needed").
KEEP_INSTRUCTIONS = (
    "Showing this memory to someone answering the question would help them, rather than distract them."
)

PRUNE_BELOW = 0.2  # absolute floor
PRUNE_BELOW_BEST = 0.4  # relative floor: below this share of the best answer
MAX_ASK_DEFAULT = 80  # verdicts bought per recall; a full 4096 block is ~55 items
MAX_MEMORY_CHARS = 16_000  # per-memory cap (≈4k tokens, the daemon's MAX_MEMORY_TOKENS)
MAX_STATE_CHARS = 8_000  # query cap inside the state (the daemon caps at 2k tokens)
#: Every setting of this gate is a flat plugin-config key under this prefix.
CONFIG_PREFIX = "recall_prune_"


class DecisionPruner:
    """Asks System One keep-questions over a recall block and drops the losers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float | None = None,
        max_gate_ms: float = 2200.0,
        min_noul: float = PRUNE_BELOW,
        min_noul_of_best: float = PRUNE_BELOW_BEST,
        max_ask: int = MAX_ASK_DEFAULT,
        min_results: int = 3,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key = api_key
        self.model = model
        # The wall clock cap is the operative bound (it protects the prefetch thread); the
        # socket timeout only reaps the orphan thread after a late verdict, so it must sit
        # ABOVE the cap: a healthy 60-item batch routinely takes 1.5-2s on a local endpoint.
        self.timeout_s = timeout_s if timeout_s else max_gate_ms / 1000.0 + 1.0
        # Wall-clock cap for the whole grading step. urlopen's timeout bounds socket
        # inactivity only: a decision endpoint stalling behind other GPU work can hold the
        # call open far longer, and this step runs on the prefetch thread whose result the
        # next turn joins for ~3s. Past the cap the block ships unpruned; verdicts are never
        # worth a lost injection.
        self.max_gate_ms = max_gate_ms
        self.min_noul = min_noul
        self.min_noul_of_best = min_noul_of_best
        self.max_ask = max(1, int(max_ask))
        self.min_results = max(2, int(min_results))

    @classmethod
    def from_config(cls, cfg: dict, api_key: str) -> DecisionPruner:
        """The plugin config's flat ``recall_prune_*`` keys (*api_key* resolved by the caller,
        it is a secret) -> pruner. Raises ValueError when unusable."""
        base_url = str(cfg.get(f"{CONFIG_PREFIX}base_url") or "").strip()
        api_key = str(api_key or "").strip()
        model = str(cfg.get(f"{CONFIG_PREFIX}model") or "").strip()
        if not (base_url and api_key and model):
            raise ValueError(
                f"recall pruning needs {CONFIG_PREFIX}base_url, {CONFIG_PREFIX}model and an API key"
            )

        def _num(name: str, default: float) -> float:
            try:
                value = float(cfg.get(f"{CONFIG_PREFIX}{name}", default))
            except (TypeError, ValueError):
                return default
            return value if value > 0 else default

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=_num("timeout_s", 0) or None,
            max_gate_ms=_num("max_gate_ms", 2200.0),
            min_noul=_num("min_noul", PRUNE_BELOW),
            min_noul_of_best=_num("min_noul_of_best", PRUNE_BELOW_BEST),
            max_ask=int(_num("max_ask", MAX_ASK_DEFAULT)),
            min_results=int(_num("min_results", 3)),
        )

    # -- wire format -------------------------------------------------------------

    def _ask(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        """One System One request -> {question key: noul}. Raises on any failure."""
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                key: {"type": "noul", "instructions": {"task": KEEP_INSTRUCTIONS, "memory": memory}}
                for key, memory in questions.items()
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode())
        answers: dict[str, float] = {}
        for key, answer in (payload.get("answers") or {}).items():
            noul = (answer or {}).get("noul")
            if isinstance(noul, (int, float)):
                answers[key] = max(0.0, min(1.0, float(noul)))
        if not answers:
            raise ValueError("decision model returned no noul answers")
        return answers

    def _ask_oversized(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        """``_ask``, retrying an over-window request in two halves (the daemon's shape).

        The endpoint's window is counted in the model's own tokens, which a client can only
        estimate; a 400 mentioning max_tokens_exceeded means ask less, not ask again.
        """
        try:
            return self._ask(state, questions)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="replace")
            except Exception:
                pass
            if exc.code != 400 or "max_tokens_exceeded" not in detail or len(questions) < 2:
                raise
        items = list(questions.items())
        half = len(items) // 2
        answers = self._ask_oversized(state, dict(items[:half]))
        answers.update(self._ask_oversized(state, dict(items[half:])))
        return answers

    # -- the gate ----------------------------------------------------------------

    def prune(self, query: str, results: Sequence[Any], *, current_date: str | None = None) -> tuple[list, list]:
        """Split ``results`` (recall-ranked) into ``(kept, dropped)``.

        Fails open: any error returns ``(list(results), [])``. The first ``max_ask`` items
        are graded; items beyond it stay (they are the rarest of the rare tail of a cut that
        is already capped). A result without ``.text`` is never asked and never dropped.
        """
        asked = [r for r in results if getattr(r, "text", None)][: self.max_ask]
        if len(asked) < self.min_results:
            return list(results), []
        state = f"Current date: {current_date}\n\nQuestion: " if current_date else "Question: "
        state += (query or "")[:MAX_STATE_CHARS]
        questions = {f"m{index}": r.text[:MAX_MEMORY_CHARS] for index, r in enumerate(asked)}
        box: dict[str, object] = {}

        def _grade() -> None:
            try:
                box["answers"] = self._ask_oversized(state, questions)
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=_grade, name="hindsight-prune", daemon=True)
        worker.start()
        worker.join(self.max_gate_ms / 1000.0)
        if worker.is_alive():
            logger.warning(
                "decision prune skipped this turn (gate exceeded %.0f ms; block ships unpruned)",
                self.max_gate_ms,
            )
            return list(results), []
        if "answers" not in box:
            logger.warning("decision prune skipped this turn (%s)", box.get("error"))
            return list(results), []
        answers: dict[str, float] = box.get("answers") or {}  # type: ignore[assignment]
        nouls = [answers.get(f"m{index}") for index in range(len(asked))]
        graded = [n for n in nouls if n is not None]
        if not graded:
            return list(results), []
        floor = max(self.min_noul, self.min_noul_of_best * max(graded))
        kept: list = []
        dropped: list = []
        for index, result in enumerate(results):
            noul = nouls[index] if index < len(nouls) else None
            if index >= len(asked) or noul is None or noul >= floor:
                kept.append(result)
            else:
                dropped.append(result)
        if dropped:
            logger.debug("decision prune: kept %d, dropped %d (floor %.3f)", len(kept), len(dropped), floor)
        return kept, dropped
