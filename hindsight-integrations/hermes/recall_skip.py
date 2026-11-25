"""Pre-recall cue gate: one System One question decides whether a message is worth a recall.

Recall works like associative memory: it needs a cue. A message that, taken alone, names
nothing concrete ("ok", "go ahead", "you didn't answer", "the second one") gives the search
nothing to match, so the daemon returns its usual 30-50 nearest neighbours and the relevance
gate keeps several of them anyway: junk in the context window on exactly the turns that
need memory least. This gate asks the decision model one ``noul`` question about the user's
message alone, *before* the search, and skips the recall when the message has no cue.

The question is the ``b1_cue_a`` wording, chosen over a dozen alternatives on 192 labelled
prompts from real history (AUC 0.92 against "the recall found nothing useful"). Asked alone,
as here, at the default floor (0.55) it skipped 16.5% of 1855 history turns and cut the
searches on cue-less messages from 78 to 22 of 192, at 5 lost recalls of messages that did
name something. The floor is calibrated for that measurement's model (d1-2B); another model
needs its own.

Fails OPEN: an unconfigured, unreachable, slow or refusing endpoint searches as usual. It
shares the relevance gate's endpoint, model and key (``recall_prune_*``).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .decision_prune import CONFIG_PREFIX as PRUNE_PREFIX
from .decision_prune import ask_nouls

logger = logging.getLogger(__name__)

NO_CUE_INSTRUCTIONS = """Taken on its own, without the earlier conversation, this message contains nothing specific to remember things by: it names no person, term, program, model, machine, file, place, event or other concrete subject.

This is the case for:
- a reaction: ok, thanks, done, great, or swearing;
- a command that only steers the current work: continue, stop, go ahead, do it, repeat, run it, check again;
- a remark about the conversation itself: you did not answer, explain what you are doing now, I already told you;
- gibberish or a typo;
- a fragment whose meaning rests entirely on the previous message: the second one, why?, what about it?
A message that names anything concrete is not such a case, even when it is a command."""

SKIP_AT = 0.55  # noul at or above which the recall is skipped (d1-2B calibration)
MAX_GATE_MS = 1000.0  # one short question; ~50 ms on a local endpoint
MAX_MESSAGE_CHARS = 4000  # the measured state carried the message up to this length
CONFIG_PREFIX = "recall_skip_"


def message_state(message: str) -> str:
    """The System One state, exactly as measured."""
    return f"Message from the user to their AI assistant:\n<<<\n{message[:MAX_MESSAGE_CHARS]}\n>>>"


@dataclass(frozen=True)
class CueVerdict:
    """One gate pass. ``noul`` is the probability the message has no cue (None when the gate
    could not answer, then ``error`` says why and ``skip`` is False: fail open)."""

    skip: bool
    noul: float | None
    elapsed_ms: float
    error: str | None = None


class CueGate:
    """Asks whether a user message has any concrete cue to recall by."""

    def __init__(
        self, *, base_url: str, api_key: str, model: str, min_noul: float = SKIP_AT, max_gate_ms: float = MAX_GATE_MS
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key = api_key
        self.model = model
        self.min_noul = min_noul
        self.max_gate_ms = max_gate_ms

    @classmethod
    def from_config(cls, cfg: dict, api_key: str) -> CueGate:
        """``recall_prune_*`` endpoint keys plus ``recall_skip_*`` tuning -> gate.
        Raises ValueError when the endpoint is not configured."""
        base_url = str(cfg.get(f"{PRUNE_PREFIX}base_url") or "").strip()
        model = str(cfg.get(f"{PRUNE_PREFIX}model") or "").strip()
        api_key = str(api_key or "").strip()
        if not (base_url and api_key and model):
            raise ValueError(f"recall skipping needs {PRUNE_PREFIX}base_url, {PRUNE_PREFIX}model and an API key")

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
            min_noul=_num("min_noul", SKIP_AT),
            max_gate_ms=_num("max_gate_ms", MAX_GATE_MS),
        )

    def judge(self, message: str) -> CueVerdict:
        """Skip the recall when the message has no cue. Never raises."""
        started = time.perf_counter()
        box: dict[str, object] = {}

        def _ask() -> None:
            try:
                box["noul"] = ask_nouls(
                    self.url, self.api_key, self.model, message_state(message),
                    {"no_cue": NO_CUE_INSTRUCTIONS}, self.max_gate_ms / 1000.0 + 1.0,
                )["no_cue"]
            except Exception as exc:
                box["error"] = exc

        # Same shape as the relevance gate: urlopen's timeout bounds socket inactivity, not a
        # stall behind other GPU work, so the wall clock is the bound.
        worker = threading.Thread(target=_ask, name="hindsight-cue-gate", daemon=True)
        worker.start()
        worker.join(self.max_gate_ms / 1000.0)
        elapsed = (time.perf_counter() - started) * 1000
        if "noul" in box:
            noul = float(box["noul"])  # type: ignore[arg-type]
            return CueVerdict(noul >= self.min_noul, noul, elapsed)
        why = f"over {self.max_gate_ms:.0f} ms" if worker.is_alive() else f"error: {type(box.get('error')).__name__}"
        logger.warning("recall cue gate could not answer (%s); recalling as usual", why)
        return CueVerdict(False, None, elapsed, why)
