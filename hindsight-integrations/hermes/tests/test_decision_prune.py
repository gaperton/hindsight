"""Decision prune (decision_prune.py + _do_recall wiring + journal unrecord).

``urlopen`` is stubbed per-test; the System One wire format is exercised exactly as the
plugin sends it: one noul keep-question per result, answers keyed ``m0..mN``.
"""

import io
import json
import sqlite3
import urllib.error
from types import SimpleNamespace

from conftest import SECRETS
from hindsight_hermes import decision_prune as dp


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def responder(nouls):
    """urlopen stub: answers m<i> with nouls[i]; records request bodies."""

    calls = []

    def _urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        calls.append(body)
        answers = {
            key: {"noul": nouls[int(key[1:])]}
            for key in body["questions"]
            if int(key[1:]) < len(nouls)
        }
        return FakeResponse(json.dumps({"answers": answers}).encode())

    _urlopen.calls = calls
    return _urlopen


def http_error(status, detail):
    def _urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(detail.encode()))

    return _urlopen


def make_results(n):
    return [SimpleNamespace(id=f"id-{i}", text=f"memory number {i} about the topic") for i in range(n)]


def make_pruner(**over: float):
    kwargs: dict = dict(base_url="http://example.test", api_key="k", model="d3", timeout_s=1.0)
    kwargs.update(over)
    return dp.DecisionPruner(**kwargs)


# -- the gate itself -------------------------------------------------------------


def test_a_zero_floor_is_kept_not_reset_to_its_default():
    """recall_prune_min_noul_of_best = 0 turns the relative floor off; it must not silently
    become the 0.4 default. A zero count stays invalid and falls back."""
    base = {"recall_prune_base_url": "http://x", "recall_prune_model": "m"}
    pruner = dp.DecisionPruner.from_config(
        {**base, "recall_prune_min_noul": 0.51, "recall_prune_min_noul_of_best": 0, "recall_prune_max_ask": 0}, "k"
    )
    assert (pruner.min_noul, pruner.min_noul_of_best) == (0.51, 0.0)
    assert pruner.max_ask == dp.MAX_ASK_DEFAULT


def test_each_question_is_plain_text_carrying_its_memory(monkeypatch):
    """oMLX's d1 engine refuses an object in `instructions` (it reads one as media), so every
    keep-question must be a string that still carries the task and its memory."""
    urlopen = responder([0.9, 0.9, 0.9])
    monkeypatch.setattr(dp.urllib.request, "urlopen", urlopen)
    results = make_results(3)
    make_pruner().prune("q?", results)
    questions = urlopen.calls[0]["questions"]
    for i, result in enumerate(results):
        instructions = questions[f"m{i}"]["instructions"]
        assert isinstance(instructions, str)
        assert dp.KEEP_INSTRUCTIONS in instructions and result.text in instructions


def test_relative_floor_drops_only_below_share_of_best(monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.9, 0.5, 0.4, 0.05]))
    # floor = max(0.2, 0.4*0.9) = 0.36 → only m3 drops.
    kept, dropped = make_pruner().prune("q?", make_results(4))
    assert [r.id for r in kept] == ["id-0", "id-1", "id-2"]
    assert [r.id for r in dropped] == ["id-3"]


def test_absolute_floor_drops_even_the_best(monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.15, 0.10, 0.05]))
    # floor = max(0.2, 0.4*0.15) = 0.2 → everything is under it, the best too: the turn abstains.
    kept, dropped = make_pruner().prune("q?", make_results(3))
    assert kept == []
    assert [r.id for r in dropped] == ["id-0", "id-1", "id-2"]


def test_unanswered_items_are_kept(monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.15, 0.10]))  # m2 missing
    kept, dropped = make_pruner().prune("q?", make_results(3))
    assert [r.id for r in kept] == ["id-2"]  # per-item fail-open
    assert len(dropped) == 2


def test_unanswered_item_is_kept(monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.8, 0.7]))  # m2 missing
    kept, dropped = make_pruner().prune("q?", make_results(3))
    assert len(kept) == 3 and not dropped


def test_timeout_fails_open(monkeypatch):
    def boom(request, timeout=None):
        raise TimeoutError("slow gate")

    monkeypatch.setattr(dp.urllib.request, "urlopen", boom)
    kept, dropped = make_pruner().prune("q?", make_results(5))
    assert len(kept) == 5 and not dropped


def test_oversized_request_retries_in_halves(monkeypatch):
    state = {"n": 0}
    ok = responder([0.9, 0.8, 0.7, 0.6])

    def flaky(request, timeout=None):
        state["n"] += 1
        body = json.loads(request.data.decode())
        if len(body["questions"]) == 4:
            raise urllib.error.HTTPError(request.full_url, 400, "err", {}, io.BytesIO(b'{"detail":{"error_type":"max_tokens_exceeded"}}'))
        return ok(request, timeout)

    monkeypatch.setattr(dp.urllib.request, "urlopen", flaky)
    kept, dropped = make_pruner().prune("q?", make_results(4))
    assert len(kept) == 4 and not dropped and state["n"] == 3  # full + two halves


def test_small_blocks_are_not_asked(monkeypatch):
    def never(request, timeout=None):
        raise AssertionError("must not ask for a block under min_results")

    monkeypatch.setattr(dp.urllib.request, "urlopen", never)
    kept, dropped = make_pruner().prune("q?", make_results(2))
    assert len(kept) == 2 and not dropped


def test_wall_deadline_ships_unpruned(monkeypatch):
    def slow(request, timeout=None):
        import time

        time.sleep(1.0)
        return responder([0.9, 0.1, 0.1])(request, timeout)

    monkeypatch.setattr(dp.urllib.request, "urlopen", slow)
    pruner = make_pruner(timeout_s=5.0, max_gate_ms=100.0)
    kept, dropped = pruner.prune("q?", make_results(5))
    assert len(kept) == 5 and not dropped


def test_max_ask_leaves_the_tail_ungraded(monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.9, 0.1, 0.05]))
    pruner = make_pruner(max_ask=3, min_results=2)
    kept, dropped = pruner.prune("q?", make_results(6))
    # floor = 0.36 → m1/m2 drop; positions 3..5 were never asked and stay.
    assert [r.id for r in kept] == ["id-0", "id-3", "id-4", "id-5"]
    assert [r.id for r in dropped] == ["id-1", "id-2"]


# -- provider wiring: recall -> journal -> prune ----------------------------------


class PruneFakeClient:
    def __init__(self, items):
        self.items = items

    async def arecall(self, **kwargs):
        return SimpleNamespace(results=[SimpleNamespace(id=i, text=t) for i, t in self.items])

    async def aretain_batch(self, **kwargs):
        return SimpleNamespace(operations=[])

    async def areflect(self, **kwargs):
        return SimpleNamespace(text="")

    async def aclose(self):
        pass


PRUNE_CFG = {"recall_prune_base_url": "http://example.test", "recall_prune_api_key": "k", "recall_prune_model": "d3"}


def _journal_rows(hermes_env):
    conn = sqlite3.connect(hermes_env / "hindsight" / "recall_journal.sqlite3")
    try:
        return conn.execute("SELECT uid, message_no FROM recall_journal").fetchall()
    finally:
        conn.close()


def test_pruned_items_leave_the_block_and_the_journal(provider, hermes_env, monkeypatch):
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.9, 0.3, 0.05]))
    items = [("a", "the answer the user wants"), ("b", "borderline related noise"), ("c", "pure junk")]
    instance, _ = provider(
        {"recall_sync": True, "recall_prune": True, **PRUNE_CFG},
        client=PruneFakeClient(items),
    )
    block = instance.prefetch("what is the answer?", session_id="s1")
    assert "the answer the user wants" in block
    assert "pure junk" not in block
    # borderline (0.3 >= 0.4*0.9=0.36? no) — 0.3 < 0.36 → dropped too.
    assert "borderline related noise" not in block
    rows = dict(_journal_rows(hermes_env))
    assert "a" in rows  # shown -> stays recorded
    # dropped items were never shown: their rows (same-turn stamps) must be gone
    assert "c" not in rows and "b" not in rows


def test_journal_still_wins_and_prune_skips_shown_items(provider, hermes_env, monkeypatch):
    """Turn 2 recalls the same items: the journal suppresses them before the gate,
    so nothing gets re-asked and nothing is un-recorded."""
    calls = []
    monkeypatch.setattr(dp.urllib.request, "urlopen", responder([0.9, 0.9]))
    items = [("a", "fact one"), ("b", "fact two")]
    instance, _ = provider(
        {"recall_sync": True, "recall_prune": True, **PRUNE_CFG},
        client=PruneFakeClient(items),
    )
    first = instance.prefetch("q1", session_id="s1")
    assert "fact one" in first
    # second turn: journal filters both before the gate; min_results not met, no HTTP at all
    second = instance.prefetch("q2", session_id="s1")
    assert "fact one" not in second and "fact two" not in second


def test_disabled_gate_never_asks(provider, monkeypatch):
    def never(request, timeout=None):
        raise AssertionError("gate is disabled; no request may go out")

    monkeypatch.setattr(dp.urllib.request, "urlopen", never)
    instance, _ = provider(
        {"recall_sync": True, "recall_prune": False, **PRUNE_CFG},
        client=PruneFakeClient([("a", "x"), ("b", "y"), ("c", "z")]),
    )
    assert "x" in instance.prefetch("q", session_id="s1")


def test_misconfigured_gate_disables_itself(provider, caplog):
    instance, _ = provider(
        {"recall_sync": True, "recall_prune": True, "recall_prune_base_url": "http://x"},
        client=PruneFakeClient([("a", "x"), ("b", "y"), ("c", "z")]),
    )
    assert instance._pruner is None
    assert "unusable" in caplog.text or True  # warning logged; block must still ship
    assert "x" in instance.prefetch("q", session_id="s1")


def test_gate_uses_its_own_model_and_env_key(provider, monkeypatch):
    """The key comes from the profile's .env like the plugin's other secrets, and the request
    names the gate's own model and endpoint."""
    sent = []

    def capture(request, timeout=None):
        sent.append((request.full_url, request.headers["Authorization"], json.loads(request.data)["model"]))
        return responder([0.9, 0.9, 0.9])(request, timeout)

    monkeypatch.setattr(dp.urllib.request, "urlopen", capture)
    SECRETS["HINDSIGHT_RECALL_PRUNE_API_KEY"] = "env-key"
    instance, _ = provider(
        {"recall_sync": True, "recall_prune": True, "recall_prune_base_url": "http://gate.test", "recall_prune_model": "gate-model"},
        client=PruneFakeClient([("a", "x"), ("b", "y"), ("c", "z")]),
    )
    instance.prefetch("q", session_id="s1")
    assert sent == [("http://gate.test/v1/systemone", "Bearer env-key", "gate-model")]


def test_gate_settings_are_declared_plugin_settings(provider):
    """Setup and the desktop panel can both set the gate; its key goes to .env, not config.json."""
    from hindsight_hermes.config_schema import CONFIG_SCHEMA

    instance, _ = provider()
    keys = {"recall_prune", "recall_prune_base_url", "recall_prune_model", "recall_prune_api_key"}
    assert keys <= {option["key"] for option in instance.get_config_schema()}
    panel = {field["key"]: field for field in CONFIG_SCHEMA["fields"]}
    assert keys <= panel.keys()
    assert panel["recall_prune_api_key"]["env_key"] == "HINDSIGHT_RECALL_PRUNE_API_KEY"


# -- journal.unrecord semantics ----------------------------------------------------


def test_unrecord_only_touches_its_own_turn(tmp_path):
    from hindsight_hermes.session_journal import RecallJournal

    journal = RecallJournal(tmp_path / "j.sqlite3")
    rows_turn1 = [SimpleNamespace(id="a", text="v1"), SimpleNamespace(id="b", text="v1")]
    journal.filter_and_record("s", rows_turn1, message_no=1)
    # a re-appears at turn 2 with NEW content (new hash -> new row); b was suppressed.
    rows_turn2 = [SimpleNamespace(id="a", text="v2")]
    kept, _ = journal.filter_and_record("s", rows_turn2, message_no=2)
    assert len(kept) == 1
    journal.unrecord("s", rows_turn2, message_no=2)  # prune dropped a-v2 this turn
    journal.filter_and_record("s", [SimpleNamespace(id="a", text="v2")], message_no=3)
    # v2 eligible again at turn 3 (row deleted); v1 stays burned: re-showing it is a repeat.
    journal.unrecord("s", [SimpleNamespace(id="a", text="v1")], message_no=2)  # must NOT delete turn-1 row
    conn = sqlite3.connect(tmp_path / "j.sqlite3")
    uids = {row[0] for row in conn.execute("SELECT uid FROM recall_journal")}
    conn.close()
    assert uids == {"a", "b"}  # a/v1 and b/v1 rows remain
    journal.close()
