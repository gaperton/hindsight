"""Per-request recall audit (recall_audit.py + _do_recall wiring).

One ``recall_requests`` row per automatic recall, in the journal DB: the query, one count
column per phase (``n_found = n_duplicated + n_pruned + n_injected``), the gate status, and
every returned item tagged with its fate — ``injected``, ``already_shown`` / ``repeat``
(session dedup), ``pruned`` (relevance gate, with the noul and the floor).
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

_COLUMNS = (
    "session_id, message_no, query, n_found, n_duplicated, n_pruned, n_injected,"
    " gate_status, prune_floor, items_json, error"
)


class AuditFakeClient:
    """arecall returns *answers[i]* on the i-th call (last answer repeats), items (id, text)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    async def arecall(self, **kwargs):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return SimpleNamespace(results=[SimpleNamespace(id=i, text=t) for i, t in answer])

    async def aretain_batch(self, **kwargs):
        return SimpleNamespace(operations=[])

    async def areflect(self, **kwargs):
        return SimpleNamespace(text="")

    async def aclose(self):
        pass


def _request_rows(hermes_env):
    conn = sqlite3.connect(hermes_env / "hindsight" / "recall_journal.sqlite3")
    try:
        return conn.execute(f"SELECT {_COLUMNS} FROM recall_requests ORDER BY id").fetchall()
    finally:
        conn.close()


@pytest.fixture
def sync_provider(provider):
    """recall_sync=True: prefetch() recalls inline, so turns are deterministic."""

    def _make(answers, **config):
        instance, _ = provider({"recall_sync": True, **config}, client=AuditFakeClient(answers))
        return instance

    return _make


def test_one_row_per_request_with_phase_counts_and_drops(sync_provider, hermes_env):
    p = sync_provider([[("a", "fact A"), ("b", "fact B")], [("a", "fact A"), ("b", "fact B"), ("c", "fact C")]])
    p.prefetch("q1")
    p._turn_counter = 1
    p.prefetch("q2 what else")

    rows = _request_rows(hermes_env)
    assert len(rows) == 2
    first, second = rows
    assert first[2] == "q1" and first[3:7] == (2, 0, 0, 2)  # query; found/duplicated/pruned/injected
    assert json.loads(first[9])[0]["stage"] == "injected"

    assert second[2] == "q2 what else"
    assert second[3:7] == (3, 2, 0, 1)  # found 3, two already shown, only the new fact injected
    items = json.loads(second[9])
    assert [i["stage"] for i in items] == ["already_shown", "already_shown", "injected"]
    assert items[2]["text"] == "fact C"
    assert second[7] == "off" and second[8] is None and second[10] is None
    for row in rows:  # the phase counts partition every returned item
        assert row[3] == row[4] + row[5] + row[6]


def test_prune_verdicts_are_recorded(sync_provider, hermes_env, monkeypatch):
    from hindsight_hermes.decision_prune import PruneResult

    p = sync_provider([[("a", "fact A"), ("b", "fact B"), ("c", "fact C")]])

    def fake_gate(query, results):
        by = {r.id: r for r in results}
        return PruneResult([by["a"], by["c"]], [by["b"]], True, None, 12.5, (0.11,), 0.2)

    monkeypatch.setattr(p, "_gate", fake_gate)
    out = p.prefetch("q")
    assert "fact A" in out and "fact B" not in out

    (row,) = _request_rows(hermes_env)
    assert row[3:7] == (3, 0, 1, 2)  # found / duplicated / pruned / injected
    assert row[7] == "graded" and row[8] == 0.2  # gate_status, prune_floor
    items = json.loads(row[9])
    assert [i["stage"] for i in items] == ["injected", "pruned", "injected"]
    assert items[1]["uid"] == "b" and items[1]["noul"] == 0.11


def test_skipped_gate_is_recorded(sync_provider, hermes_env, monkeypatch):
    from hindsight_hermes.decision_prune import PruneResult

    p = sync_provider([[("a", "fact A")]])
    monkeypatch.setattr(p, "_gate", lambda q, r: PruneResult(list(r), [], False, "over 2200 ms", 1.0))
    assert "fact A" in p.prefetch("q")

    (row,) = _request_rows(hermes_env)
    assert row[7] == "skipped:over 2200 ms"  # n_pruned=0 alone would not say the gate never ran
    assert row[3:7] == (1, 0, 0, 1)
    assert [json.loads(row[9])[0]["stage"]] == ["injected"]


def test_failed_recall_records_an_error_row(sync_provider, hermes_env, monkeypatch):
    p = sync_provider([[("a", "fact A")]])

    def _boom(query, **kw):
        raise RuntimeError("daemon down")

    monkeypatch.setattr(p, "_recall", _boom)
    p.prefetch("q")

    (row,) = _request_rows(hermes_env)
    assert row[2] == "q" and row[3:7] == (0, 0, 0, 0)
    assert row[10] == "daemon down"


def test_audit_write_failure_fails_open(sync_provider, monkeypatch):
    p = sync_provider([[("a", "fact A")]])

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(p._audit, "record", _boom)
    assert "fact A" in p.prefetch("q1")  # an audit error never costs a recall


def test_audit_disabled_creates_no_table(sync_provider, hermes_env):
    p = sync_provider([[("a", "fact A")]], recall_audit=False)
    assert p._audit is None
    assert "fact A" in p.prefetch("q1")
    conn = sqlite3.connect(hermes_env / "hindsight" / "recall_journal.sqlite3")  # the journal still exists
    try:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'recall_requests'").fetchone()[0] == 0
    finally:
        conn.close()


def test_env_kill_switch_disables_audit(hermes_env, monkeypatch):
    """The env switch rides in as the config default — it applies when the profile has no
    ``hindsight/config.json`` (exactly the ``recall_session_dedup`` semantics)."""
    import hindsight_hermes as plugin
    from conftest import SECRETS

    SECRETS.update({"HINDSIGHT_MODE": "cloud", "HINDSIGHT_API_KEY": "test-key", "HINDSIGHT_RECALL_AUDIT": "false"})
    instance = plugin.HindsightMemoryProvider()
    monkeypatch.setattr(instance, "_new_cloud_client", lambda: AuditFakeClient([[("a", "fact A")]]))
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)
    instance.initialize("session-1")
    assert instance._audit is None


def test_trim_bounds_the_table(tmp_path):
    from hindsight_hermes.recall_audit import RecallAudit

    audit = RecallAudit(tmp_path / "a.sqlite3", max_rows=2)
    for i in range(5):
        audit.record(
            session_id="s", message_no=i, duration_ms=1.0, query=f"q{i}",
            n_found=1, n_duplicated=0, n_pruned=0, n_injected=1,
            gate_status="off", prune_floor=None, items=[],
        )
    # A fresh connection: proves every write committed (uncommitted sqlite rows are invisible
    # to other connections — the journal's known pitfall).
    conn = sqlite3.connect(tmp_path / "a.sqlite3")
    try:
        assert [r[0] for r in conn.execute("SELECT query FROM recall_requests ORDER BY id")] == ["q3", "q4"]
    finally:
        conn.close()


def test_journal_names_the_drop_reason(tmp_path):
    from hindsight_hermes.session_journal import RecallJournal

    journal = RecallJournal(tmp_path / "j.sqlite3")
    x1, x2 = SimpleNamespace(id="a", text="v"), SimpleNamespace(id="a", text="v")
    kept, dropped = journal.filter_and_record("s", [x1, x2], 1)
    assert len(kept) == 1 and [reason for _, reason in dropped] == ["repeat"]  # same-response dupe

    kept2, dropped2 = journal.filter_and_record("s", [SimpleNamespace(id="a", text="v")], 2)
    assert kept2 == [] and [reason for _, reason in dropped2] == ["already_shown"]  # earlier turn
