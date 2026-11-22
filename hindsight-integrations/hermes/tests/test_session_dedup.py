"""Session-scoped recall dedup (session_journal.py + _do_recall/_tool_recall wiring).

The fake client hands out scripted recall batches, each item ``(id, text)`` —
like the SDK's RecallResult, items carry an id and the recall response is an
object with ``.results``.
"""

import sqlite3
from types import SimpleNamespace

import pytest


class JournalFakeClient:
    """arecall returns *answers[i]* on the i-th call (last answer repeats)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.retains = []
        self.reflects = []

    async def arecall(self, **kwargs):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return SimpleNamespace(results=[SimpleNamespace(id=i, text=t) for i, t in answer])

    async def aretain_batch(self, **kwargs):
        self.retains.append(kwargs)
        return SimpleNamespace(operations=[])

    async def areflect(self, **kwargs):
        self.reflects.append(kwargs)
        return SimpleNamespace(text="")

    async def aclose(self):
        pass


def _journal_rows(hermes_env):
    path = hermes_env / "hindsight" / "recall_journal.sqlite3"
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT session_id, uid, content_hash, message_no, recalled_at"
            " FROM recall_journal ORDER BY message_no, uid"
        ).fetchall()
    finally:
        conn.close()


@pytest.fixture
def sync_provider(provider):
    """recall_sync=True: prefetch() recalls inline, so turns are deterministic."""

    def _make(answers, **config):
        client = JournalFakeClient(answers)
        instance, _ = provider({"recall_sync": True, **config}, client=client)
        return instance, client

    return _make


def test_second_turn_suppresses_already_shown(sync_provider, hermes_env):
    p, _ = sync_provider(
        [
            [("a", "fact A"), ("b", "fact B")],
            [("a", "fact A"), ("b", "fact B"), ("c", "fact C")],
        ]
    )
    first = p.prefetch("q1")
    assert "fact A" in first and "fact B" in first
    assert p._last_recall_count == 2

    p._turn_counter = 1  # turn 2's block lands on user message 2
    second = p.prefetch("q2")
    assert "fact A" not in second and "fact B" not in second and "fact C" in second
    assert p._last_recall_count == 1

    rows = _journal_rows(hermes_env)
    assert [r[1] for r in rows] == ["a", "b", "c"]
    assert [r[3] for r in rows] == [1, 1, 2]  # session position recorded per row


def test_updated_observation_shows_again(sync_provider):
    p, _ = sync_provider(
        [
            [("a", "fact A"), ("b", "B v1")],
            [("a", "fact A"), ("b", "B v2")],  # consolidation rewrote b under the same id
        ]
    )
    assert "B v1" in p.prefetch("q1")
    second = p.prefetch("q2")
    assert "fact A" not in second  # unchanged fact stays suppressed
    assert "B v2" in second  # content epoch changed -> new information


def test_tool_recall_shows_all_and_marks_the_journal(sync_provider, hermes_env):
    p, _ = sync_provider([[("a", "fact A"), ("b", "fact B")]])
    out = p.handle_tool_call("hindsight_recall", {"query": "q"})
    assert "fact A" in out and "fact B" in out  # tool answers are never deduped
    assert [r[1] for r in _journal_rows(hermes_env)] == ["a", "b"]  # ...but are recorded
    # The next automatic block must not re-ship what the model already pulled.
    assert p.prefetch("q2") == ""


def test_injected_fact_never_repeated_by_tool_on_later_turn(sync_provider):
    p, _ = sync_provider([[("a", "fact A")], [("a", "fact A"), ("b", "fact B")]])
    assert "fact A" in p.prefetch("q1")  # injection marks a
    out = p.handle_tool_call("hindsight_recall", {"query": "q2"})
    # Mark-only: the tool itself still returns everything, including a.
    assert "fact A" in out and "fact B" in out


def test_disabled_flag_creates_no_journal(hermes_env, provider):
    client = JournalFakeClient([[("a", "fact A")]])
    p, _ = provider({"recall_sync": True, "recall_session_dedup": False}, client=client)
    assert p._journal is None
    assert "fact A" in p.prefetch("q1")
    assert "fact A" in p.prefetch("q2")  # no suppression without the journal
    assert not (hermes_env / "hindsight" / "recall_journal.sqlite3").exists()


def test_journal_failure_fails_open(sync_provider, monkeypatch):
    p, _ = sync_provider([[("a", "fact A")]])

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(p._journal, "filter_and_record", _boom)
    out = p.prefetch("q1")
    assert "fact A" in out  # bookkeeping errors never cost a recall


def test_journal_survives_restart_and_seeds_the_counter(sync_provider, provider, hermes_env):
    p, _ = sync_provider([[("a", "fact A")]])
    p._turn_counter = 4  # 4 completed turns: the next block lands on user message 5
    assert "fact A" in p.prefetch("q1")

    # A restarted session keeps its id: dedup continues and the position counter
    # resumes past the highest recorded message, never restarting at 1.
    p2, _ = provider({"recall_sync": True}, client=JournalFakeClient([[("a", "fact A"), ("b", "fact B")]]))
    assert p2._turn_counter == 5
    out = p2.prefetch("q2")
    assert "fact A" not in out and "fact B" in out
    rows = _journal_rows(hermes_env)
    assert [r[1] for r in rows] == ["a", "b"]
    assert rows[-1][3] == 6  # p2's block lands on message 6


def test_recalled_at_is_exact_utc(sync_provider, hermes_env):
    from datetime import datetime

    p, _ = sync_provider([[("a", "fact A")]])
    p.prefetch("q1")
    (row,) = _journal_rows(hermes_env)
    stamp = datetime.fromisoformat(row[4])
    assert stamp.utcoffset().total_seconds() == 0  # explicit UTC, not naive


def test_id_less_items_pass_through(sync_provider):
    """Results without an id have nothing to key on: shown, never recorded."""
    p, _ = sync_provider([[(None, "fact A"), ("b", "fact B")]])
    out = p.prefetch("q1")
    assert "fact A" in out and "fact B" in out
    out = p.prefetch("q2")
    assert "fact A" in out  # still unkeyable, still passes
    assert "fact B" not in out
