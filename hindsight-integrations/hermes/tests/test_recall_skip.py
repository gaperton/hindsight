"""Pre-recall cue gate (recall_skip.py + _do_recall wiring + audit row).

``urlopen`` is stubbed per test; the gate's single noul question is answered with a fixed
probability, and the daemon client records whether it was ever searched.
"""

import io
import json
import logging
import sqlite3
import urllib.request
from types import SimpleNamespace


GATE_CFG = {
    "recall_skip_no_cue": True,
    "recall_prune_base_url": "http://example.test",
    "recall_prune_api_key": "k",
    "recall_prune_model": "d1",
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def answering(noul):
    calls = []

    def _urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        calls.append(body)
        return FakeResponse(json.dumps({"answers": {key: {"noul": noul} for key in body["questions"]}}).encode())

    _urlopen.calls = calls
    return _urlopen


class CountingClient:
    def __init__(self):
        self.searches = []

    async def arecall(self, **kwargs):
        self.searches.append(kwargs["query"])
        return SimpleNamespace(results=[SimpleNamespace(id="a", text="the gateway runs on aurelius")])

    async def aretain_batch(self, **kwargs):
        return SimpleNamespace(operations=[])

    async def aclose(self):
        pass


def _audit(hermes_env):
    conn = sqlite3.connect(hermes_env / "hindsight" / "recall_journal.sqlite3")
    try:
        return conn.execute("SELECT gate_status, n_found, cue_noul FROM recall_requests").fetchall()
    finally:
        conn.close()


def test_a_message_without_a_cue_is_never_searched(provider, hermes_env, monkeypatch, caplog):
    urlopen = answering(0.71)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client, lines = CountingClient(), []
    instance, _ = provider({"recall_sync": True, **GATE_CFG}, client=client, status_callback=lines.append)
    with caplog.at_level(logging.INFO):
        assert instance.prefetch("ну давай", session_id="s1") == ""
    assert client.searches == []
    assert "ну давай" in urlopen.calls[0]["state"]
    assert "recall: skipped, the message has no cue (p=0.71)" in caplog.text
    assert any("recall skipped" in line for line in lines)
    assert _audit(hermes_env) == [("recall-skipped", 0, 0.71)]


def test_a_message_with_a_cue_is_searched_and_its_probability_audited(provider, hermes_env, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", answering(0.2))
    client = CountingClient()
    instance, _ = provider({"recall_sync": True, **GATE_CFG}, client=client)
    assert "aurelius" in instance.prefetch("restart the gateway on aurelius", session_id="s1")
    assert client.searches == ["restart the gateway on aurelius"]
    assert _audit(hermes_env) == [("off", 1, 0.2)]


def test_the_gate_reads_the_whole_message_not_the_recall_cut(provider, monkeypatch):
    """The recall query is cut to recall_max_input_chars; a cue past the cut still counts."""
    urlopen = answering(0.2)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    message = "ok " * 400 + "aurelius"
    instance, _ = provider({"recall_sync": True, "recall_max_input_chars": 800, **GATE_CFG}, client=CountingClient())
    instance.prefetch(message, session_id="s1")
    assert "aurelius" in urlopen.calls[0]["state"]


def test_an_unreachable_gate_recalls_as_usual(provider, monkeypatch):
    def unreachable(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    client = CountingClient()
    instance, _ = provider({"recall_sync": True, **GATE_CFG}, client=client)
    assert "aurelius" in instance.prefetch("ok", session_id="s1")
    assert client.searches == ["ok"]


def test_the_floor_is_configurable(provider, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", answering(0.71))
    client = CountingClient()
    instance, _ = provider({"recall_sync": True, "recall_skip_min_noul": 0.8, **GATE_CFG}, client=client)
    instance.prefetch("ok", session_id="s1")
    assert client.searches == ["ok"]


def test_the_gate_is_off_unless_enabled(provider, monkeypatch):
    def never(request, timeout=None):
        raise AssertionError("the gate is off; no request may go out")

    monkeypatch.setattr(urllib.request, "urlopen", never)
    client = CountingClient()
    instance, _ = provider({"recall_sync": True, **GATE_CFG, "recall_skip_no_cue": False}, client=client)
    instance.prefetch("ok", session_id="s1")
    assert client.searches == ["ok"]


def test_an_old_audit_file_gains_the_cue_column(tmp_path):
    from hindsight_hermes.recall_audit import RecallAudit

    path = tmp_path / "journal.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE recall_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
        " message_no INTEGER NOT NULL, recalled_at TEXT NOT NULL, duration_ms REAL NOT NULL DEFAULT 0,"
        " query TEXT NOT NULL, n_found INTEGER NOT NULL DEFAULT 0, n_duplicated INTEGER NOT NULL DEFAULT 0,"
        " n_pruned INTEGER NOT NULL DEFAULT 0, n_injected INTEGER NOT NULL DEFAULT 0,"
        " gate_status TEXT NOT NULL DEFAULT 'off', prune_floor REAL, items_json TEXT NOT NULL DEFAULT '[]', error TEXT)"
    )
    conn.commit()
    conn.close()
    audit = RecallAudit(path)
    audit.record(
        session_id="s", message_no=1, duration_ms=1, query="ok", n_found=0, n_duplicated=0, n_pruned=0,
        n_injected=0, gate_status="recall-skipped", prune_floor=None, items=[], cue_noul=0.7,
    )
    audit.close()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT cue_noul FROM recall_requests").fetchall() == [(0.7,)]
    conn.close()
