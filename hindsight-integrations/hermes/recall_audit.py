"""Per-request recall audit — one SQLite row per automatic recall, in the journal DB.

``session_journal.py`` answers "has this item ever been shown?". This table answers
"what happened on that turn": the query, one count column per filtering phase —
``n_found = n_duplicated + n_pruned + n_injected`` (duplicated = session-dedup drops:
already shown this session or a repeat within one response; pruned = relevance-gate drops)
— plus every item the daemon returned tagged with its fate (``injected``, ``already_shown``,
``repeat``, ``pruned`` with the noul that sank it and the floor it failed). The three drop
reasons are all plugin-side decisions; the daemon's token cut is not visible from the
client, so no row can name items the server never returned. The injected block itself is
not stored: it is exactly ``"\\n".join("- " + text)`` over the ``injected`` items in recall
order, so a block column would only duplicate the row's text.

Written once per ``_do_recall``; a failed recall gets a row with ``error`` set. Reflect is
not audited (synthesized prose, no items to tag — the same reason ``RecallReport`` counts
nothing there), and neither is the ``hindsight_recall`` tool (it drops nothing and its
answer is already in the transcript).

Retention trims to the newest ``max_rows`` on every write. Rows carry full item text
(~15-20 KB for a full block), so the cap bounds the file. Like every other bookkeeping
path this fails OPEN: an audit write error costs the audit row, never the recall.
"""

from __future__ import annotations

import json
import sqlite3
import threading

from .session_journal import _utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    message_no    INTEGER NOT NULL,
    recalled_at   TEXT NOT NULL,
    duration_ms   REAL NOT NULL DEFAULT 0,
    query         TEXT NOT NULL,
    n_found       INTEGER NOT NULL DEFAULT 0,
    n_duplicated  INTEGER NOT NULL DEFAULT 0,
    n_pruned      INTEGER NOT NULL DEFAULT 0,
    n_injected    INTEGER NOT NULL DEFAULT 0,
    gate_status   TEXT NOT NULL DEFAULT 'off',
    prune_floor   REAL,
    items_json    TEXT NOT NULL DEFAULT '[]',
    error         TEXT,
    cue_noul      REAL
);
CREATE INDEX IF NOT EXISTS recall_requests_session_pos
    ON recall_requests (session_id, message_no);
"""

#: ``stage`` values inside ``items_json``.
STAGES = ("injected", "already_shown", "repeat", "pruned")


class RecallAudit:
    """One ``recall_requests`` row per automatic recall: query, phase counts, item fates.

    ``gate_status`` says whether the relevance gate could speak: ``off`` (disabled),
    ``graded``, ``too-few`` (fewer surviving items than ``min_results``),
    ``recall-skipped`` (the cue gate skipped the search: no items, see ``cue_noul``), or
    ``skipped:<why>`` — without it, ``n_pruned = 0`` cannot distinguish "all survived"
    from "never graded". Same file and threading shape as ``RecallJournal`` (turn thread
    + prefetch worker, instance lock over a ``check_same_thread=False`` connection).
    """

    def __init__(self, path, max_rows: int = 1000) -> None:
        self.path = str(path)
        self.max_rows = max(1, int(max_rows))
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            # cue_noul arrived with the pre-recall cue gate; older files lack the column.
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(recall_requests)")}
            if "cue_noul" not in columns:
                self._conn.execute("ALTER TABLE recall_requests ADD COLUMN cue_noul REAL")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(
        self,
        *,
        session_id: str,
        message_no: int,
        duration_ms: float,
        query: str,
        n_found: int,
        n_duplicated: int,
        n_pruned: int,
        n_injected: int,
        gate_status: str,
        prune_floor: float | None,
        items: list,
        error: str | None = None,
        cue_noul: float | None = None,
    ) -> None:
        """Write one request row, then trim the table to the newest ``max_rows`` rows.

        *items* is the per-item fate list (``uid``/``text``/``stage``/``noul``) dumped as
        JSON here, in recall order, so the row alone replays the whole decision. *cue_noul* is
        the cue gate's "no cue" probability for the query (None when the gate is off or failed).
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO recall_requests"
                " (session_id, message_no, recalled_at, duration_ms, query,"
                "  n_found, n_duplicated, n_pruned, n_injected, gate_status, prune_floor,"
                "  items_json, error, cue_noul)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, int(message_no), _utc_now(), float(duration_ms), query,
                    int(n_found), int(n_duplicated), int(n_pruned), int(n_injected),
                    gate_status, prune_floor, json.dumps(items, ensure_ascii=False), error,
                    cue_noul,
                ),
            )
            # Every write trims; one bounded DELETE per insert keeps the table (and this
            # statement) cheap without a separate janitor. Keeps the newest max_rows rows.
            self._conn.execute(
                "DELETE FROM recall_requests WHERE id <= (SELECT MAX(id) - ? FROM recall_requests)",
                (self.max_rows,),
            )
            # Must commit: an uncommitted sqlite3 row is invisible to every other connection.
            self._conn.commit()
