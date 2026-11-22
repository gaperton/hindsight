"""Session-scoped recall journal — the cross-turn dedup tracker.

Each memory item an automatic recall shows to the model is recorded here with its
id, a content hash, the session position it was shown at (1-based ordinal of the
user message the block attached to) and the exact UTC time. On the next turn a
result whose ``(id, content_hash)`` pair is already in the journal drops out of the
injected block, so the same observation stops being re-shipped turn after turn.

Design notes:

- The Hindsight recall API has no exclude-ids parameter, so suppression must be
  client-side, after the recall response and before block assembly (the plugin's
  ``_do_recall``).
- Keying on ``id + content hash`` (not id alone) is deliberate: ``RecallResult``
  carries no ``updated_at``, and consolidation rewrites an observation in place
  under the same id. The hash is the content epoch — a re-consolidated observation
  counts as new information and shows again.
- ``message_no`` is the plugin's own user-turn count (Hermes core exposes no
  message index to providers). It is durable across a restart: ``initialize()``
  re-seeds the counter from ``MAX(message_no)`` for the session.
- v1 never re-shows a fact within a session. After a compaction cutoff drops an
  old block from the model's window, its facts stay suppressed until the next
  session; the stored ``message_no`` is exactly the data a later window-bound
  re-anchor rule needs, and sessions are cheap — a new session resets scope.
- Callers treat every exception as "journal unavailable" and fail OPEN: memory
  bookkeeping must never cost the model a recall.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_journal (
    session_id   TEXT NOT NULL,
    uid          TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    message_no   INTEGER NOT NULL,
    recalled_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, uid, content_hash)
);
CREATE INDEX IF NOT EXISTS recall_journal_session_pos
    ON recall_journal (session_id, message_no);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def content_hash(text: str | None) -> str:
    """First 12 hex of sha1: the content epoch for a recalled item's id."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


class RecallJournal:
    """Small SQLite journal of items already shown in a session.

    One instance per provider (per chat session); the file is shared process-wide.
    Safe for the plugin's two calling threads (turn thread + prefetch worker):
    every statement runs under an instance lock on a ``check_same_thread=False``
    connection.
    """

    def __init__(self, path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def max_message_no(self, session_id: str) -> int:
        """Highest recorded session position, or 0 for an unseen session."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(message_no) FROM recall_journal WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0] or 0)

    def filter_and_record(
        self, session_id: str, results: Iterable[Any], message_no: int
    ) -> tuple[list[Any], int]:
        """Drop already-shown items, record what passes. -> ``(kept, suppressed)``.

        Items without an id are passed through unrecorded (nothing to key on).
        Order of *results* is preserved.
        """
        with self._lock:
            seen = {
                (row[0], row[1])
                for row in self._conn.execute(
                    "SELECT uid, content_hash FROM recall_journal WHERE session_id = ?",
                    (session_id,),
                )
            }
            items = list(results)
            kept: list[Any] = []
            new_rows: list[tuple[str, str, str, int, str]] = []
            now = _utc_now()
            for item in items:
                uid = getattr(item, "id", None)
                if not uid:
                    kept.append(item)
                    continue
                pair = (str(uid), content_hash(getattr(item, "text", None)))
                if pair in seen:
                    continue
                seen.add(pair)  # a within-response repeat shows once, like the block would
                kept.append(item)
                new_rows.append((session_id, pair[0], pair[1], int(message_no), now))
            if new_rows:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO recall_journal"
                    " (session_id, uid, content_hash, message_no, recalled_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    new_rows,
                )
                self._conn.commit()
        return kept, len(items) - len(kept)

    def unrecord(self, session_id: str, results: Iterable[Any], message_no: int) -> int:
        """Undo the recording of items that were recalled but never shown (prune drops).

        Only rows stamped with *message_no* are deleted: an item shown on an earlier turn
        keeps its mark and stays suppressed. Without this, a memory the relevance gate
        dropped once would be journal-suppressed for the whole session — the gate's
        verdict is per-turn (the next turn's query may make it grade as keep): never burn
        what was not shown.
        """
        rows = []
        for item in results:
            uid = getattr(item, "id", None)
            if uid:
                rows.append((session_id, str(uid), content_hash(getattr(item, "text", None)), int(message_no)))
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "DELETE FROM recall_journal WHERE session_id = ? AND uid = ? AND content_hash = ? AND message_no = ?",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def record(self, session_id: str, results: Iterable[Any], message_no: int) -> None:
        """Record items as shown without filtering (the ``hindsight_recall`` tool path:
        what the model pulled is in the transcript, so auto-recall must not re-show it,
        but the tool answer itself is never suppressed)."""
        now = _utc_now()
        rows = []
        for item in results:
            uid = getattr(item, "id", None)
            if uid:
                rows.append((session_id, str(uid), content_hash(getattr(item, "text", None)), int(message_no), now))
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO recall_journal"
                " (session_id, uid, content_hash, message_no, recalled_at)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
