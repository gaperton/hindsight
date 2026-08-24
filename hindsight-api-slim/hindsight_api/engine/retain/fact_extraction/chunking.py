"""Chunking helpers for fact extraction."""

import json
import re
from collections.abc import Iterator

# Below this size, splitting an over-long chunk further cannot help: if a chunk
# this small still overflows the model's output cap, the cause is degenerate or
# looping model output rather than genuinely dense input, and halving it just
# recurses toward a single character. A few-hundred-character floor bounds that
# runaway (an all-sizes-overflow 3000-char chunk drops in ~17 extraction calls
# instead of ~5000) while staying well under any chunk that legitimately holds
# enough facts to exceed the cap.
_MIN_SPLIT_CHUNK_CHARS = 500


def _split_chunk_for_output_retry(chunk: str) -> tuple[str, str] | None:
    """Split an oversized extraction chunk without corrupting structured input."""
    stripped = chunk.strip()
    if len(stripped) <= _MIN_SPLIT_CHUNK_CHARS:
        return None

    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        if len(parsed) >= 2:
            mid = len(parsed) // 2
            return json.dumps(parsed[:mid]), json.dumps(parsed[mid:])

        if len(parsed) == 1 and isinstance(parsed[0], dict):
            turn = parsed[0]
            content = turn.get("content")
            if isinstance(content, str) and len(content) > 1:
                cut = len(content) // 2
                first_turn = dict(turn)
                second_turn = dict(turn)
                first_turn["content"] = content[:cut]
                second_turn["content"] = content[cut:]
                return json.dumps([first_turn]), json.dumps([second_turn])

        return None

    # Split plain text at the midpoint, preferring sentence boundaries nearby.
    mid_point = len(stripped) // 2
    search_range = int(len(stripped) * 0.2)
    search_start = max(0, mid_point - search_range)
    search_end = min(len(stripped), mid_point + search_range)

    best_split = mid_point
    for ending in [". ", "! ", "? ", "\n\n"]:
        pos = stripped.rfind(ending, search_start, search_end)
        if pos != -1:
            best_split = pos + len(ending)
            break

    first_half = stripped[:best_split].strip()
    second_half = stripped[best_split:].strip()
    if not first_half or not second_half or first_half == stripped or second_half == stripped:
        return None
    return first_half, second_half


# Separators for sentence-aware recursive text splitting, ordered most- to
# least-preferred. The final "" lets the splitter break mid-word as a last
# resort so a chunk can never exceed the size budget.
_RECURSIVE_TEXT_SEPARATORS = [
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    ". ",  # Sentence endings
    "! ",  # Exclamations
    "? ",  # Questions
    "; ",  # Semicolons
    ", ",  # Commas
    " ",  # Words
    "",  # Characters (last resort)
]


def _iter_separator_splits(text: str, separator: str) -> Iterator[str]:
    """Yield ``text`` cut at every occurrence of ``separator``, separator kept on the right.

    The lazy equivalent of what ``RecursiveCharacterTextSplitter`` gets from
    ``re.split("(sep)", text)`` under its default ``keep_separator=True``: the piece before
    the first match, then one piece per match running from that match to the next. Empty
    pieces are dropped, matching the ``[s for s in splits if s]`` filter there.

    Lazy because the eager form is the expensive half of chunking a large body — splitting a
    45 MB document on ``". "`` materialises 646k substrings (~80 MB live, and far more RSS
    once the allocator has fragmented) purely to feed a greedy packer that reads them once,
    in order (#3756). An empty separator degrades to per-character iteration, which is the
    same last-resort behaviour as ``list(text)`` without the 47M single-character strings.
    """
    if not separator:
        yield from text
        return
    previous = 0
    for match in re.finditer(re.escape(separator), text):
        piece = text[previous : match.start()]
        if piece:
            yield piece
        previous = match.start()
    tail = text[previous:]
    if tail:
        yield tail


def _iter_recursive_splits(text: str, max_chars: int, separators: list[str]) -> Iterator[str]:
    """Sentence-aware split of ``text`` into chunks of at most ``max_chars``, streamed.

    A faithful, lazy re-implementation of ``RecursiveCharacterTextSplitter._split_text`` for
    the one configuration retain used it in (``chunk_overlap=0``, ``keep_separator=True``,
    ``strip_whitespace=True``, ``length_function=len``). The algorithm is unchanged — pick
    the most-preferred separator present, cut on it, pack the pieces greedily, and recurse
    with the remaining separators into any piece that is still over budget — so the
    boundaries it produces are identical.

    That identity is load-bearing rather than incidental: chunk boundaries are content
    hashes that delta retain matches against, and ``chunk_id`` is derived from a chunk's
    index. Boundaries that shifted would make every stored chunk of every document look
    changed. ``test_chunking_streams.py`` pins the output against the langchain splitter
    directly, so a drift shows up as a test failure rather than as a silent re-ingest.

    The greedy packing is inlined rather than run over a collected list of "good" pieces the
    way ``_split_text`` collects ``good_splits``. Collecting first is what an eager
    implementation can afford: prose has no over-budget piece to interrupt the run, so the
    list grows to hold every piece in the document — 646k of them for a 45 MB body, ~80 MB,
    which is the whole cost this function exists to avoid. Packing as pieces arrive keeps at
    most one chunk's worth alive (#3756).
    """
    separator = separators[-1]
    remaining: list[str] = []
    for index, candidate in enumerate(separators):
        if not candidate:
            separator = candidate
            break
        if re.search(re.escape(candidate), text):
            separator = candidate
            remaining = separators[index + 1 :]
            break

    # The chunk being packed. Whitespace-stripping and the drop of an empty result mirror
    # ``_join_docs``; resetting at an over-budget piece mirrors ``_split_text`` starting a
    # fresh ``good_splits`` run after one.
    buffered: list[str] = []
    buffered_len = 0

    def _flush() -> Iterator[str]:
        nonlocal buffered, buffered_len
        if buffered:
            packed = "".join(buffered).strip()
            buffered = []
            buffered_len = 0
            if packed:
                yield packed

    for piece in _iter_separator_splits(text, separator):
        if len(piece) < max_chars:
            if buffered and buffered_len + len(piece) > max_chars:
                yield from _flush()
            buffered.append(piece)
            buffered_len += len(piece)
            continue
        # Over budget even alone: close the run so ordering is preserved, then split this
        # piece further, or emit it whole when no separator is left to try.
        yield from _flush()
        if remaining:
            yield from _iter_recursive_splits(piece, max_chars, remaining)
        else:
            yield piece
    yield from _flush()


def iter_chunks(text: str, max_chars: int, structured_chunk_size: int | None = None) -> Iterator[str]:
    """Stream the chunks of ``text``, in order — the lazy form of :func:`chunk_text`.

    Yields exactly what ``chunk_text`` returns, one chunk at a time, so a caller that
    consumes chunks as it goes never holds the whole document's chunk list. Retain's
    producer works that way: a 45 MB body is 32k chunks that cost ~130 MB live and several
    hundred MB of RSS once materialised together, and the pipeline only ever needs the one
    it is extracting from (#3756).

    See :func:`chunk_text` for what the chunking itself guarantees.
    """
    # If text is small enough, return as-is
    if len(text) <= max_chars:
        yield text
        return

    structured_limit = structured_chunk_size if structured_chunk_size is not None else max_chars

    # Try to parse as JSON conversation array
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
        # This looks like a conversation - chunk at turn boundaries
        yield from _iter_conversation_chunks(parsed, max_chars, structured_limit)
        return

    if isinstance(parsed, dict):
        # A single JSON object — e.g. one JSONL line handed back to the extractor
        # after the producer already pre-chunked it. It is one structured unit:
        # keep it whole up to the structured limit, else split it as text within
        # the chunk budget. Without this, a lone object (one line, so _chunk_jsonl
        # declines) would fall through to plain-text splitting and re-split a chunk
        # the producer deliberately kept whole — breaking idempotency (issue #2301).
        if len(text) <= structured_limit:
            yield text
        else:
            yield from _iter_recursive_splits(text, max_chars, _RECURSIVE_TEXT_SEPARATORS)
        return

    # Try to parse as JSONL (newline-delimited JSON objects, e.g. session logs)
    if _looks_like_jsonl(text):
        yield from _iter_jsonl_chunks(text, max_chars, structured_limit)
        return

    # Fall back to sentence-aware text splitting
    yield from _iter_recursive_splits(text, max_chars, _RECURSIVE_TEXT_SEPARATORS)


def chunk_text(text: str, max_chars: int, structured_chunk_size: int | None = None) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns) and JSONL (newline-delimited
    JSON objects), splits at turn/line boundaries so no object is split across chunks.
    A single turn/line that overflows ``max_chars`` is kept whole only up to
    ``structured_chunk_size``. When unset, that limit defaults to ``max_chars``.
    For plain text, uses sentence-aware splitting.

    The result is idempotent: re-chunking any chunk this returns yields that chunk
    unchanged. The streaming retain pipeline pre-chunks each document once and then
    re-chunks every piece during extraction; if a piece re-split, its sub-chunks
    would inherit one chunk_index and collide on ``chunk_id`` (issue #2301).

    Materialises every chunk. Prefer :func:`iter_chunks` for anything document-sized —
    this is the convenience form for callers that need random access or a length.

    Args:
        text: Input text to chunk (plain text, JSON conversation, or JSONL)
        max_chars: Target maximum characters per chunk
        structured_chunk_size: Maximum characters for a single JSONL line or
            conversation turn to keep whole. Defaults to ``max_chars``.

    Returns:
        List of text chunks, roughly under max_chars
    """
    return list(iter_chunks(text, max_chars, structured_chunk_size=structured_chunk_size))


def _iter_conversation_chunks(turns: list[dict], max_chars: int, structured_limit: int) -> Iterator[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk
        structured_limit: Maximum characters for a single turn to keep whole

    Yields:
        JSON-serialized chunks, each containing complete turns
    """
    current_chunk: list[dict] = []
    current_size = 2  # Account for "[]"
    emitted = False

    def _flush() -> Iterator[str]:
        nonlocal current_chunk, current_size, emitted
        if current_chunk:
            emitted = True
            yield json.dumps(current_chunk, ensure_ascii=False)
            current_chunk = []
            current_size = 2  # Reset to "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_unit_size = len(turn_json)
        turn_size = turn_unit_size + 1  # +1 for comma

        # A turn too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if turn_unit_size > structured_limit:
            yield from _flush()
            for fragment in _iter_recursive_splits(
                turn_json, min(structured_limit, max_chars), _RECURSIVE_TEXT_SEPARATORS
            ):
                emitted = True
                yield fragment
            continue

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            yield from _flush()

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    yield from _flush()

    if not emitted:
        yield json.dumps(turns, ensure_ascii=False)


def _iter_nonblank_lines(text: str) -> Iterator[str]:
    """Yield ``text``'s non-blank lines without materialising them all.

    ``str.splitlines()`` on a document-sized body allocates a second copy of it as N
    separate strings; the JSONL path reads its lines strictly in order and twice (once to
    decide the format, once to pack), so it can afford to re-scan instead of retaining
    (#3756). Splits on ``\\n`` only, and strips a trailing ``\\r``, which is what
    ``splitlines`` does for the CR/LF forms JSONL can realistically arrive in.
    """
    start = 0
    length = len(text)
    while start < length:
        end = text.find("\n", start)
        if end == -1:
            end = length
        line = text[start:end]
        if line.endswith("\r"):
            line = line[:-1]
        # `not line.isspace()` rather than `line.strip()`: both answer "does this line have a
        # non-whitespace character", but strip() BUILDS the stripped copy to answer it. A body
        # with no newline at all is one line, so on a 45 MB one that is a 45 MB copy allocated
        # to decide the line is not blank (#3756). isspace() scans and allocates nothing.
        if line and not line.isspace():
            yield line
        start = end + 1


def _looks_like_jsonl(text: str) -> bool:
    """Whether ``text`` is newline-delimited JSON: 2+ non-blank lines, each a JSON object.

    Every line has to be checked — one line that is not an object disqualifies the whole
    body — but none of the parsed objects is kept, so this scans rather than collects.
    Non-JSONL input is rejected on its first line, so prose never gets scanned twice.
    """
    seen = 0
    for line in _iter_nonblank_lines(text):
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(parsed, dict):
            return False
        seen += 1
    return seen >= 2


def _iter_jsonl_chunks(text: str, max_chars: int, structured_limit: int) -> Iterator[str]:
    """Chunk newline-delimited JSON (JSONL) at line boundaries.

    Packs whole lines into chunks so no line is split across chunks (multiple short lines
    may share a chunk). A line that overflows ``max_chars`` is kept whole only up to
    ``structured_limit``. Call only for text :func:`_looks_like_jsonl` accepted.

    Args:
        text: Input text to chunk.
        max_chars: Maximum characters per chunk.
        structured_limit: Maximum characters for a single JSONL line to
            keep whole.

    Yields:
        JSONL chunks (lines joined by newline).
    """
    current_chunk: list[str] = []
    current_size = 0

    def _flush() -> Iterator[str]:
        nonlocal current_chunk, current_size
        if current_chunk:
            yield "\n".join(current_chunk)
            current_chunk = []
            current_size = 0

    for line in _iter_nonblank_lines(text):
        line_unit_size = len(line)
        line_size = len(line) + 1  # +1 for the joining newline

        # A line too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if line_unit_size > structured_limit:
            yield from _flush()
            yield from _iter_recursive_splits(line, min(structured_limit, max_chars), _RECURSIVE_TEXT_SEPARATORS)
            continue

        # If adding this line would exceed the limit and we have lines, flush.
        # A line up to structured_limit is kept whole (a bounded overflow).
        if current_size + line_size > max_chars and current_chunk:
            yield from _flush()

        current_chunk.append(line)
        current_size += line_size

    yield from _flush()
