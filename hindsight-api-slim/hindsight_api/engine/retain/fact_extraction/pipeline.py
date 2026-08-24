"""Multi-content fact extraction orchestration."""

import asyncio

from ...response_models import TokenUsage
from ..types import ChunkMetadata, RetainContent
from ..types import ExtractedFact as ExtractedFactType
from .batch import extract_facts_from_contents_batch_api
from .chunking import chunk_text
from .normalization import (
    _add_temporal_offsets,
    _convert_causal_relations,
    _inject_label_tags,
    _parse_datetime,
)
from .streaming import extract_facts_from_text


def _extract_facts_chunks(
    contents: list[RetainContent],
    config,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    chunks mode: no LLM call, no entity extraction.

    Each chunk becomes one memory unit with the raw text as fact_text.
    User-provided entities from RetainContent.entities are picked up downstream
    by entity_processing.py — they are the sole source of entity data in this mode.
    """
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    global_chunk_idx = 0

    for content_index, content in enumerate(contents):
        chunks = chunk_text(
            content.content,
            config.retain_chunk_size,
            structured_chunk_size=config.retain_structured_chunk_size,
        )
        for chunk in chunks:
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk,
                    fact_count=1,
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                )
            )
            extracted_facts.append(
                ExtractedFactType(
                    fact_text=chunk,
                    fact_type="world",
                    entities=[],
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                    context=content.context,
                    mentioned_at=content.event_date,
                    metadata=content.metadata,
                    tags=content.tags,
                    observation_scopes=content.observation_scopes,
                )
            )
            global_chunk_idx += 1

    _add_temporal_offsets(extracted_facts, contents)
    return extracted_facts, chunks_metadata, TokenUsage()


async def extract_facts_from_contents(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts from multiple content items in parallel.

    This function:
    1. Extracts facts from all contents in parallel using the LLM
    2. Tracks which facts came from which chunks
    3. Adds time offsets to preserve fact ordering within each content
    4. Returns typed ExtractedFact and ChunkMetadata objects

    Routes to batch API mode if config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration for fact extraction
        agent_name: Name of the agent (for agent-related fact detection)
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (passed to batch API for state storage)
        operation_id: Async operation ID (passed to batch API for crash recovery)
        schema: Database schema (passed to batch API for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)
    """
    if not contents:
        return [], [], TokenUsage()

    # chunks mode: skip LLM entirely, store each chunk as-is
    # Must come before the batch-API check so no LLM queue/locks are acquired
    if config.retain_extraction_mode == "chunks":
        return _extract_facts_chunks(contents, config)

    # Route to batch API if enabled
    if config.retain_batch_enabled:
        return await extract_facts_from_contents_batch_api(
            contents, llm_config, agent_name, config, pool, operation_id, schema
        )

    # Step 1: Create parallel fact extraction tasks
    fact_extraction_tasks = []
    for item in contents:
        # Call extract_facts_from_text directly.
        task = extract_facts_from_text(
            text=item.content,
            event_date=item.event_date,
            context=item.context,
            llm_config=llm_config,
            agent_name=agent_name,
            config=config,
            metadata=item.metadata or None,
        )
        fact_extraction_tasks.append(task)

    # Step 2: Wait for all fact extractions to complete.
    # return_exceptions=True so a failing item doesn't cancel its still-running
    # siblings (which would leave orphaned LLM calls / partial work); we await
    # them all, then propagate.
    all_fact_results = await asyncio.gather(*fact_extraction_tasks, return_exceptions=True)

    # Step 3: Flatten and convert to typed objects
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    total_usage = TokenUsage()

    global_chunk_idx = 0
    global_fact_idx = 0

    # Never silently drop a document's memory. Any extraction failure (provider
    # rate-limit / timeout / 5xx, malformed response, token-limit, etc.)
    # propagates so the streaming producer surfaces it and the worker's
    # RetryTaskAt machinery retries the task — and ultimately fails it *loudly*
    # if the problem persists. Swallowing the error and substituting an empty
    # result here used to commit the document with 0 facts and mark the
    # operation `completed`, losing the memory with no signal. See issue #1833.
    valid_results = []
    for content, result in zip(contents, all_fact_results):
        if isinstance(result, Exception):
            raise result
        valid_results.append((content, result))

    for content_index, (content, (facts_from_llm, chunks_from_llm, content_usage)) in enumerate(valid_results):
        total_usage = total_usage + content_usage
        chunk_start_idx = global_chunk_idx

        # Convert chunk tuples to ChunkMetadata objects
        for chunk_index_in_content, (chunk_value, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_metadata = ChunkMetadata(
                chunk_text=chunk_value,
                fact_count=chunk_fact_count,
                content_index=content_index,
                chunk_index=global_chunk_idx,
            )
            chunks_metadata.append(chunk_metadata)
            global_chunk_idx += 1

        # Convert facts to ExtractedFact objects with proper indexing
        fact_idx_in_content = 0
        for chunk_idx_in_content, (_chunk_value, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_global_idx = chunk_start_idx + chunk_idx_in_content
            extraction_group_start_idx = global_fact_idx
            chunk_facts = facts_from_llm[fact_idx_in_content : fact_idx_in_content + chunk_fact_count]

            for fact_from_llm in chunk_facts:
                # Convert Fact model from LLM to ExtractedFactType dataclass
                # mentioned_at is always the event_date (when the conversation/document occurred)
                extracted_fact = ExtractedFactType(
                    fact_text=fact_from_llm.fact,
                    fact_type=fact_from_llm.fact_type,
                    entities=list(fact_from_llm.entities or []),
                    # occurred_start/end: from LLM only, leave None if not provided
                    occurred_start=_parse_datetime(fact_from_llm.occurred_start)
                    if fact_from_llm.occurred_start
                    else None,
                    occurred_end=_parse_datetime(fact_from_llm.occurred_end) if fact_from_llm.occurred_end else None,
                    causal_relations=_convert_causal_relations(
                        fact_from_llm.causal_relations or [], extraction_group_start_idx, len(chunk_facts)
                    ),
                    content_index=content_index,
                    chunk_index=chunk_global_idx,
                    context=content.context,
                    # mentioned_at: always the event_date (when the conversation/document occurred)
                    mentioned_at=content.event_date,
                    metadata=content.metadata,
                    tags=content.tags,
                    observation_scopes=content.observation_scopes,
                )

                extracted_facts.append(extracted_fact)
                global_fact_idx += 1
                fact_idx_in_content += 1

    # Step 4: For verbatim mode, collapse to one fact per chunk with original text
    if config.retain_extraction_mode == "verbatim":
        extracted_facts = _collapse_to_verbatim(extracted_facts, chunks_metadata)

    # Step 5: Add time offsets to preserve ordering within each content
    _add_temporal_offsets(extracted_facts, contents)

    # Step 6: Auto-tag facts from label groups with tag=True
    _inject_label_tags(extracted_facts, config)

    return extracted_facts, chunks_metadata, total_usage


def _collapse_to_verbatim(facts: list[ExtractedFactType], chunks: list[ChunkMetadata]) -> list[ExtractedFactType]:
    """
    For verbatim mode: ensure one fact per chunk with the original chunk text preserved.

    The LLM prompt asks for exactly one fact per chunk, but if it returns more,
    this collapses them: keeps the first fact as representative, overrides its
    fact_text with the raw chunk text, and merges entities from any extra facts.
    """
    chunk_text_map = {c.chunk_index: c.chunk_text for c in chunks}
    seen: dict[int, ExtractedFactType] = {}
    result: list[ExtractedFactType] = []

    for fact in facts:
        if fact.chunk_index not in seen:
            fact.fact_text = chunk_text_map.get(fact.chunk_index, fact.fact_text)
            seen[fact.chunk_index] = fact
            result.append(fact)
        else:
            # Merge entities from extra facts into the representative
            representative = seen[fact.chunk_index]
            for entity in fact.entities:
                if entity not in representative.entities:
                    representative.entities.append(entity)

    return result
