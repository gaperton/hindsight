"""Batch API fact extraction."""

import asyncio
import json
import logging
from typing import Any

from ...llm_wrapper import parse_llm_json
from ...operation_metadata import RetainExtractionErrors
from ...response_models import TokenUsage
from ..entity_labels import build_labels_lookup, is_label_entity, parse_entity_labels
from ..types import ChunkMetadata, RetainContent
from ..types import ExtractedFact as ExtractedFactType
from .chunking import chunk_text
from .models import CausalRelation, Fact
from .normalization import (
    _add_temporal_offsets,
    _coerce_entity_strings,
    _coerce_fact_response,
    _convert_causal_relations,
    _extract_map_entities,
    _infer_temporal_date,
    _inject_label_tags,
    _parse_datetime,
)
from .prompts import (
    _build_extraction_prompt_and_schema,
    _build_request_body,
    _build_user_message,
    _retain_mission_preamble,
)

logger = logging.getLogger(__name__)


async def _write_batch_extraction_errors(
    pool: Any,
    operation_id: str | None,
    schema: str | None,
    errors: RetainExtractionErrors,
) -> None:
    """Persist non-fatal Batch API extraction errors into operation result_metadata."""
    if not pool or not operation_id or errors.count == 0:
        return

    from ...db_utils import acquire_with_retry
    from ...task_backend import fq_table

    # `errors` is the complete set for this extraction run, so overwrite the
    # extraction_errors_* keys rather than folding in what's already stored. On
    # batch crash recovery the resumed batch reprocesses every result and
    # recomputes `errors` from scratch; reading + merging the prior run's
    # counters here would double-count them. The SQL `||` merge still preserves
    # unrelated keys (e.g. batch_id) already on result_metadata.
    table = fq_table("async_operations", schema)
    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {table}
            SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $2::jsonb,
                updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
            json.dumps(errors.to_dict()),
        )


async def extract_facts_from_contents_batch_api(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts using LLM Batch API (OpenAI/Groq).

    Submits all chunks as a single batch, polls until complete, then processes results.
    Only called when config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration with batch API support
        agent_name: Name of the agent
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (for storing batch state)
        operation_id: Async operation ID (for crash recovery)
        schema: Database schema (for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)
    """
    if not contents:
        return [], [], TokenUsage()

    logger.info(f"Using Batch API for fact extraction ({len(contents)} contents)")

    # Check config for causal link extraction (used throughout)
    extract_causal_links = config.retain_extract_causal_links

    # Check if we're resuming an existing batch (crash recovery). This is read
    # BEFORE the serving member is resolved: a resume must target the account
    # that owns the batch, not whichever member the current configuration would
    # pick for a fresh one.
    batch_id = None
    submitted_account: str | None = None
    submitted_provider: str | None = None
    if operation_id and pool:
        from ...db_utils import acquire_with_retry
        from ...task_backend import fq_table

        table = fq_table("async_operations", schema)
        async with acquire_with_retry(pool) as conn:
            row = await conn.fetchrow(
                f"SELECT result_metadata FROM {table} WHERE operation_id = $1",
                operation_id,
            )

        if row and row["result_metadata"]:
            metadata = row["result_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            batch_id = metadata.get("batch_id")
            if batch_id:
                submitted_account = metadata.get("batch_account")
                submitted_provider = metadata.get("batch_provider")

    # Resolve the provider implementation that serves the batch. For a multi-LLM
    # chain a fresh batch goes to the first batch-capable member (not necessarily
    # the primary); for a single provider it is the primary itself, and ``None``
    # when nothing configured can serve a batch at all. The whole batch lifecycle
    # (submit → poll → retrieve) must target this ONE impl, so resolve it once
    # and reuse it.
    #
    # Resuming pins the lookup to the account that submitted the batch. The chain
    # configuration can change between submit and resume — a member added,
    # removed, reordered, or given batch capacity — and two members of the same
    # provider on different credentials are indistinguishable by provider name,
    # so "first batch-capable member" can resolve to an account that has never
    # seen this batch id (#3671).
    batch_impl = await llm_config.batch_provider_impl(account_key=submitted_account)

    if batch_impl is None:
        if batch_id:
            # Polling an account that does not own the batch would hang until the
            # wall clock ran out and then report a provider error nobody can act
            # on. Fail before the first poll instead, naming both sides.
            raise RuntimeError(
                f"Cannot resume batch {batch_id}: it was submitted by the LLM member "
                f"'{submitted_account or submitted_provider}', which the retain LLM "
                f"configuration no longer serves batch from. Restore the LLM member "
                f"(provider, base URL and API key) that submitted it, or fail this "
                f"operation and retain again."
            )
        raise RuntimeError(
            f"retain_batch_enabled=True but provider '{llm_config.provider}' does not "
            f"support the batch API. This should have been caught at startup — check "
            f"HINDSIGHT_API_RETAIN_BATCH_ENABLED and your LLM provider configuration."
        )

    if batch_id:
        # Batches submitted before ``batch_account`` was persisted carry only the
        # provider name; keep guarding those on the coarse signal we do have.
        if submitted_account is None and submitted_provider and submitted_provider != batch_impl.provider:
            raise RuntimeError(
                f"Cannot resume batch {batch_id}: it was submitted to "
                f"'{submitted_provider}' but the retain LLM configuration now "
                f"serves batch from '{batch_impl.provider}'. Restore the LLM "
                f"member that submitted it, or fail this operation and retain again."
            )
        logger.info(f"Resuming existing batch: batch_id={batch_id} (crash recovery)")

    # Step 1: Chunk all contents and build batch requests (skip if resuming)
    all_chunks_info = []  # List of (chunk_text, content_index, chunk_index_in_content, event_date, context)
    batch_requests = []

    # Build prompt and schema once (same for all chunks)
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    for content_index, item in enumerate(contents):
        chunks = chunk_text(
            item.content,
            max_chars=config.retain_chunk_size,
            structured_chunk_size=config.retain_structured_chunk_size,
        )

        for chunk_index_in_content, chunk in enumerate(chunks):
            all_chunks_info.append((chunk, content_index, chunk_index_in_content, item.event_date, item.context))

            # Build batch request for this chunk
            custom_id = f"chunk_{len(all_chunks_info) - 1}"  # Global chunk index

            # Build user message using helper function
            user_message = _build_user_message(
                chunk,
                chunk_index_in_content,
                len(chunks),
                item.event_date,
                item.context,
                item.metadata or None,
                agent_name,
                mission_preamble=_retain_mission_preamble(config),
            )

            # Build request body using helper function
            request_body = _build_request_body(batch_impl, config, prompt, user_message, response_schema)

            batch_requests.append(
                {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
            )

    if not batch_requests and not batch_id:  # No requests and not resuming
        return [], [], TokenUsage()

    # Step 2: Submit batch (skip if resuming)
    if not batch_id:
        logger.info(f"Submitting batch with {len(batch_requests)} chunk requests")

        batch_metadata = await batch_impl.submit_batch(batch_requests)
        batch_id = batch_metadata["batch_id"]

        logger.info(f"Batch submitted: {batch_id}, polling every {config.retain_batch_poll_interval_seconds}s")

        # CRITICAL: Store minimal batch state in operation metadata for crash recovery
        # This allows resuming polling if worker restarts
        if operation_id and pool:
            batch_state = {
                "batch_id": batch_id,
                "batch_provider": batch_impl.provider,
                # Binds the batch to the exact account that owns it, so a resume
                # after a member reorder resolves that account instead of a
                # same-provider lookalike (#3671). Non-secret by construction.
                "batch_account": batch_impl.batch_account_key,
                "chunk_count": len(batch_requests),
            }

            # Update operation result_metadata
            from ...db_utils import acquire_with_retry
            from ...task_backend import fq_table

            table = fq_table("async_operations", schema)
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET result_metadata = result_metadata || $1::jsonb, updated_at = now()
                    WHERE operation_id = $2
                    """,
                    json.dumps(batch_state),
                    operation_id,
                )
            logger.info(f"Stored batch state for operation {operation_id} (crash recovery enabled)")
    else:
        logger.info(f"Resuming polling for existing batch: {batch_id}")

    # Step 3: Poll until complete
    import time

    start_time = time.time()
    while True:
        status_info = await batch_impl.get_batch_status(batch_id)
        status = status_info["status"]

        elapsed = time.time() - start_time
        logger.info(
            f"Batch {batch_id}: status={status}, "
            f"completed={status_info['request_counts']['completed']}/{status_info['request_counts']['total']}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            break
        elif status in ("failed", "expired", "cancelled"):
            error_msg = status_info.get("errors", "Unknown error")
            raise RuntimeError(f"Batch {batch_id} failed with status {status}: {error_msg}")

        # Wait before polling again
        await asyncio.sleep(config.retain_batch_poll_interval_seconds)

    logger.info(f"Batch {batch_id} completed in {elapsed:.0f}s, retrieving results")

    # Step 4: Retrieve results
    batch_results = await batch_impl.retrieve_batch_results(batch_id)

    # Map results by custom_id
    results_by_id = {result["custom_id"]: result for result in batch_results}

    # Step 5: Parse results into facts (same as sync mode)
    all_facts_from_llm = []
    chunks_metadata = []
    total_usage = TokenUsage()
    extraction_errors = RetainExtractionErrors()

    for chunk_idx, (chunk_content, content_index, chunk_index_in_content, event_date, context) in enumerate(
        all_chunks_info
    ):
        custom_id = f"chunk_{chunk_idx}"
        result = results_by_id.get(custom_id)

        if not result:
            message = f"{custom_id}: missing batch result"
            logger.warning(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Check for errors
        if result.get("error"):
            message = f"{custom_id}: {result['error']}"
            logger.error(f"Error in {message}")
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Extract response
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])

        if not choices:
            message = f"{custom_id}: no choices in response"
            logger.warning(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse JSON content
        message = choices[0].get("message", {})
        content_str = message.get("content", "{}")

        try:
            # #2701: use the lenient parser (strips markdown fences, scrubs
            # embedded control chars) so recoverable batch responses — e.g.
            # transient Gemini quirks — aren't dropped along with all their facts.
            extraction_response_json = parse_llm_json(content_str)
        except json.JSONDecodeError as e:
            message = f"{custom_id}: failed to parse JSON: {e}"
            logger.error(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        response_type_name = type(extraction_response_json).__name__
        extraction_response_json = _coerce_fact_response(extraction_response_json)
        if extraction_response_json is None:
            message = f"{custom_id}: LLM returned non-dict JSON ({response_type_name})"
            logger.error(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse facts (reuse existing logic from _extract_facts_from_chunk)
        raw_facts = extraction_response_json.get("facts", [])
        chunk_facts = []

        for i, llm_fact in enumerate(raw_facts):
            if not isinstance(llm_fact, dict):
                continue

            def get_value(field_name):
                value = llm_fact.get(field_name)
                if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                    return value
                return None

            what = get_value("what")
            if not what:
                what = get_value("factual_core")
            if not what:
                what = get_value("text")
            if not what:
                # Same schema-drift signal as the streaming path (#3708): a fact object
                # carrying none of the text keys means the model ignored the schema.
                # The batch API cannot re-prompt a single request, so record it on the
                # operation instead — that is what extraction_errors is for, and
                # HINDSIGHT_API_FAIL_ON_EXTRACTION_ERRORS can escalate it to a failure.
                # A key that is present but empty/"N/A" stays a quiet skip.
                if not any(key in llm_fact for key in ("what", "factual_core", "text")):
                    message = f"{custom_id}: fact {i} has no 'what'/'factual_core'/'text' field"
                    logger.warning(message)
                    extraction_errors.add(message)
                continue

            when = get_value("when")
            who = get_value("who")
            why = get_value("why")

            # Critical field: fact_type — only "assistant" maps to "experience", everything else is "world"
            # Critical field: fact_type — "assistant" maps to "experience", everything else is "world".
            # If fact_type is unexpected, fall back to fact_kind before defaulting to "world".
            raw_fact_type = llm_fact.get("fact_type")
            if raw_fact_type == "assistant":
                fact_type = "experience"
            elif raw_fact_type == "world":
                fact_type = "world"
            else:
                raw_fact_kind = llm_fact.get("fact_kind")
                fact_type = "experience" if raw_fact_kind == "assistant" else "world"

            # Build combined fact text
            combined_parts = [what]
            if when:
                combined_parts.append(f"When: {when}")
            if who:
                combined_parts.append(f"Involving: {who}")
            if why:
                combined_parts.append(why)
            combined_text = " | ".join(combined_parts)

            # Temporal fields
            fact_data = {}
            fact_kind = llm_fact.get("fact_kind", "conversation")
            if fact_kind not in ["conversation", "event", "other"]:
                fact_kind = "conversation"

            if fact_kind == "event":
                occurred_start = get_value("occurred_start")
                occurred_end = get_value("occurred_end")

                if not occurred_start:
                    fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                else:
                    fact_data["occurred_start"] = occurred_start

                if occurred_end:
                    fact_data["occurred_end"] = occurred_end
                elif fact_data.get("occurred_start"):
                    fact_data["occurred_end"] = fact_data["occurred_start"]

            # Entities are plain strings. Older prompts taught a {"text": ...}
            # object form, so keep unwrapping it for models that still emit it.
            validated_entities = _coerce_entity_strings(get_value("entities"))

            # Post-process label entities from structured labels object
            entity_labels_raw = config.entity_labels
            labels_cfg_batch = parse_entity_labels(entity_labels_raw)
            free_form_entities_batch = config.entities_allow_free_form
            if labels_cfg_batch and labels_cfg_batch.attributes:
                labels_lookup_batch = build_labels_lookup(labels_cfg_batch)
                labels_data = llm_fact.get("labels") or {}
                if isinstance(labels_data, dict):
                    existing_texts_lower = {e.lower() for e in validated_entities}
                    for group in labels_cfg_batch.attributes:
                        value = labels_data.get(group.key)
                        if not value:
                            continue
                        # Map-type groups: recursively extract key:field:value strings
                        if group.type == "map" and group.fields:
                            entities_list = value if isinstance(value, list) else [value]
                            for entity_obj in entities_list:
                                if isinstance(entity_obj, dict):
                                    _extract_map_entities(
                                        entity_obj,
                                        group.fields,
                                        f"{group.key}:",
                                        validated_entities,
                                        existing_texts_lower,
                                    )
                            continue
                        values_list = value if isinstance(value, list) else [value]
                        for v in values_list:
                            if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                                continue
                            label_str = f"{group.key}:{v.strip()}"
                            if group.type == "text":
                                if label_str.lower() not in existing_texts_lower:
                                    validated_entities.append(label_str)
                                    existing_texts_lower.add(label_str.lower())
                            elif (
                                label_str.lower() in labels_lookup_batch
                                and label_str.lower() not in existing_texts_lower
                            ):
                                validated_entities.append(label_str)
                                existing_texts_lower.add(label_str.lower())

                if not free_form_entities_batch:
                    validated_entities = [
                        e for e in validated_entities if is_label_entity(e, labels_cfg_batch, labels_lookup_batch)
                    ]
            elif not free_form_entities_batch:
                validated_entities = []

            if validated_entities:
                fact_data["entities"] = validated_entities

            # Causal relations
            if extract_causal_links:
                validated_relations = []
                causal_relations_raw = get_value("causal_relations")
                if causal_relations_raw:
                    for rel in causal_relations_raw:
                        if not isinstance(rel, dict):
                            continue
                        target_idx = rel.get("target_index")
                        relation_type = rel.get("relation_type")

                        if target_idx is None or relation_type is None:
                            continue
                        if target_idx < 0 or target_idx >= i:
                            continue

                        try:
                            validated_relations.append(
                                CausalRelation(target_fact_index=target_idx, relation_type=relation_type)
                            )
                        except Exception:
                            pass

                if validated_relations:
                    fact_data["causal_relations"] = validated_relations

            # Set mentioned_at to the event_date (when the conversation/document occurred),
            # or None when the caller opted into no timestamp.
            fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

            try:
                fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                chunk_facts.append(fact)
            except Exception as e:
                message = f"{custom_id}: failed to create Fact model for fact {i}: {e}"
                logger.error(message)
                extraction_errors.add(message)
                continue

        all_facts_from_llm.extend(chunk_facts)
        chunks_metadata.append(
            ChunkMetadata(
                chunk_text=chunk_content,
                fact_count=len(chunk_facts),
                content_index=content_index,
                chunk_index=chunk_idx,
            )
        )

        # Track token usage
        usage_data = response_body.get("usage", {})
        if usage_data:
            total_usage = total_usage + TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

    # Step 6: Convert to ExtractedFact objects with proper chunk mapping
    # Group facts by chunk
    facts_by_chunk = []  # List of (chunk_metadata, [facts])
    fact_start_idx = 0

    for chunk_meta in chunks_metadata:
        chunk_facts = all_facts_from_llm[fact_start_idx : fact_start_idx + chunk_meta.fact_count]
        facts_by_chunk.append((chunk_meta, chunk_facts))
        fact_start_idx += chunk_meta.fact_count

    # Now convert to ExtractedFactType
    extracted_facts = []
    global_fact_idx = 0

    for chunk_meta, chunk_facts in facts_by_chunk:
        content = contents[chunk_meta.content_index]
        extraction_group_start_idx = global_fact_idx

        for fact_from_llm in chunk_facts:
            extracted_fact = ExtractedFactType(
                fact_text=fact_from_llm.fact,
                fact_type=fact_from_llm.fact_type,
                entities=list(fact_from_llm.entities or []),
                occurred_start=_parse_datetime(fact_from_llm.occurred_start) if fact_from_llm.occurred_start else None,
                occurred_end=_parse_datetime(fact_from_llm.occurred_end) if fact_from_llm.occurred_end else None,
                causal_relations=_convert_causal_relations(
                    fact_from_llm.causal_relations or [], extraction_group_start_idx, len(chunk_facts)
                ),
                content_index=chunk_meta.content_index,
                chunk_index=chunk_meta.chunk_index,
                context=content.context,
                mentioned_at=content.event_date,
                metadata=content.metadata,
                tags=content.tags,
                observation_scopes=content.observation_scopes,
            )

            extracted_facts.append(extracted_fact)
            global_fact_idx += 1

    # Step 7: Add temporal offsets
    _add_temporal_offsets(extracted_facts, contents)

    # Step 8: Auto-tag facts from label groups with tag=True
    _inject_label_tags(extracted_facts, config)

    await _write_batch_extraction_errors(pool, operation_id, schema, extraction_errors)

    logger.info(f"Batch API extracted {len(extracted_facts)} facts from {len(all_chunks_info)} chunks")

    return extracted_facts, chunks_metadata, total_usage
