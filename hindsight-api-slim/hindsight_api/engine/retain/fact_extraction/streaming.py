"""Streaming fact extraction with retries and output auto-splitting."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ...llm_interface import ProviderContentPolicyError, ProviderRateLimitResetError
from ...llm_wrapper import LLMConfig, OutputTooLongError
from ...response_models import TokenUsage
from ..entity_labels import build_labels_lookup, is_label_entity, parse_entity_labels
from .chunking import _split_chunk_for_output_retry, chunk_text
from .models import CausalRelation, Fact
from .normalization import (
    _coerce_entity_strings,
    _coerce_fact_response,
    _extract_map_entities,
    _infer_temporal_date,
)
from .prompts import _build_extraction_prompt_and_schema, _build_user_message, _retain_mission_preamble

logger = logging.getLogger(__name__)


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: "LLMConfig",
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    import logging

    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Build prompt and schema using helper function
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    # Check config for extraction mode and causal link extraction
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # Build user message — the bank mission rides here (not in the cached prefix).
    user_message = _build_user_message(
        chunk,
        chunk_index,
        total_chunks,
        event_date,
        context,
        metadata,
        agent_name,
        mission_preamble=_retain_mission_preamble(config),
    )

    # Opt into context caching when the provider supports it. The prompt and
    # response_schema are bank-agnostic (the mission lives in the user message),
    # so one cached prefix serves every bank; reusing it across many small-payload
    # retain calls dramatically lowers per-call input
    # cost. ``get_or_create_cached_prefix`` returns None when caching is
    # disabled, unsupported, or the prefix is too small; the LLM call
    # transparently falls back to the uncached path in that case.
    cached_prefix_name: str | None = None
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is not None and provider_impl.supports_prompt_caching():
        try:
            cached_prefix_name = await provider_impl.get_or_create_cached_prefix(
                system_instruction=prompt,
                response_schema=response_schema,
            )
        except Exception:
            # Caching is a soft optimisation — never let a cache-side
            # error block a retain operation.
            logger.exception("Cache prefix lookup failed; falling back to uncached call")
            cached_prefix_name = None

    # Retry logic for JSON validation errors
    # Use retain-specific overrides if set, otherwise fall back to global LLM config
    llm_max_retries = (
        config.retain_llm_max_retries if config.retain_llm_max_retries is not None else config.llm_max_retries
    )
    # OUTER content-validation attempts (re-prompts on malformed JSON). Follows the
    # same `N + 1` convention as the providers' transport-retry loops — N retries after
    # the initial request — so a zero budget still performs one request (#2731). The raw
    # budget is forwarded unchanged to llm_config.call(), which owns transport retries.
    outer_attempts = llm_max_retries + 1
    last_error: Exception | None = None

    usage = TokenUsage()  # Track cumulative usage across retries
    for attempt in range(outer_attempts):
        try:
            initial_backoff = (
                config.retain_llm_initial_backoff
                if config.retain_llm_initial_backoff is not None
                else config.llm_initial_backoff
            )
            max_backoff = (
                config.retain_llm_max_backoff if config.retain_llm_max_backoff is not None else config.llm_max_backoff
            )

            call_kwargs: dict[str, Any] = dict(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=response_schema,
                scope="retain_extract_facts",
                temperature=config.llm_temperature_retain,
                strict_schema=config.llm_strict_schema_retain,
                max_completion_tokens=config.retain_max_completion_tokens,
                max_retries=llm_max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=True,  # Get raw JSON, we'll validate leniently
                return_usage=True,
            )
            if cached_prefix_name is not None:
                call_kwargs["cached_prefix"] = cached_prefix_name

            extraction_response_json, call_usage = await llm_config.call(**call_kwargs)
            usage = usage + call_usage  # Aggregate usage across retries

            # Lenient parsing of facts from raw JSON
            chunk_facts = []
            has_malformed_facts = False

            # Handle malformed LLM responses
            coerced_response_json = _coerce_fact_response(extraction_response_json)
            if coerced_response_json is None:
                if attempt < outer_attempts - 1:
                    logger.warning(
                        f"LLM returned non-dict JSON on attempt {attempt + 1}/{outer_attempts}: {type(extraction_response_json).__name__}. Retrying..."
                    )
                    continue
                else:
                    # A non-dict response is malformed (the schema is {"facts": [...]}).
                    # Raise instead of returning [] so the failure propagates to the
                    # worker's retry machinery and ultimately fails loudly — never
                    # silently commit the document with 0 facts. See issue #1833.
                    raise RuntimeError(
                        f"Fact extraction failed: LLM returned non-dict JSON after {outer_attempts} attempts "
                        f"({type(extraction_response_json).__name__}). Raw: {str(extraction_response_json)[:500]}"
                    )
            extraction_response_json = coerced_response_json

            raw_facts = extraction_response_json.get("facts", [])

            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat() if event_date else 'unset'}, "
                    f"context: {context if context else 'none'}, "
                    f"text: {chunk}"
                )

            for i, llm_fact in enumerate(raw_facts):
                # Skip non-dict entries but track them for retry
                if not isinstance(llm_fact, dict):
                    logger.warning(f"Skipping non-dict fact at index {i}")
                    has_malformed_facts = True
                    continue

                # Helper to get non-empty value
                def get_value(field_name):
                    value = llm_fact.get(field_name)
                    if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                        return value
                    return None

                # NEW FORMAT: what, when, who, why (all required)
                what = get_value("what")
                when = get_value("when")
                who = get_value("who")
                why = get_value("why")

                # Fallback to old format if new fields not present
                if not what:
                    what = get_value("factual_core")
                if not what:
                    what = get_value("text")
                if not what:
                    # In verbatim mode, 'what' is intentionally absent — text is backfilled from chunk
                    if extraction_mode != "verbatim":
                        logger.warning(f"Skipping fact {i}: missing 'what' field")
                        # Count it as malformed so the re-prompt below covers this case too.
                        # A model that emits well-formed JSON with the wrong field shape (no
                        # schema enforcement, e.g. JSON-mode-only models) otherwise drops every
                        # fact on the first attempt and returns [] without ever retrying, and
                        # the retain still completes — silent data loss (#3708).
                        #
                        # Only *absent* text keys count. A key that is present but empty or
                        # "N/A" is the model saying "nothing to extract here", which re-prompting
                        # cannot improve — skip it as quietly as before.
                        if not any(key in llm_fact for key in ("what", "factual_core", "text")):
                            has_malformed_facts = True
                        continue

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

                # Get fact_kind for temporal handling (but don't store it)
                fact_kind = llm_fact.get("fact_kind", "conversation")
                if fact_kind not in ["conversation", "event", "other"]:
                    fact_kind = "conversation"

                # Build combined fact text from the 4 dimensions: what | when | who | why
                # In verbatim mode, leave combined_text empty — _collapse_to_verbatim backfills it
                fact_data = {}
                if extraction_mode == "verbatim":
                    combined_text = ""
                else:
                    combined_parts = [what]

                    if when:
                        combined_parts.append(f"When: {when}")

                    if who:
                        combined_parts.append(f"Involving: {who}")

                    if why:
                        combined_parts.append(why)

                    combined_text = " | ".join(combined_parts)

                # Add temporal fields
                # For events: occurred_start/occurred_end (when the event happened)
                if fact_kind == "event":
                    occurred_start = get_value("occurred_start")
                    occurred_end = get_value("occurred_end")

                    # If LLM didn't set temporal fields, try to extract them from the fact text
                    if not occurred_start:
                        fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                    else:
                        fact_data["occurred_start"] = occurred_start

                    # For point events: if occurred_end not set, default to occurred_start
                    if occurred_end:
                        fact_data["occurred_end"] = occurred_end
                    elif fact_data.get("occurred_start"):
                        fact_data["occurred_end"] = fact_data["occurred_start"]

                # Entities are plain strings. Older prompts taught a {"text": ...}
                # object form, so keep unwrapping it for models that still emit it.
                validated_entities = _coerce_entity_strings(get_value("entities"))

                # Post-process label entities from structured labels object
                entity_labels_raw = config.entity_labels
                labels_cfg = parse_entity_labels(entity_labels_raw)
                free_form_entities = config.entities_allow_free_form
                if labels_cfg and labels_cfg.attributes:
                    labels_lookup = build_labels_lookup(labels_cfg)
                    labels_data = llm_fact.get("labels") or {}
                    if isinstance(labels_data, dict):
                        existing_texts_lower = {e.lower() for e in validated_entities}
                        for group in labels_cfg.attributes:
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
                                    label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower
                                ):
                                    validated_entities.append(label_str)
                                    existing_texts_lower.add(label_str.lower())
                                else:
                                    logger.warning(f"Label '{label_str}' not in valid label values, skipping")

                    # In labels-only mode, keep only label entities
                    if not free_form_entities:
                        validated_entities = [
                            e for e in validated_entities if is_label_entity(e, labels_cfg, labels_lookup)
                        ]
                elif not free_form_entities:
                    # No labels but free_form disabled: clear all entities
                    validated_entities = []

                if validated_entities:
                    fact_data["entities"] = validated_entities

                # Add per-fact causal relations (only if enabled in config)
                if extract_causal_links:
                    validated_relations = []
                    causal_relations_raw = get_value("causal_relations")
                    if causal_relations_raw:
                        for rel in causal_relations_raw:
                            if not isinstance(rel, dict):
                                continue
                            # New schema uses target_index
                            target_idx = rel.get("target_index")
                            relation_type = rel.get("relation_type")

                            if target_idx is None or relation_type is None:
                                continue

                            # Validate: target_index must be < current fact index
                            if target_idx < 0 or target_idx >= i:
                                logger.debug(
                                    f"Invalid target_index {target_idx} for fact {i} (must be 0 to {i - 1}). Skipping."
                                )
                                continue

                            try:
                                validated_relations.append(
                                    CausalRelation(
                                        target_fact_index=target_idx,
                                        relation_type=relation_type,
                                    )
                                )
                            except Exception as e:
                                logger.debug(f"Invalid causal relation {rel}: {e}")

                    if validated_relations:
                        fact_data["causal_relations"] = validated_relations

                # Set mentioned_at to the event_date (when the conversation/document occurred),
                # or None when the caller opted into no timestamp.
                fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

                # Build Fact model instance
                try:
                    fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                    chunk_facts.append(fact)
                except Exception as e:
                    logger.error(f"Failed to create Fact model for fact {i}: {e}")
                    has_malformed_facts = True
                    continue

            # If we got malformed facts and haven't exhausted retries, try again
            if has_malformed_facts and len(chunk_facts) < len(raw_facts) * 0.8 and attempt < outer_attempts - 1:
                logger.warning(
                    f"Got {len(raw_facts) - len(chunk_facts)} malformed facts out of {len(raw_facts)} on attempt {attempt + 1}/{outer_attempts}. Retrying..."
                )
                continue

            # Every fact the model returned was unusable, on every attempt. Raise
            # instead of returning [] so the failure reaches the worker's retry
            # machinery and ultimately fails the operation loudly — the same rule the
            # non-dict response above follows (#1833). Without this the retain commits
            # a document with 0 memory units and reports `completed`, so callers cannot
            # tell schema-drifted extraction from content that genuinely held no facts
            # (#3708). A model that legitimately returns `"facts": []` never lands here:
            # nothing was dropped, so has_malformed_facts stays False.
            if has_malformed_facts and not chunk_facts:
                raise RuntimeError(
                    f"Fact extraction failed: all {len(raw_facts)} facts returned by the LLM were "
                    f"unusable after {outer_attempts} attempts (wrong shape or missing required fields). "
                    f"Model '{llm_config.model}' may not honour the extraction schema — consider enabling "
                    f"HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN or using a model with strict schema support."
                )

            return chunk_facts, usage

        except BadRequestError as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is related to max_tokens/completion_tokens not being supported
            if any(
                keyword in error_str
                for keyword in [
                    "max_tokens",
                    "max_completion_tokens",
                    "maximum context",
                    "token limit",
                    "context length",
                ]
            ):
                # Provide helpful error message with configuration suggestions
                raise ValueError(
                    f"Model does not support the required output token limit.\n\n"
                    f"The model '{llm_config.model}' (provider: {llm_config.provider}) failed with: {e}\n\n"
                    f"You have two options to fix this:\n"
                    f"  1. Use a different model that supports at least {config.retain_max_completion_tokens} output tokens\n"
                    f"  2. Decrease HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS to a value your model supports\n"
                    f"     (current value: {config.retain_max_completion_tokens}, must be > RETAIN_CHUNK_SIZE={config.retain_chunk_size})"
                ) from e

            # Don't retry json_validate_failed here — the inner provider
            # loop already retried the 400 error.  Re-entering the LLM call
            # with the same input just multiplies wasted calls.
            raise

    # If we exhausted all retries, raise the last error or a descriptive fallback
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Fact extraction failed after {outer_attempts} attempts: LLM did not return valid JSON")


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: LLMConfig,
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a chunk with automatic splitting if output exceeds token limits.

    If the LLM output is too long (OutputTooLongError), this function automatically
    splits the chunk in half and processes each half recursively.

    Args:
        chunk: Text chunk to process
        chunk_index: Index of this chunk in the original list
        total_chunks: Total number of original chunks
        event_date: Reference date for temporal information
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        config: Resolved HindsightConfig for this bank
        agent_name: Optional agent name (memory owner)
        metadata: Optional document metadata key-value pairs

    Returns:
        Tuple of (facts list, token usage) extracted from the chunk (possibly from sub-chunks)
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Try to extract facts from the full chunk
        return await _extract_facts_from_chunk(
            chunk=chunk,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            config=config,
            agent_name=agent_name,
            metadata=metadata,
        )
    except OutputTooLongError:
        # Output exceeded token limits - split the chunk and retry. Conversation
        # chunks are JSON arrays, so preserve array/turn boundaries when possible.
        logger.warning(
            f"Output too long for chunk {chunk_index + 1}/{total_chunks} "
            f"({len(chunk)} chars). Splitting and retrying..."
        )

        split_chunks = _split_chunk_for_output_retry(chunk)
        if split_chunks is None:
            logger.warning(
                f"Cannot make progress splitting chunk {chunk_index + 1}/{total_chunks} "
                f"({len(chunk)} chars); dropping this sub-chunk."
            )
            return [], TokenUsage()

        first_half, second_half = split_chunks

        logger.info(
            f"Split chunk {chunk_index + 1} into two sub-chunks: {len(first_half)} chars and {len(second_half)} chars"
        )

        # Process both halves recursively (in parallel)
        sub_tasks = [
            _extract_facts_with_auto_split(
                chunk=first_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                config=config,
                agent_name=agent_name,
                metadata=metadata,
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                config=config,
                agent_name=agent_name,
                metadata=metadata,
            ),
        ]

        sub_results = await asyncio.gather(*sub_tasks)

        # Combine results from both halves
        all_facts = []
        total_usage = TokenUsage()
        for sub_facts, sub_usage in sub_results:
            all_facts.extend(sub_facts)
            total_usage = total_usage + sub_usage

        logger.info(f"Successfully extracted {len(all_facts)} facts from split chunk {chunk_index + 1}")

        return all_facts, total_usage


async def extract_facts_from_text(
    text: str,
    event_date: datetime | None,
    llm_config: LLMConfig,
    agent_name: str,
    config,
    context: str = "",
    metadata: dict[str, str] | None = None,
) -> tuple[list[Fact], list[tuple[str, int]], TokenUsage]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>3000 chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    If a chunk produces output that exceeds token limits (OutputTooLongError), it is
    automatically split in half and retried recursively until successful.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        llm_config: LLM configuration to use
        agent_name: Agent name (memory owner)
        config: Resolved HindsightConfig for this bank
        context: Context about the conversation/document
        metadata: Optional document metadata key-value pairs

    Returns:
        Tuple of (facts, chunks, usage) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
        - usage: Aggregated token usage across all LLM calls
    """
    chunks = chunk_text(
        text,
        max_chars=config.retain_chunk_size,
        structured_chunk_size=config.retain_structured_chunk_size,
    )

    # Log chunk count before starting LLM requests
    total_chars = sum(len(c) for c in chunks)
    if len(chunks) > 1:
        logger.debug(
            f"[FACT_EXTRACTION] Text chunked into {len(chunks)} chunks ({total_chars:,} chars total, "
            f"chunk_size={config.retain_chunk_size:,}) - starting parallel LLM extraction"
        )

    # Transient LLM failures (timeouts, rate limits) are already retried inside
    # the provider's inner loop.  Content-quality retries (malformed facts) are
    # handled by the middle loop in _extract_facts_from_chunk.  Adding a third
    # retry layer here would multiply wasted calls on deterministic failures
    # (see https://github.com/vectorize-io/hindsight/issues/1412).
    tasks = [
        _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            config=config,
            agent_name=agent_name,
            metadata=metadata,
        )
        for i, chunk in enumerate(chunks)
    ]

    # return_exceptions=True so we can collect all results even if some chunks
    # exhausted their retries. We check for failures below and fail the retain
    # if ANY chunk could not be extracted — partial extraction is not acceptable.
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_facts = []
    chunk_metadata = []  # [(chunk_text, fact_count), ...]
    total_usage = TokenUsage()
    failed_chunks = []
    for i, (chunk, result) in enumerate(zip(chunks, chunk_results)):
        if isinstance(result, Exception):
            failed_chunks.append((i, result))
            continue
        chunk_facts, chunk_usage = result
        all_facts.extend(chunk_facts)
        chunk_metadata.append((chunk, len(chunk_facts)))
        total_usage = total_usage + chunk_usage

    if failed_chunks:
        # Include the exception message — not just the type — so operators
        # can tell a structured-JSON parse failure apart from a rate limit
        # apart from a network 5xx, all of which can surface as the same
        # exception types. The error_message we propagate to the
        # async_operations row is the only inspection surface a worker-side
        # failure leaves behind, and a bare "chunk 0: RuntimeError" is not
        # actionable.
        failed_summary = ", ".join(f"chunk {idx}: {type(err).__name__}: {err}" for idx, err in failed_chunks[:5])
        quota_errors = [err for _, err in failed_chunks if isinstance(err, ProviderRateLimitResetError)]
        if quota_errors and len(quota_errors) == len(failed_chunks):
            retry_at = max(err.retry_at for err in quota_errors)
            raise ProviderRateLimitResetError(
                retry_at=retry_at,
                message=(
                    f"Fact extraction deferred by provider quota: {len(failed_chunks)}/{len(chunks)} chunks failed. "
                    f"First failures: {failed_summary}. Provider detail: {quota_errors[0]}"
                ),
            ) from quota_errors[0]

        # A content-policy refusal is deterministic: the offending chunk earns
        # the same refusal on every replay, so no amount of task-level retrying
        # can complete this retain. Re-raise the permanent type (rather than a
        # generic RuntimeError) so the worker fails the operation immediately
        # instead of burning a full retry schedule on it (issue #3690). One
        # refused chunk is enough — the retain cannot succeed while it is in the
        # batch, whatever the other failures were.
        policy_errors = [err for _, err in failed_chunks if isinstance(err, ProviderContentPolicyError)]
        if policy_errors:
            raise ProviderContentPolicyError(
                f"Fact extraction refused by provider content policy: {len(policy_errors)} of "
                f"{len(failed_chunks)} failed chunks ({len(chunks)} total) were refused; retrying cannot "
                f"succeed. First failures: {failed_summary}"
            ) from policy_errors[0]

        # Fail the entire retain — partial extraction is not acceptable.
        # All successfully extracted facts are discarded because the transaction
        # hasn't committed yet. The worker poller will retry the entire task.
        raise RuntimeError(
            f"Fact extraction failed: {len(failed_chunks)}/{len(chunks)} chunks failed. "
            f"First failures: {failed_summary}"
        )

    return all_facts, chunk_metadata, total_usage
