"""Shared normalization helpers for fact extraction."""

import re
from datetime import datetime, timedelta
from typing import Any

from ...llm_wrapper import sanitize_llm_output
from ..entity_labels import (
    MapField,
    parse_entity_labels,
)
from ..types import CausalRelation as CausalRelationType
from ..types import ExtractedFact as ExtractedFactType
from ..types import RetainContent


def _extract_map_entities(
    entity_obj: dict,
    fields: dict[str, MapField],
    prefix: str,
    validated_entities: list[str],
    existing_texts_lower: set[str],
) -> None:
    """Recursively extract key:field:value entity strings from a map entity dict."""
    for field_name, map_field in fields.items():
        field_val = entity_obj.get(field_name)
        if field_val is None or field_val == "":
            continue
        if map_field.type == "map" and map_field.fields:
            # Nested map: recurse into each sub-entity
            sub_list = field_val if isinstance(field_val, list) else [field_val]
            for sub_obj in sub_list:
                if isinstance(sub_obj, dict):
                    _extract_map_entities(
                        sub_obj,
                        map_field.fields,
                        f"{prefix}{field_name}:",
                        validated_entities,
                        existing_texts_lower,
                    )
        elif map_field.type == "multi-values":
            vals = field_val if isinstance(field_val, list) else [field_val]
            for v in vals:
                if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                    continue
                label_str = f"{prefix}{field_name}:{v.strip()}"
                if label_str.lower() not in existing_texts_lower:
                    validated_entities.append(label_str)
                    existing_texts_lower.add(label_str.lower())
        else:
            # text or value — single string
            if not isinstance(field_val, str) or not field_val.strip() or field_val.lower() in ("none", "null", "n/a"):
                continue
            label_str = f"{prefix}{field_name}:{field_val.strip()}"
            if label_str.lower() not in existing_texts_lower:
                validated_entities.append(label_str)
                existing_texts_lower.add(label_str.lower())


def _infer_temporal_date(fact_text: str, event_date: datetime | None) -> str | None:
    """
    Infer a temporal date from fact text when LLM didn't provide occurred_start.

    This is a fallback for when the LLM fails to extract temporal information
    from relative time expressions like "last night", "yesterday", etc.
    """
    if event_date is None:
        return None

    fact_lower = fact_text.lower()

    # Map relative time expressions to day offsets
    temporal_patterns = {
        r"\blast night\b": -1,
        r"\byesterday\b": -1,
        r"\btoday\b": 0,
        r"\bthis morning\b": 0,
        r"\bthis afternoon\b": 0,
        r"\bthis evening\b": 0,
        r"\btonigh?t\b": 0,
        r"\btomorrow\b": 1,
        r"\blast week\b": -7,
        r"\bthis week\b": 0,
        r"\bnext week\b": 7,
        r"\blast month\b": -30,
        r"\bthis month\b": 0,
        r"\bnext month\b": 30,
    }

    for pattern, offset_days in temporal_patterns.items():
        if re.search(pattern, fact_lower):
            target_date = event_date + timedelta(days=offset_days)
            return target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # If no relative time expression found, return None
    return None


def _sanitize_text(text: str | None) -> str | None:
    return sanitize_llm_output(text)


def _coerce_entity_strings(v: Any) -> Any:
    """
    Normalize the LLM's `entities` field to a plain list of strings.

    The schema previously asked for `Entity` objects ({"text": "..."}) while the
    prompt's few-shot examples taught a flat string array. Models that followed
    the examples literally returned strings, and the entities were silently
    dropped — none were ever persisted (#2749). The `Entity` wrapper carried no
    information beyond the string, so it was removed rather than taught to the
    prompt; the object form is still unwrapped here for models that learned it
    and for in-flight batch jobs.

    Returns non-list input untouched so pydantic reports the type error itself.
    """
    if v is None:
        return []
    if not isinstance(v, list):
        return v
    coerced = []
    for item in v:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                coerced.append(text)
        else:
            coerced.append(item)
    return coerced


def _coerce_fact_response(response: Any) -> dict[str, Any] | None:
    """Accept the schema wrapper, or a recoverable top-level facts array."""
    if isinstance(response, dict):
        return response
    if isinstance(response, list) and all(isinstance(item, dict) for item in response):
        return {"facts": response}
    return None


# Each fact gets 10ms offset to preserve ordering within a document
SECONDS_PER_FACT = 0.01


def _parse_datetime(date_str: str):
    """Parse ISO datetime string."""
    from dateutil import parser as date_parser

    try:
        return date_parser.isoparse(date_str)
    except Exception:
        return None


def _convert_causal_relations(
    relations_from_llm, extraction_group_start_idx: int, extraction_group_size: int
) -> list[CausalRelationType]:
    """
    Convert causal relations from LLM format to ExtractedFact format.

    Adjusts target_fact_index from content-relative to global indices.
    """
    causal_relations = []
    for rel in relations_from_llm:
        target_fact_index = rel.target_fact_index
        if (
            not isinstance(target_fact_index, int)
            or isinstance(target_fact_index, bool)
            or not 0 <= target_fact_index < extraction_group_size
        ):
            continue
        causal_relation = CausalRelationType(
            relation_type=rel.relation_type,
            target_fact_index=extraction_group_start_idx + target_fact_index,
        )
        causal_relations.append(causal_relation)
    return causal_relations


def _add_temporal_offsets(facts: list[ExtractedFactType], contents: list[RetainContent]) -> None:
    """
    Add time offsets to preserve fact ordering across all contents.

    This allows retrieval to distinguish between facts from different documents/conversations
    even when they have the same base event_date, and also between facts within the same
    conversation.

    Uses absolute position across all facts to ensure unique timestamps.

    Modifies facts in place.
    """
    from ..orchestrator import parse_datetime_flexible

    for i, fact in enumerate(facts):
        # Use absolute position across all facts to ensure uniqueness across different contents
        offset = timedelta(seconds=i * SECONDS_PER_FACT)

        # Apply offset to all temporal fields (handle both datetime objects and ISO strings)
        if fact.occurred_start:
            fact.occurred_start = parse_datetime_flexible(fact.occurred_start) + offset
        if fact.occurred_end:
            fact.occurred_end = parse_datetime_flexible(fact.occurred_end) + offset
        if fact.mentioned_at:
            fact.mentioned_at = parse_datetime_flexible(fact.mentioned_at) + offset


def _inject_label_tags(facts: list[ExtractedFactType], config) -> None:
    """
    For label groups with tag=True, add extracted key:value label entities
    to each fact's tags list. Modifies facts in place.

    This lets entity labels double as tags, enabling filtering via the
    existing tags API without any extra query infrastructure.
    """
    labels_cfg = parse_entity_labels(config.entity_labels)
    if not labels_cfg:
        return
    tag_group_keys = {g.key.lower() for g in labels_cfg.attributes if g.tag}
    if not tag_group_keys:
        return
    for fact in facts:
        label_tags = [e for e in fact.entities if ":" in e and e.split(":", 1)[0].lower() in tag_group_keys]
        if label_tags:
            existing = set(fact.tags)
            fact.tags = fact.tags + [t for t in label_tags if t not in existing]
