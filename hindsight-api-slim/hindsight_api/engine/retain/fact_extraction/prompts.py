"""Prompts, schemas, and request builders for fact extraction."""

from datetime import datetime
from typing import cast

from pydantic import ConfigDict, Field, create_model

from ...structured_output import strict_json_schema
from ..entity_labels import EntityLabelsConfig, MapField, build_labels_model, parse_entity_labels
from .models import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    FactExtractionResponse,
    FactExtractionResponseNoCausal,
    FactExtractionResponseVerbose,
    VerbatimExtractedFact,
    VerbatimFactExtractionResponse,
)
from .normalization import _sanitize_text

# Base prompt template (shared by concise and custom modes)
# Uses {extraction_guidelines} placeholder for mode-specific instructions
_BASE_FACT_EXTRACTION_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}{extraction_guidelines}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. "what": Core fact - concise but complete (1-2 sentences max)
2. "when": Temporal info if mentioned. "N/A" if none. Use day name when known.
3. "where": Location if relevant. "N/A" if none.
4. "who": People involved with relationships. "N/A" if just general info.
5. "why": Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": Objective/external facts, including the user's preferences, rules, corrections, constraints, plans, traits, or context. These stay "world" even when the user states them during an assistant interaction (e.g., "User prefers browser_navigate over web_search", "User corrected the project deadline").
- "assistant": Actions, experiences, or observations the assistant/agent actually performed (e.g., "I changed X", "I discovered Y", "I debugged Z"). Use this for the assistant/agent doing, trying, learning, deciding, recommending, or responding — not merely for user facts mentioned in conversation.

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

ALWAYS return "entities" as an array of plain strings — never objects, never null.
Correct: entities=["Alice", "Kubernetes", "CKA"]
Wrong:   entities as an array of objects with a "text" key ← never use this form
Use an empty array [] only when the fact truly names nothing.

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user.{examples}"""

# Concise mode guidelines
_CONCISE_GUIDELINES = """══════════════════════════════════════════════════════════════════════════
SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)
══════════════════════════════════════════════════════════════════════════

ONLY extract facts that are:
✅ Personal info: names, relationships, roles, background
✅ Preferences: likes, dislikes, habits, interests (e.g., "Alice likes coffee")
✅ Significant events: milestones, decisions, achievements, changes
✅ Plans/goals: future intentions, deadlines, commitments
✅ Expertise: skills, knowledge, certifications, experience
✅ Important context: projects, problems, constraints
✅ Sensory/emotional details: feelings, sensations, perceptions that provide context
✅ Observations: descriptions of people, places, things with specific details

DO NOT extract:
❌ Generic greetings: "how are you", "hello", pleasantries without substance
❌ Pure filler: "thanks", "sounds good", "ok", "got it", "sure"
❌ Process chatter: "let me check", "one moment", "I'll look into it"
❌ Repeated info: if already stated, don't extract again

CONSOLIDATE related statements into ONE fact when possible."""

# Concise mode examples
_CONCISE_EXAMPLES = """

══════════════════════════════════════════════════════════════════════════
EXAMPLES (shown in English for illustration; for non-English input, ALL output values MUST be in the input language)
══════════════════════════════════════════════════════════════════════════

Example 1 - Selective extraction (Event Date: June 10, 2024):
Input: "Hey! How's it going? Good morning! So I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee on the way."

Output: ONLY 2 facts (skip greetings, weather, coffee):
1. what="User planning wedding, wants small outdoor ceremony", who="user", why="N/A", entities=["user", "wedding"]
2. what="Emily married Sarah at rooftop garden", who="Emily (user's friend), Sarah", occurred_start="2024-06-09", entities=["Emily", "Sarah", "wedding"]

Example 2 - Professional context:
Input: "Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output: ONLY 2 facts (skip coffee preference - too trivial):
1. what="Alice has 5 years Kubernetes experience, CKA certified", who="Alice", entities=["Alice", "Kubernetes", "CKA"]
2. what="Alice leads infrastructure team since March", who="Alice", entities=["Alice", "infrastructure"]

══════════════════════════════════════════════════════════════════════════
QUALITY OVER QUANTITY
══════════════════════════════════════════════════════════════════════════

Ask: "Would this be useful to recall in 6 months?" If no, skip it.

IMPORTANT: Sensory/emotional details and observations that provide meaningful context
about experiences ARE important to remember, even if they seem small (e.g., how food
tasted, how someone looked, how loud music was). Extract these if they characterize
an experience or person."""

# Assembled concise prompt
CONCISE_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_CONCISE_GUIDELINES,
    examples=_CONCISE_EXAMPLES,
)

# Custom prompt uses same base but without examples
CUSTOM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines="{custom_instructions}",
    examples="",  # No examples for custom mode
)

# Verbatim mode: preserve the original text exactly, but still extract metadata
_VERBATIM_GUIDELINES = """══════════════════════════════════════════════════════════════════════════
VERBATIM MODE — Extract metadata only
══════════════════════════════════════════════════════════════════════════

The original text will be stored as-is in code. Your ONLY job is to extract metadata.

RULES:
- Produce EXACTLY ONE entry per input chunk.
- DO NOT include a "what" field — it is not part of the output schema.
- Extract all entities (people, places, organizations, objects, concepts).
- Extract temporal information (occurred_start, occurred_end, fact_kind, when).
- Extract location (where) and people (who).
- fact_type: use "world" for user preferences, rules, corrections, constraints, traits, and other objective facts, even when stated during an assistant interaction. Use "assistant" only for actions or experiences the assistant/agent actually performed."""

VERBATIM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_VERBATIM_GUIDELINES,
    examples="",
)


# Verbose extraction prompt - detailed, comprehensive facts (legacy mode)
VERBOSE_FACT_EXTRACTION_PROMPT = """Extract facts from text into structured format with FIVE required dimensions - BE EXTREMELY DETAILED.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}══════════════════════════════════════════════════════════════════════════
FACT FORMAT - ALL FIVE DIMENSIONS REQUIRED - MAXIMUM VERBOSITY
══════════════════════════════════════════════════════════════════════════

For EACH fact, CAPTURE ALL DETAILS - NEVER SUMMARIZE OR OMIT:

1. **what**: WHAT happened - COMPLETE description with ALL specifics (objects, actions, quantities, details)
2. **when**: WHEN it happened - ALWAYS include temporal info with DAY OF WEEK (e.g., "Monday, June 10, 2024")
   - Always include the day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
   - Format: "day_name, month day, year" (e.g., "Saturday, June 9, 2024")
3. **where**: WHERE it happened or is about - SPECIFIC locations, places, areas, regions (if applicable)
4. **who**: WHO is involved - ALL people/entities with FULL relationships and background
5. **why**: WHY it matters - ALL emotions, preferences, motivations, significance, nuance
   - For assistant facts: MUST include what the user asked/requested that triggered this!

Plus: fact_type, fact_kind, entities, occurred_start/end (for structured dates), where (structured location)

VERBOSITY REQUIREMENT: Include EVERY detail mentioned. More detail is ALWAYS better than less.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION (CRITICAL)
══════════════════════════════════════════════════════════════════════════

When text uses BOTH a generic relation AND a name for the same person → LINK THEM!

Example input: "I went to my college roommate's wedding last June. Emily finally married Sarah after 5 years together."

CORRECT output:
- what: "Emily got married to Sarah at a rooftop garden ceremony"
- when: "Saturday, June 8, 2024, after dating for 5 years"
- where: "downtown San Francisco, at a rooftop garden venue"
- who: "Emily (user's college roommate), Sarah (Emily's partner of 5 years)"
- why: "User found it romantic and beautiful, dreams of similar outdoor ceremony"
- where (structured): "San Francisco"

WRONG output:
- what: "User's roommate got married" ← LOSES THE NAME!
- who: "the roommate" ← WRONG - use the actual name!
- where: (missing) ← WRONG - include the location!

══════════════════════════════════════════════════════════════════════════
FACT_KIND CLASSIFICATION (CRITICAL FOR TEMPORAL HANDLING)
══════════════════════════════════════════════════════════════════════════

⚠️ MUST set fact_kind correctly - this determines whether occurred_start/end are set!

fact_kind="event" - USE FOR:
- Actions that happened at a specific time: "went to", "attended", "visited", "bought", "made"
- Past events: "yesterday I...", "last week...", "in March 2020..."
- Future plans with dates: "will go to", "scheduled for"
- Examples: "I went to a pottery workshop" → event
           "Alice visited Paris in February" → event
           "I bought a new car yesterday" → event
           "The user graduated from MIT in March 2020" → event

fact_kind="conversation" - USE FOR:
- Ongoing states: "works as", "lives in", "is married to"
- Preferences: "loves", "prefers", "enjoys"
- Traits/abilities: "speaks fluent French", "knows Python"
- Examples: "I love Italian food" → conversation
           "Alice works at Google" → conversation
           "I prefer outdoor dining" → conversation

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING (CRITICAL - USE EVENT DATE AS REFERENCE)
══════════════════════════════════════════════════════════════════════════

⚠️ IMPORTANT: Use the "Event Date" provided in the input as your reference point!
All relative dates ("yesterday", "last week", "recently") must be resolved relative to the Event Date, NOT today's date.

For EVENTS (fact_kind="event") - MUST SET BOTH occurred_start AND occurred_end:
- Convert relative dates → absolute using Event Date as reference
- If Event Date is "Saturday, March 15, 2020", then "yesterday" = Friday, March 14, 2020
- Dates mentioned in text (e.g., "in March 2020") should use THAT year, not current year
- CRITICAL: If the content mentions an absolute date (e.g., "March 15, 2024", "2024-03-15"), you MUST extract it and set occurred_start in ISO format
- Always include the day name (Monday, Tuesday, etc.) in the 'when' field
- Set occurred_start AND occurred_end to WHEN IT HAPPENED (not when mentioned)
- For single-day/point events: set occurred_end = occurred_start (same timestamp)

For CONVERSATIONS (fact_kind="conversation"):
- General info, preferences, ongoing states → NO occurred dates
- Examples: "loves coffee", "works as engineer"

══════════════════════════════════════════════════════════════════════════
FACT TYPE
══════════════════════════════════════════════════════════════════════════

- **world**: User's life, preferences, rules, corrections, constraints, other people, and events (facts that would exist without this conversation)
- **assistant**: Actions or experiences the assistant/agent actually performed while helping the user (requests, recommendations, help)
  ⚠️ CRITICAL for assistant facts: ALWAYS capture the user's request/question in the fact!
  Include: what the user asked, what problem they wanted solved, what context they provided

══════════════════════════════════════════════════════════════════════════
ENTITIES - EXTRACT EVERYTHING
══════════════════════════════════════════════════════════════════════════

Extract ALL of the following from the fact:
- People names (Emily, Alice, Dr. Smith)
- Organizations (Google, MIT, local coffee shop)
- Places (San Francisco, Brooklyn, Paris)
- Significant objects mentioned (coffee maker, new car, wedding dress)
- Abstract concepts/themes (friendship, career growth, loss, celebration)

ALWAYS include "user" when fact is about the user.
Extract anything that could help link related facts together."""


# Causal relationships section - appended when causal extraction is enabled
CAUSAL_RELATIONSHIPS_SECTION = """

══════════════════════════════════════════════════════════════════════════
CAUSAL RELATIONSHIPS
══════════════════════════════════════════════════════════════════════════

Link facts with causal_relations (max 2 per fact). target_index must be < this fact's index.
Type: "caused_by" (this fact was caused by the target fact)

Example: "Lost job → couldn't pay rent → moved apartment"
- Fact 0: Lost job, causal_relations: null
- Fact 1: Couldn't pay rent, causal_relations: [{target_index: 0, relation_type: "caused_by"}]
- Fact 2: Moved apartment, causal_relations: [{target_index: 1, relation_type: "caused_by"}]"""


def _append_map_fields_prompt(fields: dict[str, "MapField"], lines: list[str], indent: int = 4) -> None:
    """Recursively append map field descriptions to the prompt lines."""
    pad = " " * indent
    for field_name, map_field in fields.items():
        field_desc = f": {map_field.description}" if map_field.description else ""
        if map_field.type == "map" and map_field.fields:
            lines.append(f"{pad}• {field_name} (object){field_desc}")
            _append_map_fields_prompt(map_field.fields, lines, indent + 4)
        elif map_field.type == "multi-values":
            vals = ", ".join(v.value for v in map_field.values if v.value)
            type_hint = f"multi-values: {vals}" if vals else "multi-values"
            lines.append(f"{pad}• {field_name} ({type_hint}){field_desc}")
        elif map_field.type == "value" and map_field.values:
            vals = ", ".join(v.value for v in map_field.values if v.value)
            lines.append(f"{pad}• {field_name} (one of: {vals}){field_desc}")
        else:
            lines.append(f"{pad}• {field_name} (text){field_desc}")


def _build_labels_prompt_section(labels_cfg: EntityLabelsConfig | list | None, free_form_entities: bool = True) -> str:
    """Build the entity labels classification section for the extraction prompt."""
    if labels_cfg is None:
        return ""

    # Accept raw list for backwards compatibility
    if isinstance(labels_cfg, list):
        if not labels_cfg:
            return ""
        labels_cfg = parse_entity_labels(labels_cfg)
        if labels_cfg is None:
            return ""

    if not labels_cfg.attributes:
        return ""

    if free_form_entities:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Continue extracting regular named entities in the 'entities' field."
    else:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Do NOT add regular named entities — labels-only mode."

    lines = [
        "\n\n══════════════════════════════════════════════════════════════════════════",
        "ENTITY LABELS - CLASSIFICATION ATTRIBUTES",
        "══════════════════════════════════════════════════════════════════════════",
        "",
        entities_instruction,
        "",
        "For each fact, fill the 'labels' object. Each field is a label group:",
        "",
    ]

    has_classification_attrs = False
    has_map_attrs = False

    for attr in labels_cfg.attributes:
        if attr.type == "map":
            has_map_attrs = True
            continue
        has_classification_attrs = True
        if attr.type == "text":
            # Free-text: no predefined values — LLM writes any relevant string or null
            lines.append(f"- {attr.key} (free text or null): {attr.description}")
        else:
            mode = "multi-value (list)" if attr.type == "multi-values" else "single value or null"
            lines.append(f"- {attr.key} ({mode}): {attr.description}")
            for v in attr.values:
                desc = f" — {v.description}" if v.description else ""
                lines.append(f'    • "{v.value}"{desc}')
        lines.append("")

    if has_classification_attrs:
        lines.append("Only assign labels when clearly applicable. Leave null/empty if the fact does not match.")
        lines.append("")

    # Add structured entity types (map groups)
    if has_map_attrs:
        lines.append("")
        lines.append("══════════════════════════════════════════════════════════════════════════")
        lines.append("STRUCTURED ENTITY TYPES")
        lines.append("══════════════════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append("For each fact, extract structured entities into the corresponding list field in 'labels'.")
        lines.append("Each structured entity type has defined fields. Return a list of objects, one per entity found.")
        lines.append("")
        for attr in labels_cfg.attributes:
            if attr.type != "map" or not attr.fields:
                continue
            desc = f": {attr.description}" if attr.description else ""
            lines.append(f"- {attr.key}{desc}")
            _append_map_fields_prompt(attr.fields, lines, indent=4)
            lines.append("")
        lines.append(
            "Only extract structured entities when clearly present in the text. Leave the list empty if none found."
        )

    return "\n".join(lines)


def _build_extraction_prompt_and_schema(config) -> tuple[str, type]:
    """
    Build extraction prompt and response schema based on config.

    When a taxonomy is configured, dynamically builds a Pydantic model with a
    typed `taxonomy_entities` field using an Enum built from valid taxonomy values.
    This enables JSON schema enforcement for structured outputs.

    Returns:
        Tuple of (prompt, response_schema)
    """
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # The per-bank retain mission is NOT baked into this system prompt: it would
    # make the prompt bank-specific and force a separate Gemini context cache per
    # mission (one per bank). Instead the prompt is bank-agnostic so a single
    # CachedContent serves every bank, and the mission rides in the per-request
    # user message via _retain_mission_preamble(). The {retain_mission_section}
    # placeholder is kept (templates still reference it) but always empty here.
    from ...prompt_utils import escape_for_prompt

    retain_mission_section = ""

    # Select base prompt based on extraction mode
    if extraction_mode == "custom":
        if not config.retain_custom_instructions:
            base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
            )
        else:
            base_prompt = CUSTOM_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
                custom_instructions=escape_for_prompt(config.retain_custom_instructions),
            )
    elif extraction_mode == "verbose":
        prompt = VERBOSE_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    elif extraction_mode == "verbatim":
        prompt = VERBATIM_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    else:
        base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
        prompt = base_prompt.format(
            retain_mission_section=retain_mission_section,
        )

    # Add causal relationships section if enabled
    # Verbatim mode never uses causal relations (no fact text to relate causally)
    if extraction_mode == "verbatim":
        base_fact_class = VerbatimExtractedFact
        base_response_class = VerbatimFactExtractionResponse
    elif extract_causal_links:
        prompt = prompt + CAUSAL_RELATIONSHIPS_SECTION
        base_fact_class = ExtractedFactVerbose if extraction_mode == "verbose" else ExtractedFact
        base_response_class = FactExtractionResponseVerbose if extraction_mode == "verbose" else FactExtractionResponse
    else:
        base_fact_class = ExtractedFactNoCausal
        base_response_class = FactExtractionResponseNoCausal

    # Add entity labels section if configured and build dynamic schema
    entity_labels_raw = config.entity_labels
    labels_cfg = parse_entity_labels(entity_labels_raw)
    free_form_entities = config.entities_allow_free_form
    labels_section = _build_labels_prompt_section(labels_cfg, free_form_entities)
    if labels_section:
        prompt = prompt + labels_section

    # Force the LLM to emit fact text in the configured language, regardless of
    # the source content's language. Same directive is applied to consolidation
    # and reflect so HINDSIGHT_API_LLM_OUTPUT_LANGUAGE has a uniform effect
    # across the pipeline. This is independent of the BM25 indexing language
    # (HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE) by design — search
    # tokenization and LLM output language are separate concerns.
    from ...prompt_utils import output_language_directive

    prompt = prompt + output_language_directive(config.llm_output_language)

    response_schema = base_response_class

    if labels_cfg and labels_cfg.attributes:
        LabelsModel = build_labels_model(labels_cfg)
        if LabelsModel is not None:
            dynamic_fields: dict = {
                "labels": (
                    LabelsModel,
                    Field(
                        description="Classification labels for this fact. Fill each applicable field; leave others null/empty."
                    ),
                )
            }
            if not free_form_entities:
                dynamic_fields["entities"] = (
                    list[str],
                    Field(default_factory=list, description="Leave empty — labels-only mode"),
                )
            # Inherit parent's required fields and add 'labels' so it appears in the JSON schema
            # required array (the base class json_schema_extra overrides required entirely)
            base_extra = base_fact_class.model_config.get("json_schema_extra")
            base_required = cast(dict, base_extra).get("required", []) if isinstance(base_extra, dict) else []
            DynamicFact = create_model(
                "LabelsFact",
                __base__=base_fact_class,
                __config__=ConfigDict(
                    json_schema_mode="validation",
                    json_schema_extra={"required": [*base_required, "labels"]},
                ),
                **dynamic_fields,
            )
            DynamicResponse = create_model("LabelsResponse", facts=(list[DynamicFact], ...))  # type: ignore[valid-type]
            response_schema = DynamicResponse

    return prompt, response_schema


def _retain_mission_preamble(config) -> str:
    """The bank's retain mission, formatted for the per-request user message.

    Kept OUT of the cached system prompt (which must stay bank-agnostic so one
    CachedContent serves every bank — otherwise each distinct mission spawns its
    own cache) and prepended to the user message instead. Returns "" when unset.
    No brace-escaping needed: unlike the system template, the user message is
    used verbatim, not passed through str.format().
    """
    retain_mission = config.retain_mission
    if not retain_mission:
        return ""
    return (
        "══════════════════════════════════════════════════════════════════════════\n"
        "FOCUS — What to retain for this bank (takes priority over the general guidelines)\n"
        "══════════════════════════════════════════════════════════════════════════\n\n"
        f"{retain_mission}\n\n"
    )


def _build_user_message(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    metadata: dict[str, str] | None = None,
    agent_name: str | None = None,
    mission_preamble: str = "",
) -> str:
    """Build user message for fact extraction.

    ``mission_preamble`` (the bank's retain mission, possibly empty) is prepended
    so the bank-specific focus lives in the variable user turn rather than the
    cached, bank-agnostic system prompt.
    """
    from ..orchestrator import parse_datetime_flexible

    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    if event_date is not None:
        event_date = parse_datetime_flexible(event_date)
        event_date_str = f"{event_date.strftime('%A, %B %d, %Y')} ({event_date.isoformat()})"
    else:
        event_date_str = "Unknown"

    metadata_section = ""
    if metadata:
        metadata_lines = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
        metadata_section = f"\nMetadata:\n{metadata_lines}"

    narrator_section = ""
    if agent_name:
        narrator_section = (
            f"\nNarrator: {agent_name} (the AI agent whose memory this is). By default, "
            f'first-person statements like "I did X" are {agent_name}\'s own actions → classify as '
            f'"assistant".'
        )
        # Only defer to the Context when one was actually provided — otherwise this
        # clause points at a "Context: none" line and just adds noise.
        if context:
            narrator_section += (
                " BUT the Context above takes precedence: if it identifies a different "
                "first-person speaker (e.g. a user or customer in a transcript), attribute those "
                'statements to that speaker and classify them as "world", not "assistant".'
            )

    return f"""{mission_preamble}Extract facts from the following text chunk.

Chunk: {chunk_index + 1}/{total_chunks}
Event Date: {event_date_str}
Context: {sanitized_context}{metadata_section}{narrator_section}

Text:
{sanitized_chunk}"""


def _build_request_body(batch_impl, config, prompt: str, user_message: str, response_schema: type) -> dict:
    """Build request body for the batch LLM API call.

    ``batch_impl`` is the provider implementation that will serve the batch. For
    a multi-LLM chain this is the first batch-capable member (see
    ``MultiLLMProvider.batch_provider_impl``), not necessarily the primary — so
    ``model``/``provider``/``service_tier`` must come from THIS impl, matching the
    account the batch is submitted to.
    """
    request_body = {
        "model": batch_impl.model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
    }

    # Honour the configured retain temperature. ``None`` omits the parameter
    # entirely (for models like Azure GPT-5.5 that reject explicit temperatures),
    # mirroring LLMProvider.call, which drops temperature when it is None. The
    # batch path builds the request body directly instead of going through
    # LLMProvider.call (#2469 only de-hardcoded the streaming path), so it must
    # apply the same rule here.
    if config.llm_temperature_retain is not None:
        request_body["temperature"] = config.llm_temperature_retain

    # Add max_completion_tokens if configured
    if config.retain_max_completion_tokens:
        request_body["max_completion_tokens"] = config.retain_max_completion_tokens

    # Add service_tier for OpenAI Flex Processing. ``provider`` is set by every
    # LLMInterface, and the short-circuit keeps impls without a service tier
    # (gemini/anthropic/fireworks) from ever reaching the second attribute — so a
    # renamed field fails loudly here instead of silently dropping flex pricing.
    if batch_impl.provider == "openai" and batch_impl.openai_service_tier:
        request_body["service_tier"] = batch_impl.openai_service_tier

    # Add response_format (JSON schema). The batch path builds the request body
    # directly instead of going through LLMProvider.call(), so resolve the
    # strict-schema flag here too: strict=True grammar-enforces the output on capable
    # backends rather than relying on the model to emit clean JSON. Reads the
    # retain-scoped field, which already folds in the global HINDSIGHT_API_LLM_STRICT_SCHEMA
    # fallback, so the batch and streaming paths can't disagree.
    if hasattr(response_schema, "model_json_schema"):
        retain_strict_schema = config.llm_strict_schema_retain
        schema = strict_json_schema(response_schema) if retain_strict_schema else response_schema.model_json_schema()
        request_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "facts", "schema": schema, "strict": retain_strict_schema},
        }

    return request_body
