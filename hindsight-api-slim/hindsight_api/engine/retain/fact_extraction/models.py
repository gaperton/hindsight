"""Pydantic models used by fact extraction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .normalization import _coerce_entity_strings


class Fact(BaseModel):
    """
    Final fact model for storage - built from lenient parsing of LLM response.

    This is what fact_extraction returns and what the rest of the pipeline expects.
    Combined fact text format: "what | when | where | who | why"
    """

    # Required fields
    fact: str = Field(description="Combined fact text: what | when | where | who | why")
    fact_type: Literal["world", "experience"] = Field(description="Perspective: world/experience")

    # Optional temporal fields
    occurred_start: str | None = None
    occurred_end: str | None = None

    # Optional location field
    where: str | None = Field(
        None, description="WHERE the fact occurred or is about (specific location, place, or area)"
    )

    # Optional structured data
    entities: list[str] | None = None
    causal_relations: list["CausalRelation"] | None = None


class CausalRelation(BaseModel):
    """Causal relationship from this fact to a previous fact (stored format)."""

    target_fact_index: int = Field(description="Index of the related fact in the facts array (0-based).")
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target: 'caused_by' = this fact was caused by the target"
    )


class FactCausalRelation(BaseModel):
    """
    Causal relationship from this fact to a PREVIOUS fact (embedded in each fact).

    Uses index-based references but ONLY allows referencing facts that appear
    BEFORE this fact in the list. This prevents hallucination of invalid indices.
    """

    target_index: int = Field(
        description="Index of the PREVIOUS fact this relates to (0-based). "
        "MUST be less than this fact's position in the list. "
        "Example: if this is fact #5, target_index can only be 0, 1, 2, 3, or 4."
    )
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target fact: 'caused_by' = this fact was caused by the target fact"
    )


class ExtractedFact(BaseModel):
    """A single extracted fact."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(description="Core fact - concise but complete (1-2 sentences)")
    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")
    why: str = Field(description="Context/significance if important. 'N/A' if obvious.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts, including user preferences, rules, corrections, and constraints even when stated during a conversation. 'assistant' = actions, experiences, or observations the assistant/agent actually performed."
    )
    entities: list[str] = Field(
        default_factory=list, description='People, places, concepts - plain strings, e.g. ["Alice", "Kubernetes"]'
    )
    causal_relations: list[FactCausalRelation] | None = Field(
        default=None, description="Links to previous facts (target_index < this fact's index)"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts (causal relations are embedded in each fact)."""

    facts: list[ExtractedFact] = Field(description="List of extracted factual statements")


class ExtractedFactVerbose(BaseModel):
    """A single extracted fact with verbose field descriptions for detailed extraction."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(
        description="WHAT happened - COMPLETE, DETAILED description with ALL specifics. "
        "NEVER summarize or omit details. Include: exact actions, objects, quantities, specifics. "
        "BE VERBOSE - capture every detail that was mentioned. "
        "Example: 'Emily got married to Sarah at a rooftop garden ceremony with 50 guests attending and a live jazz band playing' "
        "NOT: 'A wedding happened' or 'Emily got married'"
    )

    when: str = Field(
        description="WHEN it happened - ALWAYS include temporal information if mentioned. "
        "Include: specific dates, times, durations, relative time references. "
        "Examples: 'on June 15th, 2024 at 3pm', 'last weekend', 'for the past 3 years', 'every morning at 6am'. "
        "Write 'N/A' ONLY if absolutely no temporal context exists. Prefer converting to absolute dates when possible."
    )

    where: str = Field(
        description="WHERE it happened or is about - SPECIFIC locations, places, areas, regions if applicable. "
        "Include: cities, neighborhoods, venues, buildings, countries, specific addresses when mentioned. "
        "Examples: 'downtown San Francisco at a rooftop garden venue', 'at the user's home in Brooklyn', 'online via Zoom', 'Paris, France'. "
        "Write 'N/A' ONLY if absolutely no location context exists or if the fact is completely location-agnostic."
    )

    who: str = Field(
        description="WHO is involved - ALL people/entities with FULL context and relationships. "
        "Include: names, roles, relationships to user, background details. "
        "Resolve coreferences (if 'my roommate' is later named 'Emily', write 'Emily, the user's college roommate'). "
        "BE DETAILED about relationships and roles. "
        "Example: 'Emily (user's college roommate from Stanford, now works at Google), Sarah (Emily's partner of 5 years, software engineer)' "
        "NOT: 'my friend' or 'Emily and Sarah'"
    )

    why: str = Field(
        description="WHY it matters - ALL emotional, contextual, and motivational details. "
        "Include EVERYTHING: feelings, preferences, motivations, observations, context, background, significance. "
        "BE VERBOSE - capture all the nuance and meaning. "
        "FOR ASSISTANT FACTS: MUST include what the user asked/requested that led to this interaction! "
        "Example (world): 'The user felt thrilled and inspired, has always dreamed of an outdoor ceremony, mentioned wanting a similar garden venue, was particularly moved by the intimate atmosphere and personal vows' "
        "Example (assistant): 'User asked how to fix slow API performance with 1000+ concurrent users, expected 70-80% reduction in database load' "
        "NOT: 'User liked it' or 'To help user'"
    )

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence (set occurred dates), 'conversation' = general info (no occurred dates)",
    )

    occurred_start: str | None = Field(
        default=None,
        description="WHEN the event happened (ISO timestamp). Only for fact_kind='event'. Leave null for conversations.",
    )
    occurred_end: str | None = Field(
        default=None,
        description="WHEN the event ended (ISO timestamp). Only for events with duration. Leave null for conversations.",
    )

    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts about the user, other people, events, general knowledge, preferences, rules, corrections, or constraints. 'assistant' = actions, experiences, or observations the assistant/agent actually performed (e.g., 'I changed X', 'I discovered Y')."
    )

    entities: list[str] = Field(
        default_factory=list,
        description="Named entities, objects, AND abstract concepts from the fact, as plain strings (e.g. [\"Alice\", \"friendship\"]). Include: people names, organizations, places, significant objects (e.g., 'coffee maker', 'car'), AND abstract concepts/themes (e.g., 'friendship', 'career growth', 'loss', 'celebration'). Extract anything that could help link related facts together.",
    )

    causal_relations: list[FactCausalRelation] | None = Field(
        default=None,
        description="Causal links to PREVIOUS facts only. target_index MUST be less than this fact's position. "
        "Example: fact #3 can only reference facts 0, 1, or 2. Max 2 relations per fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class FactExtractionResponseVerbose(BaseModel):
    """Response for verbose fact extraction."""

    facts: list[ExtractedFactVerbose] = Field(description="List of extracted factual statements")


class ExtractedFactNoCausal(BaseModel):
    """A single extracted fact WITHOUT causal relations (for when causal extraction is disabled)."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    # Same fields as ExtractedFact but without causal_relations
    what: str = Field(description="WHAT happened - COMPLETE, DETAILED description with ALL specifics.")
    when: str = Field(description="WHEN it happened - include temporal information if mentioned.")
    where: str = Field(description="WHERE it happened - SPECIFIC locations if applicable.")
    who: str = Field(description="WHO is involved - ALL people/entities with relationships.")
    why: str = Field(description="WHY it matters - emotional, contextual, and motivational details.")

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence, 'conversation' = general info",
    )
    occurred_start: str | None = Field(default=None, description="WHEN the event happened (ISO timestamp).")
    occurred_end: str | None = Field(default=None, description="WHEN the event ended (ISO timestamp).")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = about the user/others, including user preferences, rules, corrections, and constraints. 'assistant' = actions or experiences the assistant/agent actually performed."
    )
    entities: list[str] = Field(
        default_factory=list,
        description='Named entities, objects, and concepts from the fact, as plain strings (e.g. ["Alice", "Kubernetes"]).',
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class FactExtractionResponseNoCausal(BaseModel):
    """Response for fact extraction without causal relations."""

    facts: list[ExtractedFactNoCausal] = Field(description="List of extracted factual statements")


class VerbatimExtractedFact(BaseModel):
    """
    Schema for verbatim extraction mode.

    Omits 'what' entirely — the original chunk text is used as fact_text in code.
    The LLM only extracts metadata: entities, temporal info, location, people.
    """

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["when", "where", "who", "fact_type"]},
    )

    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts. 'assistant' = first-person actions, experiences, or observations by the speaker."
    )
    entities: list[str] = Field(
        default_factory=list, description='People, places, concepts - plain strings, e.g. ["Alice", "Kubernetes"]'
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class VerbatimFactExtractionResponse(BaseModel):
    """Response for verbatim extraction mode (one entry per chunk, no fact text)."""

    facts: list[VerbatimExtractedFact] = Field(description="List of metadata entries (one per chunk)")
