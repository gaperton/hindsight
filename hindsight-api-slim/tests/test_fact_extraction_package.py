"""Compatibility coverage for the fact_extraction package split."""

from hindsight_api.engine import llm_interface
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.fact_extraction import batch, chunking, pipeline, prompts, streaming


def test_legacy_fact_extraction_imports_resolve_to_owning_modules():
    assert fact_extraction.chunk_text is chunking.chunk_text
    assert fact_extraction.iter_chunks is chunking.iter_chunks
    assert fact_extraction.extract_facts_from_text is streaming.extract_facts_from_text
    assert fact_extraction.extract_facts_from_contents_batch_api is batch.extract_facts_from_contents_batch_api
    assert fact_extraction.extract_facts_from_contents is pipeline.extract_facts_from_contents
    assert fact_extraction._build_extraction_prompt_and_schema is prompts._build_extraction_prompt_and_schema


def test_fact_extraction_keeps_canonical_output_too_long_error():
    assert fact_extraction.OutputTooLongError is llm_interface.OutputTooLongError
