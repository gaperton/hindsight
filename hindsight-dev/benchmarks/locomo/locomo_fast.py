"""LoCoMo, preset for fast prompt iteration.

Same benchmark, same dataset, same scoring as ``locomo_benchmark.py`` -- this only
picks defaults tuned for a short feedback loop, so you can A/B a prompt change in
minutes instead of an hour. All of it reuses ``run_benchmark`` from
``locomo_benchmark``; nothing is reimplemented here.

The defaults, and why each one:

* ``--conversation conv-26`` -- the cheapest useful unit. Of the 10 conversations
  it has the joint-smallest haystack (19 sessions) with near-top question yield
  (199 available), so ingestion cost per question is the lowest on offer.

* ``--context-format structured`` -- the answer prompt gets facts rendered as text
  instead of ``json.dumps(recall_result)``. Measured on conv-26: ~101k -> ~44k
  tokens over three queries, a **2.3x** cut, and larger when source chunks are in
  play. Answer generation is ~90% of the QA phase's wall clock and its prompts do
  not prefix-cache (every question retrieves different context, so the cache hit
  rate is ~4.5% against retain's ~76%), which makes this the single biggest speed
  lever in the run.

* ``--max-concurrent-questions 2`` -- applies to BOTH the answer phase and the
  judge phase, and must track the answer/judge server's slot count (``parallel=``
  in models-0.ini, currently 2 for qwen3.6-35b). Requests beyond the slots only
  queue, and Hindsight starts its per-call timer *before* acquiring the LLM
  permit, so queue wait is billed to the call: oversubscribing once made
  identical work report 30-45s per call instead of 11-23s. Note the judge phase
  previously ran a hardcoded 8 concurrent, which was 4x the slots.

* ``--max-questions 100`` -- ~+/-10 points at 95% confidence. Enough to see a real
  prompt change; raise toward 199 (~+/-7) to confirm a marginal one.

Everything is overridable, so this stays a preset rather than a fork.

    # baseline
    uv run python hindsight-dev/benchmarks/locomo/locomo_fast.py --label baseline
    # after editing a prompt (retain edits need --no-skip-ingestion)
    uv run python hindsight-dev/benchmarks/locomo/locomo_fast.py --label variant

Results are written per ``--label`` so an A/B pair coexists instead of the variant
overwriting the baseline.

Two things this cannot paper over:

* ``--skip-ingestion`` (on by default here, since re-ingesting is the slow part) is
  only valid while the **retain** prompt is unchanged. Editing retain and reusing
  banks silently scores the *old* prompt. Pass ``--no-skip-ingestion`` after any
  retain-side edit.
* the judge is read from ``HINDSIGHT_API_JUDGE_LLM_*`` (note: a different namespace
  from the ``HINDSIGHT_TEST_JUDGE_*`` vars pytest uses). Leave it unset and it falls
  back to the model under test, which then grades its own answers. This script
  refuses to run in that state unless you pass ``--allow-self-judge``.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# NOTE: locomo_benchmark is imported lazily inside main(), *after* --profile has been
# loaded. benchmark_runner calls get_config() at module level, which builds and caches
# the config singleton from the environment as it stands at import time -- so importing
# first would freeze in the pre-profile environment and silently ignore the profile
# (observed: embeddings fell back to bge-small instead of the configured bge-m3, whose
# different dimension would not even match an existing bank).


def _check_judge_independence(allow_self_judge: bool) -> None:
    """Refuse to produce a self-judged number unless explicitly allowed.

    LLMAnswerEvaluator reads HINDSIGHT_API_JUDGE_LLM_* and falls back to
    HINDSIGHT_API_LLM_*, so an unset judge silently means "the model grades
    itself" -- which inflates the score and is easy to miss in the output.
    """
    judge_model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL")
    judge_url = os.getenv("HINDSIGHT_API_JUDGE_LLM_BASE_URL")
    sut_model = os.getenv("HINDSIGHT_API_LLM_MODEL")
    sut_url = os.getenv("HINDSIGHT_API_LLM_BASE_URL")

    independent = bool(judge_model) and (judge_model != sut_model or judge_url != sut_url)
    if independent:
        print(f"judge: {judge_model} @ {judge_url}  (independent of {sut_model} @ {sut_url})")
        return

    msg = (
        "judge is NOT independent: HINDSIGHT_API_JUDGE_LLM_* is unset or matches the model "
        "under test, so it would grade its own answers and inflate the score.\n"
        "  Set HINDSIGHT_API_JUDGE_LLM_{PROVIDER,MODEL,BASE_URL,API_KEY} to a different model, "
        "or pass --allow-self-judge if you accept a biased number."
    )
    if not allow_self_judge:
        raise SystemExit(f"error: {msg}")
    print(f"WARNING: {msg}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--label", default="fast", help="names the results file: benchmark_results_<label>.json")
    p.add_argument(
        "--profile",
        type=Path,
        help=(
            "env file to load (override=True) before configuring anything, e.g. .env.bench-gemma. "
            "Use one file per configuration you compare. Relying on an ad-hoc .env is how a run "
            "silently loses a setting the real profile has -- a missing LLM_STRICT_SCHEMA cost 10 "
            "of 100 answers to schema-validation failures in one run, which read as a model deficit."
        ),
    )
    p.add_argument("--conversation", nargs="+", default=["conv-26"], help="conversation ids (default: conv-26)")
    p.add_argument("--max-questions", type=int, default=100, help="questions per conversation (default: 100)")
    p.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=2,
        help=(
            "concurrent questions, AND the judge-phase concurrency. Keep it equal to the answer/judge "
            "server's slot count (parallel= in models-0.ini): above that it only builds queue, which "
            "Hindsight bills into each call's duration_ms. Default: 2."
        ),
    )
    p.add_argument(
        "--context-format",
        choices=["json", "structured"],
        default="structured",
        help="'structured' cuts answer prefill ~2.3x (default: structured)",
    )
    p.add_argument(
        "--no-skip-ingestion",
        dest="skip_ingestion",
        action="store_false",
        help="re-ingest. REQUIRED after any retain-prompt change, or you score the old prompt.",
    )
    p.add_argument("--allow-self-judge", action="store_true", help="proceed even if the judge is the model under test")
    p.add_argument(
        "--no-wait-consolidation",
        dest="wait_consolidation",
        action="store_false",
        help=(
            "skip waiting for consolidation. Off by default because consolidation is queued "
            "asynchronously after ingestion: without the wait the process reaches QA (and exits) "
            "before any observation is built, so the consolidation prompt is never exercised and "
            "a regression in it scores identically -- worse than no signal, since it looks like "
            "evidence of safety."
        ),
    )
    p.set_defaults(skip_ingestion=True, wait_consolidation=True)
    args = p.parse_args()

    if args.profile:
        from dotenv import load_dotenv

        if not args.profile.is_file():
            raise SystemExit(f"error: profile not found: {args.profile}")
        load_dotenv(args.profile, override=True)
        print(f"profile: {args.profile}")

    # Settings whose absence quietly changes what is being measured rather than
    # failing loudly. STRICT_SCHEMA is the sharp one: without it a weaker model
    # emits the JSON *schema* instead of an instance, those answers are scored
    # wrong, and the run looks like a reasoning deficit.
    for key, why in (
        ("HINDSIGHT_API_LLM_STRICT_SCHEMA", "answers may fail schema validation and be scored wrong"),
        ("HINDSIGHT_API_RERANKER_MAX_CANDIDATES", "recall breadth/latency differs from your deployed config"),
        ("HINDSIGHT_API_LLM_MAX_CONCURRENT", "concurrency may oversubscribe the router's slots"),
    ):
        if not os.getenv(key):
            print(f"WARNING: {key} is unset -- {why}")

    # Print all three roles. Only the first is under test; the other two must be
    # held constant across runs or the comparison varies more than one thing.
    sut = f"{os.getenv('HINDSIGHT_API_LLM_MODEL')} @ {os.getenv('HINDSIGHT_API_LLM_BASE_URL')}"
    ans_model = os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL")
    print(f"under test (retain+consolidate): {sut}")
    if ans_model:
        print(f"answer generation (fixed)      : {ans_model} @ {os.getenv('HINDSIGHT_API_ANSWER_LLM_BASE_URL')}")
        if ans_model == os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL"):
            # Not an error: a deliberate trade (see the profile comment). Surfaced
            # so the absolute score is never mistaken for a calibrated one.
            print(
                "  note: answer model == judge model -- absolute scores are inflated by self-grading; "
                "the bias is constant across runs, so read DIFFERENCES between runs, not the level"
            )
    else:
        print(
            "WARNING: HINDSIGHT_API_ANSWER_LLM_* unset -- answer generation falls back to the model "
            "under test, so each run varies memory quality AND answering ability together"
        )

    _check_judge_independence(args.allow_self_judge)

    # Imported here, not at module scope -- see the note at the top of this file.
    from benchmarks.locomo.locomo_benchmark import run_benchmark

    if args.skip_ingestion:
        print("ingestion: SKIPPED (reusing banks) -- pass --no-skip-ingestion after a retain-prompt edit")
    else:
        print("ingestion: full re-ingest")
    print(f"consolidation: {'WAIT (observations built before QA)' if args.wait_consolidation else 'not awaited'}")

    results = asyncio.run(
        run_benchmark(
            conversation=args.conversation,
            max_questions_per_conv=args.max_questions,
            max_concurrent_questions_override=args.max_concurrent_questions,
            context_format=args.context_format,
            eval_concurrency=args.max_concurrent_questions,
            skip_ingestion=args.skip_ingestion,
            wait_consolidation=args.wait_consolidation,
            results_filename=f"benchmark_results_{args.label}.json",
        )
    )
    return results


if __name__ == "__main__":
    main()
