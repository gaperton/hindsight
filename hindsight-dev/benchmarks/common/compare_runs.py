"""Compare two labelled benchmark runs and emit a Markdown report.

Intended workflow: keep one ``.env`` profile per configuration you care about,
run the benchmark once per profile with a distinct ``--label``, then diff a
variant against the baseline.

    uv run python hindsight-dev/benchmarks/locomo/locomo_fast.py --label baseline
    # edit a prompt / swap the .env profile
    uv run python hindsight-dev/benchmarks/locomo/locomo_fast.py --label variant

    uv run python hindsight-dev/benchmarks/common/compare_runs.py baseline variant
    uv run python hindsight-dev/benchmarks/common/compare_runs.py baseline variant -o report.md

**The report leads with whether a difference is real, not just its sign.** A
100-question run carries roughly +/-10 points of sampling noise at 95%, so a
5-point "improvement" is indistinguishable from luck. Each delta is therefore
reported with the standard error of the difference between two proportions, and
flagged as significant only when it clears ~2 SE. Ignoring that is the fastest
way to ship a prompt change that did nothing.

Per-question outcomes are also diffed, so you can see churn -- questions the
variant fixed and questions it broke -- which a headline accuracy identical to
baseline can completely hide.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Results live next to each benchmark; LoCoMo is the default home.
DEFAULT_RESULTS_DIRS = [
    Path(__file__).resolve().parents[1] / "locomo" / "results",
    Path(__file__).resolve().parents[1] / "longmemeval" / "results",
]


def _find_results(label: str, results_dir: Optional[Path]) -> Path:
    """Resolve a label (or an explicit path) to a results JSON file."""
    candidate = Path(label)
    if candidate.is_file():
        return candidate
    names = [f"benchmark_results_{label}.json", f"{label}.json", label]
    search = [results_dir] if results_dir else DEFAULT_RESULTS_DIRS
    for d in search:
        if not d:
            continue
        for name in names:
            p = d / name
            if p.is_file():
                return p
    searched = ", ".join(str(d) for d in search if d)
    raise SystemExit(f"error: no results for label {label!r} (looked in {searched})")


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _questions(run: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """Flatten every judged question, keyed by (item_id, question_index)."""
    out: Dict[Any, Dict[str, Any]] = {}
    for item in run.get("item_results", []):
        item_id = item.get("item_id")
        for q in item.get("metrics", {}).get("detailed_results", []) or []:
            out[(item_id, q.get("question_index"))] = q
    return out


def _category_stats(run: Dict[str, Any]) -> Dict[Any, Dict[str, int]]:
    """Merge per-item category_stats across items."""
    merged: Dict[Any, Dict[str, int]] = {}
    for item in run.get("item_results", []):
        for cat, st in (item.get("metrics", {}).get("category_stats") or {}).items():
            acc = merged.setdefault(cat, {"correct": 0, "total": 0, "invalid": 0})
            for k in ("correct", "total", "invalid"):
                acc[k] += st.get(k, 0)
    return merged


def _pct(correct: int, valid: int) -> Optional[float]:
    return (correct / valid * 100) if valid else None


def _delta_significance(c1: int, n1: int, c2: int, n2: int) -> Tuple[Optional[float], Optional[float], bool]:
    """Return (delta_pp, standard_error_pp, is_significant) for two proportions.

    SE of a difference of independent proportions; "significant" means the delta
    clears ~2 SE (about 95%). With n=100 per side this lands near +/-10pp, which
    is why small deltas on a short run should not be trusted.
    """
    if not n1 or not n2:
        return None, None, False
    p1, p2 = c1 / n1, c2 / n2
    delta = (p2 - p1) * 100
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) * 100
    return delta, se, abs(delta) > 2 * se


def _fmt_delta(delta: Optional[float], se: Optional[float], significant: bool) -> str:
    if delta is None:
        return "n/a"
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
    base = f"{arrow} {delta:+.1f} pp"
    if se is None:
        return base
    verdict = "**significant**" if significant else "within noise"
    return f"{base} (± {2 * se:.1f}) — {verdict}"


def _model_line(run: Dict[str, Any]) -> str:
    cfg = run.get("model_config") or {}
    parts = []
    for role in ("hindsight", "answer_generation", "judge"):
        r = cfg.get(role) or {}
        if r:
            parts.append(f"{role}=`{r.get('provider','?')}/{r.get('model','?')}`")
    return " · ".join(parts) if parts else "_(no model_config recorded)_"


def build_report(base_label: str, var_label: str, base: Dict, var: Dict, base_p: Path, var_p: Path) -> str:
    L: List[str] = []
    A = L.append

    A(f"# Benchmark comparison: `{var_label}` vs `{base_label}`\n")

    bc, bv = base.get("total_correct", 0), base.get("total_valid", 0)
    vc, vv = var.get("total_correct", 0), var.get("total_valid", 0)
    delta, se, sig = _delta_significance(bc, bv, vc, vv)

    A("## Headline\n")
    A(f"| | `{base_label}` | `{var_label}` |")
    A("|---|---|---|")
    A(f"| accuracy | **{_pct(bc, bv) or 0:.1f}%** ({bc}/{bv}) | **{_pct(vc, vv) or 0:.1f}%** ({vc}/{vv}) |")
    A(f"| invalid | {base.get('total_invalid', 0)} | {var.get('total_invalid', 0)} |")
    A(f"| questions | {base.get('total_questions', 0)} | {var.get('total_questions', 0)} |")
    A("")
    A(f"**Δ accuracy: {_fmt_delta(delta, se, sig)}**\n")
    if delta is not None and not sig:
        A(
            f"> The difference does not clear the ~±{2 * (se or 0):.1f} pp noise band for these sample "
            f"sizes. Treat `{var_label}` and `{base_label}` as indistinguishable on this evidence; "
            "raise the question count to resolve a real but smaller effect.\n"
        )

    A("## Configuration\n")
    A(f"- `{base_label}`: {_model_line(base)}")
    A(f"- `{var_label}`: {_model_line(var)}")
    A(f"- sources: `{base_p}` · `{var_p}`\n")

    # ---- per category -----------------------------------------------------
    bcat, vcat = _category_stats(base), _category_stats(var)
    cats = sorted(set(bcat) | set(vcat), key=str)
    if cats:
        A("## By category\n")
        A(f"| category | `{base_label}` | `{var_label}` | Δ |")
        A("|---|---|---|---|")
        for cat in cats:
            b = bcat.get(cat, {"correct": 0, "total": 0, "invalid": 0})
            v = vcat.get(cat, {"correct": 0, "total": 0, "invalid": 0})
            bvalid = b["total"] - b["invalid"]
            vvalid = v["total"] - v["invalid"]
            d, s, g = _delta_significance(b["correct"], bvalid, v["correct"], vvalid)
            bp, vp = _pct(b["correct"], bvalid), _pct(v["correct"], vvalid)
            bcell = f"{bp:.0f}% ({b['correct']}/{bvalid})" if bp is not None else "n/a"
            vcell = f"{vp:.0f}% ({v['correct']}/{vvalid})" if vp is not None else "n/a"
            A(f"| {cat} | {bcell} | {vcell} | {_fmt_delta(d, s, g)} |")
        A("")
        A(
            "> Per-category counts are small, so these deltas are noisier than the headline. "
            "Use them to form a hypothesis, not to conclude one.\n"
        )

    # ---- churn ------------------------------------------------------------
    bq, vq = _questions(base), _questions(var)
    shared = set(bq) & set(vq)
    fixed = [k for k in shared if not bq[k].get("is_correct") and vq[k].get("is_correct")]
    broke = [k for k in shared if bq[k].get("is_correct") and not vq[k].get("is_correct")]

    A("## Churn\n")
    if not shared:
        A("_No per-question detail in one of the runs; churn unavailable._\n")
    else:
        A(f"- compared on **{len(shared)}** questions present in both runs")
        A(f"- **fixed** by `{var_label}`: {len(fixed)}")
        A(f"- **broken** by `{var_label}`: {len(broke)}")
        A("")
        if len(fixed) or len(broke):
            A(
                f"> Net {len(fixed) - len(broke):+d}. Churn in both directions means the change moved "
                "behaviour even where the headline barely moved — worth reading the examples before "
                "concluding it is neutral.\n"
            )
        for title, keys in (("Fixed", fixed), ("Broken", broke)):
            if not keys:
                continue
            A(f"### {title} ({len(keys)})\n")
            for k in sorted(keys, key=str)[:10]:
                q = vq[k]
                A(f"- **Q:** {str(q.get('question', ''))[:160]}")
                A(f"  - expected: `{str(q.get('correct_answer', ''))[:100]}`")
                A(f"  - `{base_label}`: {str(bq[k].get('predicted_answer', ''))[:100]}")
                A(f"  - `{var_label}`: {str(q.get('predicted_answer', ''))[:100]}")
            if len(keys) > 10:
                A(f"\n_({len(keys) - 10} more not shown)_")
            A("")

    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", help="baseline label (or path to a results JSON)")
    p.add_argument("variant", help="variant label (or path to a results JSON)")
    p.add_argument("-o", "--output", type=Path, help="write Markdown here instead of stdout")
    p.add_argument("--results-dir", type=Path, help="directory holding the results files")
    args = p.parse_args()

    bp = _find_results(args.baseline, args.results_dir)
    vp = _find_results(args.variant, args.results_dir)
    report = build_report(args.baseline, args.variant, _load(bp), _load(vp), bp, vp)

    if args.output:
        args.output.write_text(report)
        print(f"wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
