"""Per-phase speed report from Hindsight's own LLM request trace.

Reads the ``llm_requests`` table Hindsight populates during any run
(HINDSIGHT_API_LLM_TRACE_ENABLED defaults to true). Adds no instrumentation --
it only reads what was already recorded.

Phases are reported SEPARATELY, because they run on different servers with
different slot counts and are not comparable to each other:

    retain          } Hindsight's own work, model under test on :8081,
    consolidation   } whose router runs parallel=3 (HINDSIGHT_API_LLM_MAX_CONCURRENT=3)
    memory          answer generation -- the benchmark's agent stand-in, :8080
    judge           scoring -- also :8080

Columns, and why each exists:

  * ``solo s`` -- **the only figure comparable across runs**: raw / in-flight, i.e.
    what one call costs with the server to itself.
  * ``raw s`` -- the recorded ``duration_ms``. Hindsight starts that timer BEFORE
    acquiring the concurrency permit, so it absorbs both queue wait and the
    slowdown from sharing the GPU. Retain read 10.8s raw at 2.0x in-flight when
    the real cost was ~5.4s.
  * ``in-flight`` -- peers sharing the GPU, sampled at each call's own midpoint
    (NOT sum(durations)/span, which idle gaps dilute toward a meaningless <1x).
  * ``calls/min`` -- throughput over ACTIVE time only, gaps between runs removed.
  * ``idle %`` -- how much of the span was idle; high values mean the rows span
    multiple runs, so use --since to isolate one.

The table accumulates across runs, so use --since to isolate one.

    uv run python hindsight-dev/benchmarks/common/bench_speed_report.py pg0://bench-gemma
    ... bench_speed_report.py pg0://bench-gemma --since 30      # last 30 minutes
    ... bench_speed_report.py --port 5436 --since 30            # already-running instance
"""

import argparse
import asyncio
import statistics
import sys

import asyncpg
from rich.console import Console
from rich.table import Table

console = Console()

# Which server/slot budget each phase runs against, for the report header.
PHASE_HOME = {
    "retain": ":8081 (model under test)",
    "consolidation": ":8081 (model under test)",
    "consolidation_dedup": ":8081 (model under test)",
    "memory": ":8080 (answer)",
    "judge": ":8080 (judge)",
}


def _median_inflight(rs: list) -> float:
    """Median number of calls in flight, sampled at each call's own midpoint.

    Measured this way rather than as ``sum(durations)/wall``: that ratio is
    diluted by idle stretches, so a window containing gaps reports ~0.9x for work
    that was genuinely running 2-3 wide. Sampling at midpoints asks "how many
    peers did this call actually share the GPU with", which is what inflates its
    recorded duration.

    Sweep via sorted start/end arrays -- in-flight at t is
    (#started <= t) - (#ended <= t) -- so this stays O(n log n).
    """
    import bisect

    starts = sorted(r["started_at"] for r in rs)
    ends = sorted(r["ended_at"] for r in rs)
    counts = []
    for r in rs:
        mid = r["started_at"] + (r["ended_at"] - r["started_at"]) / 2
        counts.append(bisect.bisect_right(starts, mid) - bisect.bisect_right(ends, mid))
    return statistics.median(counts) if counts else 1.0


def _active_seconds(rs: list) -> float:
    """Seconds during which at least one call of this phase was in flight.

    Throughput must not be calls / (last_end - first_start): the table accumulates
    across runs, so that span swallows restarts and idle stretches between them and
    understates the rate badly (qwen's 123 answer calls spanned three separate runs
    and reported 2.06 calls/min for work that ran at roughly twice that). Merging the
    intervals counts only time the phase was actually working.
    """
    ivs = sorted((r["started_at"], r["ended_at"]) for r in rs)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:  # gap -- close the current block, it is idle time
            total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    total += (cur_e - cur_s).total_seconds()
    return total


async def _connect(db_url: str | None, port: int | None):
    if port:
        return await asyncpg.connect(f"postgresql://hindsight:hindsight@127.0.0.1:{port}/hindsight")
    from hindsight_api.pg0 import resolve_database_url

    return await asyncpg.connect(await resolve_database_url(db_url))


async def report(db_url: str | None, port: int | None, since_min: float | None) -> None:
    conn = await _connect(db_url, port)
    where = "status = 'success'"
    args: list = []
    if since_min:
        where += " AND started_at > now() - ($1 || ' minutes')::interval"
        args.append(str(since_min))
    try:
        rows = await conn.fetch(
            f"""
            SELECT COALESCE(operation, scope, '?') AS phase,
                   model,
                   duration_ms,
                   input_tokens,
                   output_tokens,
                   cached_tokens,
                   started_at,
                   ended_at
            FROM public.llm_requests
            WHERE {where}
            """,
            *args,
        )
    finally:
        await conn.close()

    if not rows:
        console.print("[yellow]no successful llm_requests rows in that window[/yellow]")
        return

    buckets: dict[tuple[str, str], list] = {}
    for r in rows:
        buckets.setdefault((r["phase"], r["model"] or "?"), []).append(r)

    table = Table(title=f"LLM phases{f' (last {since_min:g} min)' if since_min else ''}")
    cols = ("phase", "model", "runs on", "calls", "solo s", "raw s", "in-flight", "calls/min", "idle %", "tok/s", "in tok", "out tok")
    for col in cols:
        table.add_column(col)

    for (phase, model), rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        durs = [r["duration_ms"] for r in rs if r["duration_ms"] is not None]
        if not durs:
            continue
        active = _active_seconds(rs)  # excludes idle gaps between runs
        span = (max(r["ended_at"] for r in rs) - min(r["started_at"] for r in rs)).total_seconds()
        inflight = _median_inflight(rs)
        raw = statistics.median(durs) / 1000
        out_med = statistics.median([r["output_tokens"] or 0 for r in rs])
        # The always-comparable figure: what one call costs with the GPU to itself.
        solo = raw / inflight if inflight else raw
        table.add_row(
            phase,
            str(model),
            PHASE_HOME.get(phase, "-"),
            str(len(rs)),
            f"[bold]{solo:.1f}[/bold]",
            f"{raw:.1f}",
            f"{inflight:.1f}x",
            f"{len(rs) / (active / 60):.2f}" if active > 0 else "-",
            f"{100 * (1 - active / span):.0f}%" if span > 0 else "-",
            f"{out_med / solo:.0f}" if solo else "-",
            f"{statistics.median([r['input_tokens'] or 0 for r in rs]):.0f}",
            f"{out_med:.0f}",
        )

    console.print(table)
    console.print(
        "  [bold]solo s[/bold] = raw / in-flight -- what one call costs with the server to itself.\n"
        "  [dim]This is the only figure comparable across runs. `raw s` is the recorded duration_ms,\n"
        "  which at concurrency > 1 is inflated by however many calls shared the GPU: retain measured\n"
        "  10.8s raw at 2.0x in-flight is really ~5.4s of work. `tok/s` is per-stream decode at solo\n"
        "  speed. Do not compare `raw s` between runs that used different concurrency.[/dim]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db_url", nargs="?", help="pg0://instance-name (omit if using --port)")
    p.add_argument("--port", type=int, help="connect directly to a running pg0 instance's port")
    p.add_argument("--since", type=float, help="only calls from the last N minutes (isolates one run)")
    args = p.parse_args()
    if not args.db_url and not args.port:
        p.error("give a pg0:// url or --port")
    asyncio.run(report(args.db_url, args.port, args.since))


if __name__ == "__main__":
    sys.exit(main())
