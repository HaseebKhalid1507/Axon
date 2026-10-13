#!/usr/bin/env python3
"""Thread-count sweep: every engine at every thread count, then each at its best.

For each thread count in --threads-list, measures pp560 and tg128 for every
engine (axon BINs, --ollama, --llama-server), interleaved across engines within
that thread count, --warmup discarded rounds then R timed rounds, medians.
Prints the full grid per metric and a "best config per engine" table (best
median tok/s and the thread count that got it), so engines are compared each
at its own best thread count rather than at one arbitrary T.

Usage:
  scripts/sweep.py [-m MODEL] [--threads-list 1,2,4,...] [-r ROUNDS] [--warmup N]
                   [--load N] [--ollama NAME] [--llama-server PATH]
                   [--metrics pp560,tg128] [--json PATH] [--dry-run] [BIN ...]

BIN accepts the same PATH or LABEL=PATH[,ENV=VAL...] forms as bench.py.
llama-server instances are started per thread count and stopped once that
thread count is done.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bench.add_common_args(ap)
    ap.add_argument("--threads-list", default="1,2,4,8,12,16,18,20,24")
    ap.add_argument("--metrics", default="pp560,tg128",
                    help="comma list (no @1 metrics; the sweep sets threads)")
    a = ap.parse_args()

    threads_list = [int(t) for t in a.threads_list.split(",") if t]
    metrics = [m for m in a.metrics.split(",") if m]
    for m in metrics:
        if m.endswith("@1"):
            ap.error(f"{m}: @1 metrics make no sense in a sweep")
        try:
            bench.metric_job(m, 1)
        except (KeyError, ValueError):
            ap.error(f"unknown metric {m}")
    engines = bench.make_engines(a)
    if not engines:
        ap.error("nothing to benchmark: give BINs, --ollama or --llama-server")

    # grid[engine][metric][threads] = samples
    grid = {e.name: {m: {} for m in metrics} for e in engines}
    bench.sigterm_as_interrupt()
    try:
        with bench.Hogs(0 if a.dry_run else a.load):
            for t in threads_list:
                jobs = [(m, m, t) for m in metrics]
                res = bench.run_rounds(engines, jobs, a.rounds, a.warmup, a.dry_run,
                                       tag=f"t={t} ")
                for name, by in res.items():
                    for m, v in by.items():
                        grid[name][m][t] = v
                for e in engines:
                    e.close()  # stop this thread count's llama-servers
    finally:
        for e in engines:
            e.close()

    print(f"threads={a.threads_list} rounds={a.rounds} warmup={a.warmup} "
          f"load={a.load} model={a.model}")
    best = {}
    for m in metrics:
        print(f"\n### {m} (tok/s, median (min–max))\n")
        print("| engine | " + " | ".join(f"t={t}" for t in threads_list) + " |")
        print("|---|" + "---|" * len(threads_list))
        for e in engines:
            row = grid[e.name][m]
            print(f"| {e.name} | " + " | ".join(bench.cell(row[t]) for t in threads_list) + " |")
            meds = {t: bench.statistics.median(v) for t, v in row.items() if v}
            if meds:
                bt = max(meds, key=meds.get)
                best.setdefault(e.name, {})[m] = (meds[bt], bt)

    print("\n### best config per engine\n")
    print("| engine | " + " | ".join(f"{m} (threads)" for m in metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for e in engines:
        b = best.get(e.name, {})
        cells = [f"{b[m][0]:.1f} (t={b[m][1]})" if m in b else "-" for m in metrics]
        print(f"| {e.name} | " + " | ".join(cells) + " |")

    if a.json:
        meta = bench.metadata(a, engines, metrics=metrics, threads_list=threads_list,
                              best={n: {m: {"tok_s": v, "threads": t} for m, (v, t) in b.items()}
                                    for n, b in best.items()})
        doc_results = {name: {f"{m}@t{t}": v for m, by_t in by.items() for t, v in by_t.items()}
                       for name, by in grid.items()}
        bench.write_json(a.json, meta, doc_results, a.dry_run)


if __name__ == "__main__":
    main()
