#!/usr/bin/env python3
"""passhist.py TRACE ARM [C]: the pre-registered P1 lines of docs/design/w4a8-prefill.md section 4 (post-P1 re-run;
scripts/w4a8_coverage_lp.py `p1` mode) from one AXON_SERVE_TRACE file. Only the steps of the measured cell count:
from the {"mark":"cell",...,"c":C} line (default C 128) to the next mark (warm-ups excluded).
  PASSHIST <arm> <fed>x<count> ...    steps with one feed pass: tokens fed, how many such steps
  PASSMULTI <arm> <fed>/<feeds> ...   steps with >= 2 feed passes (the H100's two-feed-pass rate; M10)
  FEEDSHARE <arm> <feed_ms> <wall_ms> the cell's feed time and step wall time
  PASSSTAT <arm> ...                  counts, for the log"""
import collections, json, sys

path, arm = sys.argv[1], sys.argv[2]
want_c = int(sys.argv[3]) if len(sys.argv) > 3 else 128
hist, multi = collections.Counter(), []
feed_ms = wall_ms = 0.0
on = False
steps = 0
marks = 0


def fatal(why):
    # Loud, on stdout (the job log) and in the exit code: a cell that sent nothing must never read as data.
    print(f"PASSHIST FATAL {arm}: {why} ({path})")
    sys.exit(2)


try:
    lines = open(path).readlines()
except OSError as e:
    fatal(f"no trace file: {e}")
for line in lines:
    try:
        x = json.loads(line)
    except ValueError:
        continue
    if "mark" in x:
        on = x.get("mark") == "cell" and int(x.get("c", -1)) == want_c
        marks += on
        continue
    if not on or "feeds" not in x:
        continue
    steps += 1
    wall_ms += float(x.get("ms", 0.0))
    feed_ms += float(x.get("feed_ms", 0.0))
    k, fed = int(x["feeds"]), int(x.get("fed", 0))
    if k == 1:
        hist[fed] += 1
    elif k >= 2:
        multi.append(f"{fed}/{k}")
if marks == 0:
    fatal(f"no C={want_c} cell mark in the trace")
n_feed = sum(hist.values()) + len(multi)
if steps == 0:
    fatal(f"the C={want_c} cell has no steps: it sent nothing")
if n_feed == 0 or feed_ms <= 0.0:
    fatal(f"the C={want_c} cell has {steps} steps but no feed passes: no prompt was fed")
print(f"PASSHIST {arm} " + " ".join(f"{f}x{c}" for f, c in sorted(hist.items())))
print(f"PASSMULTI {arm} " + " ".join(multi))
print(f"FEEDSHARE {arm} {feed_ms:.3f} {wall_ms:.3f}")
n1 = sum(hist.values())
print(f"PASSSTAT {arm} C={want_c}: {steps} steps, {n1} with one feed pass, {len(multi)} with two or more "
      f"({100 * len(multi) / max(1, n1 + len(multi)):.2f} % of feed steps), feeds {feed_ms / 1000:.2f} s of {wall_ms / 1000:.2f} s")
