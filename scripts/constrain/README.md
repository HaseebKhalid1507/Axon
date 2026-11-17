# Constrained tool calls: the gate's script and inputs (#274)

`cbcheck.sh` is #274's bit and validity gate for constrained tool calls (PR B of #259), one GPU job. Its header says
what each verdict checks; it prints `CBCHECK PASS` when they all pass. It drives `../constrain_check.py`, which has
the modes `run`, `replay`, `cost` and `diff`. The inputs are kept here so a gate can be rerun as it ran:

- `replays/`: `constrain_check.py replay` cells, one JSON file each: `kind`, `what`, `model`, `route`, `body`,
  `orig`.
  - `runaway`: the 13 acceptance-run calls that tripped the first stall windows. Each is the request that led to
    the call, and `orig` names that call.
  - `legit`: data that repeats one value, with the size asked for (`want`: a unit and a count). A stall while the
    call holds no more than that cut the data asked for: a failure. A stall after it held more is a real runaway
    the rule caught: the row passes and prints `overran U/N`. On sm_89 (a 4090), `14-c` greedy and seed 2 write
    683 zeros on one line for the 256 asked and stall; with `--no-constrain` the same text runs on to
    `max_tokens`, so the mask didn't cause it (11-17, jobs `7f0e`, `6f9d`, `899c`). On sm_120 (bella) it writes the
    table.
  - `known-fp` (reported, not judged): none now. `16-csv` was one under the few-ids rule (1,024 tokens of at most
    8 ids: it tripped after about 1,050 tokens), which Fable's review dropped (`0932813`). Under the one rule left
    (2,048 tokens of one block) it stalls at 120 rows of the 80 asked, on both cards, so it is `legit` with a
    `want`.
  - A runaway cell that no seed loops on prints `NO-LOOP`: it tests nothing at that head. At `3c205cc` that was
    `02-codex-apply-patch`, `03-pi-edit`, `10-pi-edit` and the 0.5B's `00-pi-edit`.
- `bodies/`: each harness's captured first-turn body (codex 0.144.5, opencode 2.2.6, pi 0.80.3) for
  `constrain_check.py run`'s cells.
- `serve_md5_share.py`: the md5 of tool-free serve traffic, base against candidate (from #216).
- `cbbar.py`: the harness bar. It reads `harness_accept.py`'s RESULT lines and gives malformed replies, stalls,
  fallbacks and tasks passed per harness and engine, and splits stalls per task run (first, repeats). Pass
  `--stalls-malformed` for runs at `1b88537` or later: a stalled call arrives malformed and its reply finishes
  `length`, so stalls aren't counted twice and the length note leaves them out.
