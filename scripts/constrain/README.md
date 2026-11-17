# Constrained tool calls: the gate's script and inputs (#274)

`cbcheck.sh` is #274's bit and validity gate for constrained tool calls (PR B of #259), one GPU job. Its header says
what each verdict checks; it prints `CBCHECK PASS` when they all pass. It drives `../constrain_check.py`, which has
the modes `run`, `replay`, `cost` and `diff`. The inputs are kept here so a gate can be rerun as it ran:

- `replays/`: `constrain_check.py replay` cells, one JSON file each: `kind`, `what`, `model`, `route`, `body`,
  `orig`.
  - `runaway`: the 13 acceptance-run calls that tripped the first stall windows. Each is the request that led to
    the call, and `orig` names that call.
  - `legit`: data that repeats one value.
  - `known-fp`: the CSV the rule is known to cut.
  - A runaway cell that no seed loops on prints `NO-LOOP`: it tests nothing at that head. At `3c205cc` that was
    `02-codex-apply-patch`, `03-pi-edit`, `10-pi-edit` and the 0.5B's `00-pi-edit`.
- `bodies/`: each harness's captured first-turn body (codex 0.144.5, opencode 2.2.6, pi 0.80.3) for
  `constrain_check.py run`'s cells.
- `serve_md5_share.py`: the md5 of tool-free serve traffic, base against candidate (from #216).
- `cbbar.py`: the harness bar. It reads `harness_accept.py`'s RESULT lines and gives malformed replies, stalls,
  fallbacks and tasks passed per harness and engine. Pass `--stalls-malformed` for runs at `3c205cc` or later,
  where a stalled call already arrives malformed.
