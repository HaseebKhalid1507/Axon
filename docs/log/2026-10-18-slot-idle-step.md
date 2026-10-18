# Slot-idle test: the freeing bound counts the engine steps it waits for  (PR #N)

- Date: 2026-10-18. Branch: fix/slot-idle-step. Base: main c8e34fd. Status: open
- Pick / design: follow-up to #80 / #85. Under 8 pool threads plus 64 spinning threads `serve_slot_idle_frees_a_stalled_stream`
  failed 10/10 on #80's code (1018-044914-c136): freed 4.00-4.20 s after the window filled (8 runs) and 7.8-7.9 s (2 runs),
  against a 2.8 s bound. A number that constant pointed at a mechanism.

## What changed
`src/serve/tests.rs` only. The test measures the longest engine step it sees (the time between token counts, from the
first token on) and bounds the freeing time by idle + 2.5 s + 2 x that step instead of idle + 2.5 s. It also prints
when the eviction happened, next to when the slot was free. Server code is unchanged.

## Why: the mechanism
Read in `src/serve.rs` / `src/serve/engine.rs`:
1. **Detection runs on the connection thread, not the engine.** `Events::next` polls every min(200 ms, idle/4) =
   75 ms. `ops::Stall::check` calls a stream stalled when its acknowledged byte count hasn't moved for `idle` while
   bytes wait unacknowledged (SIOCOUTQ > 0). A send blocked for `idle` (the write timeout) also counts.
2. **But at the moment the window fills, no bytes are waiting.** Each token goes out as one small write, so when the
   client's receive window closes, everything written so far has been delivered and acknowledged (outq = 0, which
   resets the check). Bytes wait only once the engine produces the next token and the connection thread writes it.
   Then `idle` later the stream is evicted. Eviction = fill + up to one engine step + idle + a poll.
3. **The slot is freed between steps.** The cancel flag lands while a step is running, and `Batch::reap` frees the
   slot at the start of the next step. `/slots` is published at the top of the engine loop. Freed = eviction + up to
   one more step.

So freeing takes idle plus up to two engine steps. On an idle host a step of the 0.5B is ~0.02-0.1 s, which the
2.5 s slack covers. With 8 pool threads among 64 spinning threads on pod3 (6.8 vCPU), a step takes 2-6 s
(measured below), most likely because each matvec waits for its slowest, preempted pool worker. With 1 pool
thread there's no such wait and a step stays well under a second, which is why that stress passed.

Confirmed, one diagnostic job (1018-052002-351a, pod3, 8 pool threads + 64 spinning, x3), times after the fill:

| run | evicted | freed | longest step seen |
|---|---|---|---|
| 1 | 4.28 s | 7.79 s | 4.20 s |
| 2 | 2.39 s | 4.00 s | 6.09 s |
| 3 | 6.10 s | 11.70 s | 6.00 s |

Eviction comes up to one step after the fill, and freeing up to one step after eviction. The earlier probe
(1018-050906-543f) freed at 4.11 / 4.10 / 4.19 s with longest steps of 2.2 / 4.0 / 13.6 s.

Not a serve weakness worth a server change:
- Stall detection doesn't wait on engine progress, except that it needs bytes to be waiting. A client that stopped
  reading while the server has nothing to send isn't holding anything up yet.
- The slot can't be used before the step in flight ends anyway, because the engine is the slot's only user and it's
  busy until then.

The cost of a stalled client is at most one extra generated token. Detecting the zero window itself (TCP_INFO's
`tcpi_snd_wnd`) would start the clock one step earlier, but it frees nothing sooner in wall time that anyone can
use, so it wasn't done.

## How it was verified
| gate | result | job |
|---|---|---|
| stress x10, 8 pool threads + 64 spinning (pod3), #80's code (before) | 10/10 failed (freed 4.0-7.9 s > 2.8 s) | 1018-044914-c136 |
| same stress x10, this branch | pending | 1018-052924-0f34 |
| stress x10, 1 pool thread + 64 spinning, this branch (jade) | pending | 1018-052926-f051 |
| same 1-thread stress on pod3: main (before #80) / #80 | 10/10 failed "never filled" / 0/10 failed | 1018-031600-534b / 1018-032125-50cf |
| the test unstressed, release and debug builds (jade) | pending | |
| clippy -D warnings (bin + tests), rustfmt | pending | local |

## What didn't work / wasn't done
- A server-side change (zero-window detection, reaping mid-step): no slot is freed any earlier in usable time.
- The test's step measurement comes from its own 10 ms polling, so under heavy stress it can overstate a step (it once
  saw 13.6 s). That only loosens the bound under stress. On an idle host it adds ~0.2 s to the old bound.
