# serve: a request leaves `axon_requests_in_flight` before its last byte  (PR #113, closes #112)

- Date: 2026-10-20. Branch: fix/inflight-gauge. Base: main 6f1efb5. Status: open (gates passed on 59bd41c).
- Pick / design: issue #112: `serve_metrics_slots_max_prompt` read `axon_requests_in_flight == 1` right after a
  `/v1/completions` stream (job 1020-030717-aa65, debug build, loaded bella). Not reproducible alone (10/10).

## What changed
- `src/serve/http.rs`: `Conn::on_end`, called once just before a response's last write (`respond`, and
  `end_stream`). `end_stream(last)` now takes the body's last part, so the last data (`[DONE]`, ollama's
  `done: true` line, the Responses API's `response.completed`) and the chunked terminator go out in one write,
  after `on_end`. The bytes on the wire are the same; one write fewer per stream.
- `src/serve.rs` `run_request`: the `InFlight` guard (now holding an `Arc<Metrics>`) moves into `on_end`.
  A response that never gets to its last write (a write error, a client gone, a stalled reader) drops it when
  `run_request` returns (`c.on_end = None`), and the connection loop clears it after `route` too: no path
  can leak a count.
- The slot gauges had the same order: the engine sent `Done`, then published the slot free (solo: after
  `job()` returned; batch: at the top of the next engine loop). Now `Engine::publish_free` (solo) and
  `Batch::publish` (batch, `finish` and `fail_all`) publish the slot free before the request's `Done`/`Error`.
  `Batch::publish` replaces the engine loop's inline copy of the same code.
- `axon_requests_total` already had the right order (`on_status`, before the first byte; #93,
  `serve_counts_a_request_before_answering_it`).
- Tests: `serve_in_flight_released_before_answering` (deterministic): `Server::hold` gained a hook at the
  end of `run_request`, between its last write and its return. With hold on, the old order reads 1 on every
  route; the test walks the five generation routes, whole and streamed. `serve_gauges_settle_back_to_back`
  (no hooks): 200 back-to-back requests (`AXON_TEST_GAUGE_ROUNDS` x 10), each followed at once by a /metrics
  read on an already-open keep-alive connection; in-flight, slots busy and queue depth must read 0 every time.
  `http::on_end_before_the_last_write` pins the hook's order at the `Conn` level.
- README: the /metrics paragraph says when a request is counted and uncounted.

## Why
The client sees a complete response as soon as the last chunk is on the wire; the handler thread only
dropped its guard after `stream()`/`whole()` returned, cleaned up and set `cancel`. A slow debug build on a
loaded host stretches that window past one connect + /metrics round trip. The test can't fix this by polling:
any client (a load balancer, an autoscaler, a test) reading /metrics right after a response would see it.

## How it was verified
The A/B base is the branch's first commit, b760917: main's server code plus the tests and the hold hook, no fix.
Every gate ran on 59bd41c. Later commits are docs only.

| gate | result | job ID / log |
|---|---|---|
| serve:: suite, release (bella sm_120) | 115 passed / 0 failed / 2 ignored | 1020-062924-bebd |
| serve:: suite, debug-assertion (bella) | 115 / 0 / 2 | 1020-062924-314c |
| fast tier, release (avante sm_89, GPU serial) | 617 / 0 / 47, 18 filtered | 1020-062928-661f |
| fast tier, debug-assertion (bella CPU queue) | 635 / 0 / 47 (first attempt OOM-killed by its 10 GB scope, requeued, passed) | 1020-062928-5898 |
| hold test, base vs fix, rel + dbg | base FAILS (rel and dbg), fix PASSES (rel and dbg) | 1020-062935-fb1c |
| stress test x50, base vs fix, rel + dbg | see Results | 1020-062935-fb1c |
| md5 of `gen`, main 6f1efb5 vs 59bd41c, same host | 20/20 equal (4 small models x 3 cuda + 2 cpu prompts, -n 64) | 1020-062935-df1f |
| clippy -D warnings (bin + tests), rustfmt | clean | local |

## Results
MEASURED, bella (Core Ultra 9 275HX, CPU test server, load average ~12.7 from co-tenant jobs), one process at a
time, base/fix interleaved:

| | hold test (deterministic) | stress x50 (200 requests + /metrics reads each) |
|---|---|---|
| base b760917, release | FAIL: `/v1/completions (stream false): answered, still in flight` (1 vs 0) | 30 of 50 runs failed |
| fix 59bd41c, release | pass | 50 of 50 pass |
| base b760917, debug | FAIL (same assert) | 37 of 50 runs failed |
| fix 59bd41c, debug | pass | 50 of 50 pass |

Base failures hit every route, whole and streamed. Each unsettled read was `axon_requests_in_flight 1`, plus
`axon_queue_depth 1`, which follows from it. About 115 (rel) and 124 (dbg) late reads out of 10,000. In the
failure lines the log kept, `axon_slots_busy` never read nonzero, so the slot reorder is supported by code
reading, not by a measured failure. No pp/tg change is expected or measured: the hot path gains one closure
call per response and loses one write per stream.

## What didn't work / not done
- The stress test's message says "N of 200 reads", but N counts unsettled gauge values (2 per late read).
  It's a cosmetic label in a test, left alone so as not to touch gated code.
- Ordering is `Relaxed` on the counter, as before: the socket write and read in between are the
  synchronisation (system calls), not the atomic.
- The draft-model and pipeline solo paths share `Engine::job`, so they're covered; nothing else sends `Done`.
