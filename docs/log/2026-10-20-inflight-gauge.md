# serve: a request leaves `axon_requests_in_flight` before its last byte  (PR #113, closes #112)

- Date: 2026-10-20. Branch: fix/inflight-gauge. Base: main 6f1efb5. Status: open.
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
Base for the A/B is the branch's first commit (main + the tests and the hold hook, no fix).

| gate | result | job ID / log |
|---|---|---|
| serve:: suite, release and debug-assertion | pending | PR body |
| fast tier, release (+ debug on the CPU queue) | pending | PR body |
| hold test, base vs fix | pending | PR body |
| stress test x50, base vs fix, release and debug | pending | PR body |
| md5 of `gen`, main vs candidate, same host | pending | PR body |
| clippy -D warnings (bin + tests), rustfmt | clean | local |

## Results
Filled in when the jobs finish.

## What didn't work / not done
- Ordering is `Relaxed` on the counter, as before: the socket write and read in between are the
  synchronisation (system calls), not the atomic.
- The draft-model and pipeline solo paths share `Engine::job`, so they're covered; nothing else sends `Done`.
