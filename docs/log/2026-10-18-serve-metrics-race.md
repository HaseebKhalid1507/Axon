# serve: count a request before its response is written (PR #TBD)

- Date: 2026-10-18. Branch: fix/serve-metrics-race. Base: main 50cd609. Status: open
- Pick / design: flake fix, `serve::tests::serve_api_key` ("no entry found for key"), 2 failures in ~130 logs.

## What changed
- `src/serve/http.rs`: `Conn::on_status`, a one-shot `FnOnce(u16)` taken and called by `respond` and
  `start_stream` (the new `Conn::start`) with the status, before the response's first byte is written. An interim
  `100 Continue` doesn't call it.
- `src/serve.rs` `connection()`: instead of counting (`Metrics::request`) after the response was written, it arms
  `on_status` with the request's route label before answering (error path: before the 401/413/...; routed path:
  before `route()`), and disarms it afterwards. Labels are unchanged (error path: the refused request's route, else
  "other"; routed path: `route_label(path)`). A request that sends nothing is still not counted.
- Tests: `Server::hold` (cfg(test)): while on, each connection thread stops right after its response, before
  whatever follows it (the old count, the linger). New test `serve_counts_a_request_before_answering_it`.
  `serve_api_key` is unchanged (still strict).

## Why
Both failures panic on the **`/v1/completions` 200** lookup, not on the 401 one: the panic line is 2679 at
0b2f029 (job 1018-163238-d187, avante, release) and 2678 at q23k 48f0d03 (job 1018-014632-e746, bella, debug);
in both trees the `/v1/models` 401 lookup is 4 lines earlier (2675 / 2674). That POST is the last request before
the test's `GET /metrics`. The client reads the 200 by its Content-Length and connects again at once, while the
server thread that wrote it has yet to return from `route()` and call `metrics.request(...)`. The error path had
the same order (respond, then count, then linger); there it just never lost the race in a log.

## How it was verified
| gate | result | job ID |
|---|---|---|
| new test on the OLD order (71fac91: hook + test, no fix), release | FAILED as expected, first case: `GET no key: answered, not yet counted` (0.35 s) | 1018-175131-1144 |
| new test on the fix (b704847), release | passed | 1018-175131-37a7 |
| `serve::` release, bella `--vram 6000` | pending | 1018-175516-b054 |
| `serve::` debug-assertions, bella `--vram 6000` | pending | 1018-175517-05c4 |
| AXON_REPEAT=20 serve_api_key + new test, release (stress.o) | pending | 1018-175517-dd54 |
| AXON_REPEAT=20 serve_api_key + new test, debug (stress.o) | pending | 1018-175518-76d2 |
| fast tier, release | pending | 1018-175525-1b05 |
| fast tier, debug-assertions (CPU) | pending | 1018-175525-3554 |
| clippy -D warnings (bin + tests), rustfmt | clean on b704847 | local |
| md5 of `gen` | not run: no model, sampling or engine code changed (HTTP connection layer only) | |

## Results
The new test is deterministic, not timing-based: with the hold on, the old order fails on every run on its first
case; the fix passes. No performance effect expected (one Arc clone and one boxed closure per request); not measured.

## What didn't work
The brief's first suspect (the 401 error path) has the same ordering bug and is fixed too, but it isn't what
failed: the 401 count is made 10+ requests before `/metrics` is read.
