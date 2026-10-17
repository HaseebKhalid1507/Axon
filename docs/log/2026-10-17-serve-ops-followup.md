# Serve ops follow-up: shady's non-blocking items from the #61 re-review  (PR #62)

- Date: 2026-10-17. Branch: fix/serve-ops-followup. Base: main 3310ba3 (#61 merged). Status: open.
- Source: shady's re-review of #61 (MERGE 9/10, r10 thread, reply to msg-b4e103d6): non-blocking follow-ups.

## What changed
- (a) Unauthenticated bodies: I chose the stronger fix over rewording. A request that carries a body
  (`Content-Length` > 0 or chunked) and lacks the key is refused right after its head, whether or not it sent
  `Expect: 100-continue` (before, only an `Expect` request was). Its body is never buffered or parsed, and no
  `100 Continue` is sent. The connection then closes, and `linger` drops what the client already sent, at most
  2 s / 4 MiB, so the 401 isn't lost to a reset. That is the honest bound: the server still reads and discards
  those bytes, but it no longer holds a MAX_BODY buffer or parses JSON for an unauthenticated client. A request
  without a body that lacks the key still gets its 401 on a kept-alive connection. #61's log line and the
  comment in `connection()` are reworded to match.
- (b) `AXON_API_KEY` is read with `var_os`. A value that isn't UTF-8 is a startup error, where before it read as
  unset and auth silently turned off.
- (c) `--api-key` really wins: the variable is read only when the flag is absent, so a blank or non-UTF-8
  variable no longer aborts a server started with `--api-key`. The comments say so.
- (d) `connection()` resets `head_only` before answering a read error. A HEAD followed by a refused request on
  the same keep-alive connection used to get a 401 with headers only and no body.
- (e) A 401 from the head-time check is counted under its route label (`Conn::refused` keeps the path), not
  under `other`.
- (f) `api_key_env_and_empty` calls `start()` directly and asserts `ErrorKind::InvalidInput` for "", " " and "\n"
  (no model needed). It also covers a non-UTF-8 value.
- (g) The SIOCOUTQ warn-once is silent under `--quiet`. The arch list lives only in `ops::OUTQ_PROBE`:
  `unacked()` is compiled on every Linux and returns None where the probe is off.
- No changes to `/slots` or `engine.rs` (#60 owns them).

## How it was verified
| gate | result | job |
|---|---|---|
| serve:: release / debug-assertion, avante | 83/0, 83/0 | 1017-072023-ad59, 1017-072023-e4af |
| serve:: release / debug-assertion, bella | 83/0, 83/0 | 1017-072024-10aa, 1017-072024-ffb7 |
| md5 main 3310ba3 vs candidate (gen + serve with ops flags on) | 26/26 same | 1017-072246-7480 |
| fuzz_serve.py --ops / plain | 226 / 183 checks, 0 findings | 1017-072655-0fa1 |
| clippy -D warnings (bin + tests), rustfmt | clean | local |

New test: `serve_api_key_error_after_head`. It sends HEAD /health, then a POST with Expect and without the key on
the same connection, and expects a 401 with a body. A POST without the key or Expect is closed after its 401.
Both are counted under their routes, and nothing is counted under `other`.

## Note on the slot-idle bound (MEASURED, not changed here)
In this fuzz run (bella CPU, 0.5B, `--slot-idle-ms 1000`), the default-buffer stalled reader absorbed the whole
1900-token reply: 21.7 s and no eviction. In #61's run (1017-061629-9526) it was evicted after 792 tokens and
6.3 s. Linux autotunes the receive and send buffers, so with default buffers the point where the idle clock
starts varies from run to run, and it can exceed a 1900-token reply. #61's "roughly 500+ tokens" is a lower
bound, not a typical value. The 4 KiB case was stable: 158 tokens and 1.7 s here, 202 and 1.7 s before.

## Update: shady's #62 review (r10 thread msg-90ee277d)
- MUST, fixed: the head-time gate reused `refused()`, which exempts /health and every OPTIONS, so `POST /health` or
  an `OPTIONS` with a body and no key still had its whole body (up to 1 MiB) read. With a key set, the gate now
  checks every body-carrying request (Content-Length > 0 or chunked) against `ops::authorized` alone, so no body
  gets through without the key, whatever the route. Bodiless /health and OPTIONS stay open.
- A refused HEAD gets its 401 with headers only (`Conn::refused` keeps the method).
- Auth comes before the 413: the MAX_BODY check moved from `parse_head` to `read_request`, after the gate. An
  unauthenticated client sees a 401 for an oversize body, and a client with the key still gets the 413.
- Caveat, documented: an unauthenticated upload still going after `linger`'s 2 s (or past 4 MiB) can be reset
  before the client reads the 401.
- Key resolution is `api_key_of(flag, env)`; test `api_key_flag_wins_over_bad_env` covers a flag set with a
  blank or non-UTF-8 env var (the variable isn't read at all).
- Tests: `serve_api_key_no_body_without_the_key` sends only the head, so a server that read the body would
  answer 408 after 30 s. It covers POST /health with 1 MiB declared, OPTIONS with a body, chunked /health, GET
  /health with a body, a 9 MB body (401, not 413; 413 with the key), and HEAD with a body (a 401 with headers
  only). Fuzz cases `auth:no-key-*` cover the same and add an Expect variant on /health.

## What didn't work
Nothing was dropped. The weaker option, rewording the docs and keeping the Expect-only check, was rejected:
any client could still make the server buffer and parse a 1 MiB body without the key.
