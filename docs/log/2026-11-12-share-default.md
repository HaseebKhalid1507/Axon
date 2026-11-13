# Shared prefix KV pages on by default; #218's hold order; refeed claims counted apart (#219)

- Date: 2026-11-12. Branch: dec/share-default (on dec/cache-ram-auto, #226). Base: main 213465a. Status: open
- Pick / design: the 11-12 decision (Haseeb's answers to the 11-07 Fable 5.2 review, §4 "Shared prefix pages on by
  default: flip", with its conditions: fix #218, the debug suite, a soak of 4 h or more under a small pool, the kill
  switch, flip together with the tier); the foreman's M3/M4 rulings on #224's review (the flip's A/B cells, a
  sharing-on, tier-off cell).

## What changed
- **#218** (`index_pages`): a swapped block's index hold, and the call's keys, are taken only once `replace_pages`
  has mapped the index's page into the slot. A new block is still held at once (its page is the slot's own: the
  index holds a page per slot mapping it, and `release_rows` lets go by page). On success the index ends as before.
  The nit: a refeed claim alone no longer counts as a prefill step's progress (`feed` returns the tokens fed).
- **#219**: refeed claims count in `axon_refeed_shared_tokens_total`; `axon_prompt_shared_tokens_total` is
  admissions' prompt tokens again (the split, the issue's recommended option).
- **The flip**: `AXON_SHARE_KV` on unless `0` (the kill switch), was on for `1` only.

## Why
#210 and #211 passed at #216's code: same bits, no measured cell worse (C=1 decode, ITL p99, C=8 and C=16
throughput 1.000× in #207's servetime), eight subagents' TTFT 0.591× on a shared 6,006-token prefix. The review's
conditions before the flip: #218 (a hold taken before the swap it depends on), the debug suite with sharing on, a
soak under pool pressure. The board's Synaps-shaped trace shares only 4–7 % of its prompt tokens: the win is
subagent fan-out, not every agent workload.

## How it was verified
PR body (job IDs): the full tier on bella (the serve tests that don't set `share` now run with it on, the
default), a `serve::` run with `AXON_SHARE_KV=0`, `md5share-v4.sh` against main (gen; serve on, off, tier on and
off, shuffled, undersized pools; pressure), the A/B of rule 4 (four arms: main, both flips, sharing alone, the tier
alone; stop rule in the PR body before the run), and the 4.5 h soak on a debug-assertions build (page invariants
checked after every step) with sharing on, `--cache-ram 2G` and a 128-page pool.

## Results
Pending at the time of writing; the PR body has them as they land.
