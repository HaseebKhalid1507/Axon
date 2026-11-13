# Shared prefix KV pages on by default; #218's hold order; refeed claims counted apart (#219)

- Date: 2026-11-12, recast 11-13 (sharing alone). Branch: dec/share-default (on dec/cache-ram-auto, #226, with its
  flip stripped). Base: main 213465a. Status: open
- Pick / design: the 11-12 decision (Haseeb's answers to the 11-07 Fable 5.2 review, §4 "Shared prefix pages on by
  default: flip", with its conditions: fix #218, the debug suite, a soak of 4 h or more under a small pool, the kill
  switch, flip together with the tier); the foreman's M3/M4 rulings on #224's review (the flip's A/B cells, a
  sharing-on, tier-off cell); the lead's 11-13 rulings: the 4090 reading of the A/B stands, the tier stays off by
  default (#226), sharing flips on its own, and the soak runs the shipped config (sharing on, tier off).

## What changed
- **#218** (`index_pages`): a swapped block's index hold, and the call's keys, are taken only once `replace_pages`
  has mapped the index's page into the slot. A new block is still held at once (its page is the slot's own: the
  index holds a page per slot mapping it, and `release_rows` lets go by page). On success the index ends as before.
  The nit: a refeed claim alone no longer counts as a prefill step's progress (`feed` returns the tokens fed).
- **#219**: refeed claims count in `axon_refeed_shared_tokens_total`; `axon_prompt_shared_tokens_total` is
  admissions' prompt tokens again (the split, the issue's recommended option).
- **The flip**: `AXON_SHARE_KV` on unless `0` (the kill switch), was on for `1` only. The KV RAM tier stays off by
  default (#226), so a server with default flags shares pages and has no tier.

## The A/B, and why sharing flips alone (11-13)
`abshare.sh`, job `1112-233524-d38d`. The stop rule was in the PR body before the run. It was queued `--timing
bella` and rerouted by hand to an RTX 4090 (pod1, exclusive, FOREIGN_GPU 0 %), where the arms ran interleaved. Arm
÷ main medians over 5 rounds (rule: tok/s ≥ 0.98×, ITL p99 ≤ 1.02×):
- `share` (sharing on, tier off): C=1 1.001 / 0.998, C=8 1.000 / 0.980, C=16 1.001 / 1.004. **PASS.**
- `tier` (`AXON_SHARE_KV=0 --cache-ram 4G`): C=8 ITL p99 1.103, C=16 tok/s 0.977. FAIL.
- `both`: C=8 ITL p99 1.208, C=16 tok/s 0.977. FAIL.

The decision paired the two flips assuming both would pass. The failure is the tier's cost (#233, #234), not
sharing's, so the lead ruled that sharing flips alone.

Update 11-13 (the lead's roadmap ruling, asked for by shady on #226): the flips are decoupled in `docs/roadmap.md`
(week 1, item 2). Shared prefix pages go default-on on their own evidence (#218, the A/B's sharing-alone arm, the
4.5 h soak) and no longer wait on #233 or #234. `--cache-ram auto` default-on waits on #233 (first-request TTFT) and
#234 (the write-behind's 2–3 % tok/s and ITL p99), then a new A/B. This branch merged main (`4f347a3`, #224's new
roadmap; docs and infra only) to carry that edit.

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
alone; stop rule in the PR body before the run; result above). Then **the soak, the shipped config**: 4.5 h on a
debug-assertions build (page invariants checked after every step), sharing on, the tier off, a 128-page pool. A
second soak with sharing on and `--cache-ram 2G` (the opt-in combination) runs at `--low` as extra evidence.

## Results
Pending at the time of writing; the PR body has them as they land.
