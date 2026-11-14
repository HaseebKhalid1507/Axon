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
PR body (job IDs): the full tier on RTX 4090 pods (the serve tests that don't set `share` now run with it on, the
default), a `serve::` run with `AXON_SHARE_KV=0`, `md5share-v4.sh` against main (gen; serve on, off, tier on and
off, shuffled, undersized pools; pressure), the A/B of rule 4 (four arms: main, both flips, sharing alone, the tier
alone; stop rule in the PR body before the run; result above). Then **the soak, the shipped config**: 4.5 h on a
debug-assertions build (page invariants checked after every step), sharing on, the tier off, a 128-page pool. A
second soak with sharing on and `--cache-ram 2G` (the opt-in combination) ran at `--low` as extra evidence.

## Results
Gated code fc845c8.
- **Bits.** `md5share-v4.sh` against main (1113-004233-e238): gen, 0 of 30 differ. Serve: every arm's replies were
  equal to main's in every cell. That held under a 128-page pool too (`on128`: 3,840 shared tokens, 743 evictions,
  3 preemptions, 45 unfeeds).
- **Suites.**
  - Full tier, release: 223, 379 and 243 passed. Shard 2 is 5bb0: the first shard 2, 88ee, was killed while stuck
    behind #232's test, and that test passed alone (48ac).
  - Full tier, debug: 223, 380 and 243 passed.
  - `serve::` with `AXON_SHARE_KV=0`: 184 passed.
  - `serve_share_cancel_frees`, 30 fresh processes in a row, the lead's requirement: 30 of 30 on bella (7c83). It
    was also 30 of 30 on a packed pod (c7ee), but that job ended rc 1 on the script's own output check, which broke
    under `--nocapture`. The check was fixed before 7c83.
- **The A/B** (d38d): sharing alone passed all six cells (above).
- **The soak, the gate** (1113-171545-892b, bella). The run: 4.58 h, 12,173 requests, debug-assertions build,
  sharing on, the tier off, 128 pages.
  - RSS 198.6–200.6 MiB, a slope of 0.03 MiB/h (0.7 MiB over 24 h against a 64 MiB bound). VRAM flat at 1,856 MiB.
  - No panic, so the page invariants held after every step.
  - 19 identity runs under load, each equal to main's solo run.
  - No 503, and every slot idle at the end.
  - 112,320 prompt tokens were mapped from the index, and 5,101 warm pages were evicted for room.
  - **No preemption and no unfeed in 4.5 h.** The small pool's pressure fell on the index (evictions), never on a
    running request: the mix's requests are short and its long ones fit. Preemption with sharing on rests on
    `serve_share_pressure`, in every suite, and md5's pressure arm (above), not on the soak.
- **The soak with the tier** (1113-222946-bd87, `--cache-ram 2G`, `--low`; extra evidence). The run: 4.60 h, 12,889
  requests.
  - RSS 202.9–205.5 MiB, a slope of 0.31 MiB/h (7.3 MiB over 24 h, within the 64 MiB bound). VRAM flat.
  - 19 identity runs, each equal to main's solo run.
  - 127,040 shared prompt tokens.
  - The tier: 776 hits, 193,997 restored tokens, 25,334 pages written, none spilled. No preemption here either.
- **The soak harness failed first** (11-13). The first smoke and soak died at load: `soakshare.sh` passed no
  `--device`, and serve's default device is the CPU, which refuses `--parallel 8`. Fixed with `--device cuda` and
  rerun (smoke 1dd6, then the two soaks).

## Review round (11-14)
shady 8/10 SHIP and Fable (second family) 8/10 APPROVE at `bf45114`, no MUSTs. Fixed in `e07adff` (code), `56d01d7`
(docs) and `8d4fef0` (the soak harness); what was left out is in #267, each with its reason.
- **Fast kernels** (shady): under `--numerics fast` on an FP8 checkpoint (which `--datacenter` implies) the rows
  change with a prompt's chunking, so two requests' copies of a block can differ in their bytes. Debug builds assert
  byte equality on every duplicate block, and no gate held fast mode and sharing together. Sharing is now off there
  unless `AXON_SHARE_KV=1` (then within fast mode's tolerance, the debug check skipped).
  `serve_share_off_under_fast_kernels` pins the default on an sm_89+ GPU.
- **The kill switch where it's needed** (shady): an engine failure's exit line adds "(shared prefix KV pages are on;
  `AXON_SHARE_KV=0` turns them off)" while sharing is on. The start line says off too, and why, and echoes a value
  that reads as on without being `1` (`off`, `false`).
- **`cached_tokens`** (Fable): it now counts pages another client computed. serving.md says what a client can learn
  from it and when to set `AXON_SHARE_KV=0`. A per-key salt is #267.
- **`serve_share_cancel_frees`** (Fable): it dropped the third request after the others generated 6 more tokens,
  which doesn't prove the third was admitted (a09b1bd's one red). It now waits for the third's admission (the
  shared-tokens counter's delta).
- **Docs that still said opt-in** (both): START-HERE.md, results.md, methodology.md and the design.
- **A soak that preempts** (both). shady worked out why the two soaks couldn't: admission reserves pages for at most
  1024 of a running request's tokens to come (`GROW_TOKENS`), and the mix's replies are shorter, so 128 pages only
  meant more evictions. `scripts/soak.py --hogs K` adds replies that don't stop (a count, 2000–3000 tokens), and
  the verdict then needs `preempted`, `prefills_unfed` and `axon_refeed_shared_tokens_total` above 0 at the end
  (else INCONCLUSIVE, "not evidence"). The CPU smoke (one slot, no pages) ran the hogs and read INCONCLUSIVE, as it
  should. The soak itself (`soakpreempt.sh`: 4.5 h, 4 hogs + 12 clients, 128 pages, debug build, sharing on, tier
  off) is queued at `--low` on bella; it isn't a gate of this round, and its result goes on the PR.
