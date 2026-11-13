# `--cache-ram auto`: the smaller of 4 GiB and ¼ of the RAM available, off under a binding cgroup (default still off)

- Date: 2026-11-12, flip stripped 11-13. Branch: dec/cache-ram-auto (on dec/help-docs, #225). Base: main 213465a.
  Status: open
- Pick / design: the 11-12 decision (Haseeb's answers to the 11-07 Fable 5.2 review, §4 "`--cache-ram auto` default:
  flip, smaller"; roadmap week 1, #224), with the foreman's ruling on #224's review: the cap is the smaller of the two,
  "free" is the figure `auto` already read, and "under a cgroup limit" means the cgroup is the tighter bound.

## Outcome (11-13): the default flip failed its A/B and was stripped
- #228's A/B (`1112-233524-d38d`; queued `--timing bella`, rerouted by hand to an RTX 4090, interleaved on that one
  host, FOREIGN_GPU 0 %) has a `tier` arm, `AXON_SHARE_KV=0 --cache-ram 4G`, which is this PR's default on its own.
  It **failed 2 of 6 judged cells: C=8 ITL p99 1.103×, C=16 agg tok/s 0.977×** (rule: tok/s ≥ 0.98×, ITL p99 ≤
  1.02×). The first request after each start also waited 3.2–4.0 s (against 42–44 ms) while 4 GiB was pinned.
- The lead's ruling: the 4090 reading stands (no re-run on bella to get a pass), `auto` stays off by default, and
  the rest merges. Filed: #233 (the first-request TTFT) and #234 (the write-behind cost). The default flip gets
  re-A/B'd in its own PR once both are fixed.

## What changed
- **`axon serve`'s default is unchanged: `--cache-ram 0`, off.** `auto` is now the smaller of 4 GiB (`tier::AUTO_CEIL`,
  was 8 GiB) and a quarter (`AUTO_DIV`, was a half) of the RAM available after the model loads
  (`gguf::mem::avail()`: the host's `MemAvailable`, or the tightest memory cgroup's room if that is less).
- **`auto` is off when a cgroup is the tighter bound** (`Bound::Cgroup`: a container, a systemd scope with `MemoryMax`).
  Pinned memory counts against the limit and can't be reclaimed. A cgroup looser than the host's free RAM doesn't
  count. The banner says why (`CacheRam::auto_off`), e.g. `KV RAM tier off (--cache-ram auto: a memory limit binds:
  cgroup /app.slice/x.scope memory.max 6144 MiB, 5.00 GiB available, under the host's free RAM; --cache-ram SIZE
  turns it on)`.
- **An explicit `--cache-ram SIZE` is unchanged**, and works under a cgroup too. **New:** `--cache-ram off`, the
  same as `0`.
- `--help`, `docs/serving.md`, CHANGELOG.

## Why (the 11-12 case for the flip; the cap and the cgroup rule stand without it)
The tier is exclusive since #201: an evicted 25k-token session resumes in 0.141 s against a 2.713 s re-prefill on an
RTX 4090, and in 0.36–0.51 s against 8.6–8.8 s on bella (two TAINTED reads). Bella missed the old gate's 0.3 s, which
the decision waives (17–24×). The objection to the default was the size: `auto` pinned up to 8 GiB at server start on
a 30 GB desktop whose session was OOM-killed twice on 10-13. 4 GiB holds a 7B f16 session of about 75k positions
(56 KiB each). A 16 GB desktop with 6 GiB free after the model gets 1.5 GiB, still one warm 27k-position session.

Under a binding cgroup `auto` is off because pinned memory there is charged to the limit and can't be reclaimed, so
`auto` taking it could get the server, or its neighbours in the cgroup, OOM-killed.

## How it was verified
- New tests: `tier::tests::cache_ram_auto_cap` (the cap arithmetic: a quarter below 16 GiB available, 4 GiB from 16
  GiB up, 0 when unknown or cgroup-bound, the reason's text), `cache_ram_auto_off_under_a_binding_cgroup` (the rule
  on what `gguf::mem::avail_at` reads from a synthetic /proc and cgroupfs: a 6 GiB scope under a 12 GiB host turns
  `auto` off and an explicit size still on; the scope at 64 GiB leaves the host as the bound and `auto` at 3 GiB),
  `cache_ram_parses` (`off`), `cli::tests::help_is_current`.
- Gates (job IDs in the PR body): the fast tier, release and debug; the md5 of `gen` against main; a serve md5
  (`md5ram.sh`: #216's `serve_md5_share-v3.py`, 7 sessions × 3 turns over 4 slots, greedy and seeded-sampled) where
  every reply equals main's at default flags for the candidate at default flags, at `--cache-ram 4G`, at
  `--cache-ram off`, and on an undersized pool (`AXON_KV_PAGES=48`) at default and at 4G, on 7B f16 and q8,
  gemma-3-1b and Qwen3-0.6B q8 on the GPU and the 0.5B on the CPU.
- The A/B: #228's `tier` arm (the foreman's ruling, 11-12), FAIL, above. After the strip, this PR makes no timing
  claim. Its gates at the new head are for correctness only: the fast tier and the md5s against main.

Jobs (axon-run) at `a09b1bd`:

| gate | job | result |
|---|---|---|
| fast tier, release, `--shard 3` | 1113-003537-9d65, -7e61, -d7f5 | 328/1 (`serve_share_cancel_frees`, see below), killed behind `gpu_fp8_graded_vs_fp8_ref` (#232), 251/0 |
| shard 1 again; the test alone | 1113-014731-36af; 1113-014732-89c0 | 329/0; ok |
| shard 2 with `--skip …gpu_fp8_graded_vs_fp8_ref`; that test alone (bella) | 1113-011046-9828; 1113-010621-7899 | 247/0; ok |
| `serve_share_cancel_frees`, 60 fresh processes | 1113-130728-fd2c | in the PR body |
| fast tier, debug, CPU queue | 1113-003537-916f | 846/0 |
| md5 of `gen`; serve md5 (`md5ram.sh`) | 1113-003537-7892; 1113-003538-4cac | 0 of 30 differ; every reply equal |

The one shard-1 failure (`axon_prompt_shared_tokens_total` 0 where the test wants the third request's 5 pages) is in
main's test on main's code path: the PR changes neither `serve/tests.rs` nor `serve/engine.rs`, and the test asks for
no tier. The classification is on the PR (comment 6861109452).

## After shady's review (11-13)
8/10 SHIP-AFTER-FIXES. Its MUST was Process v2 rule 1: the diff touches `src/serve/tier.rs`, so the full tier runs, or
the lead waives it in writing. The lead ruled for a focused run instead of either: the debug-assertion build's tier
and cache-RAM tests on a GPU host (`dec-456ec4c-tier-dbg-gpu`, job IDs in the PR body). The reason: the change is the
sizing arithmetic in `CacheRam` (`bytes`, `auto_off`, `off`) plus its tests, all of which run before the tier starts.
`PoolTier`, `CacheTier` and the write-behind are untouched. The full tier's 18 extra tests (bigger models) set no
`--cache-ram`. The fast tier's debug half ran on the CPU queue, where the GPU tier tests skip themselves. The branch
merged #225's fixes (`456ec4c`). The SHOULDs are in #253.

## Measuring what users get
axon-run puts every job in a systemd scope with `MemoryMax` (on bella 22,000 MiB, under `app.slice`'s 23,552 MiB).
Whether that scope binds depends on what the host has free at the time: on 11-12 at 23:10 a job's scope had more room
than the host's 14 GB `MemAvailable`, so the host bound and `auto` would be on; with the host freer, the scope binds and
`auto` is off. Gate and board runs therefore pass `--cache-ram` explicitly (here `4G`, `auto`'s size on a roomy host)
to measure the default users get, and print the banner of each default-flags server to say which way it went.
RunPod containers carry a memory limit too, so the same holds on pods.
