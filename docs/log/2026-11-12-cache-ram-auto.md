# `--cache-ram auto` by default: the smaller of 4 GiB and ¼ of the RAM available, off under a binding cgroup

- Date: 2026-11-12. Branch: dec/cache-ram-auto (on dec/help-docs, #225). Base: main 213465a. Status: open
- Pick / design: the 11-12 decision (Haseeb's answers to the 11-07 Fable 5.2 review, §4 "`--cache-ram auto` default:
  flip, smaller"; roadmap week 1, #224), with the foreman's ruling on #224's review: the cap is the smaller of the two,
  "free" is the figure `auto` already read, and "under a cgroup limit" means the cgroup is the tighter bound.

## What changed
- **`axon serve` defaults to `--cache-ram auto`** (was `0`, off). `auto` is the smaller of 4 GiB (`tier::AUTO_CEIL`,
  was 8 GiB) and a quarter (`AUTO_DIV`, was a half) of the RAM available after the model loads
  (`gguf::mem::avail()`: the host's `MemAvailable`, or the tightest memory cgroup's room if that is less).
- **Off when a cgroup is the tighter bound** (`Bound::Cgroup`: a container, a systemd scope with `MemoryMax`).
  Pinned memory counts against the limit and can't be reclaimed. A cgroup looser than the host's free RAM doesn't
  count. The banner says why (`CacheRam::auto_off`), e.g. `KV RAM tier off (--cache-ram auto: a memory limit binds:
  cgroup /app.slice/x.scope memory.max 6144 MiB, 5.00 GiB available, under the host's free RAM; --cache-ram SIZE
  turns it on)`.
- **An explicit `--cache-ram SIZE` is unchanged**, and works under a cgroup too. **Kill switch:** `--cache-ram 0`,
  or the new `--cache-ram off`.
- `--help`, `docs/serving.md`, CHANGELOG.

## Why
The tier is exclusive since #201: an evicted 25k-token session resumes in 0.141 s against a 2.713 s re-prefill on an
RTX 4090, and in 0.36–0.51 s against 8.6–8.8 s on bella (two TAINTED reads). Bella missed the old gate's 0.3 s, which
the decision waives (17–24×). The objection to the default was the size: `auto` pinned up to 8 GiB at server start on
a 30 GB desktop whose session was OOM-killed twice on 10-13. 4 GiB holds a 7B f16 session of about 75k positions
(56 KiB each). A 16 GB desktop with 6 GiB free after the model gets 1.5 GiB, still one warm 27k-position session.

Under a cgroup the default stays off because pinned memory there is charged to the limit and can't be reclaimed, so
a default that took it could get the server, or its neighbours in the cgroup, OOM-killed.

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
- No timing round, per the foreman's brief: the bits are equal and the tier is opt-out. #224's review ruling (rule 4)
  since asks a default flip for an A/B on bella (C=1 tg128 decode, ITL p99, serve throughput at C=8 and C=16):
  whether this PR runs it, or the sharing flip's A/B (the tier and sharing on together, plus a sharing-on, tier-off
  cell) covers it, is the foreman's call (pending).

## Measuring what users get
axon-run puts every job in a systemd scope with `MemoryMax` (on bella 22,000 MiB, under `app.slice`'s 23,552 MiB).
Whether that scope binds depends on what the host has free at the time: on 11-12 at 23:10 a job's scope had more room
than the host's 14 GB `MemAvailable`, so the host bound and `auto` would be on; with the host freer, the scope binds and
`auto` is off. Gate and board runs therefore pass `--cache-ram` explicitly (here `4G`, `auto`'s size on a roomy host)
to measure the default users get, and print the banner of each default-flags server to say which way it went.
RunPod containers carry a memory limit too, so the same holds on pods.
