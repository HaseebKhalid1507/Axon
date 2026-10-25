# RAM available honours cgroup memory limits  (PR #PRNUM)

- Date: 2026-10-25. Branch: fix/ram-cgroup. Base: main b2068e7. Status: open
- Pick / design: issue #129 (found while diagnosing #127's bella gate, #128).

## What changed
- New `src/gguf/mem.rs`: `mem::avail()` = the smaller of the host's `/proc/meminfo` `MemAvailable` and the
  tightest memory cgroup level from the process's cgroup up to the hierarchy's mount root, plus what bound it
  (`Bound::Host` or `Bound::Cgroup { path, limit, v1 }`).
  - cgroup v2, per level: `memory.max − (memory.current − inactive_file − active_file)` (`memory.stat`; file pages
    are page cache the kernel reclaims before it OOM-kills). `max` = no limit; the v2 root has no `memory.max`.
  - cgroup v1 (memory controller on a v1 hierarchy, which wins on a hybrid system): the same walk with
    `memory.limit_in_bytes` (≥ 2^62 = no limit: the kernel's PAGE_COUNTER_MAX), `memory.usage_in_bytes`,
    `total_inactive_file + total_active_file` (else the cgroup's own `inactive_file + active_file`). Every ancestor
    counts (kernels ≥ 5.x always account hierarchically).
  - The process's cgroup from `/proc/self/cgroup`, the mount from `/proc/self/mountinfo` (`cgroup2`, or `cgroup`
    with `memory` in the super options), else `/sys/fs/cgroup` (v2) / `/sys/fs/cgroup/memory` (v1). A mount whose
    root is a subtree (a container without its own cgroup namespace) is mapped by stripping that prefix.
  - Anything unreadable or unparsable is "no limit at that level"; no cgroupfs at all leaves the host's figure,
    as before. Everything is read under a `root` path, so the tests build synthetic v1/v2 trees.
  - Not counted: `memory.high` (throttles, doesn't kill), swap (`MemAvailable` doesn't count it either).
- `gguf::mem_available()` returns `mem::avail()`'s bytes, so every caller follows the limit: `moe::check_ram`,
  the copy-vs-map choice (`copy_choice`), the read-ahead guard, `axon serve`'s CPU context plan (`cpu_ctx`).
- `moe::check_ram`'s refusal now ends `(bound by host MemAvailable)` or
  `(bound by cgroup <path> memory.max <N> MB)`; the map-instead-of-copy note names the bound too.

## Why
Inside a container or a systemd scope whose memory limit is below the host's `MemAvailable`, Axon accepted a MoE
load it couldn't fit and the kernel OOM-killed it, instead of the clean "needs N GB, M GB available" refusal.
Datacenter / Docker / K8s deployments almost always run under a memory limit. On bella every agent process already
runs under `app.slice` (MemoryMax 23 GB) on a 30 GB host, and every axon-run job in its own scope below that: the
real-machine unit test prints `mem avail: 17765 MB (cgroup /user.slice/user-1000.slice/user@1000.service/app.slice
memory.max 23552 MB)` from a bash shell there.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| refusal demo, bella, MoE at `MemoryMax=1000M` | main: OOM-killed (rc 137, scope Result=oom-kill); candidate: clean refusal (rc 2) naming the scope | 1025-053958-9d5b, 1025-054020-70da, 1025-054259-1d9c (adds a 4500M fits case) |
| release suite (fast tier) | REL | 1025-054002-f395 (bella) |
| debug-assertion suite (fast tier, CPU) | DBG | 1025-054002-957f (avante, CPU) |
| md5 of `gen`, main b2068e7 vs candidate, same host | 24/24 identical (4 models × 3 prompts × cpu, cuda f16) | 1025-053959-b9a6 |
| clippy -D warnings (lib, lib tests, bin), rustfmt | clean | local |
| unit tests on synthetic trees | 7 in `gguf::mem::tests`, `moe::tests::ram_refusal_names_the_bound` | in the suites |

## Results
MEASURED, bella (Core Ultra 9 275HX, 30 GB, Linux 7.2.11 CachyOS, systemd 262, cgroup v2), job 1025-054020-70da: each binary
runs `gen <model> "The capital of France is" -n 8 --device cpu` in its own `systemd-run --user --scope
-p MemoryMax=… -p MemorySwapMax=0`, host MemAvailable ~21 GB.

| case | cap | main b2068e7 | candidate |
|---|---|---|---|
| Qwen3-30B-A3B-Pruned-2B Q4_K_M (MoE, 1.5 GB file; check needs 3.2 GB) | 1000M | OOM-killed, rc 137, no message | refused, rc 2: `qwen3moe: the CPU model needs about 3.2 GB of RAM (its repacked experts 1.1 GB) and only 1.0 GB is available (bound by cgroup /user.slice/…/memcg-demo-…-moe-cand.scope memory.max 1000 MB)` |
| qwen2.5-0.5b Q4_K_M (dense, 491 MB) | 400M | OOM-killed, rc 137 | OOM-killed, rc 137 (unchanged: see below) |
| qwen2.5-0.5b Q4_K_M | none | rc 0, md5 8df6a47e… | rc 0, md5 8df6a47e… |

No speed claim; pp/tg n/a (a few small file reads per load; no kernel or numerics change).

## What didn't work / gaps
- **Dense CPU models have no RAM check at all** (main and candidate alike): `check_ram` only runs for MoE models.
  Under a cap below the dense model's repacked r8 arena the load is still OOM-killed (the 0.5B at 400M above). With
  this change a dense model that reads in place now maps instead of copying under a tight cgroup, but nothing
  refuses a dense load that can't fit. Follow-up issue: a `check_ram` for dense CPU models.
- `memory.high` could be added as a softer bound (a load over it is throttled hard, without swap nearly stalled);
  left out to keep the issue's "will be OOM-killed" semantics.
