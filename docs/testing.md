# Testing

How the suite is built, what makes it slow, and which tests to run on which round of a PR. Building and
running follow the builder rules (no cargo on bella, every model or GPU run through `axon-run`). The commands
below are the plain ones for a machine without those rules.

## The suite

One libtest binary (`src/main.rs --test`, about 550 tests, 35 `#[ignore]`d benchmarks and measurements). Two
builds of it gate a PR:

- **release**: `-C opt-level=3` (Cargo's `[profile.test]` does the same).
- **debug-assertion**: the same build plus `-C debug-assertions -C overflow-checks`.

Most of the time goes to real-model tests: CPU against GPU logits, and the bit-identity contracts (decode ==
batched prefill == verify, under every batching) on 0.5B to 8B models. Some things that matter when you read
or time it:

- **CPU forwards don't run in parallel across tests.** `pool::Pool` is process-wide and runs one job at a
  time. Four test threads that run CPU models take turns on one pool, so a test's CPU work costs suite wall
  time no matter how many test threads there are. Removing CPU work is the lever (see below).
- **GPU tests are serialized on every card.** `testutil::gpu` takes a process-wide lock until the test's
  thread ends. Any GPU test opens the GPU through it. `AXON_TEST_GPU_SERIAL=0` turns it off, for hunting
  only; any other value, or none, leaves it on. Below 16 GiB, concurrent tests ran out of VRAM. On 24 GB
  pods, GPU tests in flight together were observed to lose the CUDA context to ILLEGAL_ADDRESS. That was
  seen on EPYC hosts with the AMD IOMMU on, while processes were being started. The cause is a hypothesis:
  [log](log/2026-10-18-cuda-concurrency.md). A thread spawned by a GPU test must not call `testutil::gpu()`.
  That would deadlock on its parent's lock, so it panics after `AXON_TEST_GPU_LOCK_TIMEOUT` s (default 2700).
- **Packed pods are slow.** Three jobs share a 4090 and its vCPUs, and a test there takes 3-8x its exclusive
  time. Serial on an exclusive 4090, the whole release suite is ~2200 s. (The packed 4-thread figure,
  ~2400 s, was measured with GPU tests concurrent. It hasn't been re-measured with the lock.)
- **Model loads are cheap once the page cache is warm.** A copy load or `Qwen2::new` repack of a 0.5-1B model
  takes 0.1-0.4 s uncontended. The 8-26 s seen on cold, packed pods were disk and contention, not a reason
  for process-wide model fixtures.

### Per-test times

libtest can time every test (nightly options; `RUSTC_BOOTSTRAP=1` allows them on stable):

```
RUSTC_BOOTSTRAP=1 ./axon-test -Z unstable-options --report-time --format json --test-threads 1 > t.json
```

`axon-run` forwards only `AXON_*` variables, so for a remote run, link a constructor that sets
`RUSTC_BOOTSTRAP` into a measurement-only build (`-C link-arg=bootstrap.o`; the log entry has it). Time on an
exclusive host (`--timing 4090`) with `--test-threads 1`. Packed or multi-threaded timings mostly measure
the neighbours.

### Writing tests that stay fast

- **One reference pass per input.** If several models are graded against the same CPU reference, run the
  reference once: `trace` + `compare_trace`, `compare_each!`, `compare_forced2` and `continuations_by` in
  `src/cuda/qwen2/tests.rs`. Each model under test still gets exactly the calls `compare` would make, and the
  reference makes the same calls once instead of once per model.
- Don't load a model just to check that its file is there (use the `*_path()` helpers).
- Don't shorten a prompt or a position range unless you can say why the assertion doesn't depend on its
  length. The 1200-token batchings exist to cross the max batch and the attention tiles.

## Tiers

`scripts/test-tier.sh` submits a tier through `axon-run`:

```
scripts/test-tier.sh fast REL_BIN [DBG_BIN]    # intermediate review rounds
scripts/test-tier.sh full REL_BIN DBG_BIN      # the final round, before merge
```

- **fast**: the release suite minus `scripts/test-fast-skip.txt`, plus, with `DBG_BIN`, the debug-assertion
  suite on the CPU queue (`axon-run --cpu`, where GPU tests skip themselves, so it takes no GPU slot). The
  skip list holds the bigger-model tests (Llama 3B/8B, Qwen3 4B, the Qwen2.5 1.5B batchings and grading) and
  the NVRTC builds for sm_70/75/80/90a. They aren't redundant: each is the only test of some shape or weight
  type (head size 128 with GQA 3/4/6, K-quant weights at head size 128). The file says which, and the fast tier
  accepts that gap between review rounds. The NVRTC builds come back in whenever the branch or the working tree
  changes `src/cuda/*.cu`, `src/cuda/mod.rs` or `src/cuda/ffi.rs`. On an exclusive 4090 the list is ~39 % of the
  release suite, and the fast tier's release part is ~55 % of main's full release suite. The script fails if a
  name in the list no longer matches exactly one test, or if the binary lists no tests.
- **full**: every test in both builds on GPU hosts, then the md5 gate (`gen` outputs main vs candidate,
  `axon-run --same-host`).

The script sets `AXON_TEST_GPU_SERIAL=1` unless you set it (also `testutil::gpu`'s default since #82), so
GPU tests run one at a time on every host. A pod run with GPU tests concurrent (`AXON_TEST_GPU_SERIAL=0`) can
hit the ILLEGAL_ADDRESS fault, and a POISON-RERUN pass isn't clean.

Options (environment): `TIER_SHARD=N` splits each suite with `axon-run --shard N`, `TIER_DETACH=1` submits
and returns, `TIER_RUN_OPTS="--big-models --on pods"` adds `axon-run` options, `TIER_BASE` is the ref the
kernel-change check diffs against (default `origin/main`).

The policy: fast for every intermediate round. Full once, on the commit that gets merged. Also full whenever
a round changes a kernel's numerics, a cache layout, a head-size or GQA path, or anything the bit-identity
contracts cover on more than one model, because the skipped bigger-model tests are the only ones that see
those shapes.
