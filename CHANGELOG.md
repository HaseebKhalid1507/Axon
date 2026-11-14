# Changelog

All notable changes to Axon. Versions follow [SemVer](https://semver.org/); before 1.0 the library API and the CLI
can change in any minor release. Each entry names its pull requests (#N on GitHub); the engineering log (`docs/log/`) has the
measurements behind them.

## Unreleased (after 0.1.0's draft)

### Chat prompts follow each GGUF's own template, token for token (#246, closes #238, #239, #240, #241, #249)
- **The template picks the variant.** The vocabulary still picks the format. Fixed substrings of
  `tokenizer.chat_template` (no Jinja) choose within it, so `axon chat` and serve frame these files differently now:
  - **Llama 3.0** (#241): no "Cutting Knowledge Date" lines, and a system block only when a system message is given.
    Llama 3.1, 3.2 and 3.3 are unchanged.
  - **DeepSeek R1** (R1-0528, the distills) and **V3** (#238): a reply opens with `<｜Assistant｜>` alone, so R1
    reasons. Before, they got V3.1's `</think>` and reasoning was switched off. BOS is written whenever the template
    writes it.
  - **Earlier DeepSeek replies** drop their reasoning as the template does, in serve and in `axon chat` alike
    (`axon chat` used to keep all of it). R1's (R1-0528, the distills): the text after the last `</think>`, newlines
    kept. V3.1's: what follows the first `</think>`. Qwen3's rule is unchanged: after the last, newlines dropped.
  - **DeepSeek multi-turn** (#240): an earlier reply is `<｜Assistant｜>{text}<｜end▁of▁sentence｜>`. Before, the
    marker was missing and the reply was glued onto the user turn.
- **The default system message is the template's** (#239): Qwen2's is "You are a helpful assistant.", and Qwen1.5, the
  R1 distills and Qwen3 have none. Qwen2.5's "You are Qwen, ..." now comes only from its template, or for a `qwen2`
  file without one. Any ChatML template with a fixed default gets that default whatever its architecture
  (Qwen2.5-Math's, for example). A fine-tune whose template has no default loses Qwen2.5's, except on a tool
  request, which still gets it (#256).
- **Tools need a template that renders them.** A request with tools, or a tool turn, for Qwen2, Qwen1.5 or Llama 3.0
  is now a 400, as ollama refuses it. These used to render in Qwen2.5's or Llama 3.1's format.
- **DeepSeek-R1's Qwen2.5 distills load** (#249): pre-tokenizer `deepseek-r1-qwen` is Qwen2's, as llama.cpp maps it.
- **A broken `bos_token_id` or `eos_token_id`** (past the vocabulary) never reaches a prompt. A bad BOS is dropped,
  as the tokenizer drops it. A bad EOS on a format that ends turns with EOS (DeepSeek, Mistral) leaves no chat
  framing, instead of a panic in the model: the chat endpoint gives a 400, `axon chat` an error and serve's banner a
  note, each naming the id ("eos_token_id N is past the vocabulary").
- **Pinned by a fixture**, `tests/fixtures/chat_templates.json`, built by `scripts/chat_template_fixtures.py` from
  pinned Hugging Face revisions: 16 families and 148 conversations, Axon's ids equal to `apply_chat_template`'s.
  The conversations also run through `axon chat` turn by turn. Message text that spells special tokens stays text
  in every family.
- **Saved sessions:** a resumed `axon chat` session keeps its old tokens, and the new turns use the new framing. A
  conversation that changed framing (Llama 3.0, R1, Qwen2 or Qwen1.5 without a system message) mixes the two.
  Serve's prompt cache is keyed by tokens, so changed prompts only miss it.

### Tests: the FP8 gate off the GPU test lock; GPU fast vs GPU exact in the suites (#232)
- **`gpu_fp8_fast_vs_exact_sanity` replaces `gpu_fp8_graded_vs_fp8_ref` in every suite.** It runs the FP8 0.5B
  checkpoint's fast mode against its exact mode on the same GPU, every arm at >= 1000 positions, with no
  statistics: finite logits, max |dlogit| < 12, loose deterministic bars (top-1 >= 80 % an arm, KL mean <= 3× the
  floor's, verify rows >= 85 %), a fast kernel in every arm, and the router's FP8 table. It holds the GPU test
  lock for about a minute (55–174 s on packed pods). The old test held it through its CPU reference, over 2700 s
  on packed pods, and 138 GPU tests of one suite failed on the lock timeout behind it.
- **The run of record, `gpu_fp8_graded_vs_fp8_ref_1000`, is `#[ignore]`d.** It grades against the CPU's fp8-ref with
  the statistical bars, unchanged. It runs its CPU nets before it takes the lock. It runs by hand on PRs that touch
  the FP8 kernels, their routing or the FP8 load path, on sm_120 and on sm_89 (two different FP8 GEMM paths); not
  yet nightly (docs/testing.md, the PR template).
  - Until #245 lands, a failure means look, not block: a bootstrap suggests its bars flag about 1 in 3 harmless
    numerics changes.
  - Its CPU reference is slow because the CPU's float matmul is a serial scalar dot (#244).
- `graded_in_two_passes_is_graded_at_once`: the two-pass grading equals the interleaved one, field for field
  (docs/log/2026-11-13-fp8-gate-lock.md).

### Precompiled GPU kernels: `--device cuda` with only the NVIDIA driver (#227)
- **A release binary carries its CUDA kernels precompiled** (the kernel pack): cubins for sm_75, sm_80, sm_86,
  sm_89, sm_90, sm_90a and sm_120, plus compute_120 PTX for newer GPUs. So `--device cuda` needs no CUDA toolkit
  and no NVRTC, only a driver for CUDA 13 (R580+). Each entry is NVRTC 13.4's output for the run-time compile's
  source, options and target, so the same code: the md5 gate matches main's NVRTC output.
- **Not covered:** data-center Blackwell (sm_100/sm_103, below sm_120), Pascal, Volta and drivers before R580 still
  need NVRTC; with none, `--device cuda` stops with one line naming the driver and the way out.
- **`axon kernels list [PACK]`** says what the binary (or a pack file) carries and where this GPU's kernels come
  from; it exits non-zero when the pack lacks some of the binary's kernel modules. **`axon kernels build OUT`**
  compiles a pack with NVRTC (no GPU needed); OUT must name a file, and the build's temp files go in a new
  directory beside it, the only thing it deletes.
- **New variables:** `AXON_KERNEL_PACK=0` (the kill switch: NVRTC only, as a source build) or `only` (never
  NVRTC), and `AXON_KERNEL_PACK_FILE=PATH` (a pack file instead of the built-in one; an unreadable or invalid one
  is a warning, then NVRTC).
- Source builds are unchanged: no pack, NVRTC as before.

### `axon serve --cache-ram auto` capped at 4 GiB, off under a binding cgroup; `--cache-ram off` (#226)
- **`auto` is now the smaller of 4 GiB and a quarter of the RAM available after the model.** Before, it was half
  that RAM, up to 8 GiB, page-locked from start.
- **`auto` is off under a binding memory cgroup** (a container or systemd scope whose limit is tighter than the host's
  free RAM), because pinned memory counts against the limit. The banner says why. An explicit `--cache-ram SIZE`
  still turns the tier on.
- **`--cache-ram off`** is the same as `0`.
- **The default is still off.** The 11-12 decision made `auto` the default, but its A/B failed the 2 % rule on an
  RTX 4090 (C=16 tok/s 0.977×, C=8 ITL p99 1.103×), and the first request after each start waited 3.2–4.0 s for the
  pinning. The flip waits for #233 and #234 (the lead's ruling, 11-13).

### Shared prefix KV pages C3.1: duplicate pages swapped, refed prefills claim again (#216, closes #212, #213, #214)
- Still opt-in (`AXON_SHARE_KV=1`). Unset, every changed path takes main's decisions.
- **#213: a duplicate page is swapped for the index's.** Two prompts prefilling one prefix side by side each wrote
  their own copy of its blocks, and the one behind stopped indexing its later pages. It now swaps its copy for the
  index's page (`PageMap::replace`, the copy freed) and indexes the rest under it; debug builds compare the bytes
  first.
- **#212: a preempted or unfed prefill claims the shared prefix again** when it is fed next (`refeed_claim`, as
  admission does), instead of prefilling from token 0. Those pages count in `axon_prompt_shared_tokens_total`
  too, so with preemption it is no longer prompt tokens only (#219).
- **#214:** `busy_pages` counts with generation stamps (no vector of every page per call); admission walks the index
  with the prompt's block keys computed once; `reclaim` passes over an idle slot whose pages are all shared (its
  release frees nothing) while another would give a page up; `release_rows` returns an error naming the page
  instead of asserting.
- Same bits: gen and serve md5 vs main, now with a section where requests are preempted
  (docs/log/2026-11-06-shared-prefix-c3-1.md).

### RAM pre-load checks, round 2: the loader's conversions checked first, a split needs its larger step (#208, #209)
- **#208: a safetensors checkpoint's converted tensors are checked before they're made.** The loader converts BF16
  tensors to F16 or F32 in anonymous memory (a BF16 model's matrices; an FP8 checkpoint's embedding, head and
  vectors: 519 MiB on the 0.5B) before any RAM check ran, so under a memory limit the FP8 0.5B was OOM-killed at 765
  MiB and below. The conversion's need (the converted tensors, the conversion threads' buffers) is now checked
  first, on every device, worded as the CPU model's refusal (#185). Near the edge it waits for the CUDA context,
  which the CLI makes beside the load, and reads the RAM again. Its tolerance is the memory cgroup's charge batches
  only (256 KiB a CPU): a conversion gives nothing back, and its estimate is exact in anonymous memory. The cgroup
  also charges about 1.2 MiB of kernel memory for it (likely the arena's page tables), which the estimate doesn't
  count, so under a cap a few MiB above the refusals it's still killed converting (on bella about 742–745 MiB for
  the FP8 0.5B; refused through 735). The CUDA kernels' module (about 25 MiB) is still loaded after this check and
  before the split's (#217, which also takes the page tables).
- **#209: a split needs the larger of its two steps, not their sum.** The GPU half's upload staging (#184) is freed
  before the CPU half is built; the need is now the larger, each with its own residents (the GPU half's credit
  offsets the CPU half only). The refusal says so, and its `--ctx` is the CPU half's. A trade: within the split
  check's tolerance (about 32 MiB) above the need, a run the sum refused cleanly can now be OOM-killed instead (two
  caps on a 3B GGUF split, one on the 7B FP8's under a real cap), against 17 caps where the 7B FP8 split now runs.
- **An FP8 checkpoint's CPU rows are counted whole.** The estimate credited every matrix's source as memory the load
  hands back; an FP8 checkpoint's are read from its file, so its CPU model's and a split's CPU half's need was the
  rows' scales alone (7B at 20 of 28 GPU layers: 1.74 GiB of rows uncounted).
- An FP8 split in fast mode (sm_89 and up) is refused with what works (`--numerics exact`, or every layer on the GPU),
  not "fewer --gpu-layers"; the CPU model's refusal of an FP8 checkpoint in fast mode names `--gpu-layers N and
  --numerics exact`. `scripts/fp8cap.sh` runs the checks under real memory limits (systemd scopes).

### Shared prefix KV pages: the engine (#207, closes #187; follows #164)
- **Opt-in: `AXON_SHARE_KV=1`.** Off by default, so a server without it runs main's admission, eviction and spills,
  decision for decision. #187 ships sharing off by default if any timing clause fails, and two did on the RTX 4090:
  the fan-out siblings' TTFT over the leader's (0.26× main idle, 0.49× busy, against ≤ 0.25×) and s4's new-session
  TTFT (1.00×, against ≤ 0.6×). What it does when on: the 7 siblings after the leader each prefill 54 of the
  6,006 shared tokens (main: two of them all 6,006, the next ~3k and ~1.5k), and their TTFT p50 falls from 2.43 s to 1.05 s idle; the
  trace replay re-prefills 65 % fewer tokens at speed 4 (1,665 → 577); decode and ITL p99 within 0.5 %
  (docs/log/2026-11-06-shared-prefix-pages-c3.md).
- **Requests share the KV pages of a common prefix** (the GPU's paged pool, `serve --parallel` > 1, with
  `AXON_SHARE_KV=1`). Admission
  walks the prefix index over the prompt's whole pages and maps the pages it has, another slot's or warm ones kept
  from a finished request, instead of prefilling them again (`claim_prefix`, design §3.4; phase 1: a prompt that
  continues mid-way into a shared page is fed again from that page's start, no copy-on-write). Pages are indexed and
  pinned as they fill (fed, decoded, restored, or written by the request running alone). Room comes from the
  index's least recently used warm leaves first (`pop_lru_leaf`), then idle slots; with `--cache-ram` an evicted
  warm page is copied to the RAM tier first (`spill_page` → `Tier::drop_resident` → `unpin_page`), and a slot that
  gives up indexed pages copies none of them then: they stay on the GPU until evicted.
- Unset (or anything but `1`), admission, eviction and the tier's spills are main's, decision for decision.
- `/slots`: `pages_warm`, `pages_shared`. `/metrics`: `axon_kv_pages_warm`, `axon_kv_pages_shared`,
  `axon_prompt_shared_tokens_total`, `axon_pages_evicted_total`. `AXON_KV_PAGES=N`: an N-page pool (tests, gates).
- Debug builds check after every step that no page leaks (each page in one state, `refs` == its occurrences in slot
  lists, the index's holds == `refs`, every indexed page pinned), and the immutability rule (E3) on the request
  running alone (`lend_solo`'s range), as #164 did on the batched writes.
- Same bits: gen and serve md5 vs main, with `AXON_PAGE_SHUFFLE`, the kill switch, `--cache-ram` and an undersized
  pool (docs/log/2026-11-06-shared-prefix-pages-c3.md).

### RAM pre-load checks: FP8 without the repack, a split checked before its upload, refusals name the way out (#183, #184, #185)
- **#183: an FP8 checkpoint with `AXON_REPACK=0` is counted.** The dense and split RAM estimates stopped before the
  arena when the repack was off, but FP8's matrices take arena rows either way: the FP8 0.5B's ~352 MiB arena went
  uncounted by the CPU model's check (#142) and the split's (#180). The estimate now counts what the build allocates.
- **#184: a split's RAM check runs before the GPU half's upload.** It ran after the upload, so a memory limit the
  upload itself didn't fit in was an OOM kill, not a refusal. It now runs once the CUDA context and the module are
  made (so the RAM figure is what they leave), adds the upload's own host memory (its page-locked staging ring: 16
  MiB, 64 MiB for a GPU half of 1 GiB or more, none under 64 MiB, or `AXON_UPLOAD_RING`'s size; or, if larger, the
  biggest matrix it converts to Q8_0 on the host, as an FP8 checkpoint's split does: the 0.5B's head is 138 MiB, a
  7B's 552 MiB), and, when the model's bytes are in RAM (a copied file, a checkpoint's converted tensors), credits
  the GPU half's share, which goes back as it's uploaded. The refusal names both terms. Memory taken before the
  check is still an OOM kill under a tight cap: the CUDA context (about 0.2 GiB, inferred from #180's caps; a floor
  before the device is opened stays open in #184) and the safetensors loader's F16 copies (about 520 MiB on the FP8
  0.5B, about 2 GiB on a 7B FP8; #208).
- **#185: the refusal names its ways out.** After the bound: the longest `--ctx` (in steps of 256) whose KV cache fits
  beside the weights, when the weights fit alone (always under the `--ctx` refused); then a GPU for the CPU model
  (whole, or in part with `--gpu-layers`, as far as the model can take it: not a BF16/F16 safetensors checkpoint,
  which runs on the CPU only, and no `--gpu-layers` for gemma3, which doesn't split), more `--gpu-layers` for a
  split chosen by hand, and for a split picked to fit the GPU's memory, that RAM is now the limit and a smaller
  `--ctx` the way out.

### KV RAM tier: exclusive, and an idle server's backlog in pieces (#201, follows #166; #170)
- **The tier keeps what the GPU doesn't.** Copies of pages a GPU slot still holds are written only into the tier's
  free room and are evicted first (newest first); RAM's own pages, those whose slot gave its rows up, go least
  recently used. `--cache-ram`'s useful capacity is now about its size, where #166's inclusive tier held the GPU
  pool's pages too (useful ≈ `--cache-ram` − the pool's KV bytes). Its gate's five clauses pass at the PR's code
  (9829783, design note §6 and §14), all three timing cells read on an RTX 4090, not bella (the lead's reroute): a 25k
  session evicted from the GPU resumes in 0.141 s against main's 2.713 s (1105-211420-0618). On bella that resume
  likely misses the clause's 0.3 s: two TAINTED reads, medians 0.513 s (1105-211421-8e35) and 0.358 s
  (1106-051210-8062), against main's 8.83 and 8.58 s. The lead waived bella's resume cell as a gate of this PR. Still
  opt-in, default off: the `auto` default is a later change. It waits for the RAM-pinning question (note §7) and a
  clean bella s4 replay at its own head; for bella's resume it works from the two tainted reads (note §14).
- **#170: no unbounded flush.** An idle server writes its backlog in 64 MiB pieces and takes a new request between
  two, instead of one flush of every slot's backlog after a request that ran alone.
- A conversation's next turn no longer copies the pages its slot keeps before its first token. A spill stores what
  the slot gives up only under a stored parent, and drops it otherwise, as with no tier. Before, a turn that gave up
  its reply's tail page copied its whole unstored chain first, ≈ 73 ms per 16k tokens on an RTX 4090
  (docs/log/2026-11-05-kvtier-exclusive.md).
- `/metrics`: `axon_kv_ram_gpu_pages`, `axon_kv_ram_spilled_pages_total`. `AXON_SERVE_TRACE`'s step lines carry the
  tier's copies (`spilled`/`spill_ms`, `restored`/`restore_ms`, `wb`/`wb_ms`).
- Default flags reach none of it: md5 vs main at default flags is unchanged (gen and serve), and the tier's own
  replies are main's bits.

### A GPU session's identity names the split-attention count and the code target (#203)
- **A `chat --session` file saved on the GPU now records the two parts of the fingerprint's `gpu` field that change
  the cache's bits on one architecture**, where the fingerprint names them: the split attention's count, which the SM
  count sets (`sm_89, split attention 40`), and code other than the device's plain cubin (PTX the driver compiles,
  `[code compute_90 PTX, driver 12.8]`; an older cubin of its major; sm_90a, Hopper's default). Before, the identity
  had the architecture only, so a Gemma 3 4B session saved on an RTX 4050 Laptop (20 SMs, 40 splits) loaded as bytes on
  a 4090 (128 SMs, 256), both sm_89, though past 32 × 40 positions the two reduce the attention in another order.
- **One-time cost:** GPU sessions of Gemma 3 (the split attention on every GPU), of any model below sm_80, and of a
  build that doesn't run the device's plain cubin restore once as tokens (their conversation prefilled again), then as
  bytes. Dense models on sm_80 and later with the device's plain cubin keep their identity, so their sessions still
  restore as bytes. gpt-oss takes the split attention too, but nothing changes for it: it runs as a MoE, which has no
  GPU session image yet (its sessions are always prefilled again).
- **Every H100 / H200 GPU session saved before this pays that cost once, by default.** sm_90a is the default code on
  sm_90 (NVRTC 12 or later; `AXON_HOPPER=0` builds the plain sm_90 cubin), so those identities gain
  `gpu: "sm_90 [code sm_90a]"`. Their bytes weren't stale: `numerics` already recorded the choice (`hopper true`), and
  the new key only tells apart the builds where `hopper true` didn't give sm_90a (an NVRTC older than 12). The
  re-prefill is the price of the session identity using the fingerprint's own definition rather than a second one
  that could drift from it.
- **A PTX build re-prefills once per driver CUDA-version update.** Its code names the driver's CUDA version
  (`compute_90 PTX, driver 12.8`), because the driver compiles that PTX. So after a driver update that changes that
  version, its sessions restore once as tokens. A cubin build's sessions still load as bytes across driver updates:
  the identity's `info.driver` (the same version) isn't compared.

### Sessions hash the GPU module the net runs; one session counter; `axon info` predicts FP8 fingerprints (#196)
- **A saved session's GPU identity now hashes the source and options of the kernel module the net actually runs**
  (`cuda::numerics_tag`), not the plain kernels' whatever ran. A change to fp8.cu, oss.cu, or the extra types'
  options can no longer restore stale KV bytes if the version bump is forgotten. GPU choices made in Rust (launch
  parameters such as `max_split`, which kernel a shape takes) aren't in the hash: they still need the
  `NUMERICS_VERSION_EXACT` bump, which a session's `host N` carries. A dense GGUF without an extra type
  runs the plain module, whose tag is unchanged in both modes, so its saved sessions still restore as bytes.
  **One-time cost:** sessions saved on the GPU by a gpt-oss model, a model with an extra quant type (any of Q2_K,
  Q3_K, Q4_0, Q4_1, Q5_1, Q5_K, IQ4_NL: e.g. a Q5_K_M or Q3_K_M file) or an FP8 checkpoint in fast mode on sm ≥ 89
  restore once as tokens (their conversation prefilled again), then as bytes.
- **One version counter for sessions:** `kvsave::NUMERICS` (the CPU's and the host's part of a session's identity)
  is now `NUMERICS_VERSION_EXACT` itself, so the two can't drift. Both are 1: no session changes.
- **`axon info` predicts an FP8 checkpoint's fingerprints right.** Its predictions lacked the `weights`, `act` and
  `head` fields (`fp_6fec…` predicted vs `gen`'s `fp_28bb…` on the CPU). It now computes them as a load would, and
  prints fast mode's full tuple where fast runs its own kernels (fp8.cu). A test checks that every predicted
  placement matches `gen`'s, in both modes.
- An FP8 split's streaming note names the CPU's FP8 types (`F8S`, ...) instead of `??`.
- No output bit changes; no reply fingerprint changes (only `axon info`'s FP8 predictions, which were wrong).

### Identity: an FP8 checkpoint's `act` is the net's, and a split names its CPU layers' emulation (#200, part of #196)
- The reply fingerprint's `act` for an FP8 checkpoint on the CPU now comes from the net (the type its matrices were
  built as), as the KV identity's does since #199, not from `AXON_FP8_CPU_ACT` read again when the fingerprint is
  made. The CLI's fingerprints don't change: there the two always agreed.
- A CPU/GPU split (`--gpu-layers`) of an FP8 checkpoint whose CPU layers run `AXON_FP8_CPU_ACT`'s emulation now says
  so: `act=q8 (cpu layers e4m3-tok-emu)` (or `e4m3-128-emu`). It was `act=q8`, as with f32 activations, which it
  still is. A split's sessions are unaffected: it has no KV identity, so they always restore as tokens.
- No other fingerprint and no output bit changes.

### Sessions: a CPU session of an FP8 checkpoint records `AXON_FP8_CPU_ACT`'s emulation (#199, part of #196)
- **Fix: a `--device cpu` `chat --session` of an FP8 checkpoint no longer restores as KV bytes across
  `AXON_FP8_CPU_ACT`.** Its emulations (`e4m3-tok`, `e4m3-128`, a diagnostic: the activations rounded through e4m3)
  make other cache rows than the default f32 activations, but the CPU's KV identity was the same. An emulating net's
  identity now has `act: e4m3-tok-emu` (or `e4m3-128-emu`, the reply fingerprint's names), and a session saved one
  way is prefilled again the other way.
- Nothing else changes: f32 activations (the default), dense models and every GPU net keep their identities and
  their saved sessions. No output bit changes (the md5 set is unchanged).
- Tests: `fp8_session_across_cpu_act` restores sessions both ways across the variable (a child process sets it, as
  #198's GPU test does), and `tiny_fp8_kv_ident_names_cpu_act` checks the field on the tiny checkpoint in CI. Both
  session tests now compare the rows the session file holds with the restore's, for the same tokens.

### Sessions: an FP8 checkpoint's fast-mode session records whether fp8.cu made it (#198, part of #196)
- **Fix: a fast-mode `chat --session` of an FP8 checkpoint no longer restores as KV bytes across
  `AXON_FP8_NATIVE`.** With fp8.cu (fast mode, sm_89+) and with `AXON_FP8_NATIVE=0` (the Q8_0 form) the cache rows
  differ, but the KV identity was the same, so a session saved one way loaded its bytes the other way and the
  conversation went on from rows a fresh prefill wouldn't make. The identity of a net running fp8.cu now has
  `weights: fp8-native`, and a session saved one way is prefilled again the other way.
- **Fast-mode FP8 sessions saved with fp8.cu before this restore once as tokens** (prefilled again). Nothing else
  changes: dense GGUF models, either mode, and FP8 checkpoints in exact mode or under `AXON_FP8_NATIVE=0` keep their
  identities and their saved sessions. No output bit changes (the md5 set is unchanged).
- #196 stays open for the full fix: the KV identity should hash the source and options of the module the net runs.

### Tests: FP8 test skips show in a captured log; `AXON_FP8_NATIVE=0` fails the FP8 engine tests (#197)
- **A skipped FP8 test now says so in a normal test log, not only under `--nocapture`.** The FP8 checkpoint's skip
  (missing, with `AXON_SKIP_FP8_MODEL_TESTS` or `AXON_SKIP_MODEL_TESTS` set) and the FP8 tests' skip on a GPU below
  sm_89 go straight to stderr (`testutil::skip_note`) and name the test. libtest captures `eprintln!` in a passing
  test, so those skips read as a bare `ok` in a gate log, the blind spot #193 was about (shady's review of #195).
  Only a test that skips prints the line: the checkpoint rule's own unit tests print none, so a `skipping:` line in a
  suite's log means a test skipped (axon-run reads one near a log's end as a run whose test skipped itself).
- **`AXON_FP8_NATIVE=0` fails `gpu_fp8_graded_vs_fp8_ref`** (and the ignored `gpu_fp8_load_diag`) with a message
  naming it. It turns the engine's native FP8 off, so the test has nothing to grade. It used to skip every FP8 test
  quietly, as "no e4m3 tensor cores". The FP8 kernel tests no longer read it: they run on any sm_89+ GPU. README
  documents the variable.
- CONTRIBUTING's FP8 checkpoint download uses `curl -fL` and checks all 9 files' sha256 (`sha256sum -c`).

### Tests: a missing FP8 checkpoint fails the FP8 checkpoint tests (#193)
- **`gpu_fp8_exact_is_q8_and_bit_identical` and `gpu_fp8_graded_vs_fp8_ref` now fail when the FP8 0.5B checkpoint
  is missing** (`models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic`, or `AXON_TEST_FP8_MODEL`), as the GGUF
  test models already do. They used to return early, which libtest reports as `ok`. Three 4090 gate runs showed the
  FP8 graded gate passing that way without running it (#193). `AXON_SKIP_MODEL_TESTS=1` (CI) or the new
  `AXON_SKIP_FP8_MODEL_TESTS=1` skips them instead. The download and its sha256 are in CONTRIBUTING.md. Only a host
  with a usable GPU reaches the check: the exact-mode test runs on any, the graded test on sm_89+.

### FP8 on sm_89: the fast-mode GEMM runs f16 mmas below sm_120 (#190, #192)
- **Correction to #177: below sm_120, `--numerics fast` no longer runs FP8 prefill on e4m3 tensor cores.** On sm_89
  the e4m3 `mma.sync` keeps only about 14 bits of each k32 step's sum, and the FP8 GEMM missed its f64 accuracy anchor
  by up to 173× on a 4090 (#190). For every compile target below sm_120, `gemm_f8` now converts the e4m3 codes to f16
  (exact: e4m3 is a subset of f16) and runs each k32 step as two f16 mmas with f32 accumulate. That is within 0.18× of
  the anchor's tolerance on the 4090. Cost: 1.30–1.55× the e4m3 GEMM time (4090, `gemm_f8` alone at the 0.5B and 7B
  shapes). Decode (`gemv_f8`, CUDA cores) and the activation quantizer are unchanged. Exact mode (the default, Q8_0)
  and every GGUF model never run this code
  ([docs/log/2026-11-04-fp8-sm89.md](docs/log/2026-11-04-fp8-sm89.md)).
- **End to end, MEASURED on the FP8 0.5B only:** on a 4090 in `--numerics fast`, prefill is 17–22 % slower (586- and
  2260-token prompts) and decode is unchanged. The 7B is not measured. At its shapes the GEMM alone is 1.32–1.42×,
  24–30 % less throughput, and more of its prefill is GEMM than the 0.5B's, so its prefill cost is likely nearer
  that (ESTIMATED).
- **sm_120 is unchanged.** Its cubin keeps the e4m3 mma: the code is the same, and the md5 gate against main is
  identical, FP8 `--numerics fast` included.
- **H100 (sm_90) and sm_100 change too, unmeasured.** They are below sm_120, so FP8 fast-mode prefill takes the f16
  path there as well, with new bits. No anchor, graded or speed run exists on either until the #176 probe.
- **Every fast-mode fingerprint changes once.** `NUMERICS_VERSION_FAST` 2 → 3, so `numerics=fast/1.2` becomes
  `fast/1.3` in every fast-mode identity, on every GPU (sm_120 too) and for GGUF models; their bits don't change.
  Exact-mode fingerprints don't change. A `chat --session` saved in fast mode carries `fast/1.2`, so it restores once
  as tokens (prefilled again) instead of as KV bytes the old e4m3 path made. On sm_120 that happens although its bits
  are the same: the
  [lead ruling](https://github.com/HaseebKhalid1507/Axon/pull/191#issuecomment-6651890843)
  is the bump with no waiver, and accepts that.
- **The FP8 graded gate's wide-flip count is set relative to its CPU floor** (#192, test-only). On the 0.5B's
  1014-position decode arm, the floor alone has 10 top-1 flips at margin ≥ 0.5, which is all of the old
  `max(2, n/100)` = 10. A 4090 had 12. The bound is now the floor's count plus 2 binomial SE (16 there), for the FP8
  gate only; the order-only gates are unchanged. The difference starts in shared code: without any FP8 kernel, a 4090
  and bella differ at 931 of 933 decode positions.

### Shared-prefix KV pages: groundwork, no behaviour change (#164)
- **The paged KV pool counts who holds each page, and a prefix index can pin pages** (C1 + C2 of the shared-prefix
  design, [docs/design/shared-prefix-pages.md](docs/design/shared-prefix-pages.md)). Nothing maps, pins or indexes
  pages yet, so every page has one holder and the pool allocates, releases and copies exactly as before: the same
  bits (the md5 and serve identity gates against main). The rule sharing rests on, "a page shared or indexed is never
  written again", is checked in debug builds on the pool's batched writes (feeds, steps, `adopt`) and the RAM
  tier's page copies (#166). A single request's writes through a slot view aren't checked yet: that is C3's, with
  the engine work (claim, index, evict). In-flight dedupe (C4) is a later PR too.

### KV RAM tier, opt-in (#166)
- **`axon serve --cache-ram BYTES|auto`: keep conversations' KV pages in host RAM, off by default.** When a slot goes to
  another conversation, its KV pages stay in host memory, and its next turn copies the longest stored prefix back
  instead of prefilling it again: the same bits (the md5 gate against main, with the tier off and on). Default 0:
  without the flag the server is main's, byte for byte (`/metrics` included).
- **The whole size is pinned from server start.** On the GPU, `--cache-ram` is page-locked (unswappable) host RAM
  from start, used or not; `auto` is half the RAM available after the model, at most 8 GiB.
- **The tier is inclusive, so it helps only when sized well above the GPU KV pool.** Pages are copied as they fill,
  so the tier also holds what the GPU still has: useful capacity ≈ `--cache-ram` − the GPU pool's KV bytes. On
  bella's 7B f16 (pool ≈ 5.4–6.0 GiB) 6G buys almost nothing; about 20G is where it works (the 4090 replay
  1027-155046-1ed0: TTFT-resume p50 13.5 → 0.11 s at 20G).
- **It failed its pre-registered bella cell.** 1031-063509-f763 (7B f16, `--cache-ram 6G`, a 25k session evicted by
  real pressure): 6,592 of 26,496 tokens restored, TTFT-resume 7.05 s vs main 8.68 s, short of the 0.3 s bar. Hence
  opt-in (rule B's else branch). Next: an exclusive tier (C3), then the bella cell again
  ([docs/log/2026-10-27-kv-ram-tier.md](docs/log/2026-10-27-kv-ram-tier.md)).
- **A tier error no longer exits the server when it hits a spill that makes room for a restore** (#181). It turns the
  tier off, as every other tier error does, and serving goes on as with `--cache-ram 0`. The off line now names what
  failed on every path: `axon serve: --cache-ram: spill|write-behind|restore: ...; the KV RAM tier is off from here`.
  Tests now cover the off path on the CPU, a paged pool, a plain pool and `--parallel 1`
  ([docs/log/2026-11-04-kvtier-off-path.md](docs/log/2026-11-04-kvtier-off-path.md)).

### FP8 checkpoints (#177)
- **FP8 checkpoints load, as Q8_0 by default.** compressed-tensors per-channel e4m3 checkpoints with dynamic per-token
  activations (RedHatAI's `*-FP8-dynamic`) load from their directory. In exact mode (the default) the FP8 matrices
  become Q8_0 at load and run on the existing kernels under the exact contract (identity `weights=fp8-q8`). Accuracy:
  the 7B passes both bars (KL ≤ 0.01, top-1 ≥ 97 %) against the CPU FP8 reference (KL 1.2e-3, top-1 98.1 %); the 0.5B
  scores 96.7 % top-1 against HF F32, below the 97 % bar. On the CPU the checkpoint runs as its exact dequant
  (`weights=fp8-ref`, the accuracy reference). Block-scaled FP8 and other FP8 forms are refused by name.
- **Native FP8 is opt-in and fails its bars on sm_120.** `--numerics fast` on sm_89+ runs the weights natively (e4m3
  tensor cores, `weights=fp8-native`). On bella (RTX 5070 Ti Laptop) the 7B misses both pre-registered bars: tg128
  32 tok/s vs a bar of 47–55 (Q4_K_M: 85–101), top-1 95.9 % vs 97 %. It isn't measured on an H100. Follow-up: decode
  graphs for FP8 nets ([docs/log/2026-10-30-fp8-weights.md](docs/log/2026-10-30-fp8-weights.md)).
- **Every GGUF fast-mode fingerprint changes once; no output does.** `NUMERICS_VERSION_FAST` 1 → 2 (the first fast
  kernels), so `numerics=fast/1.1` becomes `fast/1.2` in every fast-mode identity, GGUF models included; their
  output bits are unchanged (the md5 gate against main, exact and fast). Exact-mode fingerprints don't change.
- **Fast-mode sessions saved before restore once as tokens.** A `chat --session` file saved in fast mode carries
  `fast/1.1` in its identity. That no longer matches, so the session restores once as tokens (prefilled again)
  instead of as KV bytes, which is slower but never wrong (`src/kvsave.rs`). Exact-mode sessions are unaffected.

### Release note: every fingerprint changes once (numerics modes, #173)
- **Every `system_fingerprint` changes, once; no output does.** The identity tuple's first line was `numerics=1`; it is
  now `numerics=exact/1` (the numerics mode and its own version), so every fingerprint (`axon gen`'s `identity:`,
  `axon serve`'s responses, `GET /identity`, `axon info`) has a new value. The kernels, the sampler and every output
  bit are unchanged (the md5 gate against main: same greedy bytes). A client that caches or replays by
  `(system_fingerprint, prompt, params, seed)` will miss once and repopulate. `GET /identity`'s `numerics` is now a
  string (`"exact/1"`, was the number `1`). `chat --session` files from before still restore as bytes (exact mode's
  session identity didn't change). Decision: Haseeb, 2026-10-29 (ship it now rather than with the next numerics bump);
  [docs/design/determinism-modes.md](docs/design/determinism-modes.md) §7.
- **`--numerics exact|fast`** on `gen`, `logits`, `chat` and `serve` (default `exact`, today's contract), and
  `axon::Numerics` / `BackendOptions::numerics` in the library. `fast` may run kernels that change the float order,
  graded against exact instead of matched bit for bit, under its own identity (`numerics=fast/1.1` as #173 shipped it, `fast/1.2` since #177: exact's counter
  and fast's own, so a change to code both modes run moves both; its own NVRTC modules and session identity). Fast
  mode promises tolerance on every path, bits on none (batching, cache reuse, chunking, drafting). **No fast kernel
  exists yet**: `fast` runs the exact kernels and says so once; `Backend::fast_kernels()` tells an embedder which.
  `axon info` prints each placement's fingerprint in both modes.
- **`serve --datacenter`**: the datacenter bundle; today it only implies `--numerics fast` (an explicit `--numerics`
  wins).
- `scripts/numerics_dnll.py` and `axon logits` with `AXON_LOGITS_NLL=1` and `AXON_LOGITS_PREFIX=N` (the first N tokens
  in batched passes): teacher-forced ΔNLL between two binaries or modes, per batched prefix (`--prefix`, default
  0,24,160), with an SE clustered by window.

### Merged after the draft, not listed above (10-25 → 10-31)
- **Behaviour change (#167):** `serve`'s defaults are `--admit-order sjf --step-ms 80` (#131 added both, with fifo
  and 60 ms the default). #165's rule S on an H100 SXM, mix C=128 against fifo/60: aggregate +7.6 %, goodput
  +60.5 % (job 1027-084258-1ce4, [log](docs/log/2026-10-27-sjf80-default.md)). `--admit-order fifo --step-ms 60`
  (`AXON_ADMIT=fifo`) is the scheduling of before; with `AXON_STEP_COST=legacy` too, the scheduling before #131.
- **`--gpu-experts N|auto`** (`AXON_GPU_EXPERTS`, default 0): routed experts resident on the GPU, deepseek2 (#127)
  and qwen3moe (#146), same bits as the CPU experts. The bench of record (#144, one RTX 4090): Qwen3-30B-A3B all
  resident prefills ×1.31 llama.cpp at pp2100 and decodes ×0.87 (×0.95 after #154)
  ([docs/results.md](docs/results.md#new-models-and-types-since-the-boards)).
- Faster, same bits: MLA decode and prefill kernels (#145, #147), exact batched decode attention for the serve pool
  at nb ≥ 32 (#151), the MoE decode step (#154), Q4_K / Q5_K `down` on gemm2t from nb 32 (#178). Each one's
  numbers are in [docs/history.md](docs/history.md) (Phases 21, 22).
- `serve`, gpt-oss with tools: a reply with no harmony header streams as the answer (#136, #114). The CUDA deepseek2
  net refuses a session image, as the CPU does (#161, #159).
- Diagnostics: `AXON_SERVE_TRACE=FILE`, a per-step trace of `serve`'s pool (#125); `AXON_MOE_PROFILE` counts resident
  layers' steps (#153).

## [0.1.0] - unreleased (draft; not tagged)

The first version: everything merged to `main` from the start of the project (2026-10-10) through PR #113
(main `aecc49d`, 2026-10-20), 103 pull requests, plus the library split below. Release notes:
[docs/release-0.1.md](docs/release-0.1.md). The tag and the date are the maintainer's call.

### Library
- The crate is now a library (`axon`) plus the `axon` binary on top of it. `src/lib.rs` is the crate root and the
  public API: `load`, `Model`, `Backend`, `BackendCache`, `Device`, `KvType`, `Tokenizer`, `Utf8Stream`, the `Lm`
  trait, `generate`, `GenOpts` (`GenOpts::greedy`; sampling stays crate-internal in 0.1), `GenStats`, and `serve`.
  The command line moved from `src/main.rs` to `src/cli.rs` (a new file: git doesn't track it as a rename) and the
  decode loop and its tests to `src/generate.rs`; `examples/embed.rs` shows the API. No change for the CLI: same
  greedy bits, same output, same tests (14 renamed: `tests::*` → `cli::tests::*` or `generate::tests::*`).
- The KV type is an argument: `Backend::new` (its `BackendOptions::kv`) reads no `AXON_KV` (the CLI and `examples/embed.rs` parse it
  with `KvType::parse` and pass it), `KvType::F32` takes effect on the CPU (it used to be silently f16 through the
  library), and a type that conflicts with the process's earlier f16/f32 choice is an error.
- `Model` is read-only outside the crate: its fields and the methods that hand out or release weight bytes are
  crate-private (a safe caller could zero weights a live `Backend` borrows); accessors `arch`, `meta_u32`,
  `n_layers`, `embed_dim`, `n_heads`, `n_kv_heads`, `ctx_len`, `total_params`, `total_bytes`. No `memmap2` type in
  the API.
- `generate` refuses drafting (`max_draft > 0`) on a backend whose decode steps stream (`Lm::decode_streams`:
  `AXON_STREAM_DECODE=1` on a CPU/GPU split), where drafting wouldn't be lossless; the CLI and `serve` already
  refused the pair.
- `Backend` and `BackendCache` are opaque (#118 A): no variants to match and no fields, so the model types inside
  (CPU, GPU, split, pipeline, mixture-of-experts) and their geometry can't be reached or written from outside the
  crate, and a new kind is not a breaking change. `Lm` is sealed (implemented by the crate's models only).
  `Device`, `KvType` and `GenStats` are `#[non_exhaustive]` (a match outside the crate needs a `_` arm;
  `GenStats::default` stays).
- `Backend::new(&model, BackendOptions)`: device, context, GPU layers, KV type and fit are a `#[non_exhaustive]`
  options struct with `Default` and chainable setters (`BackendOptions::default().device(Device::Cuda).max_ctx(8192)`),
  not six positional arguments, so new options (#118 item 2) aren't breaking. `Lm::forward_cands` / `verify_cands`
  are `#[doc(hidden)]` until the sampler is exported (#118 item 8).
- A bad `AXON_SIMD` (unknown, or a kernel set the CPU lacks), `AXON_DECODE_ROWS` or `AXON_PREFETCH` is an error
  from `load` and `Backend::new` (the CLI: its usual message, exit 2) instead of a panic inside a later call (#118 B).
- Not hardened for embedders yet (stderr notes, env-only config, a CUDA leak per `Backend::new`, `serve` exits the
  process): #118.

### Models and formats
- Qwen2 / Qwen2.5 GGUFs on the CPU from the first commit; Qwen3 dense on the CPU and CUDA (#18); Llama 3.x
  (#22, which also carried the sm_89 capture fix) and its prompt framing in `serve` (#23); SentencePiece tokenizer
  and Mistral 7B's chat format (#32); Gemma 3 1B / 4B on the CPU and CUDA (#35).
- Mixture of experts: qwen3moe on the CPU (#63), then attention and router on the GPU with CPU experts (#67, #76);
  gpt-oss-20b with MXFP4, o200k and harmony on the CPU (#78), then on the GPU with CPU experts (#98);
  DeepSeek V2/V3 / Kimi K2 / Moonlight (MLA, DeepSeek routing) on the CPU (#79), then GPU MLA with CPU routed
  experts (#89); one router kernel and a `MoeKind` enum (#104).
- Quantization: Q5_K and Q4_0 (#22); Q2_K, Q3_K and IQ4_NL on the CPU and CUDA (#72); clear refusals for i-quants
  and repacked GGUFs (#83).
- Hugging Face safetensors checkpoints (BF16/F16/F32) on the CPU, run as the GGUF F16 of the same weights (#102).
- YaRN context extension, opt-in: `--rope-scaling yarn` (#49).

### CPU
- Interleaved quantized kernels, FMA attention, vectorized exp, f16 KV cache (#2); bench harness, pool under load,
  kernels, attention, decode, hygiene, prompt-lookup drafting (#1).
- Prefill: Q8_0/Q5_0 GEMM fix, fewer pool joins, batch 128 (#20); decode pool participation on hybrid CPUs (#32).
- Startup: map instead of copy when repacked, CUDA init at t = 0, read-ahead, faster upload (#21); pinned upload
  ring, faster tokenizer build and GGUF string parse (#41).
- **Behaviour change (#142):** a dense model whose repacked weights plus the KV cache of its first 8192 tokens (or
  `--ctx`, if shorter) don't fit in the RAM available (host `MemAvailable` or the tightest cgroup limit, #133) is
  refused at load with a message naming the limit (exit 2), instead of being OOM-killed. A longer `--ctx` that
  wouldn't fit whole prints a note. The MoE RAM refusal's figures are now GiB, as the dense one's, and a cgroup
  limit is printed in MiB.
- **Behaviour change (#143):** the CPU half of a `--gpu-layers` split gets the same refusal, for its layers' repacked
  weights and their KV floor, after the GPU half has loaded (`the CPU half of the split (k of n layers) needs
  about …`), instead of being OOM-killed. A split cache that would stream prefill through the GPU page-locks the CPU
  layers' whole KV cache at once; when the RAM available can't hold that, the cache doesn't stream (a note says so).

### CUDA
- GPU backend: foundations (#7), decode (#8), batched prefill and speculative verify (#10), an 8-bit KV cache
  `--kv q8` (#11), the canonical block order and tensor-core GEMMs (#12), shared-score batched attention and
  1024-token passes (#15), tensor-core flash attention (#19), r5 prefill epilogues (#24), narrow-row decode
  kernels (#30).
- Placement: CPU/GPU split `--gpu-layers` (#13), streaming the CPU layers' weights for prefill (#16), pipeline
  parallel over several GPUs `--gpus 0,1` (#34), older NVRTC fallback for old GPUs (#39).
- Hopper (sm_90a) wgmma + TMA prefill GEMMs (#33); long-context decode (#50) and prefill (#56) attention;
  batched-decode step cost and gemm2 partial tiles (#81), fused q|k|v and gate|up launches at nb ≤ 16 (#86);
  multi-query block attention (#84); prompt passes sized to the GEMM waves (#92, #97); a 128 × 64 Q4_K/Q5_K GEMM
  tile (#100).
- Paged KV: every cache kernel through a page table (#64, #68), then the pool allocates pages and `serve` plans,
  admits and preempts by pages (#71).

### Generation
- Sampling: temperature, top-k, top-p, min-p, repeat penalty, seeds, on the CPU and CUDA (#17).
- Speculative decoding with a draft model `--draft-model` (#27), and on the CPU/GPU split (#52); greedy output
  unchanged by drafting.
- Identity tuple and `system_fingerprint`: every output says which bits it is (#51).

### `axon chat`
- Terminal chat with the model's template (#4), line editing (#6), exact KV save/restore `--session` (#48, #54),
  forgetting down to half the context when full `--forget-to` (#75).

### `axon serve`
- OpenAI-compatible HTTP server (#22); tool calling (#26); continuous batching of decode rows (#38); the trained
  context by default (#42); chunked prefill `--prefill-chunk` (#45, #55, #57); one cache for the single-request
  path (#60); ops: `--api-key`, `/metrics`, `/slots`, `--max-prompt-tokens`, `--slot-idle-ms` (#61, #62);
  `/v1/responses` and the ollama API (#70); gpt-oss tool calling via harmony (#111).
- Fixes: metrics counted before the response is written (#93); a request leaves in-flight before its client has
  the response (#113).
- **Behaviour change (#141):** an explicit `--parallel N` that doesn't fit in GPU memory at start now exits 2 with
  the numbers (it used to start with fewer slots, one request at a time at worst, said only with logging on). A
  supervisor that restarts on exit will keep exiting until `--parallel`, `--ctx` or `--kv` changes; leave
  `--parallel` out for the automatic sizing, which still falls back (and now says so, also under `--quiet`).

### Hardening, tests and CI
- Fuzz-found fixes for GGUF, tokenizer, serve and CLI (#37); soak harness and model-zoo sweep (#83).
- Test suite: qemu-robust pool test (#14), attention sweep skipped under qemu (#5), PTX arch choice on A100 (#31),
  frozen natural-text fixture (#36), GPU tests serialized on small cards (#40, #43) and on every card (#82), GPU
  OOM test tolerant of other processes (#46), slot-idle test bounds (#87), one CPU reference per input and the
  fast/full tiers (#80), graded logit tests with a repacked reference (#105), Gemma 3 near-tie handling (#107).
- Dropped the model viewer and ratatui: 61 → 27 crates (#9).

### Docs
- README, BENCH.md, START-HERE, GPU benchmarks, methodology, engineering history, rejected experiments, results
  (#25, #28, #47, #88, #91, #103); roadmap (#29, #44, #59, #65, #66, #69, #73, #96); drafting and placement note
  (#58); ILLEGAL_ADDRESS investigation (#53); test speed (#85); contributor contract, code of conduct, security
  policy (#74, #77); determinism-modes ADR (#101).
