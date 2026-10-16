# Research behind the roadmap

Read-only research notes written on 2026-10-16, before round 10. Each one maps papers and competing engines
(llama.cpp, vLLM, SGLang and others; read at source level where the report cites files) onto Axon's code and
its bit-identity contract. Each ends in ranked proposals, each with a change, an estimated gain, a
bit-identity risk, a size, and an experiment that would settle it. Labels inside: **MEASURED**, **ESTIMATED**,
**CODE** (file:line on main `e998d6f` unless stated) and **CLAIMED** (a paper's own number, not reproduced).

They were copied in unchanged, except that paths to the authors' scratch directories were replaced with a
plain description. Citations (papers, arXiv IDs, file:line references) are kept. References to `PROGRESS.md`,
`r4-*.md` … `r9-pick.md` and builders' `NOTES.md` point to the foreman's workspace, which is not in the
repo. [../history.md](../history.md) and [../experiments-rejected.md](../experiments-rejected.md) carry the
numbers from those files that matter.

| report | question | first proposal | led to |
|---|---|---|---|
| [r10-scope.md](r10-scope.md) | scope for the four product themes: bigger models, longer sessions, multi-user, concurrent requests. Includes the KV-bytes-per-token table per model | ranked list of ten items | roadmap "Scale" section (#44) |
| [research-attention.md](research-attention.md) | attention kernels on consumer Blackwell (sm_120): long-context decode and prefill | fold the decode partials while `attn_blk` runs ("chase fold"), order-preserving | follow-ups to lcdec (lcdec and lcpre themselves came from `r9-pick.md`, earlier the same day) |
| [research-bigmodels.md](research-bigmodels.md) | dense 14B–70B and MoE on a 12 GB card: hybrid, streaming, experts on the CPU | MoE with per-layer expert placement; speculation on the 32B split ranked first to run | r10 track `spec32` |
| [research-determinism.md](research-determinism.md) | batch-invariant inference, and the bit-identity contract as a product feature | identity tuple, `system_fingerprint`, a published determinism spec | r10 track `fingerprint` |
| [research-kvmem.md](research-kvmem.md) | KV-cache memory for many concurrent sessions (paged/radix KV, KV quantization) | host-RAM KV store on eviction plus save/restore | r10 track `kvsave`; "drop the N+1 cache" |
| [research-longctx.md](research-longctx.md) | context past the trained length, long chats, session persistence (YaRN, sinks, KV save) | bit-exact KV save/restore | r10 tracks `kvsave`, `yarn` |
| [research-sched.md](research-sched.md) | batching, scheduling, fairness and batched speculation for concurrent requests | prefill chunk and decode rows in one pass (Sarathi-style piggybacking) | deferred until #45 (chunked prefill) lands |

Earlier research, not copied here (in the foreman's workspace):

- `research-papers.md` (night 1): the quantized-GEMM and speculation literature. It led to the r8 interleave in #2.
- `research-airllm.md`: nothing for speed; streaming ideas for models bigger than RAM. It shaped G4 (#13, #16).
- `research-r8-bitident.md`: whether r8 could be made bit-identical. It became PR #3, closed.
- `research-verify-cost.md`: why small verify batches cost 1.5–3× a forward pass.

The round specs (`r1-gemm2.md` … `r9-pick.md`, `g4-offload.md`, `r7-hopper.md`, `r7-multigpu.md`) are design
documents for single rounds and are also not copied. Two of them were turned into repo docs earlier:
[../design/hopper.md](../design/hopper.md) and [../design/multi-gpu.md](../design/multi-gpu.md).
