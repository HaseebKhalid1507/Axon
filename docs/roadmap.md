# Roadmap

Short and opinionated. "On main" means merged at `5f96039`. Everything else is planned or in flight.
Estimates come from the design notes and are labeled.

## The thinking behind the order

Single-stream decode on a 7B/8B model is near the hardware limit. On Parabellum (RTX 5070 Ti Laptop),
7B decode GEMVs run at 93% of the measured 648 GB/s ceiling, and the 7B prefill GEMMs are power-bound
at parity with llama.cpp's MMQ kernels (MEASURED). More kernel tuning won't move single-stream 7B much.
Two things will:

- **fewer full-weight passes per token:** speculation. The draft model is now on main.
- **more useful rows per weight pass:** batching many requests (S2), and later multiple GPUs.

The exception is the H100, where prefill is issue-bound at 7.6% of its int8 peak. That gap is real
and has a specific fix ([design/hopper.md](design/hopper.md)).

## Recently landed

- **Draft-model speculative decoding** (`--draft-model`, #27). Lossless, with K = 3 by default.
  Greedy tg128 medians were +37% on Qwen2.5-7B, +44% on Qwen3-8B and +63% on Llama-3.1-8B at K = 3
  (MEASURED in the go/no-go A/B, recorded in `src/draft.rs`). Sampled at temp 0.8 the gain was
  +0.8% on 7B (README).

## Priorities

| # | item | why | status |
|---|---|---|---|
| 1 | **S2: continuous batching in `serve`.** Slot KV cache, per-row positions and sampling, a FIFO scheduler; whole-prompt prefill as its own step. | The largest product gap: ollama and llama-server already serve parallel requests, while `serve` runs one at a time. ESTIMATED aggregate on 7B: ~650 tok/s at 8 clients (5.5× single-stream) and ~1000 at 16. Gate: a batched run is token- and KV-byte-identical to a solo run. | in flight |
| 2 | **Startup round 2.** Pinned staging for the weight upload, upload/prefill overlap, faster tokenizer and GGUF parsing. | Time to first token. ESTIMATED 7B warm short-prompt TTFT 624 → ~480 ms, from the measured 25.4 GB/s pinned upload. No change to output bits. | in flight |
| 3 | **Multi-GPU, pipeline parallel (stage 1).** | Models bigger than one card, and KV room for serving. Bit-identical to one GPU by construction. With S2 in place, stage 1b overlaps stages for ~2× aggregate (ESTIMATED). | in progress ([design](design/multi-gpu.md)) |
| 4 | **Hopper wgmma/TMA prefill.** | H100 prefill trails the 4090 and 5090. ESTIMATED 7B pp2100 12.5k → 22–32k tok/s. Gated on a ~1-hour pod experiment. | design; experiment pending ([design](design/hopper.md)) |
| 5 | **Gemma 3** (1B/4B/12B). | The most-requested missing family. Medium bit-identity risk: new attention variants (head dim 256, sliding window) must keep decode == batched. Comes after S2, since both change the attention kernels. | in flight |
| 6 | **Old GPUs (sm_61, Pascal).** | Fixes for Pascal cards, tested on Jaded's GTX 1050. Main has no tested sm_61 path; it refuses a GPU the installed NVRTC can't target. | in flight |
| 7 | **SentencePiece tokenizer** (Mistral 7B v0.3, Llama 2, TinyLlama) on the existing `llama` path. | Cheap coverage: ESTIMATED 6–10 engineer-hours and no kernel risk. Main accepts only GPT-2-style BPE (`tokenizer.rs`). | planned |
| 8 | **CPU re-board on main, then what it shows.** | The last full CPU scoreboard predates several merged CPU changes. Remaining items (pool spin, P/E-core tail) are ESTIMATED at a few percent. | planned |
| 9 | **GPU decode: long-context attention; locked-clock trial.** | ESTIMATED +3% decode at 4k context. A locked SM clock may recover part of the 9% laptop power-cap cost (MEASURED cost; the fix is untested). | planned |

Also open:
- **Tool calling in `serve`.** Main refuses `tools` with a 400. A PR is pending.
- **Qwen3 0.6B Q8_0 long prefill on sm_89.** The one cell Axon loses on the RTX 4050
  ([benchmarks-gpu.md](benchmarks-gpu.md)).

## Not planned, and why

- **Split-KV or f16-accumulate attention, NCCL-style ring all-reduce, atomics in reductions.** Each
  would break the bit-identity contract (decode == prefill == verify, deterministic output). Multi-GPU
  will use fixed-order reductions instead.
- **More single-stream 7B GEMM tuning on consumer GPUs.** The kernels are at bandwidth or power limits
  (above). An earlier experimental GEMM ("gemm3") was a no-go. Hopper is the exception.
- **AVX-512 and ARM/NEON CPU kernels.** No machine is available to measure them on. A
  correctness-only round under emulation is possible, but it couldn't make speed claims.
- **macOS and Windows builds.** CPU-only ports of dlopen, mmap and thread affinity can't be verified
  here, and Metal would be a new backend.
- **Mixture-of-experts** (Qwen3-30B-A3B, gpt-oss-20b). Deferred, not rejected: they don't fit a 12 GB
  card unsplit. Estimated at about two weeks of work, after Gemma 3 and S2.
- **Beating vLLM tensor parallel on NVLink H100/A100 for one stream.** Their all-reduce is ~5 µs there
  and their kernels are mature. Axon's multi-GPU case is PCIe consumer boxes and reproducible output
  ([design/multi-gpu.md](design/multi-gpu.md) §4).
