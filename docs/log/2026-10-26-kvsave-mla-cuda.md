# kvsave: the CUDA deepseek2 net refuses a session image (#159)

- Date: 2026-10-26. Branch: fix/kvsave-mla-cuda, off main d8a296c. Closes #159.

## What changed
`src/kvsave.rs`: `KvIo for CudaQwen2::kv_ident` returns `None` for an MLA (deepseek2) net, as `KvIo for Qwen2`
(the CPU) already did. A refused identity makes `chat --session` prefill the saved tokens again instead of loading
rows.

## Why
#159, found in review (#158). The CUDA impl only refused split models.
- On deepseek2 its save would write no rows under a header that claims a valid image, and a restore would decode
  from a cache it never wrote.
- Product reach (CODE): `chat` saves through `Backend`, and deepseek2 on CUDA is `Inner::Moe`, whose `kv_ident` is
  already `None`. So `chat --session` was not exposed. The gap was the GPU net's own `KvIo`, reachable by any caller
  that drives `CudaQwen2` directly.
- The `Backend` impl (Cpu / Cuda arms, `_ => None`) needs no change.

## How it was verified
`moe::ds2_tests::ds2_cuda_session_image_refused`:
- the deepseek2 fixture's GPU net (`MoeQwen3::gpu_net`) must refuse a session image;
- so must the product `Backend` (asserted `Inner::Moe`).

It fails at 050287f (test only, main's code) and passes at c19ba6c. The kvsave / session / chat tests and the
deepseek2 fixture tests run in release and debug; job ids are in the PR. No gen path changes, so there is no md5.
