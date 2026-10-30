# Gate set `fp8-7b-bbd9821`: ALL PASS; TAINTED timing

Published 2026-10-30 08:10:58 EDT by axon-gatelogs. Tested sha(s): `bbd9821`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1029-201935-bd1a | fp8-7b-acc-cpu-bbd9821 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | bbd9821 | acc7b_cpu.sh | 10-30 01:07–02:59 | [log](1029-201935-bd1a.log) |
| 1029-201935-949b | fp8-7b-acc-gpu-bbd9821 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bbd9821 | acc7b_gpu.sh | 10-30 00:52–01:07 | [log](1029-201935-949b.log) |
| 1029-201935-2a32 | fp8-7b-bench-bbd9821 | PASS TAINTED (FOREIGN_GPU max=71% mean=10.9%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bbd9821 | bench7b.sh | 10-30 08:08–08:10 | [log](1029-201935-2a32.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
