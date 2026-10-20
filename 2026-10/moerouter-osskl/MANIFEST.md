# Gate set `moerouter-osskl`: FAIL (3 of 5)

Published 2026-10-20 00:46:43 EDT by axon-gatelogs. Tested sha(s): `764792f`, `0fac171`, `fdd7816`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-004332-e3d1 | axon-test-base-764792f moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | axon-test-base-764792f | 10-20 00:43–00:44 | [log](1020-004332-e3d1.log) |
| 1020-004332-a0e9 | axon-test-base-764792f moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | axon-test-base-764792f | 10-20 00:44–00:45 | [log](1020-004332-a0e9.log) |
| 1020-004332-588c | axon-test-0fac171 moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 0fac171 | axon-test-0fac171 | 10-20 00:45–00:45 | [log](1020-004332-588c.log) |
| 1020-004333-4d84 | axon-test-fdd7816 moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-fdd7816 | 10-20 00:45–00:46 | [log](1020-004333-4d84.log) |
| 1020-004333-a2c3 | axon-test-fdd7816 moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-fdd7816 | ?–00:45 | [log](1020-004333-a2c3.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
