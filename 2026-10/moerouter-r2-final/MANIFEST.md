# Gate set `moerouter-r2-final`: FAIL (2 of 9)

Published 2026-10-20 00:52:48 EDT by axon-gatelogs. Tested sha(s): `fdd7816`, `764792f`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-002429-907c | axon-test-fdd7816 moe:: mla:: oss_ gptoss deepseek2 chat::tests::moe --test-threads 1 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-fdd7816 | 10-20 00:24–00:30 | [log](1020-002429-907c.log) |
| 1020-002430-c70d | axon-test-dbg-fdd7816 moe:: mla:: oss_ gptoss deepseek2 chat::tests::moe --test-threads 1 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-dbg-fdd7816 | 10-20 00:26–00:32 | [log](1020-002430-c70d.log) |
| 1020-002430-e156 | axon-test-fdd7816 --ignored router_bits_match_legacy --nocapture | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | fdd7816 | axon-test-fdd7816 | 10-20 00:24–00:24 | [log](1020-002430-e156.log) |
| 1020-002431-db39 | axon-test-fdd7816 ds2_gpu_moonlight_bits deepseek2_moonlight_bits oss_gpu_contract_model oss_gpu_logits_graded_model oss | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-fdd7816 | 10-20 00:32–00:36 | [log](1020-002431-db39.log) |
| 1020-002431-81be | md5-moerouter | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | md5.sh | 10-20 00:36–00:42 | [log](1020-002431-81be.log) |
| 1020-004401-2b1e | md5-moerouter | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | md5.sh | 10-20 00:46–00:52 | [log](1020-004401-2b1e.log) |
| 1020-004332-e3d1 | axon-test-base-764792f moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | axon-test-base-764792f | 10-20 00:43–00:44 | [log](1020-004332-e3d1.log) |
| 1020-004332-a0e9 | axon-test-base-764792f moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | axon-test-base-764792f | 10-20 00:44–00:45 | [log](1020-004332-a0e9.log) |
| 1020-004333-4d84 | axon-test-fdd7816 moe::oss_tests::oss_gpu_logits_graded_model --exact --test-threads 1 --nocapture | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | fdd7816 | axon-test-fdd7816 | 10-20 00:45–00:46 | [log](1020-004333-4d84.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
