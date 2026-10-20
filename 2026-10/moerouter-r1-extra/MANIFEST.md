# Gate set `moerouter-r1-extra`: ALL PASS

Published 2026-10-20 00:15:29 EDT by axon-gatelogs. Tested sha(s): `764792f`, `0fac171`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1019-222233-1991 | md5-moerouter | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | md5.sh | 10-19 23:53–23:58 | [log](1019-222233-1991.log) |
| 1019-222728-91cd | axon-legacy-0fac171 router_bits_match_legacy --nocapture | PASS | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | 0fac171 | axon-legacy-0fac171 | 10-19 22:27–22:28 | [log](1019-222728-91cd.log) |
| 1019-222212-6a02 | axon-test-0fac171 moe:: mla:: oss_ gptoss deepseek2 chat::tests::moe --test-threads 1 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0fac171 | axon-test-0fac171 | 10-19 22:22–22:25 | [log](1019-222212-6a02.log) |
| 1019-222212-81ce | axon-test-dbg-0fac171 moe:: mla:: oss_ gptoss deepseek2 chat::tests::moe --test-threads 1 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0fac171 | axon-test-dbg-0fac171 | 10-19 22:25–22:28 | [log](1019-222212-81ce.log) |
| 1019-223417-c09e | perf-moerouter-base | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 764792f | axon-base-764792f | 10-19 22:44–22:45 | [log](1019-223417-c09e.log) |
| 1019-224518-0544 | perf-moerouter-cand | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0fac171 | axon-0fac171 | 10-19 22:45–22:45 | [log](1019-224518-0544.log) |
| 1019-224619-05ca | perf-moerouter-base | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 764792f | axon-base-764792f | 10-19 23:02–23:02 | [log](1019-224619-05ca.log) |
| 1019-224619-ec6c | perf-moerouter-cand | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0fac171 | axon-0fac171 | 10-19 23:02–23:02 | [log](1019-224619-ec6c.log) |
| 1019-230220-573c | perf-moerouter-base | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 764792f | axon-base-764792f | 10-19 23:02–23:02 | [log](1019-230220-573c.log) |
| 1019-230221-d977 | perf-moerouter-cand | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0fac171 | axon-0fac171 | 10-19 23:02–23:02 | [log](1019-230221-d977.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [jade](hw-jade.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
