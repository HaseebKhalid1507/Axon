# Gate set `opaque-daada6f`: FAIL (3 of 5)

Published 2026-10-24 18:52:51 EDT by axon-gatelogs. Tested sha(s): `daada6f`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1024-171216-7030 | opaque-daada6f-md5all | FAIL (rc 130) | None | ? | ? | daada6f | md5all.sh | ?–17:36 | [log](1024-171216-7030.log) |
| 1024-171219-bdcb | opaque-daada6f-fast-rel [shard 1/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | daada6f | axon-test-daada6f | 10-24 17:18–17:43 | [log](1024-171219-bdcb.log) |
| 1024-171219-da22 | opaque-daada6f-fast-rel [shard 2/2] | FAIL (rc 130) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | daada6f | axon-test-daada6f | ?–17:36 | [log](1024-171219-da22.log) |
| 1024-171219-a517 | opaque-daada6f-fast-dbg-cpu [shard 1/2] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | daada6f | axon-test-dbg-daada6f | 10-24 17:18–17:30 | [log](1024-171219-a517.log) |
| 1024-171219-bb9b | opaque-daada6f-fast-dbg-cpu [shard 2/2] | FAIL (rc 130) | jade | CPU only (jade; GPU hidden) | 580.159.04 | daada6f | axon-test-dbg-daada6f | 10-24 17:17–18:52 | [log](1024-171219-bb9b.log) |

Redactions: none.

Hardware records: [None](hw-None.txt), [avante](hw-avante.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
