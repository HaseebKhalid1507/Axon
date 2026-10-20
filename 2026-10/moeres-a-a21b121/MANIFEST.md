# Gate set `moeres-a-a21b121`: FAIL (1 of 8)

Published 2026-10-20 16:59:12 EDT by axon-gatelogs. Tested sha(s): `a21b121`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-162850-7c8a | moeres-a21b121-twin-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | a21b121 | axon-test-a21b121 | 10-20 16:42–16:43 | [log](1020-162850-7c8a.log) |
| 1020-162853-5647 | moeres-a21b121-twin-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a21b121 | axon-test-a21b121 | 10-20 16:28–16:30 | [log](1020-162853-5647.log) |
| 1020-162900-ea13 | moeres-a21b121-twin-jade | PASS | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | a21b121 | axon-test-a21b121 | 10-20 16:29–16:29 | [log](1020-162900-ea13.log) |
| 1020-162854-8999 | moeres-a21b121-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | a21b121 | md5job.sh | 10-20 16:43–16:46 | [log](1020-162854-8999.log) |
| 1020-162900-74db | moeres-a21b121-fast-rel [shard 1/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a21b121 | axon-test-a21b121 | 10-20 16:30–16:59 | [log](1020-162900-74db.log) |
| 1020-162900-6eb0 | moeres-a21b121-fast-rel [shard 2/2] | FAIL (rc 137) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | a21b121 | axon-test-a21b121 | 10-20 16:53–16:54 | [log](1020-162900-6eb0.log) |
| 1020-162900-75b7 | moeres-a21b121-fast-dbg-cpu [shard 1/2] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | a21b121 | axon-test-dbg-a21b121 | 10-20 16:29–16:48 | [log](1020-162900-75b7.log) |
| 1020-162900-8b31 | moeres-a21b121-fast-dbg-cpu [shard 2/2] | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | a21b121 | axon-test-dbg-a21b121 | 10-20 16:40–16:42 | [log](1020-162900-8b31.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
