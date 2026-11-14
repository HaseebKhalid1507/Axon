# Gate set `chat-bac1332-regate`: FAIL (4 of 8)

Published 2026-11-13 20:28:57 EST by axon-gatelogs. Tested sha(s): `bac1332`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-171545-0c9e | chat-bac1332-full-rel | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | bac1332 | axon-test-bac1332 | 11-13 17:27–18:20 | [log](1113-171545-0c9e.log) |
| 1113-171546-d73f | chat-bac1332-full-dbg | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bac1332 | axon-test-dbg-bac1332 | ?–20:28 | [log](1113-171546-d73f.log) |
| 1113-171546-09ec | chat-bac1332-fp8graded-rel | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bac1332 | axon-test-bac1332 | 11-13 17:27–18:08 | [log](1113-171546-09ec.log) |
| 1113-171547-d090 | chat-bac1332-fp8graded-dbg | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bac1332 | axon-test-dbg-bac1332 | 11-13 17:27–17:43 | [log](1113-171547-d090.log) |
| 1113-171550-b79c | chat-bac1332-templates | PASS | jade | CPU only (jade; GPU hidden) | 580.159.04 | bac1332 | axon-test-bac1332 | 11-13 17:15–17:16 | [log](1113-171550-b79c.log) |
| 1113-171550-dd6e | chat-bac1332-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bac1332 | bash | 11-13 17:31–17:33 | [log](1113-171550-dd6e.log) |
| 1113-171551-3fca | chat-bac1332-servemd5 | FAIL (rc 1) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bac1332 | bash | 11-13 17:33–17:33 | [log](1113-171551-3fca.log) |
| 1113-171551-93e4 | chat-bac1332-golden11 | FAIL (rc 130) | None | ? | ? | bac1332 | bash | ?–18:08 | [log](1113-171551-93e4.log) |

Redactions: none.

Hardware records: [pod2-11-13_17:27](hw-pod2-11-13_17:27.txt), [bella](hw-bella.txt), [pod1-11-13_17:27](hw-pod1-11-13_17:27.txt), [jade](hw-jade.txt), [pod1-11-13_17:31](hw-pod1-11-13_17:31.txt), [pod1-11-13_17:33](hw-pod1-11-13_17:33.txt), [None](hw-None.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
