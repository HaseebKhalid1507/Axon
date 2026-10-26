# cuda/mla: faster MLA prefill (batched heads GEMM, shared K tiles), same bits  (PR #147)

- Date: 2026-10-26. Branch: perf/mla-prefill (stacked on #145; main 680a2c5 plain-merged in). Status: open, draft;
  gates pending.
- Pick / design: spike's round-2 plan (`~/.cache/axon-moeres/spec-res-r2.md`, S2.5). His trace 8bc8: at pp512
  `mla_heads_mv` + `mla_attn` take 159 of 232 ms of GPU busy time; at pp2100, 1,170 of 1,420 ms. Kill rule, fixed before
  building: pp2100 at `--gpu-experts auto` ≥ 2,800 tok/s (0.6 × llama.cpp's 4,680), in clean windows.

## The S2 decode kill rule (#145), from the clean rerun's job 1026-054428-1455
Job 1455 (`perf3.sh`, `--timing bella`, main be13f1f vs 2be8bfe vs llama.cpp `-t 8 -ncmoe 1`, 3 rounds, order rotated).
**TAINTED:** foreign GPU max 82 %, mean 13.0 %; the max is one kitty/Xorg spike at 07:51, near the end.

| round | short tg128: base / cand / llama.cpp | long tg128 (after pp2101): base / cand / llama.cpp `-d 2048` | pp2101: base / cand |
|---|---|---|---|
| 1 | 120.73 / 150.02 / 177.99 | 118.61 / 139.48 / 168.89 | 1,510.9 / 1,834.6 |
| 2 | 121.44 / 150.96 / 173.28 | 118.70 / 139.85 / 169.53 | 1,526.5 / 1,847.7 |
| 3 | 126.27 / 152.00 / 173.00 | 118.91 / 139.03 / 150.05 | 1,509.2 / 1,844.3 |
| median | 121.4 / **151.0** / 173.3 | 118.7 / 139.5 / 168.9 | 1,510.9 / 1,844.3 (+22 %) |

**Verdict:** ≥145 tok/s clause MET on a tainted run: all 3 rounds ≥150, median 151.0. Foreign load can only slow our
arm, so the absolute clause holds a fortiori. Ratio 0.87× llama.cpp, same run, also tainted. No clean window has been
obtained on bella's desktop.
