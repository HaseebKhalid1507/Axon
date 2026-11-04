# `pool_wide_long_steps_match_solo` on 6 GB cards: skip the case that doesn't fit (PR #168)

- Date: 2026-10-27. Branch: fix/pool-wide-small-card. Base: main 6f06544. Status: open.
- Cause: #151 grew the test to 40 slots × 1,152 positions (≥ `ATT_NB_MIN`). Qwen3-0.6B's f16 pool then needs
  5.07 GiB, more than a 6 GB card has free (avante RTX 4050: 4.98 GiB), and `net.pool(..).unwrap()` panicked.
  - Main fails alone on avante (1027-095951-0faf).
  - So did #167's fast and debug gates (1027-090606-5804, -b980).
  - Main's gates never ran the test on a card that small.
- Fix (test only): a case whose pool the card can't hold (`net.pool` → "… of GPU memory") is skipped with a line
  naming the model and cache; the other cases still run; any other error panics. Each case says `ran` or
  `skipping` on stderr (visible with `--nocapture`).
- Verified (PR §6):
  - On avante, Qwen3 f16 is skipped and Qwen3 q8 and Llama-1B f16 / q8 run.
  - On bella (12 GB) every case runs: 146–151 s with a 12 GB host-RAM claim, the first measured times.
  - A bella run under a 3.9 GB claim (6.2 GB MemoryHigh) thrashed and hit its 10-min cap: the test keeps ≈ 2.6 GB of
    solo KV on the host.
- Audit: no other test from #151 / #153–#166 on main fails on a 6 GB card (PR §1).
