# The north star on the roadmap: run Synaps on Axon (agent workload track)  (docs only)

- Date: 2026-10-27. Branch: docs/agent-north-star. Base: main d8a296c (#152). Status: open.
- Pick / design: Haseeb set the north star on 10-26 ("run Synaps on Axon: many concurrent sessions, each spawning
  lots of subagents, with sessions staying warm in cache, then multiple users"). Fable wrote the track spec, which
  the lead asked to apply.

## What changed
- New `docs/design/agent-workload.md`: Fable's track spec, copied from the foreman's workspace
  (`specs/agent-workload.md`). Changed:
  - a header line (north star: owner, 10-26; ranking: lead (Jawz), 10-27; the design itself stays PROPOSED);
  - relative links for the repo location (determinism.md, multi-gpu.md, the #71 / H100-anatomy / W4A8-probe logs,
    history for #48, the roadmap);
  - a re-count note in §0.
  The §0 aggregate table is numbers and structure only. I read it line by line and found no message text, no
  names, no paths beyond the log file pattern, and no credentials.
- `docs/roadmap.md`:
  - the north-star paragraph (design §7) opens "Mission and positioning";
  - "Next: proposed order": a "north star: owner, 10-26; ranking: lead (Jawz), 10-27" note and rows **2A** (PR A, in shelved item 2's slot) and
    **2B, 2C** (two tracks, ahead of item 4), each with the design's costs and pre-registered rules. Item 4 is
    marked as behind them and blocked on the ADR. The shelved item 2 row stays as it was, since item 4's text
    refers to it;
  - "Agent-harness workloads" drops "(later: not scheduled)" and takes the design's §7 rows (anchors updated);
  - the two "below the line" mentions of stage B and shared-prefix pages now point at 2B / 2C.
- `docs/START-HERE.md`: one sentence pointing at the north star.

## Sources and checks
- The design's §0 numbers were re-counted from the `Token Usage` lines of `~/.synaps-cli/synaps.log.2026-10-2{0..7}`,
  numbers only. The re-count found 22,838 requests (the 10-27 log is live) against the spec's 22,795. The prompt
  p50 / p90 / p99 (165,532 / 410,485 / 809,911) and the 97.6 % cached share match. The output and new-token
  percentiles are within the extra requests (output p50 351, p90 1,543, p99 5,554, mean 700; new tokens p50 1,045).
  The gap, fan-out and lifetime rows weren't re-counted (they need the dispatch lines); they stay as the spec gives
  them.
- #48's 67 ms page restore: `docs/history.md` (#48 row). The 57 % attention share at depth 4,096:
  `log/2026-10-26-h100-anatomy-results.md`. Item 2 shelved: `log/2026-10-26-w4a8-probe.md`.
- Relative links and anchors in README, docs/, docs/log, docs/design, docs/research, docs/reviews: script (GitHub slug
  rules, negative-tested), 0 bad.
- No cargo, builds or model runs (docs only).

## Review round 1 (shady 7/10 APPROVE-WITH-FIXES, https://github.com/HaseebKhalid1507/Axon/pull/158#issuecomment-6433633068)
- **Attribution.** "North star: owner, 10-26; ranking: lead (Jawz), 10-27" in the design header, §8 (was
  "Proposal:"), the roadmap and START-HERE. Haseeb set the north star and agreed to queue it after the H100
  anatomy; the placement against items 1–5 is the lead's.
- **Stale roadmap sentences.**
  - "Haseeb decides" now excepts the agent rows.
  - Item 4's "moves up" is gone.
  - Scale item 4's stage B points at 2B.
  - The Scale section's llama.cpp claim now quotes the README exactly.
- **Capacity.**
  - The H100 7B row was the showdown's *configured* 1.05M pool. It is now ≈ 1.35M ESTIMATED (72 GiB / 56 KiB)
    ≈ 54 sessions (q8 ≈ 2.5M ≈ 101), and §5's warm-session gain is ≈ 54 → ≈ 90 with a shared 10k prefix.
  - §5 no longer says sharing would move the 10-19 board: its prompts are salted.
  - The capacity table's caption labels every cell ESTIMATED except #71's two MEASURED bella cells.
  - 25.4 GB/s is cited (research-kvmem, pinned H2D on bella) and the D2H rate is marked ESTIMATED.
  - The re-prefill is now ≈ 10 s (25k / 2.43k tok/s from #71), not 5–10 s.
- **Kill rules.**
  - A counts re-prefill after preemption.
  - C fixes the fan-out cell (A's p50 shared span plus a private delta of A's p50 `new`, with a fallback clause
    if the shared span is < 4× the delta), puts warm/GB on agent-32k at speed 4, and gates dedupe-only itself.
  - F gates 0 cross-tenant hits, TTFT medians within the round spread at n = 1,000 × 3 rounds, tenancy overhead
    ≤ 2 % on C=1 decode and agent-32k aggregate, and a claim rule for the H100 agent board.
- **Exactness (§2.1 / §2.2), checked against main d8a296c.**
  - Joiners take only pages completed in earlier steps: chunks aren't page-aligned (`plan_charged`).
  - Shared pages stop at `min(common, len − 1)` (`admit`: `reuse = common.min(p.len() - 1)`).
  - Condition 2 moved to the tiers: there is no pool on a split (`pool_with`), and `kv_ident` is `None` for every
    split and for MLA.
  - Identity keying: `kvstore::ROOT` is constant today, so the disk tier and FP8 mode must key by identity.
  - New coverage notes for the solo path (`Engine::job` → `generate` → `adopt`), refcounted page accounting
    (`admit_check`, `reclaim`, `idle_pages`) and a preempted prefix owner.
- **llama.cpp.** The exact `cache_prompt` wording from `tools/server/README.md` at 4413a19.
- **Privacy.** The repo is public. Raw traces never go in the repo; only the scripts and a small synthetic example
  are committed; any shared trace drops its dates and shifts each day to start at t = 0.

