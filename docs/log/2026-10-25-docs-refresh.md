# Docs brought up to main 39624db: roadmap, README flags, the #131 serve A/B, results, history  (docs only)

- Date: 2026-10-25. Branch: docs/refresh-20261025. Base: main 39624db (#131). Status: open.
- Pick / design: maintainer request after the 10-20 to 10-25 merges (#104 through #131). The last refresh was
  [2026-10-19-docs-refresh.md](2026-10-19-docs-refresh.md) at 764792f (#103).

## What changed
- `docs/roadmap.md`:
  - "On main" = 39624db.
  - "Recently landed": a new first list, newest first, for the 18 merges after 764792f (#103 to #131). Each entry
    has its PR's own headline and log link, with the PR's labels kept (#127's kill rule is TAINTED, and the PR
    says its absolute tok/s aren't clean numbers, so only the ratios are quoted). A "Not landed" line covers #119
    (rejected), #126 (superseded by #131) and #125 (draft). The 764792f list follows, then the 69adfa2 list.
  - "Next: proposed order": item 1 (a) is draft #125, rehearsed (bella 1025-153025-5e47 PASS), with no pod run:
    it waits for Haseeb's approval. (b) stays REJECTED (#130). (c)/(d) landed in #131: (d) is the default, and (c)
    is opt-in after missing its kill-rule clause, with the H100 clauses and the sjf default pending in #139.
    Item 3: PRs A to C are on main for deepseek2 / Moonlight (kill rule PASS, TAINTED); Qwen3-30B-A3B is next.
    Item 5: done except the tag. #117 delivered `lib.rs`, the README board, "Why not vLLM?", `CHANGELOG.md` and the
    draft release notes; #124 closed #118's v0.1 blockers A and B. v0.1.0 is not tagged.
  - Mixture-of-experts: a status paragraph for resident experts, a status line on the Qwen3-30B-A3B plan item, and
    the "Not planned" MoE line scoped to 12 GB cards.
  - deepseek2 (Frontier item 2): resident routed experts are done. Frontier item 1 and the DeepSeek V3 table row now
    say what's left. Long-term goals and agent-harness item 4 cite #127. Scale "below the line" notes that
    shortest-first admission with an age bound is on main as an opt-in (#131).
- `README.md`:
  - **Stale claims fixed.** gpt-oss in `serve` said "no tools yet", which has been wrong since #111. "Not on main
    yet" listed keeping MoE experts on the GPU, which deepseek2 can do since #127. The qwen3moe paragraph named
    streaming as next, where the roadmap now has residency. Line counts were 90k Rust / 677 tests; they are now
    99,119 / 8,424 CUDA / 738.
  - **Flags checked against `src/cli.rs` at 39624db.**
    - `--gpu-experts`: added to the gen / chat usage; it is opt-in, default 0, does nothing on the CPU or with dense
      models (`lm.rs`: only the MoE CUDA path reads it), qwen3moe and gpt-oss fall back with a reason, and `serve`
      reads `AXON_GPU_EXPERTS` and runs a MoE model one request at a time.
    - `--step-ms`: 0 means exactly `--prefill-chunk` tokens (the README said "N"); a step feeds at most the prompts
      that fit; `--prefill-ms` is still accepted as an alias (`serve.rs` 3125).
    - `--step-tokens` also lifts the bound on prompts per step.
    - `--admit-order sjf`: opt-in until #139, and it cuts the short-prompt tail, not yet the median.
    - Rollback: `AXON_STEP_COST=legacy` with the default fifo.
    - `AXON_ADMIT` / `AXON_STEP_COST` rows added to the environment table.
    - MoE models are added to the list of what `serve` runs one request at a time (`moe_parallel_refusal`).
    - `priority` no longer says requests always run in arrival order.
    - The serve usage line names `--step-ms` and `--admit-order`.
- `docs/benchmarks-gpu.md`: a new section, "Serve scheduler A/B: laptop GPUs (2026-10-25, #131)". It has two compact
  tables (main / fifo60 = default / sjf60 on the three bella kill-rule cells, avante 1.5B, and sdab15 C=16/32),
  copied from the PR's §5, and links to the PR for every arm (sjf50 and the a18ff8d run). The page holds boards
  against other engines, and each serving section has one summary table and links to its log, so this follows that.
  The intro names the section.

Second commit, at the foreman's request (the stale docs this refresh first only reported):
- `docs/experiments-rejected.md`, the hybrid-step row: "(c)/(d) continue … via #126" now says that they landed in #131.
  (d) is the default; (c) is opt-in after missing "short TTFT p50 ≤ 0.5× fifo" (0.56×, 0.79×), pending #139.
- `docs/START-HERE.md`:
  - the table is brought up to 39624db (Phases 17–20), with line counts at 39624db;
  - a Library row (#117, #124);
  - deepseek2 resident experts in Architectures (#127);
  - the step budget, sjf opt-in and MoE one-at-a-time in Serving (#131);
  - the "Not on main" line;
  - **In flight as of 39624db**, every item open work: #125 (draft, rehearsed, pod waiting), #139, the Moonlight
    bench of record and Qwen3-30B-A3B step 0 (item 3's next steps), the #135 / #137 hardening fixes, the rest of
    #118, the ADR, batched speculation (gate v2 designed), FP8 stage B.
- `docs/results.md`:
  - main = 39624db;
  - the box's caveat table gains rows for every merge since 764792f, each effect copied from its PR body, with the
    `merged after` column `c15c90e` (the H100 board's commit) and the exception note adjusted;
  - #127 in "New models and types", against Axon's own CPU-expert path, TAINTED, ratios only;
  - known losses and gaps as of 39624db: resident Moonlight not run against llama.cpp; `serve` runs MoE one
    request at a time; sjf missed its kill rule; #135, #137.
- `docs/history.md`: **Phase 20** (764792f → 39624db, 18 merges, times from the merge commits in EDT) in Phase 19's
  table style, plus the closed #119 / #126 and what is open at 39624db. The intro now runs to 39624db.
  `docs/log/README.md` points to Phase 20.

Third commit, shady's review (8/10 FIX-THEN-MERGE, https://github.com/HaseebKhalid1507/Axon/pull/140#issuecomment-6406948239):
- MUST: history.md Phase 20's split was wrong. `git log --first-parent 764792f..39624db` gives 8 merges (#103–#120,
  10-19 23:30 to 10-20 17:44) and 10 (#123–#131, 10-24 17:37 to 10-25 21:19), not 7 and 11.
- Roadmap item 1 (a): the pod session is now "≈ $1.20–2.00, ESTIMATED". #125's review gives $1.20 if the stage
  estimates hold and $2.00 if the stages run to budget.
- Roadmap item 1: "60.0 ms at 50" keeps #131's caveat (measured on the sjf50 arm; no fifo50 arm was run).
- `docs/design/determinism-modes.md:411`: the link inside `docs/design/` pointed at `design/determinism-modes.md`, which
  is broken (it was already broken on main). It now points at `determinism-modes.md`.
- README: "8.4k of CUDA", as in START-HERE.

## Sources
- PR bodies and comments: `gh pr view N` for #103, #104, #105, #107, #111, #113, #117, #120, #122, #123, #124, #125
  (draft, with comments), #127 (with the lead's kill-rule comment, 6403248891), #128, #130, #131, #133, #134 and #136;
  issues #118 and #139. `gh pr list --state merged` and `git log --first-parent 764792f..39624db` give the same 18
  merges.
- Log entries on main: 2026-10-19-moe-router-unify, 2026-10-20-{graded-ref-repack, gemma3-inplace-window,
  gpt-oss-tools, inflight-gauge, lib-split, moe-resident-a}, 2026-10-24-{moe-resident-b, moe-resident-c,
  opaque-backend}, 2026-10-25-{reject-hybrid-step, testutil-moe-mapped, ram-cgroup, harmony-unheaded, testutil-map,
  sched-budget-main}.
- For the second commit: `git log --first-parent` merge-commit times (EDT) for Phase 20; `gh pr view 119 / 126`
  (closed 10-25 00:56 / 01:06 EDT); `gh pr list --state open` (only #125 and this PR); issues #121, #135, #137.
  The in-flight tracks without a PR (Moonlight bench of record, Qwen3-30B-A3B step 0, #135 / #137 fixes) come from
  the foreman and are listed as open work, with no claims about them.
- Code at 39624db, for the flag claims: `src/cli.rs` (USAGE, flags, environment), `src/serve.rs` (`--prefill-ms`
  alias, `PAGED_PARALLEL`, `moe_parallel_refusal`), `src/serve/engine.rs` (`Sched::from_env`, `lanes_for`,
  `two_pass_budget`, `step_prompts`), `src/lm.rs` (`gpu_experts` only on the MoE CUDA path), `src/moe.rs`
  ("resident experts: deepseek2 only for now").
- Line counts at 39624db: `git ls-files '*.rs' | xargs cat | wc -l` 99,119; `'*.cu' '*.cuh'` 8,424;
  `grep -rc '#\[test\]' src` summed, 738. This is #103's method: it gives 89,986 at 764792f, as #103 recorded.

## How it was verified
| check | result |
|---|---|
| every number in the new text, against its PR body or log (copied, not recomputed; ratios as the PRs give them) | by hand |
| relative links and anchors in README.md, docs/*.md, docs/log/*.md (GitHub slug rules; negative-tested with a bad anchor and a missing file) | 0 bad |
| CI's checks | cargo only (fmt, clippy, tests, qemu); the repo has no docs lint in CI or `scripts/`. README isn't a doctest (`src/lib.rs` doesn't `include_str!` it) |
| cargo, builds, model runs | none (docs only) |

## Found, not changed here
- `axon --help` says `--parallel N` defaults to 8 (USAGE and the flag text), but with paged KV (the default) it is
  16 (`PAGED_PARALLEL`, `serve.rs`). The README already says 16 / 8, so the help text is the stale one. It's a code
  change, so the foreman has routed it to the #137 fix.
