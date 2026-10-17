# Roadmap and README status refresh  (docs only)

- Date: 2026-10-17. Branch: docs/roadmap-refresh. Base: main 69adfa2 (after #68). Status: open.
- Pick / design: maintainer request to bring the status columns up to date with what is merged.

## What was stale
`docs/roadmap.md` said "on main means merged at `e998d6f`" (#35), and about twenty PRs had merged since. Rows
still read "in flight", "PR open" or "planned" for work on main: chunked prefill (#45, #55, #57), one cache
instead of N+1 (#60), long-context decode attention (#50), KV save stage A (#48, #54), serve ops (#61, #62), paged
KV stages 0–1 (#64, #68), YaRN (#49), old GPUs (#39), and Hopper wgmma (#33, still "experiment pending").
"Recently landed" listed only the draft model (#27). The README said chunked prefill and the long-context decode
fix were not on main, that any RoPE scaling but Gemma 3's is refused (YaRN is accepted since #49), that a GPU
NVRTC can't target is refused (since #39 the next NVRTC is tried), and didn't mention `chat --session`.

## What changed
- `docs/roadmap.md`:
  - The sha is `69adfa2`; "Recently landed" lists the merged work, newest first, with each PR's MEASURED headline.
  - Priorities: 1 (#38), 4 (#34), 7 (#39), 8 (#32) and the long-context half of 10 (#50) are done; 5 has its
    first round (#33: +12 % 7B prefill on an H100 PCIe, below the design estimate); 2 has the fuzzers and fixes
    (#37), and its soak and model-zoo sweep are in flight (feat/soak-zoo).
  - Scale: 1, 2, 3 done; 4 stage A done, stage B (serve) planned; 5 stages 0–1 done, stages 2–3 in flight
    (feat/paged-kv-pool); 10 is planned, its prerequisite done.
  - Also open: `/v1/responses` + ollama-native API in flight (feat/serve-apis); new rows for Q2_K/Q3_K
    (feat/q2k-q3k) and MoE stage 2 (PR #67), both in flight.
  - The Dependencies line is rewritten for what is done. The H100 sentence in "The thinking" mentions #33.
- `README.md`: line counts (`git ls-files` + `wc -l`: 67k Rust, 6.5k CUDA), the YaRN exception to the RoPE
  refusal, NVRTC fallback (#39), a `chat --session` bullet and usage flag (#48), chat templates (Mistral, Gemma 3),
  serve's chunked prefill, one-cache slots and ops flags, the "Not on main yet" list, and a #33 note under the
  H100 benchmark (whose numbers predate #33 and are left as measured).

## Left alone on purpose
- Scale row 7 and the "Mixture-of-experts" section: PR #67 edits them. Rows 6 and 8 sit on the lines next to
  row 7, and git reports a conflict for edits on adjacent lines, so their status cells are unchanged too.
  A paragraph under the table says that item 6 (#61, #62) and item 8's YaRN half (#49) are done, and that
  `chat` forgetting in halves is still planned (`chat.rs` still drops the fewest exchanges that make room).
  Fold this into the cells once #67 merges. The README's `qwen3moe` lines, which #67 also edits, are unchanged.
- The mission statement and "Long-term goals". Its "no kernels for Q2_K/Q3_K" is still true on main.

## How it was verified
| check | result |
|---|---|
| every status against `git log --merges origin/main`, `gh pr view N` and docs/log | done; numbers copied from the PR bodies / logs |
| README claims against `src/` (`ARCHS` and `SUPPORTED_TYPES` in model.rs, YaRN in `check_supported`, `nvrtc_where`, `--session` in main.rs, gemma3 q8 refusal) | match |
| relative links and `#anchors` in the changed files (script) | all resolve |
| `git merge-tree` of this branch with PR #67's head, `docs/roadmap.md` and `README.md` | no conflict in either |

The in-flight branches (feat/paged-kv-pool, feat/serve-apis, feat/q2k-q3k, feat/soak-zoo) are local only, not
on origin, and have no commits past main yet. Their status comes from the maintainer's brief, not from a PR.
