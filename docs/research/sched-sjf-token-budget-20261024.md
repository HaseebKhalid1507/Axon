# Spec: shortest-first admission + a cost-derived token budget for `axon serve` (roadmap item 1 (c) and (d))

Okarin (subagent), 2026-10-20, read-only. Sources: main `6e3b19c` (`src/serve/engine.rs`, `src/serve.rs`),
`origin/feat/hybrid-step` (PR #119, `plan_fused` / `join_fused`), `docs/roadmap.md` "Next: proposed order" item 1,
`docs/reviews/2026-10-19-outside-review-next-steps.md` §2, `docs/results.md` H100 section, `docs/research/research-sched.md`
(Chrollo, 10-16; P2 is the ancestor of this spec), and the showdown's per-request data at
`~/.cache/axon-h100sd/pod-pod1/results/h100-1019-2058/{cells,requests}.jsonl` (run `h100-1019-2058`, 11,992 requests).
Labels: **CODE** (file:line on main unless noted), **MEASURED** (a cell / request record / PR table cited),
**ESTIMATED** (arithmetic shown), **OPINION**.

Nothing was built or run. Line numbers are main's; the hybrid branch moves them (its `engine.rs` is +178 lines).

---

## 0. Findings first

1. **The 2.5 s short-prompt TTFT at mix C=128 is three admission rules, not kernel speed** (CODE, MEASURED below):
   (i) FIFO: `serve.rs:1335` admits `waiting.iter().position(fits_slot)`, the oldest that fits; (ii) **one admission
   per engine iteration** (`serve.rs:1335-1347`: one `admit`, then `b.step()`), so 128 simultaneous arrivals take ≥ 128
   steps to get in; (iii) `MAX_PREFILLING = 2` (`engine.rs:211`, applies to paged pools, which the H100 run used: its
   banner says "16384 pages x 64"), so two long prompts in prefill block *every* admission, short or long, until one
   finishes. In the first wave at C=128 (128 arrivals at t=0), 104 short prompts of ~60 tokens (6,240 tokens, ≈ 0.45 s
   of prefill at Axon's 13.8k tok/s) got their first token spread from 16 ms to **14.7 s** (MEASURED, requests.jsonl).
2. **(d) matters most *with* #119, not without it.** On the hybrid branch the pass is a fixed `--prefill-chunk 256`
   tokens *including the decode rows* (CODE hybrid `plan_fused`: `share = floor.max(rows+1).next_multiple_of(64) - rows`).
   At nb = 128 that is **128 prompt tokens per step**. ESTIMATED: a 256-token fused pass on the H100 ≈ 20 ms (it is the
   same 2–4-tile GEMM shape as the 18.6 ms decode step), so prefill while decoding runs at ≈ 6.4k tok/s against
   today's ≈ 570 tokens / 58 ms ≈ 9.8k; the mix cell's 251k prompt tokens would need ≈ 39 s of prefill alone, more than
   today's whole 38 s wall. **#119 as written is a likely mix C=128 regression unless the budget scales with rows and
   the measured cost.** Its own kill rule (mix C=64 on a 4090) is at nb = 64 where the same rule leaves 192 tokens; it
   may pass there and still lose at 128. This spec's (d) is the fix; it should land in the same train as #119 or before.
3. **slo_ok at C=128 is a TPOT problem, not TTFT** (MEASURED): every kind's TPOT p50 is 55–58 ms against the 50 ms
   SLO; at C=64 TPOT is 26–35 ms and shorts fail on TTFT (p90 3.9 s against 2.0 s). So (c) alone can move slo_ok at
   C=64 (review's 0.74 → ≥ 0.95: plausible, §7) but **not at C=128**; there only a step budget ≤ 50 ms does, which is (d)
   with `--step-ms 50`, at a prefill-throughput cost (§5.4).
4. **Bits: no change on any path this spec touches** (§6). Admission order, lane count and chunk size decide which
   tokens share a step and where chunk boundaries fall; both are outside the identity (`feed_slot` rows == one feed of
   the whole prompt, engine.rs header; "pass boundaries are outside the identity", results.md 7f42c02 row). The
   existing identity gates (`serve_identity`, `serve_chunked_prefill_matches_solo`, `serve_batched_matches_solo`) are
   the proof obligation; two new unit tests pin the policy.
5. Expected from (c)+(d) alone (ESTIMATED, §7): short-prompt TTFT p50 at mix C=128 2.4–2.9 s → **0.1–0.3 s**; slo_ok at
   C=64 0.69–0.77 → **0.90–0.95**; mix C=128 aggregate **+5–12 %** (waiting rows 30 → ~19 of 128, vLLM's figure);
   C=128 slo_ok 0.20–0.32 → ≥ 0.7 **only** with `--step-ms ≤ 50`. The review's "+25–40 %" needs (b) too.

---

## 1. What IS (CODE audit, main `6e3b19c`)

### 1.1 The engine loop (`serve.rs:1300-1385`)

| step | code | consequence |
|---|---|---|
| pull from the channel | `:1322-1330`: pull while `fit <= b.free() && waiting.len() <= b.slots` | `waiting` holds at most free+1 fitting jobs: when 2 slots are free the scheduler sees 3 candidates. SJF needs to see the queue. |
| admission | `:1335-1347`: `first_fit = waiting.iter().position(fits_slot)`; if `can_admit` → **one** `admit`, then fall through to `b.step()` | FIFO; ≤ 1 admission per step. 128 arrivals ⇒ ≥ 128 steps of ramp (ESTIMATED 2–4 s at 16–30 ms per early step; MEASURED first-wave short TTFTs at C=128 rise 16 → 131 → 1,287 → … ms in arrival order) |
| `can_admit` (`engine.rs:288-320`) | a free slot; paged pool: `pre.len() < MAX_PREFILLING` (= 2); pages for prompt + min(n_gen, GROW_TOKENS=1024) beside what prefilling + running requests still need | the lane cap was tuned for 7 × 15k agents on bella (comment at `:205-210`: "finish together, late"). On the H100 (1M tokens of pages) it only serialises. |
| solo path | `:1347-1380`: a lone request with nothing else waiting runs S1 (`lend_solo`) | unchanged by this spec |

### 1.2 The chunk budget (`engine.rs:76-98, 395-409`)

- `budget()`: with rows decoding and a measured cost, `Micros((prefill_ms − decode_ms) · 1e3, cost)`; `decode_ms` is
  the **last** decode step's wall time (`:676`), `cost` an EMA (α = 0.5) of µs per token per (depth + DEPTH_K) learned
  only from feeds ≥ LEARN_MIN = 128 tokens (`:488-491`). Before a cost exists: `SEED_CHUNK = 512` tokens. Idle (no rows):
  `IDLE_MS = 250` ms worth, or `IDLE_CHUNK = 4096` tokens.
- `plan()` (`:908-962`) shares the budget among the ≤ 2 prefills: the starved first (STARVE = 4 misses), then shortest
  remainder first; the first gets at least `floor` (= `--prefill-chunk`, 256); a later one only if its share ≥ floor.
- `prefill()` then calls `feed(i, n)` per prompt → **one `feed_slot` pass per prompt** (`:482`). Two short prompts in
  one step are two passes, each paying the pass's fixed part (weights read once per pass).
- `--prefill-ms 60` is a fixed wall budget for chunk + decode. At nb = 128 on the H100: 60 − 18.6 ≈ 41 ms of chunk ≈
  570 tokens (ESTIMATED at 13.8k tok/s); the step lands at 58 ms (MEASURED ITL p50 58.0, p99 77.5). The controller
  works as designed; the design's target is 8 ms over the SLO.

### 1.3 What #119 changes (CODE, `origin/feat/hybrid-step` engine.rs)

- `step()` → `plan_fused()` when rows decode and prompts wait: the chunks and the rows go in one `step_fused` pass.
- The pass is `--prefill-chunk` tokens **in all**, rows included, rounded up to 64-token tiles, capped at
  `pass_max − rows − pre.len()` (`pass_max` = PREFILL_BATCH = 1152, `cuda/qwen2.rs:143`). `--prefill-ms` is dead under
  hybrid (README/CLI on the branch say so). `decode_ms` is updated only on decode-only steps.
- Several prompts' chunks ride one pass (`Vec<FeedChunk>`): the per-prompt pass cost of §1.2 disappears in hybrid mode.
- Order inside the pass: `plan(Budget::Tokens(share), floor.min(16), …)`, the same starved-then-shortest rule.
- Admission is untouched (same `serve.rs` loop, same `MAX_PREFILLING`). P2 "not in this PR" per its body.

### 1.4 Showdown evidence (MEASURED, `h100-1019-2058`, Axon mix cells; my reduction of requests.jsonl)

Per kind, round 1 (round 2 within ±2 % at C ≥ 64):

| C | kind | n | TTFT p50 ms | p90 | p99 | TPOT p50 ms | slo_ok |
|---|---|---|---|---|---|---|---|
| 64 | short (~57 tok) | 104 | 172 | 3,886 | 5,519 | 33.5 | 0.70 |
| 64 | long 2k/4k/6k/7.5k | 24 | 473 / 1,376 / 2,566 / 956 | 3.3–4.8 s | 3.7–5.9 s | 26–35 | 0.67–0.83 |
| 128 | short (~60 tok) | 208 | 2,867 | 12,937 | 14,690 | 56.5 | 0.19 |
| 128 | long 2k/4k/6k/7.5k | 48 | 2.9 / 3.5 / 6.1 / 4.9 s | 7.0–14.3 s | 8.2–15.6 s | 55–57 | 0.17–0.33 |

By wave (the closed loop sends every client's first request at t = 0, its second when the first ends):

| C | wave | short n | short TTFT p50 / p90 / max | long p50 / max |
|---|---|---|---|---|
| 64 | 1 (t_send < 0.5 s) | 53 | 1,909 / 4,487 / 5,590 | 3,345 / 6,132 |
| 64 | 2 | 51 | **57** / 305 / 3,078 | 541 / 2,386 |
| 128 | 1 | 104 | 4,603 / 14,427 / 14,766 | 7,252 / 15,707 |
| 128 | 2 | 104 | 2,482 / 4,134 / 12,036 | 3,244 / 9,675 |

Row occupancy, Σ(completion − 1) × TPOT / wall and Σ TTFT / wall:

| engine, mix | C | wall s | mean rows decoding | mean rows waiting for a first token |
|---|---|---|---|---|
| Axon | 64 | 20.8 | 48.5–49.6 / 64 | 8.1 |
| Axon | 128 | 38.0–38.6 | 86.3–89.9 / 128 | **26.9–30.0** |
| vLLM-FP8 | 128 | 9.9–10.1 | 86.6–87.1 / 128 | 19.4–19.7 |
| vLLM-AWQ | 128 | 12.1–12.2 | 88.6–91.6 / 128 | 19.4–22.0 |

Reading: Axon's decoding occupancy equals vLLM's; its *waiting* occupancy is 1.5× vLLM's, and in a closed loop a
waiting client is a row not decoding. The C=64 wave-2 short p50 of 57 ms shows what admission costs when nothing
blocks it: about one step. Everything above that is queueing.

### 1.5 vLLM / SGLang on the same cells (MEASURED, for the targets)

vLLM-FP8 mix C=128: short TTFT p50 716 ms, p99 2,106; long 528–726 ms p50; TPOT 11–15 ms. SGLang-FP8: short 443 ms p50.
Their short-prompt TTFT at C=128 is **not** < 0.3 s either; at C=64 it is 310–400 ms p50. The review's "< 0.3 s" would
put Axon ahead of both on that one number; §7 says why it is reachable (Axon's step is slower but a 60-token prompt
needs one of them).

---

## 2. Research appendix: what each scheduler does about short-prompt TTFT under long-prompt load, and starvation

Read for this spec: vLLM `vllm/config/scheduler.py` and `v1/core/sched/scheduler.py` (main, fetched 10-20); SGLang
`srt/managers/schedule_policy.py` (main, fetched) and the hyperparameter-tuning doc; arXiv abstracts for the papers;
`research-sched.md` §2–3 for the ones read in full there (Sarathi-Serve, Orca, DistServe, FastServe, VTC, DLPM).
Items marked *recalled* were not re-read today; verify before quoting numbers.

| system | admission / order | chunking & budget | short prompts under long load | starvation guard |
|---|---|---|---|---|
| **Orca** (Yu et al., OSDI '22) | FCFS, iteration-level: admit and return per step | none (whole prompts) | a short prompt waits for whatever prefill is in flight | none beyond FCFS |
| **Sarathi-Serve** (Agrawal et al., OSDI '24, arXiv 2403.02310) | FCFS; **stall-free batching**: decode rows first, then prefill chunks fill the per-iteration **token budget** (Alg. 1) | budget chosen from the TBT SLO against chunking overhead (§4.3): 256–512 tokens for Mistral-7B on one A100; ≤ 25 % prefill overhead at 512, ≈ 0 at 2048 (§5.4.1, MEASURED in the paper) | bounded TBT for the decoders; a short prompt still queues FCFS behind long ones' chunks | none (FCFS) |
| **vLLM V1** (`config/scheduler.py`, `sched/scheduler.py`) | `policy: fcfs | priority` (priority = user-given integer, ties by arrival); running requests scheduled first, then waiting, each taking up to the remaining `max_num_batched_tokens` (default 2048 in the config class; the engine args raise it by device, *recalled* 8192 for an API server on an 80 GB card: verify) | chunked prefill on by default; `long_prefill_token_threshold` caps one long prompt's chunk so it cannot take the whole budget ("not applied when the request is the only one in the batch"); `long_prefill_token_threshold_adaptive` floors that cap at budget / (queued + running) | the cap leaves budget for other prompts in the same step; order stays FCFS unless the client sets `priority`; `max_num_queued_tokens` rejects (503) when the prefill backlog exceeds target_TTFT × prefill_throughput ("a TTFT QoS mechanism", docstring) | FCFS; `scheduler_reserve_full_isl` admits only if the whole prompt fits the KV (no thrash); preemption is LIFO on the running list |
| **SGLang** (`schedule_policy.py`; server args) | `--schedule-policy fcfs` default; `lpm` (longest prefix match: cache-aware, "reorders requests to encourage more cache hits", tuning doc), `dfs-weight`, `lof`, `random`; **prefill-prioritised**: a new prefill batch runs before the next decode batch (`get_next_batch_to_run`, *recalled*), with `--enable-mixed-chunk` to merge | `--chunked-prefill-size` by GPU memory (*recalled* 2048 / 4096 / 8192 for < 25 / < 60 / ≥ 60 GB; −1 off); `PrefillAdder` admits by remaining KV tokens with `--schedule-conservativeness` (default 1.0) scaling the reserve for `max_new_tokens` | prefill-first means a short prompt arriving mid-decode gets the *next* iteration; decoders stall for it (the opposite trade from Sarathi) | in-batch prefix dedup deprioritises a request whose prefix is already being prefilled in the same batch (`IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`); otherwise FCFS |
| **DistServe** (Zhong et al., OSDI '24) / **Splitwise** (Patel et al., ISCA '24) | separate prefill and decode instances; KV shipped | n/a (phase separation) | TTFT and TPOT tuned independently; the interference chunking only *mitigates* (DistServe §2.3) | n/a; needs ≥ 2 GPUs. research-sched.md §2.2: not for one GPU |
| **FastServe** (Wu et al., arXiv 2305.05920) | **skip-join MLFQ**: input length (known) picks the starting queue; preemption per output token; KV swapped to host under pressure | n/a | a short input starts in a high-priority queue and runs ahead of long jobs already demoted | MLFQ promotion with a starvation threshold (periodic lift to the top queue) |
| **Learning to Rank** (Fu et al., arXiv 2408.15792, `vllm-ltr`) | predict the *rank* of output lengths with a small ranker (Kendall's τ loss); schedule ≈ SJF on predicted rank | vLLM's budget | 2.8× lower chatbot latency vs FCFS (CLAIMED); the input-length part of the job is already known | explicit starvation prevention: a request waiting past a quantum gets a priority boost ("promote" after `max_waiting_iters`, *recalled*) |
| **μ-Serve** (Qiu et al., USENIX ATC '24; *recalled*, not re-read) and **SSJF** (Qiu et al., 2024, arXiv 2404.08509, *recalled*) | a proxy model predicts output length; SSJF = speculative shortest-job-first on that prediction; μ-Serve uses the same predictor for power-aware frequency scaling | | SJF on predicted total service | SSJF reports a fairness/starvation knob (*recalled*; verify) |
| **VTC** (Sheng et al., OSDI '24) / **DLPM** (Cao et al., 2025) | fairness across *clients*: smallest virtual-token counter first; DLPM charges only uncached ("extend") tokens and orders by prefix match | | not about length per se; a client's short turn after another client's long prompt is favoured by the counter | the fairness bound is the guard (gap ≤ 2·max(w_p L, w_q M)) |
| **llama-server** (research-sched.md §3.1) | FIFO task queue; slots iterated by index; Sarathi-style hybrid batch up to `n_batch` 2048 | `n_ubatch` 512 | none; a short prompt waits its slot's turn | none |

What this spec takes from each (OPINION): Sarathi's token budget sized from a TBT target (§5); vLLM's per-prompt
chunk cap and "reserve the whole prompt's KV" (Axon's `can_admit` already does the latter); FastServe's point that
*input* length is known and is the cheap SJF key (no predictor: Axon orders on prompt tokens, not predicted output);
Learning-to-Rank's starvation quantum as an **age bound** on the waiting queue; SGLang's prefill-first only for the
*idle* case (nothing to stall), which Axon already has (`IDLE_MS`). Not taken: output-length prediction (needs a
model per target; Axon serves arbitrary GGUFs), disaggregation, SGLang's decode stall for a new prefill.

---

## 3. Design goals and non-goals

Goals: (G1) a short prompt arriving while long prompts are being prefilled gets its first token within ~2 steps;
(G2) a wave of N simultaneous arrivals is admitted in O(prompt tokens / budget) steps, not O(N) steps; (G3) long
prompts cannot be starved by shorter ones past a bounded age; (G4) the step's wall time tracks a target TBT at the
*measured* cost of rows and chunk tokens on this GPU and this model, in both two-pass (main) and fused (#119) modes;
(G5) no change to any request's tokens, text, or KV bytes; (G6) every default reproduces today's behaviour on bella at
C ≤ 8 within noise, so the 4090 / H100 cells are the only ones that move.

Non-goals: per-client fairness (research-sched.md P3), prefix-aware (LPM) order, output-length prediction, preemption
policy changes (`victim`, `grow`, `preempt` untouched), the hybrid pass itself (#119).

---

## 4. Design (c): shortest-first admission, lanes scaled with slots

### 4.1 Vocabulary

- **short prompt**: `uncached(job) ≤ short_tokens`, where `uncached = prompt.len() − 1 − reuse` (reuse = the prefix a
  free slot holds, as `admit` computes it) and `short_tokens = max(--prefill-chunk, current token budget B)` (§5.3):
  "fits in one step's chunk". Default effective value on the H100 at nb=128 ≈ 500–700; on bella 256.
- **lane**: a long prompt in `pre`. `lanes = clamp(slots / 16, 2, 8)` (`--prefill-lanes N`, 0 = this rule). 2 at
  ≤ 32 slots (today's behaviour), 4 at 64, 8 at 128 (the review's ~C/16).
- **age bound**: `--admit-age-ms` (default 3000): a waiting job older than this is admitted before any younger one,
  FIFO among the aged, regardless of length (still subject to a free slot and pages).

### 4.2 Admission order (`serve.rs`, new fn `pick_admissions`)

Replace `first_fit` + single `admit` with:

```
loop {                                   // several admissions per iteration
    let now = Instant::now();
    let i = waiting.iter().enumerate()
        .filter(|(_, j)| fits_slot(j, b.ctx))
        .min_by_key(|(_, j)| admit_key(j, now, age_ms, |p| b.uncached(p)))?;   // None → break
    if !b.can_admit(&waiting[i]) { break; }   // the first refusal ends the round (see 4.4)
    admit(waiting.remove(i)) ...
}
```

`admit_key(job) = (aged: bool /*false sorts first*/, if aged { 0 } else { uncached(job) }, job.arrived)`.
Equivalent to FIFO when every candidate is aged or equally long; SJF otherwise; ties by arrival. `b.uncached(prompt)`
= `free_slot(prompt).map(|(_, common)| prompt.len() - 1 - common.min(prompt.len()-1))`. Cost: `best_slot` is O(slots ×
prefix) per candidate; with 128 slots and 128 waiting it is 16k `take_while` walks per iteration, each ending at the
first mismatch (a few tokens for unrelated prompts; up to 8k for a continued conversation). Bound it: compute
`uncached` only for the `2 × lanes` shortest-by-`prompt.len()` candidates and the aged ones; the rest use
`prompt.len()`. (OPINION: on the showdown's workload reuse is ~0 and `prompt.len()` alone would do; agent workloads
with cached 20k prefixes are where `uncached` matters, and there the candidate set is small.)

### 4.3 How many at once (`engine.rs can_admit`)

Replace `if self.pre.len() >= MAX_PREFILLING { return false }` with:

```
let b = self.budget_tokens();                      // §5.3: this step's chunk tokens (rows excluded)
let short = |rem| rem <= self.short_tokens();      // = b.max(policy.floor)
let (longs, short_rem) = self.pre.iter().fold((0, 0), |(l, s), f| {
    let rem = f.seq.hist.len() - 1 - f.fed;
    if short(rem) { (l, s + rem) } else { (l + 1, s) } });
let want = job.prompt.len() - 1 - reuse;
if short(want) {
    if short_rem + want > 2 * b { return false; }   // shorts already queued fill ~2 steps
} else if longs >= self.lanes() { return false; }
```

Then the existing pages test, unchanged (it already counts every prefilling and running request's reservation, so
admitting 8 long prompts at once reserves pages for all of them: correct on an 80 GB card, and on bella the pages
test, not the lane count, refuses — which is the right refusal).

Why `2 × b` for shorts (OPINION): `plan` serves shortest remainder first, so queued shorts finish within about two
steps; more than that only lengthens the youngest's wait without helping throughput. Why lanes at all when `plan` is
already shortest-first among prefills (OPINION): each admitted long prompt's pages are reserved against the pool in
`can_admit`'s `others` term even before they are fed, so unbounded lanes on a small card refuse the next short prompt
for pages it will never need; and a long prompt admitted into a slot it cannot progress in holds that slot (closed-loop
clients count slots as concurrency). On the H100 with 1M page-tokens the lane cap is slack at 8.

### 4.4 Pull depth (`serve.rs:1322-1330`)

Pull while `waiting.len() < b.slots.max(16)` (not `fit <= b.free()`): SJF must see the queue. `--max-queue` keeps
bounding the channel, so 503 behaviour is unchanged in total (channel + waiting ≤ max_queue + slots instead of
max_queue + free + 1; document the shift in `serve.rs`'s comment at `:1318-1321`). The cancelled-client `retain` at
`:1321` already covers the bigger `waiting`.

### 4.5 The idle step

Unchanged: `budget()`'s idle branch (IDLE_MS / IDLE_CHUNK) and `plan`'s whole-pass rounding stay. With 4.2–4.4, an
idle server receiving a 128-request wave admits every short that fits `2 × b` (b idle = IDLE_CHUNK = 4096 before a cost
is known → 104 × 60 = 6,240 tokens: all of them) plus `lanes` long ones in the first iteration. In two-pass mode each
short is still its own `feed_slot` pass (§1.2): 104 passes × t0 (ESTIMATED t0 ≈ 8–10 ms on the H100 from the C=1 short
TTFT of 14 ms, MEASURED, which includes HTTP and tokenize) ≈ 1 s for the wave, spread over the steps by §5.3's budget
(each step feeds as many shorts as fit `step_ms`). In fused mode (#119) the shorts share passes of up to `pass_max`
tokens: 6 passes ≈ 0.5 s. Either is ≥ 10× better than today's 14.7 s spread; fused is 2× cheaper in GPU time.

### 4.6 Starvation argument

Among waiting jobs: a long prompt waits behind shorter ones only until `admit_age_ms`, then goes first (G3). The bound
on its admission wait is therefore `age_ms + (time to a free lane)`; with lanes served shortest-remainder-first inside
`plan`, a lane frees at least every `longest-prompt / (budget per step)` steps. Inside `pre`, the existing STARVE = 4
rule still promotes a prompt passed over 4 steps in a row. Among shorts: FIFO by arrival at equal length; a short is
never blocked by lanes. Default 3,000 ms (OPINION): the showdown's TTFT SLO for a 7.6k prompt is 2,000 + 250 × 7.6 ≈
3,900 ms; an age bound of 3 s plus ~1 s of chunked prefill at today's rate lands at the SLO edge; lower it when the
prefill GEMM (item 2) gets faster. Make it a flag; test both 0 (pure FIFO, today's order) and 3000 in the A/B.

### 4.7 Flags (serve; `src/serve.rs Opts`, `src/cli.rs` help)

| flag | default | meaning |
|---|---|---|
| `--admit-order sjf|fifo` | `sjf` | `fifo` = today's `first_fit`, the rollback switch |
| `--admit-age-ms MS` | 3000 | a waiting request this old goes before shorter ones (0: strict shortest-first) |
| `--prefill-lanes N` | 0 = clamp(slots/16, 2, 8) | long prompts in prefill at once |
| env `AXON_ADMIT=fifo` | | same as `--admit-order fifo`, for A/B jobs that cannot change the command line |

`GET /slots` already shows each slot's request id and arrival (`slot_states`); add `waiting: [{id, prompt_tokens,
age_ms}]` so a queue inversion is visible, and `/metrics` gauges `axon_waiting_requests`, `axon_prefill_lanes_busy`.

---

## 5. Design (d): the step budget as tokens at the measured cost

### 5.1 The cost model (`engine.rs`, new `struct StepCost`)

Keep the shape the code already uses (a token at depth d costs `c · (d + DEPTH_K)` µs) and add the two things it
lacks: a per-pass fixed part, and a decode cost per row count that is *known for the step about to run*, not the last
one's wall time.

```
struct StepCost {
    /// µs per prompt token per (depth + DEPTH_K): EMA over feeds / fused passes with ≥ LEARN_MIN prompt tokens.
    tok: Option<f64>,
    /// Fixed µs of one pass (launches + one weight read): EMA over feeds < PASS_FIXED_MAX (64) tokens; seed 6000.
    pass_us: f64,
    /// Decode-only step ms by log2 bucket of rows (1,2,4,…,256): EMA α=0.25; None until seen.
    dec_ms: [Option<f64>; 9],
    /// Fused pass: µs per token-unit over rows and chunk tokens together (rows count at their own depth); EMA.
    fused_tok: Option<f64>,
}
predict_two_pass(rows, chunks) = dec(rows) + Σ_i (pass_us + tok·n_i·(d_i + n_i/2 + DEPTH_K))
predict_fused(rows, chunks)    = pass_us + fused_tok·( Σ_rows (d_r + DEPTH_K) + Σ_i n_i·(d_i + n_i/2 + DEPTH_K) )
dec(rows) = dec_ms[bucket(rows)] or, unseen, the nearest seen bucket scaled by rows ratio^0.5, or the last decode step.
```

Learning sites: `feed()` (two-pass; today's `cost` update at `:488-491` moves into `StepCost::learn_feed`, and feeds
< 64 tokens update `pass_us` instead of being ignored); the decode-only `step()` (`dec_ms[bucket]`); the fused
`step()` on the hybrid branch (`fused_tok`, when Σ chunk tokens ≥ LEARN_MIN). Nothing is measured that is not already
timed (`Instant::now()` around `feed_slot` / `step` exists on both branches).

Why not a single linear model over all tokens: MEASURED on the H100, decode rows at nb=128 cost 18.6 ms / (128 ×
(790 + 8192)) ≈ 0.016 µs per token-unit; prefill tokens cost 555 ms / (7651 × (3825 + 8192)) ≈ 0.006. The decode GEMM at
nb=128 runs 2.7× slower per token than prefill (the review's 56-CTA point), so one slope mispredicts one of the two by
2–3×. Two measured quantities, each used where it was measured, is the smallest honest model. Seeds before anything is
measured: today's constants (SEED_CHUNK 512, IDLE_CHUNK 4096) stay as the pre-measurement fallbacks.

### 5.2 Target

`--step-ms MS` (default **50**; `--prefill-ms` kept as an alias with a deprecation note): the step's target wall time
while requests decode = the TBT the server aims at. 0 = no time budget: exactly `--prefill-chunk` tokens (today's
semantics of `--prefill-ms 0`). `--step-tokens N` (default 0 = derive): pin the chunk tokens per step (vLLM's
`max_num_batched_tokens`), overriding the cost model; the A/B and the showdown config use it to hold tokens constant.

Why 50 and not today's 60 (OPINION, with the trade stated): the showdown SLO is ITL ≤ 50 ms and Axon's mix C=128 step
is 58 ms p50 — every request fails by the same 8 ms (review §2.2). On bella at C ≤ 8 the floor (256 tokens) binds
before the time budget does, so 50 vs 60 is invisible there (ESTIMATED: decode ~10 ms + 256 tokens at ~4k tok/s ≈ 74
ms either way). On the H100 at nb=128 it cuts the chunk from ≈ 570 to ≈ 430 tokens (−25 % prefill throughput while
decoding) and buys slo_ok. If Haseeb prefers throughput, the default stays 60 and the showdown config passes
`--step-ms 50`; the A/B runs both.

### 5.3 Deriving the budget

Two-pass (`budget()` replaces `(ms − decode_ms)` with the model):

```
fn budget_tokens(&self) -> usize {                          // chunk tokens this step may feed
    if let Some(n) = self.policy.step_tokens.filter(|&n| n > 0) { return n; }
    if self.policy.step_ms == 0 { return self.policy.floor; }
    if self.seqs.is_empty() { return idle as today (IDLE_MS at tok, or IDLE_CHUNK); }
    let left_us = (self.policy.step_ms as f64 * 1e3 - self.cost.dec(self.seqs.len()) * 1e3).max(0.0);
    // the quadratic plan() already solves per prompt at its depth; return the µs, let plan() share it:
    Budget::Micros(left_us, tok)   // as today, but `left_us` subtracts the modelled decode, and plan() subtracts
                                   // pass_us per prompt it feeds (so eight shorts don't look free)
}
```

`plan()` change (two lines): `spent = pass_us + c·n·(…)` and a later prompt is skipped when `share < floor` **or**
`left_us < pass_us + c·floor·(…)`. Floor and the no-sliver rules stay. Clamp: chunk ≥ `floor`, ≤ `pass_max` per prompt.

Fused (#119 `plan_fused`): replace `share = floor.max(rows+1).next_multiple_of(TILE) − rows` with

```
let b_total = match policy.step_tokens { Some(n) if n > 0 => n + rows,
    _ => solve fused_tok·(Σ_rows(d_r+DK) + n·(d̄+n/2+DK)) + pass_us = step_ms·1e3 for n, then n + rows };
let share = b_total.max(floor.max(rows + 1)).next_multiple_of(TILE).min(pass_max - rows - pre.len()).saturating_sub(rows);
```

Before `fused_tok` exists: `share = SEED_CHUNK` (512) prompt tokens, not 256 total. At nb=128 on the H100 this gives
ESTIMATED b_total ≈ 50 ms / (0.006 µs × 12k) ≈ 690 tokens → 560 chunk tokens (today's 570), i.e. the fused step keeps
today's prefill rate *and* drops the separate 18.6 ms decode pass. That is the +15 % the review attributes to (b); as
#119 stands (128 chunk tokens) it would not materialise at C=128 (§0 finding 2).

### 5.4 What (d) does and does not buy (ESTIMATED)

- Does: the step tracks the target instead of the target minus a stale `decode_ms` (the last decode step; at ramp,
  when rows change every step, it is wrong by the row delta); shorts are costed by passes, not tokens; the fused pass
  scales with rows and TBT. With `--step-ms 50` mix C=128 TPOT p50 58 → ≈ 48–50 ms, so slo_ok becomes a TTFT question
  (which (c) answers): ESTIMATED 0.2–0.3 → ≥ 0.7.
- Does not: raise prefill throughput at a fixed TBT. At 13.8k tok/s a 50 ms step holds ≈ 690 tokens; the 1152-token
  pass that fills the GEMM waves is an 83 ms step. The review's "wider chunks" and "≤ 50 ms TPOT" are in tension until
  item 2's GEMM is faster; (d) makes the knob honest, it does not remove the trade. Expect mix C=128 aggregate −0 to
  −10 % at 50 vs 60 from (d) alone, offset by (c)'s +5–12 %.

### 5.5 Flags

| flag | default | meaning |
|---|---|---|
| `--step-ms MS` | 50 | target wall time of a step while requests decode (chunk + rows, or the fused pass). 0: exactly `--prefill-chunk` tokens |
| `--prefill-ms MS` | alias of `--step-ms` | kept one release; help says so |
| `--step-tokens N` | 0 | pin chunk tokens per step (two-pass: per step across prompts; fused: prompt tokens beside the rows). Overrides the model |
| `--prefill-chunk N` | 256 | unchanged: the floor (two-pass) / the minimum pass incl. rows (fused, #119's meaning) |
| `/metrics` | | `axon_step_ms` histogram, `axon_step_prompt_tokens` histogram, `axon_step_cost_tok_us`, `axon_step_cost_pass_us`, `axon_step_cost_dec_ms{rows_bucket}` gauges: the model is observable, so the H100 trace (item 1 (a) 3) reads it off `/metrics` |

---

## 6. Bit-identity contract: paths touched, why none changes bits, and the tests that prove it

| path | change | why bits are unchanged |
|---|---|---|
| admission order (`serve.rs` loop) | which job gets which free slot, when | `best_slot` still picks per job; a different slot means a different reused prefix, and "serve's prefix reuse equals running the request alone" (START-HERE.md:75; `serve_identity`'s cache-reuse wave). `cached_tokens` in `usage` may differ; tokens and text may not |
| lanes (`can_admit`) | more prompts in `pre` at once | each prompt is fed into its own slot at its own positions (`feed_slot`); other prompts' chunks in the same step (two-pass: other passes; fused: other rows of one pass) are other slots' rows. Identity A (decode == batched prefill at any batch position, ADR §1.1; #119's `pool_hybrid_steps_match_feed_whole` with 2 prompts per pass) |
| budget (`plan`, `plan_fused`) | chunk sizes, how many passes | "its rows are the bits one feed of the whole prompt writes, whatever the chunk sizes" (engine.rs header); `serve_chunked_prefill_matches_solo` at budgets 1 / 16 / 64 / 256 (tests.rs:2254, 2307, 2426, 2635) |
| preemption pressure | more reserved pages → `grow` may preempt / unfeed more often | `preempt` refeeds "the same bits" (engine.rs:617-626); `serve_paged_pressure` (tests.rs:1946) covers it with `set_grow_tokens(0)` |
| sampling | none | per-request seeded `Sampler`, host pick; untouched |

Proof obligation for the builder (all exist; run them, don't write new identity tests): `serve_identity` /
`serve_identity_cpu` digests equal main's for the same load (the hybrid PR's `servab.sh` does this across builds: "1
distinct digest over 6 runs"); `serve_batched_matches_solo`; `serve_chunked_prefill_matches_solo` with
`--admit-order sjf` and `fifo`; `pool_hybrid_steps_match_feed_whole` on the hybrid branch; the md5 suite of `gen`
(untouched code, but the train gate requires it).

New **policy** tests (unit, no GPU), `engine.rs`:
- `admit_key_sjf_then_age`: 3 jobs (7,000 / 60 / 2,000 tokens, arrived in that order), age 0 → order 60, 2000, 7000;
  the 7,000 aged past the bound → it goes first; equal lengths → arrival order.
- `can_admit_lanes_and_short_budget`: with lanes = 2 and two long prefills, a 60-token job is admitted and a 3,000-token
  one is not; with Σ short remainder = 2·b, another short is not; a plain pool (no pages) keeps slot-bound behaviour.
- `plan_two_pass_charges_pass_fixed`: budget 50 ms, pass_us 8,000, tok such that a 60-token chunk costs ~1 ms: at most
  ⌊(50 − dec) / 9⌋ shorts are fed, not all of them.
- `plan_fused_budget_scales_with_rows` (hybrid branch): rows 128, fused_tok as the H100 numbers, step 50 ms → share
  ≥ 512 and share + rows ≤ pass_max; rows 4 → share rounds to the tile; `step_tokens` pins it.
- `step_cost_buckets`: `dec(rows)` returns the seen bucket, interpolates to an unseen one, falls back to the last step.

Integration (GPU, test model, existing harness style of `serve_paged_pressure`): `serve_sjf_short_ahead_of_long`:
4 decoders, admit a 6k prompt then a 60-token prompt 1 step later → the short's `Done` arrives before the long's first
token; with `--admit-order fifo` it does not. Asserts text equality against solo for both (the identity, again).

---

## 7. Expected gains against the review's estimates (ESTIMATED unless marked)

| metric (mix) | today (MEASURED) | review's estimate | this spec's estimate | how |
|---|---|---|---|---|
| short TTFT p50, C=128 | 2,361 (cell) / 2,867 (round 1 shorts) ms | < 300 ms | **100–300 ms** (wave 2 ≈ 1–2 steps = 60–120 ms; wave 1 ≈ 6,240 tokens of shorts through t0-bound passes ≈ 0.5–1 s spread, p50 ≈ 0.3 s; weighted p50 ≈ 0.2 s). vLLM-FP8 does 716 ms here | §4.5. The two-pass t0 per short is the uncertainty: if t0 ≈ 10 ms, wave-1 shorts p50 ≈ 0.5 s and the overall p50 ≈ 0.3 s — the estimate is on its edge **without** #119, comfortably under it **with** |
| short TTFT p90, C=128 | 12.9 s | — | ≤ 1 s | the long prompts no longer sit between a short and a lane |
| long TTFT max, C=128 | 15.7 s | — | 8–12 s (age bound 3 s + ~1 s/lane + SRF queue of 24 longs at ~8k tok/s ≈ 9 s for the last 7.5k) | §4.6. Not an SLO pass for the 7.5k's (3.9 s) — only item 2 fixes that |
| slo_ok, C=64 | 0.69–0.77 | ≥ 0.95 | **0.90–0.95**: 104 shorts pass TTFT (≤ 2 steps ≈ 70 ms; TPOT 33 ms already passes) → 104/128 = 0.81 from shorts alone; longs: the 2k/4k pass today at 0.83, the 6k/7.5k's in wave 1 (≈ 6 of 24) still miss | wave-1 longs decide the last 5 points |
| slo_ok, C=128 | 0.20–0.32 | — (review implied ×3–5 goodput at C=64) | ≈ 0.25 with `--step-ms 60` (TPOT 58 > 50 fails all); **≥ 0.7** with `--step-ms 50` (shorts pass both; longs' TTFT still long) | §0 finding 3 |
| aggregate tok/s, C=128 | 1,840–1,883 | item 1 total +25–40 % (incl. (b)) | (c): **+5–12 %** (waiting rows 30 → ~19 of 128 ⇒ decoding 86 → ~97; step time at 97 vs 86 rows ≈ equal, weight-bound); (d) at 50 ms: −0 to −10 %; net **+0 to +10 %** at 50, **+5 to +12 %** at 60 | §1.4 occupancy table |
| ITL p99, C=128 | 77.5 ms (best in the table) | keep | ≤ today's: the budget targets the step; shorts' passes are costed (§5.3) so eight of them don't blow a step | the A/B's gate |
| C ≤ 8 bella / avante | — | — | within noise: lanes = 2, floor binds, SJF has ≤ 2 candidates | G6 |

The review's +15 % "from the merged pass alone" (b) is reachable only if (d) sizes the fused pass (§5.3); with #119's
fixed 256 it is ESTIMATED negative at C=128 (§0 finding 2).

---

## 8. A/B on bella / avante, and the kill rule (written now)

### 8.1 Harness

Two existing drivers cover it; use both, interleaved per round, fresh server per arm, `same_text` checked across arms
as `serve_hybrid_ab` does:

1. **`serve_hybrid_ab` (tests.rs on the hybrid branch; port the "admit" and "loop" parts to main as `serve_sched_ab`
   if #119 is rejected)** — add a part **`mix`**: C clients, 2 requests each, the showdown's 104:24 short:long ratio
   scaled (C=16: 13 short + 3 long of 2k/4k/6k; C=32: 26 + 6), long prompts from a fixed corpus salted per round, 128
   output tokens; report per kind TTFT p50/p90/max, TPOT p50, ITL p99, aggregate tok/s, slo_ok at the showdown's SLO.
   Arms: `main`, `cand --admit-order fifo` (isolates (d)), `cand` (c+d, `--step-ms 60`), `cand --step-ms 50`. Env:
   `AXON_AB_PARTS=mix AXON_AB_C=32 AXON_AB_ROUNDS=3`.
2. **`sd_client.py cell --workload mix`** (axon-devstack `runbooks/h100sd/sd_client.py`) against a local `axon serve`
   — the showdown's own client and prompt set (`prompts-mix-v2.jsonl`; token counts are Qwen2.5's tokenizer, shared by
   the 0.5B / 1.5B / 7B). bella: 1.5B Q4_K_M, `--parallel 32 --ctx 8192`, C = 16 and 32, n = 2C, 2 rounds, `--out`
   per arm, then `sd_table.py` on `ttft_short_ms_p50, ttft_long_ms_p99, itl_ms_p99, agg_tok_s, slo_ok`. KV at 1.5B is
   ≈ 28 KiB/token (ESTIMATED from shapes); 32 × 8k = 7.3 GB worst case, paged so only what is held: fits 12 GB with the
   1 GB model, tight — fall back to `--parallel 24`. avante: 0.5B, `--parallel 32 --ctx 8192`, same cells (≈ 3 GB KV).
   7B on bella: C=8 and 16 with `--ctx 8192 --parallel 16` (the 60-token shorts and 2k longs only; the 6k's don't fit
   beside the model), the arm that matters for the timing claim.

Run under `axon-run --timing <host> --on <host> --big-models --vram N` as the hybrid PR did (its jobs show the
FOREIGN_GPU taint check; a tainted bella run is directional only). Cost: ~25 min per host per round-set; no pod.

Then **one** pod session (item 1 (a)'s budget, ~$2): the two Axon mix cells at C=64 and C=128 with the prebuilt
candidate, arms `--admit-order fifo` vs default, `--step-ms 50` vs 60 — four 2-minute cells. That is where the
numbers in §7 are tested; bella decides bits and direction, not magnitude.

### 8.2 Kill rule

Written before building; the candidate is (c)+(d) on main (or on #119 if merged), measured with harness 1 on bella 7B
C=16 mix and harness 2 on bella 1.5B C=32 mix, 3 interleaved rounds, medians, foreign GPU ≤ 5 %:

- **Bits (hard gate):** every identity test above passes; `serve_identity` digests equal main's; `same_text` = 1 across
  arms in every round. Any difference → stop, it is a bug, not a trade.
- **(c) stands if**, vs `--admit-order fifo` on the same build: short-prompt TTFT p50 under long load ≤ **0.5×** fifo's
  and ≤ 3 step times (ESTIMATED 3 × 60 ms on the H100; on bella 7B, 3 × ~75 ms); long-prompt TTFT max ≤ **1.5×** fifo's
  (the age bound holds); aggregate ≥ fifo − 3 % (noise). Fail on short TTFT → the pull depth or lanes are wrong,
  fix before re-measuring; fail on long TTFT max → lower `--admit-age-ms` default; fail on aggregate by > 5 % → revert
  to fifo default, keep the flag.
- **(d) stands if**, vs main at `--prefill-ms 60` / cand `--step-ms 60`: ITL p99 ≤ main + 10 %, aggregate ≥ main − 3 %,
  step wall p50 within ±15 % of the target when rows ≥ 16 (from `axon_step_ms`); and at `--step-ms 50`: ITL p50 ≤ 50 ms
  on the H100 cell with slo_ok at C=128 ≥ 0.5. Fail on the ±15 % → the cost model is miscalibrated (check `pass_us` /
  `dec_ms` on `/metrics`); fail on aggregate → keep the model, raise the default back to 60.
- **Together:** on the pod's mix C=128 cell, short TTFT p50 ≤ 0.5 s **and** aggregate ≥ today's 1,868 − 3 %; else the
  spec's §7 was wrong and the write-up goes to experiments-rejected.md with the trace.

---

## 9. Builder's checklist

Order: (d) first (it protects #119 at nb ≥ 64 and is pure `engine.rs`), then (c). Both land behind flags defaulting
on; `--admit-order fifo --step-ms 60 --prefill-lanes 2` reproduces main exactly (the rollback path without a revert).

| # | file | change | test |
|---|---|---|---|
| d1 | `src/serve/engine.rs` | `struct StepCost` (§5.1), replacing `cost: Option<f64>` and `decode_ms: f64` in `Batch`; `learn_feed`, `learn_decode`, `learn_fused`, `dec(rows)`, `predict_*` | `step_cost_buckets` |
| d2 | `engine.rs budget()` / `plan()` | `left_us` from `dec(rows)`; `pass_us` charged per fed prompt; skip rule (§5.3) | `plan_two_pass_charges_pass_fixed`; existing `plan_tests` unchanged (pass_us = 0 keeps them) |
| d3 | hybrid `plan_fused()` | `share` from the model / `step_tokens` (§5.3) | `plan_fused_budget_scales_with_rows`; `pool_hybrid_steps_match_feed_whole` re-run (bits) |
| d4 | `ChunkPolicy` | `ms` → `step_ms`, `+ step_tokens: usize` | |
| d5 | `src/serve.rs Opts`, `src/cli.rs` | `--step-ms` (+ `--prefill-ms` alias), `--step-tokens`; README serve section | `serve_cli_parses_step_flags` |
| d6 | `src/serve/ops.rs` | `axon_step_ms`, `axon_step_prompt_tokens` histograms; cost gauges | `/metrics` text test |
| c1 | `engine.rs` | `lanes()`, `short_tokens()`, `uncached(prompt)`; `can_admit` lanes + short budget (§4.3); `MAX_PREFILLING` → default of `lanes()` | `can_admit_lanes_and_short_budget` |
| c2 | `serve.rs` | pull depth (§4.4); `pick_admissions` loop with `admit_key` (§4.2) | `admit_key_sjf_then_age` (pure fn) |
| c3 | `serve.rs Opts`, `cli.rs` | `--admit-order`, `--admit-age-ms`, `--prefill-lanes`; `AXON_ADMIT` | |
| c4 | `ops.rs` | `/slots` `waiting` list; gauges | |
| c5 | `tests.rs` | `serve_sjf_short_ahead_of_long`; `serve_chunked_prefill_matches_solo` × {sjf, fifo}; `serve_sched_ab` mix part | |
| docs | `docs/log/2026-10-2x-sched-sjf.md`, roadmap item 1 (c)(d) status, `docs/serve-apis.md` flags | |

Gates before a PR (the train's usual): fast tier release on avante; debug-assertion `serve::` + `cuda::qwen2::slots::`
on bella; `serve::` with `--admit-order fifo` (the kill switch); md5 of `gen` vs base (untouched code; the train
requires it); `servab.sh` identity A/B across base / cand-sjf / cand-fifo; then §8.1's timing jobs.

Rollback: flags (above); or `git revert` of the two commits (d: engine.rs only; c: serve.rs + engine.rs `can_admit`).
`MAX_PREFILLING = 2` behaviour is `--prefill-lanes 2`; `--prefill-ms 60` behaviour is `--step-ms 60 --step-tokens 0`
with `pass_us = 0`, which the builder can keep reachable as `AXON_STEP_COST=legacy` for one release if the A/B needs
it (OPINION: not worth a flag unless the A/B disagrees with the model).

---

## 10. Risks, open questions, and what to measure first on the pod

1. **t0 (the pass's fixed cost) on the H100 is unmeasured.** It decides whether two-pass shorts in a wave take 0.5 s or
   1.5 s (§4.5). Item 1 (a)'s `pool_step_cost` run should add `feed_slot` at 16 / 64 / 128 tokens; `/metrics`'s
   `axon_step_cost_pass_us` reads it off a serve run otherwise.
2. **`uncached()` cost at 128 × 128** (§4.2): bounded by the 2 × lanes rule; if `/slots` shows the loop iteration
   exceeding ~1 ms of host time, order on `prompt.len()` only.
3. **Interaction with `grow`/`victim`:** 8 lanes of long prompts reserve pages in `can_admit`'s `others` term; on a
   12 GB card this refuses shorts earlier than today. Correct but visible: bella `serve_paged_pressure` must still pass,
   and the A/B's 1.5B C=32 cell watches `preempted` / `unfed` counters (`/metrics`), which should not rise vs main.
4. **Closed-loop artefact:** the showdown's client makes every slot's second request arrive when its first ends, so
   admission latency converts directly into lost rows. An open-loop (Poisson) run would show (c)'s TTFT gain and little
   throughput gain. State the loop type in every number.
5. **SJF favours the workload that is 81 % short.** On an all-long workload (classic: 519-token prompts) SJF = FIFO and
   the lane count is the only change; expect classic C=128 within noise of today (it is 2.7× behind for kernel reasons).
6. **The hybrid branch's `decode_ms` is stale** (updated only on decode-only steps); `StepCost::dec` fixes that by
   bucket, but under hybrid the fused model (`fused_tok`) is the live one and `dec` is used only for the two-pass
   fallback (MoE / MLA pools, `AXON_HYBRID=0`). Both must be learned on the branch.
7. **Not in scope but adjacent:** vLLM's `max_num_queued_tokens`-style early 503 when the prefill backlog exceeds
   `target_TTFT × prefill_rate` — Axon's `--max-queue` counts requests, not tokens. A follow-up once the H100 numbers
   exist; the research-sched.md P3 fairness work is where it belongs.

El Psy Kongroo.
