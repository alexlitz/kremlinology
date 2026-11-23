# Breakpoint semantics on open sliding-window and hybrid models (2026-11-17)

Does the "resume only at breakpoints / request ends" behaviour seen on Claude and GPT-5.6+ also appear on open models
whose architecture keeps state that can't be rebuilt from a slice of KV? Same four tests as the closed-model probe
(`kremlin cache-prefix`: repeat; first half shared then new text; extended prompt; follow-up turn), ~4.1k-token
prompts, 3 s apart, every request witnessed. 21 routes × 2 trials, then 10 informative routes × 4 more (6 total).
Evidence: `data/kremlinology-cache-semantics-open-v1`, `-open-v2`; figure `reports/cache-semantics-2026-11-17.png`
(pooled with the closed-model runs v1-v6). Spend $0.57.

Architecture from published configs (`reports/kv-cache-economics/fig1_bytes_per_token_32k.csv`, HF `config.json`):
sliding window + global (gpt-oss-120b: 128-token window; Gemma 4: 1,024), chunked (Llama 4: 8,192), linear attention
+ global (Qwen3.5, Qwen3.6: 3 linear : 1 full; Qwen3-Next), Mamba hybrid (Nemotron-3 Nano), dense controls.

## Results, pooled to 20 trials per cell (open-v1–v3; hits = trials reading > 10% of the reusable prefix)

| route | repeat | first half shared | follow-up | extended | reading |
|---|---|---|---|---|---|
| Llama 3.3-70B · CoreWeave (dense control) | 99% (14/20) | **99% (15/20)** | 99% (16/20) | 99% (15/20) | any prefix (when it hits) |
| gpt-oss-120b · Nebius (sliding) | 100% (20/20) | **100% (20/20)** | 100% (20/20) | 99% (20/20) | any prefix |
| Gemma 4 31B · DeepInfra (sliding) | 100% (20/20) | **6/20: 6/6 before 05:20 UTC, 0/14 at 08:14** | 99% (20/20) | 99% (18/20) | any prefix, then not (see below) |
| gpt-oss-120b · CoreWeave (sliding) | 99% (16/20) | **3% (0/20)** | 100% (18/20) | 2% (9/19) | request ends only (extensions sometimes) |
| gpt-oss-120b · Parasail (sliding) | 100% (20/20) | **3% (0/20)** | 100% (20/20) | 2% (6/20) | request ends only (extensions sometimes) |
| Gemma 4 26B · CoreWeave (sliding) | 99% (14/20) | **0% (0/20)** | 50% (10/20) | 0% (7/20) | request ends only; flaky cache |
| Qwen3.5-397B · Parasail (linear hybrid) | 76% (20/20) | **0% (0/20)** | 76% (20/20) | 76% (20/20) | last whole block of the previous request |
| Qwen3.5-397B · DeepInfra (linear hybrid) | 51% (18/20) | **0% (0/20)** | 51% (17/20) | 51% (18/20) | same, coarser |
| Qwen3.6-35B · CoreWeave (linear hybrid) | 76% (19/20) | **0% (0/20)** | 76% (15/20) | 76% (18/20) | same |
| Qwen3-Next 80B · Parasail (linear hybrid) | 92% (20/20) | **0% (1/20)** | 92% (20/20) | 92% (20/20) | same |

(Percentages: median share of the reusable prefix read from cache.) The 14-trial top-up (open-v3, 08:12–08:18 UTC,
witnessed, `kremlin cache-prefix --trials 14`) used the same prompt seeds as open-v2 for its first four trials,
three hours later; cache lifetimes on these endpoints are minutes, so the runs are effectively independent.

- **Gemma 4 31B on DeepInfra changed within the day.** Shared-half resumes hit 6 of 6 at 05:12–05:19 UTC and 0 of 14
  at 08:14 UTC, while repeats, follow-ups and extensions kept hitting (14/14, 14/14, 12/14). Either the serving stack
  moved to the windowed pattern (keep the sliding-window KV only near request ends) or the router's prefix affinity
  now sends a prompt that diverges halfway to a different replica. These tests can't tell the two apart.
- **Hybrid totals:** 1 hit in 80 shared-half trials (95% Wilson 0.2–7%); the one Qwen3-Next hit read 79%.
- **Sliding window, request-end-only stacks** (CoreWeave and Parasail gpt-oss, CoreWeave Gemma 26B): 0 of 60 on the
  shared half (0–6%). Extensions, which resume a few template tokens before the earlier request's end, hit
  sometimes (22 of 59).

Earlier 6-trial table and readings follow (unchanged conclusions except Gemma 4 31B on DeepInfra).

## Earlier results, 6 trials (shared half: share of the half reused; hits = trials reading > 10% of the reusable prefix)

| route | repeat | first half shared | follow-up | extended | reading |
|---|---|---|---|---|---|
| Llama 3.3-70B · CoreWeave (dense control) | 99% (4/6) | **99% (6/6)** | 99% (5/6) | 99% (4/6) | any prefix |
| gpt-oss-120b · Nebius (sliding) | 100% (6/6) | **100% (6/6)** | 100% | 99% | any prefix |
| Gemma 4 31B · DeepInfra (sliding) | 99% (6/6) | **94% (6/6)** | 99% | 99% | any prefix |
| gpt-oss-120b · CoreWeave (sliding) | 99% (5/6) | **2% (0/6)** | 99% (5/6) | 2% (1/5) | request ends only |
| gpt-oss-120b · Parasail (sliding) | 100% (6/6) | **3% (0/6)** | 100% (6/6) | 2% (2/6) | request ends only |
| Gemma 4 26B · CoreWeave (sliding) | 99% (4/6) | **0% (0/6)** | 99% (5/6) | 0% (2/6) | request ends only |
| Qwen3.5-397B · Parasail (linear hybrid) | 76% (6/6) | **0% (0/6)** | 77% (6/6) | 76% (6/6) | last whole block of the previous request |
| Qwen3.5-397B · DeepInfra (linear hybrid) | 51% (4/6) | **0% (0/6)** | 25% (3/6) | 51% (4/6) | same, coarser |
| Qwen3.6-35B · CoreWeave (linear hybrid) | 76% (6/6) | **0% (0/6)** | 77% (6/6) | 76% (6/6) | same |
| Qwen3-Next 80B · Parasail (linear hybrid) | 91% (6/6) | **0% (0/6)** | 92% (6/6) | 91% (6/6) | same |

- **Linear hybrids never resume at an arbitrary shared prefix:** 0 of 24 trials on the shared half (95% Wilson
  interval 0–14%) while repeats, follow-ups and extensions hit on every endpoint. The hit is always the same whole
  number of large blocks: 3,168 = 3 × 1,056 tokens (Qwen3.5/3.6 on Parasail and CoreWeave), 3,808 = 7 × 544
  (Qwen3-Next), 2,112 = 2 × 1,056 (DeepInfra): the last whole block before the end of the earlier request. The
  shared half (~2,060 tokens) spans at least one whole block yet gets nothing, so recurrent state is saved only at
  request ends, not at every block: an implicit breakpoint per request, as on GPT-5.6+.
- **Sliding window depends on the serving stack:** Nebius (gpt-oss) and DeepInfra (Gemma 4) reuse any prefix; CoreWeave
  (gpt-oss, Gemma 4) and Parasail (gpt-oss) resume only where a request ended (0 of 18 on the shared half, 95% CI
  0–18%; follow-ups 16 of 18). CoreWeave reuses arbitrary prefixes for its dense Llama (6 of 6) but not for its
  sliding-window models, so the architecture, not the provider's policy, decides.
- **No cache at all:** Qwen3.5-35B on Parasail, DeepInfra and Darkbloom; Nemotron-3 Nano (Mamba hybrid) on Nebius,
  despite a listed cached price; Gemma 4 26B on AkashML and Darkbloom. Llama 4 Maverick and Mistral Small on Parasail:
  too flaky to read (Parasail's dense Llama also missed plain repeats in this run).
- **Implication for the Anthropic argument:** resuming only at saved points is what open models with non-reconstructible
  state actually do, and what some engines do for sliding windows. Breakpoints are consistent with that design. It
  remains an inference: tiered storage or billing design could produce the same API.

## Closed-model cells topped up to ~20 (v4–v9: $1.26 + $2.30 + $0.29)
v7a/v7b/v8 brought the closed cells to 16–22 trials (v8 stopped at its $2.30 cap with 793 of 896 events); v9
(2026-11-18) filled the remaining deficits: every closed cell now has 19–22 (GPT-6.1 repeat: 19). No reading changed. Claude: shared half 0/85, extend
without a breakpoint 0/83, with a breakpoint 86/86, follow-up 83/83. GPT-5.6/6/6.1: shared half 0/61, extend 0/63,
repeat 59/59. The nanos (GPT-5/5.4) reuse any prefix (shared half 20/22).

### Earlier (v4-v6)
Every heatmap cell now has n ≥ 3 (Sonnet, Opus) or n = 4–6 (others). No cell changed its reading. Pooled: Claude
shared-half 0/17 (0–18%), extend-without-breakpoint 0/17; GPT-5.6/6/6.1 shared-half 0/17, extended 0/13.
