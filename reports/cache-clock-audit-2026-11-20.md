# The miss distribution as a reference clock for auditing cache reports (2026-11-20; no API calls)

Idea (user's): a cache miss is compute-bound prefill, so its TTFT should be comparatively deterministic per endpoint
and prompt size, while a hit's time depends on lookup, tier and placement. Fit each endpoint's cold (miss)
distribution, then classify every warm trial by the clock alone: below the cold 10th percentile (and, stricter, below
the fastest cold) = clock-hit; inside the distribution = clock-miss. Against the provider's own reported cached
tokens this gives per-trial confusion counts: reported-hit-but-clock-miss = **phantom-hit candidate** (counter not
corroborated; where discounted, customer-favoured), reported-miss-but-clock-hit = **hidden-hit candidate** (billed in
full, provider-favoured).

Script `experiments/cache_clock_audit_v1.py`; numbers `reports/cache-clock-audit-2026-11-20.json`; input the per-trial
honesty cells (~20 cold/warm pairs per endpoint, 5.4k-token Llama / 7k DeepSeek / 10k gpt-oss prompts, 3 s apart).

## Findings

**1. The determinism premise holds only where load noise is low.** Cold-TTFT CV ranges 0.10 (Near AI) to 1.29
(Bedrock); DeepInfra's DeepSeek colds have CV 1.02 (queueing spikes). The clock reference is trustworthy exactly
where the endpoint is quiet.

**2. Where it is trustworthy, it reproduces the known cases and clears the clean ones:**

| endpoint | cold CV | verdict from the clock |
|---|---|---|
| DeepInfra Llama turbo | 0.78 | **8/18 hidden hits**, all 8 strictly below the fastest cold (0.87 s): unambiguous |
| GMI Cloud DeepSeek | 0.21 | **12/21 phantom-hit candidates** (warm hits land inside the tight cold distribution) |
| Parasail (both), CoreWeave fp4, Groq, AkashML fp8 | 0.14–0.37 | counter matches the clock on every trial |
| DeepInfra DeepSeek | 1.02 | 12/15 phantom candidates, but the cold reference itself is unstable — suggestive, not conclusive |

**3. Real hits are *more* variable than misses, in the predicted direction** — the user's mechanism shows up
directly: CoreWeave fp16 cold CV 0.27 → hit CV 1.76; Crusoe 0.23 → 1.41. Hit time is tier/lookup-dependent; miss time
is compute. (AkashML fp8 is the reverse, 0.56 → 0.07, because its colds carry the load noise and its rare hits are
clean.)

**4. Limitation — the classifier needs effect size.** Where true speedups are small (Nebius, Baseten, Near AI at
these prompt sizes: warm only 10–20% under cold), a p10-of-cold threshold flags 2–6 of ~20 real hits as phantom:
false positives, not findings. The clock audit is decisive only when the prompt is big enough that a real hit must
clear the miss distribution by a wide margin.

## Conclusions
- The method is a per-trial audit tool that needs no cooperation from the provider, and it cleanly separates the two
  known misreporting directions plus the honest counters.
- GMI Cloud is the strongest phantom-hit candidate (tight cold reference, 12/21 overlap, and it *discounts* what it
  reports at 0.23×); DeepInfra's DeepSeek endpoint leans the same way but needs a quieter reference.
- To use this at the small-speedup endpoints, rerun at 16–32k prompts where a real hit must save ~1 s: that is the
  proposed discrimination run (~$1–1.50, witnessed).

## First-party clock tests: Anthropic and OpenAI (2026-11-22; witnessed, $1.23)

`experiments/clock_cache_probe_v1.py`, 20 trials per size, unique prompts, cold then identical warm 3 s later
(`haiku-clock-v1`, `nano-clock-v1`; both sessions audit clean).

| endpoint | size | cold → warm (median, client clock) | reported hits | warm below fastest cold | warm ÷ cold cost |
|---|---|---|---|---|---|
| Anthropic Haiku 4.5 (breakpoint) | 8k | 0.98 → 0.93 s | 20/20 | 4/20 | 0.08 |
| | 32k | 1.42 → 1.09 s | 20/20 | **18/20** | 0.08 |
| OpenAI GPT-5.4 nano | 8k | 1.07 → 1.02 s | 19/20 | 1/20 | 0.18 |
| | 32k | 1.38 → 1.18 s | 20/20 | **17/20** | 0.12 |

Both documented caches show up in the clock at 32k and not at 8k, where the saving is inside the noise. Compare
first-party DeepSeek, whose hits are indistinguishable from misses up to 64k and save ~30% only at 128k–256k (a disk
cache; `deepseek-cache-reconciliation-2026-11-20.md`).

## The hidden cache at DeepInfra's Llama turbo endpoint lasts seconds (2026-11-22; witnessed, $0.10)

`experiments/cache_ttl_generic_v1.py`, 5.4k-token unique prompts, identical repeat after a delay, 17–20 trials per
delay; no request reported any cached tokens. Repeats faster than the fastest of 93 cold requests: 9/20 at 2 s,
1/18 at 60 s, 0/19 at 5 min, 0/17 at 15 min, 0/19 at 30 min. Unreported, billed in full, and gone within a minute.
