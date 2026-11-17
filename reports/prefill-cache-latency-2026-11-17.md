# Prefill speed, cache latency, cache age and the disaggregation question (2026-11-17)

All timing is witnessed (the relay's record timestamps): TTFT = first token-bearing event minus the end of the
request. OpenRouter, pinned endpoints, reasoning off. Scripts: `experiments/prefill_cache_sweep_v1.py`,
`prefill_cache_sweep_analysis_v1.py`, `cache_latency_grid_age_v1.py`, `cache_latency_analysis_v1.py`. Figures:
`reports/prefill-cache-sweep-2026-11-17.png`, `cache-latency-grid-2026-11-17.png`, `cache-latency-age-2026-11-17.png`.

## 1. Prefill sweep (four runs pooled: 2,518 calls, $6.13; 19–28 trials per size)

Unique prompts of 1k–32k tokens, sent cold and repeated 3 s later. DeepSeek-V4.1 Flash on 8 providers (sizes 1k–32k),
Kimi K2.6 on 3 (1k, 4k, 16k, 32k). Fit: cold TTFT = overhead + tokens / prefill speed.

| endpoint | says it disaggregates? | prefill fit (tok/s) | fixed overhead | cold → warm TTFT at 32k | first→second token gap vs size |
|---|---|---|---|---|---|
| Together (DeepSeek) | third-party claims only | ~116k | 0.15 s | 0.38 → 0.15 s | flat (−0.4 ms per 1k tokens) |
| CoreWeave (DeepSeek) | not stated | ~92k | 0.54 s | 0.88 → 0.57 s | flat (−1.6) |
| Fireworks (DeepSeek) | yes, [own docs](https://fireworks.ai/inference) | ~72k | 0.40 s | 0.85 → 0.44 s | flat (0.0) |
| DeepInfra (DeepSeek) | not stated | ~62k | 0.44 s | 0.84 → 0.61 s | flat (+0.2) |
| Baseten (DeepSeek) | yes, [own docs](https://www.baseten.co/resources/guide/the-baseten-inference-stack/) | ~51k | 0.32 s | 0.71 → 0.45 s | flat (0.0) |
| Parasail (DeepSeek) | not stated | ~44k | 0.35 s | 1.00 → 0.55 s | flat (+0.3) |
| Novita (DeepSeek) | not stated | ~30k | 0.81 s | 1.94 → 1.54 s | +0.7 |
| GMI Cloud (DeepSeek) | not stated | ~26k | 1.65 s | 2.88 → 2.44 s | flat (+0.1) |
| Moonshot (Kimi) | yes, [Mooncake](https://github.com/kvcache-ai/Mooncake) | ~3.6k | 2.25 s | 9.52 → 1.72 s | flat, but a constant 256–311 ms vs ~38 ms later steps |
| DeepInfra (Kimi) | not stated | ~23k | 0.74 s | 1.65 → 0.88 s | +1.2 |
| Parasail (Kimi) | not stated | ~16k | 0.57 s | 2.54 → 0.89 s | +2.4 |

- **DeepSeek-V4.1 Flash prefills so fast that a cache hit rarely matters below 16k tokens:** cold TTFT is 0.1–2.9 s
  at every size; warm/cold drops below 0.75 only at 16–32k (except Together, from 4k).
- **Kimi K2.6 at Moonshot is the opposite:** cold TTFT 1.4 s at 1k to 9.5 s at 32k (prefill ~3.6k tok/s); warm
  1.1–1.7 s, so a hit saves 82% at 32k.
- **CoreWeave reports no DeepSeek cache below 32k** (cached share 0 at 1k–16k, 1.0 at 32k), and its warm TTFT only
  improves at 32k: a minimum cacheable length somewhere between 16k and 32k.
- **No clean disaggregation signature.** The test assumed the first token comes from the prefill node and the KV must
  reach a decode node before the second, so the first gap would grow with the prompt. It doesn't on any
  self-described disaggregated provider; it grows slightly on Parasail and DeepInfra (Kimi) and Novita, which make no
  such claim. Most disaggregated engines stream the first token from the decode node, which puts the KV transfer
  inside TTFT, and layer-wise transfer overlaps it with prefill. Moonshot's constant 7× first gap is consistent with
  a fixed hand-off. Inconclusive by design, not by sample size.
- Several fits moved between the 6-trial and 20-trial pools (Fireworks ~125k → ~72k, DeepInfra ~90k → ~62k tok/s):
  a provider's prefill speed varies with load across a day, so these are snapshots.

## 2. Cache grid: TTFT at a given total size with part of it cached (grid runs v1–v5 pooled; 19–21 trials per point)

Request 1 writes C tokens; request 2 is a follow-up adding U new tokens. Compared with uncached prompts of the same
total size:
- **Kimi K2.6 · Moonshot:** 8.5k total with 8k cached 1.51 s (1.41–1.59) vs 3.48 s (3.27–3.75) for 8k uncached; 32.5k
  with 32k cached 1.94 s vs ~9.5 s uncached. Reported cached: 6,144 of 8k and 30,720 of 32k (whole blocks).
- **DeepSeek-V4.1 · Parasail:** 32.5k with 32k cached 0.54 s vs ~1.0 s uncached; 16.5k with 16k cached 0.51 s vs
  0.67 s; at 4k, within noise.
- **gpt-oss-120b · Nebius:** 16.5k with 16k cached 0.47 s (0.46–0.52) vs 0.62 s (0.60–0.68) for 16k uncached;
  smaller sizes within noise.

## 3. Cache age: same cached prefix, follow-up after 5 s to 20 min (age runs v1–v3 pooled; 18–40 per delay)

The 20-trial run (age-v2) sent 126 Moonshot writes within 4 minutes; Moonshot's slow prefill queued them behind each
other (cold TTFTs ~10 s, early follow-ups 9 s). Requests sent while 3+ of our own were waiting at the endpoint are
now excluded (`cache_latency_analysis_v1.py`), and Moonshot was re-run with writes spread over 30 minutes (age-v3,
140 pairs, $1.70).

| endpoint | no cache (unloaded cold sends) | follow-up 5 s – 5 min | 10 min | 20 min |
|---|---|---|---|---|
| Kimi K2.6 · Moonshot (8k cached) | 3.18 s (n = 158) | 1.51–1.98 s; 6,144 cached reported, billed at the cached rate (139 hits) | **1.76 s, 0 cached reported, billed in full** (0/40 report a hit) | **1.65 s, same** (0/22) |
| gpt-oss-120b · Nebius (16k) | 0.63 s | 0.48–0.53 s, hits (20/20 per delay) | 0.69 s, 19/20 hits | 1.08 s, 1/2 hits |
| DeepSeek-V4.1 · DeepInfra (16k) | 0.62 s | 0.52–0.61 s, hits (20/20) | 0.57 s, 20/20 | 0.50 s, 2/2 |

- **Moonshot after 10 minutes (replicated, n = 62):** every follow-up at 10 or 20 minutes reported 0 cached tokens and
  was billed at the full input price ($0.0081 vs $0.0032 for a reported hit). Yet 52 of the 62 answered faster than
  the fastest of 158 unloaded cold sends of the same prompt size (2.95 s); median 1.74 s vs 3.18 s cold and 1.56 s
  for reported hits. Under exchangeability that many beats are p ≈ 1e-40. The prefix is still being reused,
  probably from a slower tier (Mooncake keeps KV in DRAM/SSD), but reported and billed as uncached.
  - Caveats: OpenRouter's usage mapping, not Moonshot, could drop the count; not checked against Moonshot's own API.
- **Nebius:** hits up to 10 minutes; at 10–20 minutes hits get slower (0.69–1.08 s vs 0.63 s cold), a hint of a
  slower tier. The 20-minute point has only 2 trials.
- **DeepInfra:** hits for the whole 20 minutes with no slowdown (20-minute point: 2 trials).

