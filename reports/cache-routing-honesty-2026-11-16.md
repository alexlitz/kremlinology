# Cache-aware routing and cached-token honesty (2026-11-16)

Two probes through the witness relay (every exchange attested; audited and accepted on the hub):
- gpt-oss-120b routing + 20-endpoint honesty run: https://alexlitzenberger.com/kremlinology/runs/f30459ff73c39797e69c/
  (168 calls, $0.128)
- DeepSeek-V3.2 routing run: https://alexlitzenberger.com/kremlinology/runs/61175138068b9e03de82/ (40 calls, $0.018)

Scripts: `experiments/cache_routing_honesty_v1.py`, `experiments/cache_routing_honesty_analysis_v1.py`.
Data: `reports/cache-honesty-2026-11-16.json`, `reports/cache-routing-deepseek-2026-11-16.json`.

## A. Does OpenRouter route a repeated prompt back to the provider holding its cache?

Unique ~2,650-token prompts, each sent 4 times, 2 s apart, with default (unpinned) routing.

| model | prompts | providers used | repeats on the first request's provider | expected from the overall split |
|---|---|---|---|---|
| gpt-oss-120b | 12 | Crusoe 27, Google 18, Venice 2, Near AI 1 | 15/36 (42%) | 46% |
| DeepSeek-V3.2 | 10 | DeepInfra 24, GMI Cloud 16 | 18/30 (60%) | 52% |

- No stickiness on either model. On DeepSeek, counting only requests where exactly one provider held the prompt,
  the next request went to it 8 of 15 times (53%) against a 52% chance rate.
- Repeats that land on a holder get the cache: median 2,608 (gpt-oss) / 2,304 (DeepSeek) cached tokens; repeats
  that switch get only the chat-template header (gpt-oss: 48–73 tokens) or nothing (DeepSeek: 0).
- With only two providers in rotation (DeepSeek), most later repeats hit anyway, because both providers end up
  holding the prompt: the prompt is cached (and the first-time prefill paid) twice.
- Block sizes differ: GMI Cloud cached exactly 2,048 of ~2,700 tokens; DeepInfra 2,560–2,624.
- Confound for gpt-oss: default routing mostly used providers with no cache discount (Crusoe lists its cached
  price equal to its input price) or no cache (Google), so a price-aware router had little reason to stick.
  DeepSeek's providers do discount (GMI $0.022 vs $0.209/M; DeepInfra $0.13 vs $0.26/M) and still no stickiness.

## B. Do reported cached tokens match the time to first token?

Per pinned endpoint, 3 trials: a unique long prompt (Llama ~5.4k, DeepSeek ~7k, gpt-oss ~10k tokens), then the
identical request 3 s later. TTFT = first token-bearing event minus end of request, from the witness's record
timestamps. "Faster" = warm TTFT < 0.8 × cold. "Reported" = > 50% of the prompt reported as cached.

| endpoint | TTFT cold → warm (median) | trials faster / reported | warm cost ÷ cold cost | reading |
|---|---|---|---|---|
| CoreWeave, gpt-oss-120b | 0.74 → 0.26 s | 2/2 · 2/2 | 1.00 | real, no discount (listed cached price = input price) |
| CoreWeave, Llama 3.3-70B | 0.35 → 0.15 s | 3/3 · 3/3 | 1.00 | real, no discount (same) |
| Parasail, Llama 3.3-70B | 1.26 → 0.48 s | 2/3 · 2/3 | 0.50 | real, discounted |
| Parasail, gpt-oss-120b | 0.70 → 0.52 s | 2/3 · 3/3 | 0.56 | real, discounted |
| Crusoe, gpt-oss-120b | 0.75 → 0.49 s | 2/2 · 2/2 | 1.00 | real, no discount (listed) |
| Nebius, gpt-oss-120b | 0.52 → 0.43 s | 2/3 · 3/3 | 0.67 | real, discounted |
| Near AI, gpt-oss-120b | 1.51 → 1.14 s | 3/3 · 3/3 | 0.51 | real, discounted |
| GMI Cloud, DeepSeek-V3.2 | 1.62 → 1.32 s | 1/3 · 3/3 | 0.11 | reported and discounted; speed-up not visible |
| DeepInfra, DeepSeek-V3.2 | 0.73 → 1.03 s | 0/3 · 2/3 | 0.51 | reported and discounted; warm not faster |
| Baseten, gpt-oss-120b | 0.25 → 0.20 s | 0/3 · 2/3 | 1.00 | reported; prefill too fast to see a gain |
| AkashML, Llama 3.3-70B | 0.78 → 2.12 s | 1/3 · 1/3 | 1.00 | one hit in three (asynchronous caching, as before) |
| Groq, Llama 3.3-70B | 0.56 → 0.96 s | 0/3 · 0/3 | 1.00 | no cache, although a cached price ($0.295/M) is listed |
| DeepInfra Llama (turbo); SambaNova, Bedrock, Vertex, AkashML (gpt-oss) | | ≤ 1 faster, 0 reported | 1.00 | no cache |
| Baidu (DeepSeek), SambaNova (Llama), DeepInfra (gpt-oss) | | | | rate-limited (429), no data |

- **No hidden caching:** no endpoint was faster on repeats in 2+ trials without reporting a cache. The margin case
  (cache used, full price charged, not reported) did not appear.
- **"Cached" doesn't always mean cheaper:** CoreWeave, Crusoe and Baseten report hits but bill them at full price,
  as their listings say (cached price = input price). Caching there saves time, not money.
- **Reported-but-not-faster** (DeepInfra and GMI on DeepSeek-V3.2) errs in the customer's favour: the discount is
  applied. A cache tier slower than recomputing a 7k prompt, or DeepSeek's sparse attention making prefill cheap,
  could explain it; not identified.

## Caveats
- 2–3 trials per endpoint; single-trial speed-ups (AkashML, Bedrock, Vertex gpt-oss) are within noise.
- Short prompts on fast hardware leave little prefill to save (Baseten).
- Routing: 12 and 10 prompts on two models, one afternoon; OpenRouter's routing may differ by model, load or
  account.

## Update 2026-11-17: ~20 trials per endpoint (pooled)

Two more honesty runs (`data/kremlinology-cache-honesty-v2`, `-v3`, same protocol, witnessed) pooled with the
first: 1,031 calls, $0.84. Pooled analysis: `reports/cache-honesty-pooled-2026-11-17.json` (per-trial cells included).
With ~20 trials, a few "faster" repeats happen by queueing noise alone, so the hidden-cache rule is now statistical
(in `cache_routing_honesty_analysis_v1.py`): among trials that report no cache, either warm beats cold significantly
more than half the time (one-sided sign test, p < 0.01, median ratio ≥ 1.25), or significantly many warm trials are
faster than the fastest cold trial (floor test: P = C(n,m)/C(2n,m) under exchangeability, p < 0.01). The floor test
catches a hit/miss mix (a cache on some replicas only) that a median hides.

| endpoint | n | reported hits | TTFT cold → warm on reported hits | warm ÷ cold cost on hits | reading |
|---|---|---|---|---|---|
| CoreWeave, gpt-oss-120b | 20 | 20/20 | 0.67 → 0.26 s | 1.00 | real, no discount (as listed) |
| CoreWeave, Llama 3.3-70B | 21 | 19/21 | 0.34 → 0.15 s | 1.00 | real, no discount (as listed) |
| Crusoe, gpt-oss-120b | 19 | 18/19 | 0.58 → 0.32 s | 1.00 | real, no discount (as listed) |
| Nebius, gpt-oss-120b | 20 | 20/20 | 0.58 → 0.46 s | 0.67 | real, discounted |
| Parasail, gpt-oss-120b | 21 | 21/21 | 0.65 → 0.52 s | 0.56 | real, discounted |
| Near AI, gpt-oss-120b | 21 | 21/21 | 1.23 → 0.98 s | 0.51 | real, discounted |
| Parasail, Llama 3.3-70B | 21 | **10/21** | 1.35 → 0.58 s | 0.50 | real when it hits; misses on about half the repeats (replicas) |
| AkashML, Llama 3.3-70B / gpt-oss | 21 / 20 | 4/21 / 3/20 | 0.80 → 0.35 s (Llama) | 0.50 / 1.00 | occasional hits (asynchronous caching, as before) |
| GMI Cloud, DeepSeek-V3.2 | 21 | 21/21 | 1.70 → 1.53 s | 0.23 | reported and discounted; speed-up small |
| DeepInfra, DeepSeek-V3.2 | 20 | 15/20 | 0.73 → 0.70 s | 0.51 | reported and discounted; no speed-up |
| Baseten, gpt-oss-120b | 20 | 11/20 | 0.23 → 0.18 s | 1.00 | reported; prefill too fast to see much |
| **DeepInfra, Llama 3.3-70B (turbo)** | 18 | **0/18** | — | 1.00 | **hidden cache: 8 of 18 repeats answered in 0.31–0.44 s, faster than every one of the 18 first requests (≥ 0.87 s); floor test p = 0.0014. Never reported, billed in full** |
| Groq, Llama 3.3-70B | 20 | 0/20 | — | 1.00 | no cache (0/20 faster than the cold floor), although a cached price is listed |
| SambaNova, Bedrock, Vertex (gpt-oss) | 21 each | 0, 0, 1 | — | 1.00 | no cache (floor test p = 0.5) |
| Baidu (DeepSeek), SambaNova (Llama), DeepInfra (gpt-oss) | | | | | rate-limited (429) throughout, no data |

- **This revises "no hidden caching".** With 3 trials, DeepInfra's Llama turbo endpoint showed ≤ 1 fast repeat; with
  18, it is bimodal: about 4 in 9 repeats are 2-3x faster than any first request and none report cached tokens or
  get a discount. That is the margin case (cache used, not reported, billed in full), seen 3 s after the first
  request. The miss half is consistent with the repeat landing on a different replica (as Parasail's hits do).
  Moonshot's 10-minute case (prefill-cache-latency-2026-11-17) is the other instance.
- **Parasail Llama's hits are real but replica-dependent**: 10/21, each one 2x faster and billed at half price.
- The remaining verdicts stand, now on ~20 trials each.
