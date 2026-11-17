# Who can read your prompt cache? Cross-tenant and per-user cache scope (2026-11-13)

**How it was run:**
- Two paid probes through OpenRouter, $0.25 in total. Only our own fresh prompts were used (random nonce first, about 6k tokens of wikitext). Each trial writes with one identity and re-sends 4 s later with another. Cache reads come from OpenRouter usage fields.
- `experiments/cross_tenant_cache_probe_v1.py` → `data/kremlinology-cross-tenant-cache-v1` ($0.14, 96 requests).
- `experiments/cache_scope_probe_v1.py` → `data/kremlinology-cache-scope-v1` ($0.11, 104 requests). Single account; a cross-account mode is built in (`--second-key-env`) and not run yet.

## 1. Venice → Parasail: a cross-tenant cache test

Parasail sees Venice's traffic as one customer and our direct pinned requests (via OpenRouter) as another.

| model (Venice / Parasail tags) | V→P | P→V | P→P | V→V |
|---|---|---|---|---|
| Gemma 4 26B (bf16 / bf16) | **2/3** | **2/3** | 2/3 | 2/3 |
| GLM-5.3 Flash (— / fp4) | 0/3 | 0/3 | 3/3 | 3/3 |
| MiniMax-M3 (fp8 / fp8) | 0/3 | 0/3* | 3/3 | 3/3 |
| Kimi K2.6 (int4 / int4) | 0/3 | 0/3 | 0/1 (429s) | 3/3 |

\* 128-token hits on Venice are MiniMax's shared chat-template preamble, not our document.

- **Gemma 4 26B: the caches are one cache.** A prompt written through either route is read by the other at the same rate as a same-route repeat. So Venice serves this model from Parasail's replicas, and **Parasail's prefix cache is shared across its customers**: a prefix one tenant computed is reused, and billed at cache-read rates, for another.
- **GLM, MiniMax and Kimi: separate caches.** Each route caches for itself but never for the other. Venice runs (or buys) these elsewhere, so the Parasail relationship is per model. On Qwen3.5-397B (no caching on either side) the link was shown by bit-identical logprobs instead; see `reports/qwen35-dataset-tv-2026-11-13.md`.

## 2. Does the `user` field or `prompt_cache_key` isolate caches? (one account)

| target | same (control) | other `user` | `user` dropped | other `prompt_cache_key` |
|---|---|---|---|---|
| Claude Haiku 5.5 (breakpoint) | 2/2 | 2/2 | 2/2 | — |
| GPT-5.4 nano | 2/2 | 2/2 | 2/2 | **0/2** |
| GPT-6 Luna | 2/2 | 2/2 | 2/2 | **0/2** |
| DeepSeek-V4 Flash, DeepInfra | 1/2 | 2/2 | 2/2 | — |
| Llama-3.3-70B, CoreWeave | 1/2 | 2/2 | 2/2 | — |
| Llama-3.3-70B, AkashML | 1/2 | 0/2 | 0/2 | — |
| Llama-3.3-70B, Parasail | 0/2 | 0/2 | 1/2 | — |
| Gemma 4 26B, Parasail | 1/2 | 0/2 | 1/2 | — |

- **The `user` field scopes nothing.** On every route that caches reliably, a different or absent `user` hits as often as the control. On flaky routes (Parasail, AkashML), the control misses as often. Caches are at least account-wide.
- **OpenAI's `prompt_cache_key` splits caches:** 0/4 vs 4/4 for the same key. It's documented as a routing hint, so a different key reaches a different machine.

## Caveats

- 2–3 trials per cell; replica routing causes misses even on same-route repeats.
- Everything goes through one OpenRouter account. Upstream providers see all OpenRouter customers as one tenant. Whether *different OpenRouter customers* share upstream caches needs a second account (supported by `cache_scope_probe_v1.py --second-key-env`).
- Literature: Gu et al. (2025), "Auditing Prompt Caching in Language Model APIs", detect cross-user cache sharing with timing alone. Cached-token usage fields make it direct here.
