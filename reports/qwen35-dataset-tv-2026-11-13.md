# Qwen3.5-397B across providers: dataset-average first-token TV and reseller detection (2026-11-13)

**How it was run:**
- Run: `data/kremlinology-qwen35-dataset-tv-v2`, config `configs/bench/qwen35-dataset-tv-v2.json`. It stopped at 3,421 of 4,000 requests on an unaccounted-cost check; cost $0.127.
- Prompts: the Llama-70B campaign's 400 (200 CodeAlpaca, 200 wikitext).
- Settings: first token, top-5 logprobs (Alibaba and GMI reject more), `reasoning.enabled=false`, 2 repeats.
- Reference: Alibaba, the first party.
- Analysis: `experiments/dataset_tv_analysis_v1.py` → `derived/dataset_tv_analysis.json`.
- GMI Cloud was dropped after v1: it ignored `max_tokens=1` and ran into the 30 s deadline.

## Average TV vs Alibaba (allowance 0; assessable prompts; betting 95% CI)

| target | code | wiki |
|---|---|---|
| Alibaba vs itself | 4.8% [2.9, 10.0] | 6.3–10.4% [4.6, 15.4] |
| Novita | 4.4% [3.0, 8.4] | 6.9–10.2% [5.1, 14.2] |
| Parasail fp8 | 6.7% [5.1, 11.0] | 8.8–13.0% [6.7, 17.1] |
| Venice | 7.2% [5.4, 11.5] | 8.5–12.5% [6.6, 16.5] |
| StreamLake | 9.1% [7.1, 13.4] | 14.8–18.9% [11.8, 23.5] |

## Bitwise identity (same prompt and repeat, all top-5 logprobs within 0.002 nats)

| pair | identical |
|---|---|
| Parasail–Venice | **424 / 586 (72%)** |
| Alibaba–Novita | **108 / 566 (19%)** |
| every other cross pair | 0–4 of 560–600 |
| self, repeat 0 vs 1 | Parasail 70%, Venice 65%, StreamLake 38%, Novita 20%, Alibaba 18% |
| Venice r0 vs Parasail r1 | 221 / 297 (74%) |

## Findings

1. **Venice serves Qwen3.5 from the same backend as Parasail.** Its logprobs match Parasail's bit for bit as often as Parasail matches itself, and match nobody else.
2. **Novita matches Alibaba at their self-match rate.** That suggests the same stack or a shared upstream. Its TV is also indistinguishable from Alibaba's own noise.
3. **Alibaba is nondeterministic.** Repeats agree exactly only 18% of the time, and the self-TV is 5–10%. Provider differences have to be judged against that floor, not against zero.
4. **StreamLake differs most,** at about 2× the floor. Its distributions are sharper; in the smoke test, "leve" sat at −7.0 against −5.6 at Alibaba.

## Venice in detail (OpenRouter generation records, free lookups)

| evidence | Parasail | Venice |
|---|---|---|
| upstream request ID | `chatcmpl-` + 32 hex (vLLM format), 692/692 | same format, 657/679 (97%) |
| | | `gen-<unix>-<20 chars>`, 18/679: OpenRouter generation IDs timestamped at our request's second |
| median latency (OpenRouter) | 293 ms | 384 ms (+90 ms, one relay hop) |
| exact logprob match with Parasail | 70% self (repeat vs repeat) | 72% same repeat, 74% across repeats |
| when Parasail's two repeats differ | — | Venice equals one of them 115/132 (87%) |
| list price (input / output per M, OpenRouter catalog 2026-11-13) | $0.50 / $3.60 | $0.75 / $4.50 (+50% input, +25% output) |
| prompt token counts | — | identical to Parasail on 588/589 prompts |

- **Venice lands on Parasail's numeric states**, including when Parasail disagrees with itself. It never produces a third variant, as an independent deployment of the same recipe would sometimes do.
- **The 18 OpenRouter-ID responses never match Parasail.** 5 match Alibaba exactly and 3 match Novita (the Alibaba-like stack). Their latency is 0.5–3.7 s, like Alibaba. 4 of them returned no logprobs.
- So Venice appears to relay about 97% of these requests to Parasail and the rest back through OpenRouter, where they reach an Alibaba-family endpoint. That's a request from OpenRouter, to Venice, back into OpenRouter.

## Caveats

- The upstream ID is what OpenRouter recorded from Venice's response. A `gen-` ID could in principle be Venice's own format. Our requests carry matching timestamps, and those responses carry Alibaba-family numerics, which makes a relay the simplest explanation.

- Top-5 only, so the upper bounds are wide.
- Bitwise matches show shared numerics, not a commercial relationship.
- Cause (quantization, kernels, template) isn't attributed.
