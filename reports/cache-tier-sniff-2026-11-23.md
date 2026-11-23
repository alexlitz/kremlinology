# Sniffing cache tiers from outside (2026-11-23)

Question (user's): can a tiered cache, or a cache hierarchy, be detected through the API? Two signatures can be read
from data already collected (no new requests); a third needs a new probe, written but not yet run.

Scripts: `experiments/cache_tier_sniff_v1.py` (signatures 1 and 2), `experiments/prefill_recovery_analysis_v1.py`,
`experiments/cache_tier_doubletap_v1.py` (signature 3, not run). Numbers: `reports/cache-tier-sniff-2026-11-23.json`,
`reports/prefill-recovery-2026-11-23.json`. All timing is witnessed TTFT except where "client clock" is stated.

## Signature 1: what a hit costs per cached token

A hit served from GPU memory costs almost nothing per cached token. A hit that has to be brought in from host memory,
disk or another node costs time in proportion to its size. So fit, per endpoint, the slope of hit TTFT against cached
tokens (reported hits, 1k to 32k–256k, 3 s after the write) and compare it with the slope of miss TTFT against prompt
tokens (the recompute cost).

| endpoint | miss, µs/token | hit, µs/token (95% CI) | hit ÷ miss | hit path |
|---|---|---|---|---|
| Together (DeepSeek-V4.1 Flash) | 8.3 | 1.2 (0.5–2.2) | 0.15 | ~resident |
| Fireworks | 15.9 | 1.8 (0.2–7.8) | 0.11 | ~resident |
| DeepInfra fp8 (to 64k) | 14.2 | 1.7 (−0.1–3.2) | 0.12 | resident |
| CoreWeave | 11.2 | 3.1 (−9.9–9.0) | — | not distinguishable from resident |
| Baseten | 17.0 | 4.3 (1.4–15.1) | 0.25 | unclear |
| Parasail | 23.5 | 9.5 (8.2–11.6) | 0.40 | **loaded: ~106k tok/s** |
| GMI Cloud | 44.9 | 19.9 (8.0–34.5) | 0.44 | **loaded: ~50k tok/s** |
| Novita (to 64k) | 31.6 | 18.6 (16.8–20.7) | 0.59 | **loaded: ~54k tok/s** |
| **DeepSeek first-party (to 256k)** | 6.9 | **4.66 (4.09–5.07)** | **0.67** | **loaded: ~214k tok/s** |
| Moonshot (Kimi K2.6) | 268.6 | 20.8 (−9.9–69.5) | 0.08 | ~resident (wide CI) |
| DeepInfra fp4 / Parasail int4 (Kimi K2.6) | 31.3 / 69.3 | 10.1 (6.7–13.4) / 14.0 (9.3–21.1) | 0.32 / 0.20 | loaded: ~99k / ~71k tok/s |

- **Two populations.** At Together, Fireworks and DeepInfra a hit is nearly free per token: the cache is where the
  model is. At DeepSeek first-party, Novita, GMI and Parasail **a hit costs 40–67% of a recompute per token**, with
  tight intervals at DeepSeek and Novita. That is what a cache read from a slower tier looks like, and it is why
  those endpoints' "recovered fraction" stays near 0.3–0.6 at every size instead of approaching 1.
- **DeepSeek says so itself.** Its docs call the feature "Context Caching on Disk" ("Each user request will trigger
  the construction of a hard disk cache"; cleared "usually within a few hours to a few days"; best-effort). Our
  numbers are what that predicts: correct prefix bookkeeping, a 2–4% bill, and a per-token read cost two-thirds of
  recomputing on what is the fastest prefill we have measured (~144k tok/s). If V4.1's KV is the size of V3.2's
  (78 KB/token at bf16; an assumption), 214k tok/s is ~17 GB/s: NVMe-array or host-memory territory, far below HBM.
- **Caveat: hit slope is not only load.** Tokenising, templating and prefix-hashing the prompt are also paid per token
  on a hit. Together's 1.2 µs/token bounds what an efficient front end costs; a slow front end could explain part of
  a 4–20 µs/token hit slope. Age dependence (signature 2) and the double-tap (signature 3) separate the two, because
  front-end cost does not depend on how old the cache is.

## Signature 2: hit latency that steps up with the cache's age

Same prompt, same reported hit, different age. Retention probes (~1.5k-token prompts, client clock, reported hits
only; n in brackets) and the 8k/16k age runs:

| endpoint | miss | hit at 2–5 s | 1 min | 5 min | 15–30 min | reading |
|---|---|---|---|---|---|---|
| CoreWeave, Llama 3.3-70B | 0.22 s | 0.17 [16] | 0.17 [16] | 0.18 [15] | 0.17 [14], 0.17 [15] | **one tier for its whole 30-minute life**, then gone |
| Nebius, gpt-oss-120b | 0.35 | 0.26 [20] | 0.28 [19] | **0.39 [11]** | 0.36 [7] | fast for ~1 min, then hits at miss speed: a second, slower tier (the 16k age runs disagree by day; unresolved) |
| Chutes, Gemma 4 31B | 1.68 | 1.51 [20] | **1.80 [12]** | 1.74 [5] | — | fast tier ≤ seconds; by 1 min a "hit" costs a miss |
| DeepInfra turbo, Gemma 4 31B | 0.29 | 0.25 [18] | 0.43 [5] | — | — | same pattern, thin |
| Moonshot, Kimi K2.6 (8k) | 3.18 | 1.56 | 1.5–2.0 | 1.5–2.0 | **1.74, unreported, full price** | a second tier at 10 min that is slower and billed as a miss |
| DeepInfra, DeepSeek-V4.1 (16k) | 0.62 | 0.58 | 0.52 | 0.60 | 0.57–0.59 | one tier for 20 min |

- **Three hierarchies visible:** Moonshot (reported fast tier → unreported slower tier at ~10 min); Nebius and Chutes
  (a seconds-to-a-minute fast tier, then reported hits that are no faster than recomputing); and single-tier caches
  with a hard expiry (CoreWeave 30–60 min, DeepInfra ≥ 20 min).
- **DeepInfra's hidden Llama cache is the shortest tier of all** (turbo endpoint, 5.4k-token prompts, 17–20 trials
  per delay, witnessed): repeats beat the fastest of 93 cold requests 9/20 times at 2 s, 1/18 at 60 s and 0/55 from
  5 to 30 min. It lives for seconds, is never reported and is billed in full.
- Thin cells: several 5–15 minute cells have 4–12 hits at 1.5k tokens, where a tier's load time is small.

## Signature 3 (designed, not run): the double-tap

Write a prompt; at age A read it (R1); two seconds later read it again (R2). If R1 − R2 > 0 and grows with prompt
size, R1 paid a load that R2 did not: the entry had been demoted and the hit promoted it, and S ÷ (R1 − R2) is the
load rate of the tier that age maps to. Everything is within one prompt, so replica placement and load mostly cancel,
and front-end cost (paid by both) drops out. Stepping A from seconds to hours locates the tier boundaries; at Moonshot
it also asks whether a hit on the unreported tier re-enters the reported one.

`cache_tier_doubletap_v1.py`: DeepSeek first-party (16k, 64k; 5 s to 2 h), Nebius, Moonshot, Chutes, DeepInfra and
OpenAI nano at 20 trials per cell, about $3.1; CoreWeave and Anthropic's 1-hour cache (is the 2×-priced 1 h cache a
slower tier?) about $4.1 more. **Blocked: the API key's spending limit has ~$0.2 left** after the 24-hour TTL run.

## Related results from the same batch
- **First-party cache lifetime:** 33 prompts written to DeepSeek and re-sent byte-for-byte 34 hours later (witnessed):
  0/33 hits. A 20-trial curve (30 min to 24 h) is running (`kremlinology-dsttl-v3`, resumable).
- **Anthropic and OpenAI first-party caches are latency-real at 32k** (20 trials each, witnessed): Haiku 4.5 with a
  breakpoint 1.42 → 1.09 s, 18/20 warm requests faster than the fastest cold; GPT-5.4 nano 1.38 → 1.18 s, 17/20. At
  8k neither is distinguishable (4/20, 1/20): the clock test needs size.
