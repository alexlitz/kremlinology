# Do DeepSeek endpoints' reported cache rates match the clock? (2026-11-20; no API calls)

Prompted by the observation that DeepSeek's reported cache hit rates look suspiciously high. DeepSeek has no
first-party endpoint on OpenRouter (checked 2026-11-20: every DeepSeek model there is a reseller), so this reconciles
what the resellers **report** against what the **clock** corroborates, using the prefill-sweep data (DeepSeek-V4.1
Flash, 8 providers, 1k–32k unique prompts re-sent 3 s later, ~20 trials per size, witnessed timing).

**Metric.** From each endpoint's own cold fit, TTFT = overhead + tokens / prefill_rate; the prefill-attributable time
at size S is S/rate. The **recovered fraction** of a reported hit is (cold − warm) ÷ (S/rate): the share of the
recompute a full hit should save that actually disappeared. Reported cached share is the provider's own field. At
1–4k the denominator is tens of milliseconds, so only 8k–32k readings count.

## Reported share vs recovered fraction (8k–32k, where measurable)

| endpoint (DeepSeek-V4.1 Flash) | reported share | recovered fraction, 16k → 32k | prefill fit | reading |
|---|---|---|---|---|
| Together | 0.99–1.00 | 0.81 → 0.83 | 116k tok/s | counter corroborated |
| Fireworks | 0.98–1.00 | 0.81 → 0.92 | 72k | corroborated |
| CoreWeave (hits only at 32k) | 1.00 | 0.88 | 92k | corroborated |
| Parasail | 0.99–1.00 | 0.49 → 0.63 | 44k | about half recovered |
| Baseten | 0.97–1.00 | 0.71 → 0.41 | 51k | about half |
| DeepInfra | 0.99–1.00 | 0.30 → 0.44 | 62k | a third to a half |
| Novita | 0.99–1.00 | 0.19 → 0.38 | 30k | a quarter to a third |
| **GMI Cloud** | 0.99–1.00 | **0.16 → 0.36** | 26k | **a sixth to a third** |
| (Kimi contrast: Moonshot first-party) | 0.91–0.99 | 1.02 → 0.88 | 3.6k | corroborated |

## Reading

- **Everyone reports ≈ 1.00; the clock recovers between 0.16 and 0.92 of the claimed saving.** Same model, same
  prompt sizes, same claim — six-fold spread in corroboration. If the counters were honest and caches healthy, the
  recovered fraction should approach 1 everywhere (Together, Fireworks and CoreWeave show it does).
- Two explanations fit "reported 1.00, recovered 0.2–0.4", and they are not separable at these sizes:
  1. **Optimistic counter** (bookkeeping marks the prefix cached; the serving replica recomputes) — matches the
     phantom-hit evidence from the clock audit (`cache-clock-audit-2026-11-20.md`: GMI 12/21, DeepInfra-DeepSeek
     12/15 reported hits inside the cold distribution).
  2. **A cache whose read costs half or more of recompute** — a real hit served from a slow tier (host memory, disk,
     cross-node). Then the report is true and the cache is nearly worthless for latency while still earning the
     discounted hit price.
  Either way the customer's mental model — "cached_tokens ≈ 1 means I skipped the compute" — is wrong at these
  endpoints, in the direction that makes the hit-rate numbers look impressive.
- **The suspicion extends to small prompts:** these endpoints report 0.75–0.97 cached share even at 1k–2k, where the
  whole prompt is one or two blocks and hits barely matter. A counter that reports near-total caching everywhere,
  corroborated or not, is what "suspiciously high hit rates" looks like from outside.
- Why DeepSeek models specifically: V3.x/V4.1 use MLA with tiny per-token KV and (V4.1) a small active-parameter MoE,
  so prefill compute is genuinely cheap (fits: 26–116k tok/s). Cheap prefill shrinks the absolute saving a real hit
  can show, which is exactly the cover an optimistic counter (or a slow-tier cache) needs — the discrepancy only
  becomes unambiguous at large sizes.

## What would settle it
- **64k–128k prompts** (compute 1–2.5 s): if GMI/Novita/DeepInfra still recover ≤ 0.4 while reporting 1.00, the
  read-cost excuse dies (no tier reads that slowly and still serves) and the counter is optimistic. ~$1.50 witnessed.
- **First-party DeepSeek** (api.deepseek.com): the one platform with documented per-request hit/miss token fields —
  needs a DeepSeek key (stdin, never stored), ~$1 for the same reconciliation. DeepSeek first-party is also where the
  public "90%+ hit rate" impressions come from, and the same recovered-fraction test applies directly.

Data: `reports/prefill-cache-sweep-2026-11-17.json`; scripts `experiments/prefill_cache_sweep_analysis_v1.py` (fits),
this reconciliation computed from the same. Evidence dirs on the release archive.

## First-party DeepSeek, measured (2026-11-20, later the same day; witnessed, $0.20)

The user's key made first-party DeepSeek routable on OpenRouter (BYOK: `deepseek/deepseek-v4.1-flash` pinned to
provider `deepseek`; the endpoint is absent from the public catalog). The prefill sweep was run against it: unique
prompts 1k–64k, 10 trials per size, cold then identical warm 3 s later, all 140 requests witnessed
(`data/kremlinology-prefill-sweep-v5`, audit passes). **First-party prefill is the fastest we have measured:
~122k tok/s, fixed overhead 0.81 s.**

| size | cold → warm TTFT (median) | reported cached (warm) | warm ÷ cold billed cost | recovered fraction |
|---|---|---|---|---|
| 16k | 0.89 → 0.95 s | 15,616 of 15,787 | 0.04 | ≈ 0 (0/10 below the fastest cold) |
| 32k | 0.93 → 0.99 s | 31,360 of 31,562 | 0.03 | ≈ 0 (0/10) |
| 64k | 1.20 → 1.23 s | 62,848 of 63,093 | **0.02** | ≈ 0 (3/10, medians equal) |

- **The reported hit rate is ~100% on every repeat (block granularity 64 tokens), the bill follows it — and the
  clock shows no benefit whatsoever.** At 64k a full recompute is ~0.52 s of the 1.20 s cold TTFT (the fit is
  serial: overhead + tokens/rate), so a true full-cache hit should have landed near 0.81 s; warm sits at 1.23 s.
- So on first-party DeepSeek, `cached_tokens` functions as a **billing category, not a latency claim**: repeats are
  billed at 2–4% of cold input cost, yet are not measurably faster. Whether KV is physically reused is undetectable
  from outside — which is precisely the sense in which the reported rates are "suspiciously high": nothing in the
  observable behaviour corroborates them except the invoice.
- Customer impact is nonetheless favourable (the discount is real and large). The finding matters for anyone using
  cached_tokens as a *measurement* of reuse — as we do throughout — and it caps what ping/keep-alive economics can
  buy on first-party DeepSeek at latency: nothing, only the discount.
- Caveats: 10 trials per size, one route (OpenRouter → DeepSeek), one day; if their pipeline overlaps prefill with
  queueing the saving could hide in TTFT, but the cold fit shows no such overlap (cold ≈ overhead + compute
  serially). Warm requests were sent 3 s after cold, same prompt, `max_tokens: 1`.

## Could the claimed hits be impossible? Prefix-semantics probe (2026-11-20; witnessed, $0.02)

Six one-token requests on ~30k-token prompts, provider `deepseek` first-party (`deepseek-impossible-v1-session`,
audit passes):

| request | prompt | reported cached | billed |
|---|---|---|---|
| A fresh unique | 30,209 | **0** | full |
| A2 identical repeat, 3 s later | 30,209 | 30,075 (99.5%) | discounted |
| B same body, **first ~140 tokens changed** | 30,300 | **0** | full |
| C same body, last ~140 tokens changed | 30,291 | 29,952 (all but the tail) | discounted |
| D a different fresh prompt | 30,191 | 0 | full |

The counter behaves exactly like an honest block-prefix cache: a fresh prompt claims nothing, a changed head kills the
whole prefix (as it must: prefix KV depends on everything before it, so those tokens cannot legitimately be reused),
a changed tail keeps everything before it, and billing follows the true prefix match in every case. **No impossible
claims.** What remains anomalous is only the clock: a prefix-consistent, block-granular, correctly-billed hit that
saves no measurable time at any size to 64k — i.e. the reuse is real as *bookkeeping* and as a discount, and
undetectable as compute.

## Multi-turn, empty-think replay, and in-flight writes (2026-11-20; witnessed, $0.05)

Note: v4.1-flash is a thinking model by default (a `reasoning` field on replies; at `max_tokens: 8` the visible
content is empty). ~16k-token unique prompts, one-token-ish replies, provider `deepseek`
(`deepseek-multiturn-v1-session`, audit passes). The B1 prefix is 14,976 cacheable tokens in every row:

| request | reported cached |
|---|---|
| cold, then identical repeat | 0, then 14,976 |
| + assistant turn ("OK") + ask | 14,976 |
| + two assistant turns | 14,976 |
| + empty `<think></think>` inside a replayed assistant turn | 14,976 |
| identical repeat of the two-assistant-turn conversation (twice) | 14,976 |
| repeat with temperature 2 and a seed | 14,976 |
| **two truly-simultaneous first-ever requests** on a fresh prompt | **both 0, both full price**; a third, after the write commits, hits |

- **The empty-thinking-tag issue does not bite, because the cache never extends through assistant turns at all.**
  The hit always ends at the last block boundary of the final user-side content; the trailing ask, the template tail
  and every replayed assistant turn — with or without an empty think tag — miss, even on identical repeats. So the
  big user prefix keeps its discount turn over turn, while replayed assistant content (on this thinking-by-default
  model, potentially long stripped reasoning) is re-prefilled and billed as a miss every turn. The tag makes it no
  worse: the region does not hit either way.
- **No in-flight over-claim:** concurrent first requests report nothing cached and pay full price until a write has
  actually committed. Combined with the earlier probes (cold 0, changed head 0, changed tail partial), every
  impossibility test we can construct comes back honest.
- Sampler parameters (temperature, seed) do not affect the hit, as expected for prefill-level caching.
- The one anomaly remains the original one: all of this correct, prefix-true, concurrency-safe bookkeeping buys zero
  measurable latency at any size we tested.

## Correction and completion (2026-11-23): larger sizes, 20 trials, and what kind of cache it is

Witnessed top-ups: first-party at 128k and 256k (20 trials each; `prefill-sweep-v6`, `-v6b`), DeepInfra and Novita at
16/32/64k (20 trials; `-v7`, `-v7b`). GMI Cloud no longer serves this model (404 on every request), so the strongest
phantom-hit suspect could not be retested. Analysis: `experiments/prefill_recovery_analysis_v1.py`
(`reports/prefill-recovery-2026-11-23.json`): recovered fraction with bootstrap 95% intervals.

| endpoint | size | cold → warm TTFT | recovered fraction (95% CI) | warm below fastest cold |
|---|---|---|---|---|
| DeepSeek first-party | 64k (n=10) | 1.20 → 1.23 s | −0.06 (−0.42 to 0.62) | 3/10 |
| | 128k | 1.64 → 1.39 s | **0.29 (0.04–0.56)** | 0/20 |
| | 256k | 2.44 → 1.90 s | **0.31 (0.23–0.48)** | 18/20 |
| DeepInfra fp8 | 16k / 32k / 64k | 0.61→0.53 / 0.80→0.56 / 1.31→0.66 s | 0.31 / 0.51 / **0.68 (0.55–0.82)** | 8, 17, 18 of 20 |
| Novita fp8 | 16k / 32k / 64k | 1.22→1.09 / 1.76→1.43 / 2.79→2.11 s | 0.26 / 0.31 / **0.33 (0.23–0.40)** | 3/20, 10/19, 12/16 |

- **"Invisible as speed" was true only up to 64k and is withdrawn as a general statement.** At 128k–256k a
  first-party hit removes about 30% of the recompute time, and at 256k 18 of 20 warm requests beat every cold one.
  The benefit is real, small, and proportional to size.
- **It is a disk cache, as DeepSeek documents** ("Context Caching on Disk ... Each user request will trigger the
  construction of a hard disk cache"; cleared "usually within a few hours to a few days"; "best-effort"). A hit's
  cost grows 4.66 µs per cached token (95% CI 4.09–5.07) against 6.9 µs per token for a miss: reading the cache costs
  about two-thirds of recomputing it, on the fastest prefill we have measured (~144k tok/s). Hence correct prefix
  bookkeeping, a 2–4% bill, and little time saved. See `cache-tier-sniff-2026-11-23.md`.
- **Lifetime:** 33 prompts written first-party and re-sent byte-for-byte 34 hours later: 0/33 hits (witnessed,
  `dsttl-v1b-session`). The 3-second repeats hit 100%. A 20-trial curve from 30 minutes to 24 hours is running.
- **Resellers split by mechanism.** DeepInfra's recovered fraction climbs with size (0.31 → 0.68): a fixed cost per
  hit on top of a resident cache. Novita's stays flat near 0.3 at every size (hit slope 18.6 µs/token vs 31.6 for a
  miss): a read that scales with size, like first-party's. Neither is a phantom counter; both are real caches whose
  hits are much less than free.
