# How fast do providers load cached KV? Bounds from time to first token (2026-11-17)

No new requests: re-analysis of the witnessed prefill sweep (v1–v4), cache grid (v1–v5) and cache age (v1–v3) runs.
Script: `experiments/cache_tier_throughput_v1.py`; numbers: `reports/cache-tier-throughput-2026-11-17.json`.

**Idea.** A cache hit's time to first token contains the time to bring the cached prefix to the GPU. So:
- **Fast hits, marginal rate:** going from C1 to C2 cached tokens can't add less time than the extra loading, because
  the other costs (overhead, attention over a longer context) only stay flat or grow. Hence
  load rate ≥ (C2 − C1) / ΔTTFT. One-sided 95%: the bootstrap 95th percentile of ΔTTFT (medians).
- **A slower tier, sandwiched** (follow-ups ≥ 10 min after the write that still reuse the prefix, vs follow-ups at
  5 s–5 min with the same prefix):
  - upper: rate_slow ≤ C / (TTFT_slow − TTFT_fast): the slow hit does the fast hit's work plus a longer load;
  - lower: rate_slow ≥ C / (TTFT_slow − floor), floor = 5th-percentile TTFT of the endpoint's smallest uncached
    requests (overhead every request pays).
- **Bytes need an assumption:** rate × KV bytes per token. That depends on the KV precision (fp8 halves it) and, for
  sliding-window models, on whether the sliding layers' cache is kept for every token or only for the last window.
  Both products are given.

## Fast hits (minimum load rate, one-sided 95%)

| endpoint | cached tokens | ΔTTFT (median; 95th pct.) | ≥ tokens/s | ≥ GB/s at bf16 KV |
|---|---|---|---|---|
| Together, DeepSeek-V4.1 Flash | 0.9k → 31.5k | +0.05 s (+0.07) | 448k | (V4.1 KV size unpublished) |
| DeepInfra, DeepSeek-V4.1 Flash | 0.9k → 31.5k | +0.02 s (+0.14) | 223k | |
| Fireworks, DeepSeek-V4.1 Flash | 0.9k → 31.4k | +0.03 s (+0.19) | 162k | |
| Parasail, DeepSeek-V4.1 Flash (sweep / grid) | → 31.5k | +0.32 / +0.20 s | 81k / 116k | |
| Baseten / Novita / GMI, DeepSeek-V4.1 Flash | | +0.14 / +0.56 / +0.47 s | 68k / 48k / 36k | |
| CoreWeave, DeepSeek-V4.1 Flash | 1.9k → 31.5k | −0.14 s | no bound (larger hits were faster) | |
| DeepInfra, Kimi K2.6 | 1.0k → 31.5k | +0.31 s (+0.41) | 75k | 5.3 (MLA, as K2.5) |
| Parasail, Kimi K2.6 | 1.0k → 31.6k | +0.47 s (+0.64) | 48k | 3.4 |
| Moonshot, Kimi K2.6 (sweep / grid) | → 30.7k | +0.64 / +0.43 s | 18k / 26k | 1.3 / 1.8 |
| Nebius, gpt-oss-120b (grid) | 3.9k → 30.9k | +0.31 s (+0.40) | 68k | 2.5 (windows only) / 5.0 (all sliding-layer KV) |

These are floors: an HBM-resident hit loads nothing, so a small ΔTTFT mostly measures attention over the longer
context. The fastest stacks (Together, DeepInfra, Fireworks) show no measurable cost of a 30k-token hit.

## The slower tier, sandwiched

| endpoint | prefix | fast hits → hits ≥ 10 min | reported? | load rate | at bf16 KV |
|---|---|---|---|---|---|
| Nebius, gpt-oss-120b | 15.5k | 0.50 → 0.74 s (+0.23) | yes, 22/22 | 27k–158k tok/s | 1.0–5.8 GB/s (windows only); 2.0–11.6 GB/s (all sliding-layer KV) |
| Moonshot, Kimi K2.6 | 6.1k | 1.56 → 1.74 s (+0.17) | no, 0/62 | ≥ 7k tok/s (upper side not significant) | ≥ 0.5 GB/s |
| DeepInfra, DeepSeek-V4.1 Flash | 15.7k | 0.57 → 0.56 s | yes, 22/22 | no slowdown: still the fast tier at 20 min | — |

- **Withdrawn (2026-11-18): the Nebius row is not a slower tier.** The 20-minute top-up (age-v4, 18 trials, run the
  next day) hit 19/20 at 0.52 s, as fast as at 5 s; the 10-minute slowdown came from one run and was most likely load
  at that time. With no tier change the sandwich bounds nothing. The row is kept for the record.
- **Moonshot's unreported reuse** (0 cached tokens reported, billed in full) moves at least 0.5 GB/s; its upper side
  can't be pinned because the 0.17 s slowdown is within noise at 95%.

## Can timing tell whether sliding-window caches are kept (and loaded) in full?

Structurally, yes for some stacks already (`cache-semantics-open-models-2026-11-17.md`): a sliding-window model can
only resume at an arbitrary shared prefix if the sliding layers' cache is kept for every position there. Nebius
(gpt-oss) and DeepInfra (Gemma 4 31B, until midday) do that; CoreWeave and Parasail resume only where a request ended,
consistent with keeping just the windows. Keeping everything costs 2× per token for gpt-oss-120b (73,728 vs 36,864
B) and about 6–8× for Gemma (Gemma 3 27B: 507,904 vs 81,920 B; Gemma 4 31B: 901,120 vs 81,920 B per token beyond
the window).

Whether a *load* moves all of it can't be read from one model's numbers, because bytes per token and bandwidth
enter as a product. It can with a full-attention control on the same provider and tier: the difference between
slow-tier and fast-tier per-token slopes is pure load time, and its ratio between a sliding-window model and the
control cancels the bandwidth (and, if both use the same KV dtype, the precision). Predicted ratios vs Qwen3-235B
(192,512 B/token) at Nebius: gpt-oss-120b 0.19 (windows only) vs 0.38 (all); Gemma 3 27B 0.43 vs 2.64. Proposed run:
`cache_tier_swa_v1` (not yet run): those three Nebius endpoints, prefixes of 8k and 32k tokens, follow-ups after 5 s
and 10 min, 20 trials per cell, writes paced to avoid self-queueing; ~$1.6 estimated.

## Caveats
- Lower bounds assume the non-load part of a hit doesn't shrink as the cached prefix grows; upper bounds assume the
  slow hit does at least the fast hit's work. Layer-wise loading that overlaps compute keeps both valid.
- Rates are per request, not the tier's aggregate bandwidth.
- KV precision is unknown at every provider; DeepSeek-V4.1's KV size isn't in our configs.
