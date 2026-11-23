# Provider TV from samples alone: an adaptive estimator, validated, then run on hosted Llama 3.3-70B (2026-11-17)

**Question:** can you measure how far two providers' next-token distributions are apart (total variation, the
yardstick of the logprob campaigns) when one of them returns no logprobs?

**Scripts:** `experiments/adaptive_tv_v1.py` (estimator and local validation), `adaptive_tv_hosted_v1.py`,
`adaptive_tv_hosted_v2.py` (hosted runs; `--fixed` top-ups), `adaptive_tv_hosted_analysis_v1.py` (final numbers from the raw samples).
Data: `reports/adaptive-tv-2026-11-17.json` (+ `-records.json`), `reports/adaptive-tv-hosted-2026-11-17.json`;
evidence `data/kremlinology-adaptive-tv-hosted-v1`, `-v2` (every one-token sample kept). Hosted spend $2.07 for
136,318 samples (plus a few cents of logprob requests); evidence also in `-hosted-v3`, `-v3b`.

## The estimator

Per provider, in rounds of 500 one-token samples (half each setting):
- **plain** samples at temperature 1 give absolute shares, with an OTHER bucket for everything rarely seen;
- **hot** samples at temperature 2 with top-K 10 reach the tail; a pooled multinomial fit over both settings gives the
  named tokens' logit gaps, which are turned into absolute probabilities using the plain samples' mass on the named
  set.

TV is computed over the union of named tokens plus OTHER (a lower bound on the full TV, like the logprob
comparisons, which also see only the top tokens). A bootstrap that resamples every run gives the interval; the point
estimate is bias-corrected (plug-in TV is biased upward: two identical distributions still differ by noise). Rounds go
to whichever provider's own noise contributes more of the TV variance, until the 95% interval is narrower than ±0.5
points or the budget runs out.

**Interval near zero.** The basic bootstrap interval collapses to [0, 0] when two providers agree (it did on one hosted
prompt, which also stopped sampling early). The upper bound is now never below the percentile bound, which is biased
upward and so conservative (`adaptive_tv_v1.interval`). All numbers here use it.

## 1. Local validation (ground truth known)

Llama-3.2-3B fp32 logits as provider A on 60 dataset prompts; B is either the bf16 model (a real, small difference) or
fp32 logits plus Gaussian noise (a provider-sized difference). Budget 8,000 samples per provider per prompt.

| pair | true TV (median) | estimate error (median) | bias after correction | 95% CI half-width | coverage of the true value |
|---|---|---|---|---|---|
| fp32 vs fp32 + noise | 8.5% | 0.48 points | +0.17 | ±1.7 | 93% (56/60) |
| fp32 vs bf16 | 1.1% | 0.70 points | +0.29 | ±1.8 | 87% (52/60) |

- **Provider-sized differences (5–10%) are measured to about half a point** with honest intervals.
- **Differences near 1% can only be bounded**: the bias correction can't fully remove noise when the true gap is the
  size of the noise, so the interval undercovers somewhat (87%). At this budget, "under ~3%" is the resolution.
- The ±0.5-point target was reached on 1 of 120 prompt pairs within 8,000 samples per provider; the rest ran to budget.

## 2. Hosted: Llama 3.3-70B, CoreWeave fp16 vs Parasail fp8 vs DeepInfra turbo (no logprobs)

10 near-tie prompts from the Llama dataset campaign, 4,000 samples per provider per prompt (one-token requests,
OpenRouter, pinned endpoints). DeepInfra's shared upstream pool returned 429 for ~3 hours during the main run; its
6 missing prompts were sampled the next day (`-hosted-v3`, fixed 4,000 per prompt) and pooled with the other two
providers' samples of the same prompts. code-0172 had stopped early (1,000/500 samples) on the old interval and was
topped up to 4,500/4,000 (`-hosted-v3b`).

| prompt | CoreWeave ~ Parasail (sampled) | logprob path, same pair | CoreWeave ~ DeepInfra | Parasail ~ DeepInfra |
|---|---|---|---|---|
| code-0094 (run 1) | 4.3% (1.5–6.9) | — | 7.7% (4.7–10.7) | 11.0% (7.8–14.3) |
| code-0094 (run 2) | 3.7% (0.8–6.7) | 3.8% | | |
| wiki-0333 | 14.0% (11.4–16.5) | 8.7% | 13.5% (10.9–16.3) | 1.4% (0–4.2) |
| code-0034 | 6.7% (4.6–9.2) | 14.0% | 6.3% (4.2–8.4) | 12.5% (10.2–14.8) |
| code-0171 | 6.4% (4.3–8.4) | 2.6% | 7.8% (5.9–9.8) | 2.0% (0.6–3.8) |
| code-0005 | 5.6% (3.0–8.1) | 2.5% | 0.3% (0–3.3) | 6.4% (3.9–9.1) |
| code-0172 | 3.9% (1.9–6.5) | 12.7% | 0.5% (0–3.1) | 4.8% (2.4–7.1) |
| code-0116 | 8.2% (5.6–10.9) | 5.8% | 2.5% (0.1–5.6) | 10.5% (7.7–13.2) |
| code-0137 | 8.2% (5.4–10.9) | 5.3% | 2.6% (0–5.2) | 5.7% (2.9–8.7) |
| code-0071 | 1.0% (0–4.7) | 2.5% | 4.7% (2.7–7.3) | 4.0% (1.9–6.7) |
| code-0087 | 0.0% (0–2.6) | 9.1% | 5.0% (3.1–6.9) | 5.1% (3.3–7.1) |

- **It works on hosted endpoints:** CoreWeave vs Parasail median 6.0% sampled, the same size as the logprob
  campaign's 5–7%, with ±2.4-point intervals at 4,000 samples per provider. Run 1 and run 2 of code-0094 agree
  (4.3% vs 3.7%).
- **DeepInfra's turbo endpoint**, which returns no logprobs and so was never in the TV comparison, sits a median
  4.7% from CoreWeave (0.3–13.5%) and 5.1% from Parasail (1.4–12.5%) over 10 prompts: the same range as the fp8
  providers from each other. Nothing suggests a different model. On some prompts it sides with one provider
  (wiki-0333 and code-0171: with Parasail; code-0005, code-0116, code-0172: with CoreWeave), as independent
  low-precision deployments would.
- **The sampled TV and the logprob TV disagree prompt by prompt** (wiki-0333 14.0% vs 8.7%; code-0172 3.9% vs 12.7%)
  even though both are the same pair. That isn't the estimator: requests that ask for logprobs get a different
  prompt at CoreWeave and Parasail (the Llama default system block is added only on the logprob path), and each
  provider's no-logprob samples sit 16–17% (median TV) from its own logprob path. See
  `reports/logprob-template-2026-11-17.md`. Sampling measures what ordinary requests get.

## Caveats
- One model, first token only, prompts chosen as near-ties (they amplify differences); typical prompts are peakier,
  and their TV is smaller and cheaper to resolve.
- The TV covers named tokens plus one OTHER bucket: a lower bound on the full TV.
- Hosted requests went direct to OpenRouter, not through the witness relay (about 105,000 one-token calls). Every
  sample is kept with its generation id.
- 87% coverage on 1%-sized differences: treat intervals on near-zero TVs as approximate.

## Witnessed replication (2026-11-19)

The whole experiment was re-run through the kremlin witness proxy (`data/kremlinology-adaptive-tv-hosted-w` +
`-w-session`: 116,633 records, 112,165 with relay receipts and zero mismatches; 4,468 requests were refused by the
relay's rate cap and never executed; `kremlin audit --require-witness` passes). All 10 prompts, three providers, 4,000
samples each ($1.65). Results match the direct runs: CoreWeave~Parasail median 5.3% (direct: 6.0%), DeepInfra turbo in
the same band (e.g. code-0171: 6.2% from CoreWeave, 0.9% from Parasail), per-prompt sampled-vs-logprob disagreements as
before. The template finding's A/B runs were also re-run witnessed (`sample-vs-logprob-ab-w*`, $0.55, audit passes;
the relay rate-capped the burst, so arms have n=249-640) and replicate (e.g. code-0087: code-fence opener 72%/90% with
logprobs vs 0% without, both providers).