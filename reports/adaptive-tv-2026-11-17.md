# Provider TV from samples alone: an adaptive estimator, validated, then run on hosted Llama 3.3-70B (2026-11-17)

**Question:** can you measure how far two providers' next-token distributions are apart (total variation, the
yardstick of the logprob campaigns) when one of them returns no logprobs?

**Scripts:** `experiments/adaptive_tv_v1.py` (estimator and local validation), `adaptive_tv_hosted_v1.py`,
`adaptive_tv_hosted_v2.py` (hosted runs), `adaptive_tv_hosted_analysis_v1.py` (final numbers from the raw samples).
Data: `reports/adaptive-tv-2026-11-17.json` (+ `-records.json`), `reports/adaptive-tv-hosted-2026-11-17.json`;
evidence `data/kremlinology-adaptive-tv-hosted-v1`, `-v2` (every one-token sample kept). Hosted spend $1.85 for
105,318 samples (plus a few cents of logprob requests).

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
OpenRouter, pinned endpoints). DeepInfra's shared upstream pool returned 429 for ~3 hours, so DeepInfra has 4 prompts.

| prompt | CoreWeave ~ Parasail (sampled) | logprob path, same pair | CoreWeave ~ DeepInfra | Parasail ~ DeepInfra |
|---|---|---|---|---|
| code-0094 (run 1) | 4.2% (1.4–7.2) | — | 7.6% (4.5–10.8) | 11.0% (7.7–14.5) |
| code-0094 (run 2) | 3.7% (0.9–6.6) | 3.8% | | |
| wiki-0333 | 14.0% (11.5–16.4) | 8.7% | | |
| code-0034 | 6.8% (4.9–8.8) | 14.0% | | |
| code-0171 | 6.3% (4.3–8.4) | 2.6% | | |
| code-0005 | 5.6% (3.0–8.1) | 2.5% | | |
| code-0172 | 0.0% (0–5.7) | 12.7% | | |
| code-0116 | 8.2% (5.6–10.9) | 5.8% | | |
| code-0137 | 8.2% (5.5–10.8) | 5.3% | 2.7% (0–5.3) | 5.9% (3.2–8.6) |
| code-0071 | 1.1% (0–4.4) | 2.5% | 4.8% (2.8–7.2) | 4.0% (1.5–7.1) |
| code-0087 | 0.0% (0–2.6) | 9.1% | 5.0% (3.1–7.1) | 5.2% (3.2–7.1) |

- **It works on hosted endpoints:** CoreWeave vs Parasail medians 5.9% sampled, the same size as the logprob
  campaign's 5–7%, with ±2.5-point intervals at 4,000 samples per provider. Run 1 and run 2 of code-0094 agree
  (4.2% vs 3.7%).
- **DeepInfra's turbo endpoint**, which returns no logprobs and so was never in the TV comparison, sits 2.7–7.6% from
  CoreWeave on the 4 prompts it served: the same range as the fp8 providers. Nothing suggests a different model.
- **The sampled TV and the logprob TV disagree prompt by prompt** (wiki-0333 14.0% vs 8.7%; code-0172 0% vs 12.7%)
  even though both are the same pair. That isn't the estimator: requests that ask for logprobs get a different
  prompt at CoreWeave and Parasail (the Llama default system block is added only on the logprob path), and each
  provider's no-logprob samples sit 16–20% (median TV) from its own logprob path. See
  `reports/logprob-template-2026-11-17.md`. Sampling measures what ordinary requests get.

## Caveats
- One model, first token only, prompts chosen as near-ties (they amplify differences); typical prompts are peakier,
  and their TV is smaller and cheaper to resolve.
- The TV covers named tokens plus one OTHER bucket: a lower bound on the full TV.
- Hosted requests went direct to OpenRouter, not through the witness relay (about 105,000 one-token calls). Every
  sample is kept with its generation id.
- 87% coverage on 1%-sized differences: treat intervals on near-zero TVs as approximate.
