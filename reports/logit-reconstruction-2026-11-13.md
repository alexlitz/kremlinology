# How well does sampling reconstruct logits? Local ground truth (2026-11-13)

**How it was run:**
- Scripts: `experiments/logit_reconstruction_benchmark_v1.py`, `experiments/logit_reconstruction_figure_v1.py`.
- Outputs: `reports/logit-reconstruction-2026-11-13{.json,-records.json,.png}`.
- Local GPU0, no API calls.
- **Truth:** fp32 next-token logits of Llama-3.2-3B-Instruct on 100 dataset prompts (the Llama-70B campaign's CodeAlpaca and wikitext manifest, chat template applied). Median top-1 probability is 0.52 (IQR 0.38–0.71).
- **Sampling:** exact provider order (temperature → top-K → inclusive top-P → multinomial). This isolates the statistics; HTTP semantics and fault modes are covered by `local_sampler_pipeline_v1`. `n` doesn't change the statistics: n choices in one request are n independent draws, so it only changes how many requests a budget costs.
- **Estimands:** logit gaps z_r − z_1 for the true ranks 2–10 (900 gaps), and TV over the top-10 conditional distribution.
- **Estimator:** pooled multinomial MLE (`pooled_logit_fit`) on labels with ≥ 5 samples, keeping at most the 40 most-sampled labels.

## Results (fp32 sampling, naive support)

| strategy | budget | ranks 2–10 recovered | median abs. error | 95% CI coverage | top-10 TV |
|---|---|---|---|---|---|
| plain T=1 | 1,000 | 69% | 0.12 nats | 0.95 | 0.024 |
| plain T=1 | 10,000 | 88% | 0.05 | 0.96 | 0.007 |
| plain T=1 | 100,000 | 97% | 0.02 | 0.93 | 0.003 |
| T=2, top-K=10 | 1,000 | 96% | 0.20 | 0.95 | 0.034 |
| T=2, top-K=10 | 10,000 | 100% | 0.07 | 0.96 | 0.012 |
| T=2, top-K=10 | 100,000 | 100% | 0.02 | 0.95 | 0.004 |
| top-K ladder 2/4/10 | 10,000 | 100% | 0.10 | 0.95 | 0.012 |
| adaptive pair design | 10,000 | 100% | 0.08 | 0.95 | 0.011 |

**By true gap (T=2 top-10 vs plain, % recovered and median error):**

| gap below top-1 | 1k: hot | 1k: plain | 10k: hot | 10k: plain |
|---|---|---|---|---|
| 0–2 nats | 100%, 0.14 | 100%, 0.07 | 100%, 0.05 | 100%, 0.02 |
| 2–5 | 100%, 0.19 | 92%, 0.15 | 100%, 0.07 | 100%, 0.05 |
| 5–10 | 92%, 0.30 | 9%, 0.48 | 100%, 0.11 | 65%, 0.16 |
| > 10 | 14% | 0% | 100%, 0.18 | 0% |

## Findings

1. **It works, and the error bars are honest.**
   - Errors fall as 1/√budget, about 0.2 nats at 1k samples, 0.07 at 10k and 0.02 at 100k.
   - Nominal 95% intervals cover 93–97% of true gaps at every budget of 1,000 or more.
   - Sample-splitting the support (to avoid label-selection bias) wasn't needed at this label threshold. It cost precision without improving coverage.
2. **Temperature plus top-K is what reaches the tail.**
   - At T=1 a token 5–10 nats down appears less than once in a few thousand draws.
   - T=2 with top-K=10 halves every gap and stops spending samples outside the top 10: 92% of those gaps are recovered at 1k samples vs 9% for plain sampling, and 100% at 10k vs 65%.
   - Plain T=1 is the most precise on easy gaps under 5 nats.
   - The top-K ladder and the adaptive pair design add nothing over the single hot top-10 run on these distributions.
3. **Top-P bisection pins absolute mass.** The runner-up appears in the nucleus iff p1 < top-P, so bisection brackets the top-1 probability.
   - At 1k samples, 91% of prompts are bracketed (median width 2e-5). From 10k up, 40 bisection steps are completed, reaching floating-point width.
   - Coverage is 0.96 from 10k samples up, but only 0.83–0.86 below that, where the pilot's estimate of the runner-up share is noisy. There's no multiplicity correction across probes.
   - Combined with the gaps, this makes the log-probs absolute, not just up to a shift.
4. **The real limit is the server's numerics, not sampling.**
   - Feeding the estimators bf16-model samples and scoring against fp32 truth plateaus at about 0.06 nats median error, with 95% coverage falling to 51–76%. The bf16 vs fp32 full-distribution TV has a median of 1.3%.
   - Top-P bisection on the bf16 model brackets the bf16 p1 so tightly that it covers the fp32 p1 in 0 of 100 prompts.
   - So beyond about 10k samples per prompt, sampling measures the provider's precision and kernels, which is the forensic signal, not noise to average away.

## Bug found and fixed

`pooled_logit_fit` (`src/kremlinology/sampled_logit_estimates.py`) used undamped Newton steps from zero. On long-tailed counts (40 labels spanning 11,087 to 89 samples) it overshot and went singular, so most large-budget fits failed silently. It now starts from smoothed log counts and halves steps until the likelihood stops falling. There's a regression test in `tests/test_sampled_logit_estimates.py`.

## Caveats

- One 3B model, first token only. Peakier distributions (lower entropy) need more samples for the tail.
- The sampler is exact and the stated semantics are trusted. A provider ignoring top-K or temperature, applying penalties, or varying logits between requests breaks the estimates. Those are tested separately (canaries, overdispersion).
- The budget is in sampled tokens. With `max_tokens=1` and `n`, 10k samples cost 10k output tokens plus a prompt per request.
