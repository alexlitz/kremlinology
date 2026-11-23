# Asking for logprobs changes the prompt: Llama 3.3-70B on CoreWeave and Parasail (2026-11-17)

**Found by:** the adaptive sampling-only TV run (`adaptive-tv-2026-11-17.md`). Its control compares sampled
frequencies with the same provider's logprobs, and on near-tie prompts the samples ran 7–12 points away from the
provider's own logprobs, on CoreWeave fp16 as well as Parasail fp8.

**Scripts:** `experiments/sample_vs_logprob_v1.py`. Evidence: `data/kremlinology-sample-vs-logprob-v1`,
`-ab-v1`, `-ab-v2` (every response kept). OpenRouter, pinned endpoints, one-token requests. Total cost $0.56.

## 1. Within one response, samples follow the logprobs

600 requests at temperature 1 with `logprobs=true, top_logprobs=5`, so each response carries the sampled token and
the distribution it was drawn from:

| prompt · provider | token | sampled (95% CI) | mean reported probability |
|---|---|---|---|
| wiki-0333 · CoreWeave | `ban` | 51.8% (47.8–55.8) | 52.4% |
| wiki-0333 · Parasail | `pro` | 54.0% (50.0–57.9) | 53.1% |
| code-0094 · CoreWeave | `###` | 52.2% (48.2–56.1) | 55.0% |
| code-0094 · Parasail | `**` | 50.7% (46.7–54.7) | 49.6% |

So when logprobs are requested, the sampler and the reported numbers agree.

## 2. With vs without logprobs: different distributions

Interleaved A/B: alternating requests with and without logprobs, same prompt, same minutes, temperature 1
(800 per arm on the first two prompts, 400 on the other eight).

| prompt | CoreWeave fp16: with → without logprobs | TV | Parasail fp8: with → without | TV |
|---|---|---|---|---|
| wiki-0333 | `pro` 44.9% → 54.7% | 0.10 | `pro` 51.4% → 71.1% | 0.20 |
| code-0094 | `**` 42.7% → 56.9% | 0.14 | `**` 51.0% → 52.4% | 0.01 |
| code-0034 | ```` ``` ```` 63% → 0%, `###` 37% → 84% | 0.63 | ```` ``` ```` 81% → 0% | 0.81 |
| code-0171 | ```` ``` ```` 72% → 8%, `Here` 28% → 83% | 0.64 | ```` ``` ```` 76% → 0% | 0.76 |
| code-0087 | ```` ``` ```` 74% → 0%, `**` 26% → 100% | 0.74 | ```` ``` ```` 86% → 0% | 0.86 |
| code-0172 | `Here` 48% → 81% | 0.33 | `Here` 54% → 69% | 0.15 |
| code-0116 | `**` 50% → 64% | 0.15 | `**` 45% → 71% | 0.26 |
| code-0071 | `Here` 44% → 25% | 0.19 | `Here` 46% → 28% | 0.19 |
| code-0005 | `**` 57% → 52% | 0.06 | `**` 65% → 63% | 0.02 |
| code-0137 | `###` 50% → 57% | 0.07 | `**` 51% → 50% | 0.01 |

Median TV between the two arms: 0.19 (sampling noise at n = 400 is about 0.03). Both providers move the same way on
every prompt where they move, which points at the input rather than at either provider's numerics.

## 3. The cause: a different chat template

`usage.prompt_tokens` for the same one-message request (`"Create a program using any scripting language to print the
multiplication table of 5."`):

| endpoint | without logprobs | with logprobs |
|---|---|---|
| Llama 3.3-70B · CoreWeave fp16 | **26** | **51** |
| Llama 3.3-70B · Parasail fp8 | **27** | **51** |
| Llama 3.3-70B · AkashML, Cloudflare, Groq | 51 | 51 |
| Llama 3.3-70B · SambaNova | 26 | (429) |
| Qwen3.5-397B · Parasail, Alibaba, Novita, Venice | 26 | 26 |
| Gemma 4 31B · DeepInfra, Parasail | 29 | 29 |
| gpt-oss-120b · Nebius, Parasail, CoreWeave | 83 | 83 |

The 25 extra tokens match the system block Meta's published Llama 3.x template inserts when the request has no system
message ("Cutting Knowledge Date: December 2023 / Today Date: …"). The logprob-campaign analysis
(`llama70b-dataset-tv-2026-11-10.md`) found that template reproduced every provider's count on all 3,200 requests,
all of which asked for logprobs. Without logprobs, CoreWeave and Parasail (and SambaNova) serve the same message without
that block.

- **What it means for logprob comparisons:** every logprob-based Llama measurement (the dataset TV campaign, the
  reseller matching) compared providers on the 51-token path. That comparison is internally consistent (all
  providers saw the same prompt) but is not the prompt CoreWeave's and Parasail's ordinary users get. Which path is
  the "real" deployment is not visible from outside: the two paths could be one engine with two templates (e.g.
  OpenRouter or the provider formatting logprob requests differently) or two deployments.
- **What it means for sampling-only checks:** they measure the path users actually get, which is the reason to have
  them. Between CoreWeave and Parasail the no-logprob path differs by more than the logprob path on the same prompts
  (wiki-0333: 13.9% vs 8.7% TV; see the adaptive TV report).
- **Not an accuracy bug as such:** the default system block only adds a date; but it shifts first-token distributions
  by up to 0.86 TV on short coding prompts (code-block vs prose openers), so any comparison has to hold the template
  fixed.

## Caveats
- One model (Llama 3.3-70B). Qwen3.5, Gemma 4 and gpt-oss counts don't change, so the effect is template-specific.
- First token only; the templates' effect on whole answers isn't measured.
- 10 prompts, all chosen as near-ties from the logprob campaign, which amplify any change in the logits.
