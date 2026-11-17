# Dataset-average first-token TV: Llama 3.3-70B across OpenRouter providers

Date: 2026-11-10.

**Run:**
- Model: `meta-llama/llama-3.3-70b-instruct` via OpenRouter, routes pinned with no fallbacks.
- Reference: CoreWeave (declared fp16). Challengers: AkashML (declared fp8), Parasail (declared fp8), SambaNova-turbo (precision undisclosed).
- 400 prompts sampled uniformly at random (seed 20261110, without replacement): 200 from CodeAlpaca-20k instructions and 200 wikitext-103 validation continuation prompts.
- First token only, top-20 logprobs, non-streaming, 2 repeats per provider.
- 3,200 of 3,200 requests completed with no retries. Reported cost $0.1087. Manifest verified.
- Every response was structurally aligned with top-20 logprobs, so no prompt needed conservative imputation.

**Evidence:**
- Run: `data/kremlinology-llama70b-dataset-tv-v2`. The v1 run is retained; it stopped after 93 requests on a Parasail 429.
- Analysis: [JSON](llama70b-dataset-tv-2026-11-10.json).
- Code: `experiments/build_dataset_tv_campaign_v1.py`, `experiments/dataset_tv_analysis_v1.py`, `kremlinology.bench.dataset_bounds`.

## Estimand and method

For each prompt, repeat-matched challenger/reference pairs give an interval for first-token TV from the partial top-20 reports. Each listed probability is allowed a report error of ±ε nats; unlisted mass is unknown. A prompt's interval is the mean over its two repeat pairs.

The dataset average lies between the means of the per-prompt lower and upper bounds. The 95% interval combines a one-sided Waudby-Smith–Ramdas betting bound on each side (α/2 each). These bounds are distribution-free and finite-sample.

Also reported:
- Clopper–Pearson intervals for the fraction of prompts whose TV lower bound exceeds 0.05.
- The fraction whose cross-provider lower bound exceeds CoreWeave's own repeat-pair upper bound.

**Why ε=0.01 is the primary allowance:** at ε=0, 17–19% of comparisons were infeasible, because reported top-20 probabilities sum slightly above 1 (rounded reports). The ε=0 averages therefore cover a selected subset and are not used. At ε=0.01 and 0.05, all 200 prompts per dataset are assessable.

## Results (ε = 0.01 nats)

| Dataset | Provider | Mean TV bounds | 95% interval, dataset-average TV | Prompts with TV > 5% | Prompts beyond CoreWeave self-noise |
| --- | --- | --- | --- | --- | --- |
| code | CoreWeave vs itself | 0.004–0.015 | [0, 0.051] | 1% [0, 4%] | — |
| code | AkashML fp8 | 0.054–0.067 | **[0.042, 0.102]** | 38% [31, 45] | 77% [71, 83] |
| code | Parasail fp8 | 0.046–0.059 | **[0.035, 0.094]** | 31% [25, 38] | 68% [61, 74] |
| code | SambaNova | 0.075–0.088 | **[0.054, 0.124]** | 42% [35, 49] | 73% [66, 79] |
| wiki | CoreWeave vs itself | 0.004–0.018 | [0, 0.053] | 1% [0, 4%] | — |
| wiki | AkashML fp8 | 0.061–0.076 | **[0.043, 0.112]** | 43% [36, 50] | 54% [46, 61] |
| wiki | Parasail fp8 | 0.049–0.063 | **[0.034, 0.099]** | 32% [25, 38] | 51% [43, 58] |
| wiki | SambaNova | 0.028–0.043 | **[0.017, 0.078]** | 20% [15, 27] | 38% [31, 45] |

At ε=0.05 every challenger's interval still excludes zero. The lowest is SambaNova on wiki, [0.010, 0.114].

Top-1 token agreement with CoreWeave over 800 paired requests:

| Provider | Agreement | 95% interval |
| --- | --- | --- |
| CoreWeave (repeat vs repeat) | 98.3% | [97.1, 99.0] |
| AkashML | 92.7% | [90.7, 94.4] |
| Parasail | 95.0% | [93.3, 96.4] |
| SambaNova | 94.1% | [92.3, 95.7] |

The median absolute shift in the reference top token's logprob is 0.020–0.030 nats; the 90th percentile is 0.19–0.23.

Mean TV lower bound by CoreWeave's top-1 margin tercile (ε=0.01, from near-tie to confident):

| Dataset | Provider | Near-tie | Middle | Confident |
| --- | --- | --- | --- | --- |
| code | AkashML | 0.096 | 0.052 | 0.017 |
| code | Parasail | 0.076 | 0.049 | 0.014 |
| code | SambaNova | 0.110 | 0.077 | 0.038 |
| wiki | AkashML | 0.142 | 0.042 | 0.001 |
| wiki | Parasail | 0.110 | 0.037 | 0.000 |
| wiki | SambaNova | 0.064 | 0.021 | 0.000 |

## Interpretation

1. **The differences are real at the dataset level.** On both prompt distributions, all three challengers' average first-token distributions differ from CoreWeave's, with 95% lower bounds of 0.017–0.054 TV, against CoreWeave's own repeat noise (mean lower bound 0.004). The earlier 8-prompt "max" figures (0.29–0.34) were worst cases. Typical averages are about 5–8% TV.
2. **The pattern looks like a numerical difference, not a different model.** Differences concentrate on prompts where CoreWeave's top two tokens are close. On wikitext they almost vanish when the reference is confident (lower bound ≤0.001). Top-1 agreement is 93–95%, and the typical logprob shift is a few hundredths of a nat. This is what systematically perturbed logits from the same model look like, for example fp8 against fp16. A substituted or differently trained model would also differ on confident prompts, as Qwen 0.5B against 1.5B did locally (gaps up to 5 nats).
3. **SambaNova differs most on code, and is the only provider still different on confident code prompts** (0.038 in the top tercile). Its precision is undisclosed. This is the most interesting lead, though still not attributable.
4. **The reference is not ground truth.** CoreWeave's "fp16" is a declaration. The comparisons measure distance from CoreWeave's serving path, not from the true model.

## Limits

- **First token only,** for two public datasets and this request format (no system prompt, chat template applied by each provider). Other traffic may differ.
- **The per-prompt intervals assume a shared contract:** identical hidden templates and token events across providers, reports from the unprocessed distribution, and report error within ε. Rounded reports showed that ε=0 is violated; ε=0.01 and 0.05 are assumptions, not measured coverage.
- **The tercile pattern supports, but does not establish, a numerical explanation.** No fp8 Llama 3.3-70B reference was run. It would need about 70 GB of GPU memory, against 24 GB per local card.
- **No penalty or provenance conclusion follows from these results.**

## Evidence completeness (added after the run)

The run predates metadata promotion, so serving metadata was backfilled from the retained raw captures into `derived/`. The original files are unchanged, and `verify` checks the run manifest and the derived manifest.

**OpenRouter generation-stats reconciliation:** 3,200 of 3,200 lookups succeeded (free lookups, no inference).
- Every recorded cost equals OpenRouter's `total_cost`; the total is $0.108745.
- No response came from OpenRouter's response cache.
- No request was cancelled.
- Each provider was served by exactly one OpenRouter endpoint ID.

**Declared serving metadata:**
- AkashML returned two `system_fingerprint`s behind its single endpoint: `vllm-0.31.0-tp2-260b81aa` (656 requests) and `…-e1f0aa81` (144). That suggests two replicas.
- SambaNova returned `fastcoe` on all 800. CoreWeave and Parasail returned none.
- Prefix-cache hits: CoreWeave 637/800, Parasail 506/800, AkashML 800/800, SambaNova 0/800.

Mean TV midpoint by stratum (ε=0.01, CLT ±95%). The strata are observational, not randomized:

| Stratum | Pairs | Mean TV |
| --- | --- | --- |
| AkashML replica `260b81aa` | 656 | 0.069 (about ±0.006) |
| AkashML replica `e1f0aa81` | 144 | 0.046 (about ±0.010) |
| CoreWeave vs itself, both repeats uncached | 47 | 0.0058 ± 0.0007 |
| CoreWeave vs itself, both repeats cached | 284 | 0.0104 ± 0.0014 |
| Parasail, both sides uncached | 145 | 0.048 ± 0.008 |
| Parasail, both sides cached | 488 | 0.058 ± 0.006 |

Replica and cache state are plausible contributors to the measured TV. Testing that needs randomized or explicitly controlled cache state, for example a per-request nonce prefix against repeated identical prefixes.

**Retrospective provenance:** the prompt manifest, dataset revisions (CodeAlpaca-20k `152bb5e9…`, wikitext `b08601e0…`) and an endpoint catalog fetched after the run are stored in `derived/retrospective_provenance.json`, marked retrospective. New runs record these at run time through `evidence.catalog_snapshot` and `evidence.attachments`.

**Backup:** v1, v2 and the local-pipeline v3 runs are copied to a second disk and verified against their manifests (`backup-log.jsonl`).
