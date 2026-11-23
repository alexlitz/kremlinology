# Live sampling-policy audit

## Extended top-k/top-p campaign

`traces/sampling-audit-extended-v1/results.json` completed 15,360 HTTP requests
and correctly identified **40/40 deployments**: two models × four weight
paths × honest/argmax/ignored-top-k/ignored-top-p/ignored-both. Each deployment
uses the original two128-trial top-k conditions plus128 trials with top-k0,
temperature1 and top-p=.001. The returned top1 full-model probability must
exceed .001; this makes the nucleus a fully observed singleton without
assuming that missing tail tokens have probability0. If that precondition
fails, the experiment stops rather than fabricating support.

These are fixed local policy adversaries, not all forms of steering. A small
nucleus probes gross enforcement, not every boundary at normal top-p values.
Separate k and p interventions avoid confusing the ordering of filters with
ignoring one of them. All one-token trials leave compressed-KV decode unused;
the configured KV setting is not validated by this result. Original traces
remain retained, and the larger compound-validation extension is prepared
but has not run yet. No hosted spend.

The sampling reference now supports threshold-style top-k boundary ties,
matching the local fixture. An exact-k API can request the alternate convention;
this is a semantic calibration requirement. The extended live campaign ran
before that subsequent tie-handling adjustment; its accuracy artifact is not
silently recomputed. This does not eliminate ambiguity when unreported tokens
tie at a censored boundary or when API nucleus tie ordering is unspecified.

## Original three-policy campaign

`traces/sampling-audit-v1/results.json` records 6,144 completed one-token HTTP
requests across 24 deployments: Qwen2.5-0.5B and TinyLlama-1.1B, four weight
formats, and three declared sampling policies. All 24 policies were correctly
identified. Each serving stack also enabled prefix reuse, FP4 KV settings,
and stream coalescing. Because completions have one token, this experiment
does not exercise compressed-KV decoding or validate coalescing detection.

The fixed prompt asks for a choice of colors. Each deployment is sampled 128
times with top-k=1 and 128 times with top-k=5, temperature=3, top-p=1. The
predictor uses selected token bytes and pre-temperature exposed logprobs.
No weight precision labels, private sampler labels, or endpoint identifiers
enter the decision rule. Predictions are saved before aggregate grading.

Ignoring top-k is detected by a k1 support violation. Forced argmax is
distinguished from honest sampling by deviation from the k5 distribution,
using coordinate-wise Hoeffding bounds and a union bound (alpha=.001 per
audit). Observed empirical TV is measured relative to the exposed truncated
distribution, not independent full-checkpoint TV. A nonrejection is not proof
of honesty. Alpha is per audit, not a familywise guarantee for the campaign.

This closed-policy test does not establish detection of every steering
adversary, hidden temperature transformations, top-p noncompliance, or fake
logprob reporting. Providers with post-truncation logprobs require a different
semantic calibration. Near top-k ties and repeated byte spellings need care.
All prompts, responses, timestamps, and usage are retained. No hosted spend.
