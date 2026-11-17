# Kremlinology for LLM inference APIs

Code, reports and data behind the blog post *[title TBD]* (https://alexlitzenberger.com/blog/... — link added at
publication). Everything here measures public inference APIs from the outside, using only our own synthetic and
public-dataset prompts.

- **`kremlin/`**: the toolkit: one command per investigation, evidence directories with replayable audits, the TLS
  witness relay, the findings hub, and a witnessed proxy usable from any language. See [`kremlin/README.md`](kremlin/README.md).
- **`experiments/`, `src/`, `tools/`, `configs/`**: the probes and analyses the post's numbers come from.
- **`reports/`**: one report per study (method, numbers, caveats), with figures and derived data (CSV/JSON).
- **Raw evidence**: every request and response (API keys redacted), as archives attached to the GitHub release.
- **Attested runs**: witnessed through the public relay and re-audited by the findings page at
  https://alexlitzenberger.com/kremlinology/ (each run page links its evidence bundle).

## Quick start

```
pip install './kremlin[witness]'          # the toolkit (or: pipx install from the findings page)
pip install -r requirements.txt           # analysis and figure dependencies

# reproduce a figure from the raw evidence
mkdir data && tar -xzf kremlinology-data-cache-semantics.tar.gz -C data
KREMLINOLOGY_DATA=data python experiments/cache_semantics_figure_v2.py
```

Probes that send requests need `OPENROUTER_API_KEY` (and spend money; each has a hard spend cap argument). Analyses and
figures run offline on the archives.

## Map: post section → report → code → data

| post section | report(s) | code | raw data archive | attested runs |
|---|---|---|---|---|
| Caching: the breakpoint tell | `cache-semantics-2026-11-13`, `cache-semantics-open-models-2026-11-17`, `cache-isolation-2026-11-13` | `cache_semantics_probe_v1`–`v8`, `cache_semantics_analysis_v1`, `cache_semantics_figure_v2`, `cross_tenant_cache_probe_v1`, `cache_scope_probe_v1`; open models: `kremlin cache-prefix` | `cache-semantics`, `cache-misc` | open-model runs (see hub) |
| How long providers keep your cache | `kv-cache-economics/README` (fig 16) | `cache_ttl_probe_v1`–`v4`, `cache_ttl_analysis_v1` | `cache-retention` | — |
| Are cache hits real, and does the router chase them? | `cache-routing-honesty-2026-11-16` (+ `cache-honesty-pooled-2026-11-17.json`), `prefill-cache-latency-2026-11-17` | `cache_routing_honesty_v1` (+ `_analysis_v1`), `cache_latency_grid_age_v1` (+ `cache_latency_analysis_v1`) | `prefill-latency`, `witnessed-sessions` | [f30459ff](https://alexlitzenberger.com/kremlinology/runs/f30459ff73c39797e69c/), [61175138](https://alexlitzenberger.com/kremlinology/runs/61175138068b9e03de82/) |
| Why caches die young: the economics | `kv-cache-economics/README` | `kv_cache_economics_v1`, `kv_cache_survey_v1`, `kv_reuse_tiers_v1`, `collect_model_configs_v1`, `configs/kv_models/` | Mooncake traces: see note | — |
| Are you getting the model you're paying for? | `llama70b-dataset-tv-2026-11-10`, `qwen35-dataset-tv-2026-11-13`, `gemma4-31b-dataset-tv-2026-11-13` | `build_dataset_tv_campaign_v1`, `dataset_tv_analysis_v1`, `dataset_tv_figure_v1`, `configs/bench/*dataset-tv*` | `dataset-tv` | — |
| Without logprobs: recovering logits, checking the sampler | `logit-reconstruction-2026-11-13`, `top-p-ladder-2026-11-17`, `adaptive-tv-2026-11-17`, `sampling-audit-2026-10-04` | `logit_reconstruction_benchmark_v1`, `logit_reconstruction_figure_v1`, `top_p_ladder_v1`, `adaptive_tv_v1` (local validation), `adaptive_tv_hosted_v1`/`v2` + `adaptive_tv_hosted_analysis_v1`; hosted check: `kremlin sampler` | `witnessed-sessions` (sampler), `adaptive-tv` | [f51dfaf1](https://alexlitzenberger.com/kremlinology/runs/f51dfaf13deacb3553df/) |
| Who's really serving you? Resellers | `qwen35-dataset-tv-2026-11-13`, `cache-isolation-2026-11-13` | `reseller_identity_figure_v1`; re-check: `kremlin fingerprint` | `dataset-tv`, `witnessed-sessions` (venice-confirm) | [09d44f3d](https://alexlitzenberger.com/kremlinology/runs/09d44f3d2ea48da10dd5/) |
| Tokenizers and templates | `llama70b-dataset-tv-2026-11-10`, `logprob-template-2026-11-17` | `dataset_tv_analysis_v1`, `sample_vs_logprob_v1` | `dataset-tv`, `adaptive-tv` | — |
| When the API tells on itself | `max-tokens-2026-11-13` (+ `max-tokens-pooled-2026-11-17.json`), `later-token-forcing-pilot-2026-11-10` | `max_tokens_probe_v1`, `max_tokens_analysis_v1`, `later_token_forcing_pilot_v1` | `cache-misc` | — |
| Timing, speculation and hardware | `timing-speculation-2026-11-13`, `timing-jitter-2026-11-16` (figures and CSV dated 11-17 pool all runs) | `configs/bench/timing-speculation-v*.json` and `experiments/timing-speculation-v4*`–`v6*.json` (run with `python -m kremlinology.bench`), `timing_speculation_analysis_v1`, `timing_jitter_analysis_v1` | `timing` | — |
| Prefill speed, and can you see disaggregation? | `prefill-cache-latency-2026-11-17` | `prefill_cache_sweep_v1`, `prefill_cache_sweep_analysis_v1` | `prefill-latency` | — |

Reports are in `reports/`, code in `experiments/` unless noted. Shared helpers: `experiments/ci_utils.py` (Wilson and
bootstrap intervals), `experiments/kv_cache_economics_v1.py` (figure style), `src/kremlinology/bench/` (the request
harness the TV and timing campaigns use: `PYTHONPATH=src python -m kremlinology.bench run CONFIG --output DIR --execute`).

## Data archives (GitHub release)

| archive | contents |
|---|---|
| `kremlinology-data-dataset-tv.tar.gz` | Llama 3.3-70B, Qwen3.5-397B, Gemma 4 31B first-token logprob campaigns |
| `kremlinology-data-timing.tar.gz` | timing-speculation runs v1–v6d (per-chunk arrival times) |
| `kremlinology-data-cache-semantics.tar.gz` | closed-model probes v1–v8 and the open-model runs open-v1–v3 |
| `kremlinology-data-cache-retention.tar.gz` | cache lifetime probes v1–v4 |
| `kremlinology-data-cache-misc.tar.gz` | cross-tenant cache v1, v2, cache scope, max-tokens v1–v3, later-token pilot |
| `kremlinology-data-prefill-latency.tar.gz` | prefill sweep v1–v4; cache grid v1–v5 and age v1–v3 |
| `kremlinology-data-witnessed-sessions.tar.gz` | witnessed sessions: Venice re-check, cache honesty and routing (v1–v3), sampler check, language tests |
| `kremlinology-data-adaptive-tv.tar.gz` | hosted sampling-only TV runs (every one-token sample), logprobs on/off A/B and same-response checks |

Each run directory has a `manifest.json` (SHA-256 of every file); witnessed runs also carry the relay's signed receipts
and can be re-audited with `kremlin audit DIR --pubkey 1pUve6CoMVNVk6p9750A5cED9hyw80ut4kOr6DJ12aY= --require-witness`.

## Notes
- The cache-economics simulation uses Moonshot's published Mooncake request traces; get them from
  https://github.com/kvcache-ai/Mooncake and extract to `data/kremlinology-mooncake-traces`.
- `reports/sampling-audit-2026-10-04.md` describes an earlier local study whose scripts aren't included here.
- API spend is reported per study in each report (cents to a few dollars per study).
- API keys are redacted from all evidence; the witness relay never sees them (see `kremlin/README.md`).

## License
MIT (see `LICENSE`) for code, reports and data.
