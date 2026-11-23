# Output caps: who honours `max_tokens`, and what over-cap tokens cost (2026-11-13)

**How it was run:**
- Evidence:
  - `data/kremlinology-max-tokens-probe-v1` (script `experiments/max_tokens_probe_v1.py`, $0.038).
  - Generation stats for `data/kremlinology-timing-speculation-v2` (`derived/generation_stats.json`, via `tools/reconcile_openrouter_generations.py`).
- Prompt: "Write an original 600-word story…", temperature 0, cap 64.
- Each pinned target got 3 repeats of each of three variants:
  - `max_tokens`, non-streamed.
  - `max_tokens`, streamed.
  - `max_completion_tokens`.
- Billing is OpenRouter's generation record (native completion and reasoning tokens, total cost). Both endpoints declare `max_tokens` support in the catalog.

## Results (cap 64)

| target | completion tokens billed | of which reasoning | visible chars | cost / request |
|---|---|---|---|---|
| SiliconFlow gpt-oss-120b | **1,176–1,403** (all 9) | 90–121 | 4.8–5.7k | $0.0007–0.0009 |
| Baseten gpt-oss-120b (control) | 64 | 0 reported | 0 (cap spent reasoning) | $0.00004 |
| GMI Cloud DeepSeek-V3.2 | **3,580–16,811** (all 9) | 3,514–16,745 | 267–318 | **$0.0011–0.0052** |
| SiliconFlow DeepSeek-V3.2 (control) | 64 | 0 | 254–299 | $0.00006 |
| DeepInfra DeepSeek-V3.2 (control) | 64 | 0 | 243–307 | $0.00003 |

## Findings

1. **SiliconFlow ignores the output cap for gpt-oss-120b**, in every variant. About 20× the requested tokens are returned and billed. The same provider honours the cap on DeepSeek-V3.2, so this is per-deployment, not a policy.
2. **GMI Cloud enables DeepSeek-V3.2 reasoning without being asked and excludes it from the cap.**
   - The visible answer stops at about 64 tokens (finish_reason `length`), but thousands of reasoning tokens are generated and billed.
   - A capped request costs 40–170× what it costs at DeepInfra.
   - In the timing run (cap 256), GMI took 11–33 s to the first visible token.
   - In a one-token logprob request it returned 7 tokens, and a 1-token dataset campaign timed out on it.
3. **The timing run (cap 256) shows the same pattern.** SiliconFlow gpt-oss: 299–1,413 tokens, finish `stop`. GMI DeepSeek: 350–3,381 tokens, of which 92–3,123 were reasoning. Generation stats confirm all were billed.

## Caveats

- The amounts are small per request. The issue is the multiplier on any capped workload.
- Billing comes from OpenRouter's records, which pass through upstream costs. Direct-from-provider billing could differ.
- Whether GMI's reasoning is a default "thinking on" setting or a template choice can't be told apart from outside.
