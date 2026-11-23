# Gemma 4 31B across providers: dataset-average first-token TV (2026-11-13)

**How it was run:**
- Run: `data/kremlinology-gemma4-31b-dataset-tv-v3`, config `configs/bench/gemma4-31b-dataset-tv-v3.json`, $0.049.
- The host stopped the process under system memory pressure (another job) with 4,758 of 4,800 cells attempted; 42 were never sent and 5 failed after retries. It was sealed with `tools/seal_interrupted_run.py`, and unattempted cells count as unusable.
- Prompts: the Llama-70B campaign's 400 (200 CodeAlpaca, 200 wikitext).
- Settings: first token, top-20 logprobs, 2 repeats; failed requests skipped up to 200 (`max_failed_requests`).
- Reference: Novita bf16.
- Analysis: `experiments/dataset_tv_analysis_v1.py` → `derived/dataset_tv_analysis.json`.

## Average TV vs Novita bf16 (allowance 0; assessable prompts; betting 95% CI)

| target | code | wiki |
|---|---|---|
| Novita vs itself | 2.2% [1.2, 6.4] | 5.8–6.2% [3.5, 10.7] |
| Parasail fp8 | 3.0% [2.0, 7.1] | 7.8–8.2% [6.0, 12.6] |
| DekaLLM | 9.9% [7.0, 14.0] | 20.8–21.2% [16.6, 27.9] |
| CoreWeave fp4 | 11.6% [8.8, 16.0] | 23.5–24.2% [18.8, 31.4] |
| Modal | 11.7% [9.1, 16.1] | 21.1–21.7% [15.6, 28.4] |
| NextBit fp4 | 14.6% [11.2, 19.4] | 24.6–25.2% [21.1, 32.7] |

## Findings

1. **Parasail fp8 is close to Novita's own noise.** The other four differ by 4–5× the reference's self-TV. These are the largest provider gaps in the project; Llama-70B's fp8 providers were at 5–7%.
2. **Two causes are visible:**
   - **Thinking-channel handling.** CoreWeave, DekaLLM and Modal put about 4% of first-token mass (mean) on Gemma's thinking-channel token, against 0.3–0.5% at Novita, Parasail and NextBit. Prompt-token counts are identical everywhere, so this is a template or thinking-mode default, not extra prompt text.
   - **Precision.** NextBit fp4 has no extra channel mass but the largest gap.
3. **Prose continuation (wiki) differs about twice as much as code instructions** for every provider, including the reference against itself. Its distributions are flatter, so the same numerical perturbation moves more mass.

## Caveats

- Novita's "bf16" is a declaration, and it is a reference, not ground truth.
- First token only. The thinking-channel attribution rests on top-20 mass, not on reading templates.
