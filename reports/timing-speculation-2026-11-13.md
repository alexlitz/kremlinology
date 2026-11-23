# Decode timing: speculation signatures and hardware bounds (2026-11-13)

**How it was run:**
- Run: `data/kremlinology-timing-speculation-v2` (backed up), 18 pinned OpenRouter targets.
- Probes: 3 per target, 6 repeats each, interleaved. That's 324 requests plus 1 retry, $0.068 reported.
- Settings: temperature 0, `max_tokens` 256.
- Analysis: `experiments/timing_speculation_analysis_v1.py`. Outputs: `reports/timing-speculation-2026-11-13.csv/.png`.

**The probes:**
- `copy`: reproduce a given 240-word text.
- `count`: write 1 to 300.
- `story`: an original story.

**Method:**
- Decode rate = (completion tokens − 1) / (time from first to last token-bearing SSE delta). Content and reasoning deltas both count, because gpt-oss reasoning tokens are included in `completion_tokens`.
- Speculation signature: copy/story and count/story speed ratios, median of 6, with a bootstrap 95% CI. Without speculation, decode speed should not depend on how predictable the text is.

## Speculation signatures (CI excludes 1 for both probes)

| target | story tok/s | copy/story | count/story |
|---|---|---|---|
| Llama-3.3-70B @ Groq | 269 | 2.16 [1.58, 2.52] | 2.23 [1.76, 2.82] |
| Llama-3.3-70B @ SambaNova | 303 | 1.82 [1.60, 1.89] | 1.94 [1.87, 1.99] |
| gpt-oss-120b @ Cerebras | 1,867 | 1.59 [1.14, 1.73] | 1.68 [1.32, 1.83] |
| gpt-oss-120b @ Nebius (fp4) | 304 | 1.52 [1.24, 2.01] | 1.85 [1.34, 2.49] |
| Gemma 4 31B @ Crusoe (bf16) | 29 | 1.58 [0.99, 2.58] | 2.69 [1.80, 4.04] (count only) |

## Flat (no detectable speculation)

| target | story tok/s | copy/story | count/story |
|---|---|---|---|
| Llama-3.3-70B @ CoreWeave (fp16) | 96 | 1.02 [0.97, 1.07] | 0.98 [0.93, 1.04] |
| gpt-oss-120b @ Groq | 475 | 0.97 [0.79, 1.03] | 1.00 [0.89, 1.07] |

Also flat: Llama @ AkashML and DeepInfra, Gemma @ Chutes, DeepInfra and Parasail, gpt-oss @ Baseten and Parasail.

Llama @ Parasail is inconclusive (count/story 3.0, CI 0.8–5.5).

**Groq difference:** Groq speculates on Llama-70B but not on gpt-oss-120b.

## Hardware bounds

Story speed × active-weight bytes gives the minimum memory bandwidth behind one stream. KV reads add to it. Story text is the least speculation-friendly probe, but where a signature exists this is still only an upper estimate of the true bandwidth need.

- **CoreWeave Llama-70B, fp16, no speculation:** 96 tok/s × 141 GB = **13.5 TB/s**. That is more than a single H100 (3.35) or B200 (8), so it implies tensor parallelism across at least 4 H100s or at least 2 B200s.
- **AkashML (fp8), 62 tok/s, and Parasail (fp8), 70 tok/s:** 4.4 and 4.9 TB/s. That is more than one H100, so at least 2 GPUs or an H200-class part.
- **SambaNova Llama, 303 tok/s:** 21 TB/s at fp8 even before the speculation discount. That fits its multi-chip RDU systems.
- **Groq and Cerebras** serve weights from on-chip SRAM, so HBM bounds don't apply.

## Data-quality findings

- **`max_tokens=256` ignored or not applied to reasoning:**
  - GMI Cloud DeepSeek-V3.2: up to 2,113 completion tokens, TTFT 11–33 s, which suggests hidden reasoning.
  - SiliconFlow gpt-oss-120b: up to 1,396 completion tokens.
  - Both are excluded from interpretation.
- **gpt-oss copy runs on Groq, Nebius and Parasail never left the reasoning channel within 256 tokens.** Rates are over reasoning tokens.

## Caveats

- Arrival times are client-side: network and SSE batching add noise, and some providers send several tokens per chunk.
- 6 repeats per cell. Load can vary across the roughly 25-minute run (requests were interleaved).
- Copy and count outputs differ in length and token mix from story, so the ratios aren't pure predictability effects. The flat controls (CoreWeave, Groq gpt-oss) argue against a generic content effect.
- Active-parameter counts are published approximations.

## Pooled update (2026-11-17): runs v2–v6, 19–45 streams per cell

Runs `data/kremlinology-timing-speculation-v2`, `-v3`, `-v4`, `-v4a`–`v4g`, `-v5a`–`v5f`, `-v6a`–`v6d` pooled
(configs `experiments/timing-speculation-v*.json`); adds a `random` probe (random characters at temperature 1: nothing
to speculate on). Every cell has ≥ 19 streams (median ~30). Figure and CSV: `reports/timing-speculation-2026-11-17.png/.csv`.
Bandwidth bound = random-text tok/s × active-weight bytes (the speculation-free speed).

| target | story tok/s | random tok/s [95% CI] | copy/story [95% CI] | count/story [95% CI] | min bandwidth (fp8 / bf16) | n (story/random) |
|---|---|---|---|---|---|---|
| deepseekv32@deepinfra-fp4 | 23 | 15 [13.35, 16.11] | 1.10 [0.95, 1.30] | 0.88 [0.78, 1.03] | 0.5 / 1.1 TB/s | 30/27 |
| deepseekv32@gmicloud-fp8 | 51 | 55 [51.03, 58.44] | 1.21 [1.11, 1.37] | 1.52 [1.36, 1.60] | 2.0 / 4.1 TB/s | 30/19 |
| gemma4-31b@chutes-fp4 | 35 | 35 [32.83, 36.30] | 0.94 [0.89, 1.03] | 1.03 [0.93, 1.09] | 1.1 / 2.1 TB/s | 29/21 |
| gemma4-31b@crusoe-bf16 | 36 | 36 [30.04, 52.49] | 1.78 [1.36, 2.16] | 2.81 [2.07, 3.41] | 1.1 / 2.2 TB/s | 40/21 |
| gemma4-31b@deepinfra-turbo | 92 | 97 [87.79, 100.35] | 0.97 [0.86, 1.11] | 0.98 [0.86, 1.09] | 3.0 / 6.0 TB/s | 26/20 |
| gemma4-31b@parasail-fp8 | 78 | 84 [60.99, 101.93] | 1.00 [0.82, 1.38] | 1.11 [0.96, 1.49] | 2.6 / 5.2 TB/s | 31/25 |
| gptoss120b@baseten-fp4 | 226 | 210 [192.93, 236.59] | 1.05 [0.93, 1.17] | 1.02 [0.90, 1.14] | 1.1 / 2.1 TB/s | 31/24 |
| gptoss120b@cerebras-fp16 | 1767 | 1438 [1266.02, 1633.37] | 1.48 [1.27, 1.61] | 1.60 [1.29, 1.82] | 7.3 / 14.7 TB/s | 44/25 |
| gptoss120b@groq | 474 | 472 [470.70, 473.23] | 0.99 [0.96, 0.99] | 1.00 [0.99, 1.01] | 2.4 / 4.8 TB/s | 39/21 |
| gptoss120b@nebius-fp4 | 319 | 288 [268.42, 313.91] | 1.50 [1.36, 1.70] | 1.66 [1.52, 1.86] | 1.5 / 2.9 TB/s | 38/21 |
| gptoss120b@parasail-fp4 | 265 | 230 [162.45, 299.17] | 1.13 [0.87, 1.26] | 1.05 [0.82, 1.20] | 1.2 / 2.4 TB/s | 27/21 |
| gptoss120b@siliconflow-fp8 | 138 | 136 [132.19, 137.53] | 1.00 [0.97, 1.02] | 1.00 [0.84, 1.02] | 0.7 / 1.4 TB/s | 32/25 |
| llama70b@akashml-fp8 | 61 | 62 [56.94, 63.16] | 0.96 [0.82, 1.02] | 1.03 [0.98, 1.07] | 4.4 / 8.7 TB/s | 26/20 |
| llama70b@coreweave-fp16 | 95 | 97 [91.96, 97.83] | 1.03 [1.00, 1.04] | 0.99 [0.97, 1.03] | 6.8 / 13.6 TB/s | 43/25 |
| llama70b@deepinfra-turbo | 24 | 28 [20.60, 30.36] | 0.92 [0.57, 1.44] | 0.85 [0.54, 1.34] | 2.0 / 3.9 TB/s | 20/21 |
| llama70b@groq | 293 | 161 [149.51, 168.34] | 2.35 [2.16, 2.50] | 2.53 [2.41, 2.61] | 11.4 / 22.7 TB/s | 38/20 |
| llama70b@parasail-fp8 | 55 | 34 [26.34, 41.17] | 1.44 [1.20, 1.78] | 3.26 [2.66, 3.96] | 2.4 / 4.8 TB/s | 41/22 |
| llama70b@sambanova-turbo | 302 | 136 [127.68, 142.07] | 1.85 [1.82, 1.87] | 1.92 [1.88, 1.94] | 9.6 / 19.1 TB/s | 39/21 |

- **Speculation (both ratios' CIs above 1):** Groq Llama (2.35, 2.53), SambaNova Llama (1.85, 1.92), Cerebras gpt-oss
  (1.48, 1.60), Nebius gpt-oss (1.50, 1.66), Crusoe Gemma (1.78, 2.81), Parasail Llama (1.44, 3.26; now resolved),
  GMI DeepSeek (1.21, 1.52).
- **On random text the speculators slow down and the flat ones don't:** Groq Llama 293 → 161 tok/s, SambaNova
  302 → 136, Parasail Llama 55 → 34; CoreWeave 95 → 97, Groq gpt-oss 474 → 472.
- **Bounds from random-text speed** (Llama-70B): CoreWeave fp16 97 tok/s → ≥ 13.6 TB/s (≥ 5 H100s or 2 B200s);
  SambaNova ≥ 9.6 TB/s at fp8; Groq ≥ 11.4 TB/s at fp8 (SRAM, so not an HBM bound); AkashML ≥ 4.4 TB/s.
- The 2026-11-13 table above (6 repeats) is superseded by these pooled numbers; no reading changed except Parasail
  Llama (inconclusive → speculation).

## Two bandwidth-bound probes that did not work (2026-11-22; recorded so they are not mistaken for results)

- **Concurrency scaling** (`concurrency_scaling_v1.py`, one pass, no repeats): per-stream speed did not fall from 1 to
  8 parallel streams at Groq (Llama: 651–690 tok/s), Cerebras (2,887–3,315) or Nebius (513–619); CoreWeave and
  SambaNova returned no usable streams. Uninformative: parallel requests from one client are not guaranteed to share
  a replica, and the prompt ("repeat this sequence 25 times") is a copy task, so the speeds are speculation-inflated
  (Groq Llama 650+ here vs 161 on random text).
- **Decode speed vs context length** (`context_bw_sweep_v1.py`, 4 repeats, same copy prompt): flat from 1k to 64k at
  AkashML (Llama, 23–32 tok/s) and Nebius (gpt-oss, 491–546); Parasail's Gemma 4 31B halves from 68 tok/s at 1k to
  30–32 at 16k–64k. No bandwidth or KV-size fit is claimed: AkashML at this speed is far below any bandwidth bound,
  so the KV term cannot show, and the prompt is wrong for the purpose.
- A redesign needs the random-text probe (no speculation), repeats, and an endpoint that runs near its bound.
