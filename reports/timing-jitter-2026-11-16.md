# Inter-token latency jitter: load, stalls or speculation? (2026-11-16)

Re-analysis of the two full timing runs (2026-11-13; `kremlinology-timing-speculation-v2` and `-v3`, 18 provider/model
pairs, copy / count / story prompts at temperature 0, 6–16 streams per cell, one request at a time). No new requests.
Script: `experiments/timing_jitter_analysis_v1.py`; data: `reports/timing-jitter-2026-11-16.json`; figure:
`reports/timing-jitter-2026-11-16.png`.

**Method.** Client-side SSE arrival times. Chunks arriving within 2 ms (one network read) are merged into a
*delivery*. Per stream: tokens per delivery (completion tokens / deliveries), and the share of decode time spent in
*stalls*, gaps longer than 4× the stream's own median gap. Speculation should make predictable text (copy, count)
arrive in bigger bursts than story text, consistently across repeats; a second story prompt (v3) is the control.
Load or scheduling stalls should not depend on content and should vary between repeats.

## Bursts: speculation shows in the shape of the stream

| pair | tokens per delivery: copy / count / story (story 2) | burst ratio |
|---|---|---|
| Parasail, Llama 3.3-70B fp8 | 11.8 / 25.6 / 8.4 (8.5) | 3.0× |
| Crusoe, Gemma 4 31B bf16 | 5.1 / 8.8 / 2.9 (3.9) | 3.0× |
| SambaNova, Llama 3.3-70B | 26.7 / 32.0 / 17.1 (17.1) | 1.9× |
| Nebius, gpt-oss-120b fp4 | 10.2 / 13.5 / 8.3 (8.1) | 1.6× |
| Groq, Llama 3.3-70B | 8.2 / 8.0 / 5.5 (5.6) | 1.5× |
| flat: CoreWeave, AkashML, DeepInfra (3), Chutes, Baseten, Parasail (Gemma, gpt-oss), SiliconFlow, GMI, Groq gpt-oss | | 0.97–1.09 |

- Matches the throughput test where both apply (Groq Llama yes, Groq gpt-oss no; SambaNova, Nebius yes; CoreWeave no).
- Adds two: Parasail's Llama and Crusoe's Gemma burst 3× on predictable text (counting especially), with the
  second story matching the first, so the effect follows content, not chance.
- Cerebras can't be read this way: deliveries saturate at 32 tokens on every prompt.

## Stalls: load and scheduling

| pair | streams with stalls | median share of decode time in stalls (p10–p90 across streams) |
|---|---|---|
| DeepInfra, Llama 3.3-70B | 17/18 | 27% (6–56%) |
| Crusoe, Gemma 4 31B | 48/66 | 24% (0–45%) |
| DeepInfra, Gemma 4 31B | 16/18 | 17% (0–34%) |
| Parasail, Gemma 4 31B | 14/18 | 14% (0–37%) |
| Groq, gpt-oss-120b | 40/66 | 11% (0–38%) |
| SiliconFlow, gpt-oss-120b | 15/18 | 5% (0–11%) |
| CoreWeave, Llama 3.3-70B | 5/66 | 0% |
| SambaNova, Llama 3.3-70B | 2/64 | 0% |
| Cerebras, gpt-oss-120b | 1/23 | 0% |
| AkashML, Llama 3.3-70B | 1/18 | 0% |
| GMI Cloud, DeepSeek-V3.2 | 0/18 | 0% |

- DeepInfra's and Parasail's Gemma stalls are similar on every prompt type (10–18% of decode time) and vary widely
  between repeats of the same prompt: the signature of load or batch scheduling (other requests' prefill
  interleaved), not content.
- DeepInfra's Llama stalls more on copy and count (30%) than on stories (13%) but with no bigger bursts (2.7–3.0
  tokens per delivery on every prompt), so it isn't the speculation pattern; the cause isn't identified.
- Groq's Llama is different: stalls take 52–54% of decode time on copy and count vs 16–19% on stories. Those pauses
  are the other half of its speculative bursts (accepted runs delivered at once, then a gap), not load.

## Caveats
- Client-side timing over OpenRouter: deliveries depend on how the provider and OpenRouter flush chunks and on the
  network; a provider that buffers more looks smoother.
- One request at a time, so stalls can't be compared across providers moment by moment (shared causes such as
  OpenRouter itself aren't separated).
- 6 streams per cell in v2; 12–18 in v3.
