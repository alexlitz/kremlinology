# What does max_tokens cap? Kimi K2.5 on four providers, and Novita's 81,921-token "1-token" replies (2026-11-19)

## How it was found

In the cache-refresh run (`cache-refresh-2026-11-18.md`; direct requests, not witnessed) every request asked for
`max_tokens: 1`. Novita's Kimi K2.5 billed a median 241 completion tokens (p90 396, p99 5,185; 7 of 159 over 1,000).
Two requests billed **81,921**: 81,920 reasoning tokens plus the visible answer "OK", after ~35 minutes each, at
$0.236 apiece. Their reasoning (275k and 339k characters) shows the model "reading the notes" and reciting the
random-word filler of the prompt word by word until it stopped at exactly 81,920 = 80 × 1,024 tokens, a reasoning
ceiling, then answering "OK". The filler is pathological input, so the loop is partly self-inflicted; the billing
exposure is not: a 1-token cap let 81,921 tokens through.

## Controlled check (witnessed)

Script `experiments/reasoning_cap_probe_v1.py`; rows `data/kremlinology-novita-cap-v1/rows.json`; every request
through a local `kremlin proxy` to the witness relay (session `data/kremlinology-novita-cap-v1-session`, 161
records, all with receipts; `kremlin audit --require-witness` passes). Two ordinary prompts (an arithmetic question, a
one-sentence summary) × 5 repeats × 4 settings × 4 providers = 160 requests, $0.05.

| provider | default, max_tokens 1 | default, max_tokens 64 | reasoning.max_tokens 64, max_tokens 1 | reasoning disabled, max_tokens 1 |
|---|---|---|---|---|
| Amazon Bedrock | 1 token, answers (no reasoning by default) | 20 tokens, answers | 1, answers | 1, answers |
| Atlas Cloud (int4) | 1 token, **all reasoning, empty answer** | 64, **all reasoning, empty answer** | 1, empty answer | 1, answers |
| SiliconFlow (int4) | median 194 billed (max 612), answer 1 token | median 241 (max 960) | **65 (budget honoured)** | 1, answers |
| Novita | median 272 billed (max 639), answer 1 token | median 230 (max 776) | **241 (budget ignored)** | 1, answers |

- **Four meanings of max_tokens for one model:** no reasoning at all (Bedrock); a cap on reasoning + answer, so a
  small cap returns an empty answer (Atlas Cloud, OpenAI's semantics); a cap on the visible answer only, with
  reasoning uncapped and billed at the output price (SiliconFlow, Novita).
- **Only Novita also ignores an explicit reasoning budget.** SiliconFlow honours `reasoning.max_tokens`.
- **Disabling reasoning is the one setting all four honour** (1 token billed, an answer returned).
- Cost of a "1-token" request to a short prompt: Novita ~$0.0009, Bedrock ~$0.00004 (median, same prompt).

## Caveats
- One model; reasoning token counts are as billed by OpenRouter (provider-reported).
- The 81,920-token ceiling is inferred from two capped runs; Novita doesn't document it.
- The runaway loop was triggered by random-word filler; on the ordinary prompts here the maximum was 776 tokens.
