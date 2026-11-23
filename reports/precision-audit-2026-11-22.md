# Declared precision vs measured TV: the long-tail audit (2026-11-22; free, from existing campaigns)

Question (user's): are any providers serving a different precision or model than they claim? Method: the three
dataset-TV campaigns (Llama 3.3-70B vs CoreWeave fp16; Qwen3.5-397B vs Alibaba; Gemma 4 31B vs the bf16 reference)
give per-provider total variation against a reference; providers that *declare* a precision calibrate the ladder, and
undeclared providers are placed on it. Declared quantizations from the campaign-time catalogs (frozen at run time),
cross-checked against today's endpoint listings. `reports/precision-audit-2026-11-22.json` carries the rows.

## The ladder is consistent

| declared | measured TV vs reference (code / wiki) | providers |
|---|---|---|
| bf16 / reference itself | 1.8–5.3% | novita-bf16 (Gemma ref), reference self |
| fp8 | 2.6–8.2% | parasail-fp8 (Gemma, Qwen, Llama), akashml-fp8 |
| fp4 | 9.9–23% | coreweave-fp4, nextbit-fp4, modelrun/fp4 (Gemma) |

Every provider that declares a precision lands in its own band. **No outright false claim found**: nobody declares
bf16 and serves fp4-class outputs. OpenRouter routing is also clean: across 802 pinned-and-succeeded requests in all
our runs, the serving provider always matched the pin (the one flag was a hyphenation artifact).

## The flags: "unknown" quantization, fp4-class outputs

| provider (model) | declares | measured TV (code / wiki) | band | $/req in campaign | fp4 peers' $/req |
|---|---|---|---|---|---|
| **Modal** (Gemma 4 31B) | unknown | 10.5% / 20.1% | **fp4** | **$0.0160** | $0.0050–0.0072 |
| DekaLLM (Gemma 4 31B) | unknown | 8.9% / 20.0% | fp4 | $0.0061 | $0.0050–0.0072 |
| StreamLake (Qwen3.5-397B) | unknown | 8.7% / 14.1% | fp4-or-worse | $0.0268 | fp8 peers $0.023–0.026 |

- **Modal is the standout: fp4-class outputs behind an "unknown" disclosure at 2.2–3.2× the price of every declared
  fp4 competitor** — and, from the output-lattice study, one of only two providers whose logits sit *off* the bf16
  grid (a different numeric stack). Undisclosed low precision at a premium is exactly the "sketchy" pattern worth
  naming, with the caveat that "unknown" is not a false claim and the TV band is evidence about precision, not proof
  (a different-but-legitimate serving stack could widen TV; the lattice result independently says their stack
  differs).
- DekaLLM: fp4-class at a fair fp4 price, also off-grid. Honest pricing, undisclosed precision, odd stack.
- StreamLake: above the fp8 band on Qwen — worse than fp8-class or a different deployment; "unknown" disclosure.
- Venice and Novita also declare "unknown" on Qwen3.5 but sit at self/fp8 levels — and we know what they are
  (Venice = Parasail's backend at a markup; Novita matches Alibaba).

## Paid follow-ups now queued (batch C, approved)
- Variant-pair TV (`variant_pair_tv_v1.py`): parasail fp4~fast, fireworks~spot, venice fp8~ultrafast,
  deepinfra~priority — same-model siblings, are the discount tiers the same weights?
- Concurrency scaling (`concurrency_scaling_v1.py`) and the context-length bandwidth/KV fit (`context_bw_sweep_v1.py`)
  for the GPU-bound question.
- Remaining suspects to extend this audit to (phase 2): Mancer, Ionstream, Wafer, Reka, Makora, Relace on the models
  where references exist (~$0.30 each via logprob TV; adaptive sampling where logprobs are absent).

## Variant pairs (2026-11-22; witnessed, 1,024 requests): one usable pair, and it differs

`experiments/variant_pair_tv_v1.py`. Only one of four planned pairs could be tested with logprobs: the Fireworks tags
returned 404, and Venice and DeepInfra return no logprobs for DeepSeek-V4.1 (those pairs need the sampling-only
method). The script's own TV summary was wrong (it differenced log-probabilities); the numbers below are recomputed
from the witnessed responses.

- **`parasail/fp4` vs `parasail/fast` (gpt-oss-120b) are different deployments.** Each is deterministic against
  itself (top-20 logprobs identical to 0.002 nats on 55/62 and 63/64 repeated prompts) and they never match each
  other (0/64; median largest gap 0.84 nats). The first token is always `<|channel|>` at probability 1, so the total
  variation at this position is zero and says nothing about quality: the difference is in the tail logits, i.e. in
  the numerics (a different quantisation or engine build), not necessarily in what users see. Both sit on a 1/16
  logit grid.
