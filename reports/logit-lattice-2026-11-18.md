# Are a provider's logits rounded to bf16 before the softmax? (2026-11-18)

No new requests: the three logprob campaigns (Llama 3.3-70B, Qwen3.5-397B, Gemma 4 31B; top-20 logprobs, ~800
responses per provider). Script: `experiments/logit_lattice_v1.py`; numbers: `reports/logit-lattice-2026-11-18.json`.

**Test.** Within one response, the gap between two tokens' logprobs equals their logit gap (the softmax shift cancels).
If the final-layer logits are stored in bf16, every logit lies on the bf16 grid, and for logits in [2^e, 2^(e+1))
their differences are multiples of 2^(e−7): 1/8 for logits of 16–32, 1/16 for 8–16. With fp32 logits the gaps are
continuous. Score: share of responses whose top-20 gaps all sit (within 2e-4) on a grid of 1/64 or coarser.

| model | on the bf16 grid (> 96% of responses) | off the grid (0%) |
|---|---|---|
| Llama 3.3-70B | AkashML fp8, CoreWeave fp16, Parasail fp8 (steps 1/8 and 1/16) | **SambaNova** |
| Qwen3.5-397B | Alibaba, Novita, Parasail fp8, StreamLake, Venice (mostly 1/8) | — |
| Gemma 4 31B | CoreWeave fp4, NextBit fp4, Novita bf16, Parasail fp8 (mostly 1/16) | **DekaLLM, Modal** |

Example (AkashML, Llama, one response): gaps below the top token 1.75, 4.0, 4.875, 7.0, 10.0, 10.125, 10.5: all
multiples of 1/8.

**Reading.**
- The grid is a property of the numeric pipeline (the output layer's result dtype and where rounding happens), not of
  the weight quantization: fp4, fp8 and bf16-weight deployments all sit on it.
- The off-grid providers compute or keep logits at higher precision, or transform them after rounding. Gemma applies
  a final logit soft-cap (30·tanh(z/30)), so off-grid Gemma logits can also mean the cap is applied in fp32 after a
  bf16 matmul. SambaNova runs its own RDU hardware rather than GPUs; DekaLLM and Modal are the two largest Gemma
  outliers in the TV comparison (9.9% and 11.7% vs CoreWeave), consistent with a different serving stack.
- It is a stack fingerprint, not a vendor identification: the same GPU with a different kernel or engine flag would
  move a provider across the line.

## Caveats
- Top-20 logprobs only, first token; reporting precision is fp32 (gaps are checked to 2e-4).
- Responses whose two logits straddle a power of two fit only the finer grid; the per-response test allows this by
  taking the coarsest step that fits all gaps.
