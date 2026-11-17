# Later-token forcing pilot: OpenRouter, Llama 3.3-70B

Date: 2026-11-10. 254 paid requests, reported cost $0.0108. All HTTP 200.

**Evidence:**
- `data/kremlinology-later-token-pilot-v1` (manifest, `requests.jsonl`, raw captures, `results.json`, `summary.json`).
- `data/kremlinology-later-token-pilot-v1-continue-flag` (the four flag tests).
- Script: `experiments/later_token_forcing_pilot_v1.py`.

## Question

Can the CoreWeave (fp16), AkashML (fp8), Parasail (fp8) and SambaNova routes be made to produce next-token distributions after an identical forced history? That would allow later-token TV and a reference-sampled upper bound on sequence-level TV.

## Method

- Ten prompts from the dataset campaign (5 code, 5 wiki).
- CoreWeave sampled a 12-token history for each prompt (T=1).
- Each provider was asked for the token at positions t ∈ {1, 4, 8} by two methods:
  - **chat prefill:** a final partial assistant message;
  - **raw `/completions`:** the Llama 3 template applied locally.
- Exact token IDs came from the Llama 3.1 tokenizer. On all 3,200 campaign requests, its chat-template token counts matched every provider exactly, and all 67,201 reported tokens mapped to vocabulary IDs.

## Results

| Check | Result |
| --- | --- |
| Forced prefix re-tokenizes to the generated IDs | 30/30, including across the template boundary |
| Chat prefill: reported minus expected prompt tokens | 0 at t=1, **+5 at t=4 and t=8 on all four providers** (`<|eot_id|>` plus a new assistant header: the turn is closed, not continued) |
| Chat prefill plus `continue_final_message` / `add_generation_prompt=false` | still +5 on all four (flags not honoured or stripped) |
| Raw `/completions`: reported minus expected prompt tokens | **+34/+35 on all four** (the raw text was wrapped in a second chat template) |
| Faithfulness: CoreWeave's forced logprob of its own sampled token vs its generation-time logprob | matches only at t=1 (median difference 8.5e-5); later positions off by up to about 12 nats |

## Conclusion

Teacher forcing is not available for these routes through OpenRouter. The prompt-token deltas show this directly. Without the token-ID and count checks, both methods would have returned plausible-looking but wrong "later-token" distributions.

Options:

- **Shared greedy paths.** Compare positions where all providers' greedy continuations have so far produced identical token IDs. Valid conditional comparisons, but on agreed high-probability paths, with an agreement survival curve.
- **Direct provider APIs** that support raw-prompt completions, or vLLM's `continue_final_message`. These need separate accounts.
