# Top-P ladder: absolute token probabilities from sampling alone (local ground truth, 2026-11-17)

**The rule.** Under top-P (nucleus) sampling a token can be sampled only if the total probability of the tokens ranked
above it is below P. So if a token ranked below the top K is ever sampled at `top_p = P`, the top-K mass is < P. Not
seeing it in m draws rules "in the nucleus" out only statistically: miss probability (1 − its share)^m.

**Method.** `experiments/top_p_ladder_v1.py`. Llama-3.2-3B-Instruct, fp32 logits on the same 100 dataset prompts as
`logit-reconstruction-2026-11-13`, exact provider sampling order, no API calls.
1. A temperature-1 pilot (a fifth of the budget) names the likely ranks 2–10 and estimates their shares.
2. For K = 1..9, bisect on `top_p` watching the (K+1)-th named token: this brackets C_K, the mass ranked above it
   (miss probability α = 0.01 per probe, using half the pilot's share estimate).
3. A token's own probability is the gap C_{K+1} − C_K, identified only when the two brackets are disjoint and ordered.
   Overlapping pairs (near-ties the pilot may have swapped) are reported as unresolved, not guessed.
The K = 1 case is what `logit_reconstruction_benchmark_v1` already used to make log-probs absolute; this runs every K.

## Results

| samples per prompt | mass above rank K+1 (C_K): coverage / median width | own probability of token K: resolved / coverage / median width |
|---|---|---|
| 10,000 | 100% at every K; 0.0000 (K=1), 0.006 (K=2), 0.13 (K=3), widening to uninformative by K=6 | K=2: 80/98, 100%, 0.0015; K=3: 54/97, 100%, 0.018; K=4: 27/94, 100%, 0.063 |
| 100,000 | 100% at every K; under 0.0005 up to K=5, 0.016 at K=6, 0.25 at K=9 | K=2: 96/100, 100%, < 0.0001; K=3: 88/100; K=4: 79/97; K=5: 60/96 (all 100%, ≤ 0.0001); K=9: 22/84, 100%, 0.0015 |

- **Every bracket covers the truth** (C_K and resolved per-token probabilities, 100% at both budgets).
- **Precision falls with K at a fixed budget**, because lower-ranked targets are rarer and need more draws per probe.
- **Unresolved = genuinely ambiguous order.** A first version that differenced in the pilot's order regardless gave
  negative "probabilities" for 77–97 near-tied pairs; refusing to difference overlapping brackets removes all of them.
- **Uses:** absolute probabilities without logprobs (on top of the relative gaps from hot top-K sampling), and a check
  that a provider applies top-p at all: the toolkit's `kremlin sampler` now sets `top_p` midway between p1 and p1 + p2
  from the reported logprobs, so only the top two may appear (honest fake route: 0 violations; route that ignores
  sampling parameters: caught).

## Caveats
- One 3B model, first token, fp32 truth; hosted servers add their own precision floor (see logit-reconstruction).
- Assumes the provider's nucleus is computed after temperature on the same distribution it samples from (the standard
  order); a provider that orders top-k/top-p differently needs the K = 1 rung only.
- α is per probe with no multiplicity correction across the ~40 bisection steps per rung (coverage was 100% anyway).
