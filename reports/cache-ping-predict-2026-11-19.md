# How well can a prefix's return be predicted? (2026-11-19; no API calls)

Follow-up to `cache-ping-policy-money-2026-11-19.md`, which showed all keep-alive value lies in predicting returns.
Script `experiments/cache_ping_predict_v1.py`; numbers `reports/cache-ping-predict-2026-11-19.json`.

**Setup.** Requests are grouped into conversations by prefix chains (a request continues the most recent conversation
whose block-hash list it extends; ≥ 2 shared blocks, since 1 is just a shared system header). This is the unit a
user-side pinger can keep alive. After each request, a predictor estimates P(another request within the cache
lifetime) from features known at decision time: idle age, requests-so-far, the conversation's own arrival rhythm,
the last output length. Fitted as binned hazard tables on a held-out half of the conversations (dead conversations
included in training, as a deployed model would train on completed histories); scored on the other half. The same
simulation as before, with every ping charged and terminal (post-death) pings counted as waste.

**Predictability (AUC of "returns within the lifetime", death counted as a negative):**

| workload | best features | AUC |
|---|---|---|
| toolagent | idle age + arrival rhythm | **0.88–0.89** |
| arxiv | idle age + arrival rhythm | **0.89** |
| conversation (chained) | requests-so-far | 0.68–0.75 |
| conversation ×8 (slow human) | last output length | 0.61–0.72 |
| synthetic (random by construction) | — | ≈ 0.5 |

**But the money still doesn't clear the bar** ($ per 1M re-presented tokens; Haiku L = 5 min / Kimi L = 10 min):

| workload | let it expire | blind pinging | predictor-gated pinging | perfect prediction |
|---|---|---|---|---|
| toolagent | 2.21 / 0.27 | 8.00 / 3.54 | 2.91 / 0.29 | **0.23 / 0.06** |
| arxiv | 2.19 / 0.28 | 7.77 / 3.43 | 2.88 / 0.30 | **0.22 / 0.06** |
| conversation ×8 | 30.1 / 17.6 | 56.9 / 33.6 | 44.2 / 26.7 | **11.6 / 8.2** |
| conversation | 7.17 / 1.64 | 23.0 / 10.7 | 9.53 / 1.90 | **0.97 / 0.41** |

- **Returns are predictable — AUC 0.88–0.89 from rhythm alone on the agent traces — and it is still not enough.**
  Predictor-gated pinging costs slightly more than letting caches expire at every price point modelled (ρ = 0.10 and
  0.168). A ping must be right about ρ ≈ 10–17% of the time per ping, sustained across a persistence chain, and the
  predictor's residual error sits exactly at that scale.
- **The value is concentrated in the last increment of accuracy**: perfect prediction cuts expiry cost 60–90%
  (toolagent: 2.21 → 0.23, a 10× reduction), an 0.89-AUC predictor captures none of it. Between 0.89 and perfection
  lies all the money.
- What the traces cannot provide: session semantics — whether the agent loop is still scheduled, whether the human
  closed the tab, whether a cron period is known. Those are the features that could plausibly approach the oracle;
  block-hash rhythm does not. (A richer model on the same features — logistic regression, gradients — would add
  perhaps a few points of AUC, not the required jump.)

## Caveats
- Conversation grouping is heuristic (greedy prefix chains, most recent wins); on the conversation trace the
  "conversations" are shared-system-header chains rather than true dialogues.
- Binned histogram predictors, 25+ samples per cell; no model tuning. Haiku/Kimi prices and lifetimes as measured on
  2026-11-18/19 (they move).
- Terminal pings charged until the per-gap cap; the ping interval is 0.9·L throughout.
- One bug found and fixed during the study (misses in (L, 2L] were undercharged when the ping loop broke early);
  the tables above are from the corrected code.

## v2 (later the same day): real models, the right target, and a different answer

`experiments/cache_ping_predict_v2.py`; numbers `reports/cache-ping-predict-v2-2026-11-19.json`. Two corrections to the
study above, then models:

1. **The decision rule was the weak link, not only the predictor.** Pinging on P(return in the *next window*) is
   myopic: it skips cheap alive-pings and later pays full rewrites. The event that matters is a return within the
   *affordable bridge horizon* (⌊w/ρ⌋−1 pings). My first "oracle" had this bug too — random noise beat it by
   accidentally imitating bridging — which is what exposed it.
2. **Expiry was never updated after a ping** in the v2 policy draft, charging phantom misses (fixed; the v1 tables
   above were already correct).

**Predictability, model-based** (logistic regression and a small net on idle age, requests-so-far, arrival rhythm,
last output length, prompt size, conversation age; threshold tuned on a train-side split, never on test; held-out
conversations). Target: P(returns within the bridge horizon):

| workload | logistic AUC | $/1Mtok: miss → predictor-gated (best model) | oracle |
|---|---|---|---|
| toolagent | 0.877 | 2.21 → **1.36** (−38%) / kimi 0.27 → **0.20** (−25%) | 0.26 / 0.10 |
| arxiv | 0.864 | 2.19 → 1.37 (−38%) / 0.28 → 0.22 (−22%) | 0.27 / 0.10 |
| conversation | 0.830 | 7.17 → 4.12 (−43%) / 1.64 → 1.11 (−33%) | 1.06 / 0.62 |

(Raw cost units rescaled; see JSON. The MLP sometimes matched the logistic and sometimes collapsed to not pinging —
the logistic is the robust choice.)

**Accuracy → dollars** (degraded oracle: the bridging answer flipped at random, so errors are independent per
decision — a harsher error structure than real models produce):

| AUC | toolagent Haiku-like cost | toolagent Kimi-like cost |
|---|---|---|
| 1.00 | 1,424 | 591 |
| 0.95 | 2,769 | 868 |
| 0.90 | 3,078 | 1,270 |
| 0.80 | 4,611 | 1,971 |
| 0.60 | 7,593 | 3,883 |
| (no pinging) | 12,342 | 1,582 |

- **At read ratios ≈ 0.10 (Anthropic/OpenAI-like), pinging profits even at AUC ≈ 0.6 under random errors**, and the
  trained models (AUC 0.83–0.88) save 22–43% of expiry cost. The earlier negative result came from the one-window
  decision rule.
- **At read ratios ≈ 0.17 (Moonshot-like) the bar is near AUC 0.9 under random errors** — yet the real logistic at
  AUC 0.72 still profits there (25%), because its errors cluster benignly: it confidently skips dead conversations.
  Headline AUC is not the whole story; the error structure per conversation is.
- **Perfect prediction saves 63–88%** — the remaining 2–3× beyond the models. Closing it needs signals about the
  session itself (agent loop still scheduled, human still typing, known cron period), which block-hash traces do not
  carry. That is the concrete spec for a per-conversation duration predictor.
