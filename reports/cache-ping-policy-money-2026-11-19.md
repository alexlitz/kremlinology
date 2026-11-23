# Would keep-alive pings have saved money? Honest accounting, and what prediction is worth (2026-11-19)

No API calls: `experiments/cache_ping_policy_sim_v2.py` on the four Mooncake traces (plus the conversation trace with
timestamps stretched ×3 and ×8, a slower human in the loop). Numbers:
`reports/cache-ping-policy-money-2026-11-19.json`. This corrects the first simulation
(`cache_ping_policy_sim_v1.py`), which scored pinging only on idle gaps that ended in a reuse — clairvoyant about
which prefixes were dead. Everything here charges those pings.

**Model.** Per 512-token block, its use times from the trace. A gap over the lifetime L is a miss (the block is
recomputed; cost w = 1, or 1.25 at Anthropic's write premium). A ping costs ρ (the cache-read price ratio: 0.10 for
Claude Haiku 4.5, 0.168 for Kimi K2.6) and refreshes. Ping interval I = 0.9·L; both pinging policies give up after
⌊w/ρ⌋−1 pings per idle period. **Pings sent after a block's last-ever use are waste**, charged to the fixed and
hazard policies (the oracle knows). The hazard policy pings only while the fitted probability of a return within I
(estimated on a held-out half of the blocks, from idle age and the block's uses-so-far) exceeds ρ. Cost is converted
at list prices per 1M tokens of re-presented prefix.

## Results ($ per 1M re-presented prefix tokens; miss = doing nothing)

| workload (Haiku 4.5: L = 5 min, ρ = 0.10) | miss | fixed ping | hazard ping | oracle |
|---|---|---|---|---|
| conversation (dense, 1-hour trace) | 0.304 | 0.563 (92% of pings wasted) | 0.563 | 0.041 |
| conversation ×3 (slow human) | 0.759 | 0.738 | 0.724 | 0.178 |
| conversation ×8 (very slow) | 1.158 | 1.207 | 1.119 | 0.470 |
| toolagent | 0.099 | 0.216 | 0.216 | 0.010 |
| arxiv | 0.100 | 0.215 | 0.215 | 0.010 |
| synthetic | 0.108 | 0.268 | 0.133 | 0.009 |

Kimi K2.6 (L = 10 min, ρ = 0.168) behaves the same with smaller numbers (misses 0.009–0.689).

In whole-workload terms (conversation trace: 145M input tokens, 24M re-presented prefix tokens): at Haiku list
prices the workload's input bill is ~$145, of which expiry misses are ~$7.3 (5%). Pinging naively raises that to
~$13.5; a perfect predictor cuts it to ~$1.0. Per $100 of input spend: misses ≈ $5, oracle ≈ $0.7, naive pinging
≈ $9.3.

## Findings

- **Naive fixed pinging loses money on every natural pattern simulated** (it only breaks even on the ×8 stretched
  trace, and only just). 73–99% of its pings go to prefixes that never return: most blocks (76–78% in these traces)
  are used exactly once. The earlier "removes 80–90% of expiry cost" result was an artefact of scoring only gaps
  that ended in a reuse.
- **All the value is in prediction.** The oracle — knowing which prefixes return and when — removes 59–91% of the
  expiry cost. That is the entire prize; the interval/cap tuning is worth little once death-waste is charged.
- **Simple prefix statistics predict returns poorly.** The hazard policy (idle age + uses-so-far, fitted out of
  sample) beats fixed pinging only on the slowest traces (×8: waste 52% → 37%; synthetic: 0.268 → 0.133) and ties it
  elsewhere. From the prefix's own history, "will this ever come back" is close to unpredictable.
- **Where prediction could actually work** — and what these traces cannot test: the missing signal is
  *whose* prefix it is. An agent mid-loop, a chat session with a human still typing, a cron job with a known period
  are predictable from session state, not from block statistics; a one-shot eval request is predictably dead. A
  per-conversation duration/continuation predictor attaches exactly there: at write
  time choose Anthropic's 5-min vs 1-h lifetime, and during idle ping only conversations whose predicted return
  probability exceeds ρ. On these traces that signal is unavailable, so our oracle bounds what it could be worth
  (up to ~4–5% of the input bill at Haiku-like prices, ~60–90% of all expiry cost).

## Caveats
- One hour per trace; lifetimes and prices from 2026-11-18 measurements (and they move day to day).
- Block-level accounting: a ping of a B-block prefix is charged as B block-reads; correct in aggregate.
- The traces have no user/session identity, so the per-conversation predictor is bounded, not measured.
- Moonshot's unreported reuse would change its economics further (see `cache-refresh-2026-11-18.md`).
