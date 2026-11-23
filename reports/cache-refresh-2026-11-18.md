# Does hitting a cache keep it alive, and does a partial hit? (2026-11-18)

Script: `experiments/cache_refresh_probe_v1.py`; analysis `experiments/cache_refresh_analysis_v1.py`; numbers
`reports/cache-refresh-2026-11-18.json`; evidence `data/kremlinology-cache-refresh-v1` (1,440 requests, every
response kept). Spend $3.57. OpenRouter, pinned endpoints, one-token requests, ~5k-token unique prompts (Anthropic
~9.4k, with breakpoints at 50% and 100%).

**Design.** Per endpoint, 20 trials of each arm, every trial with its own prompt:
- control: write at 0, probe with the same prompt at D;
- full ping: write at 0, re-send the whole prompt at P, probe at D;
- partial ping: write at 0, send only the first 25% of the prompt plus new text at P (Anthropic: only the first
  breakpoint, i.e. the first half), probe at D.
D was chosen past each endpoint's lifetime in the earlier retention runs. The probe's cached share says what survived:
the whole prompt, just the head, or nothing. A ping that misses re-writes the entry, so conditional counts use only
trials whose ping hit.

## Results (probe at D: whole prompt cached / head only / nothing)

| endpoint | P → D | control | full ping (ping hit) | partial ping (ping hit) | reading |
|---|---|---|---|---|---|
| Anthropic, Claude Haiku 4.5 | 4 → 7 min | 0 / 0 / 20 | **20** / 0 / 0 (20/20) | 0 / **20** / 0 (20/20) | a hit refreshes the entry it reads, and only that one: pinging the first breakpoint keeps the first half alive, not the longer entry |
| Moonshot, Kimi K2.6 | 5 → 10 min | 1 / 0 / 19 | **20** / 0 / 0 (20/20) | 0 / **20** / 0 (20/20) | refreshed per block (or per prefix): a partial hit keeps exactly the part it touched; the rest expires on the original clock |
| OpenAI, GPT-5.4 nano | 7.5 → 15 min | 0 / 0 / 20 | 13 / 0 / 7 (13/20); **11/13** when the ping hit | 0 / 0 / 20 (**0/20** pings hit) | lifetime counts from the last use; a 25% (~1.3k-token) partial prompt never hit, so it can't be used as a cheap ping |
| Novita, Kimi K2.5 | 20 s → 5 min | 11 / 0 / 9 | 18 / 0 / 1 (17/19) | 19 / 1 / 0 (14/20) | refresh helps (18/19 vs 11/20, one-sided Fisher p = 0.005); with partial pings the whole entry survived, so possibly refreshed as a whole, though 5 of the 6 trials whose partial ping missed survived too |
| Chutes, Gemma 4 31B | 30 s → 5 min | 0 / 0 / 20 | 0 / 1 / 19 (20/20) | 0 / 0 / 20 (1/20) | a ping at 30 s doesn't carry to 5 min; the entry lasts under 4.5 min after its last use, so whether pings refresh it can't be told from this design. Partial prompts don't hit |
| CoreWeave, Llama 3.3-70B | 30 → 60 min | 0 / 0 / 20 | 4 / 0 / 16 (**0/20**) | 0 / 0 / 20 (0/20) | today's lifetime was under 30 min (every ping missed; 15/20 hit at 30 min on Nov 17), so not assessable |
| Nebius, gpt-oss-120b | 10 → 30 min | **20** / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | today the control survived 30 min (0/20 at 30 min in the retention runs): not assessable |
| DeepInfra, DeepSeek-V3.2 | 1 → 10 min | **18** / 0 / 2 | 20 / 0 / 0 | 19 / 0 / 1 | control survived 10 min today (~20% at 5–15 min before): not assessable |
| GMI Cloud, DeepSeek-V3.2 | 30 s → 5 min | **20** / 0 / 0 | 19 / 0 / 0 | 19 / 1 / 0 | control survived 5 min today (4/20 before): not assessable |

## What it means

- **Pinging works where a hit refreshes the lifetime: Anthropic, Moonshot, OpenAI, and very likely Novita.** It saved
  money in the run itself (median cost per trial):

  | endpoint | control (re-pay at D) | full ping + probe | saving on the follow-up |
  |---|---|---|---|
  | Anthropic Haiku 4.5 | probe $0.01176 (a re-write at 1.25×) | $0.00095 + $0.00095 | 84% |
  | Moonshot Kimi K2.6 | $0.00479 | $0.00094 + $0.00094 | 61% |
  | OpenAI GPT-5.4 nano | $0.00101 | $0.00014 + $0.00014 | 72% (when the ping hits; 13/20 at 7.5 min) |

- **A partial hit refreshes only what it touches** at Anthropic (per breakpoint entry) and Moonshot (per block). You
  can't keep a long prompt alive by pinging its first quarter. At OpenAI, Chutes and DeepInfra a quarter-length prompt
  didn't even register as a partial hit (0/20, 1/20 and 12/20 pings hit), so partial pings aren't an option there.
- **More caches that aren't on a fixed clock:** CoreWeave's lifetime was under 30 minutes today (every 30-minute ping
  missed) after holding 15/20 at 30 minutes on Nov 17. Nebius held 30 minutes (0/20 before), DeepInfra 10 minutes
  (~20% before) and GMI 5 minutes (4/20 before). Retention at these providers moves by day or load, so a keep-alive
  schedule tuned on one day can be wrong on the next.
- **Moonshot's unreported reuse, a third time:** all 19 control probes at 10 minutes reported no cached tokens and were
  billed the full price ($0.00479, the same as the write), yet answered in a median 1.16 s, faster than every one of the
  60 cold writes of the same prompts (2.14 s or more). A ping at 5 minutes keeps the hit *reported and discounted*.

## Caveats
- One run (2026-11-19, 00:19–01:34 UTC); four endpoints' lifetimes were longer than planned, so their refresh behaviour
  is untested.
- Chutes: P was too early to separate "no refresh" from "lifetime < 4.5 min after last use".
- Latency is client-side (non-streamed one-token requests).

## How short can a ping's reply be? (`max_tokens=1` in every request above)

| endpoint | completion tokens billed (median / max) | output share of a full ping's cost |
|---|---|---|
| Anthropic, Chutes, CoreWeave, DeepInfra, Nebius | 1 / 1 | ≤ 0.6% |
| Moonshot | 2 / 2 (1 reasoning) | 0.8% |
| OpenAI GPT-5.4 nano | 5 / 6 | 4.6% |
| GMI Cloud DeepSeek-V3.2 | 53 / 5,064 (reasoning) | 5.1% |
| **Novita Kimi K2.5** | **241 / 81,921 (reasoning)** | **58%** |

- **Novita ignores the cap while Kimi K2.5 reasons by default:** 156 of 159 requests produced more than 100 tokens;
  two ran ~35 minutes and generated 81,921 reasoning tokens each, billed $0.24 apiece (a quarter of Novita's $0.89 in
  this run) for a requested 1-token reply. With `reasoning: {enabled: false}` (or `effort: "none"`) the same
  requests return exactly 1 token in about a second ($0.00002; 4/4 checked).
- **GMI keeps reasoning** 40–50 tokens per request even with reasoning disabled (4/4) and occasionally runs away (12/159
  over 100 tokens, up to 5,064): cheap per ping at $0.31/M, but not controllable.
- **Recipe for a keep-alive ping:** the exact cached prefix, a short fixed suffix, `max_tokens: 1`, `temperature: 0`,
  `reasoning: {enabled: false}`. Output is then under 1% of the ping's cost everywhere tested except GMI (~5%) and
  OpenAI (5–6 tokens billed, ~5%).

## v2 (2026-11-19, witnessed): pings that cover almost all of the cached prompt

Script `experiments/cache_refresh_probe_v2.py`; every request through a local `kremlin proxy` to the witness relay
(session `data/kremlinology-cache-refresh-v2-session`: 639 of 642 records witnessed, `kremlin audit
--require-witness` passes; the other 3 are Novita requests the proxy refused: Novita stalled for over 120 s, the
proxy's 120 s read timeout was treated as the end of the response, and the relay's fuller copy no longer matched.
Fixed in kremlin 0.3.3: a timeout or an early close is now an explicit error, and the read timeout is 600 s and
configurable with `--read-timeout`). 640 requests, $2.01. Numbers:
`reports/cache-refresh-v2-2026-11-19.json`. Arms: control; **near** = the first 90% of the prompt plus new text
(Anthropic: both blocks, the second cut to 80% with no breakpoint at its end); **samectx** = the whole prompt with a
different final instruction. Probe cached share (of the whole prompt):

| endpoint | ping → probe | control | near (90%) | same context, new message |
|---|---|---|---|---|
| Anthropic Haiku 4.5 | 4 → 7 min | 0/20 (all ~0) | 20/20 at **~50%** (fell back to the first breakpoint) | **20/20 whole** |
| Moonshot Kimi K2.6 | 5 → 10 min | 0/20 | 20/20 at **~87%**, none whole | **19/20 whole** |
| OpenAI GPT-5.4 nano | 5 → 11 min | 5/20 whole | 15/20 at **~76%**, 3 whole, 2 none | **19/20 whole** |
| Novita Kimi K2.5 | 20 s → 5 min | 19/19 whole | 18/19 whole | 17/18 whole (control survived today: not assessable) |

- **Refresh is per block (Moonshot, OpenAI) or per breakpoint entry (Anthropic), with no rounding up:** a ping that
  covers 90% of the prompt keeps 87% (Moonshot) or ~76% (OpenAI, block-aligned) alive and the tail expires. At
  Anthropic a near-complete prefix without the exact breakpoint refreshes only the earlier breakpoint's entry.
- **The keep-alive that works is the same context with any new final message:** 19–20/20 whole on all three, so a
  ping doesn't have to repeat the last request, only everything before the point you want to keep.
- Consistent with v1: a partial ping refreshes exactly what it touches.

## When to ping: break-even and a simulation on Mooncake traces (2026-11-19)

**Break-even.** With ping interval I, per-ping failure chance q (a failed ping re-writes and costs a full miss) and
ρ = cache-read price ÷ miss price, each interval costs (1−q)·ρ + q in units of one miss, so pinging pays while the
idle gap is shorter than G\* = I / (q + (1−q)·ρ). With I = 0.9·L and q ≈ 0: Anthropic G\* ≈ 56 min, Moonshot ≈ 53 min,
OpenAI ≈ 54 min; at OpenAI's measured 7.5-minute failure rate (35%) G\* drops to ~26 min. Add a per-gap cap of
⌊w/ρ⌋ − 1 pings (w = write premium; 14 at Anthropic, 4–9 elsewhere): bridging a longer gap would cost more than the
miss it avoids.

**Simulation** (`experiments/cache_ping_policy_sim_v1.py`, no API calls): every 512-token block's consecutive reuse
times in the four Mooncake traces; a gap over L is a miss (cost w per block), a ping costs ρ per block and refreshes;
the oracle knows each gap and picks the cheaper of bridging or letting it die. Cost in recomputes per 1,000 block
reuses; both policies are scored only on gaps that end in a reuse (a deployment also wastes pings on dead
conversations, which hurts the fixed policy more):

| trace (L = 5 min, ρ = 0.08–0.10) | gaps > L | no pings | fixed, I = 4.5 min | oracle | headroom |
|---|---|---|---|---|---|
| conversation (agent-like) | 21.5% | 268 | 32 (398 pings) | 29 | 3.1 |
| toolagent | 7.4% | 92 | 8.1 | 7.3 | 0.8 |
| arxiv | 7.5% | 94 | 8.4 | 7.6 | 0.8 |
| synthetic | 8.6% | 108 | 8.0 | 7.6 | 0.4 |

- **A fixed ping at ~90% of the lifetime removes 80–90% of expiry cost** where the lifetime is minutes: conversation
  268 → 32, the others ~95 → 8. At Moonshot's ρ = 0.17 and L = 10 min: 63 → 18.
- **A perfect gap-length predictor adds almost nothing** (headroom 0.4–3 per 1,000, ~1% of the no-ping cost): with
  I = 0.9·L and a per-gap cap, the fixed policy already spends almost all its cost on pings that the oracle would also
  send. Interval tuning is not where prediction pays.
- **Short lifetimes never justify pinging**: at a Novita-like L = 20 s, fixed pinging costs more than doing nothing
  (1,103 vs 864 per 1,000) even with the cap, and the oracle's 676 comes mostly from choosing to let gaps die.
- Where prediction *does* pay: (a) not pinging conversations that are over (excluded here), and (b) Anthropic's
  per-breakpoint lifetime choice (5 min vs 1 h at 2× write), which is one prediction per conversation instead of a
  schedule.

## Witnessed replication (v3, 2026-11-19)

`data/kremlinology-cache-refresh-v3` + `-v3-session` ($3.03, all 1,440 requests witnessed, audit passes;
reasoning left on, as in v1). Core findings reproduce: Anthropic control 0/20 vs full-ping 20/20 whole and partial-ping
20/20 head-only; Moonshot the same (20/20, 19/20 head); OpenAI full ping 8/20 (its 7.5-min pings hit only 4/20 today).
Day-dependent lifetimes again: CoreWeave's under 30 min (all pings missed), Nebius 20/20 at 30 min, Chutes control
16/20 at 5 min (0/20 the first day) -- now attested examples of the instability.