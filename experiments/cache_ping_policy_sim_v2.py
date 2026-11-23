"""Would a keep-alive ping policy have saved money, and can the need be predicted? (2026-11-19; no API calls)

Block-level simulation on the Mooncake traces (loader shared with kv_reuse_tiers_v1), extended from
cache_ping_policy_sim_v1 with three things v1 lacked:
  honest death: pings sent after a block's last use are waste, charged to fixed/hazard policies (the oracle knows).
                A fixed pinger gives up after cap pings per idle period; a hazard pinger stops when the return
                probability drops below the ping's cost.
  prediction:   policies are evaluated on a held-out half of the blocks (split by hash). The predictor is the
                empirical hazard h(t) = P(a block returns within the next I s | idle for t s), fitted on the train
                half, optionally conditioned on the block's prior-use count (popularity).
  money:        costs in recompute-equivalents per 1M re-presented prefix tokens, converted at real list prices.
Variants: the four traces plus the conversation trace with timestamps stretched x3 and x8 (a slower human in the
loop). Ping interval I = 0.9 L; a failed ping (sent after expiry) re-writes and costs w.
Run: python experiments/cache_ping_policy_sim_v2.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from kv_reuse_tiers_v1 import TRACES, load_trace, block_events  # noqa: E402

BLOCK = 512
ENDPOINTS = [  # label, lifetime_s, rho (read/miss price), w (write premium)
    ('haiku4.5', 300, 0.10, 1.25),
    ('kimi-k2.6', 600, 0.168, 1.0),
]
PRICES = {'haiku4.5': dict(input=1.00, read=0.10), 'kimi-k2.6': dict(input=0.95, read=0.16)}  # $ per M tokens
VARIANTS = [('conversation', 'conversation_trace.jsonl', 1), ('conversation-x3', 'conversation_trace.jsonl', 3),
            ('conversation-x8', 'conversation_trace.jsonl', 8), ('toolagent', 'toolagent_trace.jsonl', 1),
            ('arxiv', 'arxiv_mooncake_trace.jsonl', 1), ('synthetic', 'synthetic_trace.jsonl', 1)]
AGE_BINS = [0, 30, 60, 120, 240, 480, 960, 1920, 1e9]
POP_BINS = (1, 2, 4, 8, 1 << 30)  # prior-use count buckets


def block_timelines(rows, stretch):
    uses = {}
    for r in rows:
        t = r['timestamp'] / 1000 * stretch
        for block in set(r['hash_ids']):
            uses.setdefault(block, []).append(t)
    return {b: sorted(ts) for b, ts in uses.items()}


def fit_hazard(train, I):
    """Two predictors of a block returning within I s: h(age_s, prior_uses) and h_age(age_s), fitted on the train half."""
    at_risk, returns = defaultdict(int), defaultdict(int)
    at_risk_age, returns_age = defaultdict(int), defaultdict(int)
    for times in train:
        for k, (a, b) in enumerate(zip(times, times[1:])):
            pop = min(POP_BINS, key=lambda p: abs(p - (k + 1)))  # uses so far, no peeking at the future
            if a >= AGE_BINS[-1]:
                continue
            age_bin = bin_age(a)
            at_risk[(age_bin, pop)] += 1
            at_risk_age[age_bin] += 1
            if b - a <= I:
                returns[(age_bin, pop)] += 1
                returns_age[age_bin] += 1
    table = {k: returns[k] / n for k, n in at_risk.items() if n >= 30}
    table_age = {k: returns_age[k] / n for k, n in at_risk_age.items() if n >= 30}
    return (lambda t, pop: table.get((bin_age(t), pop), 0.0)), (lambda t, pop: table_age.get(bin_age(t), 0.0))


def bin_age(t):
    for age in AGE_BINS:
        if t < age:
            return age
    return AGE_BINS[-1]


def simulate(times_list, L, rho, w, I, policy, hazard=None, cap=None):
    """Cost over a list of per-block use-time lists. Terminal pings (after the last use) are charged to fixed/hazard."""
    cost = pings = waste = 0
    for times in times_list:
        for k, (a, b) in enumerate(zip(times, times[1:])):
            pop = min(POP_BINS, key=lambda p: abs(p - (k + 1)))
            g = b - a
            if g <= L:
                continue
            if policy == 'oracle':
                cost += min(w, (-(-g // L) - 1) * rho)  # ceil(g/L)-1 pings at rho, or let it die
                continue
            expiry, t, n = L, I, 0
            while t < g and (cap is None or n < cap):
                if policy == 'hazard' and hazard(t, pop) < rho:
                    break
                if t < expiry:
                    cost += rho
                    expiry = t + L
                else:
                    cost += w
                    expiry = t + L
                pings += 1
                n += 1
                t += I
            if g - (expiry - L) > L:
                cost += w
        if policy in ('fixed', 'hazard'):  # the tail after the last use: pinging a dead prefix
            expiry, t, n = L, I, 0
            pop = min(POP_BINS, key=lambda p: abs(p - len(times)))  # after death, all uses are 'so far'
            while t < AGE_BINS[-1] and n < cap:
                if policy == 'hazard' and hazard(t, pop) < rho:
                    break
                cost += rho
                pings += 1
                waste += 1
                n += 1
                t += I
    return cost, pings, waste


def main():
    out = {}
    for vname, fname, stretch in VARIANTS:
        rows = load_trace(fname)
        uses = block_timelines(rows, stretch)
        blocks = sorted(uses)
        train = [uses[b] for b in blocks[::2]]
        test = [uses[b] for b in blocks[1::2]]
        reused = [t for t in test if len(t) > 1]
        reuses = sum(len(t) - 1 for t in reused)
        entry = dict(variant=vname, trace=fname.split('_')[0] + ('' if stretch == 1 else f' x{stretch}'),
                     requests=len(rows), blocks=len(blocks), reused_blocks=len(reused), block_reuses=reuses, endpoints={})
        print(f"\n{vname}: {len(rows)} requests, {len(blocks)} blocks, {reuses} block reuses "
              f"({len(reused)} blocks reused)", flush=True)
        for label, L, rho, w in ENDPOINTS:
            I = int(0.9 * L)
            cap = max(0, int(w / rho) - 1)
            h, h_age = fit_hazard(train, I)
            none_c = sum(w for t in reused for a, b in zip(t, t[1:]) if b - a > L)
            fixed = simulate(reused, L, rho, w, I, 'fixed', None, cap)
            hazard = simulate(reused, L, rho, w, I, 'hazard', h, cap)
            hazard_age = simulate(reused, L, rho, w, I, 'hazard', h_age, cap)
            oracle_c = simulate(reused, L, rho, w, I, 'oracle')[0]
            tok = reuses * BLOCK / 1e6  # M tokens of prefix re-presented
            p = PRICES[label]
            # cost is in block-recompute equivalents; per block-reuse that is per 512 re-presented tokens, so
            # $ per 1M re-presented tokens = (cost / reuses) x input price (rho and w are already price ratios)
            per_m = lambda c: c / max(reuses, 1) * p['input']
            e = dict(lifetime_s=L, rho=rho, cap=cap, interval_s=I,
                     miss_cost_mtok=round(per_m(none_c), 4),
                     fixed_mtok=round(per_m(fixed[0]), 4),
                     hazard_mtok=round(per_m(hazard[0]), 4),
                     hazard_age_only_mtok=round(per_m(hazard_age[0]), 4),
                     oracle_mtok=round(per_m(oracle_c), 4),
                     fixed_pings=round(fixed[1] / max(reuses, 1) * 1000, 1),
                     hazard_pings=round(hazard[1] / max(reuses, 1) * 1000, 1),
                     fixed_wasted_pings=round(fixed[2] / max(fixed[1], 1), 3),
                     hazard_wasted_pings=round(hazard[2] / max(hazard[1], 1), 3),
                     hazard_note='fitted on the train half; see json')
            entry['endpoints'][label] = e
            print(f"  {label:10s} L={L}s: miss $/Mtok {e['miss_cost_mtok']:.3f} -> fixed {e['fixed_mtok']:.3f} "
                  f"({e['fixed_pings']:.0f} ping/k-reuse, {e['fixed_wasted_pings']:.0%} wasted) "
                  f"hazard {e['hazard_mtok']:.3f} ({e['hazard_wasted_pings']:.0%} wasted), "
                  f"age-only {e['hazard_age_only_mtok']:.3f}, oracle {e['oracle_mtok']:.3f}")
        out[vname] = entry
    (ROOT / 'reports/cache-ping-policy-money-2026-11-19.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
