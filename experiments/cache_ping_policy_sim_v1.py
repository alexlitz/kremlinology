"""When does a keep-alive ping pay? Simulated on Mooncake request traces (no API calls; 2026-11-19).

Uses the per-block reuse gaps of the Mooncake traces (the same loader as kv_reuse_tiers_v1): for every 512-token
block, the times of its consecutive uses. A gap G between uses either fits inside the cache lifetime L (hit, free) or
the block is recomputed at the next use (cost 1, in units of one block recompute = its input price). A ping during the
gap costs rho per block (the cache-read price ratio). Policies per gap:
  none:    cost 1 if G > L else 0;
  fixed(I): pings every I seconds while the gap is open; a ping before expiry costs rho and refreshes; a ping after
           expiry costs 1 (it re-writes) and revives the entry;
  oracle:  knows G: pays min(1, (ceil(G/L) - 1) * rho) -- ping just before each expiry, or let it die if that is dearer.
Both policies are scored only on gaps that end in a reuse (a deployment also wastes pings on dead conversations,
which hurts fixed more than oracle). The fixed-minus-oracle difference is the most any gap-length predictor could
save. Endpoint parameters from the refresh runs: lifetime L, read ratio rho, and a write premium w (Anthropic) that
multiplies the miss cost.
Output: reports/cache-ping-policy-2026-11-19.json plus a printed table (cost per 1,000 block reuses).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from kv_reuse_tiers_v1 import TRACES, load_trace, block_events  # noqa: E402

# name, lifetime_s, read_ratio rho, write premium w (miss costs w), ping intervals to try (s)
ENDPOINTS = [
    ('anthropic-5m', 300, 0.08, 1.25, [240, 270, 300]),
    ('openai-5m', 300, 0.10, 1.0, [240, 270, 300]),
    ('openai-10m', 600, 0.10, 1.0, [480, 540, 600]),
    ('moonshot-10m', 600, 0.17, 1.0, [300, 480, 540, 600]),
    ('novita-20s', 20, 0.17, 1.0, [16, 18, 20]),
]
TRACES_FILES = ['conversation_trace.jsonl', 'toolagent_trace.jsonl', 'arxiv_mooncake_trace.jsonl',
                'synthetic_trace.jsonl']


def gap_costs(gaps, L, rho, w, I, cap=None):
    """(cost_none, cost_fixed, cost_oracle, misses_none, pings_fixed) over a list of gap seconds. cap: stop pinging a
    gap after this many pings (bridging longer gaps would cost more than the miss it avoids)."""
    none = fixed = oracle = misses = pings = 0.0
    for g in gaps:
        if g <= L:
            continue  # fits: free under every policy
        none += w
        misses += 1
        oracle += min(w, (math.ceil(g / L) - 1) * rho)
        if I is not None:
            expiry, t, n = L, I, 0
            while t < g and (cap is None or n < cap):
                if t < expiry:
                    fixed += rho
                    expiry = t + L
                else:
                    fixed += w  # the ping re-writes
                    expiry = t + L
                pings += 1
                n += 1
                t += I
            if g - (expiry - L) > L:  # died between the last refresh and the reuse
                fixed += w
                misses += 0.001  # not counted; kept for clarity
    return none, fixed, oracle, misses, pings


def main():
    out = {}
    for name in TRACES_FILES:
        rows = load_trace(name)
        uses = block_events(rows)
        gaps_by_block = [b - a for events in uses.values() for a, b in zip([t for t, _ in events], [t for t, _ in events][1:])]
        span = rows[-1]['timestamp'] / 1000 - rows[0]['timestamp'] / 1000
        entry = dict(trace=name.split('_')[0], requests=len(rows), span_hours=round(span / 3600, 1),
                     block_reuses=len(gaps_by_block), endpoints={})
        print(f"\n{entry['trace']}: {len(rows)} requests over {span / 3600:.1f} h, {len(gaps_by_block)} block reuses")
        for label, L, rho, w, intervals in ENDPOINTS:
            over = [g for g in gaps_by_block if g > L]
            none_c, _, oracle_c, _, _ = gap_costs(gaps_by_block, L, rho, w, None)
            cap = max(0, int(w / rho) - 1)
            results = [gap_costs(gaps_by_block, L, rho, w, I, cap) for I in intervals]
            best = min(results, key=lambda r: r[1])
            e = dict(lifetime_s=L, rho=rho, write_premium=w, reuse_gaps_over_lifetime=len(over),
                     share_over=round(len(over) / max(1, len(gaps_by_block)), 4),
                     cost_none_per_1000=round(none_c / len(gaps_by_block) * 1000, 2),
                     cost_oracle_per_1000=round(oracle_c / len(gaps_by_block) * 1000, 2),
                     ping_cap_per_gap=cap)
            e.update(cost_fixed_per_1000=round(best[1] / len(gaps_by_block) * 1000, 2),
                     best_interval_s=intervals[results.index(best)],
                     pings_per_1000=round(best[4] / len(gaps_by_block) * 1000, 1),
                     fixed_minus_oracle_per_1000=round((best[1] - oracle_c) / len(gaps_by_block) * 1000, 2))
            entry['endpoints'][label] = e
            print(f"  {label:14s} L={L:>4}s rho={rho:.2f}: gaps>L {len(over):6d} ({e['share_over']:.1%}) | "
                  f"cost/1000 reuses: none {e['cost_none_per_1000']:7.2f} -> fixed(I={e['best_interval_s']}s) "
                  f"{e['cost_fixed_per_1000']:7.2f} ({e['pings_per_1000']:.0f} pings), oracle {e['cost_oracle_per_1000']:7.2f} "
                  f"| predictor headroom {e['fixed_minus_oracle_per_1000']:6.2f}")
        out[name] = entry
    (ROOT / 'reports/cache-ping-policy-2026-11-19.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
