"""Recovered fraction of reported cache hits, per endpoint and size (pooled prefill-sweep runs; no API calls).

For each endpoint: fit cold TTFT = overhead + tokens / rate over all cold trials (all sizes, all runs). At each size
S the prefill-attributable time is S / rate. The recovered fraction of a reported hit is
(median cold TTFT - median warm TTFT) / (S / rate): 1 = the hit removed the whole recompute, 0 = it removed nothing.
95% intervals: bootstrap over trials (cold and warm resampled independently, the fit held fixed). Also the floor
test (warm trials faster than the fastest cold trial of that size), the reported cached share and the billed ratio.
Witnessed TTFT (first token-bearing SSE event minus request end, from the relay's record timestamps).
Run: python experiments/prefill_recovery_analysis_v1.py OUT.json RUN_DIR [RUN_DIR ...]
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_routing_honesty_analysis_v1 import generation_of  # noqa: E402
from prefill_cache_sweep_analysis_v1 import token_times  # noqa: E402

BOOT = 2000


def load(runs):
    by = defaultdict(list)
    for k, run in enumerate(runs):
        root = run / 'session'
        ttft = {}
        for line in (root / 'requests.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('status') == 200:
                gen = generation_of(record, root)
                if gen:
                    t = token_times(record)
                    if t:
                        ttft[gen] = t[0]
        for r in json.loads((run / 'log.json').read_text())['rows']:
            if r['status'] != 200 or ttft.get(r['generation']) is None or not r.get('prompt_tokens'):
                continue
            size, trial, kind = r['tag'].split(':')
            by[(r['model'], r['provider'])].append(dict(size=int(size), trial=(k, int(trial)), kind=kind,
                                                        ttft=ttft[r['generation']], prompt=r['prompt_tokens'],
                                                        cached=r['cached_tokens'], cost=r['cost']))
    return by


def main():
    out_path, runs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    rng = np.random.default_rng(20261123)
    report = []
    for (model, provider), rows in sorted(load(runs).items()):
        cold = [r for r in rows if r['kind'] == 'cold' and r['cached'] < 0.1 * r['prompt']]
        if len({r['size'] for r in cold}) < 2:
            continue
        x = np.array([r['prompt'] for r in cold], float)
        y = np.array([r['ttft'] for r in cold], float)
        # robust line: fit on per-size medians (cold TTFTs have queueing outliers)
        sizes = sorted({r['size'] for r in cold})
        mx = np.array([np.median([r['prompt'] for r in cold if r['size'] == s]) for s in sizes])
        my = np.array([np.median([r['ttft'] for r in cold if r['size'] == s]) for s in sizes])
        slope, intercept = np.polyfit(mx, my, 1)
        rate = 1 / slope if slope > 0 else float('inf')
        entry = dict(model=model, provider=provider, prefill_tokens_per_s=rate, overhead_s=float(intercept), sizes={})
        print(f"\n{model.split('/')[1]} @ {provider}: prefill ~{rate / 1000:.0f}k tok/s, overhead {intercept:.2f} s")
        for s in sizes:
            c = np.array([r['ttft'] for r in cold if r['size'] == s])
            warm = [r for r in rows if r['kind'] == 'warm' and r['size'] == s]
            hits = [r for r in warm if r['cached'] >= 0.5 * r['prompt']]
            if len(c) < 3 or len(hits) < 3:
                continue
            w = np.array([r['ttft'] for r in hits])
            compute = float(np.median([r['prompt'] for r in hits])) * slope
            rec = (np.median(c) - np.median(w)) / compute
            draws = [(np.median(rng.choice(c, len(c))) - np.median(rng.choice(w, len(w)))) / compute for _ in range(BOOT)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            cost_c = st.median(r['cost'] for r in cold if r['size'] == s)
            cost_w = st.median(r['cost'] for r in hits)
            e = dict(n_cold=len(c), n_warm=len(warm), reported_hits=len(hits), cold_ttft=float(np.median(c)),
                     warm_ttft=float(np.median(w)), compute_s=compute, recovered=float(rec), lo=float(lo), hi=float(hi),
                     warm_below_fastest_cold=int((w < c.min()).sum()),
                     reported_share=float(np.median([r['cached'] / r['prompt'] for r in hits])),
                     billed_ratio=cost_w / cost_c if cost_c else None)
            entry['sizes'][s] = e
            print(f"  {s // 1000:>3d}k: cold {e['cold_ttft']:.2f} warm {e['warm_ttft']:.2f} s (n={e['n_cold']}/{len(hits)} hits of "
                  f"{e['n_warm']}) | recompute {compute:.2f} s | recovered {rec:.2f} [{lo:.2f}, {hi:.2f}] | below fastest "
                  f"cold {e['warm_below_fastest_cold']}/{len(hits)} | reported {e['reported_share']:.2f} | billed x{e['billed_ratio']:.2f}")
        report.append(entry)
    out_path.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
