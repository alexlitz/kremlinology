"""Recompute the hosted adaptive TV results from the raw samples (adaptive_tv_hosted_v1/v2 runs).

Why post hoc: the runs' live intervals used the basic bootstrap, which collapses to [0, 0] when two providers agree
(code-0172 stopped after 1,000/500 samples with "0.000 [0.000, 0.000]"); adaptive_tv_v1.interval now keeps the upper
bound at or above the percentile bound. Every prompt is recomputed here with BOOT draws from the samples actually
collected (each sample row is kept in samples.jsonl).

Inputs: v2's samples (rows carry their probe) and v1's first prompt (code-0094, the only prompt with DeepInfra
samples before its upstream pool started returning 429s; v1 rows carry no probe, so the first 4,000 successful rows
per provider are prompt 1, checked against v1's results.json).
Also per prompt: the logprob-path TV (CoreWeave vs Parasail, from v2's temperature-0 logprob requests) and each
provider's sampled distribution vs its own logprobs. These differ because the logprob path adds the Llama default
system block (reports/logprob-template-2026-11-17.md).
Top-up runs (adaptive_tv_hosted_v2.py --fixed, e.g. v3 = DeepInfra's 6 missing prompts, v3b = code-0172) are pooled
into v2's prompts.
Run: python experiments/adaptive_tv_hosted_analysis_v1.py V1_DIR V2_DIR [TOPUP_DIR ...]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import adaptive_tv_v1 as atv  # noqa: E402

BOOT = 400
SETTINGS = {(1.0, 0, 1.0): 'plain', (2.0, 10, 1.0): 'hot'}
NAMES = {'coreweave/fp16': 'coreweave', 'parasail/fp8': 'parasail', 'deepinfra/turbo': 'deepinfra'}
PAIRS = [('coreweave', 'parasail'), ('coreweave', 'deepinfra'), ('parasail', 'deepinfra')]


class Counts:
    def __init__(self):
        self.plain, self.hot = Counter(), Counter()

    @property
    def used(self):
        return sum(self.plain.values()) + sum(self.hot.values())


def load(v1, v2, extra=()):
    data = defaultdict(lambda: defaultdict(Counts))  # probe -> provider -> counts
    first = json.loads((v1 / 'results.json').read_text())['results'][0]
    quota = Counter()
    for line in (v1 / 'samples.jsonl').read_text().splitlines():
        r = json.loads(line)
        if 'token' not in r:
            continue
        name = NAMES[r['provider']]
        if quota[name] >= first['samples'][name]:
            continue
        quota[name] += 1
        c = data[('v1', first['probe'])][name]
        getattr(c, SETTINGS[tuple(r['setting'])]).update([r['token']])
    for name, c in data[('v1', first['probe'])].items():  # the split must reproduce v1's own estimates
        est = atv.estimate(c.plain, c.hot)
        assert all(abs(est.get(k, 0) - v) < 1e-9 for k, v in first['estimates'][name].items()), name
    for line in [l for d in (v2, *extra) for l in (d / 'samples.jsonl').read_text().splitlines()]:
        r = json.loads(line)
        if 'token' in r:
            getattr(data[('v2', r['probe'])][NAMES[r['provider']]], SETTINGS[tuple(r['setting'])]).update([r['token']])
    laws = {r['probe']: r['logprob_laws'] for r in json.loads((v2 / 'results.json').read_text())['results']}
    return data, laws


def main():
    v1, v2 = Path(sys.argv[1]), Path(sys.argv[2])
    data, laws = load(v1, v2, [Path(a) for a in sys.argv[3:]])
    rng = np.random.default_rng(20261117)
    atv.BOOT = BOOT
    rows = []
    for (run, probe), providers in sorted(data.items(), key=lambda x: (x[0][1], x[0][0])):
        row = dict(run=run, probe=probe, samples={n: c.used for n, c in providers.items()}, pairs={}, self={})
        est = {n: atv.estimate(c.plain, c.hot) for n, c in providers.items()}
        row['estimates'] = est
        for a, b in PAIRS:
            if a in providers and b in providers and min(providers[a].used, providers[b].used) >= 400:
                draws = atv.bootstrap(providers[a], providers[b], rng)
                point, lo, hi = atv.interval(atv.tv(est[a], est[b]), draws)
                row['pairs'][f'{a}~{b}'] = dict(tv=point, lo=lo, hi=hi)
        law = laws.get(probe) if run == 'v2' else None
        if law and all(law.get(n) for n in ('coreweave', 'parasail')):
            row['logprob_path_tv'] = atv.tv(law['coreweave'], law['parasail'])
            for n in ('coreweave', 'parasail'):
                row['self'][n] = atv.tv(est[n], law[n])  # sampled (no-logprob path) vs own logprobs (logprob path)
        rows.append(row)
        pairs = '  '.join(f"{k} {v['tv']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in row['pairs'].items())
        extra = (f"  logprob-path cw~ps {row['logprob_path_tv']:.3f}  sampled-vs-own-logprobs cw {row['self']['coreweave']:.3f} "
                 f"ps {row['self']['parasail']:.3f}") if 'logprob_path_tv' in row else ''
        print(f"{run} {probe:10s} n={row['samples']}  {pairs}{extra}")
    v2rows = [r for r in rows if r['run'] == 'v2' and 'coreweave~parasail' in r['pairs']]
    summary = dict(
        prompts=len(v2rows),
        median_tv_sampled_cw_ps=float(np.median([r['pairs']['coreweave~parasail']['tv'] for r in v2rows])),
        median_tv_logprob_path_cw_ps=float(np.median([r['logprob_path_tv'] for r in v2rows if 'logprob_path_tv' in r])),
        median_ci_halfwidth=float(np.median([(r['pairs']['coreweave~parasail']['hi'] - r['pairs']['coreweave~parasail']['lo']) / 2
                                             for r in v2rows])),
        median_self_tv=dict(coreweave=float(np.median([r['self']['coreweave'] for r in v2rows if r['self']])),
                            parasail=float(np.median([r['self']['parasail'] for r in v2rows if r['self']]))))
    print(json.dumps(summary, indent=1))
    (ROOT / 'reports/adaptive-tv-hosted-2026-11-17.json').write_text(json.dumps(dict(boot=BOOT, summary=summary,
                                                                                    rows=rows), indent=1))


if __name__ == '__main__':
    main()
