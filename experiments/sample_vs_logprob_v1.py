"""Do hosted samples follow the provider's own reported logprobs? Same-response check (2026-11-17).

The adaptive TV run found, on near-tie prompts, sampled T=1 frequencies 7-12 points away from the provider's own
temperature-0 logprobs (both CoreWeave fp16 and Parasail fp8). This sends N one-token requests at temperature 1 with
logprobs=True, top_logprobs=5, so each response carries both the sampled token and the reported distribution it was
(supposedly) drawn from. Compares the sampled frequency with the mean reported probability, per token, with a 95%
Wilson interval on the frequency. Also N/5 temperature-0 logprob requests as the earlier baseline.
--ab: interleaved A/B at temperature 1, N requests with logprobs and N without, alternating, per provider: does
asking for logprobs change the distribution sampled from?
Run: python experiments/sample_vs_logprob_v1.py OUT [--n 600] [--probes wiki-0333,code-0094] [--ab]
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from ci_utils import wilson  # noqa: E402

API = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'meta-llama/llama-3.3-70b-instruct'
PROVIDERS = ['coreweave/fp16', 'parasail/fp8']


def call(client, key, provider, messages, temperature, logprobs=True):
    body = dict(model=MODEL, messages=messages, max_tokens=1, temperature=temperature,
                usage=dict(include=True), provider=dict(only=[provider], allow_fallbacks=False))
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    for attempt in range(4):
        try:
            r = client.post(API, json=body, headers={'Authorization': f'Bearer {key}'}, timeout=60)
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            data = r.json()
            choice = data['choices'][0]
            content = ((choice.get('logprobs') or {}).get('content') or [])
            return dict(token=choice['message'].get('content') or '', id=data.get('id'),
                        cost=(data.get('usage') or {}).get('cost') or 0,
                        top={e['token']: math.exp(e['logprob']) for e in content[0]['top_logprobs']} if content else None,
                        sampled_logprob=content[0]['logprob'] if content else None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--n', type=int, default=600)
    parser.add_argument('--probes', default='wiki-0333,code-0094')
    parser.add_argument('--ab', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    prompts = {p['probe']: p for p in json.loads((ROOT / 'experiments/adaptive_tv_prompts_v1.json').read_text())}
    summary, spent = [], 0.0
    with httpx.Client(limits=httpx.Limits(max_connections=32)) as client, ThreadPoolExecutor(16) as pool:
        for probe in args.probes.split(','):
            messages = prompts[probe]['messages']
            for provider in PROVIDERS:
                if args.ab:
                    flags = [i % 2 == 0 for i in range(2 * args.n)]
                    res = list(pool.map(lambda f: (f, call(client, key, provider, messages, 1.0, f)), flags))
                    spent += sum(r['cost'] for _, r in res if r)
                    with open(out / 'responses.jsonl', 'a') as f:
                        for flag, r in res:
                            if r:
                                f.write(json.dumps(dict(probe=probe, provider=provider, kind='ab', logprobs=flag, **r)) + '\n')
                    row = dict(probe=probe, provider=provider, ab={})
                    for flag in (True, False):
                        got = [r['token'] for fl, r in res if r and fl == flag]
                        freq = Counter(got)
                        row['ab']['with_logprobs' if flag else 'without'] = {
                            t: dict(k=k, n=len(got), share=k / len(got), ci=wilson(k, len(got))) for t, k in freq.most_common(3)}
                    summary.append(row)
                    print(probe, provider, {m: {t: f"{v['share']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] n={v['n']}"
                                                for t, v in d.items()} for m, d in row['ab'].items()}, f'spent ${spent:.3f}', flush=True)
                    continue
                hot = [r for r in pool.map(lambda _: call(client, key, provider, messages, 1.0), range(args.n)) if r]
                cold = [r for r in pool.map(lambda _: call(client, key, provider, messages, 0.0), range(args.n // 5)) if r]
                spent += sum(r['cost'] for r in hot + cold)
                with open(out / 'responses.jsonl', 'a') as f:
                    for kind, rows in (('t1', hot), ('t0', cold)):
                        for r in rows:
                            f.write(json.dumps(dict(probe=probe, provider=provider, kind=kind, **r)) + '\n')
                freq = Counter(r['token'] for r in hot)
                tokens = [t for t, _ in freq.most_common(3)]
                row = dict(probe=probe, provider=provider, n=len(hot), n_t0=len(cold), tokens={})
                for t in tokens:
                    k = freq[t]
                    lo, hi = wilson(k, len(hot))
                    rep1 = [r['top'].get(t, 0) for r in hot if r['top']]
                    rep0 = [r['top'].get(t, 0) for r in cold if r['top']]
                    row['tokens'][t] = dict(sampled=k / len(hot), lo=lo, hi=hi,
                                            reported_t1=sum(rep1) / len(rep1) if rep1 else None,
                                            reported_t0=sum(rep0) / len(rep0) if rep0 else None)
                summary.append(row)
                print(probe, provider, len(hot), {t: f"sampled {v['sampled']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] "
                                                  f"reported T1 {v['reported_t1'] or 0:.3f} T0 {v['reported_t0'] or 0:.3f}"
                                                  for t, v in row['tokens'].items()}, f'spent ${spent:.3f}', flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(model=MODEL, n=args.n, summary=summary, spent=spent), indent=1))


if __name__ == '__main__':
    main()
