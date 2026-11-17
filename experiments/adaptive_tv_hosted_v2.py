"""Adaptive sampling-only TV between hosted providers (no logprobs needed), with a logprob-based control.

Llama 3.3-70B on OpenRouter, pinned endpoints: CoreWeave fp16 (reference), Parasail fp8 (returns logprobs: control,
compared with its logprob-based TV) and DeepInfra turbo (returns no logprobs: the unknown). For each of 10 high-entropy
dataset prompts (experiments/adaptive_tv_prompts_v1.json), the estimator and stopping/allocation rules of
adaptive_tv_v1 run with one-token requests as samples (OpenRouter ignores `n`, so each sample is one request):
plain T=1 and T=2 + top_k=10, rounds of ROUND samples, until both pairs' 95% CI half-width < TARGET or the per-provider
budget is spent. Requests go direct (not through the witness relay: ~90k one-token calls); raw records are kept.
Spend cap: --cap. Output: the run directory (results.json, samples.jsonl).

v2 (after v1 stalled on prompt 2 when DeepInfra's shared upstream pool returned 429 for over an hour): each sample row
records its probe; a provider's round keeps going until it has its samples or STALL seconds pass without one (then the
provider is marked stalled for that prompt and pairs with it are reported as incomplete, not estimated); per-provider
concurrency; --skip resumes after probes finished in an earlier run. (Google Vertex, the other top_k-without-logprobs
endpoint, is bring-your-own-key only; Groq takes neither logprobs nor top_k.)
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import adaptive_tv_v1 as atv  # noqa: E402

API = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'meta-llama/llama-3.3-70b-instruct'
PROVIDERS = {'coreweave': 'coreweave/fp16', 'parasail': 'parasail/fp8', 'deepinfra': 'deepinfra/turbo'}
PAIRS = [('coreweave', 'parasail'), ('coreweave', 'deepinfra')]
THREADS = {'coreweave/fp16': 16, 'parasail/fp8': 16, 'deepinfra/turbo': 6, 'google-vertex/us-central1': 8}
STALL = 900          # seconds without a successful sample before a provider is given up for the prompt
MIN_SAMPLES = 400    # per provider, to estimate a pair
STALLED_BEFORE = set()  # providers that stalled on an earlier prompt get a short 120 s grace on later prompts


class Budget:
    def __init__(self, cap):
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()

    def add(self, cost):
        with self.lock:
            self.spent += cost or 0
            return self.spent < self.cap


class HostedSampler:
    def __init__(self, provider, messages, key, budget, log, client, probe):
        self.provider, self.messages, self.key, self.budget, self.log, self.client = provider, messages, key, budget, log, client
        self.probe, self.stalled = probe, False

    def one(self, setting):
        temperature, top_k, top_p = setting
        body = dict(model=MODEL, messages=self.messages, max_tokens=1, temperature=temperature, usage=dict(include=True),
                    provider=dict(only=[self.provider], allow_fallbacks=False))
        if top_k:
            body['top_k'] = top_k
        for attempt in range(4):
            try:
                r = self.client.post(API, json=body, headers={'Authorization': f'Bearer {self.key}'}, timeout=60)
            except httpx.HTTPError:
                time.sleep(1 + attempt)
                continue
            if r.status_code in (429, 502, 503):
                time.sleep(5 + 5 * attempt)
                continue
            if r.status_code != 200:
                self.log.write(json.dumps(dict(probe=self.probe, provider=self.provider, setting=setting, status=r.status_code)) + '\n')
                return None
            data = r.json()
            text = (data['choices'][0]['message'].get('content') or '')
            cost = (data.get('usage') or {}).get('cost') or 0
            self.log.write(json.dumps(dict(probe=self.probe, provider=self.provider, setting=setting, token=text, cost=cost,
                                           served=data.get('provider'), id=data.get('id'))) + '\n')
            if not self.budget.add(cost):
                raise RuntimeError('spend cap')
            return text
        return None

    def sample(self, setting, count):
        got, last = Counter(), time.time()
        with ThreadPoolExecutor(THREADS[self.provider]) as pool:
            while not self.stalled and sum(got.values()) < count:
                need = min(count - sum(got.values()), 4 * THREADS[self.provider])  # small batches: notice stalls
                out = [t for t in pool.map(lambda _: self.one(setting), range(need)) if t is not None]
                got.update(out)
                if out:
                    last = time.time()
                    STALLED_BEFORE.discard(self.provider)
                elif time.time() - last > (120 if self.provider in STALLED_BEFORE else STALL):
                    self.stalled = True
                    STALLED_BEFORE.add(self.provider)
                else:
                    time.sleep(30)
        return got


class HostedProvider(atv.Provider):
    def round(self, n):  # count only samples actually received
        plain = self.sampler.sample(atv.PLAIN, n // 2)
        hot = self.sampler.sample(atv.HOT, n - n // 2)
        self.plain += plain
        self.hot += hot
        self.used += sum(plain.values()) + sum(hot.values())


def logprob_law(client, key, provider, messages, repeats=5):
    """Mean first-token distribution over the reported top-20 (plus OTHER), from `repeats` logprob requests."""
    import math
    laws = []
    for _ in range(repeats):
        body = dict(model=MODEL, messages=messages, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
                    provider=dict(only=[provider], allow_fallbacks=False))
        r = client.post(API, json=body, headers={'Authorization': f'Bearer {key}'}, timeout=60)
        if r.status_code != 200:
            continue
        content = ((r.json()['choices'][0].get('logprobs') or {}).get('content') or [])
        if content:
            laws.append({e['token']: math.exp(e['logprob']) for e in content[0].get('top_logprobs', [])})
    if not laws:
        return None
    keys = set().union(*laws)
    law = {k: sum(l.get(k, 0) for l in laws) / len(laws) for k in keys}
    law['OTHER'] = max(0.0, 1 - sum(law.values()))
    return law


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=5.0)
    parser.add_argument('--budget', type=int, default=4000, help='samples per provider per prompt')
    parser.add_argument('--round', type=int, default=500)
    parser.add_argument('--skip', default='', help='comma-separated probes already done')
    args = parser.parse_args()
    atv.ROUND, atv.BUDGET = args.round, args.budget
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    prompts = json.loads((ROOT / 'experiments/adaptive_tv_prompts_v1.json').read_text())
    budget = Budget(args.cap)
    rng = np.random.default_rng(17)
    results = []
    with httpx.Client(http2=False, limits=httpx.Limits(max_connections=64)) as client, open(out / 'samples.jsonl', 'a') as log:
        for item in prompts:
            if item['probe'] in args.skip.split(','):
                continue
            providers = {name: HostedProvider(HostedSampler(slug, item['messages'], key, budget, log, client, item['probe']))
                         for name, slug in PROVIDERS.items()}
            lp = {n: logprob_law(client, key, PROVIDERS[n], item['messages']) for n in ('coreweave', 'parasail')}
            logprob_tv = atv.tv(lp['coreweave'], lp['parasail']) if all(lp.values()) else None
            status = {}
            try:
                for p in providers.values():
                    p.round(atv.ROUND)
                while True:
                    status = {}
                    for a, b in PAIRS:
                        if min(providers[a].used, providers[b].used) < MIN_SAMPLES:
                            status[(a, b)] = dict(tv=None, incomplete=True)
                            continue
                        draws = atv.bootstrap(providers[a], providers[b], rng)
                        raw = atv.tv(atv.estimate(providers[a].plain, providers[a].hot),
                                     atv.estimate(providers[b].plain, providers[b].hot))
                        point, lo, hi = atv.interval(raw, draws)
                        status[(a, b)] = dict(tv=point, raw=raw, lo=lo, hi=hi)
                    open_pairs = [k for k, v in status.items() if v['tv'] is not None and (v['hi'] - v['lo']) / 2 >= atv.TARGET]
                    open_pairs += [k for k, v in status.items() if v['tv'] is None
                                   and not any(providers[n].sampler.stalled for n in k)]
                    candidates = [n for n, p in providers.items() if p.used < atv.BUDGET and not p.sampler.stalled]
                    if not open_pairs or not candidates:
                        break
                    var = Counter()
                    for a, b in [k for k in open_pairs if status[k]['tv'] is None]:  # incomplete pairs: feed the short side
                        var[min((a, b), key=lambda n: providers[n].used)] += 1.0
                    for a, b in [k for k in open_pairs if status[k]['tv'] is not None]:  # each provider's own contribution to the open pairs' TV variance
                        var[a] += atv.bootstrap(providers[a], providers[b], rng, 'a').var()
                        var[b] += atv.bootstrap(providers[b], providers[a], rng, 'a').var()
                    target = max(candidates, key=lambda n: var[n])
                    providers[target].round(atv.ROUND)
            except RuntimeError:
                print('spend cap reached', flush=True)
            results.append(dict(probe=item['probe'], logprob_tv_coreweave_parasail=logprob_tv, logprob_laws=lp,
                                samples={n: p.used for n, p in providers.items()},
                                stalled=[n for n, p in providers.items() if p.sampler.stalled],
                                estimates={n: atv.estimate(p.plain, p.hot) for n, p in providers.items()},
                                pairs={f'{a}~{b}': status[(a, b)] for a, b in PAIRS}))
            print(item['probe'], 'logprob TV cw~ps', None if logprob_tv is None else round(logprob_tv, 3), {k: 'incomplete' if v['tv'] is None else f"{v['tv']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" for k, v in results[-1]['pairs'].items()},
                  results[-1]['samples'], f'spent ${budget.spent:.3f}', flush=True)
            (out / 'results.json').write_text(json.dumps(dict(model=MODEL, providers=PROVIDERS, results=results,
                                                              spent=budget.spent), indent=1, default=str))
            if budget.spent >= budget.cap:
                break


if __name__ == '__main__':
    main()
