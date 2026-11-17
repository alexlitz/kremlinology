"""Cache latency, two ways (witnessed timing via kremlin http_client):

grid  time to first token for a follow-up that adds U new tokens on top of C cached tokens, over a grid of (C, U).
      Request 1 = [user: P (C tokens) + "Reply with OK."]; request 2 = request 1 + [assistant: "OK.",
      user: N (U tokens) + question], 3 s later. A follow-up works for any-prefix caches and for caches that resume
      only where a previous request ended. C = 0 sends request 2's shape with an empty P (nothing to reuse).
age   C and U fixed; request 2 sent after a delay (5 s to 20 min) since request 1 wrote the cache. Tiered caches
      (HBM -> DRAM -> SSD) should show hits that get slower before they disappear.
Spend guard shared by both (--cap). Timing is analysed from the session evidence (witness timestamps).
"""
import argparse
import heapq
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kremlin

API = 'https://openrouter.ai/api/v1/chat/completions'
WORDS = ('river stone market winter lantern harbor copper meadow signal orchard ledger canyon violet engine '
         'paper thunder village silver mirror garden compass anchor feather glacier timber station velvet '
         'shadow beacon quarry saddle marble cotton falcon tunnel ember prairie walnut archive bridge').split()
GRID = [
    dict(model='deepseek/deepseek-v4.1-flash', provider='parasail/fp8', cached=[0, 4000, 16000, 32000],
         new=[500, 4000, 16000], trials=2),
    dict(model='openai/gpt-oss-120b', provider='nebius/fp4', cached=[0, 4000, 16000, 32000],
         new=[500, 4000, 16000], trials=2),
    dict(model='moonshotai/kimi-k2.6', provider='moonshotai', cached=[0, 8000, 32000], new=[500, 8000], trials=2),
]
AGE = [
    dict(model='moonshotai/kimi-k2.6', provider='moonshotai', cached=8000, new=500),
    dict(model='openai/gpt-oss-120b', provider='nebius/fp4', cached=16000, new=500),
    dict(model='deepseek/deepseek-v4.1-flash', provider='deepinfra/fp8', cached=16000, new=500),
]
DELAYS = [5, 30, 60, 120, 300, 600, 1200]
AGE_TRIALS = 2
AGE_SPREAD = 240  # seconds over which the age writes are spread (age-v2 used 240: Moonshot queued our own burst)


def text(seed, tokens, per_word=1.15):
    rng = random.Random(seed)
    out = []
    while len(out) < int(tokens / per_word):
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]
        out[-1] += '.'
    return ' '.join(out)


class Runner:
    def __init__(self, out, key, cap):
        self.http = kremlin.http_client(out / 'session', timeout=600)
        self.key, self.cap, self.spent, self.lock, self.log = key, cap, 0.0, threading.Lock(), []

    def call(self, model, provider, messages, tag):
        with self.lock:
            if self.spent >= self.cap:
                raise RuntimeError('spend guard')
        body = dict(model=model, max_tokens=16, temperature=0, stream=True, usage=dict(include=True),
                    reasoning=dict(enabled=False), provider=dict(only=[provider], allow_fallbacks=False),
                    messages=messages)
        if model == 'openai/gpt-oss-120b':
            body['reasoning'] = dict(effort='low')
        usage, gen, status = None, None, None
        with self.http.stream('POST', API, json=body, headers={'Authorization': f'Bearer {self.key}'}) as reply:
            status = reply.status_code
            for line in reply.iter_lines():
                if line.startswith('data:') and line.strip() != 'data: [DONE]':
                    try:
                        obj = json.loads(line[5:])
                    except ValueError:
                        continue
                    gen = obj.get('id') or gen
                    usage = obj.get('usage') or usage
        cost = (usage or {}).get('cost') or 0
        row = dict(tag=tag, model=model, provider=provider, status=status, generation=gen,
                   prompt_tokens=(usage or {}).get('prompt_tokens'),
                   cached_tokens=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
                   cost=cost, t=time.time())
        with self.lock:
            self.spent += cost
            self.log.append(row)
        print(json.dumps(row), flush=True)
        return row


def pair(run_id, model, provider, cached, new, label):
    prefix = f'[{run_id} {label}] ' + (text(f'{run_id}-{label}-p', cached) if cached else '')
    first = [dict(role='user', content=prefix + '\n\nReply with OK.')]
    second = first + [dict(role='assistant', content='OK.'),
                      dict(role='user', content=f'[{label} new] ' + text(f'{run_id}-{label}-n', new)
                           + '\n\nIn one sentence, what is this new text about?')]
    return first, second


def grid_job(runner, run_id, spec):
    for trial in range(spec['trials']):
        for c in spec['cached']:
            for u in spec['new']:
                label = f"grid:{c}:{u}:{trial}"
                first, second = pair(run_id, spec['model'], spec['provider'], c, u, label)
                try:
                    if c:
                        runner.call(spec['model'], spec['provider'], first, label + ':write')
                        time.sleep(3)
                    runner.call(spec['model'], spec['provider'], second, label + ':read')
                except RuntimeError:
                    return
                except Exception as error:
                    print(json.dumps(dict(label=label, error=str(error)[:200])), flush=True)


def age_schedule(runner, run_id):
    events, ready = [], {}
    rng = random.Random(run_id)
    for spec in AGE:
        for trial in range(AGE_TRIALS):
            for delay in DELAYS:
                label = f"age:{delay}:{trial}"
                start = rng.uniform(0, AGE_SPREAD)
                first, second = pair(run_id, spec['model'], spec['provider'], spec['cached'], spec['new'],
                                     f"{spec['provider']}-{label}")
                key = (spec['provider'], label)
                ready[key] = threading.Event()
                heapq.heappush(events, (start, 'write', key, spec, first, label))
                heapq.heappush(events, (start + delay, 'read', key, spec, second, label))
    t0 = time.time()
    pool = ThreadPoolExecutor(12)

    def run(kind, key, spec, messages, label):
        try:
            if kind == 'read':
                ready[key].wait(600)
            runner.call(spec['model'], spec['provider'], messages, f'{label}:{kind}')
        except Exception as error:
            print(json.dumps(dict(label=label, kind=kind, error=str(error)[:200])), flush=True)
        finally:
            if kind == 'write':
                ready[key].set()
    futures = []
    while events:
        when, kind, key, spec, messages, label = heapq.heappop(events)
        wait = t0 + when - time.time()
        if wait > 0:
            time.sleep(wait)
        futures.append(pool.submit(run, kind, key, spec, messages, label))
    for f in futures:
        f.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=0.8)
    parser.add_argument('--grid-trials', type=int, help='trials per grid point (default per endpoint in GRID)')
    parser.add_argument('--age-trials', type=int, help='trials per delay (default AGE_TRIALS)')
    parser.add_argument('--only', choices=['grid', 'age'])
    parser.add_argument('--providers', help='comma-separated providers to keep in GRID/AGE')
    parser.add_argument('--age-spread', type=float, help='seconds to spread age writes over (default AGE_SPREAD)')
    parser.add_argument('--delays', help='comma-separated age delays in seconds (default DELAYS)')
    args = parser.parse_args()
    global AGE_TRIALS, AGE_SPREAD
    if args.age_spread:
        AGE_SPREAD = args.age_spread
    if args.delays:
        DELAYS[:] = [int(d) for d in args.delays.split(',')]
    if args.providers:
        keep = set(args.providers.split(','))
        GRID[:] = [g for g in GRID if g['provider'] in keep]
        AGE[:] = [a for a in AGE if a['provider'] in keep]
    if args.grid_trials:
        for spec in GRID:
            spec['trials'] = args.grid_trials
    if args.age_trials:
        AGE_TRIALS = args.age_trials
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    run_id = f'{random.Random().getrandbits(40):010x}'
    runner = Runner(out, os.environ['OPENROUTER_API_KEY'], args.cap)
    try:
        age = threading.Thread(target=age_schedule, args=(runner, run_id))
        if args.only != 'grid':
            age.start()
        if args.only != 'age':
            with ThreadPoolExecutor(len(GRID)) as pool:
                list(pool.map(lambda s: grid_job(runner, run_id, s), GRID))
        if args.only != 'grid':
            age.join()
    finally:
        runner.http.close()
        (out / 'log.json').write_text(json.dumps(dict(run_id=run_id, spent=runner.spent, grid=GRID, age=AGE, age_spread=AGE_SPREAD,
                                                      delays=DELAYS, rows=runner.log), indent=1))
        print(f'spent ${runner.spent:.4f}; {len(runner.log)} calls')


if __name__ == '__main__':
    main()
