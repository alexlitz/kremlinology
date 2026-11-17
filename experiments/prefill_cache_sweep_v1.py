"""Prefill speed, cache effect and the disaggregation signature, by prompt size and provider (witnessed).

Per pinned endpoint and prompt size (1k-32k tokens): a unique prompt sent cold, then the identical prompt 3 s later
(warm), 3 trials. Timing comes from the witness relay's record timestamps (kremlin session evidence), not the client.
Measures, in the analysis:
  - cold TTFT vs prompt tokens: slope = seconds per prefilled token (prefill speed), intercept = fixed overhead;
  - warm TTFT vs cold at each size: what a cache hit saves as the prompt grows;
  - gap between the first and second token-bearing events vs prompt size. With prefill/decode disaggregation the
    first token comes from the prefill node and the KV cache must reach a decode node before the second, so that gap
    should grow with the prompt; colocated serving keeps it near one decode step.
Providers run in parallel (one thread each, requests sequential within a provider). Spend guard: --cap.
"""
import argparse
import json
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
PLAN = {
    'deepseek/deepseek-v4.1-flash': dict(
        providers=['fireworks', 'baseten', 'together', 'deepinfra/fp8', 'coreweave/fp8', 'parasail/fp8',
                   'novita/fp8', 'gmicloud/fp8'],
        sizes=[1000, 2000, 4000, 8000, 16000, 32000], trials=3),
    'moonshotai/kimi-k2.6': dict(providers=['moonshotai', 'parasail/int4', 'deepinfra/fp4'],
                                 sizes=[1000, 4000, 16000, 32000], trials=2),
}
TOKENS_PER_WORD = 1.15


def text(seed, tokens):
    rng = random.Random(seed)
    words, out = int(tokens / TOKENS_PER_WORD), []
    while len(out) < words:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]
        out[-1] += '.'
    return ' '.join(out)


class Runner:
    def __init__(self, out, key, cap):
        self.http = kremlin.http_client(out / 'session', timeout=600)
        self.key, self.cap = key, cap
        self.spent, self.lock, self.log = 0.0, threading.Lock(), []

    def call(self, model, provider, prompt, tag):
        with self.lock:
            if self.spent >= self.cap:
                raise RuntimeError('spend guard')
        body = dict(model=model, max_tokens=24, temperature=0, stream=True, usage=dict(include=True),
                    reasoning=dict(enabled=False), provider=dict(only=[provider], allow_fallbacks=False),
                    messages=[dict(role='user', content=prompt + '\n\nIn one sentence, what is this text about?')])
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


def provider_job(runner, run_id, model, provider, sizes, trials):
    for trial in range(trials):
        for size in sizes:
            prompt = f'[{run_id} {provider} {size} {trial}] ' + text(f'{run_id}-{model}-{provider}-{size}-{trial}', size)
            try:
                runner.call(model, provider, prompt, f'{size}:{trial}:cold')
                time.sleep(3)
                runner.call(model, provider, prompt, f'{size}:{trial}:warm')
            except RuntimeError:
                return
            except Exception as error:
                print(json.dumps(dict(provider=provider, size=size, error=str(error)[:200])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=1.2)
    parser.add_argument('--trials', action='append', default=[], metavar='MODEL=N', help='override trials per model')
    parser.add_argument('--providers', help='comma-separated providers to keep')
    args = parser.parse_args()
    if args.providers:
        keep = set(args.providers.split(','))
        for plan in PLAN.values():
            plan['providers'] = [x for x in plan['providers'] if x in keep]
    for item in args.trials:
        model, _, n = item.rpartition('=')
        PLAN[model]['trials'] = int(n)
    key = __import__('os').environ['OPENROUTER_API_KEY']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    run_id = f'{random.Random().getrandbits(40):010x}'
    runner = Runner(out, key, args.cap)
    jobs = [(model, p, plan['sizes'], plan['trials']) for model, plan in PLAN.items() for p in plan['providers']]
    try:
        with ThreadPoolExecutor(len(jobs)) as pool:
            list(pool.map(lambda j: provider_job(runner, run_id, *j), jobs))
    finally:
        runner.http.close()
        (out / 'log.json').write_text(json.dumps(dict(run_id=run_id, spent=runner.spent, plan=PLAN, rows=runner.log),
                                                  indent=1))
        print(f'spent ${runner.spent:.4f}; {len(runner.log)} calls')


if __name__ == '__main__':
    main()
