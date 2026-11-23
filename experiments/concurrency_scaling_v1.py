"""Per-stream decode speed vs concurrency: is the bandwidth bound batch-1 valid? (2026-11-22; paid, via proxy).

N parallel streams of the same unpredictable output (a random character sequence to be repeated), per endpoint:
per-stream decode tok/s = (completion_tokens - 1) / (last chunk - first chunk), client-side chunk arrivals through
the witness proxy. If per-stream speed holds as N grows, one weight read is being amortised across streams (batched
decode, or more devices); if it decays ~1/N, serving is memory-bound near batch 1 and the speed x bytes bound is
two-sided. Baseline solo speed measured before and after each N as a load control.
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/concurrency_scaling_v1.py OUT [--reps 2]
"""
import argparse, json, os, random, string, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
ENDPOINTS = [  # (model, provider, label)
    ('meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16', 'llama70b@coreweave-fp16'),
    ('meta-llama/llama-3.3-70b-instruct', 'groq', 'llama70b@groq'),
    ('meta-llama/llama-3.3-70b-instruct', 'sambanova-turbo', 'llama70b@sambanova'),
    ('openai/gpt-oss-120b', 'cerebras/fp16', 'gptoss120b@cerebras'),
    ('openai/gpt-oss-120b', 'nebius/fp4', 'gptoss120b@nebius-fp4'),
    ('google/gemma-4-31b-it', 'deepinfra/turbo', 'gemma4-31b@deepinfra'),
]
NS = [1, 2, 4, 8]


def one_stream(cl, key, model, provider, rng_seed):
    rng = random.Random(rng_seed)
    seq = ''.join(rng.choice(string.ascii_letters + string.digits) for _ in range(300))
    body = dict(model=model, max_tokens=450, temperature=1, stream=True, usage=dict(include=True),
                provider=dict(only=[provider], allow_fallbacks=False),
                messages=[dict(role='user', content=f'Repeat the following sequence exactly, 25 times, separated by spaces:\n{seq}')])
    times, usage = [], None
    with cl.stream('POST', URL, json=body, headers={'Authorization': 'Bearer ' + key}, timeout=300) as r:
        for chunk in r.iter_lines():
            if chunk.startswith('data:') and chunk.strip() != 'data: [DONE]':
                times.append(time.time())
                try:
                    obj = json.loads(chunk[5:])
                    if obj.get('usage'):
                        usage = obj['usage']
                except ValueError:
                    pass
    if not times or not usage or len(times) < 8:
        return None
    toks = usage.get('completion_tokens') or 0
    return (toks - 1) / (times[-1] - times[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--cap', type=float, default=0.6)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    cl = httpx.Client(timeout=300)
    rows, spent = [], 0.0
    for model, provider, label in ENDPOINTS:
      for rep in range(a.reps):
        for n in NS + [1]:
            def task(i):
                return one_stream(cl, key, model, provider, hash((label, n, i)) & 0xffff)
            with ThreadPoolExecutor(n) as ex:
                speeds = [s for s in ex.map(task, range(n)) if s]
            row = dict(endpoint=label, rep=rep, n=n, streams=len(speeds),
                       median_tps=sorted(speeds)[len(speeds) // 2] if speeds else None,
                       min_tps=min(speeds) if speeds else None)
            rows.append(row)
            (out / 'rows.json').write_text(json.dumps(rows, indent=1))
            print(json.dumps(row), flush=True)
            if n == NS[-1]:
                time.sleep(5)
    cl.close()
    print(json.dumps(dict(done=len(rows))))


if __name__ == '__main__':
    main()
