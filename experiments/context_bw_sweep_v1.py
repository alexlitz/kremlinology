"""Decode speed vs context length: joint bandwidth and KV-bytes fit (2026-11-22; paid, via proxy).

Per endpoint and context size C (random filler prefix + an unpredictable-output instruction), decode tok/s =
(completion_tokens - 1) / (last chunk - first chunk), R repeats. The model 1/tps = (W + k*C)/BW is linear in C:
the intercept gives BW/W (a per-stream bandwidth bound with the weight term alone), the slope k/BW, so with declared
weight bytes W the fit returns both the effective bandwidth and k, the KV bytes read per token per step -- an
independent measurement of KV precision to compare with the architecture's nominal KV size.
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/context_bw_sweep_v1.py OUT
"""
import argparse, json, os, random, string, time
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
ENDPOINTS = [  # (model, provider, label, weight_bytes, nominal_kv_bytes_per_token)
    ('meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16', 'llama70b@coreweave-fp16', 141e12, 320 * 1024),
    ('meta-llama/llama-3.3-70b-instruct', 'akashml/fp8', 'llama70b@akashml-fp8', 70.6e12, 320 * 1024),
    ('openai/gpt-oss-120b', 'nebius/fp4', 'gptoss120b@nebius-fp4', 60.4e12, 36864 + 36864),
    ('google/gemma-4-31b-it', 'parasail/fp8', 'gemma4-31b@parasail-fp8', 31.3e12, 81920 + 204800 // 5),
]
SIZES = [1000, 4000, 16000, 64000]
REPS = 8
WORDS = ('system cache token latency memory block prefix model request provider window layer state attention '
         'vector matrix budget schedule replica storage compute network kernel batch').split()


def filler(rng, tokens):
    w = int(tokens / 1.15); out = []
    while len(out) < w:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]; out[-1] += '.'
    return ' '.join(out)


def run(cl, key, model, provider, prompt):
    body = dict(model=model, max_tokens=400, temperature=1, stream=True, usage=dict(include=True),
                provider=dict(only=[provider], allow_fallbacks=False),
                messages=[dict(role='user', content=prompt)])
    times, usage = [], None
    with cl.stream('POST', URL, json=body, headers={'Authorization': 'Bearer ' + key}, timeout=600) as r:
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
    return (usage.get('completion_tokens') or 0) - 1, times[-1] - times[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--cap', type=float, default=0.8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    cl = httpx.Client(timeout=600)
    rng = random.Random(20261122)
    rows, spent = [], 0.0
    for model, provider, label, W, kv in ENDPOINTS:
        for size in SIZES:
            speeds = []
            for rep in range(REPS):
                seq = ''.join(rng.choice(string.ascii_letters + string.digits) for _ in range(300))
                prompt = (f'Reference {rng.getrandbits(64):016x}. Context:\n{filler(rng, size)}\n\n'
                          f'Ignore the context. Repeat the following sequence exactly, 25 times, separated by spaces:\n{seq}')
                got = run(cl, key, model, provider, prompt)
                if got and got[0] > 20:
                    speeds.append(got[0] / got[1])
                time.sleep(1)
            if speeds:
                row = dict(endpoint=label, context=size, n=len(speeds), median_tps=sorted(speeds)[len(speeds) // 2])
                rows.append(row); print(json.dumps(row), flush=True)
                (out / 'rows.json').write_text(json.dumps(rows, indent=1))
    cl.close()
    print(json.dumps(dict(done=len(rows))))


if __name__ == '__main__':
    main()
