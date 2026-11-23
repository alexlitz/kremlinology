"""Double-tap probe for cache tiers (2026-11-23; paid, via the witness proxy; NOT YET RUN).

Per endpoint, prompt size S and cache age A, TRIALS unique prompts each get three requests:
  W   the write (cold);
  R1  the identical request A seconds later (the cache at age A);
  R2  the identical request again, 2 s after R1 returns (the cache just touched).
Readings, all within one prompt so load and placement mostly cancel:
  R1 - R2 > 0, growing with S   the entry had been demoted: R1 paid a load that R2 did not (promotion on hit);
                                S / (R1 - R2) is the load rate of the tier that age A maps to;
  R1 ~ R2 at every age          one tier (or no promotion); compare both with W for the hit's absolute cost;
  R2 reported, R1 not           a hit on a cold tier re-enters the reported (discounted) tier.
Tier boundaries are the ages where R1 - R2 steps up. Requests are one-token, non-streamed, reasoning off; elapsed
time is the client's, through the proxy. Writes are spread over SPREAD seconds and at most 2 requests per endpoint
are in flight, so our own load does not queue behind itself.
Run: KREMLINOLOGY_OPENROUTER_URL=http://127.0.0.1:PORT/api/v1/chat/completions \
     python experiments/cache_tier_doubletap_v1.py OUT [--cap 3.5] [--trials 20] [--only cheap|all] [--providers nebius/fp4,...]
"""
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
WORDS = ('system cache token latency memory block prefix model request provider window layer state attention '
         'vector matrix budget schedule replica storage compute network kernel batch').split()
# (model, provider, sizes, ages_s, group, extra body)
ENDPOINTS = [
    ('deepseek/deepseek-v4.1-flash', 'deepseek', [16000, 64000], [5, 300, 1800, 7200], 'cheap', {}),   # documented disk cache
    ('openai/gpt-oss-120b', 'nebius/fp4', [4000, 16000], [5, 120, 600, 1500], 'cheap', {'keep_reasoning': True}),  # reasoning is mandatory there; age step at 1-5 min?
    ('moonshotai/kimi-k2.6', 'moonshotai', [8000], [5, 240, 720], 'cheap', {}),                       # unreported tier past 10 min
    ('google/gemma-4-31b-it', 'chutes/fp4', [8000], [5, 60, 240], 'cheap', {}),                       # hits at miss speed by 60 s
    ('deepseek/deepseek-v4.1-flash', 'deepinfra/fp8', [32000], [5, 300, 1200], 'cheap', {}),          # resident for 20 min?
    ('openai/gpt-5.4-nano', 'openai', [32000], [5, 240], 'cheap', {}),
    ('meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16', [16000], [5, 600, 1500], 'all', {}),      # single-tier control
    ('anthropic/claude-haiku-4.5', 'anthropic', [16000], [5, 240, 1800], 'all', {'ttl': '1h'}),      # is the 1 h cache a slower tier?
]
SPREAD = 900


def text(rng, tokens):
    w = int(tokens / 1.15); out = []
    while len(out) < w:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]; out[-1] += '.'
    return f'Reference {rng.getrandbits(64):016x}. ' + ' '.join(out)


def messages(body_text, extra):
    if extra.get('ttl'):
        return [dict(role='user', content=[
            dict(type='text', text=body_text, cache_control=dict(type='ephemeral', ttl=extra['ttl'])),
            dict(type='text', text='Reply with OK.')])]
    return [dict(role='user', content=body_text + '\n\nReply with OK.')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--cap', type=float, default=3.5)
    ap.add_argument('--trials', type=int, default=20); ap.add_argument('--only', default='cheap')
    ap.add_argument('--providers', default='', help='comma-separated provider tags to keep')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    cl = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=64))
    lock, state = threading.Lock(), dict(spent=0.0)
    sems = {}
    rng = random.Random(20261123)
    jobs = []
    for model, provider, sizes, ages, group, extra in ENDPOINTS:
        if a.only == 'cheap' and group != 'cheap':
            continue
        if a.providers and provider not in a.providers.split(','):
            continue
        sems[provider + model] = threading.Semaphore(2)
        for size in sizes:
            for age in ages:
                for trial in range(a.trials):
                    jobs.append(dict(model=model, provider=provider, size=size, age=age, trial=trial, extra=extra,
                                     start=rng.uniform(0, SPREAD), text=text(rng, size)))
    t0 = time.time() + 5

    def call(job, role):
        with lock:
            if state['spent'] >= a.cap:
                return None
        body = dict(model=job['model'], max_tokens=1, temperature=0, messages=messages(job['text'], job['extra']),
                    usage=dict(include=True),
                    provider=dict(only=[job['provider']], allow_fallbacks=False))
        if not job['extra'].get('keep_reasoning'):
            body['reasoning'] = dict(enabled=False)
        with sems[job['provider'] + job['model']]:
            t = time.time()
            try:
                r = cl.post(URL, json=body, headers={'Authorization': 'Bearer ' + key})
                j = r.json() if r.status_code == 200 else {}
                status = r.status_code
            except Exception as error:
                j, status = {}, type(error).__name__
            el = time.time() - t
        u = j.get('usage') or {}
        row = dict(model=job['model'], provider=job['provider'], size=job['size'], age=job['age'], trial=job['trial'],
                   role=role, status=status, sent=t, elapsed_s=el, prompt=u.get('prompt_tokens'),
                   cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'), cost=u.get('cost'),
                   generation=j.get('id'))
        with lock:
            state['spent'] += u.get('cost') or 0
            with (out / 'rows.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        return row

    def run(job):
        wait = t0 + job['start'] - time.time()
        if wait > 0:
            time.sleep(wait)
        w = call(job, 'W')
        if not w or w['status'] != 200:
            return
        time.sleep(max(0, w['sent'] + w['elapsed_s'] + job['age'] - time.time()))
        r1 = call(job, 'R1')
        if not r1 or r1['status'] != 200:
            return
        time.sleep(2)
        call(job, 'R2')

    with ThreadPoolExecutor(len(jobs)) as pool:
        list(pool.map(run, jobs))
    cl.close()
    (out / 'summary.json').write_text(json.dumps(dict(jobs=len(jobs), spent=state['spent'], trials=a.trials,
                                                      only=a.only, finished=time.time()), indent=1))
    print(json.dumps(dict(stop='done', jobs=len(jobs), spent=state['spent'])))


if __name__ == '__main__':
    main()
