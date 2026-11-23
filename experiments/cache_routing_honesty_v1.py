"""Two cache probes through the witness relay (kremlin http_client; every exchange attested).

A. routing: does OpenRouter send a repeat of a cached prompt back to the provider that holds the cache?
   12 unique ~3k-token prompts for openai/gpt-oss-120b, each sent 4 times without pinning a provider (2 s apart).
   Compare how often repeats land on the first request's provider with how often fresh prompts land on any one
   provider (the default spread), and whether repeats report cached tokens.
B. honesty: does a provider's reported cached-token count match what its time to first token says?
   Per pinned endpoint, 3 trials of (cold request, identical request 3 s later) with a unique long prompt.
   A cache that is used shows as a faster warm request; reported cached tokens should match. Used-but-unreported
   (faster, 0 cached tokens, full price) is the hidden-margin case.
Evidence: kremlin session directories (audit + submit like any run). Spend guard: stops at $0.45 reported cost.
"""
import json
import random
import sys
import time
from pathlib import Path

import kremlin

API = 'https://openrouter.ai/api/v1/chat/completions'
GUARD_USD = float(__import__('os').environ.get('KREMLINOLOGY_GUARD_USD', '0.45'))
WORDS = ('river stone market winter lantern harbor copper meadow signal orchard ledger canyon violet engine '
         'paper thunder village silver mirror garden compass anchor feather glacier timber station velvet '
         'shadow beacon quarry saddle marble cotton falcon tunnel ember prairie walnut archive bridge').split()

B_TARGETS = {
    'openai/gpt-oss-120b': ['coreweave/fp4', 'deepinfra/bf16', 'akashml/fp4', 'nebius/fp4', 'parasail/fp4',
                            'baseten/fp4', 'crusoe/bf16', 'near-ai/fp4', 'sambanova', 'amazon-bedrock',
                            'google-vertex/global'],
    'meta-llama/llama-3.3-70b-instruct': ['coreweave/fp16', 'parasail/fp8', 'akashml/fp8', 'groq',
                                          'deepinfra/turbo', 'sambanova-turbo'],
    'deepseek/deepseek-v3.2': ['gmicloud/fp8', 'deepinfra/fp4', 'baidu/fp8'],
}
B_WORDS = {'openai/gpt-oss-120b': 9000, 'meta-llama/llama-3.3-70b-instruct': 4500, 'deepseek/deepseek-v3.2': 6000}


def text(seed, n_words):
    rng = random.Random(seed)
    sentences = []
    while sum(len(s.split()) for s in sentences) < n_words:
        k = rng.randint(8, 16)
        sentences.append(' '.join(rng.choice(WORDS) for _ in range(k)).capitalize() + '.')
    return ' '.join(sentences)


class Runner:
    def __init__(self, out, key):
        self.http = kremlin.http_client(out, timeout=300)
        self.key = key
        self.spent = 0.0
        self.log = []

    def call(self, model, prompt, provider=None, tag=None):
        if self.spent >= GUARD_USD:
            raise SystemExit(f'spend guard reached (${self.spent:.4f})')
        body = dict(model=model, messages=[dict(role='user', content=prompt + '\n\nReply with the single word OK.')],
                    max_tokens=16, stream=True, usage=dict(include=True))
        if model == 'openai/gpt-oss-120b':
            body['reasoning'] = dict(effort='low')
        if provider:
            body['provider'] = dict(only=[provider], allow_fallbacks=False)
        usage, served, gen_id, status = None, None, None, None
        with self.http.stream('POST', API, json=body, headers={'Authorization': f'Bearer {self.key}'}) as reply:
            status = reply.status_code
            for line in reply.iter_lines():
                if not line.startswith('data:') or line.strip() == 'data: [DONE]':
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                served = obj.get('provider') or served
                gen_id = obj.get('id') or gen_id
                usage = obj.get('usage') or usage
        cost = (usage or {}).get('cost') or 0
        self.spent += cost
        cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        row = dict(tag=tag, model=model, pinned=provider, status=status, served_by=served, generation=gen_id,
                   prompt_tokens=(usage or {}).get('prompt_tokens'), cached_tokens=cached, cost=cost, t=time.time())
        self.log.append(row)
        print(json.dumps(row), flush=True)
        return row


def main():
    key = open(sys.argv[1]).read().strip() if len(sys.argv) > 1 and sys.argv[1] != '-' else \
        __import__('os').environ['OPENROUTER_API_KEY']
    out = Path(sys.argv[2] if len(sys.argv) > 2 else 'cache-probes')
    run_id = f'{random.Random().getrandbits(48):012x}'
    runner = Runner(out / 'session', key)
    if len(sys.argv) > 3 and sys.argv[3] == 'honesty':  # honesty only: ... OUT honesty TRIALS
        trials = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        try:
            for trial in range(trials):
                for model, providers in B_TARGETS.items():
                    for provider in providers:
                        prompt = f'[{run_id}-honest-{trial}-{provider}] ' + text(f'{run_id}-{trial}-{provider}', B_WORDS[model])
                        try:
                            runner.call(model, prompt, provider, tag=f'honest:{provider}:{trial}:cold')
                            time.sleep(3)
                            runner.call(model, prompt, provider, tag=f'honest:{provider}:{trial}:warm')
                        except SystemExit:
                            raise
                        except Exception as error:
                            print(json.dumps(dict(tag=f'honest:{provider}:{trial}', error=str(error)[:200])), flush=True)
        finally:
            runner.http.close()
            (out / 'log.json').write_text(json.dumps(dict(run_id=run_id, spent=runner.spent, rows=runner.log), indent=1))
            print(f'spent ${runner.spent:.4f}; {len(runner.log)} calls')
        return
    if len(sys.argv) > 3:  # routing only, another model: ... OUT MODEL PROMPTS
        model, prompts = sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10
        try:
            for i in range(prompts):
                prompt = f'[{run_id}-route-{i}] ' + text(f'{run_id}-route-{i}', 2300)
                for k in range(4):
                    runner.call(model, prompt, tag=f'route:{i}:{k}')
                    time.sleep(2)
        finally:
            runner.http.close()
            (out / 'log.json').write_text(json.dumps(dict(run_id=run_id, spent=runner.spent, rows=runner.log),
                                                      indent=1))
            print(f'spent ${runner.spent:.4f}; {len(runner.log)} calls')
        return
    try:
        # A: routing (default, unpinned)
        for i in range(12):
            prompt = f'[{run_id}-route-{i}] ' + text(f'{run_id}-route-{i}', 2300)
            for k in range(4):
                runner.call('openai/gpt-oss-120b', prompt, tag=f'route:{i}:{k}')
                time.sleep(2)
        # B: honesty (pinned)
        for trial in range(3):
            for model, providers in B_TARGETS.items():
                for provider in providers:
                    prompt = f'[{run_id}-honest-{trial}-{provider}] ' + text(f'{run_id}-{trial}-{provider}',
                                                                              B_WORDS[model])
                    try:
                        runner.call(model, prompt, provider, tag=f'honest:{provider}:{trial}:cold')
                        time.sleep(3)
                        runner.call(model, prompt, provider, tag=f'honest:{provider}:{trial}:warm')
                    except SystemExit:
                        raise
                    except Exception as error:  # an endpoint being down shouldn't stop the run
                        print(json.dumps(dict(tag=f'honest:{provider}:{trial}', error=str(error)[:200])), flush=True)
    finally:
        runner.http.close()
        (out / 'log.json').write_text(json.dumps(dict(run_id=run_id, spent=runner.spent, rows=runner.log), indent=1))
        print(f'spent ${runner.spent:.4f}; {len(runner.log)} calls; session evidence in {out / "session"}')


if __name__ == '__main__':
    main()
