"""Pilot: can OpenRouter Llama 3.3-70B providers be teacher-forced onto a shared history?

Paid, small (~250 one-token requests). Steps:
 1. CoreWeave samples a 12-token continuation per prompt (T=1, top-20 logprobs).
 2. Each provider is asked for the next-token distribution at positions t in
    {1, 4, 8} after the first t-1 generated tokens, by two methods:
      prefill     chat API with a final partial assistant message
      completions raw /completions with the Llama 3 chat template applied locally
 3. Validity, using exact token IDs from the local tokenizer:
      - re-tokenization: forced text must encode to the generated token IDs
      - prompt-token parity: reported prompt_tokens must equal the expected
        continued-history count (closing the turn adds tokens; extra BOS adds one)
      - faithfulness: CoreWeave's forced logprob of its own sampled token must
        match the logprob reported during generation
Raw request/response evidence goes to a new run directory with a manifest.
Run: HF_HUB_OFFLINE=1 python experiments/later_token_forcing_pilot_v1.py OUT
"""
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import SAFE_RESPONSE_HEADERS, redact  # noqa: E402
from kremlinology.bench.tokens import ByteVocabulary  # noqa: E402

MODEL = 'meta-llama/llama-3.3-70b-instruct'
PROVIDERS = {'coreweave-fp16': 'coreweave/fp16', 'akashml-fp8': 'akashml/fp8',
             'parasail-fp8': 'parasail/fp8', 'sambanova': 'sambanova-turbo'}
POSITIONS = (1, 4, 8)
TOKENIZER = 'meta-llama/Llama-3.1-8B-Instruct'  # same vocabulary/template; parity checked on 3,200 requests
PROMPTS = ROOT / 'configs/bench/llama70b-dataset-tv-v1.prompts.json'
BASE = 'https://openrouter.ai/api/v1'


class Recorder:
    def __init__(self, root, key):
        self.root, self.key = Path(root), key
        if self.root.exists():
            raise SystemExit('new output directory required')
        (self.root / 'captures').mkdir(parents=True)
        self.client = httpx.Client(timeout=45, follow_redirects=False, trust_env=False)
        self.count, self.cost = 0, 0.0

    def post(self, kind, body, meta):
        call = f'{self.count:05d}'
        self.count += 1
        for attempt in range(4):
            started = time.time()
            try:
                response = self.client.post(f'{BASE}/{kind}', json=body,
                                            headers=dict(authorization='Bearer ' + self.key))
                status, raw = response.status_code, response.content
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_RESPONSE_HEADERS}
            except httpx.HTTPError as error:
                status, raw, headers = None, b'', dict(error=type(error).__name__)
            record = dict(call=call, attempt=attempt, kind=kind, meta=meta, request=body, status=status,
                          headers=headers, started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(),
                          elapsed_ms=1000 * (time.time() - started))
            safe = redact(raw, [self.key])
            (self.root / 'captures' / f'{call}-{attempt}.json').write_text(
                json.dumps(dict(body_base64=base64.b64encode(safe).decode(),
                                sha256=hashlib.sha256(safe).hexdigest())))
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {}
            record['usage'] = parsed.get('usage') if isinstance(parsed, dict) else None
            if isinstance(record['usage'], dict) and isinstance(record['usage'].get('cost'), (int, float)):
                self.cost += record['usage']['cost']
            with (self.root / 'requests.jsonl').open('a') as handle:
                handle.write(redact(json.dumps(record).encode(), [self.key]).decode() + '\n')
            time.sleep(0.25)
            if status not in (429, 502, 503, 504):
                return status, parsed
            time.sleep(5 * 2 ** attempt)
        return status, parsed

    def finish(self, summary):
        (self.root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        files = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob('*') if p.is_file()}
        (self.root / 'manifest.json').write_text(json.dumps(dict(schema_version=1, files=files), indent=2))


def routing(slug):
    return dict(only=[slug], allow_fallbacks=False, require_parameters=True)


def chat_law(parsed, vocabulary):
    try:
        record = parsed['choices'][0]['logprobs']['content'][0]
    except (KeyError, IndexError, TypeError):
        return None
    return {vocabulary.token_id(a['bytes']): a['logprob'] for a in record.get('top_logprobs') or []}


def completion_law(parsed, vocabulary):
    try:
        logprobs = parsed['choices'][0]['logprobs']
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(logprobs, dict):
        return None
    if isinstance(logprobs.get('content'), list) and logprobs['content']:  # chat-style shape
        record = logprobs['content'][0]
        return {vocabulary.token_id(a['bytes']): a['logprob'] for a in record.get('top_logprobs') or []}
    tops = logprobs.get('top_logprobs')
    if isinstance(tops, list) and tops and isinstance(tops[0], dict):  # legacy {token: logprob}
        return {vocabulary.token_id(token.encode()): value for token, value in tops[0].items()}
    return None


def tv_bounds(left, right, error=0.01):
    """Partial-report TV interval (same contract as kremlinology.bench.analysis)."""
    from kremlinology.bench.analysis import partial_tv_bounds
    as_labels = [{str(k): v for k, v in law.items() if k is not None} for law in (left, right)]
    return partial_tv_bounds(*as_labels, error)


def main():
    out = Path(sys.argv[1])
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    snapshots = Path.home() / '.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots'
    vocabulary = ByteVocabulary(sorted(snapshots.glob('*/tokenizer.json'))[0])
    manifest = json.loads(PROMPTS.read_text())['prompts']
    prompts = [p for p in manifest if p['dataset'] == 'code'][:5] + [p for p in manifest if p['dataset'] == 'wiki'][:5]
    recorder = Recorder(out, key)
    (out / 'pilot_config.json').write_text(json.dumps(dict(
        model=MODEL, providers=PROVIDERS, positions=POSITIONS, tokenizer=TOKENIZER,
        tokenizer_sha256=vocabulary.sha256, prompt_ids=[p['probe_id'] for p in prompts],
        prompts_manifest_sha256=hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    results = []
    for prompt in prompts:
        messages = [dict(role='user', content=prompt['text'])]
        template_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        raw_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        status, parsed = recorder.post('chat/completions', dict(
            model=MODEL, messages=messages, max_tokens=12, temperature=1.0, logprobs=True, top_logprobs=20,
            provider=routing(PROVIDERS['coreweave-fp16'])), dict(step='history', prompt=prompt['probe_id']))
        try:
            records = parsed['choices'][0]['logprobs']['content']
        except (KeyError, IndexError, TypeError):
            results.append(dict(prompt=prompt['probe_id'], error='no history'))
            continue
        generated = [vocabulary.token_id(r['bytes']) for r in records]
        generated_logprobs = [r['logprob'] for r in records]
        item = dict(prompt=prompt['probe_id'], generated_ids=generated, positions={})
        for t in POSITIONS:
            if t - 1 > len(generated) or None in generated[:t - 1]:
                continue
            prefix_ids = generated[:t - 1]
            prefix_text = b''.join(bytes(r['bytes']) for r in records[:t - 1]).decode('utf-8', 'replace')
            retokenized = tokenizer.encode(prefix_text, add_special_tokens=False)
            full = tokenizer.encode(raw_template + prefix_text, add_special_tokens=False)
            expected = len(template_ids) + len(prefix_ids)
            position = dict(prefix_ids=prefix_ids, prefix_text=prefix_text,
                            retokenizes_exactly=retokenized == prefix_ids,
                            full_prompt_matches=full == template_ids + prefix_ids,
                            expected_prompt_tokens=expected,
                            reference_generation_logprob=generated_logprobs[t - 1] if t - 1 < len(records) else None,
                            reference_generated_id=generated[t - 1] if t - 1 < len(generated) else None,
                            methods={})
            for name, slug in PROVIDERS.items():
                for method in ('prefill', 'completions'):
                    if method == 'prefill':
                        body = dict(model=MODEL, max_tokens=1, logprobs=True, top_logprobs=20,
                                    messages=messages + ([dict(role='assistant', content=prefix_text)] if t > 1 else []),
                                    provider=routing(slug))
                        status, parsed = recorder.post('chat/completions', body,
                                                       dict(step='force', prompt=prompt['probe_id'], t=t,
                                                            provider=name, method=method))
                        law = chat_law(parsed, vocabulary)
                    else:
                        body = dict(model=MODEL, prompt=raw_template + prefix_text, max_tokens=1, logprobs=True,
                                    top_logprobs=20, provider=routing(slug))
                        status, parsed = recorder.post('completions', body,
                                                       dict(step='force', prompt=prompt['probe_id'], t=t,
                                                            provider=name, method=method))
                        law = completion_law(parsed, vocabulary)
                    usage = parsed.get('usage') if isinstance(parsed, dict) else None
                    reported = usage.get('prompt_tokens') if isinstance(usage, dict) else None
                    position['methods'].setdefault(method, {})[name] = dict(
                        status=status, prompt_tokens=reported,
                        prompt_token_delta=None if reported is None else reported - expected,
                        law={str(k): v for k, v in (law or {}).items()}, unmapped=None if law is None else sum(k is None for k in law))
            item['positions'][str(t)] = position
        results.append(item)
    # Validity summary.
    summary = dict(requests=recorder.count, reported_cost_usd=recorder.cost, methods={})
    for method in ('prefill', 'completions'):
        per = {}
        for name in PROVIDERS:
            deltas, laws, faithful = [], 0, []
            for item in results:
                for t, position in item.get('positions', {}).items():
                    entry = position['methods'].get(method, {}).get(name)
                    if not entry:
                        continue
                    deltas.append((int(t), entry['prompt_token_delta']))
                    laws += bool(entry['law'])
                    target = str(position['reference_generated_id'])
                    if name == 'coreweave-fp16' and entry['law'] and target in entry['law'] \
                            and position['reference_generation_logprob'] is not None:
                        faithful.append(abs(entry['law'][target] - position['reference_generation_logprob']))
            per[name] = dict(responses_with_logprobs=laws, prompt_token_deltas=sorted(set(deltas), key=str),
                             faithfulness_abs_logprob_diff=dict(n=len(faithful), max=max(faithful, default=None),
                                                                median=sorted(faithful)[len(faithful) // 2] if faithful else None))
        summary['methods'][method] = per
    summary['retokenization'] = dict(
        checked=sum(len(i.get('positions', {})) for i in results),
        exact=sum(p['retokenizes_exactly'] for i in results for p in i.get('positions', {}).values()),
        full_prompt_exact=sum(p['full_prompt_matches'] for i in results for p in i.get('positions', {}).values()))
    tv = {}
    for method in ('prefill', 'completions'):
        for name in PROVIDERS:
            if name == 'coreweave-fp16':
                continue
            values = []
            for item in results:
                for t, position in item.get('positions', {}).items():
                    m = position['methods'].get(method, {})
                    a, b = m.get(name, {}).get('law'), m.get('coreweave-fp16', {}).get('law')
                    if a and b:
                        bounds = tv_bounds({int(k) if k != 'None' else None: v for k, v in a.items()},
                                           {int(k) if k != 'None' else None: v for k, v in b.items()})
                        if bounds['lower'] is not None:
                            values.append((int(t), (bounds['lower'] + bounds['upper']) / 2))
            tv[f'{method}:{name}'] = {str(t): [round(v, 4) for s, v in values if s == t] for t in POSITIONS}
    summary['tv_midpoints_vs_coreweave'] = tv
    (out / 'results.json').write_text(json.dumps(results, indent=1, allow_nan=False))
    recorder.finish(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
