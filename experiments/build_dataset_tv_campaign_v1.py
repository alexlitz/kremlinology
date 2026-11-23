"""Build a dataset-average first-token TV campaign config (no API calls).

Samples prompts uniformly at random (fixed seed, without replacement) from two
locally cached public datasets so averages estimate those datasets' means:
  code - CodeAlpaca-20k instructions (plus input when present)
  wiki - wikitext-103 validation paragraphs, cut to a continuation prompt
Writes the benchmark config and a prompt manifest with dataset indices.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
from datasets import load_dataset  # noqa: E402

MODEL = 'meta-llama/llama-3.3-70b-instruct'
# OpenRouter catalog prices (USD per million tokens) checked 2026-11-10.
TARGETS = [
    ('coreweave-fp16', 'coreweave/fp16', 0.71, 0.71),
    ('akashml-fp8', 'akashml/fp8', 0.20, 0.52),
    ('parasail-fp8', 'parasail/fp8', 0.22, 0.50),
    ('sambanova', 'sambanova-turbo', 0.60, 1.20),
]


def code_prompts(count, generator):
    data = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
    order = list(range(len(data)))
    generator.shuffle(order)
    result = []
    for index in order:
        row = data[index]
        text = row['instruction'].strip() + (('\n\n' + row['input'].strip()) if row['input'].strip() else '')
        if 20 <= len(text) <= 600:
            result.append(dict(dataset='code', source_index=index, text=text))
        if len(result) == count:
            return result
    raise SystemExit('not enough code prompts')


def wiki_prompts(count, generator):
    data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    paragraphs = [i for i, row in enumerate(data) if len(row['text']) > 300 and not row['text'].strip().startswith('=')]
    generator.shuffle(paragraphs)
    result = []
    for index in paragraphs[:count]:
        words = data[index]['text'].split()
        snippet = ' '.join(words[:generator.randint(30, 60)])
        text = 'Continue the following passage. Reply with the continuation only.\n\n' + snippet
        result.append(dict(dataset='wiki', source_index=index, text=text))
    return result


def dataset_revision(repo):
    """Cached Hugging Face dataset snapshot commit(s); local lookup only."""
    from huggingface_hub import constants
    root = Path(constants.HF_HUB_CACHE) / ('datasets--' + repo.replace('/', '--')) / 'snapshots'
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-dataset', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--max-cost-usd', type=float, default=1.0)
    parser.add_argument('--output', default='configs/bench/llama70b-dataset-tv-v1.json')
    args = parser.parse_args()
    generator = random.Random(args.seed)
    prompts = code_prompts(args.per_dataset, generator) + wiki_prompts(args.per_dataset, generator)
    probes = []
    for number, prompt in enumerate(prompts):
        prompt['probe_id'] = f"{prompt['dataset']}-{number:04d}"
        probes.append(dict(id=prompt['probe_id'], messages=[dict(role='user', content=prompt['text'])],
                           max_tokens=1, stream=False))
    targets = [dict(id=name, adapter='openrouter', model=MODEL, provider=slug, api_key_env='OPENROUTER_API_KEY',
                    parameters=dict(logprobs=True, top_logprobs=20),
                    pricing=dict(input_per_million=price_in, output_per_million=price_out))
               for name, slug, price_in, price_out in TARGETS]
    requests = len(targets) * len(probes) * args.repeats
    config = dict(schema_version=1, targets=targets, probes=probes, repeats=args.repeats, seed=args.seed,
                  limits=dict(max_requests=requests, max_output_tokens=requests, max_cost_usd=args.max_cost_usd,
                              deadline_s=30, unknown_cost='stop', min_interval_s=0.25,
                              # Rate limits/gateway errors are retried with backoff;
                              # every failed attempt stays in the evidence.
                              retry_http_statuses=[429, 502, 503, 504], max_retries=5,
                              max_total_retries=400, retry_backoff_s=5),
                  analysis=dict(reference_target='coreweave-fp16', tv_budgets=[0.01, 0.05, 0.1],
                                logprob_error=0.01),
                  evidence=dict(catalog_snapshot=True, attachments=[
                      dict(path=str(Path(args.output).with_suffix('.prompts.json')), role='prompt_manifest')]))
    output = Path(args.output)
    output.write_text(json.dumps(config, indent=1) + '\n')
    manifest = dict(seed=args.seed, per_dataset=args.per_dataset, prompts=prompts,
                    dataset_revisions={'sahil2801/CodeAlpaca-20k': dataset_revision('sahil2801/CodeAlpaca-20k'),
                                       'wikitext (wikitext-103-raw-v1, validation)': dataset_revision('wikitext')},
                    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.prompts.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(config=str(output), prompts=len(probes), requests=requests)))


if __name__ == '__main__':
    main()
