"""Collect config.json for the most-used open LLMs (no weights, no inference).

Selection, recorded with download counts in configs/kv_models_top/manifest.json:
  - the 1,000 most-downloaded Hugging Face models across the text-generation and
    image-text-to-text pipelines (many current LLMs are tagged as VLMs), excluding
    obvious test fixtures (tiny-random, internal testing repos);
  - every OpenRouter model whose catalog entry names a Hugging Face repository.
Configs come from the local HF cache when present, otherwise one small download each.
Gated or missing configs are recorded, not retried. Downloads are a rolling 30-day count.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub as hub

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/kv_models_top'
OPENROUTER = ROOT / 'reports/kv-cache-economics/openrouter-models-2026-11-11.json'
TEST_REPO = re.compile(r'tiny|random|testing|dummy|-test\b|^hf-internal|^trl-internal|^peft-internal', re.I)


def main(limit=1000):
    api = hub.HfApi()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for pipeline in ('text-generation', 'image-text-to-text'):
        for model in api.list_models(pipeline_tag=pipeline, sort='downloads', limit=limit * 2):
            if TEST_REPO.search(model.id):
                continue
            entry = candidates.setdefault(model.id, dict(repo=model.id, downloads=model.downloads or 0,
                                                         pipelines=[], sources=[]))
            entry['pipelines'].append(pipeline)
    top = sorted(candidates.values(), key=lambda e: -e['downloads'])[:limit]
    for entry in top:
        entry['sources'].append(f'hf_top_{limit}')
    selected = {e['repo']: e for e in top}
    for model in json.loads(OPENROUTER.read_text())['data']:
        repo = model.get('hugging_face_id')
        if repo:
            entry = selected.setdefault(repo, dict(repo=repo, downloads=None, pipelines=[], sources=[]))
            entry['sources'].append('openrouter')
            entry.setdefault('openrouter_ids', []).append(model['id'])
    for i, entry in enumerate(selected.values()):
        try:
            path = None
            try:
                path = hub.hf_hub_download(entry['repo'], 'config.json', local_files_only=True)
                entry['config_source'] = 'local_cache'
            except Exception:
                path = hub.hf_hub_download(entry['repo'], 'config.json')
                entry['config_source'] = 'downloaded'
                time.sleep(0.05)
            snapshot = Path(path).parent.name
            target = OUT / f"{entry['repo'].replace('/', '--')}@{snapshot}.json"
            target.write_bytes(Path(path).read_bytes())
            entry.update(config=target.name, snapshot=snapshot)
        except Exception as error:
            entry['config_error'] = type(error).__name__
        if entry.get('downloads') is None:
            try:
                entry['downloads'] = api.model_info(entry['repo']).downloads
            except Exception as error:
                entry['downloads_error'] = type(error).__name__
        if i % 100 == 0:
            print(i, len(selected), flush=True)
    manifest = dict(collected_at=datetime.now(timezone.utc).isoformat(), limit=limit,
                    huggingface_hub=hub.__version__, models=list(selected.values()),
                    scope='Downloads are rolling 30-day Hub counts (proxy for use, inflated by CI and mirrors).')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    ok = sum('config' in e for e in selected.values())
    print(json.dumps(dict(selected=len(selected), configs=ok, errors=len(selected) - ok)))


if __name__ == '__main__':
    main()
