"""Build the GitHub release data archives from the run directories (2026-11-17).

Each archive is a tar.gz of whole run directories, copied first so the originals are untouched. In the copies, local
home paths (/home/<user>/...) in text files are rewritten to ~/..., but only in files no manifest references (logs,
run scripts, derived manifests' "producer" fields): hashed evidence is never edited, so every manifest still
verifies. Hashed files that contain a home path are left as they are and listed in archives.json (in practice one:
a tokenizer cache path recorded in llama70b-dataset-tv-v2/derived/token_ids.json). The build stops if any file contains
an API-key-like string or a name on the private list (env KREMLINOLOGY_PRIVATE_PATTERNS).
Run: python tools/build_release_archives.py DATA_ROOT OUT_DIR [--only NAME]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path
import os as _os
DATA_ROOT = _os.environ.get('KREMLINOLOGY_DATA', 'data')  # where the release archives are extracted

ARCHIVES = {
    'dataset-tv': ['kremlinology-llama70b-dataset-tv-v1', 'kremlinology-llama70b-dataset-tv-v2',
                   'kremlinology-qwen35-dataset-tv-v1', 'kremlinology-qwen35-dataset-tv-v2',
                   'kremlinology-gemma4-31b-dataset-tv-v1', 'kremlinology-gemma4-31b-dataset-tv-v2',
                   'kremlinology-gemma4-31b-dataset-tv-v3'],
    'timing': ['kremlinology-timing-speculation-v' + v for v in
               ['1', '2', '3', '4', '4a', '4b', '4c', '4d', '4e', '4f', '4g', '5a', '5b', '5c', '5d', '5e', '5f',
                '6a', '6b', '6c', '6d']],
    'cache-semantics': ['kremlinology-cache-semantics-v' + v for v in ['1', '2', '3', '4', '4b', '5', '6', '7a', '7b', '8']]
                       + [f'kremlinology-cache-semantics-open-v{v}' for v in (1, 2, 3)] + ['kremlinology-cache-semantics-v9'],
    'cache-retention': [f'kremlinology-cache-ttl-probe-v{v}' for v in (1, 2, 3, 4)],
    'cache-misc': ['kremlinology-cross-tenant-cache-v1', 'kremlinology-cross-tenant-cache-v2', 'kremlinology-cache-scope-v1',
                   'kremlinology-max-tokens-probe-v1', 'kremlinology-max-tokens-probe-v2', 'kremlinology-max-tokens-probe-v3',
                   'kremlinology-later-token-pilot-v1'],
    'prefill-latency': [f'kremlinology-prefill-sweep-v{v}' for v in (1, 2, 3, 4, 5, 6, '6b', 7, '7b')]
                       + ['kremlinology-cache-latency-v1', 'kremlinology-cache-latency-age-v2', 'kremlinology-cache-latency-age-v3', 'kremlinology-cache-latency-age-v4']
                       + [f'kremlinology-cache-latency-grid-v{v}' for v in (2, 3, 4, 5)],
    'witnessed-sessions': ['kremlinology-witnessed-sessions-2026-11', 'kremlinology-cache-honesty-v1',
                           'kremlinology-cache-honesty-v2', 'kremlinology-cache-honesty-v3'],
    'cache-audit': [  # 2026-11-20 to 11-23: first-party clock tests, DeepSeek probes, hidden-cache and 34 h lifetime
        'kremlinology-haiku-clock-v1', 'kremlinology-haiku-clock-v1-session', 'kremlinology-nano-clock-v1',
        'kremlinology-nano-clock-v1-session', 'kremlinology-turbo-ttl-v1', 'kremlinology-turbo-ttl-v1-session',
        'kremlinology-dsttl-v1', 'kremlinology-dsttl-v1-session', 'kremlinology-dsttl-v1b',
        'kremlinology-dsttl-v1b-session', 'kremlinology-deepseek-direct-v1-session',
        'kremlinology-deepseek-impossible-v1-session', 'kremlinology-deepseek-multiturn-v1-session',
        'kremlinology-variant-pairs-v1', 'kremlinology-variant-pairs-v1-session', 'kremlinology-concurrency-v1',
        'kremlinology-concurrency-v1-session', 'kremlinology-context-bw-v1', 'kremlinology-context-bw-v1-session'],
    'cache-refresh': ['kremlinology-cache-refresh-v1', 'kremlinology-cache-refresh-v2',
                       'kremlinology-cache-refresh-v2-session', 'kremlinology-cache-refresh-v3',
                       'kremlinology-cache-refresh-v3-session', 'kremlinology-novita-cap-v1',
                       'kremlinology-novita-cap-v1-session', 'kremlinology-sample-vs-logprob-ab-w',
                       'kremlinology-sample-vs-logprob-ab-w2', 'kremlinology-sample-vs-logprob-ab-w-session'],
    'adaptive-tv': ['kremlinology-adaptive-tv-hosted-v1', 'kremlinology-adaptive-tv-hosted-v2', 'kremlinology-adaptive-tv-hosted-v3',
                    'kremlinology-adaptive-tv-hosted-v3b', 'kremlinology-adaptive-tv-hosted-w', 'kremlinology-adaptive-tv-w-session',
                    'kremlinology-sample-vs-logprob-v1', 'kremlinology-sample-vs-logprob-ab-v1',
                    'kremlinology-sample-vs-logprob-ab-v2'],
}
HOME = re.compile(rb'/home/[a-z_][a-z0-9_-]*/')
# API-key shapes, plus private names from KREMLINOLOGY_PRIVATE_PATTERNS (comma-separated, kept out of the repo)
_PRIVATE = [re.escape(x.strip()).encode() for x in os.environ.get('KREMLINOLOGY_PRIVATE_PATTERNS', '').split(',') if x.strip()]
SECRET = re.compile(b'|'.join([rb'sk-or-v1-[0-9a-f]{16}', rb'sk-ant-[A-Za-z0-9_-]{16}', rb'sk-proj-[A-Za-z0-9_-]{16}'] + _PRIVATE),
                    re.I)
TEXT = {'.json', '.jsonl', '.md', '.txt', '.log', '.csv', '.sh', '.py', '.toml', '.yaml', '.yml', ''}


def referenced(run):
    """Relative paths of every file some manifest in the run hashes."""
    out = set()
    for m in run.rglob('manifest.json'):
        try:
            d = json.loads(m.read_text())
        except ValueError:
            continue
        base = m.parent
        entries = d.get('files') or d.get('outputs') or {}
        names = entries.keys() if isinstance(entries, dict) else [e.get('path') or e.get('name') for e in entries if isinstance(e, dict)]
        for n in names:
            if not n:
                continue
            for root in (base, run):  # paths are relative to the manifest's directory or to the run root
                p = (root / n).resolve()
                if p.exists():
                    out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root')
    ap.add_argument('out')
    ap.add_argument('--only')
    args = ap.parse_args()
    data, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'staging'
    summary = {}
    for name, runs in ARCHIVES.items():
        if args.only and name != args.only:
            continue
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        scrubbed, kept, problems = [], [], []
        for r in runs:
            src = data / r
            if not src.is_dir():
                problems.append(f'missing {r}')
                continue
            dst = tmp / r
            shutil.copytree(src, dst, symlinks=False)
            hashed = {p.relative_to(src.resolve()) for p in referenced(src.resolve())}
            for f in dst.rglob('*'):
                if not f.is_file():
                    continue
                rel = f.relative_to(dst)
                data_bytes = f.read_bytes()
                if SECRET.search(data_bytes):
                    problems.append(f'secret-like string in {r}/{rel}')
                if f.suffix not in TEXT or not HOME.search(data_bytes):
                    continue
                if rel in hashed:
                    kept.append(f'{r}/{rel}')
                    continue
                f.write_bytes(HOME.sub(b'~/', data_bytes))
                scrubbed.append(f'{r}/{rel}')
        if problems:
            raise SystemExit(f'{name}: ' + '; '.join(problems[:20]))
        archive = out / f'kremlinology-data-{name}.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for r in runs:
                tar.add(tmp / r, arcname=r)
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        summary[name] = dict(archive=archive.name, runs=runs, bytes=archive.stat().st_size, sha256=sha,
                             scrubbed_unhashed_files=scrubbed, home_paths_left_in_hashed_files=kept)
        print(f'{archive.name}: {archive.stat().st_size / 1e6:.1f} MB, {len(scrubbed)} unhashed files scrubbed, '
              f'{len(kept)} hashed files with a home path left as is {kept}', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    prev = json.loads((out / 'archives.json').read_text()) if (out / 'archives.json').exists() else {}
    prev.update(summary)
    (out / 'archives.json').write_text(json.dumps(prev, indent=1))


if __name__ == '__main__':
    main()
