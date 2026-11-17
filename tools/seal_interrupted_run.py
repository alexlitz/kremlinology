"""Seal a benchmark run that was killed before it could finalize (no API calls).

Writes summary.json (the harness summary over the recorded observations, with
run.stop_reason='interrupted' and planned/attempted/missing counts), targets.csv and
manifest.json, the same files the runner writes on a normal stop. Refuses a run that
already has a manifest. Observations are not modified; cells never attempted stay missing
and analyses treat them as unusable.
Usage: python tools/seal_interrupted_run.py RUN_DIR --reason TEXT
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kremlinology.bench.analysis import summarize  # noqa: E402


def seal(run, reason):
    root = Path(run)
    if (root / 'manifest.json').exists():
        raise ValueError('run already has a manifest')
    plan = json.loads((root / 'plan.json').read_text())
    rows = [json.loads(line) for line in (root / 'observations.jsonl').read_text().splitlines()]
    attempted_cells = {(r['target_id'], r['probe_id'], r['repeat']) for r in rows}
    missing = [c for c in plan['schedule'] if (c['target'], c['probe'], c['repeat']) not in attempted_cells]
    costs = [r.get('accounted_cost_usd') for r in rows]
    summary = summarize(rows, plan['config'])
    summary['run'] = dict(
        stop_reason='interrupted', interruption=reason, planned=len(plan['schedule']), recorded=len(rows),
        attempted=len(rows), cells_attempted=len(attempted_cells), cells_missing=len(missing),
        accounted_cost_usd=sum(c for c in costs if isinstance(c, (int, float))),
        actual_billing_unknown=True, sealed_after_interruption=True,
        synthetic_only=all(t['adapter'] == 'fixture' for t in plan['config']['targets']))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (root / 'targets.csv').open('w', newline='') as handle:
        if summary['targets']:
            flat = [{k: v for k, v in t.items() if not isinstance(v, (dict, list))} for t in summary['targets']]
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()}
    (root / 'manifest.json').write_text(json.dumps(dict(
        schema_version=1, files=files,
        scope='Sealed after interruption: file integrity of what was recorded, not signed provenance.'), indent=2))
    return summary['run']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.run, args.reason), indent=2))


if __name__ == '__main__':
    main()
