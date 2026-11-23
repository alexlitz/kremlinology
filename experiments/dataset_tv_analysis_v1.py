"""Dataset-average first-token TV bounds from a benchmark run (no API calls).

For each prompt and challenger, repeat-matched challenger/reference pairs give
partial-report TV intervals under each report-error allowance; a prompt's
interval is the mean over its assessable pairs. The reference's own repeat
pair gives a noise floor. Two estimands per dataset:
  assessable   average over prompts where both sides are structurally usable
               (conditional; abstention may correlate with TV)
  conservative unusable prompts count as [0, 1] (valid for the whole sample)
Bounds are conditional on the interval contract (same template/token events,
pre-sampling report stage, report error within the allowance).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kremlinology.bench.analysis import alignment, first_law, partial_tv_bounds  # noqa: E402
from kremlinology.bench.dataset_bounds import average_interval, proportion_interval  # noqa: E402
from kremlinology.bench.runner import write_derived  # noqa: E402

ALLOWANCES = (0.0, 0.01, 0.05)
THRESHOLD = 0.05


def usable(row):
    return row and alignment(row)['status'] == 'structurally_aligned' and first_law(row) is not None


def bounds(left, right, error):
    result = partial_tv_bounds(first_law(left), first_law(right), error)
    if result['lower'] is None:
        return None
    return result['lower'], result['upper']


def margin(row):
    values = sorted(first_law(row).values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else math.inf


def serving(run, row):
    """Declared backend fingerprint and prefix-cache hit, from the row or (older runs) its capture."""
    fingerprint = (row.get('system_fingerprints') or [None])[0]
    if 'system_fingerprints' not in row:
        import base64
        capture = Path(run) / 'captures' / f"{row['call_id']}.json"
        if capture.is_file():
            body = json.loads(base64.b64decode(json.loads(capture.read_text())['body_base64']) or b'{}')
            fingerprint = body.get('system_fingerprint')
    details = row['usage'].get('prompt_tokens_details') or {}
    cached = details.get('cached_tokens')
    return str(fingerprint), (None if cached is None else cached > 0)


def stratified(run, table, reference, challengers, repeats, error):
    """Mean TV midpoint by challenger fingerprint and by both sides' prefix-cache state.

    Strata are observational (not randomized), so prompt mix and time confound them.
    """
    import statistics
    groups = defaultdict(list)
    for probe, cells in table.items():
        for r in repeats:
            ref = cells.get((reference, r))
            if not usable(ref):
                continue
            _, ref_cached = serving(run, ref)
            for challenger in challengers + [reference]:
                if challenger == reference:
                    if r != repeats[0] or len(repeats) < 2 or not usable(cells.get((reference, repeats[1]))):
                        continue
                    other = cells[(reference, repeats[1])]
                else:
                    other = cells.get((challenger, r))
                    if not usable(other):
                        continue
                b = bounds(other, ref, error)
                if b is None:
                    continue
                fingerprint, cached = serving(run, other)
                name = 'reference_self' if challenger == reference else challenger
                key = (name, fingerprint, f'challenger_cached={cached}', f'reference_cached={ref_cached}')
                groups[key].append((b[0] + b[1]) / 2)
    out = []
    for key, values in sorted(groups.items()):
        se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        out.append(dict(target=key[0], fingerprint=key[1], challenger_cache=key[2], reference_cache=key[3],
                        pairs=len(values), mean_tv_midpoint=statistics.mean(values),
                        clt_95_halfwidth=None if se is None else 1.96 * se))
    return out


def analyze(run, reference):
    rows = [json.loads(line) for line in (Path(run) / 'observations.jsonl').read_text().splitlines()]
    # Keep the final attempt per cell; failed retried attempts stay in the evidence.
    rows = [r for r in rows if r['status'] == 'success'] + [r for r in rows if r['status'] != 'success']
    table = defaultdict(dict)
    for row in rows:
        table[row['probe_id']].setdefault((row['target_id'], row['repeat']), row)
    targets = sorted({row['target_id'] for row in rows})
    challengers = [t for t in targets if t != reference]
    repeats = sorted({row['repeat'] for row in rows})
    report = dict(run=str(run), reference=reference, allowances=ALLOWANCES, threshold=THRESHOLD, results={})
    for error in ALLOWANCES:
        by_dataset = defaultdict(lambda: defaultdict(list))
        for probe, cells in sorted(table.items()):
            dataset = probe.split('-')[0]
            self_pair = None
            if len(repeats) >= 2 and all(usable(cells.get((reference, r))) for r in repeats[:2]):
                self_pair = bounds(cells[(reference, repeats[0])], cells[(reference, repeats[1])], error)
            by_dataset[dataset]['reference_self'].append(self_pair)
            for challenger in challengers:
                pairs = [bounds(cells[(challenger, r)], cells[(reference, r)], error) for r in repeats
                         if usable(cells.get((challenger, r))) and usable(cells.get((reference, r)))]
                pairs = [p for p in pairs if p is not None]
                interval = None
                if pairs:
                    interval = (sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs))
                ref_margin = next((margin(cells[(reference, r)]) for r in repeats
                                   if usable(cells.get((reference, r)))), None)
                by_dataset[dataset][challenger].append(dict(interval=interval, self=self_pair, margin=ref_margin))
        summary = {}
        for dataset, groups in sorted(by_dataset.items()):
            summary[dataset] = {}
            selfs = groups.pop('reference_self')
            summary[dataset]['reference_self'] = describe([(s, None, None) for s in selfs])
            for challenger, items in groups.items():
                summary[dataset][challenger] = describe([(i['interval'], i['self'], i['margin']) for i in items])
        report['results'][str(error)] = summary
    report['stratified_at_0.01'] = stratified(run, table, reference, challengers, repeats, 0.01)
    return report


def describe(items):
    total = len(items)
    assessable = [(interval, own, m) for interval, own, m in items if interval is not None]
    result = dict(prompts=total, assessable=len(assessable))
    if len(assessable) < 2:
        return result
    lowers = [a[0][0] for a in assessable]
    uppers = [a[0][1] for a in assessable]
    result['assessable_average'] = average_interval(lowers, uppers)
    result['conservative_average'] = average_interval(
        lowers + [0.0] * (total - len(assessable)), uppers + [1.0] * (total - len(assessable)))
    above = sum(low > THRESHOLD for low in lowers)
    result['fraction_lower_above_threshold'] = proportion_interval(above, len(assessable))
    with_self = [(a[0][0], a[1][1]) for a in assessable if a[1] is not None]
    if with_self:
        beyond = sum(low > own for low, own in with_self)
        result['fraction_beyond_reference_self_upper'] = proportion_interval(beyond, len(with_self))
    margins = sorted((a[2], a[0][0]) for a in assessable if a[2] is not None and math.isfinite(a[2]))
    if len(margins) >= 6:
        third = len(margins) // 3
        result['mean_lower_by_reference_top2_margin_tercile'] = [
            sum(v for _, v in part) / len(part)
            for part in (margins[:third], margins[third:2 * third], margins[2 * third:])]
    return result


def main():
    run, reference = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'coreweave-fp16'
    report = analyze(run, reference)
    path = write_derived(run, 'dataset_tv_analysis', report, Path(__file__))
    print(json.dumps(dict(output=str(path)), indent=2))


if __name__ == '__main__':
    main()
