"""Sampled first-token diagnostics across sampler settings, without NumPy.

Rows with max_tokens=1 and the same target and visible messages form a
group; requests inside it differ only in declared temperature, top-K and
top-P. Tokens are identified by first-record bytes (or text), so distinct
labels lower-bound distinct tokens. Statistical flags assume independent
requests and the declared sampler order temperature, top-K, then nucleus.
None of this attributes cause, hardware or provenance.
"""

import json
import math
import random

SETTING_KEYS = ("temperature", "top_k", "top_p")


def first_label(choice, request):
    records = choice.get("records") or []
    record = records[0] if records and isinstance(records[0], dict) else None
    raw = record.get("bytes") if record else None
    if isinstance(raw, list) and raw and all(type(b) is int and 0 <= b <= 255 for b in raw):
        label = bytes(raw)
    elif request.get("max_tokens") == 1 and choice.get("text"):
        label = choice["text"].encode()
    else:
        return None, None
    logprob = record.get("logprob") if record else None
    if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 0:
        logprob = None
    return label.decode("utf-8", "backslashreplace"), logprob


def row_choices(row):
    # Older observations predate per-choice records.
    if row.get("choices"):
        return row["choices"]
    return [dict(index=0, text=row["text"], records=row["records"])]


def setting_of(request):
    return dict(
        temperature=request.get("temperature", 1.0),
        top_k=request.get("top_k", 0) or 0,
        top_p=request.get("top_p", 1.0),
    )


def collect_groups(rows):
    groups = {}
    for row in rows:
        request = row["request"]
        if row["status"] != "success" or request.get("max_tokens") != 1:
            continue
        key = (row["target_id"], json.dumps(request["messages"], sort_keys=True))
        setting = setting_of(request)
        name = json.dumps(setting, sort_keys=True)
        group = groups.setdefault(key, dict(target=row["target_id"], probes=set(), settings={}))
        group["probes"].add(row["probe_id"])
        entry = group["settings"].setdefault(name, dict(setting=setting, requests=[], unlabeled=0))
        counts, logprobs = {}, {}
        for choice in row_choices(row):
            label, logprob = first_label(choice, request)
            if label is None:
                entry["unlabeled"] += 1
                continue
            counts[label] = counts.get(label, 0) + 1
            if logprob is not None:
                logprobs.setdefault(label, []).append(logprob)
        entry["requests"].append(dict(call_id=row.get("call_id"), counts=counts, logprobs=logprobs))
    return groups


def totals(entry):
    result = {}
    for request in entry["requests"]:
        for label, count in request["counts"].items():
            result[label] = result.get(label, 0) + count
    return result


def solve(matrix, vector):
    """Gaussian elimination with partial pivoting for small dense systems."""
    size = len(vector)
    a = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(a[r][column]))
        if abs(a[pivot][column]) < 1e-300:
            raise ValueError("singular system")
        a[column], a[pivot] = a[pivot], a[column]
        for r in range(column + 1, size):
            factor = a[r][column] / a[column][column]
            for c in range(column, size + 1):
                a[r][c] -= factor * a[column][c]
    result = [0.0] * size
    for r in range(size - 1, -1, -1):
        result[r] = (a[r][size] - sum(a[r][c] * result[c] for c in range(r + 1, size))) / a[r][r]
    return result


def chi2_survival(statistic, freedom):
    """Upper tail of a chi-square law via the regularized incomplete gamma."""
    if freedom <= 0:
        return math.nan
    a, x = freedom / 2, statistic / 2
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1 / a
        for n in range(1, 10000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1 / 1e-300, 1 / (x + 1 - a)
    h = d
    for n in range(1, 10000):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def split(entry, part):
    """Alternate requests (in call order) into selection and estimation halves."""
    requests = sorted(entry["requests"], key=lambda r: r.get("call_id") or "")
    return dict(entry, requests=requests[part::2])


def frequent_labels(entries, minimum_count, maximum_labels):
    overall = {}
    for entry in entries:
        for label, count in totals(entry).items():
            overall[label] = overall.get(label, 0) + count
    ranked = sorted(overall, key=lambda label: (-overall[label], label))
    return [label for label in ranked if overall[label] >= minimum_count][:maximum_labels]


def pooled_gaps(entries, *, minimum_count=5, maximum_labels=40, sample_split=True):
    """Conditional multinomial MLE of logit gaps over frequent observed labels.

    Conditioning on any label subset keeps each supported label's renormalized
    ratio exp((z_i-z_j)/T), whatever truncation removed, so rare labels are
    simply left out. Choosing labels by their counts in the same data favours
    rare labels that happened to be oversampled, so by default labels are
    chosen on alternate requests and gaps are estimated on the others (falling
    back to all data when a setting has a single request). Gaps are relative
    to the most frequent label. Requires temperature > 0.
    """
    if sample_split and all(len(e["requests"]) >= 2 for e in entries):
        selection = [split(e, 0) for e in entries]
        entries = [split(e, 1) for e in entries]
    else:
        sample_split, selection = False, entries
    keep = set(frequent_labels(selection, minimum_count, maximum_labels))
    data = []
    for chosen, entry in zip(selection, entries):
        observed = totals(entry)
        setting = entry["setting"]
        if not sample_split:
            support = keep & set(observed)
        elif setting["top_k"] == 0 and setting["top_p"] >= 1:
            support = keep
        else:
            # Truncated support is unknown; trust labels seen in the selection half.
            support = keep & set(totals(chosen))
        # Zero counts stay in: dropping unseen labels would favour lucky ones.
        counts = {k: observed.get(k, 0) for k in support}
        if setting["temperature"] > 0 and len(counts) >= 2:
            data.append((counts, setting["temperature"]))
    seen = {k for counts, _ in data for k, v in counts.items() if v}
    data = [({k: v for k, v in counts.items() if k in seen}, t) for counts, t in data]
    data = [(counts, t) for counts, t in data if len(counts) >= 2]
    if not data:
        return None
    overall = {}
    for counts, _ in data:
        for label, count in counts.items():
            overall[label] = overall.get(label, 0) + count
    labels = sorted(overall, key=lambda label: (-overall[label], label))
    # Labels only seen alone in a setting carry no gap information.
    index = {label: i for i, label in enumerate(labels)}
    parent = list(range(len(labels)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for counts, _ in data:
        ids = [index[label] for label in counts]
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    root = find(0)
    labels = [label for label in labels if find(index[label]) == root]
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    if size < 2:
        return None
    def evaluate(point):
        loglik, gradient = 0.0, [0.0] * size
        information = [[0.0] * size for _ in range(size)]
        for counts, temperature in data:
            ids = [index[label] for label in counts if label in index]
            if len(ids) < 2:
                continue
            values = [counts[labels[i]] for i in ids]
            scaled = [point[i] / temperature for i in ids]
            peak = max(scaled)
            weights = [math.exp(v - peak) for v in scaled]
            norm = math.fsum(weights)
            probability = [w / norm for w in weights]
            total = sum(values)
            loglik += math.fsum(c * (v - peak) for c, v in zip(values, scaled)) - total * math.log(norm)
            for a, i in enumerate(ids):
                gradient[i] += (values[a] - total * probability[a]) / temperature
                for b, j in enumerate(ids):
                    delta = probability[a] * ((a == b) - probability[b])
                    information[i][j] += total * delta / temperature**2
        return loglik, gradient, information

    gaps = [0.0] * size
    loglik, gradient, information = evaluate(gaps)
    for _ in range(200):
        step = solve([row[1:] for row in information[1:]], gradient[1:])
        # Backtracking keeps Newton from overshooting into underflow.
        scale = 1.0
        while True:
            trial = [gaps[0]] + [g + scale * v for g, v in zip(gaps[1:], step)]
            result = evaluate(trial)
            if result[0] >= loglik - 1e-12 or scale < 1e-8:
                break
            scale /= 2
        gaps, (loglik, gradient, information) = trial, result
        if max(abs(scale * v) for v in step) < 1e-10:
            break
    reduced = [row[1:] for row in information[1:]]
    columns = [solve(reduced, [float(i == j) for j in range(size - 1)]) for i in range(size - 1)]
    errors = [0.0] + [math.sqrt(max(0.0, columns[i][i])) for i in range(size - 1)]
    # Goodness of fit of the shared-gap model across settings.
    statistic, freedom = 0.0, 0
    for counts, temperature in data:
        ids = [index[label] for label in counts if label in index]
        if len(ids) < 2:
            continue
        scaled = [gaps[i] / temperature for i in ids]
        peak = max(scaled)
        weights = [math.exp(s - peak) for s in scaled]
        norm = math.fsum(weights)
        total = sum(counts[labels[i]] for i in ids)
        for i, weight in zip(ids, weights):
            expected = total * weight / norm
            statistic += (counts[labels[i]] - expected) ** 2 / expected
        freedom += len(ids) - 1
    freedom -= size - 1
    return dict(
        reference_label=labels[0],
        gaps={label: dict(gap=gaps[i], standard_error=errors[i]) for label, i in index.items()},
        # Gaps share the reference label, so their errors are correlated.
        covariance_labels=labels[1:],
        covariance=columns,
        goodness_of_fit=dict(
            statistic=statistic,
            degrees_of_freedom=freedom,
            pvalue=chi2_survival(statistic, freedom) if freedom > 0 else None,
        ),
        sample_split=sample_split,
        scope="logit gaps up to a shift, assuming declared temperatures are applied",
    )


def temperature_ratios(entries):
    """Naseh et al. (2023) ratio of top-pair log-ratios between settings."""
    results = []
    usable = [e for e in entries if e["setting"]["temperature"] > 0]
    for i, left in enumerate(usable):
        for right in usable[i + 1:]:
            a, b = totals(left), totals(right)
            shared = sorted(set(a) & set(b), key=lambda label: -(a[label] + b[label]))
            if len(shared) < 2:
                continue
            first, second = shared[:2]
            if min(a[first], a[second], b[first], b[second]) < 5:
                continue
            ratio_a = math.log(a[first] / a[second])
            ratio_b = math.log(b[first] / b[second])
            if abs(ratio_a) < 1e-12:
                continue
            estimate = ratio_b / ratio_a
            variance_a = 1 / a[first] + 1 / a[second]
            variance_b = 1 / b[first] + 1 / b[second]
            error = abs(estimate) * math.sqrt(variance_a / ratio_a**2 + variance_b / ratio_b**2)
            declared = left["setting"]["temperature"] / right["setting"]["temperature"]
            results.append(
                dict(
                    settings=[left["setting"], right["setting"]],
                    labels=[first, second],
                    estimated_temperature_ratio=estimate,
                    declared_temperature_ratio=declared,
                    standard_error=error,
                    z=(estimate - declared) / error if error > 0 else None,
                )
            )
    return results


def overdispersion(requests, permutations, seed, keep=None):
    """Permutation test; labels outside keep share one 'other' column."""
    if keep is not None:
        requests = [dict(counts=collapse(r["counts"], keep)) for r in requests]
    labels = sorted({label for r in requests for label in r["counts"]})
    table = [[r["counts"].get(label, 0) for label in labels] for r in requests]
    sizes = [sum(row) for row in table]
    if len(table) < 2 or len(labels) < 2 or min(sizes) < 2:
        return None
    grand = sum(sizes)
    columns = [sum(row[c] for row in table) for c in range(len(labels))]

    def statistic(rows):
        value = 0.0
        for row, size in zip(rows, sizes):
            for c, observed in enumerate(row):
                expected = size * columns[c] / grand
                value += (observed - expected) ** 2 / expected
        return value
    observed = statistic(table)
    pool = [c for c, count in enumerate(columns) for _ in range(count)]
    generator = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        generator.shuffle(pool)
        rows, start = [], 0
        for size in sizes:
            row = [0] * len(labels)
            for c in pool[start:start + size]:
                row[c] += 1
            rows.append(row)
            start += size
        exceed += statistic(rows) >= observed
    freedom = (len(table) - 1) * (len(labels) - 1)
    dispersion = observed / freedom
    mean_size = grand / len(table)
    return dict(
        requests=len(table),
        dispersion=dispersion,
        intraclass_correlation=(dispersion - 1) / (mean_size - 1),
        permutation_pvalue=(exceed + 1) / (permutations + 1),
        single_token_request_fraction=sum(sum(v > 0 for v in row) == 1 for row in table) / len(table),
    )


def collapse(counts, keep):
    result = {}
    for label, count in counts.items():
        key = label if label in keep else "\x00other"
        result[key] = result.get(key, 0) + count
    return result


def reported_consistency(entry):
    """Reported selected-token probabilities against sampled frequencies.

    Only for temperature 1 without truncation, where the sampling law equals
    the unprocessed distribution. Unreported or rare mass forms one 'other'
    bucket.
    """
    setting = entry["setting"]
    if setting != dict(temperature=1, top_k=0, top_p=1):
        return None
    counts = totals(entry)
    reports = {}
    for request in entry["requests"]:
        for label, values in request["logprobs"].items():
            reports.setdefault(label, []).extend(values)
    if not reports:
        return None
    total = sum(counts.values())
    probability = {label: math.exp(math.fsum(v) / len(v)) for label, v in reports.items()}
    spread = max(max(v) - min(v) for v in reports.values())
    # A label is reported only once sampled; keeping rare labels would bias the
    # test upward. Labels expected fewer than five times join 'other'.
    probability = {label: p for label, p in probability.items() if total * p >= 5}
    if not probability:
        return None
    other = 1 - math.fsum(probability.values())
    statistic, cells = 0.0, 0
    for label, p in probability.items():
        statistic += (counts.get(label, 0) - total * p) ** 2 / (total * p)
        cells += 1
    observed_other = total - sum(counts.get(label, 0) for label in probability)
    if other > 1e-12:
        statistic += (observed_other - total * other) ** 2 / (total * other)
        cells += 1
    elif observed_other:
        return dict(status="reported_mass_exceeds_samples", samples=total)
    freedom = cells - 1
    return dict(
        samples=total,
        labels_reported=len(probability),
        maximum_report_spread_nats=spread,
        statistic=statistic,
        degrees_of_freedom=freedom,
        pvalue=chi2_survival(statistic, freedom) if freedom > 0 else None,
        probability_stage="assumed_unprocessed_at_temperature_1",
    )


def analyze_sampling(rows, settings):
    alpha = settings["sampling_alpha"]
    permutations = settings["sampling_permutations"]
    results = []
    for (target, _), group in sorted(collect_groups(rows).items()):
        entries = list(group["settings"].values())
        per_setting, violations, flags = [], [], []
        for entry in entries:
            counts = totals(entry)
            setting = entry["setting"]
            choices = [sum(r["counts"].values()) for r in entry["requests"]]
            item = dict(
                setting=setting,
                requests=len(entry["requests"]),
                samples=sum(choices),
                unlabeled=entry["unlabeled"],
                counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            )
            k = setting["top_k"]
            deterministic = setting["temperature"] == 0 or k == 1
            # Only one request's choices share logits; pooled supports may
            # legitimately differ when logits vary between requests.
            within = max((len(r["counts"]) for r in entry["requests"]), default=0)
            item["maximum_distinct_labels_within_request"] = within
            if k and within > k:
                violations.append(dict(kind="top_k_support_exceeded_within_request",
                                       setting=setting, distinct_labels=within))
            if deterministic and within > 1:
                violations.append(dict(kind="deterministic_setting_varied_within_request",
                                       setting=setting, distinct_labels=within))
            if (k and len(counts) > k) or (deterministic and len(counts) > 1):
                flags.append(dict(kind="pooled_support_exceeds_declared_truncation",
                                  setting=setting, distinct_labels=len(counts),
                                  note="definite only if logits are fixed across requests"))
            if max(choices, default=0) > 1:
                seed = hash_seed(target, setting)
                keep = set(frequent_labels([entry], 5, 40))
                spread = overdispersion(entry["requests"], permutations, seed, keep)
                item["overdispersion"] = spread
                if spread and spread["permutation_pvalue"] <= alpha:
                    flags.append(dict(kind="between_request_overdispersion", setting=setting,
                                      pvalue=spread["permutation_pvalue"]))
            consistency = reported_consistency(entry)
            if consistency is not None:
                item["reported_logprob_consistency"] = consistency
                p = consistency.get("pvalue")
                if consistency.get("status") or (p is not None and p <= alpha):
                    flags.append(dict(kind="reported_logprobs_inconsistent_with_samples",
                                      setting=setting, pvalue=p))
            per_setting.append(item)
        gaps = pooled_gaps(entries)
        if gaps and gaps["goodness_of_fit"]["pvalue"] is not None and gaps["goodness_of_fit"]["pvalue"] <= alpha:
            flags.append(dict(kind="settings_inconsistent_with_shared_logits",
                              pvalue=gaps["goodness_of_fit"]["pvalue"]))
        ratios = temperature_ratios(entries)
        results.append(
            dict(
                target=target,
                probes=sorted(group["probes"]),
                settings=per_setting,
                pooled_gaps=gaps,
                temperature_ratios=ratios,
                observed_violations=violations,
                statistical_flags=flags,
                tests_performed=sum(
                    ("overdispersion" in item) + ("reported_logprob_consistency" in item)
                    for item in per_setting
                )
                + bool(gaps and gaps["goodness_of_fit"]["pvalue"] is not None),
                alpha=alpha,
                operational_status="unassessed",
                penalties_authorized=False,
            )
        )
    return dict(
        groups=results,
        scope="First-token samples grouped by target and visible messages. Violations are definite only "
        "under declared parameters and one native token per label; flags are per-test alpha without "
        "multiplicity correction and can arise from ignored parameters, sampler order, varying logits "
        "or dependent choices. Not provenance, hardware, or fraud attribution.",
    )


def hash_seed(target, setting):
    import hashlib

    raw = json.dumps([target, setting], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
