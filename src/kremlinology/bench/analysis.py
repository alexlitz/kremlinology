"""Order-independent sparse probability diagnostics with mandatory abstention."""

import itertools
import math
import statistics
from .sampling import analyze_sampling


def alignment(row):
    records = row["records"]
    count = row["usage"].get("completion_tokens")
    reasons = []
    parts = []
    for record in records:
        if not isinstance(record, dict):
            reasons.append("invalid_record")
            continue
        raw = record.get("bytes")
        lp = record.get("logprob")
        if not isinstance(raw, list) or any(
            type(b) is not int or not 0 <= b <= 255 for b in raw
        ):
            reasons.append("missing_or_invalid_bytes")
        else:
            parts.append(bytes(raw))
        if type(lp) not in (int, float) or not math.isfinite(lp) or lp > 0:
            reasons.append("invalid_selected_logprob")
    if not records:
        reasons.append("no_logprob_records")
    if type(count) is not int or len(records) != count:
        reasons.append("record_count_mismatch")
    if len(parts) != len(records) or b"".join(parts) != row["text"].encode():
        reasons.append("byte_coverage_mismatch")
    if row["status"] != "success":
        reasons.append("request_not_successful")
    if row["request"].get("n", 1) != 1:
        reasons.append("multiple_choices_requested")
    if "response_mode_mismatch" in row["warnings"]:
        reasons.append("response_mode_mismatch")
    return dict(
        status="structurally_aligned" if not reasons else "unassessed",
        reasons=sorted(set(reasons)),
        physical_token_boundaries="unassessed",
        probability_fidelity="unassessed",
    )


def first_law(row):
    if not row["records"] or not isinstance(row["records"][0], dict):
        return None
    alternatives = row["records"][0].get("top_logprobs")
    if not isinstance(alternatives, list) or not alternatives:
        return None
    law = {}
    for alt in alternatives:
        if not isinstance(alt, dict):
            return None
        raw, lp = alt.get("bytes"), alt.get("logprob")
        if (
            not isinstance(raw, list)
            or not raw
            or any(type(b) is not int or not 0 <= b <= 255 for b in raw)
        ):
            return None
        if type(lp) not in (int, float) or not math.isfinite(lp) or lp > 0:
            return None
        key = bytes(raw).hex()
        if key in law:
            return None
        law[key] = lp
    return law


def partial_tv_bounds(left, right, error):
    if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
        raise ValueError("invalid report-error allowance")
    boxes = [
        {
            k: (math.exp(v - error), math.exp(min(0.0, v + error)))
            for k, v in law.items()
        }
        for law in (left, right)
    ]
    masses = [math.fsum(v[0] for v in box.values()) for box in boxes]
    if any(m > 1 for m in masses):
        return dict(status="infeasible_report_intervals", lower=None, upper=None)
    forward = []
    reverse = []
    for k in set(left) | set(right):
        a = boxes[0].get(k, (0.0, 1 - masses[0]))
        b = boxes[1].get(k, (0.0, 1 - masses[1]))
        forward.append(max(0.0, a[0] - b[1]))
        reverse.append(max(0.0, b[0] - a[1]))
    lower = max(math.fsum(forward), math.fsum(reverse))
    upper = 1 - math.fsum(
        min(boxes[0][k][0], boxes[1][k][0]) for k in set(left) & set(right)
    )
    return dict(
        status="conditional_diagnostic",
        lower=lower,
        upper=max(0.0, min(1.0, upper)),
        numerically_certified=False,
    )


def compare(left, right, settings):
    base = dict(
        operational_status="unassessed",
        provider_exclusions=[],
        penalties_authorized=False,
        missing_contracts=[
            "native_token_event_identity",
            "hidden_template_identity",
            "probability_stage",
            "report_error_coverage",
        ],
        scope="Same visible first history only; byte-labelled token events are not a universal text partition. "
        "No substitution, hardware, speculation, or provenance attribution.",
    )
    a = {
        k: v
        for k, v in left["request"].items()
        if k not in ("provider", "stream_options")
    }
    b = {
        k: v
        for k, v in right["request"].items()
        if k not in ("provider", "stream_options")
    }
    if a != b:
        return dict(base, reason="different_model_history_or_sampling_request")
    if a.get("max_tokens") != 1:
        return dict(base, reason="later_divergent_histories_not_comparable")
    if any(alignment(row)["status"] != "structurally_aligned" for row in (left, right)):
        return dict(base, reason="structural_alignment_not_qualified")
    laws = first_law(left), first_law(right)
    if any(law is None for law in laws):
        return dict(base, reason="missing_invalid_or_ambiguous_top_lists")
    bounds = partial_tv_bounds(*laws, settings["logprob_error"])
    sweep = []
    for budget in settings["tv_budgets"]:
        verdict = (
            "unassessed"
            if bounds["lower"] is None
            else "incompatible"
            if bounds["lower"] > budget
            else "compatible"
            if bounds["upper"] <= budget
            else "unassessed"
        )
        sweep.append(dict(tv_budget=budget, conditional_diagnostic=verdict))
    return dict(
        base, bounds=bounds, budget_sweep=sweep, logprob_error=settings["logprob_error"]
    )


def serving_metadata(rows):
    """Declared fingerprints, tiers and prefix-cache hits; declarations, not attestations."""
    fingerprints, tiers = {}, {}
    cached = known = 0
    for row in rows:
        for value in row.get("system_fingerprints") or [None]:
            fingerprints[str(value)] = fingerprints.get(str(value), 0) + 1
        for value in row.get("service_tiers") or [None]:
            tiers[str(value)] = tiers.get(str(value), 0) + 1
        details = row["usage"].get("prompt_tokens_details")
        if isinstance(details, dict) and type(details.get("cached_tokens")) is int:
            known += 1
            cached += details["cached_tokens"] > 0
    return dict(system_fingerprints=fingerprints, service_tiers=tiers,
                prefix_cache_hits=cached, prefix_cache_reported=known)


def summarize(rows, config):
    summaries = []
    for target in config["targets"]:
        selected = [r for r in rows if r["target_id"] == target["id"]]
        success = [r for r in selected if r["status"] == "success"]
        ttft = [r["ttft_ms"] for r in success if r.get("ttft_ms") is not None]
        summaries.append(
            dict(
                target=target["id"],
                model=target["model"],
                synthetic=target["adapter"] == "fixture",
                recorded=len(selected),
                attempts=sum(r["status"] != "skipped" for r in selected),
                successful=len(success),
                skipped=sum(r["status"] == "skipped" for r in selected),
                aligned=sum(
                    alignment(r)["status"] == "structurally_aligned" for r in success
                ),
                logprob_records=sum(len(r["records"]) for r in success),
                median_ttft_ms=statistics.median(ttft) if ttft else None,
                accounted_cost_usd=sum(
                    r.get("accounted_cost_usd") or 0 for r in selected
                ),
                actual_billing_unknown=any(
                    r.get("cost_kind") in ("estimated", "unavailable") for r in selected
                ),
                serving_metadata=serving_metadata(success),
            )
        )
    comparisons = []
    reference = config["analysis"].get("reference_target")
    first = [r for r in rows if r["request"]["max_tokens"] == 1]
    for left, right in itertools.combinations(first, 2):
        if left["probe_id"] != right["probe_id"]:
            continue
        same_target = left["target_id"] == right["target_id"]
        if same_target:
            if reference and left["target_id"] != reference:
                continue
            kind = "within_target_repeat"
        else:
            if left["repeat"] != right["repeat"]:
                continue
            if reference and reference not in (left["target_id"], right["target_id"]):
                continue
            kind = "paired_cross_target"
        comparisons.append(
            dict(
                kind=kind,
                left=left["call_id"],
                right=right["call_id"],
                left_target=left["target_id"],
                right_target=right["target_id"],
                probe=left["probe_id"],
                result=compare(left, right, config["analysis"]),
            )
        )
    return dict(
        targets=summaries,
        comparisons=comparisons,
        sampling=analyze_sampling(rows, config["analysis"]),
        operational_status="unassessed",
        provider_exclusions=[],
        penalties_authorized=False,
        scope="Repeated comparisons reuse observations; not independent trials or calibrated provider risk. "
        "Different checkpoints are not directly TV-compared. No missing report mass set to zero.",
    )
