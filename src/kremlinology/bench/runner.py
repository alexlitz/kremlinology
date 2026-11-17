"""Sequential, bounded runs with immutable plans, source snapshots and evidence."""

import asyncio
from contextlib import nullcontext
import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import httpx
from .adapters import ADAPTERS, observation, redact
from .analysis import summarize
from .config import effective_parameters, planned_output_tokens, validate_config


def plan_benchmark(config):
    from datetime import datetime, timezone

    config = validate_config(config)
    schedule = [
        dict(target=t["id"], probe=p["id"], repeat=r)
        for t in config["targets"]
        for p in config["probes"]
        for r in range(config["repeats"])
    ]
    random.Random(config["seed"]).shuffle(schedule)
    return dict(
        schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        schedule=schedule,
        planned_requests=len(schedule),
        planned_output_token_cap=planned_output_tokens(config),
        scope="Client limits and configured price estimates are not hard billing guarantees. "
        "Provider support declarations and model names are claims, not execution attestations.",
    )


def cost_of(row, target):
    if row["status"] == "skipped":
        return 0.0, "not_sent"
    if row["synthetic"]:
        return 0.0, "synthetic"
    value = row["usage"].get("cost")
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value, "reported"
    pricing = target.get("pricing", {})
    counts = [row["usage"].get("prompt_tokens"), row["usage"].get("completion_tokens")]
    if {"input_per_million", "output_per_million"} <= set(pricing) and all(
        type(v) is int and v >= 0 for v in counts
    ):
        return (
            counts[0] * pricing["input_per_million"]
            + counts[1] * pricing["output_per_million"]
        ) / 1e6, "estimated"
    return None, "unavailable"


def reservation(target, probe):
    price = target.get("pricing", {})
    if not {"input_per_million", "output_per_million"} <= set(price):
        return None
    # Configured reservation, not a bound on hidden templates or native tokenization.
    input_tokens = max(4096, sum(len(m["content"].encode()) for m in probe["messages"]))
    return (
        input_tokens * price["input_per_million"]
        + probe["max_tokens"]
        * effective_parameters(target, probe).get("n", 1)
        * price["output_per_million"]
    ) / 1e6


async def run_benchmark(config, output, *, transport=None):
    plan = plan_benchmark(config)
    config = plan["config"]
    limits = config["limits"]
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("new run directory required")
    targets = {t["id"]: t for t in config["targets"]}
    probes = {p["id"]: p for p in config["probes"]}
    keys = {}
    for target in targets.values():
        if target["adapter"] not in ADAPTERS:
            raise ValueError(f"unregistered adapter: {target['adapter']}")
        env = target.get("api_key_env")
        if env and not os.environ.get(env):
            raise ValueError(f"missing credential environment variable: {env}")
        keys[target["id"]] = os.environ.get(env, "") if env else ""
    secrets = list(keys.values())
    destination.mkdir(parents=True)
    (destination / "captures").mkdir()
    (destination / "source").mkdir()

    def save(path, obj):
        raw = json.dumps(obj, indent=2, allow_nan=False).encode()
        (destination / path).write_bytes(redact(raw, secrets))

    source = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        data = path.read_bytes()
        name = "source/" + path.name
        (destination / name).write_bytes(data)
        source[name] = hashlib.sha256(data).hexdigest()
    plan.update(
        source_snapshot_sha256=source,
        python_version=sys.version,
        httpx_version=httpx.__version__,
    )
    attachments = {}
    if config["evidence"]["attachments"]:
        (destination / "attachments").mkdir()
    for attachment in config["evidence"]["attachments"]:
        data = Path(attachment["path"]).read_bytes()
        if len(data) > 50_000_000:
            raise ValueError("attachment larger than 50 MB")
        name = "attachments/" + Path(attachment["path"]).name
        (destination / name).write_bytes(redact(data, secrets))
        attachments[name] = dict(role=attachment["role"], source_path=attachment["path"],
                                 sha256=hashlib.sha256(data).hexdigest())
    plan.update(attachments=attachments)
    save("plan.json", plan)
    rows = []
    accounted = 0.0
    reported = 0.0
    estimated = 0.0
    unknown = False
    stop = "completed"
    attempted = 0
    failed_skipped = 0
    client = httpx.AsyncClient(
        transport=transport or httpx.AsyncHTTPTransport(retries=0),
        follow_redirects=False,
        trust_env=False,
        timeout=limits["deadline_s"],
    )
    try:
        async with client:
            if config["evidence"]["catalog_snapshot"]:
                await snapshot_catalogs(client, targets.values(), destination, "start", save)
            schedule = plan["schedule"]
            index, retry, retries_used, last_sent = 0, 0, 0, None
            while index < len(schedule):
                cell = schedule[index]
                target = targets[cell["target"]]
                probe = probes[cell["probe"]]
                params = effective_parameters(target, probe)
                call_id = f"{index:06d}" + (f"-retry{retry}" if retry else "")
                declared = target.get("supported_parameters")
                unsupported = (
                    sorted(set(params) - set(declared)) if declared is not None else []
                )
                if unsupported:
                    row = observation(target, probe, cell["repeat"])
                    row.update(
                        status="skipped",
                        warnings=["unsupported_declared_parameters"],
                        unsupported_parameters=unsupported,
                    )
                    capture = None
                else:
                    reserve = reservation(target, probe)
                    if (
                        reserve is not None
                        and accounted + reserve > limits["max_cost_usd"]
                    ):
                        stop = "next_request_reservation_exceeds_budget"
                        break
                    if last_sent is not None and limits["min_interval_s"]:
                        wait = limits["min_interval_s"] - (time.monotonic() - last_sent)
                        if wait > 0:
                            await asyncio.sleep(wait)
                    last_sent = time.monotonic()
                    attempted += 1
                    adapter = ADAPTERS[target["adapter"]]()
                    try:
                        deadline = (
                            nullcontext()
                            if getattr(adapter, "manages_deadline", False)
                            else asyncio.timeout(limits["deadline_s"])
                        )
                        async with deadline:
                            row, capture = await adapter.collect(
                                target,
                                probe,
                                cell["repeat"],
                                limits,
                                client,
                                keys[target["id"]],
                            )
                    except Exception as exc:
                        row = observation(target, probe, cell["repeat"])
                        row.update(status="adapter_error", error=type(exc).__name__)
                        capture = None
                row["call_id"] = call_id
                row["attempt"] = retry
                cost, kind = cost_of(row, target)
                # Configured retryable statuses (e.g. rate limits) are retried with
                # backoff; their billing is assumed, not verified, to be zero.
                retryable = (
                    row["status"] == "http_error"
                    and row.get("http_status") in limits["retry_http_statuses"]
                    and retry < limits["max_retries"]
                    and retries_used < limits["max_total_retries"]
                )
                if retryable and kind == "unavailable":
                    kind = "unavailable_retryable"
                    row["warnings"].append("retryable_failure")
                row["accounted_cost_usd"] = cost
                row["cost_kind"] = kind
                if cost is not None:
                    accounted += cost
                    if kind == "reported":
                        reported += cost
                    if kind == "estimated":
                        estimated += cost
                if kind in ("estimated", "unavailable", "unavailable_retryable"):
                    unknown = True
                if row["status"] not in ("success", "skipped"):
                    unknown = True
                if capture is not None:
                    raw = capture.pop("_body", b"")
                    safe = redact(raw, secrets)
                    capture.update(
                        body_base64=base64.b64encode(safe).decode(),
                        sanitized_body_sha256=hashlib.sha256(safe).hexdigest(),
                        redaction_changed_length=len(raw) != len(safe),
                        retained_original_bytes=len(raw),
                    )
                    save(f"captures/{call_id}.json", capture)
                rows.append(row)
                with (destination / "observations.jsonl").open("ab") as handle:
                    handle.write(
                        redact(json.dumps(row, allow_nan=False).encode(), secrets)
                        + b"\n"
                    )
                save(
                    "progress.json",
                    dict(
                        recorded=len(rows),
                        attempted=attempted,
                        accounted_cost_usd=accounted,
                    ),
                )
                if retryable:
                    retries_used += 1
                    retry += 1
                    await asyncio.sleep(limits["retry_backoff_s"] * 2 ** (retry - 1))
                    continue
                if kind == "unavailable" and limits["unknown_cost"] == "stop":
                    stop = "unaccounted_cost"
                    break
                if row["status"] not in ("success", "skipped"):
                    if failed_skipped < limits["max_failed_requests"]:
                        failed_skipped += 1
                        index, retry = index + 1, 0
                        continue
                    stop = "request_failed"
                    break
                if accounted >= limits["max_cost_usd"]:
                    stop = "accounted_cost_limit"
                    break
                index, retry = index + 1, 0
            if config["evidence"]["catalog_snapshot"]:
                await snapshot_catalogs(client, targets.values(), destination, "end", save)
    except BaseException:
        stop = "interrupted"
        unknown = True
        raise
    finally:
        summary = summarize(rows, config)
        summary["run"] = dict(
            stop_reason=stop,
            planned=len(plan["schedule"]),
            recorded=len(rows),
            attempted=attempted,
            retries=retries_used,
            failed_skipped=failed_skipped,
            accounted_cost_usd=accounted,
            reported_cost_usd=reported,
            estimated_cost_usd=estimated,
            actual_billing_unknown=unknown,
            synthetic_only=all(t["adapter"] == "fixture" for t in targets.values()),
        )
        save("summary.json", summary)
        with (destination / "targets.csv").open("w", newline="") as handle:
            if summary["targets"]:
                flat = [{k: v for k, v in t.items() if not isinstance(v, (dict, list))}
                        for t in summary["targets"]]
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        files = {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.rglob("*")
            if p.is_file()
        }
        save(
            "manifest.json",
            dict(
                schema_version=1,
                files=files,
                scope="File integrity including source snapshots, not signed provider provenance. Manifest is not self-authenticating.",
            ),
        )
    return summary


async def snapshot_catalogs(client, targets, destination, phase, save):
    """Public OpenRouter endpoint catalogs (declared quantization, prices, parameters).

    Unauthenticated GETs; failures are recorded, never fatal. Catalog entries are
    provider declarations, not execution attestations.
    """
    from datetime import datetime, timezone
    from urllib.parse import quote

    seen = set()
    for target in targets:
        if target["adapter"] != "openrouter" or target["model"] in seen:
            continue
        seen.add(target["model"])
        url = target["base_url"].rstrip("/") + "/models/" + quote(target["model"], safe="/") + "/endpoints"
        record = dict(model=target["model"], phase=phase, url=url,
                      fetched_at=datetime.now(timezone.utc).isoformat())
        try:
            response = await client.get(url, timeout=30)
            record.update(http_status=response.status_code, body=response.json())
        except Exception as exc:
            record.update(error=type(exc).__name__)
        (destination / "catalog").mkdir(exist_ok=True)
        slug = target["model"].replace("/", "__")
        save(f"catalog/{phase}-{slug}.json", record)


def write_derived(run, name, obj, producer):
    """Write an analysis output under derived/ and bind it to the run manifest and producer source."""
    from datetime import datetime, timezone

    root = Path(run)
    if not (root / "manifest.json").is_file():
        raise ValueError("run manifest required")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("simple derived output name required")
    (root / "derived").mkdir(exist_ok=True)
    raw = json.dumps(obj, indent=2, allow_nan=False).encode()
    (root / "derived" / f"{name}.json").write_bytes(raw)
    path = root / "derived" / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, outputs={})
    source = Path(producer).read_bytes()
    manifest["outputs"][f"derived/{name}.json"] = dict(
        sha256=hashlib.sha256(raw).hexdigest(),
        run_manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        producer=str(producer),
        producer_sha256=hashlib.sha256(source).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest["scope"] = "Derived analysis integrity, unsigned; not provider attestation."
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return root / "derived" / f"{name}.json"


def verify_run(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    failures = []
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest path")
        file = root / relative
        if (
            root not in file.resolve().parents
            or file.is_symlink()
            or not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != digest
        ):
            failures.append(name)
    derived_failures = []
    derived = root / "derived" / "manifest.json"
    run_digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    outputs = json.loads(derived.read_text())["outputs"] if derived.is_file() else {}
    for name, entry in outputs.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "derived":
            raise ValueError("unsafe derived manifest path")
        file = root / relative
        if (not file.is_file() or file.is_symlink()
                or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]
                or entry["run_manifest_sha256"] != run_digest):
            derived_failures.append(name)
    return dict(
        valid=not failures and not derived_failures,
        files=len(manifest["files"]),
        failures=failures,
        derived_outputs=len(outputs),
        derived_failures=derived_failures,
        scope="Unsigned local manifest integrity, not identity or authenticity attestation.",
    )
