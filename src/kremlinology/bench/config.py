"""Strict JSON configuration, with no credentials stored in configuration."""

import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit
from .probes import expand_probe

RESERVED = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "provider",
    "stream_options",
    "api_key",
    "headers",
}


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


def _positive(value, label, maximum=1_000_000):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"invalid {label}")


def _parameters(value):
    if not isinstance(value, dict) or RESERVED & set(value):
        raise ValueError("invalid or reserved sampling parameters")
    if "temperature" in value and (
        type(value["temperature"]) not in (int, float)
        or not math.isfinite(value["temperature"])
        or value["temperature"] < 0
    ):
        raise ValueError("invalid temperature")
    if "top_p" in value and (
        type(value["top_p"]) not in (int, float) or not 0 < value["top_p"] <= 1
    ):
        raise ValueError("invalid top_p")
    if "top_k" in value and (type(value["top_k"]) is not int or value["top_k"] < 0):
        raise ValueError("invalid top_k")
    if "n" in value and (type(value["n"]) is not int or not 1 <= value["n"] <= 256):
        raise ValueError("invalid n")
    if "logprobs" in value and type(value["logprobs"]) is not bool:
        raise ValueError("logprobs must be boolean")
    if "top_logprobs" in value and (
        type(value["top_logprobs"]) is not int or not 0 <= value["top_logprobs"] <= 20
    ):
        raise ValueError("invalid top_logprobs")
    if value.get("top_logprobs", 0) and value.get("logprobs") is not True:
        raise ValueError("top_logprobs requires logprobs=true")
    json.dumps(value, allow_nan=False)


def validate_config(raw):
    import copy

    config = copy.deepcopy(raw)
    _keys(
        config,
        {
            "schema_version",
            "targets",
            "probes",
            "repeats",
            "seed",
            "limits",
            "analysis",
            "evidence",
        },
        "config",
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for name in ("targets", "probes"):
        if not isinstance(config.get(name), list) or not config[name]:
            raise ValueError(f"nonempty {name} required")
    config.setdefault("repeats", 2)
    _positive(config["repeats"], "repeats", 1000)
    config.setdefault("seed", 0)
    if type(config["seed"]) is not int:
        raise ValueError("integer seed required")
    seen = set()
    for target in config["targets"]:
        _keys(
            target,
            {
                "id",
                "adapter",
                "model",
                "base_url",
                "api_key_env",
                "provider",
                "parameters",
                "supported_parameters",
                "pricing",
                "allow_insecure_http",
                "fixture_variant",
            },
            "target",
        )
        for key in ("id", "model"):
            if not isinstance(target.get(key), str) or not target[key]:
                raise ValueError(f"target {key} required")
        if target["id"] in seen:
            raise ValueError("duplicate target ID")
        seen.add(target["id"])
        target.setdefault("adapter", "chat_completions")
        if not isinstance(target["adapter"], str) or not target["adapter"]:
            raise ValueError("adapter name required")
        target.setdefault("parameters", {})
        _parameters(target["parameters"])
        if "supported_parameters" in target and (
            not isinstance(target["supported_parameters"], list)
            or any(not isinstance(p, str) for p in target["supported_parameters"])
        ):
            raise ValueError("supported_parameters must be a string list")
        if target.get("api_key_env") is not None and (
            not isinstance(target["api_key_env"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target["api_key_env"])
        ):
            raise ValueError("api_key_env must name an environment variable")
        if target["adapter"] == "fixture":
            if target.get("api_key_env") or target.get("provider"):
                raise ValueError("fixture cannot use credentials or a provider route")
            if target.get("fixture_variant", "a") not in (
                "a",
                "b",
                "sampler",
                "sampler_jitter",
            ):
                raise ValueError("unknown fixture variant")
        else:
            if target["adapter"] == "openrouter":
                target.setdefault("base_url", "https://openrouter.ai/api/v1")
                if (
                    not isinstance(target.get("provider"), str)
                    or not target["provider"]
                ):
                    raise ValueError(
                        "OpenRouter requires an explicit pinned provider slug"
                    )
            elif target.get("provider") is not None:
                raise ValueError(
                    "provider routing is supported only by the OpenRouter adapter"
                )
            if not isinstance(target.get("base_url"), str):
                raise ValueError("base_url required")
            url = urlsplit(target["base_url"])
            if (
                not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "invalid base_url; credentials/query/fragment forbidden"
                )
            if url.scheme != "https" and not (
                url.scheme == "http"
                and (
                    url.hostname in ("localhost", "127.0.0.1", "::1")
                    or target.get("allow_insecure_http") is True
                )
            ):
                raise ValueError("HTTPS required except explicit local/insecure HTTP")
        if "pricing" in target:
            _keys(
                target["pricing"],
                {"input_per_million", "output_per_million"},
                "pricing",
            )
            for v in target["pricing"].values():
                if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                    raise ValueError("invalid pricing")
    probes = []
    seen = set()
    for raw_probe in config["probes"]:
        _keys(
            raw_probe,
            {"id", "recipe", "messages", "max_tokens", "stream", "parameters"},
            "probe",
        )
        probe = expand_probe(raw_probe)
        if (
            not isinstance(probe.get("id"), str)
            or not probe["id"]
            or probe["id"] in seen
        ):
            raise ValueError("unique probe ID required")
        seen.add(probe["id"])
        probe.setdefault("max_tokens", 64)
        _positive(probe["max_tokens"], "max_tokens")
        probe.setdefault("stream", True)
        if type(probe["stream"]) is not bool:
            raise ValueError("stream must be boolean")
        messages = probe.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages required")
        for message in messages:
            _keys(message, {"role", "content"}, "message")
            if message.get("role") not in (
                "system",
                "user",
                "assistant",
            ) or not isinstance(message.get("content"), str):
                raise ValueError("text system/user/assistant messages required")
        probe.setdefault("parameters", {})
        _parameters(probe["parameters"])
        probes.append(probe)
    config["probes"] = probes
    for target in config["targets"]:
        for probe in probes:
            effective_parameters(target, probe)
    limits = config.setdefault("limits", {})
    _keys(
        limits,
        {
            "max_requests",
            "max_output_tokens",
            "max_cost_usd",
            "unknown_cost",
            "deadline_s",
            "max_response_bytes",
            "min_interval_s",
            "retry_http_statuses",
            "max_retries",
            "max_total_retries",
            "retry_backoff_s",
            "max_failed_requests",
        },
        "limits",
    )
    for key, default in (
        ("max_requests", 100),
        ("max_output_tokens", 20000),
        ("max_response_bytes", 2_000_000),
    ):
        limits.setdefault(key, default)
        _positive(limits[key], key, 100_000_000)
    limits.setdefault("max_cost_usd", 0.05)
    limits.setdefault("deadline_s", 45.0)
    limits.setdefault("unknown_cost", "stop")
    for key in ("max_cost_usd", "deadline_s"):
        if (
            type(limits[key]) not in (int, float)
            or not math.isfinite(limits[key])
            or limits[key] <= 0
        ):
            raise ValueError(f"invalid {key}")
    limits.setdefault("min_interval_s", 0.0)
    limits.setdefault("retry_http_statuses", [])
    limits.setdefault("max_retries", 0)
    limits.setdefault("max_total_retries", 0)
    limits.setdefault("retry_backoff_s", 5.0)
    for key in ("min_interval_s", "retry_backoff_s"):
        if type(limits[key]) not in (int, float) or not 0 <= limits[key] <= 600:
            raise ValueError(f"invalid {key}")
    # Failed requests (after retries) normally stop the run; this many may be recorded and skipped.
    limits.setdefault("max_failed_requests", 0)
    for key, maximum in (
        ("max_retries", 10),
        ("max_total_retries", 100_000),
        ("max_failed_requests", 100_000),
    ):
        if type(limits[key]) is not int or not 0 <= limits[key] <= maximum:
            raise ValueError(f"invalid {key}")
    if not isinstance(limits["retry_http_statuses"], list) or any(
        type(v) is not int or v not in (408, 429, 500, 502, 503, 504)
        for v in limits["retry_http_statuses"]
    ):
        raise ValueError("retry_http_statuses must list transient HTTP statuses")
    if limits["max_failed_requests"] and limits["unknown_cost"] != "continue":
        raise ValueError("max_failed_requests requires unknown_cost 'continue'")
    if limits["unknown_cost"] not in ("stop", "continue"):
        raise ValueError("unknown_cost must be stop or continue")
    requests = len(config["targets"]) * len(probes) * config["repeats"]
    tokens = planned_output_tokens(config)
    if requests > limits["max_requests"] or tokens > limits["max_output_tokens"]:
        raise ValueError("planned campaign exceeds request/output-token limit")
    evidence = config.setdefault("evidence", {})
    _keys(evidence, {"catalog_snapshot", "attachments"}, "evidence")
    evidence.setdefault("catalog_snapshot", False)
    evidence.setdefault("attachments", [])
    if type(evidence["catalog_snapshot"]) is not bool:
        raise ValueError("catalog_snapshot must be boolean")
    names = set()
    for attachment in evidence["attachments"]:
        _keys(attachment, {"path", "role"}, "attachment")
        if not all(isinstance(attachment.get(k), str) and attachment[k] for k in ("path", "role")):
            raise ValueError("attachment path and role required")
        name = Path(attachment["path"]).name
        if name in names:
            raise ValueError("attachment file names must be unique")
        names.add(name)
    analysis = config.setdefault("analysis", {})
    _keys(
        analysis,
        {
            "reference_target",
            "tv_budgets",
            "logprob_error",
            "sampling_alpha",
            "sampling_permutations",
        },
        "analysis",
    )
    if analysis.get("reference_target") is not None and analysis[
        "reference_target"
    ] not in {t["id"] for t in config["targets"]}:
        raise ValueError("unknown reference target")
    analysis.setdefault("tv_budgets", [0.01, 0.05, 0.1])
    analysis.setdefault("logprob_error", 0.01)
    if (
        not isinstance(analysis["tv_budgets"], list)
        or not analysis["tv_budgets"]
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in analysis["tv_budgets"]
        )
    ):
        raise ValueError("invalid TV budgets")
    analysis.setdefault("sampling_alpha", 0.001)
    analysis.setdefault("sampling_permutations", 2000)
    if (
        type(analysis["sampling_alpha"]) not in (int, float)
        or not 0 < analysis["sampling_alpha"] < 1
    ):
        raise ValueError("invalid sampling_alpha")
    _positive(analysis["sampling_permutations"], "sampling_permutations", 100_000)
    if (analysis["sampling_permutations"] + 1) * analysis["sampling_alpha"] < 1:
        raise ValueError("sampling_permutations too few to reach sampling_alpha")
    v = analysis["logprob_error"]
    if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
        raise ValueError("invalid logprob_error")
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def planned_output_tokens(config):
    """Output-token cap over the plan; each of n choices may use max_tokens."""
    return config["repeats"] * sum(
        p["max_tokens"] * effective_parameters(t, p).get("n", 1)
        for t in config["targets"]
        for p in config["probes"]
    )


def effective_parameters(target, probe):
    result = {"temperature": 1.0, "top_p": 1.0}
    result.update(target["parameters"])
    result.update(probe["parameters"])
    _parameters(result)
    return result
