"""Extensible collectors. Built-ins: chat-completions HTTP, OpenRouter, fixture."""

import asyncio
import json
import math
import re
import time
from .config import effective_parameters


# Values kept only for these headers; all other headers keep their names only.
SAFE_RESPONSE_HEADERS = frozenset({
    "age", "alt-svc", "cf-cache-status", "cf-ray", "content-type", "date",
    "openai-model", "openai-processing-ms", "request-id", "server", "via",
    "x-request-id", "x-cache", "x-cache-status", "x-provider", "x-backend",
    "x-generation-id", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
})


def reject_constant(value):
    raise ValueError("nonfinite JSON number")


def redact(data, secrets):
    """Exact supplied byte patterns and truncated suffix prefixes only."""
    for secret in sorted((s.encode() for s in secrets if s), key=len, reverse=True):
        data = data.replace(secret, b"[REDACTED]")
        for n in range(min(len(secret) - 1, len(data)), 0, -1):
            if data.endswith(secret[:n]):
                data = data[:-n] + b"[REDACTED-PREFIX]"
                break
    return data


def merge_usage(previous, update):
    result = dict(previous)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_usage(result[key], value)
        else:
            result[key] = value
    return result


class SSEDecoder:
    """Byte-incremental SSE framing, including UTF-8 split across body chunks."""

    def __init__(self):
        self.buffer = b""

    @staticmethod
    def decode(frame):
        fields = []
        for line in re.split(rb"\r\n|\r|\n", frame):
            if line == b"data":
                fields.append(b"")
            elif line.startswith(b"data:"):
                value = line[5:]
                fields.append(value[1:] if value.startswith(b" ") else value)
        return b"\n".join(fields).decode("utf-8") if fields else None

    def feed(self, chunk, final=False):
        self.buffer += chunk
        frames = []
        while True:
            match = re.search(rb"\r\n\r\n|\n\n|\r\r", self.buffer)
            if not match:
                break
            value = self.decode(self.buffer[: match.start()])
            self.buffer = self.buffer[match.end() :]
            if value is not None:
                frames.append(value)
        if final and self.buffer:
            value = self.decode(self.buffer)
            self.buffer = b""
            if value is not None:
                frames.append(value)
        return frames


def request_body(target, probe):
    body = dict(
        model=target["model"],
        messages=probe["messages"],
        max_tokens=probe["max_tokens"],
        stream=probe["stream"],
        **effective_parameters(target, probe),
    )
    if probe["stream"]:
        body["stream_options"] = dict(include_usage=True)
    if target["adapter"] == "openrouter":
        body["provider"] = dict(
            only=[target["provider"]], allow_fallbacks=False, require_parameters=True
        )
    return body


def observation(target, probe, repeat):
    from datetime import datetime, timezone

    return dict(
        schema_version=1,
        target_id=target["id"],
        adapter=target["adapter"],
        model=target["model"],
        started_at=datetime.now(timezone.utc).isoformat(),
        probe_id=probe["id"],
        repeat=repeat,
        request=request_body(target, probe),
        status="pending",
        synthetic=target["adapter"] == "fixture",
        events=[],
        records=[],
        text="",
        choices=[],
        response_ids=[],
        system_fingerprints=[],
        service_tiers=[],
        response_headers={},
        response_header_names=[],
        usage={},
        warnings=[],
        reported_providers=[],
        finish_reasons=[],
        ttft_ms=None,
    )


def consume(row, obj, received_ms):
    if not isinstance(obj, dict):
        raise ValueError("JSON object required")
    if obj.get("error"):
        row["warnings"].append("provider_error")
        row["provider_error"] = obj["error"]
    if isinstance(obj.get("usage"), dict):
        row["usage"] = merge_usage(row["usage"], obj["usage"])
    if (
        isinstance(obj.get("provider"), str)
        and obj["provider"] not in row["reported_providers"]
    ):
        row["reported_providers"].append(obj["provider"])
    if isinstance(obj.get("model"), str):
        row["reported_model"] = obj["model"]
    # Response identity and serving metadata, as declared by the endpoint.
    for key, field in (("id", "response_ids"), ("system_fingerprint", "system_fingerprints"),
                       ("service_tier", "service_tiers")):
        value = obj.get(key)
        if isinstance(value, str) and value and value not in row[field]:
            row[field].append(value)
    choices = obj.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError("choices must be an array")
    requested = row["request"].get("n", 1)
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("choice must be an object")
        index = choice.get("index", 0)
        if type(index) is not int or not 0 <= index < requested:
            raise ValueError("choice index outside requested n")
        entry = next((c for c in row["choices"] if c["index"] == index), None)
        if entry is None:
            entry = dict(index=index, text="", records=[], finish_reasons=[])
            row["choices"].append(entry)
            row["choices"].sort(key=lambda c: c["index"])
        message = choice.get("delta") or choice.get("message") or {}
        text = message.get("content")
        lp = choice.get("logprobs") or {}
        if not isinstance(lp, dict):
            raise ValueError("unsupported logprob shape")
        records = lp.get("content") or []
        if not isinstance(records, list):
            raise ValueError("logprob content must be an array")
        # Top-level text/records remain choice 0 for single-choice analyses.
        targets = (entry, row) if index == 0 else (entry,)
        if isinstance(text, str) and text:
            for item in targets:
                item["text"] += text
            if row["ttft_ms"] is None:
                row["ttft_ms"] = received_ms
        for item in targets:
            item["records"].extend(records)
        if choice.get("finish_reason"):
            for item in targets:
                item["finish_reasons"].append(choice["finish_reason"])


class ChatCompletionsAdapter:
    manages_deadline = True

    async def collect(self, target, probe, repeat, limits, client, secret):
        row = observation(target, probe, repeat)
        started = time.perf_counter()
        body = bytearray()
        chunks = []
        capture = dict(
            complete=False,
            truncated=False,
            body_base64="",
            chunks=chunks,
            scope="Bounded HTTP response body chunks, not TLS packets or physical token timestamps.",
        )
        decoder = SSEDecoder()
        done = False

        def now():
            return 1000 * (time.perf_counter() - started)

        def frames(values):
            nonlocal done
            for data in values:
                if len(row["events"]) >= 10000:
                    raise ValueError("event limit exceeded")
                received = now()
                row["events"].append(dict(received_ms=received, data=data))
                if data == "[DONE]":
                    done = True
                    continue
                consume(row, json.loads(data, parse_constant=reject_constant), received)

        headers = {"content-type": "application/json", "accept-encoding": "identity"}
        if secret:
            headers["authorization"] = "Bearer " + secret
        try:
            async with asyncio.timeout(limits["deadline_s"]):
                url = target["base_url"].rstrip("/") + "/chat/completions"
                async with client.stream(
                    "POST", url, headers=headers, json=row["request"]
                ) as response:
                    row["http_status"] = response.status_code
                    row["ttfb_ms"] = now()
                    row["response_header_names"] = sorted({k.lower() for k in response.headers})
                    row["response_headers"] = {
                        k.lower(): v for k, v in response.headers.items()
                        if k.lower() in SAFE_RESPONSE_HEADERS
                    }
                    content_type = (
                        response.headers.get("content-type", "").split(";")[0].lower()
                    )
                    is_sse = content_type == "text/event-stream"
                    encoding = response.headers.get(
                        "content-encoding", "identity"
                    ).lower()
                    if encoding != "identity":
                        raise ValueError("unsupported response content encoding")
                    async for chunk in response.aiter_raw():
                        if len(chunks) >= 4096:
                            raise ValueError("chunk metadata limit exceeded")
                        take = chunk[: max(0, limits["max_response_bytes"] - len(body))]
                        body.extend(take)
                        chunks.append(
                            dict(
                                received_ms=now(),
                                size=len(chunk),
                                retained_size=len(take),
                            )
                        )
                        if len(take) != len(chunk):
                            capture["truncated"] = True
                            raise ValueError("response byte limit exceeded")
                        if is_sse and 200 <= response.status_code < 300:
                            frames(decoder.feed(chunk))
                    capture["complete"] = True
                    if not 200 <= response.status_code < 300:
                        row["status"] = "http_error"
                    elif is_sse:
                        frames(decoder.feed(b"", final=True))
                        if not done and not row["finish_reasons"]:
                            raise ValueError("missing stream termination")
                        row["status"] = (
                            "success"
                            if not row.get("provider_error")
                            else "provider_error"
                        )
                    else:
                        consume(
                            row,
                            json.loads(bytes(body), parse_constant=reject_constant),
                            None,
                        )
                        if probe["stream"]:
                            row["warnings"].append("requested_stream_returned_json")
                        row["status"] = (
                            "success"
                            if not row.get("provider_error")
                            else "provider_error"
                        )
                    if is_sse != probe["stream"]:
                        row["warnings"].append("response_mode_mismatch")
        except TimeoutError:
            row.update(status="deadline", error="total_request_deadline")
        except Exception as exc:
            # Do not retain arbitrary exception strings that may expose credentials/URLs.
            row.update(status="error", error=type(exc).__name__)
        row["elapsed_ms"] = now()
        capture["_body"] = bytes(body)
        return row, capture


FIXTURE_LOGITS = dict(A=2.0, B=1.4, C=0.9, D=0.1, E=-0.6, F=-1.5)


def fixture_sample(logits, parameters, generator):
    """Declared order: temperature, top-K, then inclusive nucleus; raw reports."""
    ranked = sorted(logits, key=lambda token: -logits[token])
    temperature = parameters.get("temperature", 1.0)
    if temperature == 0:
        return ranked[0]
    k = parameters.get("top_k", 0)
    if k:
        ranked = ranked[:k]
    peak = max(logits[t] for t in ranked)
    weights = [math.exp((logits[t] - peak) / temperature) for t in ranked]
    total = math.fsum(weights)
    keep, mass = [], 0.0
    for token, weight in zip(ranked, weights):
        if mass >= parameters.get("top_p", 1.0):
            break
        keep.append(token)
        mass += weight / total
    return generator.choices(keep, weights[: len(keep)])[0]


def fixture_sampler_payload(target, probe, repeat):
    import random

    request = request_body(target, probe)
    generator = random.Random(f"{target['id']}|{probe['id']}|{repeat}")
    logits = dict(FIXTURE_LOGITS)
    if target.get("fixture_variant") == "sampler_jitter":
        logits = {t: v + generator.gauss(0, 0.7) for t, v in logits.items()}
    peak = max(logits.values())
    normalizer = peak + math.log(math.fsum(math.exp(v - peak) for v in logits.values()))
    raw = {t: v - normalizer for t, v in logits.items()}
    choices = []
    for index in range(request.get("n", 1)):
        token = fixture_sample(logits, request, generator)
        record = dict(token=token, bytes=list(token.encode()), logprob=raw[token])
        if request.get("top_logprobs"):
            record["top_logprobs"] = [
                dict(token=t, bytes=list(t.encode()), logprob=raw[t])
                for t in sorted(raw, key=lambda t: -raw[t])[: request["top_logprobs"]]
            ]
        choices.append(
            dict(
                index=index,
                message=dict(content=token),
                logprobs=dict(content=[record]) if request.get("logprobs") else None,
                finish_reason="length",
            )
        )
    return dict(
        model=target["model"],
        choices=choices,
        usage=dict(prompt_tokens=1, completion_tokens=len(choices), cost=0),
    )


class FixtureAdapter:
    async def collect(self, target, probe, repeat, limits, client, secret):
        row = observation(target, probe, repeat)
        if target.get("fixture_variant") in ("sampler", "sampler_jitter"):
            payload = fixture_sampler_payload(target, probe, repeat)
            consume(row, payload, None)
            row.update(status="success", elapsed_ms=None)
            row["warnings"].append("synthetic_fixture_not_model_inference")
            return row, dict(
                complete=True,
                truncated=False,
                chunks=[],
                _body=json.dumps(payload).encode(),
                scope="Synthetic seeded sampler fixture; no inference, network timing or bill.",
            )
        probabilities = (
            (0.7, 0.3) if target.get("fixture_variant", "a") == "a" else (0.4, 0.6)
        )
        tokens = ("A", "B")
        selected = 0 if probabilities[0] >= probabilities[1] else 1
        record = dict(
            token=tokens[selected],
            bytes=list(tokens[selected].encode()),
            logprob=math.log(probabilities[selected]),
            top_logprobs=[
                dict(token=t, bytes=list(t.encode()), logprob=math.log(p))
                for t, p in zip(tokens, probabilities)
            ],
        )
        payload = dict(
            model=target["model"],
            choices=[
                dict(
                    message=dict(content=tokens[selected]),
                    logprobs=dict(content=[record]),
                    finish_reason="stop",
                )
            ],
            usage=dict(prompt_tokens=1, completion_tokens=1, cost=0),
        )
        consume(row, payload, None)
        row.update(status="success", elapsed_ms=None)
        row["warnings"].append("synthetic_fixture_not_model_inference")
        raw = json.dumps(payload).encode()
        return row, dict(
            complete=True,
            truncated=False,
            chunks=[],
            _body=raw,
            scope="Synthetic deterministic fixture; no inference, network timing or bill.",
        )


ADAPTERS = {
    "chat_completions": ChatCompletionsAdapter,
    "openrouter": ChatCompletionsAdapter,
    "fixture": FixtureAdapter,
}


def register_adapter(name, factory):
    """Register trusted application code, never arbitrary imports from run configuration."""
    if not isinstance(name, str) or not name or name in ADAPTERS:
        raise ValueError("unique adapter name required")
    if not callable(factory):
        raise ValueError("adapter factory required")
    ADAPTERS[name] = factory
