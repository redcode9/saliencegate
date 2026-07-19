from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import cast

import httpx
import pytest

import saliencegate.commands.pilot as pilot_module
from saliencegate.artifacts import AlgorithmWarmupPolicy
from saliencegate.commands.pilot import (
    PilotRuntimeConfigurationError,
    PilotRuntimeDependencies,
)
from saliencegate.domain import canonical_json

ENDPOINT = "http://127.0.0.1:11434/v1"
MODEL = "gpt-oss:20b"
CHECKPOINT_DIGEST = "d" * 64
CONFIGURATION_ERROR = "pilot runtime configuration is invalid"


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


def _json_response(
    request: httpx.Request,
    payload: object,
    *,
    headers: dict[str, str] | None = None,
    status: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status,
        content=canonical_json(payload),
        headers={"Content-Type": "application/json", **(headers or {})},
        request=request,
    )


def _model_record(*, digest: str = CHECKPOINT_DIGEST) -> dict[str, object]:
    return {
        "name": MODEL,
        "model": MODEL,
        "digest": digest,
        "size": 12_345,
        "details": {
            "format": "gguf",
            "quantization_level": "Q4_K_M",
        },
    }


class _RuntimeContract:
    def __init__(self, *, running: bool = True) -> None:
        self.running = running
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/version":
            return _json_response(request, {"version": "0.11.4"})
        if path == "/api/tags":
            return _json_response(request, {"models": [_model_record()]})
        if path == "/api/show":
            return _json_response(
                request,
                {
                    "capabilities": ["completion", "structured_output"],
                    "details": {"format": "gguf", "quantization_level": "Q4_K_M"},
                },
            )
        if path == "/v1/chat/completions":
            wire = json.loads(request.content)
            schema = wire["response_format"]["json_schema"]["schema"]
            assert schema["properties"]["ready"]["const"] is True
            return _json_response(
                request,
                {
                    "model": MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"canary": "sg-strict-7f4c2a91", "ready": True}
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                },
            )
        if path == "/api/ps":
            return _json_response(
                request,
                {"models": [_model_record()] if self.running else []},
            )
        raise AssertionError(path)


def _probe_payload() -> dict[str, object]:
    return {
        "model": MODEL,
        "choices": [
            {"message": {"content": json.dumps({"canary": "sg-strict-7f4c2a91", "ready": True})}}
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    }


def _assert_sanitized(error: BaseException, *secrets: str) -> None:
    rendered = f"{error!s}\n{error!r}"
    assert str(error) == CONFIGURATION_ERROR
    assert all(secret not in rendered for secret in secrets)


def test_runtime_json_parser_accepts_one_bounded_unique_object() -> None:
    assert pilot_module._parse_runtime_json(b'{"items":[1,{"ready":true}]}') == {
        "items": [1, {"ready": True}]
    }


@pytest.mark.parametrize(
    "payload",
    (
        b'{"duplicate":1,"duplicate":2}',
        b'{"constant":NaN}',
        b"[]",
        b"\xff",
        b'{"nested":' + (b"[" * 33) + b"0" + (b"]" * 33) + b"}",
        canonical_json({"items": [0] * 20_001}),
    ),
    ids=("duplicate-key", "non-finite", "non-object", "invalid-utf8", "too-deep", "too-large"),
)
def test_runtime_json_parser_rejects_ambiguous_or_unbounded_payloads(payload: bytes) -> None:
    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        pilot_module._parse_runtime_json(payload)

    _assert_sanitized(raised.value)


def test_elapsed_time_clamps_clock_rollback_and_signed_overflow() -> None:
    assert pilot_module._elapsed_us(lambda: 9, 10) == 0
    assert pilot_module._elapsed_us(lambda: 1 << 80, 0) == (1 << 63) - 1


@pytest.mark.parametrize(
    ("clock", "started"),
    (
        (lambda: -1, 0),
        (lambda: cast(int, "1"), 0),
        (lambda: 1, cast(int, "0")),
        (lambda: (_ for _ in ()).throw(RuntimeError("clock-secret")), 0),
    ),
)
def test_elapsed_time_rejects_invalid_clock_evidence(
    clock: Callable[[], int],
    started: int,
) -> None:
    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        pilot_module._elapsed_us(clock, started)

    _assert_sanitized(raised.value, "clock-secret")


async def test_runtime_request_accepts_chunked_json_and_sets_closed_headers() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            stream=_ChunkStream(b'{"ready":', b"true}"),
            headers={
                "Content-Type": "Application/JSON; charset=utf-8",
                "Content-Encoding": "identity",
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        payload, latency = await pilot_module._runtime_request(
            client,
            "POST",
            f"{ENDPOINT}/probe",
            body=b"{}",
            clock=_Clock(),
        )

    assert payload == {"ready": True}
    assert latency == 1_000
    assert observed[0].headers["accept"] == "application/json"
    assert observed[0].headers["accept-encoding"] == "identity"
    assert observed[0].headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    "case",
    (
        "status",
        "content-type",
        "content-encoding",
        "malformed-length",
        "oversized-length",
        "oversized-body",
        "oversized-stream",
        "invalid-json",
    ),
)
async def test_runtime_request_rejects_untrusted_response_framing(case: str) -> None:
    secret = "runtime-body-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        if case == "status":
            return httpx.Response(
                503,
                text=secret,
                headers={"Content-Type": "application/json"},
                request=request,
            )
        if case == "content-type":
            return httpx.Response(200, text=secret, request=request)
        if case == "content-encoding":
            return httpx.Response(
                200,
                content=b"{}",
                headers={"Content-Type": "application/json", "Content-Encoding": "br"},
                request=request,
            )
        if case in {"malformed-length", "oversized-length"}:
            declared = "1.0" if case == "malformed-length" else str(1024 * 1024 + 1)
            return httpx.Response(
                200,
                content=b"{}",
                headers={"Content-Type": "application/json", "Content-Length": declared},
                request=request,
            )
        if case == "oversized-body":
            response = httpx.Response(
                200,
                content=b"x" * (1024 * 1024 + 1),
                headers={"Content-Type": "application/json"},
                request=request,
            )
            del response.headers["content-length"]
            return response
        if case == "oversized-stream":
            return httpx.Response(
                200,
                stream=_ChunkStream(b"x" * (1024 * 1024 + 1)),
                headers={"Content-Type": "application/json"},
                request=request,
            )
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PilotRuntimeConfigurationError) as raised:
            await pilot_module._runtime_request(
                client,
                "GET",
                f"{ENDPOINT}/probe",
                body=None,
                clock=_Clock(),
            )

    _assert_sanitized(raised.value, secret)


def test_visible_model_selects_exact_model_field_and_skips_unrelated_records() -> None:
    selected = _model_record()
    del selected["name"]
    unrelated = _model_record(digest="e" * 64)
    unrelated["name"] = unrelated["model"] = "other:1b"

    assert pilot_module._visible_model({"models": [unrelated, selected]}, MODEL) == (
        CHECKPOINT_DIGEST,
        "Q4_K_M",
    )


@pytest.mark.parametrize(
    "models",
    (
        None,
        ["not-an-object"],
        [],
        [_model_record(), _model_record()],
        [{**_model_record(), "digest": "not-a-digest"}],
        [{**_model_record(), "size": True}],
        [{**_model_record(), "details": None}],
        [{**_model_record(), "details": {"format": "safetensors", "quantization_level": "Q4"}}],
        [{**_model_record(), "details": {"format": "gguf", "quantization_level": "bad value"}}],
    ),
)
def test_visible_model_rejects_ambiguous_or_incomplete_identity(models: object) -> None:
    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        pilot_module._visible_model({"models": models}, MODEL)

    _assert_sanitized(raised.value)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"capabilities": ["embedding"], "details": {}},
        {
            "capabilities": ["completion", 1],
            "details": {"format": "gguf", "quantization_level": "Q4_K_M"},
        },
        {"capabilities": ["completion"], "details": "not-an-object"},
        {
            "capabilities": ["completion"],
            "details": {"format": "safetensors", "quantization_level": "Q4_K_M"},
        },
        {
            "capabilities": ["completion"],
            "details": {"format": "gguf", "quantization_level": "Q8_0"},
        },
    ),
)
def test_show_contract_rejects_missing_capability_or_identity(payload: dict[str, object]) -> None:
    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        pilot_module._validate_show(payload, quantization="Q4_K_M")

    _assert_sanitized(raised.value)


def test_probe_evidence_accepts_provider_usage_without_redundant_total() -> None:
    payload = _probe_payload()
    usage = cast(dict[str, object], payload["usage"])
    del usage["total_tokens"]

    evidence = pilot_module._probe_evidence(
        payload,
        model=MODEL,
        request_body=b'{"request":"bounded"}',
        latency_us=17,
    )

    assert evidence.provider_input_tokens == 7
    assert evidence.provider_output_tokens == 3
    assert evidence.latency_us == 17


@pytest.mark.parametrize(
    "case",
    (
        "model",
        "choices",
        "choice-item",
        "message",
        "content",
        "duplicate-content",
        "usage",
        "input-type",
        "input-negative",
        "input-overflow",
        "output-type",
        "output-negative",
        "sum-overflow",
        "total-type",
        "total-mismatch",
        "request-body",
    ),
)
def test_probe_evidence_rejects_malformed_schema_or_token_accounting(case: str) -> None:
    payload = _probe_payload()
    request_body = b'{"request":"bounded"}'
    choices = cast(list[object], payload["choices"])
    choice = cast(dict[str, object], choices[0])
    message = cast(dict[str, object], choice["message"])
    usage = cast(dict[str, object], payload["usage"])
    if case == "model":
        payload["model"] = "other-model"
    elif case == "choices":
        payload["choices"] = []
    elif case == "choice-item":
        payload["choices"] = ["not-an-object"]
    elif case == "message":
        choice["message"] = None
    elif case == "content":
        message["content"] = 1
    elif case == "duplicate-content":
        message["content"] = '{"canary":"a","canary":"b","ready":true}'
    elif case == "usage":
        payload["usage"] = None
    elif case == "input-type":
        usage["prompt_tokens"] = True
    elif case == "input-negative":
        usage["prompt_tokens"] = -1
    elif case == "input-overflow":
        usage["prompt_tokens"] = 1 << 63
    elif case == "output-type":
        usage["completion_tokens"] = True
    elif case == "output-negative":
        usage["completion_tokens"] = -1
    elif case == "sum-overflow":
        usage["prompt_tokens"] = (1 << 63) - 1
        usage["completion_tokens"] = 1
    elif case == "total-type":
        usage["total_tokens"] = True
    elif case == "total-mismatch":
        usage["total_tokens"] = 11
    elif case == "request-body":
        request_body = b"[]"
    else:
        raise AssertionError(case)

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        pilot_module._probe_evidence(
            payload,
            model=MODEL,
            request_body=request_body,
            latency_us=17,
        )

    _assert_sanitized(raised.value, "other-model")


def test_running_model_check_requires_well_formed_digests() -> None:
    assert pilot_module._model_is_running({"models": []}, CHECKPOINT_DIGEST) is False
    assert pilot_module._model_is_running({"models": [_model_record()]}, CHECKPOINT_DIGEST) is True
    for models in (None, ["not-an-object"], [{"digest": "invalid"}]):
        with pytest.raises(PilotRuntimeConfigurationError) as raised:
            pilot_module._model_is_running({"models": models}, CHECKPOINT_DIGEST)
        _assert_sanitized(raised.value)


async def test_default_transport_path_completes_preflight_and_postflight_without_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RuntimeContract()
    constructed: list[tuple[int, bool]] = []

    def transport(*, retries: int, trust_env: bool) -> httpx.AsyncBaseTransport:
        constructed.append((retries, trust_env))
        return httpx.MockTransport(runtime)

    monkeypatch.setattr(pilot_module.httpx, "AsyncHTTPTransport", transport)
    dependencies = PilotRuntimeDependencies(transport_factory=None, monotonic_ns=_Clock())

    profile = await pilot_module._probe_runtime_once(
        ENDPOINT,
        MODEL,
        AlgorithmWarmupPolicy.WARM,
        dependencies,
    )
    postflight_latency = await pilot_module._postflight_runtime_once(
        ENDPOINT,
        MODEL,
        profile,
        dependencies,
    )

    assert profile.version == "0.11.4"
    assert profile.checkpoint_digest == CHECKPOINT_DIGEST
    assert profile.quantization == "Q4_K_M"
    assert postflight_latency == 4_000
    assert constructed == [(0, False), (0, False)]
    assert [request.url.path for request in runtime.requests].count("/v1/chat/completions") == 1


async def test_preflight_rejects_a_runtime_that_breaks_warm_residency_policy() -> None:
    runtime = _RuntimeContract(running=False)
    dependencies = PilotRuntimeDependencies(
        transport_factory=lambda: httpx.MockTransport(runtime),
        monotonic_ns=_Clock(),
    )

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await pilot_module._probe_runtime(ENDPOINT, MODEL, AlgorithmWarmupPolicy.WARM, dependencies)

    _assert_sanitized(raised.value)


@pytest.mark.parametrize("factory_fails", (False, True))
async def test_preflight_sanitizes_invalid_transport_construction(factory_fails: bool) -> None:
    def transport() -> httpx.AsyncBaseTransport:
        if factory_fails:
            raise RuntimeError("transport-factory-secret")
        return cast(httpx.AsyncBaseTransport, object())

    dependencies = PilotRuntimeDependencies(transport_factory=transport)

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await pilot_module._probe_runtime_once(
            ENDPOINT,
            MODEL,
            AlgorithmWarmupPolicy.WARM,
            dependencies,
        )

    _assert_sanitized(raised.value, "transport-factory-secret")


async def test_postflight_wrapper_sanitizes_transport_factory_failure() -> None:
    def fail() -> httpx.AsyncBaseTransport:
        raise RuntimeError("postflight-transport-secret")

    expected = pilot_module._RuntimeProfile(
        version="0.11.4",
        checkpoint_digest=CHECKPOINT_DIGEST,
        quantization="Q4_K_M",
        probe=pilot_module._ProbeEvidence(
            request_digest="a" * 64,
            provider_input_tokens=1,
            provider_output_tokens=1,
            latency_us=1,
        ),
        control_latency_us=1,
    )

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await pilot_module._postflight_runtime(
            ENDPOINT,
            MODEL,
            expected,
            PilotRuntimeDependencies(transport_factory=fail),
        )

    _assert_sanitized(raised.value, "postflight-transport-secret")


async def test_postflight_rejects_a_non_transport_dependency() -> None:
    expected = pilot_module._RuntimeProfile(
        version="0.11.4",
        checkpoint_digest=CHECKPOINT_DIGEST,
        quantization="Q4_K_M",
        probe=pilot_module._ProbeEvidence(
            request_digest="a" * 64,
            provider_input_tokens=1,
            provider_output_tokens=1,
            latency_us=1,
        ),
        control_latency_us=1,
    )
    dependencies = PilotRuntimeDependencies(
        transport_factory=lambda: cast(httpx.AsyncBaseTransport, object())
    )

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await pilot_module._postflight_runtime_once(ENDPOINT, MODEL, expected, dependencies)

    _assert_sanitized(raised.value)
