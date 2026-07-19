from __future__ import annotations

import asyncio
import json
import re
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Never, cast

import httpx
import pytest
from tests.cli.conftest import RunCli

import saliencegate.cli as cli_module
import saliencegate.commands.pilot as pilot_module
from saliencegate import __version__
from saliencegate.artifacts import (
    AlgorithmEndpointClassification,
    AlgorithmHardwareAttestation,
    AlgorithmWarmupPolicy,
    ArtifactClassification,
    ArtifactValidationCode,
    ArtifactValidationError,
    RevisionEvidence,
    RevisionSource,
    load_validated_algorithm_artifact,
)
from saliencegate.commands.pilot import (
    PilotCommandError,
    PilotEvidenceError,
    PilotRuntimeConfigurationError,
    PilotRuntimeDependencies,
    PilotRuntimeUnavailableError,
    build_paper_two_phase_pilot_trajectory,
    paper_two_phase_pilot_suite_digest,
    render_pilot_human,
    render_pilot_json,
    run_paper_two_phase_pilot,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.runtime.model_token_counting import (
    DeterministicModelTokenCounter,
    ModelTokenCounter,
)
from saliencegate.security.keys import InstallationKey

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/fixtures/pilots/stage_2_cases.json"
TRAJECTORY = ROOT / "tests/fixtures/runs/paper_two_phase_basic.jsonl"
RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl"
ENDPOINT = "http://127.0.0.1:11434/v1"
MODEL = "gpt-oss:20b"
RUN_ID = "00000000-0000-4000-8000-000000009000"
TRAJECTORY_DIGEST = "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"
SUITE_DIGEST = "cc369ea90fd34bf91eeaccc4e5254e637be6ca201eb83723a16d82eefd70481b"
CHECKPOINT_DIGEST = "d" * 64
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_FILES = {
    "attestations.json",
    "calls.json",
    "cycles.json",
    "decisions.json",
    "deliveries.json",
    "manifest.json",
    "metrics.json",
    "outcomes.json",
    "run.json",
    "trajectory.json",
}


def _reviewed_outputs() -> tuple[dict[str, object], ...]:
    outputs = tuple(
        cast(dict[str, object], json.loads(line)["result"]["output"])
        for line in RESPONSES.read_bytes().splitlines()
    )
    assert len(outputs) == 6
    return outputs


def _schema_instance(value: object) -> object:
    """Build the smallest deterministic instance of the pilot's probe schema."""

    if type(value) is not dict:
        return {}
    schema = cast(dict[str, object], value)
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if type(enum) is list and enum:
        return enum[0]
    for branch_key in ("oneOf", "anyOf"):
        branches = schema.get(branch_key)
        if type(branches) is list and branches:
            return _schema_instance(branches[0])
    schema_type = schema.get("type")
    if type(schema_type) is list:
        schema_type = next((item for item in schema_type if item != "null"), "null")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if type(properties) is not dict or type(required) is not list:
            return {}
        return {
            key: _schema_instance(properties[key])
            for key in required
            if type(key) is str and key in properties
        }
    if schema_type == "array":
        minimum = schema.get("minItems", 0)
        count = minimum if type(minimum) is int and minimum > 0 else 0
        return [_schema_instance(schema.get("items", {})) for _ in range(count)]
    if schema_type == "boolean":
        return True
    if schema_type == "integer":
        minimum = schema.get("minimum", 0)
        return minimum if type(minimum) is int else 0
    if schema_type == "number":
        minimum = schema.get("minimum", 0.0)
        return float(minimum) if type(minimum) in (int, float) else 0.0
    if schema_type == "null":
        return None
    pattern = schema.get("pattern")
    if type(pattern) is str and "0-9a-f" in pattern and "64" in pattern:
        return "a" * 64
    minimum_length = schema.get("minLength", 1)
    return "probe" if type(minimum_length) is not int else "p" * max(1, minimum_length)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class _CloseFailTransport(httpx.AsyncBaseTransport):
    def __init__(self, script: _RuntimeScript) -> None:
        self._delegate = httpx.MockTransport(script)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._delegate.handle_async_request(request)

    async def aclose(self) -> None:
        raise RuntimeError("transport-close-secret-sentinel")


class _RuntimeScript:
    def __init__(
        self,
        *,
        visible_model: bool = True,
        model_running: bool = True,
        invalid_probe: bool = False,
        follow_visible_probe_instruction: bool = False,
        integer_probe_boolean: bool = False,
        omit_diagnostic_usage: bool = False,
        invalid_diagnostic_at: int | None = None,
        failing_diagnostic_at: int | None = None,
        secret: str = "runtime-secret-sentinel",
        checkpoint_digest: str = CHECKPOINT_DIGEST,
    ) -> None:
        self.visible_model = visible_model
        self.model_running = model_running
        self.invalid_probe = invalid_probe
        self.follow_visible_probe_instruction = follow_visible_probe_instruction
        self.integer_probe_boolean = integer_probe_boolean
        self.omit_diagnostic_usage = omit_diagnostic_usage
        self.invalid_diagnostic_at = invalid_diagnostic_at
        self.failing_diagnostic_at = failing_diagnostic_at
        self.secret = secret
        self.runtime_version = "0.11.4"
        self.quantization = "Q4_K_M"
        self.checkpoint_digest = checkpoint_digest
        self.running_checkpoint_digest = checkpoint_digest
        self.requests: list[httpx.Request] = []
        self.chat_requests: list[httpx.Request] = []
        self.diagnostic_calls = 0
        self.outputs = _reviewed_outputs()

    def _json(self, request: httpx.Request, value: object, *, status: int = 200) -> httpx.Response:
        return httpx.Response(
            status,
            content=canonical_json(value),
            headers={"Content-Type": "application/json"},
            request=request,
        )

    def _probe_completion(self, request: httpx.Request) -> httpx.Response:
        self.model_running = True
        wire = json.loads(request.content)
        response_format = wire["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        instance = _schema_instance(schema)
        if self.follow_visible_probe_instruction:
            instance = {"ready": False}
        if self.integer_probe_boolean:
            assert type(instance) is dict
            instance["ready"] = 1
        content = "not-json" if self.invalid_probe else canonical_json(instance).decode("utf-8")
        return self._json(
            request,
            {
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        )

    def _diagnostic_completion(self, request: httpx.Request) -> httpx.Response:
        self.model_running = True
        ordinal = self.diagnostic_calls
        self.diagnostic_calls += 1
        if ordinal == self.failing_diagnostic_at:
            return httpx.Response(
                500,
                text=self.secret,
                headers={"Content-Type": "text/plain"},
                request=request,
            )
        content = (
            "not-json"
            if ordinal == self.invalid_diagnostic_at
            else canonical_json(self.outputs[ordinal]).decode("utf-8")
        )
        prompt_tokens = 100 + ordinal
        completion_tokens = 10 + ordinal
        payload: dict[str, object] = {
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if not self.omit_diagnostic_usage:
            payload["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return self._json(request, payload)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/version":
            return self._json(request, {"version": self.runtime_version})
        if path in ("/api/ps", "/api/tags"):
            models: list[dict[str, object]] = []
            if self.visible_model and (path == "/api/tags" or self.model_running):
                models.append(
                    {
                        "name": MODEL,
                        "model": MODEL,
                        "digest": (
                            self.checkpoint_digest
                            if path == "/api/tags"
                            else self.running_checkpoint_digest
                        ),
                        "size": 12_345,
                        "details": {
                            "family": "gptoss",
                            "format": "gguf",
                            "parameter_size": "20.9B",
                            "quantization_level": self.quantization,
                        },
                    }
                )
            return self._json(request, {"models": models})
        if path == "/api/generate":
            self.model_running = False
            return self._json(request, {"done": True, "model": MODEL})
        if path == "/v1/models":
            data = [{"id": MODEL, "object": "model"}] if self.visible_model else []
            return self._json(request, {"object": "list", "data": data})
        if path == "/api/show":
            return self._json(
                request,
                {
                    "model": MODEL,
                    "details": {
                        "format": "gguf",
                        "quantization_level": self.quantization,
                    },
                    "capabilities": ["completion", "structured_output"],
                },
            )
        if path == "/v1/chat/completions":
            self.chat_requests.append(request)
            if len(self.chat_requests) == 1:
                return self._probe_completion(request)
            return self._diagnostic_completion(request)
        return self._json(request, {"error": self.secret}, status=404)


def _hardware() -> AlgorithmHardwareAttestation:
    return AlgorithmHardwareAttestation(
        model="deterministic-test-host",
        architecture="test-arm64",
        logical_core_count=8,
        memory_capacity_bytes=16 * 1024**3,
        operating_system="test-os",
        operating_system_version="1.0",
    )


def _revision() -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version=__version__,
        commit="a" * 40,
        dirty_worktree=False,
    )


def _counter(model_id: str) -> ModelTokenCounter:
    return DeterministicModelTokenCounter(
        model_id=model_id,
        input_token_count=37,
        output_token_count=11,
    )


def _dependencies(
    script: _RuntimeScript,
    *,
    runtime_extra_available: Callable[[], bool] = lambda: True,
    model_token_counter_factory: Callable[[str], ModelTokenCounter] = _counter,
) -> PilotRuntimeDependencies:
    clock = _Clock()
    return PilotRuntimeDependencies(
        transport_factory=lambda: httpx.MockTransport(script),
        hardware_provider=_hardware,
        runtime_extra_available=runtime_extra_available,
        model_token_counter_factory=model_token_counter_factory,
        installation_key_factory=lambda: InstallationKey(b"p" * 32),
        monotonic_ns=clock,
        revision_provider=_revision,
    )


def _assert_value_free(error: BaseException, *values: str) -> None:
    rendered = f"{error!s}\n{error!r}"
    assert all(value not in rendered for value in values)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_pilot_fixture_is_canonical_content_addressed_and_matches_production() -> None:
    raw = CASES.read_bytes()
    payload = json.loads(raw)
    digest_payload = dict(payload)
    digest = digest_payload.pop("suite_digest")

    assert raw == canonical_json(payload) + b"\n"
    assert digest == SUITE_DIGEST
    assert digest == length_prefixed_sha256(
        canonical_json(digest_payload),
        domain="saliencegate:pilot:stage2-suite:v1",
    )
    assert paper_two_phase_pilot_suite_digest() == digest
    assert payload["condition"] == "fixed_step"
    assert payload["invocation_boundary_sequences"] == [1, 6, 11]
    assert [case["expected_action"] for case in payload["cases"]] == [
        "remind",
        "silence",
    ]

    trajectory = build_paper_two_phase_pilot_trajectory()
    assert trajectory.fixture_id == "paper-two-phase-basic/v1"
    assert trajectory.fixture_digest == TRAJECTORY_DIGEST
    assert trajectory.run_id.hex == "00000000000040008000000000009000"
    assert len(trajectory.records) == 11
    assert trajectory.canonical_bytes == TRAJECTORY.read_bytes()


async def test_pilot_runs_one_probe_and_six_diagnostic_calls_then_validates_artifact(
    tmp_path: Path,
) -> None:
    script = _RuntimeScript()
    output = tmp_path / "generated-parent" / "pilot"

    report = await run_paper_two_phase_pilot(
        endpoint=ENDPOINT,
        model=MODEL,
        output_path=output,
        dependencies=_dependencies(script),
    )

    payload = json.loads(render_pilot_json(report))
    assert payload["schema_version"] == "cli-paper-two-phase-pilot-report/v1"
    assert payload["status"] == "ok"
    assert payload["pilot"] == "paper-two-phase"
    assert payload["suite_id"] == "paper-two-phase-local-diagnostic/v1"
    assert payload["suite_digest"] == SUITE_DIGEST
    assert payload["condition"] == "fixed_step"
    assert payload["run_id"] == RUN_ID
    assert payload["runtime_id"] == "ollama"
    assert payload["runtime_version"] == "0.11.4"
    assert payload["model_id"] == MODEL
    assert payload["model_tag"] == MODEL
    assert payload["checkpoint_digest"] == CHECKPOINT_DIGEST
    assert payload["quantization"] == "Q4_K_M"
    assert payload["warmup_policy"] == "warm"
    assert payload["probe_calls"] == 1
    assert payload["probe_provider_input_tokens"] == 7
    assert payload["probe_provider_output_tokens"] == 3
    assert payload["probe_latency_us"] == 1_000
    assert SHA256.fullmatch(payload["probe_request_digest"]) is not None
    assert payload["control_latency_us"] == 1_000
    assert payload["postflight_latency_us"] == 4_000
    assert payload["cases"] == 2
    assert payload["cycles"] == 3
    assert payload["calls"] == 6
    assert payload["total_calls"] == 7
    assert payload["provider_input_tokens"] == 615
    assert payload["provider_output_tokens"] == 75
    assert payload["total_provider_input_tokens"] == 622
    assert payload["total_provider_output_tokens"] == 78
    assert payload["canonical_input_tokens"] == 222
    assert payload["canonical_output_tokens"] == 66
    assert payload["canonical_token_equivalents"] == 288
    assert payload["canonical_token_scope"] == "diagnostic_calls_only"
    assert payload["model_latency_us"] == 6_000
    assert payload["schema_repairs"] == 0
    assert payload["schema_invalid_outputs"] == 0
    assert payload["condition_violations"] == 0
    assert payload["memory_mutations"] == 4
    assert payload["grounded_reminders"] == 1
    assert payload["valid_silences"] >= 1
    assert payload["budget_reconciled"] is True
    assert payload["rebuild_equivalent"] is True
    assert payload["artifact_validated"] is True
    assert payload["classification"] == "synthetic_digest_only"
    assert payload["confirmatory"] is False
    assert "efficacy" not in payload
    assert ENDPOINT not in render_pilot_json(report)
    assert payload["hardware_digest"] == _hardware().hardware_digest
    for name in (
        "run_digest",
        "result_digest",
        "manifest_digest",
        "overall_content_digest",
        "execution_digest",
        "prompt_bundle_digest",
        "configuration_digest",
    ):
        assert SHA256.fullmatch(payload[name]) is not None

    assert {item.name for item in output.iterdir()} == ARTIFACT_FILES
    assert {item.name for item in output.parent.iterdir()} == {"pilot", ".pilot.lock"}
    assert stat.S_IMODE((output.parent / ".pilot.lock").stat().st_mode) == 0o600
    loaded = load_validated_algorithm_artifact(
        output / "manifest.json",
        expected_manifest_digest=payload["manifest_digest"],
    )
    assert loaded.manifest.classification is ArtifactClassification.SYNTHETIC_DIGEST_ONLY
    assert loaded.manifest.confirmatory is False
    assert loaded.manifest.execution.endpoint_classification is (
        AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE
    )
    assert loaded.manifest.execution.warmup_policy is AlgorithmWarmupPolicy.WARM
    assert loaded.manifest.execution.checkpoint.checkpoint_digest == CHECKPOINT_DIGEST
    assert loaded.manifest.execution.response_fixture is None
    assert payload["configuration_digest"] == loaded.manifest.configuration_digest
    assert payload["configuration_digest"] != loaded.manifest.condition_digest

    assert [request.url.path for request in script.requests].count("/api/version") == 2
    assert [request.url.path for request in script.requests].count("/api/tags") == 2
    assert [request.url.path for request in script.requests].count("/api/show") == 2
    assert [request.url.path for request in script.requests].count("/api/ps") == 2
    assert len(script.chat_requests) == 7
    assert script.diagnostic_calls == 6
    probe_wire = json.loads(script.chat_requests[0].content)
    assert probe_wire["messages"][1]["content"] == (
        'Return exactly {"ready":false} and no other fields.'
    )
    probe_schema = probe_wire["response_format"]["json_schema"]["schema"]
    assert probe_schema["properties"]["ready"]["const"] is True
    assert probe_schema["properties"]["canary"]["const"] == "sg-strict-7f4c2a91"
    assert all(request.url.host == "127.0.0.1" for request in script.requests)
    assert all("authorization" not in request.headers for request in script.requests)
    for request in script.chat_requests:
        wire = json.loads(request.content)
        assert wire["model"] == MODEL
        assert wire["stream"] is False
        assert wire["temperature"] == 0
        assert wire["response_format"]["type"] == "json_schema"
        assert wire["response_format"]["json_schema"]["strict"] is True


async def test_pilot_records_the_explicit_cold_control_policy(tmp_path: Path) -> None:
    script = _RuntimeScript(model_running=False)
    output = tmp_path / "cold"

    report = await run_paper_two_phase_pilot(
        endpoint=ENDPOINT,
        model=MODEL,
        output_path=output,
        warmup="cold",
        dependencies=_dependencies(script),
    )

    payload = json.loads(render_pilot_json(report))
    assert payload["warmup_policy"] == "cold"
    assert payload["control_latency_us"] == 2_000
    assert payload["postflight_latency_us"] == 4_000
    assert [request.url.path for request in script.requests].count("/api/generate") == 1
    loaded = load_validated_algorithm_artifact(output / "manifest.json")
    assert loaded.manifest.execution.warmup_policy is AlgorithmWarmupPolicy.COLD


async def test_pilot_refuses_missing_runtime_extra_without_transport_or_output(
    tmp_path: Path,
) -> None:
    script = _RuntimeScript()
    output = tmp_path / "missing-extra"

    with pytest.raises(PilotRuntimeUnavailableError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script, runtime_extra_available=lambda: False),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert script.requests == []
    assert not output.exists()


async def test_pilot_refuses_missing_model_after_bounded_preflight(tmp_path: Path) -> None:
    script = _RuntimeScript(visible_model=False)
    output = tmp_path / "model-missing"

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL, script.secret)
    assert len(script.chat_requests) == 0
    assert script.diagnostic_calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    ("checkpoint", "runtime-version", "quantization", "unloaded", "running-digest"),
)
async def test_pilot_refuses_runtime_identity_change_before_publication(
    tmp_path: Path,
    change: str,
) -> None:
    script = _RuntimeScript()
    base = _dependencies(script)
    factory_calls = 0

    def transport_factory() -> httpx.AsyncBaseTransport:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 3:
            if change == "checkpoint":
                script.checkpoint_digest = "e" * 64
            elif change == "runtime-version":
                script.runtime_version = "9.9.9"
            elif change == "quantization":
                script.quantization = "Q8_0"
            elif change == "unloaded":
                script.model_running = False
            elif change == "running-digest":
                script.running_checkpoint_digest = "e" * 64
            else:
                raise AssertionError(change)
        return httpx.MockTransport(script)

    dependencies = PilotRuntimeDependencies(
        transport_factory=transport_factory,
        hardware_provider=base.hardware_provider,
        runtime_extra_available=base.runtime_extra_available,
        model_token_counter_factory=base.model_token_counter_factory,
        installation_key_factory=base.installation_key_factory,
        monotonic_ns=base.monotonic_ns,
        revision_provider=base.revision_provider,
    )
    output = tmp_path / f"changed-{change}"

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=dependencies,
        )

    _assert_value_free(
        raised.value,
        CHECKPOINT_DIGEST,
        "e" * 64,
        "9.9.9",
        "Q8_0",
        ENDPOINT,
        MODEL,
    )
    assert factory_calls == 3
    assert script.diagnostic_calls == 6
    assert not output.exists()
    assert not (output.parent / f".{output.name}.lock").exists()


async def test_pilot_refuses_incompatible_strict_output_before_diagnostic(
    tmp_path: Path,
) -> None:
    script = _RuntimeScript(invalid_probe=True)
    output = tmp_path / "incompatible"

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert len(script.chat_requests) == 1
    assert script.diagnostic_calls == 0
    assert not output.exists()


async def test_pilot_probe_rejects_visible_prompt_following_without_schema_support(
    tmp_path: Path,
) -> None:
    script = _RuntimeScript(follow_visible_probe_instruction=True)
    output = tmp_path / "prompt-following"

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert len(script.chat_requests) == 1
    assert script.diagnostic_calls == 0
    assert not output.exists()


async def test_pilot_probe_rejects_integer_in_place_of_boolean(tmp_path: Path) -> None:
    script = _RuntimeScript(integer_probe_boolean=True)
    output = tmp_path / "integer-boolean"

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert len(script.chat_requests) == 1
    assert script.diagnostic_calls == 0
    assert not output.exists()


async def test_pilot_missing_diagnostic_usage_cannot_publish_artifact(tmp_path: Path) -> None:
    script = _RuntimeScript(omit_diagnostic_usage=True)
    output = tmp_path / "missing-diagnostic-usage"

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert script.diagnostic_calls == 6
    assert not output.exists()
    assert not (output.parent / f".{output.name}.lock").exists()


async def test_pilot_partial_case_failure_cannot_publish_evidence(tmp_path: Path) -> None:
    script = _RuntimeScript(invalid_diagnostic_at=4)
    output = tmp_path / "partial"

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, ENDPOINT, MODEL)
    assert script.diagnostic_calls == 5
    assert not output.exists()


async def test_pilot_propagates_artifact_validation_failure_without_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _RuntimeScript()

    def fail_validation(*_args: object, **_kwargs: object) -> Never:
        raise ArtifactValidationError(ArtifactValidationCode.INVALID_MANIFEST)

    monkeypatch.setattr(pilot_module, "load_validated_algorithm_artifact", fail_validation)

    with pytest.raises(ArtifactValidationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "invalid-artifact",
            dependencies=_dependencies(script),
        )

    assert raised.value.code is ArtifactValidationCode.INVALID_MANIFEST
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


async def test_pilot_runtime_failure_never_exposes_response_body(tmp_path: Path) -> None:
    secret = "private-runtime-body-do-not-emit"
    script = _RuntimeScript(failing_diagnostic_at=2, secret=secret)
    output = tmp_path / "secret"

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, secret, ENDPOINT, MODEL)
    assert not output.exists()


async def test_pilot_transport_close_failure_is_sanitized(tmp_path: Path) -> None:
    script = _RuntimeScript()
    base = _dependencies(script)
    dependencies = PilotRuntimeDependencies(
        transport_factory=lambda: _CloseFailTransport(script),
        hardware_provider=base.hardware_provider,
        runtime_extra_available=base.runtime_extra_available,
        model_token_counter_factory=base.model_token_counter_factory,
        installation_key_factory=base.installation_key_factory,
        monotonic_ns=base.monotonic_ns,
        revision_provider=base.revision_provider,
    )

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "close-failure",
            dependencies=dependencies,
        )

    _assert_value_free(raised.value, "transport-close-secret-sentinel", ENDPOINT, MODEL)


async def test_pilot_diagnostic_transport_close_failure_is_sanitized(tmp_path: Path) -> None:
    script = _RuntimeScript()
    base = _dependencies(script)
    factory_calls = 0

    def transport_factory() -> httpx.AsyncBaseTransport:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return httpx.MockTransport(script)
        return _CloseFailTransport(script)

    dependencies = PilotRuntimeDependencies(
        transport_factory=transport_factory,
        hardware_provider=base.hardware_provider,
        runtime_extra_available=base.runtime_extra_available,
        model_token_counter_factory=base.model_token_counter_factory,
        installation_key_factory=base.installation_key_factory,
        monotonic_ns=base.monotonic_ns,
        revision_provider=base.revision_provider,
    )

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "diagnostic-close-failure",
            dependencies=dependencies,
        )

    _assert_value_free(raised.value, "transport-close-secret-sentinel", ENDPOINT, MODEL)
    assert factory_calls == 2


async def test_pilot_refuses_existing_output_without_contacting_runtime(tmp_path: Path) -> None:
    script = _RuntimeScript()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-caller"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, str(output), ENDPOINT, MODEL)
    assert script.requests == []
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://localhost:11434/v1",
        "http://10.0.0.1:11434/v1",
        "http://user:credential@127.0.0.1:11434/v1",
    ),
)
async def test_pilot_refuses_ambiguous_remote_or_credential_endpoints(
    endpoint: str,
    tmp_path: Path,
) -> None:
    script = _RuntimeScript()

    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=endpoint,
            model=MODEL,
            output_path=tmp_path / "refused",
            dependencies=_dependencies(script),
        )

    _assert_value_free(raised.value, endpoint, "credential")
    assert script.requests == []


def test_pilot_renderers_are_exact_and_value_minimized(tmp_path: Path) -> None:
    script = _RuntimeScript()
    report = asyncio.run(
        run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "render",
            dependencies=_dependencies(script),
        )
    )
    payload = json.loads(render_pilot_json(report))

    assert render_pilot_json(report) == canonical_json(report).decode("utf-8") + "\n"
    assert render_pilot_human(report) == (
        "Local paper-two-phase pilot complete\n"
        "condition: fixed_step\n"
        "diagnostic cases: 2\n"
        "diagnostic calls: 6\n"
        "memory mutations: 4\n"
        "grounded reminders: 1\n"
        f"valid silences: {payload['valid_silences']}\n"
        "canonical token equivalents: 288\n"
        f"artifact digest: {payload['manifest_digest']}\n"
        "classification: exploratory diagnostic (never confirmatory)\n"
    )


def test_pilot_cli_contract_has_no_replace_or_credential_escape_hatch(
    run_cli: RunCli,
) -> None:
    pilot = run_cli("pilot", "--help")
    command = run_cli("pilot", "paper-two-phase", "--help")

    assert pilot.returncode == command.returncode == 0
    assert pilot.stderr == command.stderr == ""
    assert "{paper-two-phase}" in pilot.stdout
    assert "--endpoint ENDPOINT" in command.stdout
    assert "--model MODEL" in command.stdout
    assert "--output OUTPUT" in command.stdout
    assert "--warmup {warm,cold}" in command.stdout
    assert "--json" in command.stdout
    assert "--replace" not in command.stdout
    assert "credential" not in command.stdout.lower()

    for forbidden in ("--replace", "--api-key", "--credential"):
        completed = run_cli(
            "pilot",
            "paper-two-phase",
            "--endpoint",
            ENDPOINT,
            "--model",
            MODEL,
            "--output",
            "unused",
            forbidden,
            "private-cli-value",
        )
        assert completed.returncode == cli_module.ExitCode.INVALID_INPUT
        assert completed.stdout == ""
        assert completed.stderr == "error: invalid command line\n"
        assert "private-cli-value" not in completed.stderr


def test_core_cli_stays_lazy_when_model_runtime_extra_is_missing(tmp_path: Path) -> None:
    probe = f"""
import importlib.abc
import sys

class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname in {{'httpx', 'openai_harmony'}}:
            raise ModuleNotFoundError('optional runtime blocked', name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalRuntime())
from saliencegate.cli import ExitCode, main
assert 'saliencegate.commands.pilot' not in sys.modules
assert main(['--help']) == ExitCode.SUCCESS
assert 'saliencegate.commands.pilot' not in sys.modules
code = main([
    'pilot', 'paper-two-phase',
    '--endpoint', 'http://127.0.0.1:11434/v1',
    '--model', 'gpt-oss:20b',
    '--output', {str(tmp_path / "unavailable")!r},
    '--json',
])
assert code == ExitCode.UNAVAILABLE_DEPENDENCY
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-c", probe),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage: saliencegate" in completed.stdout
    assert completed.stderr == "error: pilot model runtime is unavailable\n"


@pytest.mark.parametrize(
    ("error_type", "expected_code", "expected_error"),
    (
        (PilotCommandError, cli_module.ExitCode.INVALID_INPUT, "pilot input or output is invalid"),
        (
            PilotRuntimeConfigurationError,
            cli_module.ExitCode.CONFIGURATION,
            "pilot runtime configuration is invalid",
        ),
        (
            PilotRuntimeUnavailableError,
            cli_module.ExitCode.UNAVAILABLE_DEPENDENCY,
            "pilot model runtime is unavailable",
        ),
        (
            PilotEvidenceError,
            cli_module.ExitCode.CORRUPTED_ARTIFACT,
            "pilot evidence requirements failed",
        ),
    ),
)
def test_pilot_cli_maps_failures_to_stable_value_free_exits(
    error_type: type[Exception],
    expected_code: int,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(**_kwargs: object) -> Never:
        raise error_type()

    monkeypatch.setattr(pilot_module, "run_paper_two_phase_pilot", fail)

    code = cli_module.main(
        (
            "pilot",
            "paper-two-phase",
            "--endpoint",
            ENDPOINT,
            "--model",
            MODEL,
            "--output",
            str(tmp_path / "unused"),
            "--json",
        )
    )

    captured = capsys.readouterr()
    assert code == expected_code
    assert captured.out == ""
    assert captured.err == f"error: {expected_error}\n"
