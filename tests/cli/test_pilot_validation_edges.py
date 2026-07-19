from __future__ import annotations

import os
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

import saliencegate.commands.pilot as pilot_module
from saliencegate import __version__
from saliencegate.artifacts import (
    AlgorithmHardwareAttestation,
    AlgorithmWarmupPolicy,
    ArtifactClassification,
    ArtifactDestinationError,
    ArtifactExportError,
    RevisionEvidence,
    RevisionSource,
)
from saliencegate.commands.pilot import (
    PaperTwoPhasePilotReport,
    PilotCommandError,
    PilotEvidenceError,
    PilotRuntimeConfigurationError,
    PilotRuntimeDependencies,
    PilotRuntimeUnavailableError,
    render_pilot_human,
    render_pilot_json,
    run_paper_two_phase_pilot,
)
from saliencegate.experiments import Stage2ConditionId
from saliencegate.models.openai_compatible import (
    OpenAICompatibleError,
    OpenAICompatibleErrorCode,
)
from saliencegate.runtime.model_token_counting import (
    DeterministicModelTokenCounter,
    ModelTokenCounterUnavailableError,
)
from saliencegate.security.keys import InstallationKey

ENDPOINT = "http://127.0.0.1:11434/v1"
MODEL = "gpt-oss:20b"
DIGEST = "a" * 64


def _hardware() -> AlgorithmHardwareAttestation:
    return AlgorithmHardwareAttestation(
        model="local-runtime",
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
        commit="b" * 40,
        dirty_worktree=False,
    )


def _dependencies(**changes: object) -> PilotRuntimeDependencies:
    dependencies = PilotRuntimeDependencies(
        hardware_provider=_hardware,
        runtime_extra_available=lambda: True,
        model_token_counter_factory=lambda model: DeterministicModelTokenCounter(
            model_id=model,
            input_token_count=1,
            output_token_count=1,
        ),
        installation_key_factory=lambda: InstallationKey(b"k" * 32),
        monotonic_ns=lambda: 0,
        revision_provider=_revision,
    )
    return replace(dependencies, **changes)


def _assert_public_error(
    error: BaseException,
    expected_message: str,
    *private_values: str,
) -> None:
    rendered = f"{error!s}\n{error!r}"
    assert str(error) == expected_message
    assert all(value not in rendered for value in private_values)
    assert error.__cause__ is None
    assert error.__context__ is None


def _report() -> PaperTwoPhasePilotReport:
    return PaperTwoPhasePilotReport(
        suite_digest=pilot_module.paper_two_phase_pilot_suite_digest(),
        condition=Stage2ConditionId.FIXED_STEP,
        run_id="00000000-0000-4000-8000-000000009000",
        run_digest=DIGEST,
        result_digest=DIGEST,
        manifest_digest=DIGEST,
        overall_content_digest=DIGEST,
        execution_digest=DIGEST,
        runtime_version="0.11.4",
        checkpoint_digest=DIGEST,
        quantization="Q4_K_M",
        warmup_policy=AlgorithmWarmupPolicy.WARM,
        hardware_digest=DIGEST,
        prompt_bundle_digest=DIGEST,
        configuration_digest=DIGEST,
        probe_request_digest=DIGEST,
        probe_provider_input_tokens=1,
        probe_provider_output_tokens=1,
        probe_latency_us=1,
        control_latency_us=1,
        postflight_latency_us=1,
        provider_input_tokens=1,
        provider_output_tokens=1,
        total_provider_input_tokens=2,
        total_provider_output_tokens=2,
        canonical_input_tokens=1,
        canonical_output_tokens=1,
        canonical_token_equivalents=2,
        model_latency_us=1,
        memory_mutations=1,
        grounded_reminders=1,
        valid_silences=1,
        classification=ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
    )


class _NoSocketClient:
    async def __aenter__(self) -> _NoSocketClient:
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None


class _NoSocketRunner:
    def __init__(self, **_kwargs: object) -> None:
        pass

    async def run(self, _trajectory: object) -> object:
        return object()


@pytest.fixture
def isolated_pilot_pipeline(monkeypatch: pytest.MonkeyPatch) -> object:
    execution = object()

    async def probe_runtime(*_args: object, **_kwargs: object) -> object:
        return pilot_module._RuntimeProfile(
            version="0.11.4",
            checkpoint_digest=DIGEST,
            quantization="Q4_K_M",
            probe=pilot_module._ProbeEvidence(
                request_digest=DIGEST,
                provider_input_tokens=1,
                provider_output_tokens=1,
                latency_us=1,
            ),
            control_latency_us=1,
        )

    async def postflight_runtime(*_args: object, **_kwargs: object) -> int:
        return 1

    monkeypatch.setattr(pilot_module, "_probe_runtime", probe_runtime)
    monkeypatch.setattr(pilot_module, "_postflight_runtime", postflight_runtime)
    monkeypatch.setattr(
        pilot_module,
        "OpenAICompatibleClient",
        lambda *_args, **_kwargs: _NoSocketClient(),
    )
    monkeypatch.setattr(pilot_module, "Stage2ExperimentRunner", _NoSocketRunner)
    monkeypatch.setattr(
        pilot_module,
        "_gate_result",
        lambda _result: pilot_module._PilotDiagnosticSummary(
            provider_input_tokens=1,
            provider_output_tokens=1,
            canonical_input_tokens=1,
            canonical_output_tokens=1,
            canonical_token_equivalents=2,
            model_latency_us=1,
            memory_mutations=1,
            grounded_reminders=1,
            valid_silences=1,
        ),
    )
    monkeypatch.setattr(pilot_module, "_execution", lambda *_args, **_kwargs: execution)
    return execution


@pytest.mark.parametrize(
    ("versions", "expected"),
    (
        ({"httpx": "1.0", "openai-harmony": "1.0"}, True),
        ({"httpx": "", "openai-harmony": "1.0"}, False),
        ({"httpx": "1.0", "openai-harmony": ""}, False),
    ),
)
def test_runtime_extra_requires_both_installed_distributions(
    versions: dict[str, str],
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_module.metadata, "version", versions.__getitem__)

    assert pilot_module._runtime_extra_available() is expected


def test_runtime_extra_hides_distribution_lookup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_distribution: str) -> str:
        raise metadata.PackageNotFoundError("private-distribution-name")

    monkeypatch.setattr(pilot_module.metadata, "version", unavailable)

    assert pilot_module._runtime_extra_available() is False


def test_token_counter_factory_passes_the_exact_model_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    observed: list[str] = []

    def build(*, model_id: str) -> object:
        observed.append(model_id)
        return marker

    monkeypatch.setattr(pilot_module, "HarmonyTokenCounter", build)

    assert pilot_module._token_counter(MODEL) is marker
    assert observed == [MODEL]


@pytest.mark.parametrize(
    ("name", "model"),
    ((MODEL, "different:tag"), ("different:tag", MODEL)),
)
def test_visible_model_rejects_discordant_name_and_model_aliases(
    name: str,
    model: str,
) -> None:
    payload = {
        "models": [
            {
                "name": name,
                "model": model,
                "digest": DIGEST,
                "size": 1,
                "details": {"format": "gguf", "quantization_level": "Q4_K_M"},
            }
        ]
    }

    with pytest.raises(PilotRuntimeConfigurationError):
        pilot_module._visible_model(payload, MODEL)


def test_hardware_attestation_uses_coarse_fallbacks_and_checked_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        pilot_module.os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1024}[name],
    )
    monkeypatch.setattr(pilot_module.platform, "machine", lambda: "")
    monkeypatch.setattr(pilot_module.platform, "system", lambda: "")
    monkeypatch.setattr(pilot_module.platform, "release", lambda: "")

    hardware = pilot_module._hardware()

    assert hardware.architecture == "unknown-architecture"
    assert hardware.operating_system == "unknown-system"
    assert hardware.operating_system_version == "unknown-release"
    assert hardware.logical_core_count == 4
    assert hardware.memory_capacity_bytes == 4096 * 1024
    assert hardware.hardware_digest is not None


@pytest.mark.parametrize(
    ("cores", "page_size", "pages"),
    (
        (None, 4096, 1024),
        (True, 4096, 1024),
        (4, 0, 1024),
        (4, True, 1024),
        (4, 4096, 0),
        (4, 4096, True),
        (4, 1 << 62, 4),
    ),
)
async def test_invalid_hardware_facts_fail_before_any_runtime_contact(
    cores: object,
    page_size: object,
    pages: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_module.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(
        pilot_module.os,
        "sysconf",
        lambda name: page_size if name == "SC_PAGE_SIZE" else pages,
    )

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "hardware",
            dependencies=_dependencies(hardware_provider=pilot_module._hardware),
        )

    _assert_public_error(raised.value, "pilot runtime configuration is invalid")


async def test_hardware_probe_failure_is_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-sysconf-value"

    def fail_sysconf(_name: str) -> int:
        raise OSError(secret)

    monkeypatch.setattr(pilot_module.os, "sysconf", fail_sysconf)

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "hardware-error",
            dependencies=_dependencies(hardware_provider=pilot_module._hardware),
        )

    _assert_public_error(
        raised.value,
        "pilot runtime configuration is invalid",
        secret,
    )


@pytest.mark.parametrize("value", ("", b"private-bytes", ".", ".."))
async def test_output_rejects_noncanonical_destinations_before_dependencies(
    value: object,
) -> None:
    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=cast(str, value),
            dependencies=cast(PilotRuntimeDependencies, object()),
        )

    _assert_public_error(
        raised.value,
        "pilot input or output is invalid",
        "private-bytes",
    )


@pytest.mark.parametrize(
    "unsafe_kind",
    ("existing", "file-parent", "symlink", "mode", "access", "owner"),
)
async def test_output_rejects_unsafe_existing_ancestor(
    unsafe_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_name = f"private-{unsafe_kind}"
    if unsafe_kind == "existing":
        output = tmp_path / private_name
        output.mkdir()
    elif unsafe_kind == "file-parent":
        parent = tmp_path / private_name
        parent.write_text("caller-owned", encoding="utf-8")
        output = parent / "pilot"
    elif unsafe_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        parent = tmp_path / private_name
        parent.symlink_to(target, target_is_directory=True)
        output = parent / "pilot"
    else:
        parent = tmp_path / private_name
        parent.mkdir(mode=0o700)
        output = parent / "pilot"
        if unsafe_kind == "mode":
            parent.chmod(0o722)
        elif unsafe_kind == "access":
            monkeypatch.setattr(pilot_module.os, "access", lambda *_args: False)
        elif unsafe_kind == "owner":
            current_uid = os.getuid()
            monkeypatch.setattr(pilot_module.os, "getuid", lambda: current_uid + 1)

    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=cast(PilotRuntimeDependencies, object()),
        )

    _assert_public_error(
        raised.value,
        "pilot input or output is invalid",
        private_name,
    )


def test_output_accepts_a_missing_nested_destination(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "nested" / "pilot"

    assert pilot_module._output_path(output) == output
    assert not output.parent.exists()


@pytest.mark.parametrize("invalid_dependency", (object(), "not-dependencies"))
async def test_dependency_bundle_requires_its_exact_type(
    invalid_dependency: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "dependency",
            dependencies=cast(PilotRuntimeDependencies, invalid_dependency),
        )

    _assert_public_error(
        raised.value,
        "pilot input or output is invalid",
        "not-dependencies",
    )


@pytest.mark.parametrize("field", ("hardware_provider", "revision_provider", "transport_factory"))
async def test_dependency_bundle_requires_callable_hooks(
    field: str,
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(**{field: object()})

    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / field,
            dependencies=dependencies,
        )

    _assert_public_error(raised.value, "pilot input or output is invalid")


@pytest.mark.parametrize("model", ("private-model", cast(str, 20)))
async def test_model_selection_is_exact_and_value_free(model: str, tmp_path: Path) -> None:
    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=model,
            output_path=tmp_path / "model",
        )

    _assert_public_error(raised.value, "pilot input or output is invalid", str(model))


@pytest.mark.parametrize("warmup", ("private-policy", AlgorithmWarmupPolicy.NOT_APPLICABLE))
async def test_warmup_policy_accepts_only_guarded_local_modes(
    warmup: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(PilotCommandError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "warmup",
            warmup=cast(AlgorithmWarmupPolicy, warmup),
        )

    _assert_public_error(
        raised.value,
        "pilot input or output is invalid",
        "private-policy",
    )


@pytest.mark.parametrize("availability", (lambda: 1, lambda: None))
async def test_runtime_availability_requires_the_boolean_true_singleton(
    availability: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(PilotRuntimeUnavailableError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "runtime-extra",
            dependencies=_dependencies(runtime_extra_available=availability),
        )

    _assert_public_error(raised.value, "pilot model runtime is unavailable")


async def test_runtime_availability_hook_failure_is_sanitized(tmp_path: Path) -> None:
    secret = "private-extra-probe"

    def fail() -> bool:
        raise RuntimeError(secret)

    with pytest.raises(PilotRuntimeUnavailableError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "runtime-extra-error",
            dependencies=_dependencies(runtime_extra_available=fail),
        )

    _assert_public_error(raised.value, "pilot model runtime is unavailable", secret)


async def test_unavailable_counter_is_reported_as_an_optional_dependency(
    tmp_path: Path,
) -> None:
    def unavailable(_model: str) -> object:
        raise ModelTokenCounterUnavailableError()

    with pytest.raises(PilotRuntimeUnavailableError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "counter",
            dependencies=_dependencies(model_token_counter_factory=unavailable),
        )

    _assert_public_error(raised.value, "pilot model runtime is unavailable")


@pytest.mark.parametrize(
    "changes",
    (
        {"hardware_provider": lambda: object()},
        {"installation_key_factory": lambda: (_ for _ in ()).throw(RuntimeError("private-key"))},
    ),
)
async def test_malformed_startup_dependencies_map_to_configuration_failure(
    changes: dict[str, object],
    tmp_path: Path,
) -> None:
    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "startup",
            dependencies=_dependencies(**changes),
        )

    _assert_public_error(
        raised.value,
        "pilot runtime configuration is invalid",
        "private-key",
    )


def test_trajectory_digest_guard_rejects_unreviewed_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_module,
        "build_stage2_trajectory",
        lambda *_args, **_kwargs: SimpleNamespace(fixture_digest="f" * 64),
    )

    with pytest.raises(PilotEvidenceError):
        pilot_module.build_paper_two_phase_pilot_trajectory()


def test_repository_identifier_factory_is_deterministic_and_ordered() -> None:
    first_factory = pilot_module._repository_id_factory(DIGEST)
    second_factory = pilot_module._repository_id_factory(DIGEST)

    first = first_factory()
    second = first_factory()

    assert isinstance(first, UUID)
    assert first != second
    assert (first, second) == (second_factory(), second_factory())


def test_diagnostic_and_execution_gates_reject_unvalidated_results() -> None:
    runtime = pilot_module._RuntimeProfile(
        version="0.11.4",
        checkpoint_digest=DIGEST,
        quantization="Q4_K_M",
        probe=pilot_module._ProbeEvidence(
            request_digest=DIGEST,
            provider_input_tokens=1,
            provider_output_tokens=1,
            latency_us=1,
        ),
        control_latency_us=1,
    )

    with pytest.raises(PilotEvidenceError):
        pilot_module._gate_result(object())
    with pytest.raises(PilotEvidenceError):
        pilot_module._execution(
            object(),
            runtime=runtime,
            hardware=_hardware(),
            warmup=AlgorithmWarmupPolicy.WARM,
        )


def test_renderers_reject_non_report_values_without_echoing_them() -> None:
    secret = "private-report-value"

    for renderer in (render_pilot_json, render_pilot_human):
        with pytest.raises(PilotCommandError) as raised:
            renderer(cast(PaperTwoPhasePilotReport, {"secret": secret}))

        assert str(raised.value) == "pilot input or output is invalid"
        assert secret not in f"{raised.value!s}\n{raised.value!r}"


def test_renderers_revalidate_forged_report_instances() -> None:
    values = _report().model_dump(mode="python")
    values["runtime_version"] = {"private": "runtime"}
    forged = PaperTwoPhasePilotReport.model_construct(**values)

    for renderer in (render_pilot_json, render_pilot_human):
        with pytest.raises(PilotCommandError) as raised:
            renderer(forged)

        assert str(raised.value) == "pilot input or output is invalid"
        assert "runtime" not in f"{raised.value!s}\n{raised.value!r}"


@pytest.mark.parametrize(
    "change",
    (
        {"total_provider_input_tokens": 3},
        {"total_provider_output_tokens": 3},
        {"canonical_token_equivalents": 3},
        {"suite_digest": "b" * 64},
        {"run_id": "00000000-0000-4000-8000-000000009001"},
    ),
)
def test_report_rejects_inconsistent_attestation_summary(
    change: dict[str, int | str],
) -> None:
    values = _report().model_dump(mode="python")
    values.update(change)

    with pytest.raises(ValueError):
        PaperTwoPhasePilotReport.model_validate(values)


def test_report_validation_rejects_a_nonidentical_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    changed = report.model_copy(update={"runtime_version": "changed"})

    def replace_report(_cls: object, _payload: object) -> PaperTwoPhasePilotReport:
        return changed

    monkeypatch.setattr(
        PaperTwoPhasePilotReport,
        "model_validate_json",
        classmethod(replace_report),
    )

    with pytest.raises(PilotCommandError):
        render_pilot_json(report)


async def test_public_wrapper_sanitizes_unexpected_internal_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-unexpected-value"

    async def fail(**_kwargs: object) -> PaperTwoPhasePilotReport:
        raise RuntimeError(secret)

    monkeypatch.setattr(pilot_module, "_run_paper_two_phase_pilot", fail)

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "unexpected",
        )

    _assert_public_error(raised.value, "pilot evidence requirements failed", secret)


async def test_revision_provider_failure_cannot_publish_a_report(
    isolated_pilot_pipeline: object,
    tmp_path: Path,
) -> None:
    del isolated_pilot_pipeline
    secret = "private-revision-value"

    def fail_revision() -> RevisionEvidence:
        raise RuntimeError(secret)

    output = tmp_path / "revision"
    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(revision_provider=fail_revision),
        )

    _assert_public_error(raised.value, "pilot evidence requirements failed", secret)
    assert not output.exists()


async def test_client_construction_failure_is_a_configuration_error(
    isolated_pilot_pipeline: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_pilot_pipeline

    def fail_client(*_args: object, **_kwargs: object) -> object:
        raise OpenAICompatibleError(OpenAICompatibleErrorCode.INVALID_REQUEST)

    monkeypatch.setattr(pilot_module, "OpenAICompatibleClient", fail_client)

    with pytest.raises(PilotRuntimeConfigurationError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "client",
            dependencies=_dependencies(),
        )

    _assert_public_error(raised.value, "pilot runtime configuration is invalid")


async def test_unexpected_diagnostic_failure_is_an_evidence_error(
    isolated_pilot_pipeline: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_pilot_pipeline

    class FailingRunner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self, _trajectory: object) -> object:
            raise RuntimeError("private-diagnostic-value")

    monkeypatch.setattr(pilot_module, "Stage2ExperimentRunner", FailingRunner)

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "diagnostic",
            dependencies=_dependencies(),
        )

    _assert_public_error(
        raised.value,
        "pilot evidence requirements failed",
        "private-diagnostic-value",
    )


@pytest.mark.parametrize(
    ("error", "expected_type", "message"),
    (
        (
            ArtifactDestinationError("private-destination"),
            PilotCommandError,
            "pilot input or output is invalid",
        ),
        (
            ArtifactExportError("private-export"),
            PilotEvidenceError,
            "pilot evidence requirements failed",
        ),
        (
            RuntimeError("private-revision-export"),
            PilotEvidenceError,
            "pilot evidence requirements failed",
        ),
    ),
)
async def test_artifact_boundary_failures_are_stable_and_value_free(
    error: Exception,
    expected_type: type[Exception],
    message: str,
    isolated_pilot_pipeline: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_pilot_pipeline

    def fail_export(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(pilot_module, "export_algorithm_artifact", fail_export)
    output = tmp_path / expected_type.__name__

    with pytest.raises(expected_type) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=output,
            dependencies=_dependencies(),
        )

    _assert_public_error(raised.value, message, "private-")
    assert not output.exists()


async def test_loaded_artifact_must_match_the_attested_execution(
    isolated_pilot_pipeline: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(manifest_digest=DIGEST)
    loaded = SimpleNamespace(manifest=SimpleNamespace(execution=object()))
    monkeypatch.setattr(
        pilot_module,
        "export_algorithm_artifact",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        pilot_module,
        "load_validated_algorithm_artifact",
        lambda *_args, **_kwargs: loaded,
    )

    with pytest.raises(PilotEvidenceError) as raised:
        await run_paper_two_phase_pilot(
            endpoint=ENDPOINT,
            model=MODEL,
            output_path=tmp_path / "mismatched",
            dependencies=_dependencies(),
        )

    _assert_public_error(raised.value, "pilot evidence requirements failed")
    assert isolated_pilot_pipeline is not loaded.manifest.execution
