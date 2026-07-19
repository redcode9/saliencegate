from __future__ import annotations

import argparse
import asyncio
import builtins
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

import saliencegate.cli as cli_module
import saliencegate.commands.shadow as shadow_module
from saliencegate.cli import ExitCode
from saliencegate.commands.shadow import (
    ATIFShadowCommandReport,
    ShadowCommandConfigurationError,
    ShadowCommandInputError,
    ShadowCommandIntegrityError,
    ShadowCommandReport,
    render_shadow_atif_json,
    render_shadow_json,
    run_shadow_analyze_atif,
)
from saliencegate.security import (
    InstallationKey,
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
    StableFileAuthorization,
    inspect_private_file_location,
)
from saliencegate.shadow import (
    ATIFProfile,
    ShadowEnvironmentBinding,
    ShadowInvariantError,
    ShadowTraceReport,
    analyze_atif_bytes,
    encode_shadow_trace_report,
)
from saliencegate.shadow.io import ShadowTraceReportBinding

RUN_ID = UUID("55555555-5555-4555-8555-555555555555")
UUID_V1 = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
ENVIRONMENT_DIGEST = "e" * 64
WORKING_DIRECTORY = "/synthetic/coverage"
KEY = InstallationKey(b"c" * 32)
FIXTURE = Path("tests/fixtures/shadow/atif/codex-bundled-synthetic.trajectory.json")


def _private_source(tmp_path: Path) -> Path:
    source = tmp_path / "codex.trajectory.json"
    source.write_bytes(FIXTURE.read_bytes())
    source.chmod(0o600)
    return source


async def _analyze(
    source: str | Path,
    output: str | Path,
    *,
    profile: ATIFProfile = ATIFProfile.HARBOR_CODEX_V1,
    run_id: UUID = RUN_ID,
    working_directory: str = WORKING_DIRECTORY,
    environment_digest: str = ENVIRONMENT_DIGEST,
    repository_path: str | Path = ":memory:",
    replace: bool = False,
) -> ATIFShadowCommandReport:
    return await run_shadow_analyze_atif(
        source,
        profile=profile,
        run_id=run_id,
        working_directory=working_directory,
        environment_digest=environment_digest,
        output_path=output,
        repository_path=repository_path,
        replace=replace,
    )


@pytest.fixture(scope="module")
def valid_trace_report() -> ShadowTraceReport:
    return asyncio.run(
        analyze_atif_bytes(
            FIXTURE.read_bytes(),
            run_id=RUN_ID,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            environment=ShadowEnvironmentBinding(
                default_working_directory=WORKING_DIRECTORY,
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            installation_key=KEY,
        )
    )


@pytest.fixture(scope="module")
def valid_summary(valid_trace_report: ShadowTraceReport) -> ATIFShadowCommandReport:
    return shadow_module._atif_command_report(valid_trace_report)


def _report_binding(report: ShadowTraceReport) -> ShadowTraceReportBinding:
    return ShadowTraceReportBinding(
        run_id=report.run_id,
        trace_binding=report.binding,
        diagnostics_digest=report.diagnostics_digest,
        mapped_record_digest=report.mapped_record_digest,
        normalized_input_digest=report.normalized_input_digest,
        redaction_policy_tag=report.shadow_report.redaction_policy_tag,
        detector_profile_digest=report.shadow_report.detector_profile_digest,
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("schema_version", 1),
        ("profile_audit_manifest_digest", "A" * 64),
        ("root_segment_only", 1),
        ("tool_call_disposition_counts", []),
        ("structured_outcome_coverage", (0,)),
        ("supported_signal_types", list(shadow_module._SUPPORTED_SIGNAL_TYPES)),
    ),
)
def test_atif_renderer_rejects_forged_non_exact_field_shapes(
    valid_summary: ATIFShadowCommandReport,
    field: str,
    invalid: object,
) -> None:
    forged = valid_summary.model_copy(update={field: invalid})

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        render_shadow_atif_json(forged)


@pytest.mark.parametrize("inconsistency", ("contract", "detector", "abstention"))
def test_atif_renderer_rejects_forged_cross_field_inconsistency(
    valid_summary: ATIFShadowCommandReport,
    inconsistency: str,
) -> None:
    if inconsistency == "contract":
        update: dict[str, object] = {
            "mapped_shadow_record_count": valid_summary.mapped_shadow_record_count + 1
        }
    elif inconsistency == "detector":
        counts = list(valid_summary.detector_outcome_counts)
        signal_type, status, count = counts[0]
        counts[0] = (signal_type, status, count + 1)
        update = {"detector_outcome_counts": tuple(counts)}
    else:
        counts = list(valid_summary.abstention_reason_counts)
        signal_type, reason, count = counts[0]
        counts[0] = (signal_type, reason, count + 1)
        update = {"abstention_reason_counts": tuple(counts)}
    forged = valid_summary.model_copy(update=update)

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        render_shadow_atif_json(forged)


@pytest.mark.parametrize(
    ("profile", "run_id", "replace"),
    (
        (cast(ATIFProfile, "harbor-codex/v1"), RUN_ID, False),
        (ATIFProfile.HARBOR_CODEX_V1, UUID_V1, False),
        (ATIFProfile.HARBOR_CODEX_V1, RUN_ID, cast(bool, 1)),
    ),
)
@pytest.mark.asyncio
async def test_atif_public_boundary_rejects_non_exact_service_arguments_before_io(
    profile: ATIFProfile,
    run_id: UUID,
    replace: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbid_key_lookup() -> InstallationKey:
        raise AssertionError("invalid service arguments reached key lookup")

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", forbid_key_lookup)

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        await _analyze(
            tmp_path / "missing.json",
            tmp_path / "report.json",
            profile=profile,
            run_id=run_id,
            replace=replace,
        )


@pytest.mark.asyncio
async def test_atif_public_boundary_rejects_environment_before_source_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbid_key_lookup() -> InstallationKey:
        raise AssertionError("invalid environment reached key lookup")

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", forbid_key_lookup)

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        await _analyze(
            tmp_path / "missing.json",
            tmp_path / "report.json",
            environment_digest="not-a-digest",
        )


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileBoundError(), ShadowCommandInputError),
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
    ),
)
@pytest.mark.asyncio
async def test_atif_source_read_has_stable_error_taxonomy_before_key_lookup(
    failure: Exception,
    expected: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_read(*_args: object, **_kwargs: object) -> object:
        raise failure

    def forbid_key_lookup() -> InstallationKey:
        raise AssertionError("failed source read reached key lookup")

    monkeypatch.setattr(shadow_module, "read_stable_file", fail_read)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", forbid_key_lookup)

    with pytest.raises(expected):
        await _analyze(tmp_path / "trace.json", tmp_path / "report.json")


@pytest.mark.parametrize("loader_mode", ("raises", "returns-none"))
@pytest.mark.asyncio
async def test_atif_key_lookup_failure_is_a_configuration_error_without_publication(
    loader_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_source(tmp_path)
    output = tmp_path / "report.json"

    def load_key() -> InstallationKey | None:
        if loader_mode == "raises":
            raise SecureFileError()
        return None

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", load_key)

    with pytest.raises(ShadowCommandConfigurationError, match="configuration is invalid"):
        await _analyze(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
@pytest.mark.asyncio
async def test_atif_output_location_inspection_fails_closed_without_publication(
    failure: Exception,
    expected: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_source(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    def fail_inspection(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(shadow_module, "inspect_private_file_location", fail_inspection)

    with pytest.raises(expected):
        await _analyze(source, output)

    assert not output.exists()


@pytest.mark.asyncio
async def test_atif_location_alias_comparison_failure_is_an_input_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_source(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    def fail_alias_check(_authorizations: object) -> bool:
        raise SecureFileError()

    monkeypatch.setattr(shadow_module, "_locations_alias", fail_alias_check)

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        await _analyze(source, output)

    assert not output.exists()


def test_absent_output_is_valid_for_the_replacement_precheck(tmp_path: Path) -> None:
    location = inspect_private_file_location(tmp_path / "absent-report.json")

    assert shadow_module._existing_output_is_valid(location, cast(object, object())) is True


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileBoundError(), ShadowCommandIntegrityError),
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
def test_atif_replacement_classifies_existing_report_read_failures(
    valid_trace_report: ShadowTraceReport,
    failure: Exception,
    expected: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"existing-private-report")
    output.chmod(0o600)
    location = inspect_private_file_location(output)

    def fail_read(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(shadow_module, "read_stable_file", fail_read)

    with pytest.raises(expected):
        shadow_module._authorize_atif_output(
            location,
            binding=_report_binding(valid_trace_report),
            replace=True,
        )

    assert output.read_bytes() == b"existing-private-report"


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
def test_atif_publication_authorization_has_stable_error_taxonomy(
    valid_trace_report: ShadowTraceReport,
    failure: Exception,
    expected: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    location = inspect_private_file_location(output)

    def fail_authorization(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(
        shadow_module,
        "authorize_shadow_trace_report_publication",
        fail_authorization,
    )

    with pytest.raises(expected):
        shadow_module._authorize_atif_output(
            location,
            binding=_report_binding(valid_trace_report),
            replace=False,
        )

    assert not output.exists()


class _ExplodingRead:
    @property
    def data(self) -> bytes:
        raise SecureFileError()


class _PublicationProbe:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def publish(
        self,
        data: bytes,
        *,
        validate_published: Callable[[bytes], bool],
    ) -> object:
        if self.mode == "unsupported":
            raise SecureFileUnsupportedError()
        if self.mode == "failed-before-callback":
            raise SecureFileError()
        if self.mode == "invalid-callback":
            assert validate_published(b"invalid-report") is False
            raise SecureFileError()
        if self.mode == "broken-reopened-read":
            assert validate_published(data) is True
            return _ExplodingRead()
        if self.mode == "callback-not-invoked":
            return SimpleNamespace(data=data)
        raise AssertionError("unknown publication probe mode")


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("unsupported", ShadowCommandConfigurationError),
        ("failed-before-callback", ShadowCommandInputError),
        ("invalid-callback", ShadowCommandIntegrityError),
        ("broken-reopened-read", ShadowCommandIntegrityError),
        ("callback-not-invoked", ShadowCommandIntegrityError),
    ),
)
def test_atif_publication_enforces_callback_and_reopened_validation(
    valid_trace_report: ShadowTraceReport,
    mode: str,
    expected: type[Exception],
) -> None:
    encoded = encode_shadow_trace_report(valid_trace_report)

    with pytest.raises(expected):
        shadow_module._publish_atif_report(
            cast(shadow_module.AtomicFilePublication, _PublicationProbe(mode)),
            valid_trace_report,
            encoded,
        )


class _AuthorizationProbe:
    def __init__(
        self,
        name: str,
        *,
        aliases: frozenset[str] = frozenset(),
        alias_error: bool = False,
    ) -> None:
        self.name = name
        self._aliases = aliases
        self._alias_error = alias_error
        self.revalidated = False

    def revalidate(self) -> None:
        self.revalidated = True

    def aliases(self, other: object) -> bool:
        if self._alias_error:
            raise SecureFileError()
        return isinstance(other, _AuthorizationProbe) and other.name in self._aliases


def test_legacy_sqlite_authorization_rejects_publication_location_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _AuthorizationProbe("source")
    output = _AuthorizationProbe("output")
    publication = _AuthorizationProbe("publication")
    sqlite = _AuthorizationProbe("sqlite")
    locations = shadow_module._LocationPlan(
        output=cast(StableFileAuthorization, output),
        sqlite_slots=(),
        sqlite_path="/private/synthetic.sqlite3",
    )
    monkeypatch.setattr(shadow_module, "authorize_private_sqlite_path", lambda _path: sqlite)

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        shadow_module._authorize_sqlite(
            cast(shadow_module.PreflightedShadowTrace, SimpleNamespace(authorization=source)),
            locations,
            cast(
                shadow_module.AtomicFilePublication,
                SimpleNamespace(authorization=publication),
            ),
        )


@pytest.mark.parametrize("conflict", ("publication", "source", "sqlite"))
def test_atif_prepublication_alias_checks_fail_closed(conflict: str) -> None:
    if conflict == "publication":
        source = _AuthorizationProbe("source")
        output = _AuthorizationProbe("output")
    elif conflict == "source":
        source = _AuthorizationProbe("source", aliases=frozenset({"output"}))
        output = _AuthorizationProbe("output", aliases=frozenset({"publication"}))
    else:
        source = _AuthorizationProbe("source", aliases=frozenset({"sqlite"}))
        output = _AuthorizationProbe("output", aliases=frozenset({"publication"}))
    publication = _AuthorizationProbe("publication")
    sqlite = None if conflict != "sqlite" else _AuthorizationProbe("sqlite")

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        shadow_module._revalidate_atif_before_publication(
            source=cast(StableFileAuthorization, source),
            output=cast(StableFileAuthorization, output),
            publication=cast(
                shadow_module.AtomicFilePublication,
                SimpleNamespace(authorization=publication),
            ),
            sqlite=cast(StableFileAuthorization | None, sqlite),
        )


def test_invalid_replacement_preview_bytes_fail_closed() -> None:
    assert (
        shadow_module._trace_report_matches_preview(
            b"not-json",
            cast(shadow_module.ShadowTrace, object()),
            cast(shadow_module.ShadowRunReport, object()),
        )
        is False
    )


def test_trace_session_rejects_a_missing_sqlite_authorization() -> None:
    session = SimpleNamespace(_repository=object())

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        shadow_module._trace_session_sqlite_authorization(
            cast(shadow_module.ShadowSession, session),
            repository_path="/private/synthetic.sqlite3",
        )


@pytest.mark.asyncio
async def test_atif_sqlite_location_is_revalidated_before_session_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_source(tmp_path)
    output = tmp_path / "report.json"
    repository = tmp_path / "shadow.sqlite3"
    original_revalidate = StableFileAuthorization.revalidate
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    def raced_revalidate(authorization: StableFileAuthorization) -> None:
        if authorization.path == os.fspath(output):
            raise SecureFileError()
        original_revalidate(authorization)

    monkeypatch.setattr(StableFileAuthorization, "revalidate", raced_revalidate)

    with pytest.raises(ShadowCommandInputError, match="input or output is invalid"):
        await _analyze(source, output, repository_path=repository)

    assert not output.exists()
    assert not repository.exists()


@pytest.mark.asyncio
async def test_atif_replacement_rejects_a_post_preview_exact_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_source(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    await _analyze(source, output)
    original = output.read_bytes()
    monkeypatch.setattr(shadow_module, "_trace_report_matches_exactly", lambda *_args: False)

    with pytest.raises(ShadowCommandIntegrityError, match="integrity check failed"):
        await _analyze(source, output, replace=True)

    assert output.read_bytes() == original


@pytest.mark.parametrize("case", ("diagnostics", "profile", "invalid-count"))
def test_atif_command_conversion_rejects_forged_outer_reports(
    valid_trace_report: ShadowTraceReport,
    case: str,
) -> None:
    if case == "diagnostics":
        forged = valid_trace_report.model_copy(update={"diagnostics": object()})
    elif case == "profile":
        binding = valid_trace_report.binding.model_copy(
            update={"adapter_profile_id": "unknown-profile/v1"}
        )
        forged = valid_trace_report.model_copy(update={"binding": binding})
    else:
        diagnostics = valid_trace_report.diagnostics.model_copy(update={"total_step_count": -1})
        forged = valid_trace_report.model_copy(update={"diagnostics": diagnostics})

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        shadow_module._atif_command_report(forged)


def test_profile_detector_evidence_rejects_an_unknown_profile() -> None:
    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        shadow_module._profile_detector_evidence("unknown-profile/v1")


@pytest.mark.parametrize("mode", ("invariant", "invalid-result"))
def test_atif_command_conversion_sanitizes_profile_evidence_failures(
    valid_trace_report: ShadowTraceReport,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def profile_evidence(_profile_id: str) -> tuple[object, ...]:
        if mode == "invariant":
            raise ShadowInvariantError()
        return ()

    monkeypatch.setattr(shadow_module, "_profile_detector_evidence", profile_evidence)

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        shadow_module._atif_command_report(valid_trace_report)


@pytest.mark.asyncio
async def test_atif_public_boundary_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel(*_args: object, **_kwargs: object) -> ATIFShadowCommandReport:
        raise asyncio.CancelledError()

    monkeypatch.setattr(shadow_module, "_run_shadow_analyze_atif", cancel)

    with pytest.raises(asyncio.CancelledError):
        await _analyze("unused-source", "unused-output")


@pytest.mark.parametrize(
    ("failure", "expected", "message"),
    (
        (ShadowInvariantError(), ShadowInvariantError, "shadow invariant is invalid"),
        (OSError("secret-os-detail"), ShadowCommandConfigurationError, "configuration is invalid"),
        (
            RuntimeError("secret-runtime-detail"),
            ShadowInvariantError,
            "shadow invariant is invalid",
        ),
    ),
)
@pytest.mark.asyncio
async def test_atif_public_boundary_sanitizes_internal_failure_taxonomy(
    failure: Exception,
    expected: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*_args: object, **_kwargs: object) -> ATIFShadowCommandReport:
        raise failure

    monkeypatch.setattr(shadow_module, "_run_shadow_analyze_atif", fail)

    with pytest.raises(expected, match=message) as captured:
        await _analyze("unused-source", "unused-output")

    assert "secret" not in str(captured.value)


def test_renderers_reject_wrong_summary_type() -> None:
    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        render_shadow_atif_json(cast(ATIFShadowCommandReport, object()))


def test_atif_summary_copy_rejects_validation_drift(
    valid_summary: ATIFShadowCommandReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    different = valid_summary.model_copy(update={"report_digest": "f" * 64})

    def drifted_validate(
        _summary_type: type[ATIFShadowCommandReport],
        _value: object,
    ) -> ATIFShadowCommandReport:
        return different

    monkeypatch.setattr(
        ATIFShadowCommandReport,
        "model_validate",
        classmethod(drifted_validate),
    )

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        render_shadow_atif_json(valid_summary)


def test_legacy_summary_copy_rejects_validation_drift(
    valid_trace_report: ShadowTraceReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = shadow_module._command_report(valid_trace_report.shadow_report)
    different = summary.model_copy(update={"report_digest": "f" * 64})

    def drifted_validate(
        _summary_type: type[ShadowCommandReport],
        _value: object,
    ) -> ShadowCommandReport:
        return different

    monkeypatch.setattr(
        ShadowCommandReport,
        "model_validate",
        classmethod(drifted_validate),
    )

    with pytest.raises(ShadowInvariantError, match="shadow invariant is invalid"):
        render_shadow_json(summary)


def test_pilot_dispatch_success_writes_the_selected_public_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import saliencegate.commands.pilot as pilot_module

    marker = object()

    async def run_pilot(**_kwargs: object) -> object:
        return marker

    monkeypatch.setattr(pilot_module, "run_paper_two_phase_pilot", run_pilot)
    monkeypatch.setattr(
        pilot_module,
        "render_pilot_json",
        lambda report: "pilot-dispatch-ok\n" if report is marker else "wrong-report\n",
    )

    code = cli_module.main(
        (
            "pilot",
            "paper-two-phase",
            "--endpoint",
            "http://127.0.0.1:1/v1",
            "--model",
            "synthetic-model",
            "--output",
            "/private/synthetic-output",
            "--json",
        )
    )

    assert code == ExitCode.SUCCESS
    assert capsys.readouterr() == ("pilot-dispatch-ok\n", "")


def test_pilot_dispatch_does_not_hide_an_unrelated_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "saliencegate.commands.pilot":
            raise ModuleNotFoundError(
                "unrelated dependency is unavailable",
                name="unrelated_dependency",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ModuleNotFoundError) as captured:
        cli_module._dispatch_pilot(argparse.Namespace(pilot_command="paper-two-phase"))

    assert captured.value.name == "unrelated_dependency"


def test_unknown_dispatch_state_is_value_free_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class UnknownCommandParser:
        def parse_args(self, _argv: object) -> argparse.Namespace:
            return argparse.Namespace(command="future-unknown-command")

    monkeypatch.setattr(cli_module, "_parser", UnknownCommandParser)

    assert cli_module.main(()) == ExitCode.INTERNAL_ERROR
    assert capsys.readouterr() == ("", "error: internal error\n")
