from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import Never
from uuid import UUID

from saliencegate import __version__
from saliencegate.artifacts import ArtifactValidationError
from saliencegate.benchmarks.state_decay.diagnostic import StateDecayDiagnosticError
from saliencegate.commands.algorithm import (
    AlgorithmReplayCommandError,
    render_algorithm_replay_human,
    render_algorithm_replay_json,
    run_algorithm_replay,
)
from saliencegate.commands.capture import (
    CaptureCommandConfigurationError,
    CaptureCommandError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandRequiresDisconnectError,
    CaptureCommandUnavailableError,
)
from saliencegate.commands.demo import (
    render_demo_human,
    render_demo_json,
    run_demo,
)
from saliencegate.commands.doctor import (
    CaptureDoctorReport,
    DoctorCheckName,
    DoctorCheckStatus,
    render_capture_doctor_human,
    render_capture_doctor_json,
    render_doctor_human,
    render_doctor_json,
    run_capture_doctor,
    run_doctor,
)
from saliencegate.commands.replay import (
    ReplayCommandError,
    ReplayConfigurationError,
    render_replay_human,
    render_replay_json,
    run_replay,
)
from saliencegate.commands.shadow import (
    ShadowCommandConfigurationError,
    ShadowCommandInputError,
    ShadowCommandIntegrityError,
    render_shadow_atif_human,
    render_shadow_atif_json,
    render_shadow_human,
    render_shadow_json,
    run_shadow_analyze,
    run_shadow_analyze_atif,
)
from saliencegate.commands.validate import (
    ArtifactPathError,
    render_validate_human,
    render_validate_json,
    run_validate,
)
from saliencegate.experiments import Stage2ConditionId
from saliencegate.shadow import ATIFProfile

_ATIF_CLI_PROFILES = {
    "harbor-terminus-2-v1": ATIFProfile.HARBOR_TERMINUS_2_V1,
    "harbor-codex-v1": ATIFProfile.HARBOR_CODEX_V1,
}

_CAPTURE_PROVIDERS = ("codex", "claude-code", "opencode", "pi")


class ExitCode(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 2
    CONFIGURATION = 3
    UNAVAILABLE_DEPENDENCY = 4
    CORRUPTED_ARTIFACT = 5
    INTERNAL_ERROR = 70


class _UsageError(ValueError):
    pass


class _InspectInputError(ValueError):
    pass


class _InspectRunMismatchError(ValueError):
    pass


class _BenchmarkInputError(ValueError):
    pass


class _BenchmarkArtifactError(ValueError):
    pass


class _PilotInputError(ValueError):
    pass


class _PilotRuntimeUnavailableError(RuntimeError):
    pass


class _PilotRuntimeConfigurationError(RuntimeError):
    pass


class _PilotEvidenceError(RuntimeError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> Never:
        del message
        raise _UsageError


def _parser() -> _SafeArgumentParser:
    parser = _SafeArgumentParser(prog="saliencegate")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo")
    demo.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--repository", default="saliencegate.sqlite3")
    doctor.add_argument("--key")
    doctor.add_argument("--endpoint")
    doctor.add_argument("--capture", action="store_true")
    doctor.add_argument("--json", action="store_true")

    connect = commands.add_parser("connect", help="Install passive project capture")
    connect.add_argument("provider", choices=_CAPTURE_PROVIDERS)
    connect.add_argument("--project")
    connect.add_argument("--dry-run", action="store_true")
    connect.add_argument("--json", action="store_true")

    disconnect = commands.add_parser("disconnect", help="Remove passive project capture")
    disconnect.add_argument("provider", choices=_CAPTURE_PROVIDERS)
    disconnect.add_argument("--project")
    disconnect.add_argument("--json", action="store_true")

    status = commands.add_parser("status", help="Show passive capture status")
    status.add_argument("provider", nargs="?", choices=_CAPTURE_PROVIDERS)
    status.add_argument("--project")
    status.add_argument("--json", action="store_true")

    sessions = commands.add_parser("sessions", help="List captured project sessions")
    sessions.add_argument("--provider", choices=_CAPTURE_PROVIDERS)
    sessions.add_argument("--state", choices=("open", "closed", "quarantined"))
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--json", action="store_true")

    report = commands.add_parser("report", help="Build a passive capture report")
    report_target = report.add_mutually_exclusive_group(required=True)
    report_target.add_argument("--latest", action="store_true")
    report_target.add_argument("session_id", nargs="?")
    report.add_argument("--output")
    report.add_argument("--replace", action="store_true")
    report.add_argument("--json", action="store_true")

    feedback = commands.add_parser("feedback", help="Record local capture feedback")
    feedback.add_argument("session_id")
    feedback.add_argument(
        "--label",
        required=True,
        choices=("memory-needed", "not-memory-needed", "uncertain"),
    )
    feedback.add_argument("--json", action="store_true")

    delete = commands.add_parser("delete", help="Delete local passive capture records")
    delete_target = delete.add_mutually_exclusive_group(required=True)
    delete_target.add_argument("session_id", nargs="?")
    delete_target.add_argument("--all", action="store_true")
    delete.add_argument("--project")
    delete.add_argument("--confirm", action="store_true")
    delete.add_argument("--json", action="store_true")

    replay = commands.add_parser("replay")
    replay.add_argument("trace")
    replay.add_argument("--output", required=True)
    replay.add_argument("--responses")
    replay.add_argument("--replace", action="store_true")
    replay.add_argument("--json", action="store_true")

    shadow = commands.add_parser("shadow")
    shadow_commands = shadow.add_subparsers(dest="shadow_command", required=True)
    shadow_analyze = shadow_commands.add_parser("analyze")
    shadow_analyze.add_argument("trace")
    shadow_analyze.add_argument("--run-id", required=True)
    shadow_analyze.add_argument("--output", required=True)
    shadow_analyze.add_argument("--repository", default=":memory:")
    shadow_analyze.add_argument(
        "--capture-scope",
        choices=(
            "unknown",
            "selected_events",
            "bounded_window",
            "complete_run_declared",
        ),
        default="unknown",
    )
    shadow_analyze.add_argument("--task-scope-digest")
    shadow_analyze.add_argument("--lineage-scope-digest")
    shadow_analyze.add_argument("--capture-manifest-digest")
    shadow_analyze.add_argument("--source-adapter", default="saliencegate-shadow/v1")
    shadow_analyze.add_argument("--replace", action="store_true")
    shadow_analyze.add_argument("--json", action="store_true")
    shadow_analyze_atif = shadow_commands.add_parser("analyze-atif")
    shadow_analyze_atif.add_argument("trace")
    shadow_analyze_atif.add_argument(
        "--profile",
        required=True,
        choices=tuple(_ATIF_CLI_PROFILES),
    )
    shadow_analyze_atif.add_argument("--run-id", required=True)
    shadow_analyze_atif.add_argument("--working-directory", required=True)
    shadow_analyze_atif.add_argument("--environment-digest", required=True)
    shadow_analyze_atif.add_argument("--output", required=True)
    shadow_analyze_atif.add_argument("--repository", default=":memory:")
    shadow_analyze_atif.add_argument("--task-scope-digest")
    shadow_analyze_atif.add_argument("--lineage-scope-digest")
    shadow_analyze_atif.add_argument("--capture-manifest-digest")
    shadow_analyze_atif.add_argument("--replace", action="store_true")
    shadow_analyze_atif.add_argument("--json", action="store_true")

    algorithm = commands.add_parser("algorithm")
    algorithm_commands = algorithm.add_subparsers(dest="algorithm_command", required=True)
    algorithm_replay = algorithm_commands.add_parser("replay")
    algorithm_replay.add_argument("trace")
    algorithm_replay.add_argument("--responses")
    algorithm_replay.add_argument(
        "--condition",
        required=True,
        choices=tuple(condition.value for condition in Stage2ConditionId),
    )
    algorithm_replay.add_argument("--output", required=True)
    algorithm_replay.add_argument("--replace", action="store_true")
    algorithm_replay.add_argument("--json", action="store_true")

    pilot = commands.add_parser("pilot")
    pilot_commands = pilot.add_subparsers(dest="pilot_command", required=True)
    paper_two_phase = pilot_commands.add_parser("paper-two-phase")
    paper_two_phase.add_argument("--endpoint", required=True)
    paper_two_phase.add_argument("--model", required=True)
    paper_two_phase.add_argument("--output", required=True)
    paper_two_phase.add_argument("--warmup", choices=("warm", "cold"), default="warm")
    paper_two_phase.add_argument("--json", action="store_true")

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("suite")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--replace", action="store_true")
    benchmark.add_argument("--json", action="store_true")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("run_id")
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--json", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("artifact")
    validate.add_argument("--expected-digest")
    validate.add_argument("--require-confirmatory", action="store_true")
    validate.add_argument("--json", action="store_true")
    return parser


def _write_stdout(value: str) -> None:
    sys.stdout.write(value)
    sys.stdout.flush()


def _write_error(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")
    sys.stderr.flush()


def _doctor_exit(report: object) -> ExitCode:
    environment = report.environment if type(report) is CaptureDoctorReport else report
    checks = getattr(environment, "checks", ())
    failed = tuple(
        check.name for check in checks if check.required and check.status is DoctorCheckStatus.FAIL
    )
    dependency_checks = {
        DoctorCheckName.PYTHON,
        DoctorCheckName.SQLITE,
        DoctorCheckName.FTS5,
    }
    if any(name in dependency_checks for name in failed):
        return ExitCode.UNAVAILABLE_DEPENDENCY
    if type(report) is CaptureDoctorReport and report.capture.status is DoctorCheckStatus.FAIL:
        return ExitCode.CONFIGURATION
    return ExitCode.CONFIGURATION if failed else ExitCode.SUCCESS


def _dispatch_demo(arguments: argparse.Namespace) -> ExitCode:
    report = run_demo()
    _write_stdout(render_demo_json(report) if arguments.json else render_demo_human(report))
    return ExitCode.SUCCESS


def _dispatch_doctor(arguments: argparse.Namespace) -> ExitCode:
    key = None if arguments.key is None else Path(arguments.key)
    if arguments.capture:
        capture_report = run_capture_doctor(
            repository_path=arguments.repository,
            installation_key_path=key,
            endpoint=arguments.endpoint,
        )
        _write_stdout(
            render_capture_doctor_json(capture_report)
            if arguments.json
            else render_capture_doctor_human(capture_report)
        )
        return _doctor_exit(capture_report)
    report = run_doctor(
        repository_path=arguments.repository,
        installation_key_path=key,
        endpoint=arguments.endpoint,
    )
    _write_stdout(render_doctor_json(report) if arguments.json else render_doctor_human(report))
    return _doctor_exit(report)


def _dispatch_connect(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.connect import (
        render_connect_human,
        render_connect_json,
        run_connect,
    )

    report = run_connect(
        provider=arguments.provider,
        project=arguments.project,
        dry_run=arguments.dry_run,
    )
    _write_stdout(render_connect_json(report) if arguments.json else render_connect_human(report))
    return ExitCode.SUCCESS


def _dispatch_disconnect(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.disconnect import (
        render_disconnect_human,
        render_disconnect_json,
        run_disconnect,
    )

    report = run_disconnect(provider=arguments.provider, project=arguments.project)
    _write_stdout(
        render_disconnect_json(report) if arguments.json else render_disconnect_human(report)
    )
    return ExitCode.SUCCESS


def _dispatch_status(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.status import (
        render_status_human,
        render_status_json,
        run_status,
    )

    report = run_status(provider=arguments.provider, project=arguments.project)
    _write_stdout(render_status_json(report) if arguments.json else render_status_human(report))
    return ExitCode.SUCCESS


def _dispatch_sessions(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.sessions import (
        render_sessions_human,
        render_sessions_json,
        run_sessions,
    )

    report = run_sessions(
        provider=arguments.provider,
        state=arguments.state,
        limit=arguments.limit,
    )
    _write_stdout(render_sessions_json(report) if arguments.json else render_sessions_human(report))
    return ExitCode.SUCCESS


def _dispatch_capture_report(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.report import (
        render_capture_session_report_human,
        render_capture_session_report_json,
        run_capture_report,
    )

    report = run_capture_report(
        latest=arguments.latest,
        session_id=arguments.session_id,
        output_path=arguments.output,
        replace=arguments.replace,
    )
    _write_stdout(
        render_capture_session_report_json(report)
        if arguments.json
        else render_capture_session_report_human(report)
    )
    return ExitCode.SUCCESS


def _dispatch_capture_feedback(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.feedback import (
        render_capture_feedback_human,
        render_capture_feedback_json,
        run_capture_feedback,
    )

    report = run_capture_feedback(
        session_id=arguments.session_id,
        label=arguments.label,
    )
    _write_stdout(
        render_capture_feedback_json(report)
        if arguments.json
        else render_capture_feedback_human(report)
    )
    return ExitCode.SUCCESS


def _dispatch_delete(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.capture.delete import (
        render_delete_human,
        render_delete_json,
        run_delete,
    )

    if not arguments.all and (arguments.project is not None or arguments.confirm):
        raise CaptureCommandInputError()
    report = run_delete(
        session_id=arguments.session_id,
        delete_all=arguments.all,
        confirm=arguments.confirm,
        project=arguments.project,
    )
    _write_stdout(render_delete_json(report) if arguments.json else render_delete_human(report))
    return ExitCode.SUCCESS


def _dispatch_replay(arguments: argparse.Namespace) -> ExitCode:
    report = asyncio.run(
        run_replay(
            arguments.trace,
            output_path=arguments.output,
            responses_path=arguments.responses,
            replace=arguments.replace,
        )
    )
    _write_stdout(render_replay_json(report) if arguments.json else render_replay_human(report))
    return ExitCode.SUCCESS


def _dispatch_shadow(arguments: argparse.Namespace) -> ExitCode:
    try:
        run_id = UUID(arguments.run_id)
    except (AttributeError, TypeError, ValueError):
        raise ShadowCommandInputError() from None
    if run_id.version != 4 or str(run_id) != arguments.run_id:
        raise ShadowCommandInputError()
    if arguments.shadow_command == "analyze":
        report = asyncio.run(
            run_shadow_analyze(
                arguments.trace,
                run_id=run_id,
                output_path=arguments.output,
                repository_path=arguments.repository,
                capture_scope=arguments.capture_scope,
                task_scope_digest=arguments.task_scope_digest,
                lineage_scope_digest=arguments.lineage_scope_digest,
                capture_manifest_digest=arguments.capture_manifest_digest,
                source_adapter=arguments.source_adapter,
                replace=arguments.replace,
            )
        )
        _write_stdout(render_shadow_json(report) if arguments.json else render_shadow_human(report))
        return ExitCode.SUCCESS
    if arguments.shadow_command == "analyze-atif":
        profile = _ATIF_CLI_PROFILES.get(arguments.profile)
        if profile is None:  # pragma: no cover - argparse enforces the closed aliases
            raise _UsageError
        atif_report = asyncio.run(
            run_shadow_analyze_atif(
                arguments.trace,
                profile=profile,
                run_id=run_id,
                working_directory=arguments.working_directory,
                environment_digest=arguments.environment_digest,
                output_path=arguments.output,
                repository_path=arguments.repository,
                task_scope_digest=arguments.task_scope_digest,
                lineage_scope_digest=arguments.lineage_scope_digest,
                capture_manifest_digest=arguments.capture_manifest_digest,
                replace=arguments.replace,
            )
        )
        _write_stdout(
            render_shadow_atif_json(atif_report)
            if arguments.json
            else render_shadow_atif_human(atif_report)
        )
        return ExitCode.SUCCESS
    raise _UsageError  # pragma: no cover - required subparser


def _dispatch_algorithm(arguments: argparse.Namespace) -> ExitCode:
    if arguments.algorithm_command != "replay":  # pragma: no cover - required subparser
        raise _UsageError
    report = asyncio.run(
        run_algorithm_replay(
            arguments.trace,
            condition=arguments.condition,
            output_path=arguments.output,
            responses_path=arguments.responses,
            replace=arguments.replace,
        )
    )
    _write_stdout(
        render_algorithm_replay_json(report)
        if arguments.json
        else render_algorithm_replay_human(report)
    )
    return ExitCode.SUCCESS


def _dispatch_pilot(arguments: argparse.Namespace) -> ExitCode:
    if arguments.pilot_command != "paper-two-phase":  # pragma: no cover - required subparser
        raise _UsageError
    try:
        from saliencegate.commands.pilot import (
            PilotCommandError,
            PilotEvidenceError,
            PilotRuntimeConfigurationError,
            PilotRuntimeUnavailableError,
            render_pilot_human,
            render_pilot_json,
            run_paper_two_phase_pilot,
        )
    except ModuleNotFoundError as error:
        if error.name in {"httpx", "openai_harmony"}:
            raise _PilotRuntimeUnavailableError from None
        raise
    try:
        report = asyncio.run(
            run_paper_two_phase_pilot(
                endpoint=arguments.endpoint,
                model=arguments.model,
                output_path=arguments.output,
                warmup=arguments.warmup,
            )
        )
    except PilotCommandError:
        raise _PilotInputError from None
    except PilotRuntimeUnavailableError:
        raise _PilotRuntimeUnavailableError from None
    except PilotRuntimeConfigurationError:
        raise _PilotRuntimeConfigurationError from None
    except PilotEvidenceError:
        raise _PilotEvidenceError from None
    _write_stdout(render_pilot_json(report) if arguments.json else render_pilot_human(report))
    return ExitCode.SUCCESS


def _dispatch_inspect(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.commands.inspect import (
        InspectInputError,
        InspectRunMismatchError,
        render_inspect_human,
        render_inspect_json,
        run_inspect,
    )

    try:
        run_id = UUID(arguments.run_id)
    except (AttributeError, TypeError, ValueError):
        raise _InspectInputError from None
    if run_id.version != 4:
        raise _InspectInputError
    try:
        report = run_inspect(run_id, artifact_path=arguments.artifact)
    except InspectInputError:
        raise _InspectInputError from None
    except InspectRunMismatchError:
        raise _InspectRunMismatchError from None
    _write_stdout(render_inspect_json(report) if arguments.json else render_inspect_human(report))
    return ExitCode.SUCCESS


def _dispatch_validate(arguments: argparse.Namespace) -> ExitCode:
    report = run_validate(
        arguments.artifact,
        expected_digest=arguments.expected_digest,
        require_confirmatory=arguments.require_confirmatory,
    )
    _write_stdout(render_validate_json(report) if arguments.json else render_validate_human(report))
    return ExitCode.SUCCESS


def _dispatch_benchmark(arguments: argparse.Namespace) -> ExitCode:
    from saliencegate.benchmarks.registry import BenchmarkNotFoundError, get_benchmark
    from saliencegate.benchmarks.state_decay.runner import (
        BenchmarkArtifactValidationError,
        BenchmarkCommandError,
        render_benchmark_human,
        render_benchmark_json,
        run_state_decay_smoke,
    )

    try:
        definition = get_benchmark(arguments.suite)
        if definition.suite_id != "state-decay-smoke":  # pragma: no cover - closed registry
            raise BenchmarkNotFoundError()
        report = run_state_decay_smoke(
            arguments.output,
            replace=arguments.replace,
        )
    except (BenchmarkCommandError, BenchmarkNotFoundError):
        raise _BenchmarkInputError from None
    except BenchmarkArtifactValidationError:
        raise _BenchmarkArtifactError from None
    _write_stdout(
        render_benchmark_json(report) if arguments.json else render_benchmark_human(report)
    )
    return ExitCode.SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command while keeping stdout machine-safe and errors value-free."""

    try:
        try:
            arguments = _parser().parse_args(argv)
        except _UsageError:
            _write_error("invalid command line")
            return ExitCode.INVALID_INPUT
        except SystemExit as error:
            return int(error.code or 0)

        if arguments.command == "demo":
            return _dispatch_demo(arguments)
        if arguments.command == "doctor":
            return _dispatch_doctor(arguments)
        if arguments.command == "connect":
            return _dispatch_connect(arguments)
        if arguments.command == "disconnect":
            return _dispatch_disconnect(arguments)
        if arguments.command == "status":
            return _dispatch_status(arguments)
        if arguments.command == "sessions":
            return _dispatch_sessions(arguments)
        if arguments.command == "report":
            return _dispatch_capture_report(arguments)
        if arguments.command == "feedback":
            return _dispatch_capture_feedback(arguments)
        if arguments.command == "delete":
            return _dispatch_delete(arguments)
        if arguments.command == "replay":
            return _dispatch_replay(arguments)
        if arguments.command == "shadow":
            return _dispatch_shadow(arguments)
        if arguments.command == "algorithm":
            return _dispatch_algorithm(arguments)
        if arguments.command == "pilot":
            return _dispatch_pilot(arguments)
        if arguments.command == "benchmark":
            return _dispatch_benchmark(arguments)
        if arguments.command == "inspect":
            return _dispatch_inspect(arguments)
        if arguments.command == "validate":
            return _dispatch_validate(arguments)
        raise _UsageError
    except ReplayCommandError:
        _write_error("replay input or output is invalid")
        return ExitCode.INVALID_INPUT
    except ReplayConfigurationError:
        _write_error("replay configuration is invalid")
        return ExitCode.CONFIGURATION
    except ShadowCommandInputError:
        _write_error("shadow input or output is invalid")
        return ExitCode.INVALID_INPUT
    except ShadowCommandConfigurationError:
        _write_error("shadow configuration is invalid")
        return ExitCode.CONFIGURATION
    except ShadowCommandIntegrityError:
        _write_error("shadow report integrity check failed")
        return ExitCode.CORRUPTED_ARTIFACT
    except AlgorithmReplayCommandError:
        _write_error("algorithm replay input or output is invalid")
        return ExitCode.INVALID_INPUT
    except _PilotInputError:
        _write_error("pilot input or output is invalid")
        return ExitCode.INVALID_INPUT
    except _PilotRuntimeUnavailableError:
        _write_error("pilot model runtime is unavailable")
        return ExitCode.UNAVAILABLE_DEPENDENCY
    except _PilotRuntimeConfigurationError:
        _write_error("pilot runtime configuration is invalid")
        return ExitCode.CONFIGURATION
    except _PilotEvidenceError:
        _write_error("pilot evidence requirements failed")
        return ExitCode.CORRUPTED_ARTIFACT
    except ArtifactPathError:
        _write_error("artifact path is invalid")
        return ExitCode.INVALID_INPUT
    except _InspectInputError:
        _write_error("artifact inspection input is invalid")
        return ExitCode.INVALID_INPUT
    except _InspectRunMismatchError:
        _write_error("inspect run does not match artifact")
        return ExitCode.INVALID_INPUT
    except _BenchmarkInputError:
        _write_error("benchmark input or output is invalid")
        return ExitCode.INVALID_INPUT
    except _BenchmarkArtifactError:
        _write_error("benchmark artifact validation failed")
        return ExitCode.CORRUPTED_ARTIFACT
    except ArtifactValidationError:
        _write_error("artifact validation failed")
        return ExitCode.CORRUPTED_ARTIFACT
    except StateDecayDiagnosticError:
        _write_error("internal error")
        return ExitCode.INTERNAL_ERROR
    except CaptureCommandInputError:
        _write_error("capture command input is invalid")
        return ExitCode.INVALID_INPUT
    except CaptureCommandRequiresDisconnectError:
        _write_error("run saliencegate disconnect before delete --all")
        return ExitCode.CONFIGURATION
    except CaptureCommandConfigurationError:
        _write_error("capture configuration is invalid")
        return ExitCode.CONFIGURATION
    except CaptureCommandUnavailableError:
        _write_error("capture integration is unavailable")
        return ExitCode.UNAVAILABLE_DEPENDENCY
    except CaptureCommandIntegrityError:
        _write_error("capture integrity check failed")
        return ExitCode.CORRUPTED_ARTIFACT
    except CaptureCommandError:
        _write_error("internal error")
        return ExitCode.INTERNAL_ERROR
    except BrokenPipeError:
        return ExitCode.SUCCESS
    except KeyboardInterrupt:
        return 130
    except Exception:
        _write_error("internal error")
        return ExitCode.INTERNAL_ERROR


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["ExitCode", "entrypoint", "main"]
