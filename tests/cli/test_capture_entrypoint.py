from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from tests.cli.conftest import RunCli

from saliencegate import cli as cli_module
from saliencegate.commands.capture import (
    CaptureCommandIntegrityError,
    CaptureCommandRequiresDisconnectError,
)

PROVIDERS = ("codex", "claude-code", "opencode", "pi")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_connect_parser_exposes_the_locked_provider_surface(provider: str) -> None:
    arguments = cli_module._parser().parse_args(
        ("connect", provider, "--project", "/synthetic/project", "--dry-run", "--json")
    )

    assert vars(arguments) == {
        "command": "connect",
        "provider": provider,
        "project": "/synthetic/project",
        "global_scope": False,
        "exclude": None,
        "dry_run": True,
        "json": True,
    }


def test_capture_query_parsers_expose_the_locked_surface() -> None:
    parser = cli_module._parser()

    disconnect = parser.parse_args(("disconnect", "pi", "--json"))
    status = parser.parse_args(("status", "opencode", "--project", "/synthetic/project"))
    sessions = parser.parse_args(
        ("sessions", "--provider", "codex", "--state", "closed", "--limit", "25", "--json")
    )
    latest = parser.parse_args(("report", "--latest", "--output", "report.json", "--replace"))
    selected = parser.parse_args(("report", "sgabcdefghijkl", "--json"))
    feedback = parser.parse_args(
        ("feedback", "sgabcdefghijkl", "--label", "memory-needed", "--json")
    )

    assert vars(disconnect) == {
        "command": "disconnect",
        "provider": "pi",
        "project": None,
        "global_scope": False,
        "json": True,
    }
    assert vars(status) == {
        "command": "status",
        "provider": "opencode",
        "project": "/synthetic/project",
        "global_scope": False,
        "json": False,
    }
    assert vars(sessions) == {
        "command": "sessions",
        "provider": "codex",
        "state": "closed",
        "limit": 25,
        "json": True,
    }
    assert vars(latest) == {
        "command": "report",
        "latest": True,
        "session_id": None,
        "output": "report.json",
        "replace": True,
        "json": False,
    }
    assert vars(selected) == {
        "command": "report",
        "latest": False,
        "session_id": "sgabcdefghijkl",
        "output": None,
        "replace": False,
        "json": True,
    }
    assert vars(feedback) == {
        "command": "feedback",
        "session_id": "sgabcdefghijkl",
        "label": "memory-needed",
        "json": True,
    }


def test_delete_parser_requires_one_explicit_scope() -> None:
    parser = cli_module._parser()

    selected = parser.parse_args(("delete", "sgabcdefghijkl", "--json"))
    all_project = parser.parse_args(
        ("delete", "--all", "--project", "/synthetic/project", "--confirm", "--json")
    )

    assert vars(selected) == {
        "command": "delete",
        "session_id": "sgabcdefghijkl",
        "all": False,
        "project": None,
        "confirm": False,
        "json": True,
    }
    assert vars(all_project) == {
        "command": "delete",
        "session_id": None,
        "all": True,
        "project": "/synthetic/project",
        "confirm": True,
        "json": True,
    }


@pytest.mark.parametrize(
    "arguments",
    (
        ("connect", "unknown"),
        ("connect", "codex", "--dry"),
        ("disconnect",),
        ("status", "unknown"),
        ("sessions", "--state", "deleting"),
        ("sessions", "--limit", "not-an-integer"),
        ("report",),
        ("report", "--latest", "sgabcdefghijkl"),
        ("feedback", "sgabcdefghijkl"),
        ("feedback", "sgabcdefghijkl", "--label", "unknown"),
        ("feedback", "sgabcdefghijkl", "--lab", "memory-needed"),
        ("delete",),
        ("delete", "sgabcdefghijkl", "--all"),
    ),
)
def test_capture_parser_rejects_ambiguous_or_abbreviated_forms(
    arguments: tuple[str, ...],
    run_cli: RunCli,
) -> None:
    completed = run_cli(*arguments)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: invalid command line\n"


def test_connect_help_is_short_and_copyable(run_cli: RunCli) -> None:
    completed = run_cli("connect", "--help")

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "{codex,claude-code,opencode,pi}" in completed.stdout
    assert "--project PROJECT" in completed.stdout
    assert "--dry-run" in completed.stdout
    assert "--json" in completed.stdout
    assert len(completed.stdout.splitlines()) <= 14


def test_feedback_help_is_exact_and_copyable(run_cli: RunCli) -> None:
    completed = run_cli("feedback", "--help")

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == (
        "usage: saliencegate feedback [-h] --label\n"
        "                             {memory-needed,not-memory-needed,uncertain}\n"
        "                             [--json]\n"
        "                             session_id\n"
        "\n"
        "positional arguments:\n"
        "  session_id\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --label {memory-needed,not-memory-needed,uncertain}\n"
        "  --json\n"
    )


def test_doctor_parser_accepts_the_read_only_capture_probe() -> None:
    arguments = cli_module._parser().parse_args(("doctor", "--capture", "--json"))

    assert isinstance(arguments, argparse.Namespace)
    assert arguments.capture is True


def test_capture_doctor_entrypoint_is_read_only(run_cli: RunCli) -> None:
    completed = run_cli("doctor", "--capture", "--repository", ":memory:", "--json")

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "capture-doctor/v1"
    assert payload["capture"]["state"] == "not_configured"


def test_read_only_capture_commands_have_stable_empty_state_output(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    status = run_cli("status", "codex", "--project", str(project), "--json")
    sessions = run_cli("sessions", "--json")

    assert status.returncode == sessions.returncode == 0
    assert status.stderr == sessions.stderr == ""
    assert json.loads(status.stdout)["providers"][0]["status"] == "not_installed"
    assert json.loads(sessions.stdout) == {
        "schema_version": "capture-sessions/v1",
        "sessions": [],
    }


def test_capture_command_failures_use_stable_content_free_exits(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    project = tmp_path / "fixture-secret-project"
    project.mkdir()

    unavailable = run_cli("report", "sgabcdefghijkl", "--json")
    invalid = run_cli("delete", "--all", "--project", str(project))
    missing_report = run_cli("report", "--latest", "--json")
    missing_feedback = run_cli(
        "feedback",
        "sgabcdefghijkl",
        "--label",
        "uncertain",
        "--json",
    )

    assert unavailable.returncode == missing_feedback.returncode == 4
    assert unavailable.stdout == ""
    assert missing_feedback.stdout == ""
    assert unavailable.stderr == "error: capture integration is unavailable\n"
    assert missing_feedback.stderr == "error: capture integration is unavailable\n"
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert invalid.stderr == "error: capture command input is invalid\n"
    assert missing_report.returncode == 4
    assert missing_report.stdout == ""
    assert missing_report.stderr == "error: capture integration is unavailable\n"
    assert "fixture-secret" not in (
        unavailable.stderr + invalid.stderr + missing_report.stderr + missing_feedback.stderr
    )


def test_delete_all_connected_error_directs_the_user_to_disconnect(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: argparse.Namespace) -> None:
        raise CaptureCommandRequiresDisconnectError()

    monkeypatch.setattr(cli_module, "_dispatch_delete", fail)

    code = cli_module.main(("delete", "--all", "--project", ".", "--confirm"))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.CONFIGURATION
    assert captured.out == ""
    assert captured.err == "error: run saliencegate disconnect before delete --all\n"


def test_delete_integrity_failure_uses_the_corrupted_artifact_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: argparse.Namespace) -> None:
        raise CaptureCommandIntegrityError()

    monkeypatch.setattr(cli_module, "_dispatch_delete", fail)

    code = cli_module.main(("delete", "sgabcdefghijkl"))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.CORRUPTED_ARTIFACT
    assert captured.out == ""
    assert captured.err == "error: capture integrity check failed\n"
