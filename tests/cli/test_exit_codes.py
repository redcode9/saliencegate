from __future__ import annotations

from pathlib import Path

import pytest

import saliencegate.cli as cli_module
import saliencegate.commands.doctor as doctor_module
from saliencegate.cli import ExitCode, main
from saliencegate.commands.shadow import (
    ShadowCommandConfigurationError,
    ShadowCommandInputError,
    ShadowCommandIntegrityError,
)
from saliencegate.shadow import ShadowInvariantError

SHADOW_ARGUMENTS = (
    "shadow",
    "analyze",
    "trace",
    "--run-id",
    "b35f05f3-555b-4f09-8996-a7b3693bb54a",
    "--output",
    "report",
)


def test_public_exit_code_values_are_stable() -> None:
    assert {name: int(code) for name, code in ExitCode.__members__.items()} == {
        "SUCCESS": 0,
        "INVALID_INPUT": 2,
        "CONFIGURATION": 3,
        "UNAVAILABLE_DEPENDENCY": 4,
        "CORRUPTED_ARTIFACT": 5,
        "INTERNAL_ERROR": 70,
    }


def test_doctor_dependency_failure_uses_the_unavailable_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(doctor_module, "_runtime_python_version", lambda: (3, 10, 0))

    code = main(
        (
            "doctor",
            "--repository",
            ":memory:",
            "--key",
            str(tmp_path / "installation.key"),
            "--json",
        )
    )

    captured = capsys.readouterr()
    assert code == ExitCode.UNAVAILABLE_DEPENDENCY
    assert '"status":"unhealthy"' in captured.out
    assert captured.err == ""


def test_unexpected_failures_are_value_free_and_use_the_internal_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: object) -> ExitCode:
        raise RuntimeError("fixture-secret internal detail")

    monkeypatch.setattr(cli_module, "_dispatch_doctor", fail)

    code = main(("doctor", "--repository", ":memory:", "--json"))

    captured = capsys.readouterr()
    assert code == ExitCode.INTERNAL_ERROR
    assert captured.out == ""
    assert captured.err == "error: internal error\n"
    assert "fixture-secret" not in captured.err


@pytest.mark.parametrize(
    ("failure", "expected"),
    ((BrokenPipeError(), 0), (KeyboardInterrupt(), 130)),
)
def test_stream_close_and_interrupt_have_conventional_exits(
    failure: BaseException,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: object) -> ExitCode:
        raise failure

    monkeypatch.setattr(cli_module, "_dispatch_doctor", fail)

    assert main(("doctor", "--repository", ":memory:", "--json")) == expected
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_error"),
    (
        (
            ShadowCommandInputError(),
            ExitCode.INVALID_INPUT,
            "error: shadow input or output is invalid\n",
        ),
        (
            ShadowCommandConfigurationError(),
            ExitCode.CONFIGURATION,
            "error: shadow configuration is invalid\n",
        ),
        (
            ShadowCommandIntegrityError(),
            ExitCode.CORRUPTED_ARTIFACT,
            "error: shadow report integrity check failed\n",
        ),
        (
            ShadowInvariantError(),
            ExitCode.INTERNAL_ERROR,
            "error: internal error\n",
        ),
        (
            RuntimeError("fixture-secret internal detail"),
            ExitCode.INTERNAL_ERROR,
            "error: internal error\n",
        ),
    ),
)
def test_shadow_failures_use_stable_value_free_exit_mapping(
    failure: Exception,
    expected_code: ExitCode,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: object) -> ExitCode:
        raise failure

    monkeypatch.setattr(cli_module, "_dispatch_shadow", fail)

    code = main(SHADOW_ARGUMENTS)

    captured = capsys.readouterr()
    assert code == expected_code
    assert captured.out == ""
    assert captured.err == expected_error
    assert "fixture-secret" not in captured.err


@pytest.mark.parametrize(
    ("failure", "expected"),
    ((BrokenPipeError(), 0), (KeyboardInterrupt(), 130)),
)
def test_shadow_stream_close_and_interrupt_have_conventional_exits(
    failure: BaseException,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: object) -> ExitCode:
        raise failure

    monkeypatch.setattr(cli_module, "_dispatch_shadow", fail)

    assert main(SHADOW_ARGUMENTS) == expected
    assert capsys.readouterr() == ("", "")
