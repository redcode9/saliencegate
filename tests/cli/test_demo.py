from __future__ import annotations

import builtins
import importlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Never, cast

import pytest
from pydantic import ValidationError
from tests.cli.conftest import RunCli

import saliencegate.cli as cli_module
import saliencegate.commands.demo as demo_module
from saliencegate.benchmarks.state_decay.diagnostic import (
    StateDecayDiagnosticError,
    run_state_decay_diagnostic,
)
from saliencegate.cli import ExitCode, main
from saliencegate.commands.demo import (
    DemoCommandReport,
    render_demo_human,
    render_demo_json,
    run_demo,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256

_RESULT_DIGEST = "13704b753086925db1abfd7467f3e202edb202d60f620d150b1cdb6099c57d0f"
_EXPECTED_REPORT = {
    "confirmatory": False,
    "diagnostic": True,
    "evidence_level": "synthetic_diagnostic",
    "external_claims_assessment": "insufficient",
    "external_claims_supported": False,
    "family_count": 8,
    "intervene_count": 16,
    "oracle_failed": 0,
    "oracle_passed": 32,
    "result_digest": _RESULT_DIGEST,
    "scenario_count": 32,
    "schema_version": "cli-demo-report/v1",
    "silence_count": 16,
    "status": "ok",
    "suite_id": "state-decay-smoke",
    "synthetic": True,
}
_EXPECTED_HUMAN = (
    "SalienceGate offline demo\n"
    "suite: state-decay-smoke\n"
    "evidence: synthetic diagnostic\n"
    "scenarios: 32 across 8 families\n"
    "decisions: 16 intervene, 16 silence\n"
    "oracle: 32 passed, 0 failed\n"
    "confirmatory: no\n"
    "external claims: insufficient\n"
    f"result digest: {_RESULT_DIGEST}\n"
    "This verifies deterministic mechanics, not agent task efficacy.\n"
)


def test_demo_service_returns_the_exact_stable_aggregate_contract() -> None:
    first = run_demo()
    second = run_demo()

    assert first == second
    assert first.model_dump(mode="json", warnings=False) == _EXPECTED_REPORT
    assert render_demo_json(first) == render_demo_json(second)
    assert render_demo_json(first) == canonical_json(_EXPECTED_REPORT).decode("utf-8") + "\n"
    assert render_demo_human(first) == render_demo_human(second) == _EXPECTED_HUMAN


def test_demo_digest_frames_the_ordered_canonical_diagnostic_tuples() -> None:
    diagnostic = run_state_decay_diagnostic()

    expected = length_prefixed_sha256(
        canonical_json(diagnostic.scenarios),
        canonical_json(diagnostic.oracle_results),
        domain="saliencegate:demo:state-decay-smoke:v1",
    )

    assert expected == _RESULT_DIGEST
    assert run_demo().result_digest == expected


def test_demo_report_is_strict_frozen_default_validating_and_revalidated() -> None:
    report = run_demo()

    with pytest.raises(ValidationError):
        DemoCommandReport.model_validate({**_EXPECTED_REPORT, "scenario_count": "32"})
    with pytest.raises(ValidationError):
        DemoCommandReport.model_validate({**_EXPECTED_REPORT, "unexpected": True})
    with pytest.raises(ValidationError):
        DemoCommandReport(result_digest="A" * 64)
    with pytest.raises(ValidationError):
        DemoCommandReport(result_digest="0" * 64)
    with pytest.raises(ValidationError):
        report.status = "changed"  # type: ignore[assignment,misc]

    forged = report.model_copy(update={"scenario_count": 31})
    for renderer in (render_demo_json, render_demo_human):
        with pytest.raises(StateDecayDiagnosticError) as raised:
            renderer(forged)
        assert str(raised.value) == "state decay diagnostic failed"
        assert raised.value.__cause__ is None


def test_demo_cli_json_and_human_output_are_byte_stable(run_cli: RunCli) -> None:
    first = run_cli("demo", "--json")
    second = run_cli("demo", "--json")
    human = run_cli("demo")

    assert first.returncode == second.returncode == human.returncode == 0
    assert first.stderr == second.stderr == human.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == _EXPECTED_REPORT
    assert first.stdout == canonical_json(_EXPECTED_REPORT).decode("utf-8") + "\n"
    assert human.stdout == _EXPECTED_HUMAN


@pytest.mark.parametrize(
    "arguments",
    (
        ("--output", "fixture-secret-path"),
        ("--json", "fixture-secret-value"),
        ("--unknown=fixture-secret-option",),
    ),
)
def test_demo_accepts_only_json_and_usage_errors_are_value_free(
    arguments: tuple[str, ...],
    run_cli: RunCli,
) -> None:
    completed = run_cli("demo", *arguments)

    assert completed.returncode == ExitCode.INVALID_INPUT
    assert completed.stdout == ""
    assert completed.stderr == "error: invalid command line\n"
    assert "fixture-secret" not in completed.stderr


def test_demo_diagnostic_failure_is_a_value_free_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> Never:
        raise StateDecayDiagnosticError() from None

    monkeypatch.setattr(demo_module, "run_state_decay_diagnostic", fail)

    assert main(("demo", "--json")) == ExitCode.INTERNAL_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: internal error\n"


@pytest.mark.parametrize(
    ("failure", "expected"),
    ((BrokenPipeError(), ExitCode.SUCCESS), (KeyboardInterrupt(), 130)),
)
def test_demo_preserves_global_stream_and_interrupt_behavior(
    failure: BaseException,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: object) -> Never:
        raise failure

    monkeypatch.setattr(cli_module, "_dispatch_demo", fail)

    assert main(("demo", "--json")) == expected
    assert capsys.readouterr() == ("", "")


class _ForbiddenEnvironment(dict[str, str]):
    def _fail(self, *args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("demo attempted environment access")

    __contains__ = _fail
    __getitem__ = _fail
    __iter__ = _fail
    __len__ = _fail
    get = _fail
    items = _fail
    keys = _fail
    values = _fail


def test_demo_service_is_pure_core_only_and_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("demo crossed a forbidden boundary")

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name in {"httpx", "openai_harmony"} or name.startswith(
            "saliencegate.models.openai_compatible"
        ):
            raise AssertionError("demo attempted optional-runtime import")
        return real_import(name, globals, locals, fromlist, level)

    for method in (
        "open",
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
        "mkdir",
        "touch",
        "unlink",
        "rename",
        "replace",
        "stat",
        "lstat",
        "exists",
        "iterdir",
        "glob",
        "rglob",
    ):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(os, "environ", cast(os._Environ[str], _ForbiddenEnvironment()))
    for function in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, function, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    for module in ("httpx", "openai_harmony", "saliencegate.models.openai_compatible"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    report = run_demo()

    assert report.model_dump(mode="json", warnings=False) == _EXPECTED_REPORT
    assert "httpx" not in sys.modules
    assert "openai_harmony" not in sys.modules
    assert "saliencegate.models.openai_compatible" not in sys.modules
