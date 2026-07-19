from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from scripts import smoke_shadow_installed as smoke

from saliencegate.cli import main as cli_main
from saliencegate.domain import canonical_json
from saliencegate.shadow import decode_shadow_trace_report

ROOT = Path(__file__).parents[1]
SOCKET_GUARD = ROOT / "scripts" / "run_without_sockets.py"
INSTALLED_SHADOW_SMOKE = ROOT / "scripts" / "smoke_shadow_installed.py"


def _make_private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _assert_private_file(path: Path) -> bytes:
    metadata = path.stat(follow_symlinks=False)
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_uid == os.getuid()
    assert metadata.st_nlink == 1
    return path.read_bytes()


def test_environment_guard_allows_local_key_roots_but_rejects_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_values = {
        "HOME": "/private/synthetic-home",
        "XDG_CONFIG_HOME": "/private/synthetic-config",
        "APPDATA": "/private/synthetic-appdata",
    }
    for key, value in local_values.items():
        monkeypatch.setenv(key, value)
    for key in smoke._PROVIDER_CREDENTIAL_KEYS:
        monkeypatch.setenv(key, f"secret-value-for-{key}")

    original = os.environ
    with smoke._guard_provider_environment_reads() as guard:
        for key, value in local_values.items():
            assert os.environ.get(key) == value
        for key in smoke._PROVIDER_CREDENTIAL_KEYS:
            with pytest.raises(RuntimeError, match="provider credential") as captured:
                os.getenv(key)
            assert f"secret-value-for-{key}" not in str(captured.value)
            with pytest.raises(RuntimeError, match="provider credential"):
                os.environ.__contains__(key)

    assert os.environ is original
    assert guard.reads >= smoke._LOCAL_KEY_ENVIRONMENT_KEYS


def test_core_only_probes_every_provider_and_optional_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def absent(module_name: str) -> None:
        probed.append(module_name)
        return None

    monkeypatch.setattr(smoke.importlib.util, "find_spec", absent)
    monkeypatch.setattr(smoke, "_assert_import_exclusions", lambda: None)

    smoke._assert_core_only()

    assert tuple(probed) == smoke._IMPORTED_MODULE_EXCLUSIONS
    assert set(probed) == {"harbor", "anthropic", "openai", "httpx", "openai_harmony"}


def test_import_exclusion_rejects_provider_submodules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_name = "harbor.synthetic_installed_smoke"
    monkeypatch.setitem(sys.modules, synthetic_name, ModuleType(synthetic_name))

    with pytest.raises(RuntimeError, match="runtime was imported"):
        smoke._assert_import_exclusions()


@pytest.mark.skipif(os.name != "posix", reason="installed private-file smoke is POSIX-only")
def test_legacy_write_trace_mode_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = _make_private_directory(tmp_path / "legacy-shadow")
    trace_path = private_root / "events.ndjson"
    monkeypatch.setattr(smoke, "_assert_core_only", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [os.fspath(INSTALLED_SHADOW_SMOKE), "write-trace", os.fspath(trace_path)],
    )

    smoke.main()

    assert capsys.readouterr() == ("shadow-trace-ok\n", "")
    rows = [json.loads(row) for row in _assert_private_file(trace_path).splitlines()]
    assert [row["kind"] for row in rows] == [
        "run_start",
        "action",
        "tool_result",
        "run_end",
    ]


@pytest.mark.skipif(os.name != "posix", reason="installed private-file smoke is POSIX-only")
def test_installed_smoke_exercises_both_atif_profiles_offline(
    tmp_path: Path,
) -> None:
    private_root = _make_private_directory(tmp_path / "atif-shadow")
    home = _make_private_directory(tmp_path / "home")
    config = _make_private_directory(tmp_path / "config")
    appdata = _make_private_directory(tmp_path / "appdata")
    environment = os.environ.copy()
    environment.update(
        {
            "APPDATA": os.fspath(appdata),
            "HOME": os.fspath(home),
            "PYTHONPATH": "",
            "XDG_CONFIG_HOME": os.fspath(config),
        }
    )
    for key in smoke._PROVIDER_CREDENTIAL_KEYS:
        environment[key] = f"installed-smoke-must-not-read-{key}"

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            os.fspath(SOCKET_GUARD),
            os.fspath(INSTALLED_SHADOW_SMOKE),
            "exercise-atif",
            os.fspath(private_root),
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "shadow-atif-installed-ok\n"
    assert completed.stderr == ""
    expected_profiles = {
        "codex": "harbor-codex/v1",
        "terminus": "harbor-terminus-2/v1",
    }
    for case_name, profile_id in expected_profiles.items():
        source_path = private_root / f"{case_name}.trajectory.json"
        report_path = private_root / f"{case_name}.report.json"
        command_path = private_root / f"{case_name}.command.json"
        source_bytes = _assert_private_file(source_path)
        report_bytes = _assert_private_file(report_path)
        command_bytes = _assert_private_file(command_path)
        report = decode_shadow_trace_report(report_bytes)
        command_report = json.loads(command_bytes)
        assert report.binding.adapter_profile_id == profile_id
        assert command_report["adapter_profile_id"] == profile_id
        assert command_report["report_digest"] == report.report_digest
        assert command_report["decision_authority"] is False
        assert report.shadow_report.decision_authority is False
        assert b"MUST_NOT_PERSIST" in source_bytes
        assert b"MUST_NOT_PERSIST" not in report_bytes
        assert b"MUST_NOT_PERSIST" not in command_bytes
        assert os.fspath(source_path).encode() not in report_bytes + command_bytes

    key_directory = config / "saliencegate"
    key_path = key_directory / "installation.key"
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(key_directory.stat().st_mode) == 0o700
    assert len(_assert_private_file(key_path)) == 32


@pytest.mark.parametrize(
    ("fixture_name", "profile_alias", "profile_id", "run_id"),
    (
        (
            "codex-minimal.trajectory.json",
            "harbor-codex-v1",
            "harbor-codex/v1",
            UUID("c0de0000-0000-4000-8000-000000000001"),
        ),
        (
            "terminus-minimal.trajectory.json",
            "harbor-terminus-2-v1",
            "harbor-terminus-2/v1",
            UUID("7e2a0000-0000-4000-8000-000000000001"),
        ),
    ),
)
def test_public_atif_fixture_report_and_summary_are_semantically_validated(
    fixture_name: str,
    profile_alias: str,
    profile_id: str,
    run_id: UUID,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = _make_private_directory(tmp_path / "public-atif")
    config_root = _make_private_directory(tmp_path / "config")
    monkeypatch.setenv("XDG_CONFIG_HOME", os.fspath(config_root))
    source_path = private_root / "source.trajectory.json"
    report_path = private_root / "report.json"
    command_path = private_root / "command.json"
    source_fixture = ROOT / "examples" / "atif-shadow" / fixture_name
    smoke._write_private(source_path, source_fixture.read_bytes())

    exit_code = cli_main(
        (
            "shadow",
            "analyze-atif",
            os.fspath(source_path),
            "--profile",
            profile_alias,
            "--run-id",
            str(run_id),
            "--working-directory",
            "/synthetic/workspace",
            "--environment-digest",
            smoke._ATIF_ENVIRONMENT_DIGEST,
            "--output",
            os.fspath(report_path),
            "--json",
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    smoke._write_private(command_path, captured.out.encode("utf-8"))

    expectation = smoke._PUBLIC_ATIF_EXPECTATIONS[profile_id]
    smoke._validate_atif_report(
        source_path=source_path,
        report_path=report_path,
        command_path=command_path,
        run_id=run_id,
        expectation=expectation,
    )

    monkeypatch.setattr(smoke, "_assert_core_only", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            os.fspath(INSTALLED_SHADOW_SMOKE),
            "validate-public-atif",
            os.fspath(source_path),
            os.fspath(report_path),
            os.fspath(command_path),
            str(run_id),
            profile_id,
        ],
    )
    smoke.main()
    assert capsys.readouterr() == ("shadow-atif-public-report-ok\n", "")

    tampered = json.loads(_assert_private_file(command_path))
    tampered["report_digest"] = "f" * 64
    if (
        tampered["report_digest"]
        == decode_shadow_trace_report(_assert_private_file(report_path)).report_digest
    ):
        tampered["report_digest"] = "0" * 64
    tampered_path = private_root / "tampered-command.json"
    smoke._write_private(tampered_path, canonical_json(tampered) + b"\n")
    with pytest.raises(RuntimeError, match="does not match the trace report"):
        smoke._validate_atif_report(
            source_path=source_path,
            report_path=report_path,
            command_path=tampered_path,
            run_id=run_id,
            expectation=expectation,
        )
