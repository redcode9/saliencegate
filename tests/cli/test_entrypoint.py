from __future__ import annotations

from tests.cli.conftest import RunCli

from saliencegate import __version__


def test_module_entrypoint_reports_version_without_diagnostics(run_cli: RunCli) -> None:
    completed = run_cli("--version")

    assert completed.returncode == 0
    assert completed.stdout == f"{__version__}\n"
    assert completed.stderr == ""


def test_module_entrypoint_missing_command_has_stable_value_free_error(run_cli: RunCli) -> None:
    completed = run_cli()

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: invalid command line\n"


def test_shadow_entrypoint_requires_the_exact_subcommand_and_flags(run_cli: RunCli) -> None:
    invalid_commands = (
        ("shadow",),
        ("shadow", "analyze"),
        ("shadow", "analyze", "trace", "--run-id", "not-a-uuid"),
        (
            "shadow",
            "analyze",
            "trace",
            "--run-id",
            "b35f05f3-555b-4f09-8996-a7b3693bb54a",
            "--output",
            "report",
            "--repo",
            ":memory:",
        ),
        (
            "shadow",
            "analyze",
            "trace",
            "--run-id",
            "b35f05f3-555b-4f09-8996-a7b3693bb54a",
            "--output",
            "report",
            "--model",
            "fixture-secret-model",
        ),
    )

    for arguments in invalid_commands:
        completed = run_cli(*arguments)
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: invalid command line\n"


def test_shadow_entrypoint_rejects_noncanonical_or_non_v4_run_ids(run_cli: RunCli) -> None:
    invalid_run_ids = (
        "b35f05f3555b4f098996a7b3693bb54a",
        "B35F05F3-555B-4F09-8996-A7B3693BB54A",
        "{b35f05f3-555b-4f09-8996-a7b3693bb54a}",
        "b35f05f3-555b-1f09-8996-a7b3693bb54a",
        " b35f05f3-555b-4f09-8996-a7b3693bb54a",
    )

    for run_id in invalid_run_ids:
        completed = run_cli(
            "shadow",
            "analyze",
            "fixture-secret-trace",
            "--run-id",
            run_id,
            "--output",
            "fixture-secret-output",
        )
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: shadow input or output is invalid\n"
        assert "fixture-secret" not in completed.stderr
