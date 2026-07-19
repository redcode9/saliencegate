from __future__ import annotations

import json
import socket
import stat
from pathlib import Path
from uuid import UUID

import pytest
from tests.cli.conftest import RunCli

import saliencegate.commands.shadow as shadow_module
from saliencegate.commands.shadow import (
    ShadowCommandConfigurationError,
    ShadowCommandInputError,
    ShadowCommandIntegrityError,
    run_shadow_analyze_atif,
)
from saliencegate.security import InstallationKey
from saliencegate.shadow import (
    ATIFProfile,
    ShadowEnvironmentBinding,
    ShadowInputError,
    ShadowSession,
    analyze_atif_bytes,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
)

RUN_ID = UUID("55555555-5555-4555-8555-555555555555")
ENVIRONMENT_DIGEST = "e" * 64
WORKING_DIRECTORY = "/synthetic/cli-default"
KEY = InstallationKey(b"k" * 32)
FIXTURES = Path("tests/fixtures/shadow/atif")


def _private_fixture(tmp_path: Path, name: str) -> Path:
    source = tmp_path / name
    source.write_bytes((FIXTURES / name).read_bytes())
    source.chmod(0o600)
    return source


def _environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
    )


def _command(
    source: Path,
    output: Path,
    *,
    profile: str = "harbor-codex-v1",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        "shadow",
        "analyze-atif",
        str(source),
        "--profile",
        profile,
        "--run-id",
        str(RUN_ID),
        "--working-directory",
        WORKING_DIRECTORY,
        "--environment-digest",
        ENVIRONMENT_DIGEST,
        "--output",
        str(output),
        *extra,
    )


@pytest.mark.parametrize(
    ("fixture_name", "profile_alias", "profile_id"),
    (
        (
            "codex-bundled-synthetic.trajectory.json",
            "harbor-codex-v1",
            "harbor-codex/v1",
        ),
        (
            "terminus-timeout-sanitized.trajectory.json",
            "harbor-terminus-2-v1",
            "harbor-terminus-2/v1",
        ),
    ),
)
def test_atif_cli_analyzes_each_explicit_profile_and_publishes_only_the_outer_report(
    run_cli: RunCli,
    tmp_path: Path,
    fixture_name: str,
    profile_alias: str,
    profile_id: str,
) -> None:
    source = _private_fixture(tmp_path, fixture_name)
    output = tmp_path / f"{profile_alias}.report.json"

    completed = run_cli(*_command(source, output, profile=profile_alias, extra=("--json",)))

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["schema_version"] == "shadow-atif-command-report/v1"
    assert summary["adapter_profile_id"] == profile_id
    assert "shadow_report" not in summary
    assert "rows" not in summary
    report = decode_shadow_trace_report(output.read_bytes())
    assert report.binding.adapter_profile_id == profile_id
    assert report.binding.capture_scope == "selected_events"
    assert report.report_digest == summary["report_digest"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert str(source) not in completed.stdout
    assert str(output) not in completed.stdout
    assert b"synthetic-codex-command" not in output.read_bytes()
    assert WORKING_DIRECTORY.encode() not in output.read_bytes()


def test_atif_cli_requires_the_documented_profile_alias_without_guessing(
    run_cli: RunCli,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"

    completed = run_cli(*_command(source, output, profile="harbor-codex/v1", extra=("--json",)))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: invalid command line\n"
    assert not output.exists()


def test_atif_cli_human_summary_is_complete_and_content_free(
    run_cli: RunCli,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"

    completed = run_cli(*_command(source, output))

    assert completed.returncode == 0, completed.stderr
    for label in (
        "scope: root_segment_only=true",
        "source totals:",
        "mapped: actions=",
        "ignored calls:",
        "ignored results:",
        "dispositions:",
        "profile detector evidence:",
        "evidence-sufficient applicable detector evaluations:",
        "producer authentication: none",
        "compatibility evidence manifest digest:",
        "evidence: descriptive observational; no decision authority",
    ):
        assert label in completed.stdout
    assert "synthetic-codex-command" not in completed.stdout
    assert str(source) not in completed.stdout
    assert str(output) not in completed.stdout


@pytest.mark.asyncio
async def test_atif_requires_a_stable_owner_private_source_before_key_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "public.trajectory.json"
    source.write_bytes((FIXTURES / "codex-bundled-synthetic.trajectory.json").read_bytes())
    source.chmod(0o644)
    output = tmp_path / "report.json"
    key_lookups = 0

    def load_key() -> InstallationKey:
        nonlocal key_lookups
        key_lookups += 1
        return KEY

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", load_key)

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze_atif(
            source,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            run_id=RUN_ID,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
            output_path=output,
        )

    assert key_lookups == 0
    assert not output.exists()


@pytest.mark.asyncio
async def test_atif_command_report_bytes_match_the_one_call_api_with_the_same_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    summary = await run_shadow_analyze_atif(
        source,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        output_path=output,
    )
    api_report = await analyze_atif_bytes(
        source.read_bytes(),
        run_id=RUN_ID,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=_environment(),
        installation_key=KEY,
    )

    assert output.read_bytes() == encode_shadow_trace_report(api_report)
    assert summary.report_digest == api_report.report_digest


@pytest.mark.asyncio
async def test_atif_command_never_uses_provider_credentials_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"
    credential = "sk-provider-credential-sentinel"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    monkeypatch.setenv("ANTHROPIC_API_KEY", credential)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    def block_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("ATIF analysis attempted network access")

    monkeypatch.setattr(socket, "socket", block_socket)

    summary = await run_shadow_analyze_atif(
        source,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        output_path=output,
    )

    assert summary.model_calls == 0
    assert credential not in output.read_text()


@pytest.mark.asyncio
async def test_invalid_atif_fails_before_key_database_or_output_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.trajectory.json"
    source.write_bytes(b"not-json")
    source.chmod(0o600)
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    key_lookups = 0

    def load_key() -> InstallationKey:
        nonlocal key_lookups
        key_lookups += 1
        return KEY

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", load_key)

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze_atif(
            source,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            run_id=RUN_ID,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
            output_path=output,
            repository_path=repository,
        )

    assert key_lookups == 0
    assert not output.exists()
    assert not repository.exists()
    assert not Path(f"{repository}-wal").exists()
    assert not Path(f"{repository}-shm").exists()


@pytest.mark.asyncio
async def test_trace_preflight_failure_does_not_materialize_the_lazy_sqlite_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    def fail_preflight(session: ShadowSession, _trace: object) -> object:
        assert session._repository is None
        assert not repository.exists()
        raise ShadowInputError()

    monkeypatch.setattr(shadow_module, "_prepare_analysis", fail_preflight)

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze_atif(
            source,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            run_id=RUN_ID,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
            output_path=output,
            repository_path=repository,
        )

    assert not repository.exists()
    assert not output.exists()


@pytest.mark.asyncio
async def test_atif_exact_replacement_succeeds_and_corruption_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    arguments = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        output_path=output,
    )

    await run_shadow_analyze_atif(source, **arguments)
    original = output.read_bytes()
    await run_shadow_analyze_atif(source, replace=True, **arguments)
    assert output.read_bytes() == original

    corrupt = b'{"corrupt":true}'
    output.write_bytes(corrupt)
    output.chmod(0o600)
    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze_atif(source, replace=True, **arguments)
    assert output.read_bytes() == corrupt


@pytest.mark.asyncio
async def test_mismatched_replacement_fails_before_a_new_sqlite_repository_is_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    output = tmp_path / "report.json"
    repository = tmp_path / "shadow.sqlite3"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        output_path=output,
    )

    await run_shadow_analyze_atif(
        source,
        environment_digest="f" * 64,
        **common,
    )
    existing = output.read_bytes()

    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze_atif(
            source,
            environment_digest=ENVIRONMENT_DIGEST,
            repository_path=repository,
            replace=True,
            **common,
        )

    assert output.read_bytes() == existing
    assert not repository.exists()
    assert not any(
        Path(f"{repository}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal")
    )


@pytest.mark.asyncio
async def test_resumed_report_cannot_create_a_new_database_before_exact_replacement_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    original_repository = tmp_path / "original.sqlite3"
    new_repository = tmp_path / "new.sqlite3"
    first_output = tmp_path / "first.json"
    resumed_output = tmp_path / "resumed.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
    )

    await run_shadow_analyze_atif(
        source,
        output_path=first_output,
        repository_path=original_repository,
        **common,
    )
    await run_shadow_analyze_atif(
        source,
        output_path=resumed_output,
        repository_path=original_repository,
        **common,
    )
    first = decode_shadow_trace_report(first_output.read_bytes())
    resumed = decode_shadow_trace_report(resumed_output.read_bytes())
    assert first.binding == resumed.binding
    assert first.report_digest != resumed.report_digest

    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze_atif(
            source,
            output_path=resumed_output,
            repository_path=new_repository,
            replace=True,
            **common,
        )

    assert not new_repository.exists()
    assert not any(
        Path(f"{new_repository}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal")
    )


@pytest.mark.asyncio
async def test_atif_sqlite_path_resumes_the_authenticated_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        repository_path=repository,
    )

    await run_shadow_analyze_atif(source, output_path=first_output, **common)
    await run_shadow_analyze_atif(source, output_path=second_output, **common)

    first = decode_shadow_trace_report(first_output.read_bytes())
    second = decode_shadow_trace_report(second_output.read_bytes())
    assert first.shadow_report.preexisting_event_count == 0
    assert (
        second.shadow_report.preexisting_event_count
        == second.shadow_report.unique_input_event_count
    )
    assert first.report_digest != second.report_digest
    assert stat.S_IMODE(repository.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_atif_rejects_source_output_and_sqlite_aliases_without_clobbering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    original = source.read_bytes()
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)

    for output, repository in (
        (source, Path(":memory:")),
        (tmp_path / "report.json", source),
    ):
        with pytest.raises(ShadowCommandInputError):
            await run_shadow_analyze_atif(
                source,
                profile=ATIFProfile.HARBOR_CODEX_V1,
                run_id=RUN_ID,
                working_directory=WORKING_DIRECTORY,
                environment_digest=ENVIRONMENT_DIGEST,
                output_path=output,
                repository_path=repository,
            )

    assert source.read_bytes() == original


@pytest.mark.asyncio
async def test_atif_revalidates_source_and_materialized_sqlite_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    original_revalidate = shadow_module._revalidate_atif_before_publication

    def weaken_bound_files(*args: object, **kwargs: object) -> object:
        source.chmod(0o644)
        repository.chmod(0o644)
        return original_revalidate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        shadow_module,
        "_revalidate_atif_before_publication",
        weaken_bound_files,
    )

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze_atif(
            source,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            run_id=RUN_ID,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
            output_path=output,
            repository_path=repository,
        )

    assert repository.exists()
    assert not output.exists()


@pytest.mark.asyncio
async def test_atif_pins_an_existing_sqlite_authorization_across_factory_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    first_output = tmp_path / "first.json"
    resumed_output = tmp_path / "resumed.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        repository_path=repository,
    )
    await run_shadow_analyze_atif(source, output_path=first_output, **common)
    await run_shadow_analyze_atif(source, output_path=resumed_output, **common)
    original_factory = ShadowSession._from_sqlite_authorization_for_trace

    def remove_after_inspection(
        _session_type: type[ShadowSession],
        authorization: object,
        **kwargs: object,
    ) -> ShadowSession:
        repository.unlink()
        return original_factory(authorization, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ShadowSession,
        "_from_sqlite_authorization_for_trace",
        classmethod(remove_after_inspection),
    )

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze_atif(
            source,
            output_path=resumed_output,
            replace=True,
            **common,
        )

    assert not repository.exists()
    assert resumed_output.exists()


@pytest.mark.asyncio
async def test_atif_pins_an_absent_sqlite_authorization_against_file_appearance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    attacker_repository = tmp_path / "attacker.sqlite3"
    repository = tmp_path / "shadow.sqlite3"
    attacker_output = tmp_path / "attacker-report.json"
    output = tmp_path / "report.json"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
    )
    await run_shadow_analyze_atif(
        source,
        output_path=attacker_output,
        repository_path=attacker_repository,
        **common,
    )
    await run_shadow_analyze_atif(source, output_path=output, **common)
    attacker_bytes = attacker_repository.read_bytes()
    original_factory = ShadowSession._from_sqlite_authorization_for_trace

    def create_after_inspection(
        _session_type: type[ShadowSession],
        authorization: object,
        **kwargs: object,
    ) -> ShadowSession:
        repository.write_bytes(attacker_bytes)
        repository.chmod(0o600)
        return original_factory(authorization, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ShadowSession,
        "_from_sqlite_authorization_for_trace",
        classmethod(create_after_inspection),
    )

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze_atif(
            source,
            output_path=output,
            repository_path=repository,
            replace=True,
            **common,
        )

    assert repository.read_bytes() == attacker_bytes
    assert output.exists()


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.asyncio
async def test_atif_pins_absent_sqlite_sidecars_against_appearance(
    suffix: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    raced_sidecar = Path(f"{repository}{suffix}")
    attacker_bytes = b"private-sidecar-that-must-not-be-authorized"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    await run_shadow_analyze_atif(
        source,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        output_path=output,
    )
    original_factory = ShadowSession._from_sqlite_authorization_for_trace

    def create_sidecar_after_inspection(
        _session_type: type[ShadowSession],
        authorization: object,
        **kwargs: object,
    ) -> ShadowSession:
        raced_sidecar.write_bytes(attacker_bytes)
        raced_sidecar.chmod(0o600)
        return original_factory(authorization, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ShadowSession,
        "_from_sqlite_authorization_for_trace",
        classmethod(create_sidecar_after_inspection),
    )

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze_atif(
            source,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            run_id=RUN_ID,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
            output_path=output,
            repository_path=repository,
            replace=True,
        )

    assert raced_sidecar.read_bytes() == attacker_bytes
    assert not repository.exists()
    assert output.exists()


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.asyncio
async def test_atif_pins_existing_sqlite_sidecars_against_replacement(
    suffix: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _private_fixture(tmp_path, "codex-bundled-synthetic.trajectory.json")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    sidecar = Path(f"{repository}{suffix}")
    displaced = tmp_path / f"authorized{suffix}"
    original_bytes = b"authorized-private-sidecar"
    replacement_bytes = b"replacement-private-sidecar"
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: KEY)
    common = dict(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        run_id=RUN_ID,
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
        output_path=output,
        repository_path=repository,
    )
    await run_shadow_analyze_atif(source, **common)
    sidecar.write_bytes(original_bytes)
    sidecar.chmod(0o600)
    original_factory = ShadowSession._from_sqlite_authorization_for_trace

    def replace_sidecar_after_inspection(
        _session_type: type[ShadowSession],
        authorization: object,
        **kwargs: object,
    ) -> ShadowSession:
        sidecar.rename(displaced)
        sidecar.write_bytes(replacement_bytes)
        sidecar.chmod(0o600)
        return original_factory(authorization, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ShadowSession,
        "_from_sqlite_authorization_for_trace",
        classmethod(replace_sidecar_after_inspection),
    )

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze_atif(source, replace=True, **common)

    assert displaced.read_bytes() == original_bytes
    assert sidecar.read_bytes() == replacement_bytes
    assert repository.exists()
    assert output.exists()
