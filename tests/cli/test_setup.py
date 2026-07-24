from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.cli.conftest import RunCli

from saliencegate import cli as cli_module
from saliencegate.commands import setup as setup_commands
from saliencegate.commands.capture import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
)
from saliencegate.commands.capture.connect import CaptureConnectReport
from saliencegate.commands.setup import (
    PreparedSetup,
    ProjectSetupHandler,
    SetupPlan,
    SetupProjectSelection,
    SetupProviderPlan,
    SetupProviderResult,
    SetupReport,
    SetupScope,
    SetupScopeRequest,
    SetupStatus,
    apply_setup,
    cancel_setup,
    normalize_setup_providers,
    planned_setup,
    prepare_setup,
    render_setup_json,
    render_setup_plan_human,
    render_setup_result_human,
    setup_confirmation_phrase,
)
from saliencegate.commands.setup_wizard import (
    collect_setup_wizard_selection,
    confirm_setup_plan,
)
from saliencegate.integrations.installation import (
    GitProjectFileDisposition,
    InstallationDisposition,
)
from saliencegate.integrations.registry import ProviderAlias


def _connect_report(provider: str, *, dry_run: bool) -> CaptureConnectReport:
    return CaptureConnectReport(
        provider=ProviderAlias(provider),
        disposition=(
            InstallationDisposition.PLANNED if dry_run else InstallationDisposition.INSTALLED
        ),
        dry_run=dry_run,
        capture_enabled=not dry_run,
        project_local_files=1,
        git_disposition=GitProjectFileDisposition.NOT_REPOSITORY,
        git_unignored_files=0,
        git_tracked_files=0,
    )


def test_project_setup_plans_every_provider_before_applying_in_canonical_order(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fixture-secret-project"
    project.mkdir()
    calls: list[tuple[str, Path, bool]] = []

    def connect_runner(
        *,
        provider: str,
        project: object,
        dry_run: bool = False,
    ) -> CaptureConnectReport:
        assert isinstance(project, Path)
        calls.append((provider, project, dry_run))
        return _connect_report(provider, dry_run=dry_run)

    prepared = prepare_setup(
        providers=("pi", "codex"),
        scope="project",
        project=project,
        project_handler=ProjectSetupHandler(connect_runner=connect_runner),
    )

    assert prepared.plan.scope is SetupScope.PROJECT
    assert prepared.plan.project_selection is SetupProjectSelection.MANUAL
    assert prepared.plan.providers == (ProviderAlias.CODEX, ProviderAlias.PI)
    assert prepared.plan.provider_trust_modified is False
    assert prepared.plan.confirmation_phrase == "apply project codex,pi"
    assert calls == [
        ("codex", project, True),
        ("pi", project, True),
    ]
    planned = planned_setup(prepared)
    encoded = render_setup_json(planned)
    human = render_setup_plan_human(prepared.plan)
    assert planned.status is SetupStatus.PLANNED
    assert str(project) not in encoded + human
    assert "fixture-secret" not in encoded + human
    assert "Provider trust changes: none" in human

    applied = apply_setup(prepared)

    assert applied.status is SetupStatus.APPLIED
    assert tuple(result.provider for result in applied.results) == (
        ProviderAlias.CODEX,
        ProviderAlias.PI,
    )
    assert calls == [
        ("codex", project, True),
        ("pi", project, True),
        ("codex", project, False),
        ("pi", project, False),
    ]
    assert render_setup_result_human(applied).endswith("Provider trust changes: none\n")


def test_current_project_selection_is_resolved_once_before_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    requests: list[SetupScopeRequest] = []

    @dataclass
    class Handler:
        def plan(self, request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
            requests.append(request)
            return (
                SetupProviderPlan(
                    provider=ProviderAlias.CODEX,
                    disposition=InstallationDisposition.PLANNED,
                    managed_files=1,
                    git_disposition=GitProjectFileDisposition.ALL_IGNORED,
                ),
            )

        def apply(self, request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
            requests.append(request)
            return (
                SetupProviderResult(
                    provider=ProviderAlias.CODEX,
                    disposition=InstallationDisposition.NOOP,
                    capture_enabled=True,
                    managed_files=1,
                ),
            )

    prepared = prepare_setup(
        providers=("codex",),
        project_handler=Handler(),
    )

    assert prepared.plan.project_selection is SetupProjectSelection.CURRENT
    assert requests[0].project == project
    assert requests[0].project_selection is SetupProjectSelection.CURRENT


def test_install_only_never_calls_a_scope_handler() -> None:
    @dataclass
    class FailingHandler:
        def plan(self, _request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
            raise AssertionError("install-only must not plan provider changes")

        def apply(self, _request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
            raise AssertionError("install-only must not apply provider changes")

    prepared = prepare_setup(
        install_only=True,
        project_handler=FailingHandler(),
        global_handler=FailingHandler(),
    )
    applied = apply_setup(prepared)

    assert prepared.plan.install_only is True
    assert prepared.plan.scope is None
    assert prepared.plan.providers == ()
    assert prepared.plan.confirmation_phrase == "apply install-only"
    assert applied.status is SetupStatus.APPLIED
    assert applied.results == ()
    assert render_setup_result_human(applied) == (
        "SalienceGate setup complete; no provider configuration was changed.\n"
    )
    assert cancel_setup(prepared).status is SetupStatus.CANCELLED


@dataclass
class _GlobalHandler:
    requests: list[tuple[str, SetupScopeRequest]] = field(default_factory=list)

    def plan(self, request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
        self.requests.append(("plan", request))
        return tuple(
            SetupProviderPlan(
                provider=provider,
                disposition=InstallationDisposition.PLANNED,
                managed_files=1,
            )
            for provider in request.providers
        )

    def apply(self, request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
        self.requests.append(("apply", request))
        return tuple(
            SetupProviderResult(
                provider=provider,
                disposition=InstallationDisposition.INSTALLED,
                capture_enabled=True,
                managed_files=1,
            )
            for provider in request.providers
        )


def test_global_scope_fails_closed_until_an_explicit_handler_is_wired() -> None:
    with pytest.raises(CaptureCommandUnavailableError):
        prepare_setup(providers=("codex",), scope="global")

    handler = _GlobalHandler()
    prepared = prepare_setup(
        providers=("pi", "codex"),
        scope=SetupScope.GLOBAL,
        global_handler=handler,
    )
    applied = apply_setup(prepared)

    assert prepared.plan.scope is SetupScope.GLOBAL
    assert prepared.plan.project_selection is None
    assert prepared.plan.confirmation_phrase == "apply global codex,pi"
    assert applied.status is SetupStatus.APPLIED
    assert [(action, request.scope, request.project) for action, request in handler.requests] == [
        ("plan", SetupScope.GLOBAL, None),
        ("apply", SetupScope.GLOBAL, None),
    ]


def test_scope_handlers_preserve_stable_capture_failures(tmp_path: Path) -> None:
    @dataclass
    class IntegrityPlanHandler:
        def plan(self, _request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
            raise CaptureCommandIntegrityError()

        def apply(self, _request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
            raise AssertionError("apply must not run after a failed plan")

    with pytest.raises(CaptureCommandIntegrityError):
        prepare_setup(
            providers=("codex",),
            project=tmp_path,
            project_handler=IntegrityPlanHandler(),
        )

    @dataclass
    class UnavailableApplyHandler:
        def plan(self, _request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
            return (
                SetupProviderPlan(
                    provider=ProviderAlias.CODEX,
                    disposition=InstallationDisposition.PLANNED,
                    managed_files=1,
                    git_disposition=GitProjectFileDisposition.NOT_REPOSITORY,
                ),
            )

        def apply(self, _request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
            raise CaptureCommandUnavailableError()

    prepared = prepare_setup(
        providers=("codex",),
        project=tmp_path,
        project_handler=UnavailableApplyHandler(),
    )
    with pytest.raises(CaptureCommandUnavailableError):
        apply_setup(prepared)


def _reader(
    values: tuple[str, ...],
    prompts: list[str],
) -> Callable[[str], str]:
    iterator: Iterator[str] = iter(values)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    return read


@pytest.mark.parametrize(
    ("answers", "install_only", "providers", "project"),
    (
        (("1",), True, (), None),
        (("2", "all"), False, tuple(ProviderAlias), None),
        (
            ("3", "pi, codex", "/synthetic/project"),
            False,
            (ProviderAlias.CODEX, ProviderAlias.PI),
            "/synthetic/project",
        ),
    ),
)
def test_wizard_collects_only_supported_local_modes(
    answers: tuple[str, ...],
    install_only: bool,
    providers: tuple[ProviderAlias, ...],
    project: str | None,
) -> None:
    prompts: list[str] = []
    output: list[str] = []

    selection = collect_setup_wizard_selection(
        read_line=_reader(answers, prompts),
        write_text=output.append,
    )

    assert selection.install_only is install_only
    assert selection.providers == providers
    assert selection.project == project
    assert selection.scope is (None if install_only else SetupScope.PROJECT)
    assert output and "global" not in output[0].lower()
    assert all("credential" not in value.lower() for value in (*prompts, *output))


def test_confirmation_requires_the_exact_displayed_phrase() -> None:
    plan = prepare_setup(install_only=True).plan

    assert confirm_setup_plan(plan, read_line=lambda _prompt: "apply install-only")
    assert not confirm_setup_plan(plan, read_line=lambda _prompt: "yes")
    assert not confirm_setup_plan(plan, read_line=lambda _prompt: "Apply install-only")


@pytest.mark.parametrize(
    "arguments",
    (
        ("setup", "--install-only", "--provider", "codex", "--yes"),
        ("setup", "--provider", "all", "--provider", "codex", "--dry-run"),
        ("setup", "--provider", "codex", "--scope", "global", "--project", "."),
        ("setup", "--yes"),
        ("setup", "--json"),
        ("setup", "--install-only", "--yes", "--dry-run"),
    ),
)
def test_setup_rejects_ambiguous_scripted_forms(
    arguments: tuple[str, ...],
    run_cli: RunCli,
) -> None:
    completed = run_cli(*arguments)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr in (
        "error: invalid command line\n",
        "error: capture command input is invalid\n",
    )


def test_setup_parser_exposes_scripted_project_and_future_global_scope() -> None:
    arguments = cli_module._parser().parse_args(
        (
            "setup",
            "--provider",
            "codex",
            "--provider",
            "pi",
            "--scope",
            "project",
            "--project",
            "/synthetic/project",
            "--confirm",
            "apply project codex,pi",
            "--json",
        )
    )

    assert vars(arguments) == {
        "command": "setup",
        "install_only": False,
        "provider": ["codex", "pi"],
        "scope": "project",
        "project": "/synthetic/project",
        "exclude": None,
        "dry_run": False,
        "yes": False,
        "confirm": "apply project codex,pi",
        "json": True,
    }
    global_arguments = cli_module._parser().parse_args(
        ("setup", "--provider", "all", "--scope", "global", "--dry-run")
    )
    assert global_arguments.scope == "global"


def test_install_only_entrypoint_is_idempotent_and_machine_safe(
    run_cli: RunCli,
) -> None:
    first = run_cli("setup", "--install-only", "--yes", "--json")
    second = run_cli(
        "setup",
        "--install-only",
        "--confirm",
        "apply install-only",
        "--json",
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload == {
        "plan": {
            "confirmation_phrase": "apply install-only",
            "install_only": True,
            "operations": [],
            "project_selection": None,
            "provider_trust_modified": False,
            "providers": [],
            "schema_version": "setup-plan/v1",
            "scope": None,
        },
        "results": [],
        "schema_version": "setup-report/v1",
        "status": "applied",
    }


def test_unwired_global_entrypoint_never_claims_success(run_cli: RunCli) -> None:
    completed = run_cli(
        "setup",
        "--provider",
        "codex",
        "--scope",
        "global",
        "--dry-run",
        "--json",
    )

    assert completed.returncode == 4
    assert completed.stdout == ""
    assert completed.stderr == "error: capture integration is unavailable\n"


@pytest.mark.skipif(
    os.name != "posix",
    reason="the deterministic fake provider executable is a POSIX script",
)
def test_global_all_installs_only_providers_with_a_root_and_available_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    codex_root = home / ".codex"
    stale_opencode_root = home / ".config" / "opencode"
    codex_root.mkdir(parents=True)
    stale_opencode_root.mkdir(parents=True)
    state = tmp_path / "state"
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    codex_executable.chmod(0o700)

    monkeypatch.setenv("HOME", os.fspath(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", os.fspath(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", os.fspath(state))
    monkeypatch.setenv("PATH", os.fspath(provider_bin))
    for name in (
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "OPENCODE_CONFIG_DIR",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    code = cli_module.main(("setup", "--provider", "all", "--scope", "global", "--yes", "--json"))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.SUCCESS
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["plan"]["providers"] == ["codex"]
    assert [result["provider"] for result in payload["results"]] == ["codex"]
    assert (codex_root / "config.toml").is_file()
    assert not (stale_opencode_root / "plugins" / "saliencegate.js").exists()
    assert not (home / ".claude").exists()
    assert not (home / ".pi").exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="the deterministic fake provider executable is a POSIX script",
)
def test_global_wizard_all_plans_and_installs_only_available_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    codex_root = home / ".codex"
    codex_root.mkdir(parents=True)
    state = tmp_path / "state"
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    codex_executable.chmod(0o700)

    monkeypatch.setenv("HOME", os.fspath(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", os.fspath(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", os.fspath(state))
    monkeypatch.setenv("PATH", os.fspath(provider_bin))
    for name in (
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "OPENCODE_CONFIG_DIR",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    answers = iter(("4", "all", "", "apply global codex"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    code = cli_module.main(("setup",))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.SUCCESS
    assert captured.err == ""
    assert "Providers: codex" in captured.out
    assert "Providers: codex,claude-code" not in captured.out
    assert (codex_root / "config.toml").is_file()
    assert not (home / ".claude").exists()
    assert not (home / ".config" / "opencode").exists()
    assert not (home / ".pi").exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="the deterministic isolated PATH is POSIX-specific",
)
def test_global_all_with_no_available_provider_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    state = tmp_path / "state"

    monkeypatch.setenv("HOME", os.fspath(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", os.fspath(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", os.fspath(state))
    monkeypatch.setenv("PATH", os.fspath(provider_bin))
    for name in (
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "OPENCODE_CONFIG_DIR",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    code = cli_module.main(("setup", "--provider", "all", "--scope", "global", "--yes", "--json"))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.UNAVAILABLE_DEPENDENCY
    assert captured.out == ""
    assert captured.err == "error: capture integration is unavailable\n"
    assert tuple(home.iterdir()) == ()
    assert not state.exists()


def test_default_wizard_applies_install_only_without_redundant_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(("1",))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    code = cli_module.main(("setup",))

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.SUCCESS
    assert captured.err == ""
    assert captured.out.index("SalienceGate setup plan") < captured.out.index(
        "SalienceGate setup complete"
    )
    assert "Provider trust changes: none" in captured.out


def _global_operation(provider: ProviderAlias = ProviderAlias.CODEX) -> SetupProviderPlan:
    return SetupProviderPlan(
        provider=provider,
        disposition=InstallationDisposition.PLANNED,
        managed_files=1,
    )


def _project_operation(
    *,
    git_disposition: GitProjectFileDisposition | None = GitProjectFileDisposition.NOT_REPOSITORY,
    git_unignored_files: int = 0,
    git_tracked_files: int = 0,
) -> SetupProviderPlan:
    return SetupProviderPlan(
        provider=ProviderAlias.CODEX,
        disposition=InstallationDisposition.PLANNED,
        managed_files=1,
        git_disposition=git_disposition,
        git_unignored_files=git_unignored_files,
        git_tracked_files=git_tracked_files,
    )


def _global_plan() -> SetupPlan:
    return SetupPlan(
        install_only=False,
        scope=SetupScope.GLOBAL,
        providers=(ProviderAlias.CODEX,),
        operations=(_global_operation(),),
        confirmation_phrase="apply global codex",
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"disposition": InstallationDisposition.INSTALLED},
        {"managed_files": 1, "git_unignored_files": 2},
        {"git_disposition": None, "git_unignored_files": 1},
        {"git_disposition": GitProjectFileDisposition.UNIGNORED},
        {
            "git_disposition": GitProjectFileDisposition.ALL_IGNORED,
            "git_unignored_files": 1,
        },
    ),
)
def test_setup_provider_plan_rejects_inconsistent_dry_run_contracts(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "provider": ProviderAlias.CODEX,
        "disposition": InstallationDisposition.PLANNED,
        "managed_files": 1,
        "git_disposition": None,
        "git_unignored_files": 0,
        "git_tracked_files": 0,
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        SetupProviderPlan.model_validate(values)

    assert (
        _project_operation(
            git_disposition=GitProjectFileDisposition.UNIGNORED,
            git_unignored_files=1,
        ).git_disposition
        is GitProjectFileDisposition.UNIGNORED
    )


@pytest.mark.parametrize(
    ("disposition", "capture_enabled"),
    (
        (InstallationDisposition.PLANNED, True),
        (InstallationDisposition.INSTALLED, False),
    ),
)
def test_setup_provider_result_requires_an_enabled_installation(
    disposition: InstallationDisposition,
    capture_enabled: bool,
) -> None:
    with pytest.raises(ValidationError):
        SetupProviderResult(
            provider=ProviderAlias.CODEX,
            disposition=disposition,
            capture_enabled=capture_enabled,
            managed_files=1,
        )


@pytest.mark.parametrize(
    "values",
    (
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "providers": (ProviderAlias.PI, ProviderAlias.CODEX),
            "operations": (
                _global_operation(ProviderAlias.PI),
                _global_operation(ProviderAlias.CODEX),
            ),
            "confirmation_phrase": "apply global pi,codex",
        },
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "providers": (ProviderAlias.CODEX,),
            "operations": (_global_operation(ProviderAlias.PI),),
            "confirmation_phrase": "apply global codex",
        },
        {
            "install_only": True,
            "scope": SetupScope.GLOBAL,
            "confirmation_phrase": "apply install-only",
        },
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "confirmation_phrase": "apply global none",
        },
        {
            "install_only": False,
            "scope": SetupScope.PROJECT,
            "providers": (ProviderAlias.CODEX,),
            "operations": (_project_operation(),),
            "confirmation_phrase": "apply project codex",
        },
        {
            "install_only": False,
            "scope": SetupScope.PROJECT,
            "project_selection": SetupProjectSelection.CURRENT,
            "providers": (ProviderAlias.CODEX,),
            "operations": (_global_operation(),),
            "confirmation_phrase": "apply project codex",
        },
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "project_selection": SetupProjectSelection.MANUAL,
            "providers": (ProviderAlias.CODEX,),
            "operations": (_global_operation(),),
            "confirmation_phrase": "apply global codex",
        },
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "providers": (ProviderAlias.CODEX,),
            "operations": (_global_operation(),),
            "confirmation_phrase": "apply wrong codex",
        },
    ),
)
def test_setup_plan_rejects_noncanonical_or_incomplete_operations(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SetupPlan.model_validate(values)


def test_setup_report_rejects_results_outside_an_applied_matching_plan() -> None:
    result = SetupProviderResult(
        provider=ProviderAlias.CODEX,
        disposition=InstallationDisposition.INSTALLED,
        capture_enabled=True,
        managed_files=1,
    )
    install_only = prepare_setup(install_only=True).plan
    global_plan = _global_plan()
    wrong_result = SetupProviderResult(
        provider=ProviderAlias.PI,
        disposition=InstallationDisposition.INSTALLED,
        capture_enabled=True,
        managed_files=1,
    )

    with pytest.raises(ValidationError):
        SetupReport(status=SetupStatus.PLANNED, plan=global_plan, results=(result,))
    with pytest.raises(ValidationError):
        SetupReport(status=SetupStatus.APPLIED, plan=install_only, results=(result,))
    with pytest.raises(ValidationError):
        SetupReport(status=SetupStatus.APPLIED, plan=global_plan, results=(wrong_result,))


@pytest.mark.parametrize(
    "values",
    (
        None,
        (),
        (object(),),
        ("all", "codex"),
        ("codex", "codex"),
        ("future",),
    ),
)
def test_provider_normalization_rejects_empty_ambiguous_or_unknown_values(
    values: object,
) -> None:
    with pytest.raises(CaptureCommandInputError):
        normalize_setup_providers(values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    (
        {
            "install_only": True,
            "scope": SetupScope.GLOBAL,
            "providers": (),
        },
        {
            "install_only": False,
            "scope": None,
            "providers": (ProviderAlias.CODEX,),
        },
        {
            "install_only": False,
            "scope": SetupScope.GLOBAL,
            "providers": (),
        },
    ),
)
def test_confirmation_phrase_rejects_inconsistent_selection(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        setup_confirmation_phrase(**arguments)  # type: ignore[arg-type]


def test_project_handler_rejects_non_project_requests() -> None:
    request = SetupScopeRequest(
        scope=SetupScope.GLOBAL,
        providers=(ProviderAlias.CODEX,),
        project=None,
        project_selection=None,
    )
    handler = ProjectSetupHandler()

    with pytest.raises(CaptureCommandConfigurationError):
        handler.plan(request)
    with pytest.raises(CaptureCommandConfigurationError):
        handler.apply(request)


def test_setup_boundaries_reject_malformed_prepared_state(tmp_path: Path) -> None:
    with pytest.raises(CaptureCommandInputError):
        prepare_setup(install_only=1)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        prepare_setup(install_only=True, providers=("codex",))
    with pytest.raises(CaptureCommandInputError):
        prepare_setup(
            providers=("codex",),
            scope=SetupScope.GLOBAL,
            project=tmp_path,
            global_handler=_GlobalHandler(),
        )
    with pytest.raises(CaptureCommandConfigurationError):
        planned_setup(object())  # type: ignore[arg-type]

    install_only = prepare_setup(install_only=True)
    malformed_install_only = PreparedSetup(
        plan=install_only.plan,
        request=None,
        handler=_GlobalHandler(),
    )
    with pytest.raises(CaptureCommandConfigurationError):
        apply_setup(malformed_install_only)

    global_plan = _global_plan()
    mismatched = PreparedSetup(
        plan=global_plan,
        request=SetupScopeRequest(
            scope=SetupScope.GLOBAL,
            providers=(ProviderAlias.PI,),
            project=None,
            project_selection=None,
        ),
        handler=_GlobalHandler(),
    )
    with pytest.raises(CaptureCommandConfigurationError):
        apply_setup(mismatched)


def test_setup_renderers_cover_all_terminal_and_git_review_states() -> None:
    prepared = prepare_setup(install_only=True)
    assert render_setup_result_human(planned_setup(prepared)).startswith("Dry-run only")
    assert render_setup_result_human(cancel_setup(prepared)).startswith("Setup cancelled")

    reviews = (
        setup_commands._render_provider_plan(
            _project_operation(
                git_disposition=GitProjectFileDisposition.UNIGNORED,
                git_unignored_files=1,
                git_tracked_files=1,
            )
        ),
        setup_commands._render_provider_plan(
            _project_operation(git_disposition=GitProjectFileDisposition.ALL_IGNORED)
        ),
        setup_commands._render_provider_plan(
            _project_operation(git_disposition=GitProjectFileDisposition.UNAVAILABLE)
        ),
    )
    assert "already tracked" in reviews[0]
    assert "ignored by Git" in reviews[1]
    assert "unavailable" in reviews[2]


def test_setup_conversion_helpers_reject_wrong_phase_and_provider_shape() -> None:
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._provider_plan(_connect_report("codex", dry_run=False))
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._provider_result(_connect_report("codex", dry_run=True))

    plan = _global_operation()
    result = SetupProviderResult(
        provider=ProviderAlias.CODEX,
        disposition=InstallationDisposition.INSTALLED,
        capture_enabled=True,
        managed_files=1,
    )
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._validated_provider_plans([plan], (ProviderAlias.CODEX,))
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._validated_provider_plans((plan,), (ProviderAlias.PI,))
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._validated_provider_results([result], (ProviderAlias.CODEX,))
    with pytest.raises(CaptureCommandConfigurationError):
        setup_commands._validated_provider_results((result,), (ProviderAlias.PI,))


def test_wizard_rejects_invalid_choices_and_preserves_terminal_interrupts() -> None:
    output: list[str] = []
    with pytest.raises(CaptureCommandInputError):
        collect_setup_wizard_selection(read_line=object(), write_text=output.append)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        collect_setup_wizard_selection(
            read_line=lambda _prompt: "9",
            write_text=output.append,
        )
    with pytest.raises(CaptureCommandInputError):
        collect_setup_wizard_selection(
            read_line=_reader(("2", ","), []),
            write_text=output.append,
        )
    with pytest.raises(CaptureCommandInputError):
        collect_setup_wizard_selection(
            read_line=_reader(("3", "codex", ""), []),
            write_text=output.append,
        )

    def invalid_reader(_prompt: str) -> str:
        raise RuntimeError

    with pytest.raises(CaptureCommandInputError):
        collect_setup_wizard_selection(read_line=invalid_reader, write_text=output.append)

    for exception in (EOFError, KeyboardInterrupt):

        def interrupted(_prompt: str, *, _exception: type[BaseException] = exception) -> str:
            raise _exception

        with pytest.raises(exception):
            collect_setup_wizard_selection(read_line=interrupted, write_text=output.append)


def test_wizard_confirmation_validates_reader_and_supplied_value() -> None:
    plan = prepare_setup(install_only=True).plan
    assert (
        repr(
            collect_setup_wizard_selection(
                read_line=lambda _prompt: "1",
                write_text=lambda _text: None,
            )
        )
        == "SetupWizardSelection(<redacted>)"
    )

    with pytest.raises(CaptureCommandInputError):
        confirm_setup_plan(plan, read_line=object())  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        confirm_setup_plan(plan, read_line=lambda _prompt: 1)  # type: ignore[return-value]

    def invalid_reader(_prompt: str) -> str:
        raise RuntimeError

    with pytest.raises(CaptureCommandInputError):
        confirm_setup_plan(plan, read_line=invalid_reader)

    for exception in (EOFError, KeyboardInterrupt):

        def interrupted(_prompt: str, *, _exception: type[BaseException] = exception) -> str:
            raise _exception

        with pytest.raises(exception):
            confirm_setup_plan(plan, read_line=interrupted)
