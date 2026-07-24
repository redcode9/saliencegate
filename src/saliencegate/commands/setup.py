"""Content-free setup planning around the existing capture lifecycle."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandError,
    CaptureCommandInputError,
    CaptureCommandUnavailableError,
    resolve_capture_project,
)
from saliencegate.commands.capture.connect import CaptureConnectReport, run_connect
from saliencegate.domain import canonical_json
from saliencegate.integrations.installation import (
    GitProjectFileDisposition,
    InstallationDisposition,
)
from saliencegate.integrations.registry import ProviderAlias


class SetupScope(StrEnum):
    """Supported setup scopes."""

    PROJECT = "project"
    GLOBAL = "global"


class SetupProjectSelection(StrEnum):
    """How one project was selected without reporting its path."""

    CURRENT = "current"
    MANUAL = "manual"


class SetupStatus(StrEnum):
    """Terminal state of one setup invocation."""

    PLANNED = "planned"
    APPLIED = "applied"
    CANCELLED = "cancelled"


class _SetupModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class SetupProviderPlan(_SetupModel):
    """One content-free provider operation approved by a scope handler."""

    provider: ProviderAlias
    disposition: InstallationDisposition
    managed_files: Annotated[int, Field(ge=0, le=16)]
    git_disposition: GitProjectFileDisposition | None = None
    git_unignored_files: Annotated[int, Field(ge=0, le=16)] = 0
    git_tracked_files: Annotated[int, Field(ge=0, le=16)] = 0

    @model_validator(mode="after")
    def counts_are_consistent(self) -> Self:
        if self.disposition is not InstallationDisposition.PLANNED:
            raise ValueError("setup provider plan is not a dry-run")
        if not (self.git_tracked_files <= self.git_unignored_files <= self.managed_files):
            raise ValueError("setup provider plan counts are inconsistent")
        if self.git_disposition is None:
            if self.git_unignored_files != 0 or self.git_tracked_files != 0:
                raise ValueError("setup provider plan Git counts are ambiguous")
        elif self.git_disposition is GitProjectFileDisposition.UNIGNORED:
            if self.git_unignored_files == 0:
                raise ValueError("setup provider plan Git disposition is inconsistent")
        elif self.git_unignored_files != 0 or self.git_tracked_files != 0:
            raise ValueError("setup provider plan Git disposition is inconsistent")
        return self


class SetupProviderResult(_SetupModel):
    """One applied provider operation without paths or connection identifiers."""

    provider: ProviderAlias
    disposition: InstallationDisposition
    capture_enabled: bool
    managed_files: Annotated[int, Field(ge=0, le=16)]

    @model_validator(mode="after")
    def result_is_enabled(self) -> Self:
        if (
            self.disposition
            not in (
                InstallationDisposition.INSTALLED,
                InstallationDisposition.NOOP,
                InstallationDisposition.UPGRADED,
                InstallationDisposition.RECOVERED,
            )
            or not self.capture_enabled
        ):
            raise ValueError("setup provider result is not enabled")
        return self


class SetupPlan(_SetupModel):
    """The complete bounded plan shown before setup mutation."""

    schema_version: Literal["setup-plan/v1"] = "setup-plan/v1"
    install_only: bool
    scope: SetupScope | None = None
    project_selection: SetupProjectSelection | None = None
    providers: Annotated[tuple[ProviderAlias, ...], Field(max_length=4)] = ()
    operations: Annotated[tuple[SetupProviderPlan, ...], Field(max_length=4)] = ()
    provider_trust_modified: Literal[False] = False
    confirmation_phrase: Annotated[str, Field(min_length=10, max_length=128)]

    @model_validator(mode="after")
    def plan_is_closed_and_canonical(self) -> Self:
        expected_providers = tuple(
            provider for provider in ProviderAlias if provider in set(self.providers)
        )
        if self.providers != expected_providers or len(set(self.providers)) != len(self.providers):
            raise ValueError("setup providers are not canonical")
        if tuple(operation.provider for operation in self.operations) != self.providers:
            raise ValueError("setup provider plans do not match the selection")
        if self.install_only:
            if (
                self.scope is not None
                or self.project_selection is not None
                or self.providers
                or self.operations
            ):
                raise ValueError("install-only setup has provider operations")
        elif (
            self.scope is None or not self.providers or len(self.operations) != len(self.providers)
        ):
            raise ValueError("setup plan is incomplete")
        elif self.scope is SetupScope.PROJECT:
            if self.project_selection is None:
                raise ValueError("project setup has no project selection")
            if any(operation.git_disposition is None for operation in self.operations):
                raise ValueError("project setup has no Git review")
        elif self.project_selection is not None:
            raise ValueError("non-project setup has a project selection")
        expected_confirmation = setup_confirmation_phrase(
            install_only=self.install_only,
            scope=self.scope,
            providers=self.providers,
        )
        if self.confirmation_phrase != expected_confirmation:
            raise ValueError("setup confirmation phrase is inconsistent")
        return self


class SetupReport(_SetupModel):
    """Machine-safe setup result containing its exact approved plan."""

    schema_version: Literal["setup-report/v1"] = "setup-report/v1"
    status: SetupStatus
    plan: SetupPlan
    results: Annotated[tuple[SetupProviderResult, ...], Field(max_length=4)] = ()

    @model_validator(mode="after")
    def report_matches_plan(self) -> Self:
        if self.status in (SetupStatus.PLANNED, SetupStatus.CANCELLED):
            if self.results:
                raise ValueError("non-applied setup has results")
        elif self.plan.install_only:
            if self.results:
                raise ValueError("install-only setup has provider results")
        elif tuple(result.provider for result in self.results) != self.plan.providers:
            raise ValueError("setup results do not match the plan")
        return self


@dataclass(frozen=True, slots=True, repr=False)
class SetupScopeRequest:
    """Private scope-handler input; project paths never enter public reports."""

    scope: SetupScope
    providers: tuple[ProviderAlias, ...]
    project: Path | None
    project_selection: SetupProjectSelection | None

    def __repr__(self) -> str:
        return "SetupScopeRequest(<redacted>)"


class SetupScopeHandler(Protocol):
    """Plan read-only work, then apply only that scope's provider operations."""

    def plan(self, request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]: ...

    def apply(self, request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]: ...


class SetupConnectRunner(Protocol):
    """Narrow existing-connect seam used by project setup and focused tests."""

    def __call__(
        self,
        *,
        provider: str,
        project: str | os.PathLike[str] | Path | None,
        dry_run: bool = False,
    ) -> CaptureConnectReport: ...


@dataclass(frozen=True, slots=True, repr=False)
class ProjectSetupHandler:
    """Project scope implemented strictly as existing connect dry-runs and connects."""

    connect_runner: SetupConnectRunner = run_connect

    def __repr__(self) -> str:
        return "ProjectSetupHandler(<redacted>)"

    def plan(self, request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
        if (
            request.scope is not SetupScope.PROJECT
            or request.project is None
            or request.project_selection is None
        ):
            raise CaptureCommandConfigurationError()
        reports = tuple(
            self.connect_runner(
                provider=provider.value,
                project=request.project,
                dry_run=True,
            )
            for provider in request.providers
        )
        return tuple(_provider_plan(report) for report in reports)

    def apply(self, request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
        if (
            request.scope is not SetupScope.PROJECT
            or request.project is None
            or request.project_selection is None
        ):
            raise CaptureCommandConfigurationError()
        reports = tuple(
            self.connect_runner(
                provider=provider.value,
                project=request.project,
                dry_run=False,
            )
            for provider in request.providers
        )
        return tuple(_provider_result(report) for report in reports)


@dataclass(frozen=True, slots=True, repr=False)
class PreparedSetup:
    """An exact plan plus the private handler input needed to apply it."""

    plan: SetupPlan
    request: SetupScopeRequest | None
    handler: SetupScopeHandler | None

    def __repr__(self) -> str:
        return "PreparedSetup(<redacted>)"


def normalize_setup_providers(
    values: Sequence[str | ProviderAlias] | None,
) -> tuple[ProviderAlias, ...]:
    """Normalize one explicit provider list, supporting `all` as a sole value."""

    try:
        selected = () if values is None else tuple(values)
        if not selected:
            raise CaptureCommandInputError()
        if any(type(value) not in (str, ProviderAlias) for value in selected):
            raise CaptureCommandInputError()
        raw = tuple(value.value if type(value) is ProviderAlias else value for value in selected)
        if raw == ("all",):
            return tuple(ProviderAlias)
        if "all" in raw:
            raise CaptureCommandInputError()
        providers = tuple(ProviderAlias(value) for value in raw)
        canonical = tuple(provider for provider in ProviderAlias if provider in set(providers))
        if len(canonical) != len(providers):
            raise CaptureCommandInputError()
        return canonical
    except CaptureCommandInputError:
        raise
    except (TypeError, ValueError):
        raise CaptureCommandInputError() from None


def setup_confirmation_phrase(
    *,
    install_only: bool,
    scope: SetupScope | None,
    providers: Sequence[ProviderAlias],
) -> str:
    """Build the exact content-free phrase used to approve one displayed plan."""

    if install_only:
        if scope is not None or tuple(providers):
            raise ValueError("install-only confirmation is inconsistent")
        return "apply install-only"
    if type(scope) is not SetupScope:
        raise ValueError("setup confirmation scope is invalid")
    selected = tuple(providers)
    if not selected:
        raise ValueError("setup confirmation providers are empty")
    return f"apply {scope.value} {','.join(provider.value for provider in selected)}"


def prepare_setup(
    *,
    install_only: bool = False,
    providers: Sequence[str | ProviderAlias] | None = None,
    scope: str | SetupScope | None = None,
    project: str | os.PathLike[str] | Path | None = None,
    project_handler: SetupScopeHandler | None = None,
    global_handler: SetupScopeHandler | None = None,
) -> PreparedSetup:
    """Build one read-only setup plan without creating capture state."""

    try:
        if type(install_only) is not bool:
            raise CaptureCommandInputError()
        selected_values = () if providers is None else tuple(providers)
        if install_only:
            if selected_values or scope is not None or project is not None:
                raise CaptureCommandInputError()
            plan = SetupPlan(
                install_only=True,
                confirmation_phrase=setup_confirmation_phrase(
                    install_only=True,
                    scope=None,
                    providers=(),
                ),
            )
            return PreparedSetup(plan=plan, request=None, handler=None)

        selected_providers = normalize_setup_providers(selected_values)
        selected_scope = SetupScope.PROJECT if scope is None else SetupScope(scope)
        if selected_scope is SetupScope.PROJECT:
            selected_project = resolve_capture_project(project)
            project_selection = (
                SetupProjectSelection.CURRENT if project is None else SetupProjectSelection.MANUAL
            )
            request = SetupScopeRequest(
                scope=selected_scope,
                providers=selected_providers,
                project=selected_project,
                project_selection=project_selection,
            )
            handler = ProjectSetupHandler() if project_handler is None else project_handler
        else:
            if project is not None:
                raise CaptureCommandInputError()
            if global_handler is None:
                raise CaptureCommandUnavailableError()
            request = SetupScopeRequest(
                scope=selected_scope,
                providers=selected_providers,
                project=None,
                project_selection=None,
            )
            handler = global_handler

        operations = _validated_provider_plans(handler.plan(request), selected_providers)
        plan = SetupPlan(
            install_only=False,
            scope=selected_scope,
            project_selection=request.project_selection,
            providers=selected_providers,
            operations=operations,
            confirmation_phrase=setup_confirmation_phrase(
                install_only=False,
                scope=selected_scope,
                providers=selected_providers,
            ),
        )
        return PreparedSetup(plan=plan, request=request, handler=handler)
    except CaptureCommandError:
        raise
    except (OSError, TypeError, ValueError):
        raise CaptureCommandInputError() from None
    except Exception:
        raise CaptureCommandConfigurationError() from None


def planned_setup(prepared: PreparedSetup) -> SetupReport:
    """Return a plan-only report without applying provider changes."""

    checked = _prepared_setup(prepared)
    return SetupReport(status=SetupStatus.PLANNED, plan=checked.plan)


def cancel_setup(prepared: PreparedSetup) -> SetupReport:
    """Return a cancellation report without applying provider changes."""

    checked = _prepared_setup(prepared)
    return SetupReport(status=SetupStatus.CANCELLED, plan=checked.plan)


def apply_setup(prepared: PreparedSetup) -> SetupReport:
    """Apply the exact prepared scope after the caller confirms its public plan."""

    checked = _prepared_setup(prepared)
    if checked.plan.install_only:
        return SetupReport(status=SetupStatus.APPLIED, plan=checked.plan)
    if checked.request is None or checked.handler is None:
        raise CaptureCommandConfigurationError()
    try:
        results = _validated_provider_results(
            checked.handler.apply(checked.request),
            checked.plan.providers,
        )
        return SetupReport(
            status=SetupStatus.APPLIED,
            plan=checked.plan,
            results=results,
        )
    except CaptureCommandError:
        raise
    except Exception:
        raise CaptureCommandConfigurationError() from None


def render_setup_json(report: SetupReport) -> str:
    checked = SetupReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_setup_plan_human(plan: SetupPlan) -> str:
    """Render a bounded plan before any provider mutation."""

    checked = SetupPlan.model_validate(plan)
    if checked.install_only:
        lines = [
            "SalienceGate setup plan",
            "Scope: install only",
            "Providers: none",
        ]
    else:
        selected_scope = checked.scope
        if selected_scope is None:
            raise CaptureCommandConfigurationError()
        project_suffix = (
            f" ({checked.project_selection.value} project)"
            if selected_scope is SetupScope.PROJECT and checked.project_selection is not None
            else ""
        )
        lines = [
            "SalienceGate setup plan",
            f"Scope: {selected_scope.value}{project_suffix}",
            "Providers: " + ", ".join(provider.value for provider in checked.providers),
        ]
        lines.extend(_render_provider_plan(operation) for operation in checked.operations)
    lines.extend(
        (
            "Provider trust changes: none",
            f'Confirmation: type exactly "{checked.confirmation_phrase}"',
        )
    )
    return "\n".join(lines) + "\n"


def render_setup_result_human(report: SetupReport) -> str:
    """Render only the terminal result after a plan was already displayed."""

    checked = SetupReport.model_validate(report)
    if checked.status is SetupStatus.PLANNED:
        return "Dry-run only; no setup changes were applied.\n"
    if checked.status is SetupStatus.CANCELLED:
        return "Setup cancelled; no setup changes were applied.\n"
    if checked.plan.install_only:
        return "SalienceGate setup complete; no provider configuration was changed.\n"
    lines = [
        "SalienceGate setup complete",
        *(
            f"{result.provider.value}: {result.disposition.value}; capture enabled."
            for result in checked.results
        ),
        "Provider trust changes: none",
    ]
    return "\n".join(lines) + "\n"


def _provider_plan(report: CaptureConnectReport) -> SetupProviderPlan:
    checked = CaptureConnectReport.model_validate(report)
    if not checked.dry_run or checked.capture_enabled:
        raise CaptureCommandConfigurationError()
    return SetupProviderPlan(
        provider=checked.provider,
        disposition=checked.disposition,
        managed_files=checked.project_local_files,
        git_disposition=checked.git_disposition,
        git_unignored_files=checked.git_unignored_files,
        git_tracked_files=checked.git_tracked_files,
    )


def _provider_result(report: CaptureConnectReport) -> SetupProviderResult:
    checked = CaptureConnectReport.model_validate(report)
    if checked.dry_run:
        raise CaptureCommandConfigurationError()
    return SetupProviderResult(
        provider=checked.provider,
        disposition=checked.disposition,
        capture_enabled=checked.capture_enabled,
        managed_files=checked.project_local_files,
    )


def _validated_provider_plans(
    values: object,
    providers: tuple[ProviderAlias, ...],
) -> tuple[SetupProviderPlan, ...]:
    if not isinstance(values, tuple):
        raise CaptureCommandConfigurationError()
    plans = tuple(SetupProviderPlan.model_validate(value) for value in values)
    if tuple(plan.provider for plan in plans) != providers:
        raise CaptureCommandConfigurationError()
    return plans


def _validated_provider_results(
    values: object,
    providers: tuple[ProviderAlias, ...],
) -> tuple[SetupProviderResult, ...]:
    if not isinstance(values, tuple):
        raise CaptureCommandConfigurationError()
    results = tuple(SetupProviderResult.model_validate(value) for value in values)
    if tuple(result.provider for result in results) != providers:
        raise CaptureCommandConfigurationError()
    return results


def _prepared_setup(value: object) -> PreparedSetup:
    if type(value) is not PreparedSetup:
        raise CaptureCommandConfigurationError()
    try:
        plan = SetupPlan.model_validate(value.plan)
    except Exception:
        raise CaptureCommandConfigurationError() from None
    if plan.install_only:
        if value.request is not None or value.handler is not None:
            raise CaptureCommandConfigurationError()
    elif (
        type(value.request) is not SetupScopeRequest
        or value.handler is None
        or value.request.scope is not plan.scope
        or value.request.providers != plan.providers
        or value.request.project_selection is not plan.project_selection
    ):
        raise CaptureCommandConfigurationError()
    return value


def _render_provider_plan(plan: SetupProviderPlan) -> str:
    if plan.git_disposition is GitProjectFileDisposition.UNIGNORED:
        review = (
            f"Git will surface {plan.git_unignored_files} "
            f"({plan.git_tracked_files} already tracked)"
        )
    elif plan.git_disposition is GitProjectFileDisposition.ALL_IGNORED:
        review = "all managed files are ignored by Git"
    elif plan.git_disposition is GitProjectFileDisposition.NOT_REPOSITORY:
        review = "no Git work tree was detected"
    elif plan.git_disposition is GitProjectFileDisposition.UNAVAILABLE:
        review = "Git visibility is unavailable"
    else:
        review = "no project Git review applies"
    return (
        f"- {plan.provider.value}: would connect; "
        f"{plan.managed_files} managed file(s); {review}; .gitignore unchanged."
    )


__all__ = [
    "PreparedSetup",
    "ProjectSetupHandler",
    "SetupPlan",
    "SetupProjectSelection",
    "SetupProviderPlan",
    "SetupProviderResult",
    "SetupReport",
    "SetupScope",
    "SetupScopeHandler",
    "SetupScopeRequest",
    "SetupStatus",
    "apply_setup",
    "cancel_setup",
    "normalize_setup_providers",
    "planned_setup",
    "prepare_setup",
    "render_setup_json",
    "render_setup_plan_human",
    "render_setup_result_human",
    "setup_confirmation_phrase",
]
