"""Small interactive front end for the content-free setup planner."""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from saliencegate.commands.capture.common import CaptureCommandInputError
from saliencegate.commands.setup import (
    SetupPlan,
    SetupScope,
    normalize_setup_providers,
)
from saliencegate.integrations.registry import ProviderAlias

ReadLine = Callable[[str], str]
WriteText = Callable[[str], None]


@dataclass(frozen=True, slots=True, repr=False)
class SetupWizardSelection:
    """Validated wizard choices without terminal output of the project path."""

    install_only: bool
    providers: tuple[ProviderAlias, ...]
    scope: SetupScope | None
    project: str | os.PathLike[str] | Path | None
    exclusions: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return "SetupWizardSelection(<redacted>)"


def collect_setup_wizard_selection(
    *,
    read_line: ReadLine = input,
    write_text: WriteText,
) -> SetupWizardSelection:
    """Collect install-only, project, or user-global setup choices."""

    try:
        if not callable(read_line) or not callable(write_text):
            raise CaptureCommandInputError()
        write_text(
            "SalienceGate setup\n"
            "1. Keep this installation without connecting a provider\n"
            "2. Connect providers to the current project\n"
            "3. Connect providers to a project selected manually\n"
            "4. Connect providers to all projects for this user\n"
        )
        mode = read_line("Select 1, 2, 3, or 4: ")
        if mode == "1":
            return SetupWizardSelection(
                install_only=True,
                providers=(),
                scope=None,
                project=None,
            )
        if mode not in ("2", "3", "4"):
            raise CaptureCommandInputError()
        raw_providers = read_line(
            "Providers (comma-separated codex, claude-code, opencode, pi; or all): "
        )
        provider_values = tuple(item.strip() for item in raw_providers.split(","))
        if not provider_values or any(not item for item in provider_values):
            raise CaptureCommandInputError()
        providers = normalize_setup_providers(provider_values)
        project: str | None = None
        exclusions: tuple[str, ...] = ()
        if mode == "3":
            project = read_line("Project path: ")
            if not project:
                raise CaptureCommandInputError()
        elif mode == "4":
            raw_exclusions = read_line("Projects to exclude (comma-separated paths, or blank): ")
            exclusions = tuple(item.strip() for item in raw_exclusions.split(",") if item.strip())
        return SetupWizardSelection(
            install_only=False,
            providers=providers,
            scope=(SetupScope.GLOBAL if mode == "4" else SetupScope.PROJECT),
            project=project,
            exclusions=exclusions,
        )
    except CaptureCommandInputError:
        raise
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        raise CaptureCommandInputError() from None


def confirm_setup_plan(
    plan: SetupPlan,
    *,
    read_line: ReadLine = input,
) -> bool:
    """Accept exactly the phrase bound to the already displayed plan."""

    try:
        checked = SetupPlan.model_validate(plan)
        if not callable(read_line):
            raise CaptureCommandInputError()
        supplied = read_line("Confirmation: ")
        if type(supplied) is not str:
            raise CaptureCommandInputError()
        return hmac.compare_digest(supplied, checked.confirmation_phrase)
    except CaptureCommandInputError:
        raise
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        raise CaptureCommandInputError() from None


__all__ = [
    "SetupWizardSelection",
    "collect_setup_wizard_selection",
    "confirm_setup_plan",
]
