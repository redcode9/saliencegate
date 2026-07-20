from __future__ import annotations

import os
from pathlib import Path

import pytest

from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"c" * 32)


def test_project_resolution_is_canonical_side_effect_free_and_defaults_to_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    assert resolve_capture_project(None) == project.resolve(strict=True)
    assert resolve_capture_project(project) == project.resolve(strict=True)
    assert capture_project_digest(project, installation_key=KEY) == capture_project_digest(
        project,
        installation_key=KEY,
    )
    assert tuple(project.iterdir()) == ()


def test_project_digest_is_keyed_and_never_contains_the_operational_path(tmp_path: Path) -> None:
    project = tmp_path / "project-secret-name"
    project.mkdir()

    digest = capture_project_digest(project, installation_key=KEY)

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert os.fspath(project) not in digest
    assert digest != capture_project_digest(
        project,
        installation_key=InstallationKey(b"d" * 32),
    )


@pytest.mark.parametrize("value", ("missing", "", "\x00", b"project"))
def test_project_resolution_rejects_invalid_or_missing_targets(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(CaptureCommandInputError) as captured:
        resolve_capture_project(value, cwd=tmp_path)  # type: ignore[arg-type]

    assert "missing" not in str(captured.value)


def test_project_digest_rejects_invalid_keys_without_leaking_paths(tmp_path: Path) -> None:
    project = tmp_path / "fixture-secret-project"
    project.mkdir()

    with pytest.raises(CaptureCommandConfigurationError) as captured:
        capture_project_digest(project, installation_key=object())  # type: ignore[arg-type]

    assert "fixture-secret" not in str(captured.value)
