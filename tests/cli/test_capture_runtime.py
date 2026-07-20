from __future__ import annotations

from pathlib import Path

import pytest

from saliencegate.capture import initialize_capture_store, resolve_capture_store_locations
from saliencegate.commands.capture.common import CaptureCommandUnavailableError
from saliencegate.commands.capture.runtime import open_capture_runtime
from saliencegate.security import load_or_create_installation_key


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def test_open_runtime_uses_existing_key_and_store_without_exposing_paths(tmp_path: Path) -> None:
    project = tmp_path / "project-secret"
    project.mkdir()
    environment = _environment(tmp_path)
    key = load_or_create_installation_key(
        Path(environment["XDG_CONFIG_HOME"]) / "saliencegate" / "installation.key"
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.state_directory.mkdir(mode=0o700, parents=True)
    initialize_capture_store(locations.database_path)

    with open_capture_runtime(project=project, environ=environment, drain=False) as runtime:
        assert runtime.installation_key == key
        assert runtime.project == project.resolve(strict=True)
        assert runtime.spool is None
        assert repr(runtime) == "CaptureCommandRuntime(<redacted>)"
        assert "project-secret" not in repr(runtime)


def test_open_runtime_absence_is_side_effect_free(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = _environment(tmp_path)

    with (
        pytest.raises(CaptureCommandUnavailableError),
        open_capture_runtime(project=project, environ=environment),
    ):
        raise AssertionError("runtime must not be yielded")

    assert tuple(tmp_path.rglob("*")) == (project,)
