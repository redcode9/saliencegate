from __future__ import annotations

from pathlib import Path

import pytest

from saliencegate.capture.locations import (
    CaptureLocationError,
    CaptureStoreLocations,
    resolve_capture_store_locations,
)


class _PlatformString(str):
    pass


def _assert_content_free(error: CaptureLocationError) -> None:
    assert str(error) == "capture store location is invalid"
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize(
    "mutation",
    (
        "platform",
        "state-type",
        "database-type",
        "spool-type",
        "relative-state",
        "database-parent",
        "database-name",
        "spool-parent",
        "spool-name",
    ),
)
def test_direct_store_locations_reject_every_invalid_boundary(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = tmp_path / "state" / "saliencegate"
    values: dict[str, object] = {
        "platform": "posix",
        "state_directory": state,
        "database_path": state / "capture.sqlite3",
        "spool_directory": state / "capture-spool",
    }
    if mutation == "platform":
        values["platform"] = "plan9"
    elif mutation == "state-type":
        values["state_directory"] = str(state)
    elif mutation == "database-type":
        values["database_path"] = str(state / "capture.sqlite3")
    elif mutation == "spool-type":
        values["spool_directory"] = str(state / "capture-spool")
    elif mutation == "relative-state":
        values["state_directory"] = Path("relative-state")
    elif mutation == "database-parent":
        values["database_path"] = tmp_path / "other" / "capture.sqlite3"
    elif mutation == "database-name":
        values["database_path"] = state / "capture.db"
    elif mutation == "spool-parent":
        values["spool_directory"] = tmp_path / "other" / "capture-spool"
    else:
        values["spool_directory"] = state / "spool"

    with pytest.raises(CaptureLocationError) as captured:
        CaptureStoreLocations(**values)  # type: ignore[arg-type]

    _assert_content_free(captured.value)
    assert not state.exists()
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize(
    "invalid_argument",
    (
        "environ-list",
        "environ-object",
        "environment-value",
        "platform-bytes",
        "platform-subclass",
        "home-string",
        "home-object",
    ),
)
def test_location_resolution_rejects_wrong_runtime_types_without_side_effects(
    tmp_path: Path,
    invalid_argument: str,
) -> None:
    state_root = tmp_path / "provider-native-state"
    values: dict[str, object] = {
        "environ": {"XDG_STATE_HOME": str(state_root)},
        "home": tmp_path / "home",
        "platform": "posix",
    }
    if invalid_argument == "environ-list":
        values["environ"] = []
    elif invalid_argument == "environ-object":
        values["environ"] = object()
    elif invalid_argument == "environment-value":
        values["environ"] = {"XDG_STATE_HOME": 1}
    elif invalid_argument == "platform-bytes":
        values["platform"] = b"posix"
    elif invalid_argument == "platform-subclass":
        values["platform"] = _PlatformString("posix")
    elif invalid_argument == "home-string":
        values["home"] = str(tmp_path / "home")
    else:
        values["home"] = object()

    with pytest.raises(CaptureLocationError) as captured:
        resolve_capture_store_locations(**values)  # type: ignore[arg-type]

    _assert_content_free(captured.value)
    assert "provider-native-state" not in repr(captured.value)
    assert not state_root.exists()
    assert not (tmp_path / "home").exists()
