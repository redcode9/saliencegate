"""Cross-platform, side-effect-free capture store location resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


class CaptureLocationError(ValueError):
    """A capture store location could not be resolved safely."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture store location is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class CaptureStoreLocations:
    """Resolved capture paths without any filesystem mutation."""

    platform: Literal["posix", "windows"]
    state_directory: Path
    database_path: Path
    spool_directory: Path

    def __post_init__(self) -> None:
        if (
            self.platform not in ("posix", "windows")
            or not isinstance(self.state_directory, Path)
            or not isinstance(self.database_path, Path)
            or not isinstance(self.spool_directory, Path)
            or not self.state_directory.is_absolute()
            or self.database_path.parent != self.state_directory
            or self.spool_directory.parent != self.state_directory
            or self.database_path.name != "capture.sqlite3"
            or self.spool_directory.name != "capture-spool"
        ):
            raise CaptureLocationError()

    def __repr__(self) -> str:
        return "CaptureStoreLocations(<redacted>)"


def _exact_absolute_path(value: object) -> Path | None:
    if type(value) is not str and not isinstance(value, Path):
        return None
    try:
        path = Path(value)
    except (OSError, TypeError, ValueError):
        return None
    if not path.is_absolute() or ".." in path.parts or any("\x00" in part for part in path.parts):
        return None
    return path


def resolve_capture_store_locations(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> CaptureStoreLocations:
    """Resolve the isolated capture DB and spool without touching the filesystem."""

    result: CaptureStoreLocations | None = None
    failed = False
    try:
        environment = os.environ if environ is None else environ
        if not isinstance(environment, Mapping):
            raise TypeError
        selected_platform = (
            ("windows" if os.name == "nt" else "posix") if platform is None else platform
        )
        if type(selected_platform) is not str:
            raise TypeError
        explicit_home = Path.home() if home is None else home
        if not isinstance(explicit_home, Path):
            raise TypeError
        if selected_platform == "posix":
            configured = environment.get("XDG_STATE_HOME")
            if configured is not None:
                root = _exact_absolute_path(configured)
            else:
                home_path = _exact_absolute_path(explicit_home)
                root = None if home_path is None else home_path / ".local" / "state"
            if root is None:
                raise ValueError
            state_directory = root / "saliencegate"
        elif selected_platform == "windows":
            configured = environment.get("LOCALAPPDATA")
            root = _exact_absolute_path(configured)
            if root is None:
                raise ValueError
            state_directory = root / "SalienceGate"
        else:
            raise ValueError
        result = CaptureStoreLocations(
            platform=cast(Literal["posix", "windows"], selected_platform),
            state_directory=state_directory,
            database_path=state_directory / "capture.sqlite3",
            spool_directory=state_directory / "capture-spool",
        )
    except Exception:
        failed = True
    if failed or result is None:
        raise CaptureLocationError()
    return result


__all__ = [
    "CaptureLocationError",
    "CaptureStoreLocations",
    "resolve_capture_store_locations",
]
