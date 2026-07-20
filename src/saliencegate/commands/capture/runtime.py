"""Shared existing-state runtime for capture query commands."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from saliencegate.capture import (
    CaptureMigrationError,
    CaptureSpool,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    CaptureStore,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreLocations,
    CaptureStoreMode,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    resolve_capture_project,
)
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InstallationKey,
    InvalidInstallationKeyError,
    default_installation_key_path,
    load_installation_key,
)


@dataclass(frozen=True, slots=True, repr=False)
class CaptureCommandRuntime:
    project: Path
    installation_key: InstallationKey
    locations: CaptureStoreLocations
    store: CaptureStore
    spool: CaptureSpool | None

    def __repr__(self) -> str:
        return "CaptureCommandRuntime(<redacted>)"


def _home(environment: Mapping[str, str]) -> Path:
    configured = environment.get("HOME")
    return Path.home() if configured is None else Path(configured).expanduser()


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def open_capture_runtime(
    *,
    project: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
    drain: bool = True,
    busy_timeout_ms: int = 5_000,
) -> Iterator[CaptureCommandRuntime]:
    """Open an existing maintenance runtime, optionally draining its existing spool."""

    if type(drain) is not bool or type(busy_timeout_ms) is not int:
        raise CaptureCommandConfigurationError()
    environment = os.environ if environ is None else environ
    store: CaptureStore | None = None
    try:
        if not isinstance(environment, Mapping):
            raise TypeError
        resolved_project = resolve_capture_project(project)
        key_path = default_installation_key_path(environ=environment)
        locations = resolve_capture_store_locations(
            environ=environment,
            home=_home(environment),
        )
        try:
            key = load_installation_key(key_path)
        except FileNotFoundError:
            if _path_present(locations.database_path) or _path_present(locations.spool_directory):
                raise CaptureCommandIntegrityError() from None
            raise CaptureCommandUnavailableError() from None
        try:
            locations.database_path.lstat()
        except FileNotFoundError:
            raise CaptureCommandUnavailableError() from None
        store = CaptureStore.open(
            locations.database_path,
            installation_key=key,
            busy_timeout_ms=busy_timeout_ms,
            mode=CaptureStoreMode.MAINTENANCE,
        )
        spool: CaptureSpool | None = None
        try:
            locations.spool_directory.lstat()
        except FileNotFoundError:
            pass
        else:
            spool = CaptureSpool.open(locations, key)
            if drain:
                spool.drain(store)
        runtime = CaptureCommandRuntime(
            project=resolved_project,
            installation_key=key,
            locations=locations,
            store=store,
            spool=spool,
        )
    except CaptureCommandUnavailableError:
        raise
    except FileNotFoundError:
        raise CaptureCommandUnavailableError() from None
    except (
        CaptureMigrationError,
        CaptureSpoolIntegrityError,
        CaptureStoreIntegrityError,
        InsecureKeyFileError,
        InvalidInstallationKeyError,
    ):
        raise CaptureCommandIntegrityError() from None
    except (CaptureSpoolError, CaptureStoreError, InsecureKeyPathError, OSError, TypeError):
        raise CaptureCommandConfigurationError() from None

    try:
        yield runtime
    finally:
        try:
            runtime.store.close()
        except CaptureStoreError:
            raise CaptureCommandIntegrityError() from None


__all__ = ["CaptureCommandRuntime", "open_capture_runtime"]
