"""Authenticated project-bound passive capture report command."""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath

from saliencegate.capture import (
    CaptureNormalizationError,
    CaptureReportError,
    CaptureSessionReport,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
    build_capture_session_report,
    decode_capture_session_report,
    encode_capture_session_report,
    normalize_capture_session_snapshot,
    render_capture_session_report_human,
    render_capture_session_report_json,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.commands.capture.runtime import open_capture_runtime
from saliencegate.security import (
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
    authorize_atomic_file_publication,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsSecurityError,
)

MAX_CAPTURE_COMMAND_REPORT_BYTES = 4 * 1024 * 1024


def _absolute_report_output_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _publish_report_windows(
    path: str | os.PathLike[str],
    encoded: bytes,
    *,
    validate_replacement: Callable[[bytes], bool] | None,
    validate_published: Callable[[bytes], bool],
) -> bytes:
    windows_path = PureWindowsPath(os.fspath(path))
    if not windows_path.is_absolute():
        windows_path = PureWindowsPath(os.fspath(_absolute_report_output_path(path)))
    if not windows_path.is_absolute():
        raise WindowsSecurityError()
    operations = NativeWindowsSecurityOperations()
    reopened = operations.publish_private_file_in_managed_directory(
        windows_path,
        encoded,
        maximum_bytes=MAX_CAPTURE_COMMAND_REPORT_BYTES,
        validate_replacement=validate_replacement,
        validate_published=validate_published,
    )
    reopened.authorization.revalidate()
    return reopened.data


def _publish_report(
    path: str | os.PathLike[str],
    report: CaptureSessionReport,
    *,
    replace: bool,
) -> None:
    encoded = encode_capture_session_report(report)

    def validate_replacement(data: bytes) -> bool:
        try:
            existing = decode_capture_session_report(data)
        except CaptureReportError:
            return False
        return existing.session_id == report.session_id

    def validate_published(data: bytes) -> bool:
        try:
            return data == encoded and decode_capture_session_report(data) == report
        except CaptureReportError:
            return False

    try:
        absolute_path = _absolute_report_output_path(path)
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            reopened_data = _publish_report_windows(
                absolute_path,
                encoded,
                validate_replacement=validate_replacement if replace else None,
                validate_published=validate_published,
            )
            if reopened_data != encoded:
                raise CaptureCommandIntegrityError()
            return
        publication = authorize_atomic_file_publication(
            absolute_path,
            maximum_bytes=MAX_CAPTURE_COMMAND_REPORT_BYTES,
            validate_replacement=validate_replacement if replace else None,
        )
        reopened = publication.publish(encoded, validate_published=validate_published)
        if reopened.data != encoded:
            raise CaptureCommandIntegrityError()
    except CaptureCommandIntegrityError:
        raise
    except SecureFileUnsupportedError:
        raise CaptureCommandConfigurationError() from None
    except (
        SecureFileBoundError,
        SecureFileError,
        WindowsSecurityError,
        OSError,
        TypeError,
        ValueError,
    ):
        raise CaptureCommandInputError() from None


def run_capture_report(
    *,
    latest: bool,
    session_id: str | None = None,
    project: str | os.PathLike[str] | Path | None = None,
    output_path: str | os.PathLike[str] | None = None,
    replace: bool = False,
    environ: Mapping[str, str] | None = None,
) -> CaptureSessionReport:
    """Drain capture state and build one deterministic authenticated report."""

    if (
        type(latest) is not bool
        or type(replace) is not bool
        or latest == (session_id is not None)
        or (replace and output_path is None)
        or (session_id is not None and type(session_id) is not str)
    ):
        raise CaptureCommandInputError()
    resolved = resolve_capture_project(project)
    try:
        with open_capture_runtime(project=resolved, environ=environ, drain=latest) as runtime:
            project_id = capture_project_digest(
                runtime.project,
                installation_key=runtime.installation_key,
            )
            if latest:
                selected = runtime.store.latest_session(project_digest=project_id)
            else:
                assert session_id is not None
                selected = runtime.store.session_by_human_id(session_id)
                if not hmac.compare_digest(selected.project_digest, project_id):
                    raise CaptureStoreStateError()
                if runtime.spool is not None:
                    runtime.spool.drain(runtime.store)
            snapshot = runtime.store.snapshot_session(
                selected.connection_id,
                selected.session_id,
            )
            normalization = normalize_capture_session_snapshot(
                snapshot,
                installation_key=runtime.installation_key,
            )
            report = build_capture_session_report(
                snapshot,
                normalization,
                installation_key=runtime.installation_key,
                spool=runtime.spool,
            )
    except CaptureStoreStateError:
        raise CaptureCommandInputError() from None
    except (
        CaptureStoreIntegrityError,
        CaptureSpoolIntegrityError,
        CaptureNormalizationError,
        CaptureReportError,
    ):
        raise CaptureCommandIntegrityError() from None
    except (CaptureSpoolError, CaptureStoreError):
        raise CaptureCommandConfigurationError() from None
    if output_path is not None:
        _publish_report(output_path, report, replace=replace)
    return report


__all__ = [
    "MAX_CAPTURE_COMMAND_REPORT_BYTES",
    "render_capture_session_report_human",
    "render_capture_session_report_json",
    "run_capture_report",
]
