from __future__ import annotations

import os
import sqlite3
import stat
import sys
from collections.abc import Mapping
from contextlib import suppress
from enum import StrEnum
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import canonical_json
from saliencegate.security import default_installation_key_path

DOCTOR_SCHEMA_VERSION: Literal["doctor/v1"] = "doctor/v1"
MINIMUM_PYTHON_VERSION = (3, 11, 0)
MAXIMUM_PYTHON_VERSION = (3, 14, 0)
MINIMUM_SQLITE_VERSION = (3, 24, 0)
MAX_ENDPOINT_LENGTH = 2_048
_PILOT_ENDPOINT_ERROR = "pilot endpoint is invalid"


class PilotEndpointError(ValueError):
    """A stable, value-free rejection of an unsafe local pilot endpoint."""

    def __init__(self) -> None:
        super().__init__(_PILOT_ENDPOINT_ERROR)


class DoctorCheckName(StrEnum):
    PYTHON = "python"
    SQLITE = "sqlite"
    FTS5 = "fts5"
    REPOSITORY_PATH = "repository_path"
    INSTALLATION_KEY = "installation_key"
    ENDPOINT = "endpoint"


class DoctorCheckStatus(StrEnum):
    PASS = "pass"
    SKIP = "skip"
    FAIL = "fail"


class DoctorSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DoctorReportStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


_CANONICAL_CHECK_ORDER = tuple(DoctorCheckName)
_DISPLAY_NAMES: dict[DoctorCheckName, str] = {
    DoctorCheckName.PYTHON: "Python runtime",
    DoctorCheckName.SQLITE: "SQLite runtime",
    DoctorCheckName.FTS5: "SQLite FTS5",
    DoctorCheckName.REPOSITORY_PATH: "Repository path",
    DoctorCheckName.INSTALLATION_KEY: "Installation key",
    DoctorCheckName.ENDPOINT: "Model endpoint",
}


class _DoctorModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class DoctorCheck(_DoctorModel):
    schema_version: Literal["doctor-check/v1"] = "doctor-check/v1"
    name: DoctorCheckName
    status: DoctorCheckStatus
    severity: DoctorSeverity
    required: bool
    message: Annotated[str, Field(min_length=1, max_length=512)]

    @field_validator("message")
    @classmethod
    def message_is_safe_for_terminal_rendering(cls, value: str) -> str:
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError("doctor message cannot contain control characters")
        return value

    @model_validator(mode="after")
    def status_matches_severity_and_requirement(self) -> Self:
        if self.status is DoctorCheckStatus.PASS:
            if self.severity is not DoctorSeverity.INFO:
                raise ValueError("passing check must have info severity")
        elif self.status is DoctorCheckStatus.SKIP:
            if self.required:
                raise ValueError("required check cannot be skipped")
            if self.severity is not DoctorSeverity.INFO:
                raise ValueError("skipped check must have info severity")
        else:
            expected = DoctorSeverity.ERROR if self.required else DoctorSeverity.WARNING
            if self.severity is not expected:
                raise ValueError("failed check severity does not match its requirement")
        return self


class DoctorReport(_DoctorModel):
    schema_version: Literal["doctor/v1"] = DOCTOR_SCHEMA_VERSION
    status: DoctorReportStatus
    ok: bool
    checks: Annotated[
        tuple[DoctorCheck, ...],
        Field(min_length=len(_CANONICAL_CHECK_ORDER), max_length=len(_CANONICAL_CHECK_ORDER)),
    ]

    @model_validator(mode="after")
    def summary_matches_checks(self) -> Self:
        names = tuple(check.name for check in self.checks)
        if names != _CANONICAL_CHECK_ORDER:
            raise ValueError("doctor checks must use the canonical order")

        required_failure = any(
            check.required and check.status is DoctorCheckStatus.FAIL for check in self.checks
        )
        optional_failure = any(
            not check.required and check.status is DoctorCheckStatus.FAIL for check in self.checks
        )
        if required_failure:
            expected_status = DoctorReportStatus.UNHEALTHY
        elif optional_failure:
            expected_status = DoctorReportStatus.DEGRADED
        else:
            expected_status = DoctorReportStatus.HEALTHY
        expected_ok = expected_status is not DoctorReportStatus.UNHEALTHY
        if self.status is not expected_status or self.ok is not expected_ok:
            raise ValueError("doctor report summary does not match its checks")
        return self


def _pass(name: DoctorCheckName, message: str, *, required: bool = True) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status=DoctorCheckStatus.PASS,
        severity=DoctorSeverity.INFO,
        required=required,
        message=message,
    )


def _skip(name: DoctorCheckName, message: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status=DoctorCheckStatus.SKIP,
        severity=DoctorSeverity.INFO,
        required=False,
        message=message,
    )


def _fail(name: DoctorCheckName, message: str, *, required: bool = True) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status=DoctorCheckStatus.FAIL,
        severity=DoctorSeverity.ERROR if required else DoctorSeverity.WARNING,
        required=required,
        message=message,
    )


def _version_text(version: tuple[int, int, int]) -> str:
    return ".".join(str(component) for component in version)


def _runtime_python_version() -> tuple[int, int, int]:
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def _check_python() -> DoctorCheck:
    try:
        version = _runtime_python_version()
    except Exception:
        return _fail(DoctorCheckName.PYTHON, "Python runtime version could not be determined.")
    observed = _version_text(version)
    if not MINIMUM_PYTHON_VERSION <= version < MAXIMUM_PYTHON_VERSION:
        return _fail(
            DoctorCheckName.PYTHON,
            f"Python {observed} is unsupported; use Python 3.11, 3.12, or 3.13.",
        )
    return _pass(DoctorCheckName.PYTHON, f"Python {observed} is supported.")


def _runtime_sqlite_version() -> tuple[int, int, int]:
    version = sqlite3.sqlite_version_info
    return (int(version[0]), int(version[1]), int(version[2]))


def _check_sqlite() -> DoctorCheck:
    try:
        version = _runtime_sqlite_version()
    except Exception:
        return _fail(DoctorCheckName.SQLITE, "SQLite runtime version could not be determined.")
    observed = _version_text(version)
    if version < MINIMUM_SQLITE_VERSION:
        minimum = _version_text(MINIMUM_SQLITE_VERSION)
        return _fail(
            DoctorCheckName.SQLITE,
            f"SQLite {observed} is unsupported; version {minimum} or newer is required.",
        )
    return _pass(DoctorCheckName.SQLITE, f"SQLite {observed} is supported.")


def _check_fts5() -> DoctorCheck:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE doctor_fts USING fts5(content)")
        connection.execute("INSERT INTO doctor_fts(content) VALUES ('available')")
        row = connection.execute(
            "SELECT count(*) FROM doctor_fts WHERE doctor_fts MATCH 'available'"
        ).fetchone()
        if row != (1,):
            raise sqlite3.OperationalError("FTS5 probe returned an unexpected result")
    except Exception:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        return _fail(DoctorCheckName.FTS5, "SQLite FTS5 is unavailable.")
    try:
        connection.close()
    except Exception:
        return _fail(DoctorCheckName.FTS5, "SQLite FTS5 probe could not be finalized.")
    return _pass(DoctorCheckName.FTS5, "SQLite FTS5 is available.")


def _path_access(path: Path, mode: int) -> bool:
    return os.access(path, mode)


def _nearest_existing_ancestor(path: Path) -> tuple[Path, os.stat_result]:
    current = path
    while True:
        try:
            return current, current.lstat()
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise
            current = parent


def _repository_target(repository_path: str | Path) -> tuple[Path | None, DoctorCheck | None]:
    try:
        raw_path = os.fspath(repository_path)
        if raw_path == ":memory:":
            return None, _pass(
                DoctorCheckName.REPOSITORY_PATH,
                "The in-memory repository target is available.",
            )
        target = Path(raw_path).expanduser()
        if not target.is_absolute():
            target = target.absolute()
        return target, None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository path configuration is invalid.",
        )


def _check_repository_path(repository_path: str | Path) -> DoctorCheck:
    target, immediate = _repository_target(repository_path)
    if immediate is not None:
        return immediate
    assert target is not None
    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        try:
            ancestor, _ = _nearest_existing_ancestor(target.parent)
        except OSError:
            return _fail(
                DoctorCheckName.REPOSITORY_PATH,
                "Repository path ancestry cannot be inspected.",
            )
        if not ancestor.is_dir():
            return _fail(
                DoctorCheckName.REPOSITORY_PATH,
                "Repository path has a non-directory ancestor.",
            )
        if not _path_access(ancestor, os.W_OK | os.X_OK):
            return _fail(
                DoctorCheckName.REPOSITORY_PATH,
                "Repository path cannot be created with the current permissions.",
            )
        return _pass(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository path can be created with the current permissions.",
        )
    except (OSError, ValueError):
        return _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository path cannot be inspected.",
        )

    if stat.S_ISLNK(target_stat.st_mode):
        return _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository path cannot be a symbolic link.",
        )
    if stat.S_ISDIR(target_stat.st_mode):
        if not _path_access(target, os.R_OK | os.W_OK | os.X_OK):
            return _fail(
                DoctorCheckName.REPOSITORY_PATH,
                "Repository directory must be readable, writable, and searchable.",
            )
        return _pass(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository directory permissions are usable.",
        )
    if not stat.S_ISREG(target_stat.st_mode):
        return _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository path must be a regular file or directory.",
        )
    if not _path_access(target, os.R_OK | os.W_OK):
        return _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository file must be readable and writable.",
        )
    if not _path_access(target.parent, os.W_OK | os.X_OK):
        return _fail(
            DoctorCheckName.REPOSITORY_PATH,
            "Repository parent must permit SQLite sidecar files.",
        )
    return _pass(DoctorCheckName.REPOSITORY_PATH, "Repository file permissions are usable.")


def _key_target(
    installation_key_path: Path | None,
    environ: Mapping[str, str] | None,
) -> tuple[Path | None, DoctorCheck | None]:
    try:
        target = (
            default_installation_key_path(environ=environ)
            if installation_key_path is None
            else Path(installation_key_path).expanduser()
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation-key path configuration is invalid.",
        )
    if not target.is_absolute():
        return None, _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation-key path must be absolute.",
        )
    return target, None


def _check_installation_key(
    installation_key_path: Path | None,
    environ: Mapping[str, str] | None,
) -> DoctorCheck:
    target, immediate = _key_target(installation_key_path, environ)
    if immediate is not None:
        return immediate
    assert target is not None
    try:
        key_stat = target.lstat()
    except FileNotFoundError:
        try:
            ancestor, _ = _nearest_existing_ancestor(target.parent)
        except OSError:
            return _fail(
                DoctorCheckName.INSTALLATION_KEY,
                "Installation-key path ancestry cannot be inspected.",
            )
        if not ancestor.is_dir() or not _path_access(ancestor, os.W_OK | os.X_OK):
            return _fail(
                DoctorCheckName.INSTALLATION_KEY,
                "Installation key cannot be created with the current permissions.",
            )
        return _pass(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation key is absent and can be created securely when first needed.",
        )
    except (OSError, ValueError):
        return _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation-key path cannot be inspected.",
        )

    if stat.S_ISLNK(key_stat.st_mode):
        return _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation key cannot be a symbolic link.",
        )
    if not stat.S_ISREG(key_stat.st_mode):
        return _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation key must be a regular file.",
        )
    if key_stat.st_size != 32:
        return _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation key must contain exactly 32 bytes.",
        )
    if os.name == "posix":
        if stat.S_IMODE(key_stat.st_mode) & 0o077:
            return _fail(
                DoctorCheckName.INSTALLATION_KEY,
                "Installation-key permissions must be owner-only.",
            )
        if hasattr(os, "getuid") and key_stat.st_uid != os.getuid():
            return _fail(
                DoctorCheckName.INSTALLATION_KEY,
                "Installation key must be owned by the current user.",
            )
    if not _path_access(target, os.R_OK):
        return _fail(
            DoctorCheckName.INSTALLATION_KEY,
            "Installation key is not readable by the current user.",
        )
    return _pass(
        DoctorCheckName.INSTALLATION_KEY,
        "Installation-key permissions and size are valid.",
    )


def _check_endpoint(endpoint: str | None) -> DoctorCheck:
    if endpoint is None:
        return _skip(
            DoctorCheckName.ENDPOINT,
            "No model endpoint is configured; deterministic offline commands do not require one.",
        )
    if (
        type(endpoint) is not str
        or not endpoint
        or len(endpoint) > MAX_ENDPOINT_LENGTH
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in endpoint)
    ):
        return _fail(
            DoctorCheckName.ENDPOINT,
            "Configured model endpoint is not a valid HTTP base URL.",
        )
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        return _fail(
            DoctorCheckName.ENDPOINT,
            "Configured model endpoint is not a valid HTTP base URL.",
        )
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and port <= 0)
    ):
        return _fail(
            DoctorCheckName.ENDPOINT,
            "Configured model endpoint is not a valid credential-free HTTP base URL.",
        )
    return _pass(
        DoctorCheckName.ENDPOINT,
        "Configured model endpoint is syntactically valid; connectivity was not attempted.",
    )


def _canonical_pilot_endpoint(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_ENDPOINT_LENGTH
        or not value.startswith("http://")
        or any(ord(character) <= 0x20 or ord(character) >= 0x7F for character in value)
        or any(character in value for character in ("%", "\\", "?", "#"))
    ):
        raise ValueError

    parsed = urlsplit(value)
    port = parsed.port
    host = parsed.hostname
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or host is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"/v1", "/v1/"}
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ValueError

    address = ip_address(host)
    if type(address) not in {IPv4Address, IPv6Address} or not address.is_loopback:
        raise ValueError
    authority = f"[{address.compressed}]" if type(address) is IPv6Address else str(address)
    if port is not None:
        authority = f"{authority}:{port}"
    canonical = f"http://{authority}/v1"
    if value not in {canonical, f"{canonical}/"}:
        raise ValueError
    return canonical


def validated_pilot_endpoint(value: object) -> str:
    """Return one canonical numeric-loopback OpenAI-compatible pilot base URL."""

    try:
        return _canonical_pilot_endpoint(value)
    except Exception:
        pass
    raise PilotEndpointError()


def run_doctor(
    *,
    repository_path: str | Path = Path("saliencegate.sqlite3"),
    installation_key_path: Path | None = None,
    endpoint: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Run deterministic environment checks without writing files or contacting a model."""

    checks = (
        _check_python(),
        _check_sqlite(),
        _check_fts5(),
        _check_repository_path(repository_path),
        _check_installation_key(installation_key_path, environ),
        _check_endpoint(endpoint),
    )
    required_failure = any(
        check.required and check.status is DoctorCheckStatus.FAIL for check in checks
    )
    optional_failure = any(
        not check.required and check.status is DoctorCheckStatus.FAIL for check in checks
    )
    if required_failure:
        status = DoctorReportStatus.UNHEALTHY
    elif optional_failure:
        status = DoctorReportStatus.DEGRADED
    else:
        status = DoctorReportStatus.HEALTHY
    return DoctorReport(
        status=status,
        ok=status is not DoctorReportStatus.UNHEALTHY,
        checks=checks,
    )


def render_doctor_json(report: DoctorReport) -> str:
    """Render one canonical machine record; the caller owns stdout."""

    validated = DoctorReport.model_validate(report)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_doctor_human(report: DoctorReport) -> str:
    """Render bounded terminal text; the caller owns stdout and stderr."""

    validated = DoctorReport.model_validate(report)
    lines = [f"SalienceGate doctor: {validated.status.value}"]
    for check in validated.checks:
        requirement = "required" if check.required else "optional"
        lines.append(
            f"[{check.status.value.upper()}] {_DISPLAY_NAMES[check.name]} "
            f"({requirement}): {check.message}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "DOCTOR_SCHEMA_VERSION",
    "DoctorCheck",
    "DoctorCheckName",
    "DoctorCheckStatus",
    "DoctorReport",
    "DoctorReportStatus",
    "DoctorSeverity",
    "PilotEndpointError",
    "render_doctor_human",
    "render_doctor_json",
    "run_doctor",
    "validated_pilot_endpoint",
]
