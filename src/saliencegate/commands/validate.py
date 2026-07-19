from __future__ import annotations

import errno
import json
import os
import re
import stat
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Never, cast

from saliencegate.artifacts import (
    AlgorithmArtifactValidationReport,
    AlgorithmSourceResultAssurance,
    ArtifactValidationCode,
    ArtifactValidationError,
    ArtifactValidationReport,
    validate_algorithm_artifact,
    validate_artifact,
)
from saliencegate.benchmarks.state_decay.runner import (
    SMOKE_MANIFEST_SCHEMA_VERSION,
    BenchmarkArtifactValidationError,
    BenchmarkValidationReport,
    validate_state_decay_artifact,
)
from saliencegate.domain import canonical_json

ValidationReport = (
    ArtifactValidationReport | AlgorithmArtifactValidationReport | BenchmarkValidationReport
)

_MAX_MANIFEST_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPLAY_SCHEMA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class _ArtifactRoute(StrEnum):
    REPLAY = "replay"
    ALGORITHM = "algorithm"
    BENCHMARK = "benchmark"


class ArtifactPathError(ValueError):
    """A value-free invalid artifact path error."""

    def __init__(self) -> None:
        super().__init__("artifact path is invalid")


def _manifest_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise TypeError
        path = Path(raw)
        metadata = path.lstat()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        raise ArtifactPathError() from None
    if stat.S_ISDIR(metadata.st_mode):
        return path / "manifest.json"
    if stat.S_ISREG(metadata.st_mode) and path.name == "manifest.json":
        return path
    raise ArtifactPathError()


def _validation_failure(code: ArtifactValidationCode) -> Never:
    raise ArtifactValidationError(code) from None


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_nlink,
        getattr(metadata, "st_uid", 0),
    )


def _read_manifest_preflight(path: Path) -> bytes:
    """Read only the bounded manifest discriminator without trusting filesystem links."""

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError:
        _validation_failure(ArtifactValidationCode.MISSING_COMPONENT)

    descriptor: int | None = None
    try:
        try:
            directory_before = os.fstat(directory_fd)
            named_directory_before = os.stat(path.parent, follow_symlinks=False)
        except OSError:
            _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_ISLNK(named_directory_before.st_mode)
            or _identity(directory_before) != _identity(named_directory_before)
        ):
            _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            _validation_failure(ArtifactValidationCode.MISSING_COMPONENT)
        except OSError as error:
            code = (
                ArtifactValidationCode.UNSAFE_COMPONENT
                if error.errno in (errno.ELOOP, errno.EMLINK, errno.ENXIO, errno.ENOTDIR)
                else ArtifactValidationCode.MISSING_COMPONENT
            )
            _validation_failure(code)

        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 2
            or before.st_size > _MAX_MANIFEST_BYTES
        ):
            _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)

        chunks: list[bytes] = []
        remaining = _MAX_MANIFEST_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)

        try:
            after = os.fstat(descriptor)
            named_file = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            directory_after = os.fstat(directory_fd)
            named_directory_after = os.stat(path.parent, follow_symlinks=False)
        except OSError:
            _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)
        if (
            len(data) > _MAX_MANIFEST_BYTES
            or len(data) != before.st_size
            or not stat.S_ISREG(named_file.st_mode)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named_file)
            or _identity(directory_before) != _identity(directory_after)
            or _identity(directory_before) != _identity(named_directory_after)
        ):
            _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)
        return data
    except ArtifactValidationError:
        raise
    except OSError:
        _validation_failure(ArtifactValidationCode.UNSAFE_COMPONENT)
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.close(directory_fd)


def _reject_nonfinite(value: str) -> Never:
    del value
    raise ValueError("non-finite JSON value")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_manifest_preflight(data: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
        if type(parsed) is not dict or canonical_json(parsed) != data:
            raise ValueError
    except Exception:
        _validation_failure(ArtifactValidationCode.INVALID_MANIFEST)
    return cast(dict[str, object], parsed)


def _manifest_route(path: Path) -> _ArtifactRoute:
    payload = _decode_manifest_preflight(_read_manifest_preflight(path))
    if "artifact_kind" in payload:
        kind = payload["artifact_kind"]
        if kind == "replay_run":
            return _ArtifactRoute.REPLAY
        if kind == "algorithm_run":
            return _ArtifactRoute.ALGORITHM
        _validation_failure(ArtifactValidationCode.INVALID_MANIFEST)

    version = payload.get("schema_version")
    # Legacy benchmark manifests predate the shared artifact-kind discriminator.
    if version == SMOKE_MANIFEST_SCHEMA_VERSION:
        return _ArtifactRoute.BENCHMARK
    # Replay v1 historically defaulted its kind during model validation when omitted.
    if type(version) is str and _REPLAY_SCHEMA_VERSION.fullmatch(version) is not None:
        return _ArtifactRoute.REPLAY
    _validation_failure(ArtifactValidationCode.INVALID_MANIFEST)


def _validate_expected_digest(value: str | None) -> None:
    if value is not None and (type(value) is not str or _SHA256.fullmatch(value) is None):
        _validation_failure(ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)


def _run_validate(
    artifact_path: str | os.PathLike[str],
    *,
    expected_digest: str | None = None,
    require_confirmatory: bool = False,
) -> ValidationReport:
    manifest = _manifest_path(artifact_path)
    _validate_expected_digest(expected_digest)
    route = _manifest_route(manifest)
    if route is _ArtifactRoute.REPLAY:
        return validate_artifact(
            manifest,
            expected_manifest_digest=expected_digest,
            require_confirmatory=require_confirmatory,
        )
    if route is _ArtifactRoute.ALGORITHM:
        algorithm_report = validate_algorithm_artifact(
            manifest,
            expected_manifest_digest=expected_digest,
        )
        if require_confirmatory:
            _validation_failure(ArtifactValidationCode.CONFIRMATORY_INELIGIBLE)
        return algorithm_report

    benchmark_failed = False
    try:
        report = validate_state_decay_artifact(
            manifest,
            expected_manifest_digest=expected_digest,
        )
    except BenchmarkArtifactValidationError:
        benchmark_failed = True
    if benchmark_failed:
        _validation_failure(ArtifactValidationCode.INVALID_MANIFEST)
    if require_confirmatory:
        _validation_failure(ArtifactValidationCode.CONFIRMATORY_INELIGIBLE)
    return report


def run_validate(
    artifact_path: str | os.PathLike[str],
    *,
    expected_digest: str | None = None,
    require_confirmatory: bool = False,
) -> ValidationReport:
    """Validate one supported artifact through a value-free public boundary."""

    path_failed = False
    failure: ArtifactValidationCode | None = None
    try:
        return _run_validate(
            artifact_path,
            expected_digest=expected_digest,
            require_confirmatory=require_confirmatory,
        )
    except ArtifactPathError:
        path_failed = True
    except ArtifactValidationError as error:
        failure = error.code
    if path_failed:
        raise ArtifactPathError()
    assert failure is not None
    raise ArtifactValidationError(failure)


def render_validate_json(report: ValidationReport) -> str:
    if type(report) is ArtifactValidationReport:
        validated: ValidationReport = ArtifactValidationReport.model_validate(report)
    elif type(report) is AlgorithmArtifactValidationReport:
        validated = AlgorithmArtifactValidationReport.model_validate(report)
    elif type(report) is BenchmarkValidationReport:
        validated = BenchmarkValidationReport.model_validate(report)
    else:
        raise ArtifactValidationError(ArtifactValidationCode.INVALID_MANIFEST)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_validate_human(report: ValidationReport) -> str:
    if type(report) is BenchmarkValidationReport:
        validated_benchmark = BenchmarkValidationReport.model_validate(report)
        return (
            "Benchmark artifact valid\n"
            "assurance: deterministic synthetic oracle\n"
            "confirmatory: no\n"
            "external claims: unsupported\n"
            f"scenarios: {validated_benchmark.scenario_count}\n"
            f"manifest digest: {validated_benchmark.manifest_digest}\n"
            f"content digest: {validated_benchmark.overall_content_digest}\n"
        )
    if type(report) is AlgorithmArtifactValidationReport:
        validated_algorithm = AlgorithmArtifactValidationReport.model_validate(report)
        source_assurance = {
            AlgorithmSourceResultAssurance.PRODUCER_ATTESTED: ("producer-attested digest only"),
            AlgorithmSourceResultAssurance.RECOMPUTED_FROM_RAW: (
                "recomputed from included synthetic raw result"
            ),
        }[validated_algorithm.source_result_assurance]
        return (
            "Algorithm artifact valid\n"
            f"source result assurance: {source_assurance}\n"
            "self-consistent: yes\n"
            "confirmatory: no\n"
            f"manifest digest: {validated_algorithm.manifest_digest}\n"
            f"content digest: {validated_algorithm.overall_content_digest}\n"
        )
    if type(report) is not ArtifactValidationReport:
        raise ArtifactValidationError(ArtifactValidationCode.INVALID_MANIFEST)
    validated = ArtifactValidationReport.model_validate(report)
    return (
        "Artifact valid\n"
        "grounding assurance: producer-attested digest-only grounding\n"
        f"confirmatory: {'yes' if validated.confirmatory else 'no'}\n"
        f"manifest digest: {validated.manifest_digest}\n"
        f"content digest: {validated.overall_content_digest}\n"
    )


__all__ = [
    "ArtifactPathError",
    "ValidationReport",
    "render_validate_human",
    "render_validate_json",
    "run_validate",
]
