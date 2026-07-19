from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Never

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from saliencegate.benchmarks.state_decay.diagnostic import (
    StateDecayDiagnosticError,
    StateDecayDiagnosticResult,
    evaluate_state_decay_scenarios,
    run_state_decay_diagnostic,
)
from saliencegate.benchmarks.state_decay.generator import (
    SMOKE_SEED,
    SmokeCoverageError,
    encode_scenarios_jsonl,
)
from saliencegate.benchmarks.state_decay.oracle import (
    ORACLE_RESULT_SCHEMA_VERSION,
    OracleResult,
)
from saliencegate.benchmarks.state_decay.schema import (
    GENERATOR_VERSION,
    InterventionLabel,
    StateDecayScenario,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256

CLI_BENCHMARK_SCHEMA_VERSION: Literal["cli-benchmark-report/v1"] = "cli-benchmark-report/v1"
SMOKE_MANIFEST_SCHEMA_VERSION: Literal["state-decay-smoke-manifest/v1"] = (
    "state-decay-smoke-manifest/v1"
)
SUITE_ID: Literal["state-decay-smoke"] = "state-decay-smoke"
SUITE_VERSION: Literal["v1"] = "v1"
ORACLE_VERSION: Literal["paired-continuation-oracle/v1"] = "paired-continuation-oracle/v1"
FIXTURE_NAME: Literal["smoke.jsonl"] = "smoke.jsonl"
MANIFEST_NAME: Literal["manifest.json"] = "manifest.json"

_EXPECTED_FILES = frozenset({MANIFEST_NAME, FIXTURE_NAME})
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_FIXTURE_BYTES = 8 * 1024 * 1024
_FIXTURE_DIGEST_DOMAIN = "saliencegate:state-decay:smoke-fixture:v1"
_ORACLE_DIGEST_DOMAIN = "saliencegate:state-decay:smoke-oracle:v1"
_CONTENT_DIGEST_DOMAIN = "saliencegate:state-decay:smoke-content:v1"
_MANIFEST_DIGEST_DOMAIN = "saliencegate:state-decay:smoke-manifest:v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

Sha256 = Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]


class BenchmarkCommandError(ValueError):
    """A value-free invalid benchmark input or destination error."""

    def __init__(self) -> None:
        super().__init__("benchmark input or output is invalid")


class BenchmarkArtifactValidationError(ValueError):
    """A value-free benchmark artifact integrity error."""

    def __init__(self) -> None:
        super().__init__("benchmark artifact validation failed")


def _fail_validation() -> Never:
    raise BenchmarkArtifactValidationError() from None


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _fixture_digest(data: bytes) -> str:
    return length_prefixed_sha256(data, domain=_FIXTURE_DIGEST_DOMAIN)


def _oracle_bytes(results: Sequence[OracleResult]) -> bytes:
    return b"".join(canonical_json(result) + b"\n" for result in results)


def _oracle_digest(results: Sequence[OracleResult]) -> str:
    return length_prefixed_sha256(_oracle_bytes(results), domain=_ORACLE_DIGEST_DOMAIN)


def _overall_content_digest(
    *, fixture_byte_count: int, fixture_digest: str, oracle_result_digest: str
) -> str:
    descriptor = {
        "fixture": {
            "byte_count": fixture_byte_count,
            "content_digest": fixture_digest,
            "path": FIXTURE_NAME,
            "record_count": 32,
        },
        "oracle": {
            "content_digest": oracle_result_digest,
            "record_count": 32,
            "source": "deterministic-reconstruction",
        },
    }
    return length_prefixed_sha256(canonical_json(descriptor), domain=_CONTENT_DIGEST_DOMAIN)


def _manifest_digest(values: Mapping[str, object]) -> str:
    payload = {key: value for key, value in values.items() if key != "manifest_digest"}
    return length_prefixed_sha256(canonical_json(payload), domain=_MANIFEST_DIGEST_DOMAIN)


class SmokeManifest(_StrictModel):
    schema_version: Literal["state-decay-smoke-manifest/v1"] = SMOKE_MANIFEST_SCHEMA_VERSION
    suite_id: Literal["state-decay-smoke"] = SUITE_ID
    suite_version: Literal["v1"] = SUITE_VERSION
    generator_version: Literal["v1"] = GENERATOR_VERSION
    oracle_version: Literal["paired-continuation-oracle/v1"] = ORACLE_VERSION
    oracle_result_schema_version: Literal["state-decay-oracle-result/v1"] = (
        ORACLE_RESULT_SCHEMA_VERSION
    )
    seed: Literal[20260711] = 20260711
    diagnostic: Literal[True] = True
    synthetic: Literal[True] = True
    balanced: Literal[True] = True
    external_claims_supported: Literal[False] = False
    external_claims_assessment: Literal["insufficient"] = "insufficient"
    scenario_count: Literal[32] = 32
    family_count: Literal[8] = 8
    intervene_count: Literal[16] = 16
    silence_count: Literal[16] = 16
    oracle_passed: Literal[32] = 32
    oracle_failed: Literal[0] = 0
    fixture_path: Literal["smoke.jsonl"] = FIXTURE_NAME
    fixture_byte_count: Annotated[int, Field(gt=0, le=_MAX_FIXTURE_BYTES)]
    fixture_digest: Sha256
    oracle_result_digest: Sha256
    overall_content_digest: Sha256
    manifest_digest: Sha256

    @model_validator(mode="after")
    def digests_are_self_attesting(self) -> SmokeManifest:
        expected_content = _overall_content_digest(
            fixture_byte_count=self.fixture_byte_count,
            fixture_digest=self.fixture_digest,
            oracle_result_digest=self.oracle_result_digest,
        )
        if self.overall_content_digest != expected_content:
            raise ValueError("benchmark content digest does not match")
        expected_manifest = _manifest_digest(
            self.model_dump(mode="json", exclude={"manifest_digest"}, warnings=False)
        )
        if self.manifest_digest != expected_manifest:
            raise ValueError("benchmark manifest digest does not match")
        return self


class BenchmarkCommandReport(_StrictModel):
    schema_version: Literal["cli-benchmark-report/v1"] = CLI_BENCHMARK_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    suite_id: Literal["state-decay-smoke"] = SUITE_ID
    suite_version: Literal["v1"] = SUITE_VERSION
    generator_version: Literal["v1"] = GENERATOR_VERSION
    seed: Literal[20260711] = 20260711
    diagnostic: Literal[True] = True
    synthetic: Literal[True] = True
    balanced: Literal[True] = True
    external_claims_supported: Literal[False] = False
    external_claims_assessment: Literal["insufficient"] = "insufficient"
    scenario_count: Literal[32] = 32
    family_count: Literal[8] = 8
    intervene_count: Literal[16] = 16
    silence_count: Literal[16] = 16
    oracle_passed: Literal[32] = 32
    oracle_failed: Literal[0] = 0
    fixture_digest: Sha256
    oracle_result_digest: Sha256
    overall_content_digest: Sha256
    manifest_digest: Sha256


class BenchmarkValidationReport(_StrictModel):
    schema_version: Literal["benchmark-validation-report/v1"] = "benchmark-validation-report/v1"
    valid: Literal[True] = True
    integrity_valid: Literal[True] = True
    structurally_valid: Literal[True] = True
    assurance: Literal["deterministic_synthetic_oracle"] = "deterministic_synthetic_oracle"
    confirmatory: Literal[False] = False
    external_claims_supported: Literal[False] = False
    expected_digest_matched: bool | None
    manifest_digest: Sha256
    overall_content_digest: Sha256
    fixture_digest: Sha256
    oracle_result_digest: Sha256
    scenario_count: Literal[32] = 32


def _compose_state_decay_manifest(
    diagnostic: StateDecayDiagnosticResult,
    fixture: bytes,
) -> SmokeManifest:
    values = diagnostic.scenarios
    results = diagnostic.oracle_results
    manifest: SmokeManifest | None = None
    try:
        if encode_scenarios_jsonl(values) != fixture:
            _fail_validation()
        fixture_digest = _fixture_digest(fixture)
        oracle_result_digest = _oracle_digest(results)
        overall_digest = _overall_content_digest(
            fixture_byte_count=len(fixture),
            fixture_digest=fixture_digest,
            oracle_result_digest=oracle_result_digest,
        )
        manifest_values: dict[str, object] = {
            "schema_version": SMOKE_MANIFEST_SCHEMA_VERSION,
            "suite_id": SUITE_ID,
            "suite_version": SUITE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "oracle_version": ORACLE_VERSION,
            "oracle_result_schema_version": ORACLE_RESULT_SCHEMA_VERSION,
            "seed": SMOKE_SEED,
            "diagnostic": True,
            "synthetic": True,
            "balanced": True,
            "external_claims_supported": False,
            "external_claims_assessment": "insufficient",
            "scenario_count": len(values),
            "family_count": len({scenario.family for scenario in values}),
            "intervene_count": sum(
                scenario.label is InterventionLabel.INTERVENE for scenario in values
            ),
            "silence_count": sum(
                scenario.label is InterventionLabel.SILENCE for scenario in values
            ),
            "oracle_passed": sum(result.matched for result in results),
            "oracle_failed": sum(not result.matched for result in results),
            "fixture_path": FIXTURE_NAME,
            "fixture_byte_count": len(fixture),
            "fixture_digest": fixture_digest,
            "oracle_result_digest": oracle_result_digest,
            "overall_content_digest": overall_digest,
        }
        manifest_values["manifest_digest"] = _manifest_digest(manifest_values)
        manifest = SmokeManifest.model_validate(manifest_values)
    except Exception:
        pass
    if manifest is None:
        _fail_validation()
    return manifest


def build_state_decay_manifest(
    scenarios: Sequence[StateDecayScenario], fixture: bytes
) -> SmokeManifest:
    """Build the native manifest only after regenerating every invariant."""

    if type(fixture) is not bytes or not fixture or len(fixture) > _MAX_FIXTURE_BYTES:
        _fail_validation()
    diagnostic: StateDecayDiagnosticResult | None = None
    with suppress(StateDecayDiagnosticError):
        diagnostic = evaluate_state_decay_scenarios(scenarios)
    if diagnostic is None:
        _fail_validation()
    return _compose_state_decay_manifest(diagnostic, fixture)


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int
    link_count: int
    owner: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
            mode=value.st_mode,
            link_count=value.st_nlink,
            owner=getattr(value, "st_uid", 0),
        )

    def matches(self, value: os.stat_result) -> bool:
        return self == type(self).from_stat(value)

    def matches_after_rename(self, value: os.stat_result) -> bool:
        current = type(self).from_stat(value)
        return (
            self.device == current.device
            and self.inode == current.inode
            and self.size == current.size
            and self.modified_ns == current.modified_ns
            and self.mode == current.mode
            and self.link_count == current.link_count
            and self.owner == current.owner
        )

    def payload(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "modified_ns": self.modified_ns,
            "changed_ns": self.changed_ns,
            "mode": self.mode,
            "link_count": self.link_count,
            "owner": self.owner,
        }

    @classmethod
    def from_payload(cls, value: object) -> _FileIdentity | None:
        fields = (
            "device",
            "inode",
            "size",
            "modified_ns",
            "changed_ns",
            "mode",
            "link_count",
            "owner",
        )
        if type(value) is not dict or set(value) != set(fields):
            return None
        if any(type(value.get(field)) is not int or value[field] < 0 for field in fields):
            return None
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True, slots=True)
class _ReadArtifact:
    manifest_bytes: bytes
    fixture_bytes: bytes
    directory_identity: _FileIdentity


def _read_regular_file(directory_fd: int, name: str, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    with suppress(OSError):
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    if descriptor is None:
        _fail_validation()
    data: bytes | None = None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 2
            or before.st_size > maximum
        ):
            _fail_validation()
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        current: os.stat_result | None = None
        with suppress(OSError):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            current is None
            or len(data) != before.st_size
            or len(data) > maximum
            or not _FileIdentity.from_stat(before).matches(after)
            or not stat.S_ISREG(current.st_mode)
            or not _FileIdentity.from_stat(before).matches(current)
        ):
            _fail_validation()
    except (BenchmarkArtifactValidationError, OSError):
        data = None
    finally:
        os.close(descriptor)
    if data is None:
        _fail_validation()
    return data


def _read_artifact(path: Path) -> _ReadArtifact:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    with suppress(OSError):
        descriptor = os.open(path, flags)
    if descriptor is None:
        _fail_validation()
    artifact: _ReadArtifact | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            _fail_validation()
        identity = _FileIdentity.from_stat(before)
        names = os.listdir(descriptor)
        if len(names) != 2 or set(names) != _EXPECTED_FILES:
            _fail_validation()
        manifest = _read_regular_file(descriptor, MANIFEST_NAME, maximum=_MAX_MANIFEST_BYTES)
        fixture = _read_regular_file(descriptor, FIXTURE_NAME, maximum=_MAX_FIXTURE_BYTES)
        after = os.fstat(descriptor)
        current: os.stat_result | None = None
        with suppress(OSError):
            current = path.lstat()
        if current is None or not identity.matches(after) or not identity.matches(current):
            _fail_validation()
        artifact = _ReadArtifact(manifest, fixture, identity)
    except (BenchmarkArtifactValidationError, OSError):
        artifact = None
    finally:
        os.close(descriptor)
    if artifact is None:
        _fail_validation()
    return artifact


def _reject_constant(value: str) -> Never:
    del value
    _fail_validation()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail_validation()
        result[key] = value
    return result


def _decode_canonical_object(data: bytes) -> dict[str, object]:
    parsed: object | None = None
    with suppress(
        BenchmarkArtifactValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        parsed = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    encoded: bytes | None = None
    if type(parsed) is dict:
        with suppress(Exception):
            encoded = canonical_json(parsed)
    if type(parsed) is not dict or encoded != data:
        _fail_validation()
    return parsed


def _decode_fixture(data: bytes) -> tuple[StateDecayScenario, ...]:
    if not data.endswith(b"\n"):
        _fail_validation()
    lines = data[:-1].split(b"\n")
    if len(lines) != 32 or any(not line for line in lines):
        _fail_validation()
    scenarios: list[StateDecayScenario] = []
    for line in lines:
        _decode_canonical_object(line)
        scenario: StateDecayScenario | None = None
        with suppress(Exception):
            scenario = StateDecayScenario.model_validate_json(line)
        if scenario is None:
            _fail_validation()
        if canonical_json(scenario) != line:
            _fail_validation()
        scenarios.append(scenario)
    return tuple(scenarios)


def _path(value: str | os.PathLike[str]) -> Path:
    if isinstance(value, bytes):
        raise BenchmarkCommandError() from None
    path: Path | None = None
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise TypeError
        path = Path(raw)
    except (OSError, TypeError, ValueError):
        pass
    if path is None or path.name in ("", ".", ".."):
        raise BenchmarkCommandError() from None
    return path


def _load_state_decay_artifact(
    artifact: str | os.PathLike[str],
    *,
    expected_manifest_digest: str | None = None,
) -> SmokeManifest:
    loaded: SmokeManifest | None = None
    try:
        if expected_manifest_digest is not None and (
            type(expected_manifest_digest) is not str
            or len(expected_manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_digest)
        ):
            _fail_validation()
        path = _path(artifact)
        if path.name == MANIFEST_NAME:
            selected = _lstat(path)
            if selected is None:
                _fail_validation()
            if stat.S_ISREG(selected.st_mode):
                path = path.parent
            elif not stat.S_ISDIR(selected.st_mode) or stat.S_ISLNK(selected.st_mode):
                _fail_validation()
        read = _read_artifact(path)
        payload = _decode_canonical_object(read.manifest_bytes)
        manifest = SmokeManifest.model_validate(payload)
        if canonical_json(manifest) != read.manifest_bytes:
            _fail_validation()
        scenarios = _decode_fixture(read.fixture_bytes)
        rebuilt = build_state_decay_manifest(scenarios, read.fixture_bytes)
        if rebuilt != manifest:
            _fail_validation()
        if (
            expected_manifest_digest is not None
            and manifest.manifest_digest != expected_manifest_digest
        ):
            _fail_validation()
        loaded = manifest
    except Exception:
        pass
    if loaded is None:
        _fail_validation()
    return loaded


def validate_state_decay_artifact(
    artifact: str | os.PathLike[str],
    *,
    expected_manifest_digest: str | None = None,
) -> BenchmarkValidationReport:
    """Validate the native artifact without claiming external or confirmatory evidence."""

    manifest = _load_state_decay_artifact(
        artifact,
        expected_manifest_digest=expected_manifest_digest,
    )
    return BenchmarkValidationReport(
        expected_digest_matched=(None if expected_manifest_digest is None else True),
        manifest_digest=manifest.manifest_digest,
        overall_content_digest=manifest.overall_content_digest,
        fixture_digest=manifest.fixture_digest,
        oracle_result_digest=manifest.oracle_result_digest,
    )


def _write_file(directory: Path, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / name, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


@contextmanager
def _destination_lock(destination: Path, parent: Path) -> Iterator[None]:
    lock_path = parent / f".{destination.name}.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    io_failed = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or (os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        ):
            raise BenchmarkCommandError() from None
        identity = _FileIdentity.from_stat(metadata)
        if not identity.matches(lock_path.lstat()):
            raise BenchmarkCommandError() from None
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        elif os.name == "nt":  # pragma: no cover - unavailable on POSIX test hosts
            import msvcrt

            if metadata.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
            identity = _FileIdentity.from_stat(os.fstat(descriptor))
        if not identity.matches(lock_path.lstat()):
            raise BenchmarkCommandError() from None
        yield
    except OSError:
        io_failed = True
    finally:
        if descriptor is not None:
            if os.name == "posix":
                with suppress(OSError):
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif os.name == "nt":  # pragma: no cover - unavailable on POSIX test hosts
                with suppress(OSError):
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(  # type: ignore[attr-defined]
                        descriptor,
                        msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                        1,
                    )
            with suppress(OSError):
                os.close(descriptor)
    if io_failed:
        raise BenchmarkCommandError() from None


def _looks_like_benchmark(path: Path) -> bool:
    try:
        return (path / MANIFEST_NAME).lstat() is not None
    except (FileNotFoundError, OSError):
        return False


def _remove_owned_tree(path: Path, identity: _FileIdentity) -> bool:
    current = _lstat(path)
    if current is None or not stat.S_ISDIR(current.st_mode) or not identity.matches(current):
        return False
    shutil.rmtree(path)
    return _lstat(path) is None


def _unlink_owned_file(path: Path, identity: _FileIdentity) -> bool:
    current = _lstat(path)
    if (
        current is None
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or not identity.matches(current)
    ):
        return False
    path.unlink()
    return _lstat(path) is None


@dataclass(frozen=True, slots=True)
class _ReplacementMarker:
    original: _FileIdentity
    replacement: _FileIdentity
    original_manifest_digest: str
    replacement_manifest_digest: str
    fixture_digest: str
    file_identity: _FileIdentity


def _replacement_paths(destination: Path, parent: Path) -> tuple[Path, Path]:
    return (
        parent / f".{destination.name}.backup",
        parent / f".{destination.name}.replace.json",
    )


def _replacement_marker_bytes(
    destination: Path,
    original: _FileIdentity,
    replacement: _FileIdentity,
    *,
    original_manifest_digest: str,
    replacement_manifest_digest: str,
    fixture_digest: str,
) -> bytes:
    return canonical_json(
        {
            "schema_version": "state-decay-replacement/v1",
            "destination_name": destination.name,
            "original": original.payload(),
            "replacement": replacement.payload(),
            "original_manifest_digest": original_manifest_digest,
            "replacement_manifest_digest": replacement_manifest_digest,
            "fixture_digest": fixture_digest,
        }
    )


def _read_replacement_marker(marker: Path, destination: Path) -> _ReplacementMarker | None:
    metadata = _lstat(marker)
    if metadata is None:
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or not 2 <= metadata.st_size <= 4096
    ):
        _fail_validation()
    identity = _FileIdentity.from_stat(metadata)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    before: os.stat_result | None = None
    after: os.stat_result | None = None
    data: bytes | None = None
    try:
        descriptor = os.open(marker, flags)
        try:
            before = os.fstat(descriptor)
            data = os.read(descriptor, 4097)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
    if (
        data is None
        or before is None
        or after is None
        or len(data) != metadata.st_size
        or len(data) > 4096
        or not identity.matches(before)
        or not identity.matches(after)
        or not identity.matches(marker.lstat())
    ):
        _fail_validation()
    payload = _decode_canonical_object(data)
    original = _FileIdentity.from_payload(payload.get("original"))
    replacement = _FileIdentity.from_payload(payload.get("replacement"))
    original_digest = payload.get("original_manifest_digest")
    replacement_digest = payload.get("replacement_manifest_digest")
    fixture_digest = payload.get("fixture_digest")
    if (
        set(payload)
        != {
            "schema_version",
            "destination_name",
            "original",
            "replacement",
            "original_manifest_digest",
            "replacement_manifest_digest",
            "fixture_digest",
        }
        or payload.get("schema_version") != "state-decay-replacement/v1"
        or payload.get("destination_name") != destination.name
        or original is None
        or replacement is None
        or not stat.S_ISDIR(original.mode)
        or not stat.S_ISDIR(replacement.mode)
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (original_digest, replacement_digest, fixture_digest)
        )
    ):
        _fail_validation()
    assert isinstance(original_digest, str)
    assert isinstance(replacement_digest, str)
    assert isinstance(fixture_digest, str)
    return _ReplacementMarker(
        original=original,
        replacement=replacement,
        original_manifest_digest=original_digest,
        replacement_manifest_digest=replacement_digest,
        fixture_digest=fixture_digest,
        file_identity=identity,
    )


def _matches_artifact(
    path: Path,
    *,
    manifest_digest: str,
    fixture_digest: str,
) -> bool:
    try:
        manifest = _load_state_decay_artifact(path, expected_manifest_digest=manifest_digest)
        return manifest.fixture_digest == fixture_digest
    except BenchmarkArtifactValidationError:
        return False


def _recover_replacement(destination: Path, parent: Path) -> None:
    backup, marker_path = _replacement_paths(destination, parent)
    backup_metadata = _lstat(backup)
    marker = _read_replacement_marker(marker_path, destination)
    destination_metadata = _lstat(destination)
    if backup_metadata is None and marker is None:
        return
    if marker is None:
        _fail_validation()

    original_at_destination = (
        destination_metadata is not None
        and marker.original.matches_after_rename(destination_metadata)
        and _matches_artifact(
            destination,
            manifest_digest=marker.original_manifest_digest,
            fixture_digest=marker.fixture_digest,
        )
    )
    replacement_at_destination = (
        destination_metadata is not None
        and marker.replacement.matches_after_rename(destination_metadata)
        and _matches_artifact(
            destination,
            manifest_digest=marker.replacement_manifest_digest,
            fixture_digest=marker.fixture_digest,
        )
    )
    if backup_metadata is None:
        if not (original_at_destination or replacement_at_destination):
            _fail_validation()
        if not _unlink_owned_file(marker_path, marker.file_identity):
            _fail_validation()
        _fsync_directory(parent)
        return

    if not marker.original.matches_after_rename(backup_metadata) or not _matches_artifact(
        backup,
        manifest_digest=marker.original_manifest_digest,
        fixture_digest=marker.fixture_digest,
    ):
        _fail_validation()
    backup_identity = _FileIdentity.from_stat(backup_metadata)
    if destination_metadata is None:
        os.replace(backup, destination)
        restored = destination.lstat()
        if (
            not marker.original.matches_after_rename(restored)
            or not _matches_artifact(
                destination,
                manifest_digest=marker.original_manifest_digest,
                fixture_digest=marker.fixture_digest,
            )
            or not _unlink_owned_file(marker_path, marker.file_identity)
        ):
            _fail_validation()
        _fsync_directory(parent)
        return
    if not replacement_at_destination:
        _fail_validation()
    if not _remove_owned_tree(backup, backup_identity):
        _fail_validation()
    if not _unlink_owned_file(marker_path, marker.file_identity):
        _fail_validation()
    _fsync_directory(parent)


def _publish_locked(
    destination: Path,
    parent: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool,
    manifest: SmokeManifest,
) -> None:
    _recover_replacement(destination, parent)
    existing = _lstat(destination)
    existing_identity: _FileIdentity | None = None
    existing_manifest: SmokeManifest | None = None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
            raise BenchmarkCommandError() from None
        if not replace:
            raise BenchmarkCommandError() from None
        try:
            previous = _load_state_decay_artifact(destination)
        except BenchmarkArtifactValidationError:
            if _looks_like_benchmark(destination):
                raise
            raise BenchmarkCommandError() from None
        if (
            previous.suite_id != manifest.suite_id
            or previous.suite_version != manifest.suite_version
            or previous.fixture_digest != manifest.fixture_digest
        ):
            raise BenchmarkCommandError() from None
        existing = destination.lstat()
        existing_identity = _FileIdentity.from_stat(existing)
        existing_manifest = previous

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    staging_identity = _FileIdentity.from_stat(staging.lstat())
    backup, marker_path = _replacement_paths(destination, parent)
    backup_identity: _FileIdentity | None = None
    marker_identity: _FileIdentity | None = None
    published = False
    io_failed = False
    try:
        if _lstat(backup) is not None or _lstat(marker_path) is not None:
            raise BenchmarkCommandError() from None
        for name in sorted(files):
            _write_file(staging, name, files[name])
        _fsync_directory(staging)
        staging_identity = _FileIdentity.from_stat(staging.lstat())
        _load_state_decay_artifact(staging, expected_manifest_digest=manifest.manifest_digest)

        if existing_identity is not None:
            assert existing_manifest is not None
            current = destination.lstat()
            if not existing_identity.matches(current):
                raise BenchmarkCommandError() from None
            previous = _load_state_decay_artifact(destination)
            if previous.fixture_digest != manifest.fixture_digest:
                raise BenchmarkCommandError() from None
            _write_file(
                parent,
                marker_path.name,
                _replacement_marker_bytes(
                    destination,
                    existing_identity,
                    staging_identity,
                    original_manifest_digest=existing_manifest.manifest_digest,
                    replacement_manifest_digest=manifest.manifest_digest,
                    fixture_digest=manifest.fixture_digest,
                ),
            )
            marker_identity = _FileIdentity.from_stat(marker_path.lstat())
            _fsync_directory(parent)
            os.replace(destination, backup)
            moved = backup.lstat()
            if not existing_identity.matches_after_rename(moved):
                raise BenchmarkCommandError() from None
            backup_identity = _FileIdentity.from_stat(moved)
            _fsync_directory(parent)

        try:
            current_staging = staging.lstat()
            if not staging_identity.matches(current_staging):
                raise BenchmarkCommandError() from None
            os.replace(staging, destination)
            moved_staging = destination.lstat()
            if not staging_identity.matches_after_rename(moved_staging):
                raise BenchmarkCommandError() from None
            _load_state_decay_artifact(
                destination, expected_manifest_digest=manifest.manifest_digest
            )
            published = True
            _fsync_directory(parent)
        except Exception:
            if marker_identity is not None:
                with suppress(
                    BenchmarkArtifactValidationError,
                    BenchmarkCommandError,
                    OSError,
                ):
                    _recover_replacement(destination, parent)
                    backup_identity = None
                    marker_identity = None
            raise

        if backup_identity is not None:
            _load_state_decay_artifact(backup)
            if not _remove_owned_tree(backup, backup_identity):
                raise BenchmarkCommandError() from None
            backup_identity = None
            _fsync_directory(parent)
        if marker_identity is not None:
            if not _unlink_owned_file(marker_path, marker_identity):
                _fail_validation()
            marker_identity = None
            _fsync_directory(parent)
    except OSError:
        io_failed = True
    finally:
        if not published:
            with suppress(OSError):
                _remove_owned_tree(staging, staging_identity)
        if marker_identity is not None:
            with suppress(
                BenchmarkArtifactValidationError,
                BenchmarkCommandError,
                OSError,
            ):
                _recover_replacement(destination, parent)
    if io_failed:
        raise BenchmarkCommandError() from None


def _publish(
    destination: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool,
    manifest: SmokeManifest,
) -> None:
    parent = destination.parent
    metadata: os.stat_result | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent = parent.resolve(strict=True)
        metadata = parent.lstat()
    except (OSError, RuntimeError):
        pass
    if metadata is None:
        raise BenchmarkCommandError() from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (
            os.name == "posix"
            and (
                (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
                or stat.S_IMODE(metadata.st_mode) & 0o022
            )
        )
    ):
        raise BenchmarkCommandError() from None
    destination = parent / destination.name
    with _destination_lock(destination, parent):
        _publish_locked(
            destination,
            parent,
            files,
            replace=replace,
            manifest=manifest,
        )


def _report(manifest: SmokeManifest) -> BenchmarkCommandReport:
    values = manifest.model_dump(
        mode="json",
        exclude={
            "fixture_path",
            "fixture_byte_count",
            "oracle_version",
            "oracle_result_schema_version",
        },
        warnings=False,
    )
    values["schema_version"] = CLI_BENCHMARK_SCHEMA_VERSION
    values["status"] = "ok"
    return BenchmarkCommandReport.model_validate(values)


def state_decay_artifact_files() -> dict[str, bytes]:
    """Return a fresh, deterministic native artifact without reading source fixtures."""

    diagnostic: StateDecayDiagnosticResult | None = None
    with suppress(StateDecayDiagnosticError):
        diagnostic = run_state_decay_diagnostic()
    if diagnostic is None:
        _fail_validation()
    fixture: bytes | None = None
    with suppress(SmokeCoverageError):
        fixture = encode_scenarios_jsonl(diagnostic.scenarios)
    if fixture is None:
        _fail_validation()
    manifest = _compose_state_decay_manifest(diagnostic, fixture)
    return {MANIFEST_NAME: canonical_json(manifest), FIXTURE_NAME: fixture}


def run_state_decay_smoke(
    output: str | os.PathLike[str], *, replace: bool = False
) -> BenchmarkCommandReport:
    """Generate, evaluate, publish, and revalidate the frozen offline smoke suite."""

    if type(replace) is not bool:
        raise BenchmarkCommandError() from None
    destination = _path(output)
    failure: Literal["command", "artifact"] | None = None
    try:
        files = state_decay_artifact_files()
        manifest = SmokeManifest.model_validate_json(files[MANIFEST_NAME])
        _publish(
            destination,
            files,
            replace=replace,
            manifest=manifest,
        )
        validated = _load_state_decay_artifact(
            destination, expected_manifest_digest=manifest.manifest_digest
        )
        return _report(validated)
    except BenchmarkCommandError:
        failure = "command"
    except BenchmarkArtifactValidationError:
        failure = "artifact"
    except (SmokeCoverageError, StateDecayDiagnosticError):
        failure = "artifact"
    if failure == "command":
        raise BenchmarkCommandError() from None
    if failure == "artifact":
        raise BenchmarkArtifactValidationError() from None
    raise AssertionError("unreachable benchmark command state")  # pragma: no cover


def render_benchmark_json(report: BenchmarkCommandReport) -> str:
    validated: BenchmarkCommandReport | None = None
    with suppress(Exception):
        validated = BenchmarkCommandReport.model_validate(report)
    if validated is None:
        raise BenchmarkCommandError() from None
    return canonical_json(validated).decode("utf-8") + "\n"


def render_benchmark_human(report: BenchmarkCommandReport) -> str:
    validated: BenchmarkCommandReport | None = None
    with suppress(Exception):
        validated = BenchmarkCommandReport.model_validate(report)
    if validated is None:
        raise BenchmarkCommandError() from None
    return (
        "StateDecayBench smoke complete\n"
        f"scenarios: {validated.scenario_count}\n"
        f"families: {validated.family_count}\n"
        f"labels: {validated.intervene_count} intervene, "
        f"{validated.silence_count} silence\n"
        f"oracle: {validated.oracle_passed} passed, {validated.oracle_failed} failed\n"
        "diagnostic: yes\n"
        "synthetic: yes\n"
        "balanced: yes\n"
        "external claims: insufficient\n"
        f"manifest digest: {validated.manifest_digest}\n"
    )


__all__ = [
    "CLI_BENCHMARK_SCHEMA_VERSION",
    "SMOKE_MANIFEST_SCHEMA_VERSION",
    "BenchmarkArtifactValidationError",
    "BenchmarkCommandError",
    "BenchmarkCommandReport",
    "BenchmarkValidationReport",
    "SmokeManifest",
    "build_state_decay_manifest",
    "render_benchmark_human",
    "render_benchmark_json",
    "run_state_decay_smoke",
    "state_decay_artifact_files",
    "validate_state_decay_artifact",
]
