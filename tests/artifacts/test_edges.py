from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from saliencegate.artifacts.export import (
    ArtifactExistsError,
    ArtifactExportError,
    SyntheticArtifactContent,
    discover_revision,
    export_replay_artifact,
)
from saliencegate.artifacts.manifest import (
    ArtifactAttestationsComponent,
    ArtifactBudgetsComponent,
    ArtifactClassification,
    ArtifactComponent,
    ArtifactComponentName,
    ArtifactDecisionsComponent,
    ArtifactDeliveriesComponent,
    ArtifactEvidenceLevel,
    ArtifactManifest,
    ArtifactOutcomesComponent,
    ArtifactRunComponent,
    CycleAttestation,
    DeliveryAttestation,
    InterventionAttestation,
    RevisionEvidence,
    RevisionSource,
    component_content_digest,
    delivery_binding_digest,
)
from saliencegate.artifacts.validate import (
    ArtifactValidationCode,
    ArtifactValidationError,
    validate_artifact,
)
from saliencegate.domain import canonical_digest, canonical_json
from saliencegate.runtime.engine import ReplayRunResult
from saliencegate.security import RedactionPolicy

artifact_export = importlib.import_module("saliencegate.artifacts.export")
artifact_tree = importlib.import_module("saliencegate.artifacts.tree")
artifact_validate = importlib.import_module("saliencegate.artifacts.validate")


def _revision(*, dirty: bool = False) -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="3" * 40,
        dirty_worktree=dirty,
        distribution_digest=None,
    )


def _export(
    root: Path,
    result: ReplayRunResult,
    *,
    evidence_level: ArtifactEvidenceLevel = ArtifactEvidenceLevel.EXPLORATORY,
) -> ArtifactManifest:
    return export_replay_artifact(
        result,
        root,
        classification=ArtifactClassification.USER_REDACTED,
        evidence_level=evidence_level,
        revision=_revision(),
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _reseal_component(
    root: Path,
    manifest: ArtifactManifest,
    name: ArtifactComponentName,
    data: bytes,
) -> ArtifactManifest:
    descriptor = next(component for component in manifest.components if component.name is name)
    replacement = ArtifactComponent(
        name=name,
        path=descriptor.path,
        byte_count=len(data),
        record_count=descriptor.record_count,
        content_digest=component_content_digest(data),
    )
    components = tuple(
        replacement if component.name is name else component for component in manifest.components
    )
    updated = ArtifactManifest.create(
        classification=manifest.classification,
        evidence_level=manifest.evidence_level,
        run_id=manifest.run_id,
        revision=manifest.revision,
        engine_configuration_digest=manifest.engine_configuration_digest,
        trace_digest=manifest.trace_digest,
        model_id=manifest.model_id,
        replay_id=manifest.replay_id,
        prompt_template_digest=manifest.prompt_template_digest,
        result_digest=manifest.result_digest,
        components=components,
        counters=manifest.counters,
    )
    (root / descriptor.path).write_bytes(data)
    (root / "manifest.json").write_bytes(canonical_json(updated))
    return updated


def _rewrite_component(
    root: Path,
    manifest: ArtifactManifest,
    name: ArtifactComponentName,
    mutate: Callable[[dict[str, object]], None],
) -> ArtifactManifest:
    descriptor = next(component for component in manifest.components if component.name is name)
    payload = json.loads((root / descriptor.path).read_bytes())
    assert isinstance(payload, dict)
    mutate(payload)
    return _reseal_component(root, manifest, name, canonical_json(payload))


def _assert_code(
    error: pytest.ExceptionInfo[ArtifactValidationError],
    code: ArtifactValidationCode,
) -> None:
    assert error.value.code is code
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


@pytest.mark.parametrize("status", (b"", b" M fixture-secret.txt\n"))
def test_discover_revision_reads_full_git_commit_and_dirty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: bytes,
) -> None:
    responses = iter(((0, b"a" * 40 + b"\n"), (0, status)))
    monkeypatch.setattr(
        artifact_export,
        "_run_git",
        lambda arguments, cwd, **kwargs: next(responses),
    )

    revision = discover_revision(tmp_path)

    assert revision.source is RevisionSource.GIT
    assert revision.commit == "a" * 40
    assert revision.dirty_worktree is bool(status)
    assert revision.distribution_digest is None
    assert revision.confirmatory_eligible is (not status)


@pytest.mark.parametrize(
    ("responses", "git_was_available"),
    (
        (((1, b""),), True),
        (((0, b"\xff"),), True),
        (((0, b"a" * 40), (1, b"")), True),
        (((0, b"not-a-full-revision"), (0, b"")), True),
    ),
)
def test_git_revision_rejects_incomplete_or_malformed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: tuple[tuple[int, bytes], ...],
    git_was_available: bool,
) -> None:
    pending = iter(responses)
    monkeypatch.setattr(
        artifact_export,
        "_run_git",
        lambda arguments, cwd, **kwargs: next(pending),
    )

    assert artifact_export._git_revision(tmp_path) == (None, git_was_available)


def test_discover_revision_uses_distribution_only_when_git_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "b" * 64
    monkeypatch.setattr(artifact_export, "_git_revision", lambda root: (None, False))
    monkeypatch.setattr(artifact_export, "_distribution_digest", lambda: digest)

    distribution = discover_revision(tmp_path)

    assert distribution.source is RevisionSource.DISTRIBUTION
    assert distribution.distribution_digest == digest
    assert distribution.confirmatory_eligible

    monkeypatch.setattr(artifact_export, "_distribution_digest", lambda: None)
    unattested = discover_revision(tmp_path)
    assert unattested.source is RevisionSource.UNATTESTED
    assert not unattested.confirmatory_eligible

    monkeypatch.setattr(artifact_export, "_git_revision", lambda root: (None, True))

    def unexpected_distribution_lookup() -> str:
        raise AssertionError("distribution fallback must not mask broken Git evidence")

    monkeypatch.setattr(artifact_export, "_distribution_digest", unexpected_distribution_lookup)
    assert discover_revision(tmp_path).source is RevisionSource.UNATTESTED


def test_git_runner_is_bounded_and_maps_process_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, output: bytes, return_code: int) -> None:
            self.stdout = io.BytesIO(output)
            self.returncode = return_code

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    class HangingProcess:
        def __init__(self, output: bytes) -> None:
            self.stdout = io.BytesIO(output)
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("git", timeout)
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fail_run(*args: object, **kwargs: object) -> object:
        raise OSError("fixture-secret process failure")

    monkeypatch.setattr(artifact_export.subprocess, "Popen", fail_run)
    assert artifact_export._run_git(("status",), tmp_path) == (-1, b"")

    oversized = b"x" * 17
    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(oversized, 7),
    )
    code, output = artifact_export._run_git(("status",), tmp_path, maximum_output=4)
    assert code == -2
    assert output == b"xxxx"

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def clean_process(*args: object, **kwargs: object) -> FakeProcess:
        calls.append((args, kwargs))
        return FakeProcess(b"clean\n", 0)

    monkeypatch.setenv("FIXTURE_SECRET", "must-not-be-inherited")
    monkeypatch.setenv("SYSTEMROOT", "fixture-system-root")
    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        clean_process,
    )
    assert artifact_export._run_git(("status",), tmp_path) == (0, b"clean\n")
    command = calls[0][0][0]
    options = calls[0][1]
    assert command[-1] == "status"
    assert options["stdin"] is artifact_export.subprocess.DEVNULL
    assert options["stderr"] is artifact_export.subprocess.DEVNULL
    assert options["stdout"] is artifact_export.subprocess.PIPE
    assert options["cwd"] == tmp_path
    assert options["env"]["GIT_NO_LAZY_FETCH"] == "1"
    assert options["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert options["env"]["SYSTEMROOT"] == "fixture-system-root"
    assert "FIXTURE_SECRET" not in options["env"]
    assert artifact_export._run_git(("status",), tmp_path, maximum_output=0) == (-1, b"")

    class MissingStreamProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__(b"", 0)
            self.stdout = None

    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: MissingStreamProcess(),
    )
    assert artifact_export._run_git(("status",), tmp_path) == (-2, b"")

    monkeypatch.setattr(artifact_export, "_GIT_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: HangingProcess(b""),
    )
    assert artifact_export._run_git(("status",), tmp_path) == (-2, b"")

    monkeypatch.setattr(artifact_export, "_GIT_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: HangingProcess(b"oversized"),
    )
    assert artifact_export._run_git(("status",), tmp_path, maximum_output=1) == (-2, b"o")

    class FailingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise OSError("fixture-secret read failure")

    failing = FakeProcess(b"", 0)
    failing.stdout = FailingStream()
    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: failing,
    )
    assert artifact_export._run_git(("status",), tmp_path) == (-2, b"")

    class FailingWaitProcess(FakeProcess):
        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired("git", timeout)

    monkeypatch.setattr(
        artifact_export.subprocess,
        "Popen",
        lambda *args, **kwargs: FailingWaitProcess(b"", 0),
    )
    assert artifact_export._run_git(("status",), tmp_path) == (-2, b"")


def test_distribution_digest_is_content_bound_and_ignores_runtime_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "saliencegate"
    (package_root / "artifacts").mkdir(parents=True)
    source = package_root / "core.py"
    source.write_text("first", encoding="utf-8")
    monkeypatch.setattr(
        artifact_export,
        "__file__",
        str(package_root / "artifacts" / "export.py"),
    )
    monkeypatch.setattr(
        artifact_export.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=artifact_export.__version__),
    )

    first = artifact_export._distribution_digest()
    assert first is not None and len(first) == 64

    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "fixture-secret.pyc").write_bytes(b"ignored")
    assert artifact_export._distribution_digest() == first

    source.write_text("second", encoding="utf-8")
    second = artifact_export._distribution_digest()
    assert second is not None and second != first


def test_distribution_digest_refuses_unverifiable_installations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "saliencegate"
    (package_root / "artifacts").mkdir(parents=True)
    source = package_root / "core.py"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(
        artifact_export,
        "__file__",
        str(package_root / "artifacts" / "export.py"),
    )

    monkeypatch.setattr(
        artifact_export.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="9.9.9"),
    )
    assert artifact_export._distribution_digest() is None

    def missing_distribution(name: str) -> object:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        artifact_export.importlib.metadata,
        "distribution",
        missing_distribution,
    )
    assert artifact_export._distribution_digest() is None

    monkeypatch.setattr(
        artifact_export.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=artifact_export.__version__),
    )
    linked = package_root / "linked.py"
    os.link(source, linked)
    assert artifact_export._distribution_digest() is None


def test_distribution_digest_refuses_empty_bounded_or_unreadable_inventories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "saliencegate"
    (package_root / "artifacts").mkdir(parents=True)
    monkeypatch.setattr(
        artifact_export,
        "__file__",
        str(package_root / "artifacts" / "export.py"),
    )
    monkeypatch.setattr(
        artifact_export.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=artifact_export.__version__),
    )

    assert artifact_export._distribution_digest() is None

    source = package_root / "core.py"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(artifact_export, "_MAX_DISTRIBUTION_FILES", 0)
    assert artifact_export._distribution_digest() is None

    monkeypatch.setattr(artifact_export, "_MAX_DISTRIBUTION_FILES", 10_000)
    monkeypatch.setattr(artifact_export, "_regular_file_bytes", lambda path, maximum: None)
    assert artifact_export._distribution_digest() is None


def test_regular_file_reader_rejects_missing_nonregular_and_oversized_paths(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"abc")

    assert artifact_export._regular_file_bytes(regular, maximum=3) == b"abc"
    assert artifact_export._regular_file_bytes(regular, maximum=2) is None
    assert artifact_export._regular_file_bytes(tmp_path / "missing", maximum=3) is None
    assert artifact_export._regular_file_bytes(tmp_path, maximum=3) is None


def test_regular_file_reader_refuses_read_errors_and_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "regular.bin"
    path.write_bytes(b"abc")

    with monkeypatch.context() as patch:
        patch.setattr(
            artifact_export.os,
            "read",
            lambda descriptor, size: (_ for _ in ()).throw(OSError("fixture-secret read failure")),
        )
        assert artifact_export._regular_file_bytes(path, maximum=3) is None

    metadata = path.stat()
    changed = SimpleNamespace(
        st_mode=metadata.st_mode,
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size + 1,
        st_mtime_ns=metadata.st_mtime_ns,
        st_ctime_ns=metadata.st_ctime_ns,
    )
    observed = iter((metadata, changed))
    with monkeypatch.context() as patch:
        patch.setattr(artifact_export.os, "fstat", lambda descriptor: next(observed))
        assert artifact_export._regular_file_bytes(path, maximum=4) is None


def test_discover_revision_handles_invalid_path_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise ValueError("fixture-secret invalid path")

    def unexpected_git_lookup(root: Path) -> tuple[RevisionEvidence | None, bool]:
        raise AssertionError("invalid explicit paths must not fall back to the current directory")

    monkeypatch.setattr(artifact_export, "_git_revision", unexpected_git_lookup)
    monkeypatch.setattr(artifact_export, "_distribution_digest", lambda: "a" * 64)

    revision = discover_revision(InvalidPath())

    assert revision.source is RevisionSource.DISTRIBUTION
    assert revision.distribution_digest == "a" * 64


def test_discover_revision_ignores_an_unrelated_git_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = tmp_path / "foreign"
    (foreign / ".git").mkdir(parents=True)
    (foreign / "pyproject.toml").write_text("[project]\nname='foreign'\n", encoding="utf-8")
    installed_module = tmp_path / "installed" / "saliencegate" / "artifacts" / "export.py"
    installed_module.parent.mkdir(parents=True)
    installed_module.write_text("# installed fixture\n", encoding="utf-8")
    monkeypatch.chdir(foreign)
    monkeypatch.setattr(artifact_export, "__file__", str(installed_module))

    def unexpected_git_lookup(root: Path) -> tuple[RevisionEvidence | None, bool]:
        raise AssertionError("the current directory must not supply package provenance")

    monkeypatch.setattr(artifact_export, "_git_revision", unexpected_git_lookup)
    monkeypatch.setattr(artifact_export, "_distribution_digest", lambda: "b" * 64)

    revision = discover_revision()

    assert revision.source is RevisionSource.DISTRIBUTION
    assert revision.distribution_digest == "b" * 64


def test_discover_revision_uses_only_the_checkout_containing_the_loaded_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / "src" / "saliencegate" / "artifacts" / "export.py"
    module.parent.mkdir(parents=True)
    module.write_text("# checkout fixture\n", encoding="utf-8")
    (checkout / "pyproject.toml").write_text("[project]\nname='saliencegate'\n", encoding="utf-8")
    (checkout / ".git").mkdir()
    expected = _revision()
    observed: list[Path] = []
    monkeypatch.setattr(artifact_export, "__file__", str(module))

    def read_checkout(root: Path) -> tuple[RevisionEvidence, bool]:
        observed.append(root)
        return expected, True

    monkeypatch.setattr(artifact_export, "_git_revision", read_checkout)

    assert discover_revision() == expected
    assert observed == [checkout]


def test_discover_revision_rejects_checkout_shaped_paths_without_regular_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / "src" / "saliencegate" / "artifacts" / "export.py"
    module.parent.mkdir(parents=True)
    module.write_text("# checkout fixture\n", encoding="utf-8")
    (checkout / "pyproject.toml").mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.setattr(artifact_export, "__file__", str(module))
    monkeypatch.setattr(artifact_export, "_distribution_digest", lambda: "c" * 64)

    def unexpected_git_lookup(root: Path) -> tuple[RevisionEvidence | None, bool]:
        raise AssertionError("invalid checkout markers must not be attested")

    monkeypatch.setattr(artifact_export, "_git_revision", unexpected_git_lookup)

    revision = discover_revision()

    assert revision.source is RevisionSource.DISTRIBUTION
    assert revision.distribution_digest == "c" * 64


def test_export_internal_failures_are_value_free_and_cleanup_staging(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("fixture-secret internal failure")

    with monkeypatch.context() as patch:
        patch.setattr(ReplayRunResult, "model_dump_json", fail)
        with pytest.raises(ArtifactExportError, match="replay result") as error:
            artifact_export._validated_result(replay_result)
        assert "fixture-secret" not in repr(error.value)

    revision = _revision()
    with monkeypatch.context() as patch:
        patch.setattr(RevisionEvidence, "model_dump_json", fail)
        with pytest.raises(ArtifactExportError, match="revision evidence") as error:
            artifact_export._validated_revision(revision)
        assert "fixture-secret" not in repr(error.value)

    for function in (
        artifact_export._intervention_attestation,
        artifact_export._cycle_attestation,
        artifact_export._delivery_attestation,
    ):
        with pytest.raises(ArtifactExportError) as error:
            function(object())
        assert "fixture-secret" not in repr(error.value)

    with monkeypatch.context() as patch:
        patch.setattr(
            artifact_export,
            "GroundingReceipt",
            SimpleNamespace(model_validate_json=fail),
        )
        with pytest.raises(ArtifactExportError, match="intervention") as error:
            artifact_export._intervention_attestation(replay_result.events[-1].cycle.intervention)
        assert "fixture-secret" not in repr(error.value)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_export, "CycleAttestation", fail)
        with pytest.raises(ArtifactExportError, match="cycle") as error:
            artifact_export._cycle_attestation(replay_result.events[1])
        assert "fixture-secret" not in repr(error.value)

    with monkeypatch.context() as patch:
        patch.setattr(
            artifact_export,
            "DeliveryAttestation",
            SimpleNamespace(model_validate=fail),
        )
        with pytest.raises(ArtifactExportError, match="delivery") as error:
            artifact_export._delivery_attestation(replay_result.events[-1])
        assert "fixture-secret" not in repr(error.value)

    broken_component = SimpleNamespace(model_dump=fail)
    with pytest.raises(ArtifactExportError, match="redaction verification") as error:
        artifact_export._assert_export_is_redacted(
            {ArtifactComponentName.RUN: broken_component},
            RedactionPolicy(),
        )
    assert "fixture-secret" not in repr(error.value)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_export, "canonical_json", fail)
        with pytest.raises(ArtifactExportError, match="serialization") as error:
            artifact_export._encode_components({ArtifactComponentName.RUN: revision})
        assert "fixture-secret" not in repr(error.value)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_export, "MAX_ARTIFACT_COMPONENT_BYTES", 1)
        with pytest.raises(ArtifactExportError, match="byte limit"):
            artifact_export._encode_components({ArtifactComponentName.RUN: revision})

    with monkeypatch.context() as patch:
        patch.setattr(artifact_export.os, "write", lambda descriptor, data: 0)
        with pytest.raises(OSError, match="short artifact write"):
            artifact_export._write_file(tmp_path, "short.json", b"{}")

    with monkeypatch.context() as patch:
        patch.setattr(
            artifact_tree,
            "_write_file",
            lambda directory, name, data: (_ for _ in ()).throw(
                ArtifactExportError("value-free write failure")
            ),
        )
        with pytest.raises(ArtifactExportError, match="value-free write failure"):
            _export(tmp_path / "publish", replay_result)
    assert not tuple(tmp_path.glob(".publish.tmp-*"))


async def test_confirmatory_validation_is_an_explicit_request(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    exploratory_root = tmp_path / "exploratory"
    _export(exploratory_root, replay_result)

    with pytest.raises(ArtifactValidationError) as ineligible:
        validate_artifact(exploratory_root / "manifest.json", require_confirmatory=True)
    _assert_code(ineligible, ArtifactValidationCode.CONFIRMATORY_INELIGIBLE)

    confirmatory_root = tmp_path / "confirmatory"
    manifest = _export(
        confirmatory_root,
        replay_result,
        evidence_level=ArtifactEvidenceLevel.CONFIRMATORY,
    )
    report = validate_artifact(
        confirmatory_root / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
        require_confirmatory=True,
    )
    assert report.confirmatory
    assert report.expected_digest_matched is True


async def test_confirmatory_export_refuses_dirty_revision_without_leaking_values(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    with pytest.raises(ArtifactExportError) as error:
        export_replay_artifact(
            replay_result,
            tmp_path / "fixture-secret-destination",
            evidence_level=ArtifactEvidenceLevel.CONFIRMATORY,
            revision=_revision(dirty=True),
        )

    assert str(error.value) == "artifact construction failed"
    assert "fixture-secret" not in repr(error.value)
    assert not (tmp_path / "fixture-secret-destination").exists()


async def test_export_revalidates_result_and_revision_boundaries_secret_free(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    with pytest.raises(ArtifactExportError, match="replay result") as bad_result:
        export_replay_artifact(  # type: ignore[arg-type]
            object(),
            tmp_path / "bad-result",
            revision=_revision(),
        )
    assert "fixture-secret" not in repr(bad_result.value)

    with pytest.raises(ArtifactExportError, match="revision evidence") as bad_revision:
        export_replay_artifact(
            replay_result,
            tmp_path / "bad-revision",
            revision=object(),  # type: ignore[arg-type]
        )
    assert "fixture-secret" not in repr(bad_revision.value)


@pytest.mark.parametrize(
    "payload",
    (
        b"[]",
        b"{not-json}",
        b'{"components":[], "schema_version":"1.0"}',
        b'{"components":[],"components":[],"schema_version":"1.0"}',
        b'{"components":[],"fixture-secret":NaN,"schema_version":"1.0"}',
        b"\xff\xfe",
    ),
)
async def test_manifest_parser_rejects_noncanonical_or_malformed_json_secret_free(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    payload: bytes,
) -> None:
    root = tmp_path / "artifact"
    _export(root, replay_result)
    (root / "manifest.json").write_bytes(payload)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_MANIFEST)


@pytest.mark.parametrize(
    ("schema_version", "expected_code"),
    (
        ("2.0", ArtifactValidationCode.UNSUPPORTED_VERSION),
        ("1.1", ArtifactValidationCode.UNSUPPORTED_VERSION),
        ("01.0", ArtifactValidationCode.INVALID_MANIFEST),
        (1, ArtifactValidationCode.INVALID_MANIFEST),
    ),
)
async def test_manifest_preflight_distinguishes_unsupported_and_malformed_versions(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    schema_version: object,
    expected_code: ArtifactValidationCode,
) -> None:
    root = tmp_path / "artifact"
    manifest = _export(root, replay_result)
    values = manifest.model_dump(mode="json")
    values["schema_version"] = schema_version
    (root / "manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, expected_code)


async def test_manifest_preflight_rejects_malformed_component_descriptors(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    mutations: tuple[Callable[[dict[str, object]], None], ...] = (
        lambda values: values.__setitem__("components", "fixture-secret"),
        lambda values: values.__setitem__("components", ["fixture-secret"]),
        lambda values: values.__setitem__(
            "components", [{"name": 7, "path": "fixture-secret.json"}]
        ),
        lambda values: values.__setitem__(
            "components", [{"name": "fixture-secret", "path": "fixture-secret.json"}]
        ),
    )
    for index, mutate in enumerate(mutations):
        root = tmp_path / f"case-{index}"
        manifest = _export(root, replay_result)
        values = manifest.model_dump(mode="json")
        mutate(values)
        (root / "manifest.json").write_bytes(canonical_json(values))

        with pytest.raises(ArtifactValidationError) as error:
            validate_artifact(root / "manifest.json")
        _assert_code(error, ArtifactValidationCode.INVALID_MANIFEST)


async def test_manifest_model_failure_is_mapped_to_a_value_free_error(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root = tmp_path / "artifact"
    manifest = _export(root, replay_result)
    values = manifest.model_dump(mode="json")
    values["manifest_digest"] = "f" * 64
    (root / "manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_MANIFEST)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    (
        (
            b'{"fixture-secret":"hidden","schema_version":"artifact-run/v2"}',
            ArtifactValidationCode.UNSUPPORTED_VERSION,
        ),
        (
            b'{"fixture-secret":"hidden","schema_version":"artifact-other/v1"}',
            ArtifactValidationCode.INVALID_COMPONENT,
        ),
        (
            b'{"fixture-secret": "hidden","schema_version":"artifact-run/v1"}',
            ArtifactValidationCode.INVALID_COMPONENT,
        ),
        (
            b'{"schema_version":"artifact-run/v1","schema_version":"artifact-run/v1"}',
            ArtifactValidationCode.INVALID_COMPONENT,
        ),
        (
            b'{"fixture-secret":NaN,"schema_version":"artifact-run/v1"}',
            ArtifactValidationCode.INVALID_COMPONENT,
        ),
    ),
)
async def test_component_parser_rejects_unknown_versions_and_noncanonical_payloads(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    payload: bytes,
    expected_code: ArtifactValidationCode,
) -> None:
    root = tmp_path / "artifact"
    manifest = _export(root, replay_result)
    _reseal_component(root, manifest, ArtifactComponentName.RUN, payload)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, expected_code)


async def test_component_schema_failure_is_mapped_to_a_value_free_error(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root = tmp_path / "artifact"
    manifest = _export(root, replay_result)
    payload = b'{"fixture-secret":"hidden","schema_version":"artifact-run/v1"}'
    _reseal_component(root, manifest, ArtifactComponentName.RUN, payload)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


async def test_resealed_cross_component_tampering_is_rejected(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    cases = (
        (
            "run-binding",
            ArtifactComponentName.RUN,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "trace-count",
            ArtifactComponentName.RUN,
            ArtifactValidationCode.INCONSISTENT_COUNTERS,
        ),
        (
            "policy-binding",
            ArtifactComponentName.RUN,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "grounding-binding",
            ArtifactComponentName.RUN,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "cycle-order",
            ArtifactComponentName.BUDGETS,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "cycle-id",
            ArtifactComponentName.BUDGETS,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "intervention-grounding",
            ArtifactComponentName.BUDGETS,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "model-requests",
            ArtifactComponentName.ATTESTATIONS,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
        (
            "outcome-order",
            ArtifactComponentName.OUTCOMES,
            ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
        ),
    )

    def mutate(case: str, payload: dict[str, object]) -> None:
        if case == "run-binding":
            payload["rebuild_equivalent"] = False
        elif case == "trace-count":
            payload["trace_event_count"] = 3
        elif case == "policy-binding":
            payload["policy_configurations"] = []
        elif case == "grounding-binding":
            payload["grounding_configurations"] = []
        elif case == "cycle-order":
            cycles = payload["cycles"]
            assert isinstance(cycles, list)
            cycles.reverse()
        elif case == "cycle-id":
            cycles = payload["cycles"]
            assert isinstance(cycles, list) and isinstance(cycles[0], dict)
            cycles[0]["cycle_id"] = "f" * 64
        elif case == "intervention-grounding":
            cycles = payload["cycles"]
            assert isinstance(cycles, list)
            cycle = next(
                item
                for item in cycles
                if isinstance(item, dict) and isinstance(item.get("intervention"), dict)
            )
            intervention = cycle["intervention"]
            assert isinstance(intervention, dict)
            intervention["grounding_configuration_digest"] = "f" * 64
        elif case == "model-requests":
            requests = payload["model_request_digests"]
            assert isinstance(requests, list)
            requests.reverse()
        elif case == "outcome-order":
            outcomes = payload["outcomes"]
            assert isinstance(outcomes, list)
            outcomes.reverse()
        else:
            raise AssertionError("unknown test case")

    for case, component, expected_code in cases:
        root = tmp_path / case
        manifest = _export(root, replay_result)
        _rewrite_component(
            root, manifest, component, lambda payload, case=case: mutate(case, payload)
        )

        with pytest.raises(ArtifactValidationError) as error:
            validate_artifact(root / "manifest.json")
        _assert_code(error, expected_code)


async def test_component_models_reject_internally_inconsistent_attestations(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root = tmp_path / "artifact"
    _export(root, replay_result)

    def payload(name: ArtifactComponentName) -> dict[str, object]:
        value = json.loads((root / f"{name.value}.json").read_bytes())
        assert isinstance(value, dict)
        return value

    def clone(value: dict[str, object]) -> dict[str, object]:
        copied = json.loads(json.dumps(value))
        assert isinstance(copied, dict)
        return copied

    def validates(model: type[object], value: dict[str, object]) -> object:
        return model.model_validate_json(canonical_json(value))  # type: ignore[attr-defined]

    def rejects(
        model: type[object],
        value: dict[str, object],
        pattern: str,
    ) -> None:
        with pytest.raises(ValidationError, match=pattern):
            validates(model, value)

    run = payload(ArtifactComponentName.RUN)
    altered = clone(run)
    altered["engine_configuration_digest"] = "f" * 64
    rejects(ArtifactRunComponent, altered, "configuration digest")

    altered = clone(run)
    policies = altered["policy_configurations"]
    assert isinstance(policies, list)
    policies.append(policies[0])
    rejects(ArtifactRunComponent, altered, "non-canonical")

    altered = clone(run)
    altered["model_execution_mode"] = "structured_model"
    rejects(ArtifactRunComponent, altered, "structured model")

    structured = clone(run)
    structured["model_execution_mode"] = "structured_model"
    for field in (
        "replay_id",
        "fixture_digest",
        "fixture_response_count",
        "fixture_consumed_count",
    ):
        structured[field] = None
    validates(ArtifactRunComponent, structured)

    altered = clone(run)
    altered["fixture_consumed_count"] = 2
    rejects(ArtifactRunComponent, altered, "consumed fixture")

    decisions = payload(ArtifactComponentName.DECISIONS)
    altered = clone(decisions)
    altered["decisions_digest"] = "f" * 64
    rejects(ArtifactDecisionsComponent, altered, "canonical ordered run")

    budgets = payload(ArtifactComponentName.BUDGETS)
    cycles = budgets["cycles"]
    assert isinstance(cycles, list)
    silence = next(
        item
        for item in cycles
        if isinstance(item, dict)
        and isinstance(item.get("intervention"), dict)
        and item["intervention"].get("action") == "silence"
    )
    reminder = next(
        item
        for item in cycles
        if isinstance(item, dict)
        and isinstance(item.get("intervention"), dict)
        and item["intervention"].get("action") == "remind"
    )
    silence_intervention = silence["intervention"]
    reminder_intervention = reminder["intervention"]
    assert isinstance(silence_intervention, dict)
    assert isinstance(reminder_intervention, dict)

    altered = clone(reminder_intervention)
    altered["claim_set_digest"] = "f" * 64
    rejects(InterventionAttestation, altered, "claim-set digest")

    altered = clone(reminder_intervention)
    fingerprints = altered["claim_fingerprints"]
    assert isinstance(fingerprints, list)
    fingerprints.append(fingerprints[0])
    altered["claim_set_digest"] = canonical_digest(tuple(fingerprints))
    rejects(InterventionAttestation, altered, "fingerprints must be unique")

    altered = clone(silence_intervention)
    altered["delivery_target"] = "next_model_call"
    rejects(InterventionAttestation, altered, "silent intervention")

    altered = clone(reminder_intervention)
    altered["cited_memory_ids"] = []
    altered["cited_event_ids"] = []
    rejects(InterventionAttestation, altered, "lacks producer grounding")

    committed_cycle = clone(silence)
    committed_cycle["first_event_sequence"] = 3
    rejects(CycleAttestation, committed_cycle, "range is reversed")

    altered = clone(silence)
    altered["state"] = "running"
    rejects(CycleAttestation, altered, "not terminal")

    altered = clone(silence)
    altered["batch_digest"] = None
    rejects(CycleAttestation, altered, "model request attestation")

    altered = clone(silence)
    altered["model_call_latencies_us"] = []
    rejects(CycleAttestation, altered, "model receipts")

    altered = clone(silence)
    altered["revision"] = 3
    rejects(CycleAttestation, altered, "committed artifact cycle")

    altered = clone(silence)
    altered["state"] = "failed"
    rejects(CycleAttestation, altered, "failed artifact cycle")

    failed = clone(silence)
    failed["state"] = "failed"
    failed["revision"] = 2
    failed["intervention"] = None
    failed["failure_reason"] = "repository_unavailable"
    validates(CycleAttestation, failed)

    altered = clone(silence)
    settlement = altered["budget_settlement"]
    assert isinstance(settlement, dict)
    settlement["model_calls"] = 0
    rejects(CycleAttestation, altered, "settled calls")

    altered = clone(budgets)
    altered_cycles = altered["cycles"]
    assert isinstance(altered_cycles, list)
    altered_cycles.append(altered_cycles[0])
    rejects(ArtifactBudgetsComponent, altered, "identities are not unique")

    altered = clone(budgets)
    consumed = altered["consumed"]
    assert isinstance(consumed, dict)
    consumed["latency_us"] = 0
    rejects(ArtifactBudgetsComponent, altered, "consumed budget")

    deliveries = payload(ArtifactComponentName.DELIVERIES)
    delivery_values = deliveries["deliveries"]
    assert isinstance(delivery_values, list) and isinstance(delivery_values[0], dict)
    delivery = delivery_values[0]

    altered = clone(delivery)
    altered["binding_digest"] = "f" * 64
    rejects(DeliveryAttestation, altered, "binding digest")

    altered = clone(delivery)
    altered["state"] = "pending"
    rejects(DeliveryAttestation, altered, "not terminal")

    altered = clone(delivery)
    altered["updated_at"] = "2026-07-11T20:00:00Z"
    rejects(DeliveryAttestation, altered, "precedes creation")

    altered = clone(delivery)
    altered["adapter_deduplicates"] = False
    rejects(DeliveryAttestation, altered, "capability attestation")

    altered = clone(delivery)
    altered["attempt_id"] = None
    rejects(DeliveryAttestation, altered, "terminal state")

    for state, outcome, reason, receipt in (
        ("unknown", "unknown", "delivery_unknown", None),
        ("delivered", "delivered", "delivery_succeeded", "f" * 64),
    ):
        terminal = clone(delivery)
        terminal["state"] = state
        terminal["outcome"] = outcome
        terminal["reason_code"] = reason
        terminal["receipt_digest"] = receipt
        validates(DeliveryAttestation, terminal)

    rejected = clone(delivery)
    rejected["state"] = "rejected"
    rejected["attempt_count"] = 0
    rejected["claim_id"] = None
    rejected["attempt_id"] = None
    rejected["outcome"] = "refused"
    rejected["reason_code"] = "target_unavailable"
    validates(DeliveryAttestation, rejected)

    altered = clone(deliveries)
    altered_deliveries = altered["deliveries"]
    assert isinstance(altered_deliveries, list)
    altered_deliveries.append(altered_deliveries[0])
    rejects(ArtifactDeliveriesComponent, altered, "unique and ordered")

    outcomes = payload(ArtifactComponentName.OUTCOMES)
    altered = clone(outcomes)
    outcome_values = altered["outcomes"]
    assert isinstance(outcome_values, list)
    outcome_values.append(outcome_values[0])
    rejects(ArtifactOutcomesComponent, altered, "one unique run")

    attestations = payload(ArtifactComponentName.ATTESTATIONS)
    altered = clone(attestations)
    altered["routing_digest"] = "f" * 64
    rejects(ArtifactAttestationsComponent, altered, "inconsistent cardinality")


@pytest.mark.parametrize(
    ("path_value", "expected_code"),
    (
        (b"fixture-secret/manifest.json", ArtifactValidationCode.UNSAFE_PATH),
        ("fixture-secret/not-manifest.json", ArtifactValidationCode.UNSAFE_PATH),
        ("fixture-secret/manifest.json", ArtifactValidationCode.MISSING_COMPONENT),
    ),
)
def test_validation_rejects_unsafe_or_missing_manifest_paths_secret_free(
    path_value: bytes | str,
    expected_code: ArtifactValidationCode,
) -> None:
    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(path_value)
    _assert_code(error, expected_code)


@pytest.mark.parametrize("digest", ("fixture-secret", "A" * 64, 7))
def test_validation_rejects_malformed_expected_digests_secret_free(digest: object) -> None:
    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact("manifest.json", expected_manifest_digest=digest)  # type: ignore[arg-type]
    _assert_code(error, ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)


def test_validation_rejects_invalid_path_protocol_and_confirmatory_flag() -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise ValueError("fixture-secret invalid path")

    with pytest.raises(ArtifactValidationError) as invalid_path:
        validate_artifact(InvalidPath())  # type: ignore[arg-type]
    _assert_code(invalid_path, ArtifactValidationCode.UNSAFE_PATH)

    with pytest.raises(ArtifactValidationError) as invalid_flag:
        validate_artifact("manifest.json", require_confirmatory=1)  # type: ignore[arg-type]
    _assert_code(invalid_flag, ArtifactValidationCode.UNSAFE_PATH)


async def test_validation_rejects_directory_empty_and_fifo_components(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    for kind in ("directory", "empty", "fifo"):
        root = tmp_path / kind
        _export(root, replay_result)
        component = root / "decisions.json"
        component.unlink()
        if kind == "directory":
            component.mkdir()
        elif kind == "empty":
            component.write_bytes(b"")
        else:
            if os.name != "posix":
                continue
            os.mkfifo(component)

        with pytest.raises(ArtifactValidationError) as error:
            validate_artifact(root / "manifest.json")
        _assert_code(error, ArtifactValidationCode.UNSAFE_COMPONENT)


async def test_validation_detects_directory_and_identity_changes_after_read(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _export(root, replay_result)
    real_listdir = artifact_validate.os.listdir
    list_calls = 0

    def changing_listing(path: int) -> list[str]:
        nonlocal list_calls
        list_calls += 1
        listed = list(real_listdir(path))
        if list_calls == 2:
            listed.append("fixture-secret-race.json")
        return listed

    monkeypatch.setattr(artifact_validate.os, "listdir", changing_listing)
    with pytest.raises(ArtifactValidationError) as directory_changed:
        validate_artifact(root / "manifest.json")
    _assert_code(directory_changed, ArtifactValidationCode.UNSAFE_COMPONENT)

    monkeypatch.setattr(artifact_validate.os, "listdir", real_listdir)
    real_stat = artifact_validate.os.stat
    manifest_stat_calls = 0

    def disappearing_manifest(
        path: os.PathLike[str] | str,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal manifest_stat_calls
        if path == "manifest.json":
            manifest_stat_calls += 1
            if manifest_stat_calls == 2:
                raise FileNotFoundError("fixture-secret race")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(artifact_validate.os, "stat", disappearing_manifest)
    with pytest.raises(ArtifactValidationError) as identity_changed:
        validate_artifact(root / "manifest.json")
    _assert_code(identity_changed, ArtifactValidationCode.UNSAFE_COMPONENT)


async def test_validation_maps_listing_and_post_listing_races(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_listdir = artifact_validate.os.listdir

    listing_error_root = tmp_path / "listing-error"
    _export(listing_error_root, replay_result)

    def fail_listing(path: int) -> list[str]:
        raise OSError("fixture-secret listing failure")

    monkeypatch.setattr(artifact_validate.os, "listdir", fail_listing)
    with pytest.raises(ArtifactValidationError) as listing_error:
        validate_artifact(listing_error_root / "manifest.json")
    _assert_code(listing_error, ArtifactValidationCode.UNSAFE_COMPONENT)

    monkeypatch.setattr(artifact_validate.os, "listdir", real_listdir)
    missing_after_listing_root = tmp_path / "missing-after-listing"
    _export(missing_after_listing_root, replay_result)
    removed = False

    def remove_after_listing(path: int) -> list[str]:
        nonlocal removed
        listed = list(real_listdir(path))
        if not removed:
            removed = True
            (missing_after_listing_root / "decisions.json").unlink()
        return listed

    monkeypatch.setattr(artifact_validate.os, "listdir", remove_after_listing)
    with pytest.raises(ArtifactValidationError) as missing_after_listing:
        validate_artifact(missing_after_listing_root / "manifest.json")
    _assert_code(missing_after_listing, ArtifactValidationCode.MISSING_COMPONENT)

    monkeypatch.setattr(artifact_validate.os, "listdir", real_listdir)
    identity_root = tmp_path / "identity-replacement"
    _export(identity_root, replay_result)
    list_calls = 0

    def replace_manifest_after_validation(path: int) -> list[str]:
        nonlocal list_calls
        list_calls += 1
        listed = list(real_listdir(path))
        if list_calls == 2:
            manifest_path = identity_root / "manifest.json"
            replacement = identity_root / ".replacement"
            replacement.write_bytes(manifest_path.read_bytes())
            os.replace(replacement, manifest_path)
        return listed

    monkeypatch.setattr(artifact_validate.os, "listdir", replace_manifest_after_validation)
    with pytest.raises(ArtifactValidationError) as identity_replacement:
        validate_artifact(identity_root / "manifest.json")
    _assert_code(identity_replacement, ArtifactValidationCode.UNSAFE_COMPONENT)

    monkeypatch.setattr(artifact_validate.os, "listdir", real_listdir)
    final_listing_root = tmp_path / "final-listing-error"
    _export(final_listing_root, replay_result)
    final_list_calls = 0

    def fail_final_listing(path: int) -> list[str]:
        nonlocal final_list_calls
        final_list_calls += 1
        if final_list_calls == 2:
            raise OSError("fixture-secret final listing failure")
        return list(real_listdir(path))

    monkeypatch.setattr(artifact_validate.os, "listdir", fail_final_listing)
    with pytest.raises(ArtifactValidationError) as final_listing_error:
        validate_artifact(final_listing_root / "manifest.json")
    _assert_code(final_listing_error, ArtifactValidationCode.UNSAFE_COMPONENT)


def test_validator_internal_read_and_format_failures_are_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = tmp_path / "component.json"
    component.write_bytes(b"{}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(tmp_path, flags)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                artifact_validate.os,
                "open",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    PermissionError("fixture-secret open failure")
                ),
            )
            with pytest.raises(ArtifactValidationError) as open_error:
                artifact_validate._read_regular_file(
                    directory_fd,
                    "component.json",
                    maximum=10,
                    missing_code=ArtifactValidationCode.MISSING_COMPONENT,
                )
            _assert_code(open_error, ArtifactValidationCode.MISSING_COMPONENT)

        with monkeypatch.context() as patch:
            patch.setattr(
                artifact_validate.os,
                "read",
                lambda descriptor, size: (_ for _ in ()).throw(
                    OSError("fixture-secret read failure")
                ),
            )
            with pytest.raises(ArtifactValidationError) as read_error:
                artifact_validate._read_regular_file(
                    directory_fd,
                    "component.json",
                    maximum=10,
                    missing_code=ArtifactValidationCode.MISSING_COMPONENT,
                )
            _assert_code(read_error, ArtifactValidationCode.UNSAFE_COMPONENT)

        metadata = component.stat()
        changed = SimpleNamespace(
            st_mode=metadata.st_mode,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_size=metadata.st_size + 1,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
            st_nlink=metadata.st_nlink,
        )
        observed = iter((metadata, changed))
        with monkeypatch.context() as patch:
            patch.setattr(artifact_validate.os, "fstat", lambda descriptor: next(observed))
            with pytest.raises(ArtifactValidationError) as identity_error:
                artifact_validate._read_regular_file(
                    directory_fd,
                    "component.json",
                    maximum=10,
                    missing_code=ArtifactValidationCode.MISSING_COMPONENT,
                )
            _assert_code(identity_error, ArtifactValidationCode.UNSAFE_COMPONENT)

        changed = SimpleNamespace(
            st_mode=metadata.st_mode,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino + 1,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
            st_nlink=metadata.st_nlink,
        )
        with monkeypatch.context() as patch:
            patch.setattr(artifact_validate.os, "stat", lambda *args, **kwargs: changed)
            with pytest.raises(ArtifactValidationError) as current_identity_error:
                artifact_validate._read_regular_file(
                    directory_fd,
                    "component.json",
                    maximum=10,
                    missing_code=ArtifactValidationCode.MISSING_COMPONENT,
                )
            _assert_code(current_identity_error, ArtifactValidationCode.UNSAFE_COMPONENT)
    finally:
        os.close(directory_fd)

    with monkeypatch.context() as patch:
        patch.setattr(
            artifact_validate,
            "canonical_json",
            lambda value: (_ for _ in ()).throw(ValueError("fixture-secret formatting failure")),
        )
        with pytest.raises(ArtifactValidationError) as formatting_error:
            artifact_validate._decode_canonical_object(b"{}", manifest=False)
        _assert_code(formatting_error, ArtifactValidationCode.INVALID_COMPONENT)


async def test_validator_cross_component_guards_cover_missing_and_rebound_records(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    missing_root = tmp_path / "missing-parsed"
    manifest = _export(missing_root, replay_result)
    with pytest.raises(ArtifactValidationError) as missing:
        artifact_validate._validate_cross_component_invariants(manifest, {})
    _assert_code(missing, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)

    delivery_root = tmp_path / "delivery-binding"
    manifest = _export(delivery_root, replay_result)

    def rebind_delivery(payload: dict[str, object]) -> None:
        deliveries = payload["deliveries"]
        assert isinstance(deliveries, list) and isinstance(deliveries[0], dict)
        delivery = deliveries[0]
        delivery["rendered_text_digest"] = "f" * 64
        delivery["binding_digest"] = delivery_binding_digest(delivery)

    _rewrite_component(
        delivery_root,
        manifest,
        ArtifactComponentName.DELIVERIES,
        rebind_delivery,
    )
    with pytest.raises(ArtifactValidationError) as delivery_error:
        validate_artifact(delivery_root / "manifest.json")
    _assert_code(delivery_error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)

    routing_root = tmp_path / "routing-binding"
    manifest = _export(routing_root, replay_result)

    def rebind_routing(payload: dict[str, object]) -> None:
        bindings = payload["routing_bindings"]
        assert isinstance(bindings, list)
        binding = next(
            item
            for item in bindings
            if isinstance(item, dict) and item.get("adapter_id") is not None
        )
        binding["adapter_id"] = "fixture-adapter/v1"
        payload["routing_digest"] = canonical_digest(tuple(bindings))

    _rewrite_component(
        routing_root,
        manifest,
        ArtifactComponentName.ATTESTATIONS,
        rebind_routing,
    )
    with pytest.raises(ArtifactValidationError) as routing_error:
        validate_artifact(routing_root / "manifest.json")
    _assert_code(routing_error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


async def test_export_boundary_rejects_wrong_exact_types_and_unsafe_destinations(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    cases: tuple[tuple[object, dict[str, object]], ...] = (
        (tmp_path / "classification", {"classification": "user_redacted"}),
        (tmp_path / "evidence", {"evidence_level": "exploratory"}),
        (tmp_path / "redaction", {"redaction_policy": object()}),
        (tmp_path / "replace", {"replace": 1}),
        (tmp_path / "synthetic", {"synthetic_content": object()}),
        (b"fixture-secret", {}),
        (object(), {}),
        (Path(".."), {}),
    )
    for destination, kwargs in cases:
        with pytest.raises(ArtifactExportError) as error:
            export_replay_artifact(
                replay_result,
                destination,  # type: ignore[arg-type]
                revision=_revision(),
                **kwargs,  # type: ignore[arg-type]
            )
        assert "fixture-secret" not in str(error.value)
        assert "fixture-secret" not in repr(error.value)

    with pytest.raises(ArtifactExportError, match="raw synthetic content"):
        export_replay_artifact(
            replay_result,
            tmp_path / "raw-without-content",
            classification=ArtifactClassification.SYNTHETIC_RAW,
            revision=_revision(),
        )

    with pytest.raises(ArtifactExportError, match="non-redacted"):
        export_replay_artifact(
            replay_result,
            tmp_path / "policy",
            revision=_revision(),
            redaction_policy=RedactionPolicy(literal_secrets=(replay_result.model_id,)),
        )


async def test_export_refuses_file_or_symlink_destinations(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    destination.write_text("fixture-secret", encoding="utf-8")
    with pytest.raises(ArtifactExistsError):
        _export(destination, replay_result)

    destination.unlink()
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(ArtifactExistsError):
        _export(destination, replay_result)

    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("fixture-secret", encoding="utf-8")
    with pytest.raises(ArtifactExportError, match="destination is unavailable") as error:
        _export(blocked_parent / "artifact", replay_result)
    assert "fixture-secret" not in repr(error.value)


async def test_failed_atomic_replacement_restores_the_original_tree(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    _export(destination, replay_result)
    original = _tree(destination)
    real_replace = os.replace

    def fail_staging_publish(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
    ) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.startswith(".artifact.tmp-") and target_path == destination:
            raise OSError("fixture-secret simulated replacement failure")
        real_replace(source, target)

    monkeypatch.setattr(artifact_export.os, "replace", fail_staging_publish)
    with pytest.raises(ArtifactExportError, match="atomic publish") as error:
        export_replay_artifact(
            replay_result,
            destination,
            revision=_revision(),
            replace=True,
        )

    assert "fixture-secret" not in repr(error.value)
    assert _tree(destination) == original
    assert tuple(tmp_path.glob(".artifact.*")) == (tmp_path / ".artifact.lock",)


def test_revision_and_manifest_version_models_reject_ambiguous_claims() -> None:
    with pytest.raises(ValidationError, match="package version"):
        RevisionEvidence(
            source=RevisionSource.UNATTESTED,
            package_version="fixture secret",
        )
    with pytest.raises(ValidationError, match="Git revision"):
        RevisionEvidence(
            source=RevisionSource.GIT,
            package_version="0.1.0",
            commit="A" * 40,
            dirty_worktree=False,
        )
    with pytest.raises(ValidationError, match="Git revision evidence"):
        RevisionEvidence(
            source=RevisionSource.GIT,
            package_version="0.1.0",
            commit="a" * 40,
        )
    with pytest.raises(ValidationError, match="distribution revision evidence"):
        RevisionEvidence(
            source=RevisionSource.DISTRIBUTION,
            package_version="0.1.0",
        )
    with pytest.raises(ValidationError, match="unattested revision"):
        RevisionEvidence(
            source=RevisionSource.UNATTESTED,
            package_version="0.1.0",
            distribution_digest="f" * 64,
        )


async def test_manifest_model_rejects_minor_eligibility_and_record_count_tampering(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root = tmp_path / "artifact"
    manifest = _export(root, replay_result)

    values = manifest.model_dump(mode="python")
    values["schema_version"] = "fixture-secret"
    with pytest.raises(ValidationError, match="schema version is malformed"):
        ArtifactManifest.model_validate(values)

    values = manifest.model_dump(mode="python")
    values["schema_version"] = "1.1"
    with pytest.raises(ValidationError, match="schema minor"):
        ArtifactManifest.model_validate(values)

    values = manifest.model_dump(mode="python")
    values["confirmatory_eligible"] = False
    with pytest.raises(ValidationError, match="eligibility"):
        ArtifactManifest.model_validate(values)

    values = manifest.model_dump(mode="python")
    values["revision"] = _revision(dirty=True)
    values["confirmatory_eligible"] = False
    values["evidence_level"] = ArtifactEvidenceLevel.CONFIRMATORY
    with pytest.raises(ValidationError, match="confirmatory artifact"):
        ArtifactManifest.model_validate(values)

    values = manifest.model_dump(mode="python")
    components = list(manifest.components)
    decision_index = next(
        index
        for index, component in enumerate(components)
        if component.name is ArtifactComponentName.DECISIONS
    )
    decision = components[decision_index]
    components[decision_index] = ArtifactComponent(
        name=decision.name,
        path=decision.path,
        byte_count=decision.byte_count,
        record_count=decision.record_count - 1,
        content_digest=decision.content_digest,
    )
    values["components"] = tuple(components)
    with pytest.raises(ValidationError, match="record counts"):
        ArtifactManifest.model_validate(values)


def test_component_digest_requires_exact_bytes() -> None:
    with pytest.raises(TypeError, match="exact bytes"):
        component_content_digest(bytearray(b"fixture-secret"))  # type: ignore[arg-type]


def test_synthetic_content_has_a_bounded_response_count() -> None:
    with pytest.raises(ValidationError):
        SyntheticArtifactContent(
            prompt={},
            responses=tuple({"index": index} for index in range(10_001)),
        )


def test_synthetic_content_has_a_bounded_canonical_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_export, "MAX_ARTIFACT_COMPONENT_BYTES", 8)
    with pytest.raises(ValidationError, match="safe input bound"):
        SyntheticArtifactContent(prompt={"fixture-secret": "hidden"})
