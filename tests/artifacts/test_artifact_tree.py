from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from saliencegate.artifacts import tree as tree_module
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ArtifactExportError,
    ClosedTreeDescriptor,
    ClosedTreeFileSpec,
    ClosedTreeRead,
    ClosedTreeReadError,
    ClosedTreeReadErrorKind,
    publish_closed_tree,
    read_closed_tree,
)
from saliencegate.domain import canonical_json

_MANIFEST_NAME = "manifest.json"
_MANIFEST_MAXIMUM_BYTES = 4096


def _component(key: str, generation: int) -> bytes:
    return canonical_json(
        {
            "generation": generation,
            "key": key,
            "payload": f"public-{key}-{generation}",
        }
    )


def _fixture_files(
    *,
    generation: int = 1,
    replacement_key: str = "stable-tree",
) -> dict[str, bytes]:
    beta = _component("beta", generation)
    alpha = _component("alpha", generation)
    manifest = canonical_json(
        {
            "files": [
                {
                    "expected_bytes": len(beta),
                    "key": "beta",
                    "maximum_bytes": 256,
                    "name": "beta.json",
                },
                {
                    "expected_bytes": len(alpha),
                    "key": "alpha",
                    "maximum_bytes": 256,
                    "name": "alpha.json",
                },
            ],
            "generation": generation,
            "replacement_key": replacement_key,
            "schema_version": "closed-tree-test/v1",
        }
    )
    # Deliberately use a non-sorted insertion order. Publication order must not depend on it.
    return {
        "beta.json": beta,
        _MANIFEST_NAME: manifest,
        "alpha.json": alpha,
    }


def _parse_manifest(
    data: bytes,
    *,
    events: list[str] | None = None,
) -> ClosedTreeDescriptor:
    if events is not None:
        events.append("parse_manifest")
    value = json.loads(data)
    if type(value) is not dict or canonical_json(value) != data:
        raise ValueError("manifest is not canonical")
    raw_files = value.get("files")
    if type(raw_files) is not list:
        raise ValueError("manifest files are invalid")
    files: list[ClosedTreeFileSpec] = []
    for raw in raw_files:
        if type(raw) is not dict:
            raise ValueError("manifest file is invalid")
        files.append(
            ClosedTreeFileSpec(
                key=raw.get("key"),
                name=raw.get("name"),
                maximum_bytes=raw.get("maximum_bytes"),
                expected_bytes=raw.get("expected_bytes"),
            )
        )
    return ClosedTreeDescriptor(
        manifest=value,
        manifest_name=_MANIFEST_NAME,
        manifest_digest=sha256(data).hexdigest(),
        replacement_key=value.get("replacement_key"),
        files=tuple(files),
    )


def _parse_file(
    key: object,
    data: bytes,
    *,
    events: list[str] | None = None,
) -> dict[str, object]:
    if events is not None:
        events.append(f"parse_file:{key}")
    value = json.loads(data)
    if type(value) is not dict or canonical_json(value) != data or value.get("key") != key:
        raise ValueError("component is invalid")
    return cast(dict[str, object], value)


def _finish(
    manifest: object,
    parsed: Mapping[object, object],
    *,
    events: list[str] | None = None,
) -> dict[str, object]:
    if events is not None:
        events.append("finish")
    manifest_value = cast(dict[str, object], manifest)
    return {
        "generation": manifest_value["generation"],
        "parsed": dict(parsed),
    }


def _read(
    root: Path,
    *,
    parse_manifest: Callable[[bytes], ClosedTreeDescriptor] = _parse_manifest,
    parse_file: Callable[[object, bytes], object] = _parse_file,
    finish: Callable[[object, Mapping[object, object]], object] = _finish,
) -> ClosedTreeRead:
    return read_closed_tree(
        root / _MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        parse_manifest=parse_manifest,
        parse_file=parse_file,
        finish=finish,
    )


def _validate_tree(
    path: Path,
    expected_digest: str | None,
) -> ClosedTreeDescriptor:
    loaded = _read(path)
    if expected_digest is not None and loaded.manifest_digest != expected_digest:
        raise ValueError("staged tree does not match its descriptor")
    return loaded.descriptor


def _publish(
    root: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool = False,
    parse_manifest: Callable[[bytes], ClosedTreeDescriptor] = _parse_manifest,
    validate_tree: Callable[[Path, str | None], ClosedTreeDescriptor] = _validate_tree,
) -> None:
    publish_closed_tree(
        root,
        files,
        manifest_name=_MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        parse_manifest=parse_manifest,
        validate_tree=validate_tree,
        replace=replace,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file() and not path.is_symlink()
    }


def _caught_publish_error(callback: Callable[[], None]) -> Exception:
    try:
        callback()
    except Exception as error:
        return error
    pytest.fail("publication unexpectedly succeeded")


def _caught_read_error(callback: Callable[[], object]) -> ClosedTreeReadError:
    with pytest.raises(ClosedTreeReadError) as error:
        callback()
    assert type(error.value.kind) is ClosedTreeReadErrorKind
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)
    return error.value


def test_closed_tree_roundtrip_is_deterministic_owner_only_and_callback_ordered(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    files = _fixture_files()
    publish_events: list[str] = []

    def parse_for_publish(data: bytes) -> ClosedTreeDescriptor:
        publish_events.append("publish:parse_manifest")
        return _parse_manifest(data)

    def validate_for_publish(
        path: Path,
        expected_digest: str | None,
    ) -> ClosedTreeDescriptor:
        publish_events.append(
            "publish:validate_destination" if path == first else "publish:validate_staging"
        )
        return _validate_tree(path, expected_digest)

    _publish(
        first,
        files,
        parse_manifest=parse_for_publish,
        validate_tree=validate_for_publish,
    )
    _publish(second, dict(reversed(tuple(files.items()))))

    assert _tree(first) == files
    assert _tree(second) == files
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in first.iterdir())
    assert stat.S_IMODE((tmp_path / ".first.lock").stat().st_mode) == 0o600
    assert publish_events[0] == "publish:parse_manifest"
    assert publish_events[1] == "publish:validate_staging"
    assert "publish:validate_destination" in publish_events[2:]

    read_events: list[str] = []
    loaded = _read(
        first,
        parse_manifest=lambda data: _parse_manifest(data, events=read_events),
        parse_file=lambda key, data: _parse_file(key, data, events=read_events),
        finish=lambda manifest, parsed: _finish(manifest, parsed, events=read_events),
    )

    assert isinstance(loaded, ClosedTreeRead)
    assert loaded.manifest == _parse_manifest(files[_MANIFEST_NAME]).manifest
    assert loaded.manifest_digest == sha256(files[_MANIFEST_NAME]).hexdigest()
    assert loaded.replacement_key == "stable-tree"
    assert loaded.directory_identity is not None
    assert loaded.value == {
        "generation": 1,
        "parsed": {
            "beta": json.loads(files["beta.json"]),
            "alpha": json.loads(files["alpha.json"]),
        },
    }
    assert read_events == [
        "parse_manifest",
        "parse_file:beta",
        "parse_file:alpha",
        "finish",
    ]


def test_reader_rejects_hostile_path_protocols_before_filesystem_access(
    tmp_path: Path,
) -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise ValueError("fixture-secret invalid path")

    paths = (
        b"fixture-secret/manifest.json",
        InvalidPath(),
        tmp_path / "Manifest.json",
    )

    for path in paths:
        error = _caught_read_error(
            lambda path=path: read_closed_tree(
                path,
                maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
                parse_manifest=_parse_manifest,
                parse_file=_parse_file,
                finish=_finish,
            )
        )
        assert error.kind is ClosedTreeReadErrorKind.UNSAFE_PATH


def test_replacement_requires_opt_in_and_an_equal_opaque_key(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement = _fixture_files(generation=2)
    _publish(root, original)

    refused = _caught_publish_error(lambda: _publish(root, replacement))
    assert "public-alpha" not in str(refused)
    assert _tree(root) == original

    wrong_key = _fixture_files(generation=2, replacement_key="different-tree")
    unauthorized = _caught_publish_error(lambda: _publish(root, wrong_key, replace=True))
    assert "public-alpha" not in str(unauthorized)
    assert _tree(root) == original

    _publish(root, replacement, replace=True)

    assert _tree(root) == replacement
    assert _read(root).value["generation"] == 2
    assert tuple(sorted(path.name for path in tmp_path.glob(".artifact.*"))) == (".artifact.lock",)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "",
        ".",
        "..",
        "../escape.json",
        "./alpha.json",
        "nested/alpha.json",
        "/absolute.json",
        "alpha\\beta.json",
        _MANIFEST_NAME,
    ),
)
def test_reader_rejects_unsafe_descriptor_names(tmp_path: Path, unsafe_name: str) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def unsafe_descriptor(data: bytes) -> ClosedTreeDescriptor:
        descriptor = _parse_manifest(data)
        first, second = descriptor.files
        return ClosedTreeDescriptor(
            manifest=descriptor.manifest,
            manifest_name=descriptor.manifest_name,
            manifest_digest=descriptor.manifest_digest,
            replacement_key=descriptor.replacement_key,
            files=(
                ClosedTreeFileSpec(
                    key=first.key,
                    name=unsafe_name,
                    maximum_bytes=first.maximum_bytes,
                    expected_bytes=first.expected_bytes,
                ),
                second,
            ),
        )

    _caught_read_error(lambda: _read(root, parse_manifest=unsafe_descriptor))


@pytest.mark.parametrize(
    ("maximum_bytes", "expected_bytes"),
    (
        (0, 0),
        (-1, 0),
        (True, 1),
        (2, -1),
        (2, True),
        (2, 3),
    ),
)
def test_reader_rejects_invalid_descriptor_bounds(
    tmp_path: Path,
    maximum_bytes: object,
    expected_bytes: object,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def unsafe_descriptor(data: bytes) -> ClosedTreeDescriptor:
        descriptor = _parse_manifest(data)
        first, second = descriptor.files
        return ClosedTreeDescriptor(
            manifest=descriptor.manifest,
            manifest_name=descriptor.manifest_name,
            manifest_digest=descriptor.manifest_digest,
            replacement_key=descriptor.replacement_key,
            files=(
                ClosedTreeFileSpec(
                    key=first.key,
                    name=first.name,
                    maximum_bytes=maximum_bytes,
                    expected_bytes=expected_bytes,
                ),
                second,
            ),
        )

    _caught_read_error(lambda: _read(root, parse_manifest=unsafe_descriptor))


@pytest.mark.parametrize("alias", ("name", "key"))
def test_reader_rejects_descriptor_aliases(tmp_path: Path, alias: str) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def aliased_descriptor(data: bytes) -> ClosedTreeDescriptor:
        descriptor = _parse_manifest(data)
        first, second = descriptor.files
        duplicate = ClosedTreeFileSpec(
            key=first.key if alias == "key" else second.key,
            name=first.name if alias == "name" else second.name,
            maximum_bytes=second.maximum_bytes,
            expected_bytes=second.expected_bytes,
        )
        return ClosedTreeDescriptor(
            manifest=descriptor.manifest,
            manifest_name=descriptor.manifest_name,
            manifest_digest=descriptor.manifest_digest,
            replacement_key=descriptor.replacement_key,
            files=(first, duplicate),
        )

    _caught_read_error(lambda: _read(root, parse_manifest=aliased_descriptor))


def test_reader_rejects_missing_extra_symlink_and_non_owner_modes(tmp_path: Path) -> None:
    roots = tuple(tmp_path / name for name in ("missing", "extra", "symlink", "mode"))
    for root in roots:
        _publish(root, _fixture_files())

    (roots[0] / "alpha.json").unlink()
    missing = _caught_read_error(lambda: _read(roots[0]))

    (roots[1] / "extra.json").write_bytes(b"{}")
    extra = _caught_read_error(lambda: _read(roots[1]))

    outside = tmp_path / "outside.json"
    outside.write_bytes(_component("alpha", 1))
    (roots[2] / "alpha.json").unlink()
    (roots[2] / "alpha.json").symlink_to(outside)
    symlink = _caught_read_error(lambda: _read(roots[2]))

    (roots[3] / "alpha.json").chmod(0o644)
    mode = _caught_read_error(lambda: _read(roots[3]))

    assert missing.kind is not extra.kind
    assert extra.kind is symlink.kind is mode.kind


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link semantics")
def test_reader_rejects_a_hardlinked_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    os.link(root / "alpha.json", tmp_path / "alias.json")

    _caught_read_error(lambda: _read(root))


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link semantics")
def test_reader_rejects_a_hardlinked_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    os.link(root / _MANIFEST_NAME, tmp_path / "manifest-alias.json")

    error = _caught_read_error(lambda: _read(root))

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


def test_reader_rejects_an_unsafe_root_mode(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    root.chmod(0o755)

    error = _caught_read_error(lambda: _read(root))

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


@pytest.mark.skipif(os.name != "posix", reason="exact POSIX mode semantics")
@pytest.mark.parametrize("mode", (0o500, 0o750))
def test_reader_rejects_directory_modes_other_than_0700(tmp_path: Path, mode: int) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    root.chmod(mode)

    try:
        error = _caught_read_error(lambda: _read(root))
    finally:
        root.chmod(0o700)

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


@pytest.mark.skipif(os.name != "posix", reason="exact POSIX mode semantics")
@pytest.mark.parametrize("name", (_MANIFEST_NAME, "alpha.json"))
@pytest.mark.parametrize("mode", (0o400, 0o640))
def test_reader_rejects_file_modes_other_than_0600(
    tmp_path: Path,
    name: str,
    mode: int,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    target = root / name
    target.chmod(mode)

    try:
        error = _caught_read_error(lambda: _read(root))
    finally:
        target.chmod(0o600)

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("manifest_digest", 7),
        ("replacement_key", 7),
        ("files", []),
    ),
)
def test_reader_maps_malformed_descriptor_fields_to_one_value_free_error(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def malformed(data: bytes) -> ClosedTreeDescriptor:
        return replace(_parse_manifest(data), **{field: value})

    error = _caught_read_error(lambda: _read(root, parse_manifest=malformed))

    assert error.kind is ClosedTreeReadErrorKind.INVALID_DESCRIPTOR


@pytest.mark.parametrize("case", ("descriptor", "file", "key"))
def test_reader_rejects_malformed_descriptor_runtime_shapes(
    tmp_path: Path,
    case: str,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def malformed(data: bytes) -> ClosedTreeDescriptor:
        descriptor = _parse_manifest(data)
        if case == "descriptor":
            return cast(ClosedTreeDescriptor, object())
        first, second = descriptor.files
        if case == "file":
            return replace(
                descriptor,
                files=(cast(ClosedTreeFileSpec, object()), second),
            )
        return replace(
            descriptor,
            files=(replace(first, key=cast(str, [])), second),
        )

    error = _caught_read_error(lambda: _read(root, parse_manifest=malformed))

    assert error.kind is ClosedTreeReadErrorKind.INVALID_DESCRIPTOR


@pytest.mark.parametrize("maximum", (1, True, 128 * 1024 * 1024 + 1))
def test_reader_rejects_invalid_manifest_bounds_before_touching_the_path(
    tmp_path: Path,
    maximum: object,
) -> None:
    with pytest.raises(ClosedTreeReadError) as error:
        read_closed_tree(
            tmp_path / "missing" / _MANIFEST_NAME,
            maximum_manifest_bytes=maximum,
            parse_manifest=_parse_manifest,
            parse_file=_parse_file,
            finish=_finish,
        )

    assert error.value.kind is ClosedTreeReadErrorKind.INVALID_DESCRIPTOR


def test_reader_rejects_exact_length_overflow(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    files = _fixture_files()
    _publish(root, files)
    (root / "alpha.json").write_bytes(files["alpha.json"] + b" ")
    (root / "alpha.json").chmod(0o600)

    _caught_read_error(lambda: _read(root))


def test_final_identity_pass_rejects_a_file_swapped_by_finish_callback(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    files = _fixture_files()
    _publish(root, files)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(files["alpha.json"])
    replacement.chmod(0o600)

    def swap_file(manifest: object, parsed: Mapping[object, object]) -> object:
        os.replace(replacement, root / "alpha.json")
        return _finish(manifest, parsed)

    _caught_read_error(lambda: _read(root, finish=swap_file))
    assert (root / "alpha.json").read_bytes() == files["alpha.json"]


def test_final_identity_pass_rejects_a_whole_root_swap(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    intruder = tmp_path / "intruder"
    saved = tmp_path / "saved-original"
    files = _fixture_files()
    _publish(root, files)
    _publish(intruder, files)

    def swap_root(manifest: object, parsed: Mapping[object, object]) -> object:
        os.rename(root, saved)
        os.rename(intruder, root)
        return _finish(manifest, parsed)

    _caught_read_error(lambda: _read(root, finish=swap_root))
    assert _tree(root) == files
    assert _tree(saved) == files


def test_reader_maps_initial_directory_identity_failure_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    real_fstat = tree_module.os.fstat
    calls = 0

    def fail_initial_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fixture-secret initial fstat")
        return real_fstat(descriptor)

    monkeypatch.setattr(tree_module.os, "fstat", fail_initial_fstat)
    error = _caught_read_error(lambda: _read(root))

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


def test_reader_maps_named_directory_stat_failure_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    real_stat = tree_module.os.stat

    def fail_root_stat(
        path: os.PathLike[str] | str | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == root and dir_fd is None and not follow_symlinks:
            raise OSError("fixture-secret root stat")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(tree_module.os, "stat", fail_root_stat)
    error = _caught_read_error(lambda: _read(root))

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


def test_reader_maps_read_and_post_read_stat_failures_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_root = tmp_path / "read-failure"
    stat_root = tmp_path / "stat-failure"
    _publish(read_root, _fixture_files())
    _publish(stat_root, _fixture_files())

    with monkeypatch.context() as patch:
        patch.setattr(
            tree_module.os,
            "read",
            lambda descriptor, maximum: (_ for _ in ()).throw(OSError("fixture-secret read")),
        )
        read_error = _caught_read_error(lambda: _read(read_root))

    real_stat = tree_module.os.stat

    def fail_manifest_stat(
        path: os.PathLike[str] | str | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == _MANIFEST_NAME and dir_fd is not None and not follow_symlinks:
            raise OSError("fixture-secret manifest stat")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(tree_module.os, "stat", fail_manifest_stat)
    stat_error = _caught_read_error(lambda: _read(stat_root))

    assert read_error.kind is stat_error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


def test_reader_maps_final_directory_identity_failure_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _publish(root, _fixture_files())

    def fail_during_finish(
        manifest: object,
        parsed: Mapping[object, object],
    ) -> object:
        monkeypatch.setattr(
            tree_module.os,
            "fstat",
            lambda descriptor: (_ for _ in ()).throw(OSError("fixture-secret final fstat")),
        )
        return _finish(manifest, parsed)

    error = _caught_read_error(lambda: _read(root, finish=fail_during_finish))

    assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY


def test_fsync_failure_never_exposes_a_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"

    def fail_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("fixture-secret lost fsync")

    monkeypatch.setattr(tree_module.os, "fsync", fail_fsync)
    error = _caught_publish_error(lambda: _publish(root, _fixture_files()))

    assert "fixture-secret" not in str(error)
    assert "fixture-secret" not in repr(error)
    assert not root.exists()
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))
    lock = tmp_path / ".artifact.lock"
    assert lock.is_file()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_publisher_sanitizes_unexpected_atomic_boundary_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    monkeypatch.setattr(
        tree_module,
        "_publish_locked",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("fixture-secret atomic failure")
        ),
    )

    error = _caught_publish_error(lambda: _publish(root, _fixture_files()))

    assert isinstance(error, ArtifactExportError)
    assert "fixture-secret" not in str(error)
    assert not root.exists()


def test_publisher_rejects_and_cleans_an_unsafe_staging_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    real_mkdtemp = tree_module.tempfile.mkdtemp

    def unsafe_mkdtemp(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(*args, **kwargs)
        Path(path).chmod(0o755)
        return path

    monkeypatch.setattr(tree_module.tempfile, "mkdtemp", unsafe_mkdtemp)
    error = _caught_publish_error(lambda: _publish(root, _fixture_files()))

    assert isinstance(error, ArtifactExportError)
    assert not root.exists()
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))


def test_publisher_rejects_an_unsafe_parent_without_creating_artifact_state(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "unsafe-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o777)
    root = parent / "artifact"

    error = _caught_publish_error(lambda: _publish(root, _fixture_files()))

    assert "public-alpha" not in str(error)
    assert not root.exists()
    assert not tuple(parent.glob(".artifact.*"))


def test_publisher_rejects_malformed_boundary_inputs_value_free(tmp_path: Path) -> None:
    files = _fixture_files()
    root = tmp_path / "artifact"

    class InvalidPath:
        def __fspath__(self) -> str:
            raise ValueError("fixture-secret invalid destination")

    class InvalidMapping(Mapping[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            if key == _MANIFEST_NAME:
                return files[key]
            raise KeyError(key)

        def __iter__(self):
            raise RuntimeError("fixture-secret mapping race")

        def __len__(self) -> int:
            return len(files)

    def invoke(
        *,
        output: object = root,
        publication_files: Mapping[str, bytes] = files,
        manifest_name: str = _MANIFEST_NAME,
        parse_manifest: Callable[[bytes], ClosedTreeDescriptor] = _parse_manifest,
        replace_tree: object = False,
    ) -> Exception:
        return _caught_publish_error(
            lambda: publish_closed_tree(
                output,
                publication_files,
                manifest_name=manifest_name,
                maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
                parse_manifest=parse_manifest,
                validate_tree=_validate_tree,
                replace=replace_tree,
            )
        )

    extra = {**files, "extra.json": b"{}"}
    undersized = {**files, "alpha.json": b"x"}
    cases = (
        invoke(output=b"fixture-secret/artifact"),
        invoke(output=InvalidPath()),
        invoke(output=Path(".")),
        invoke(replace_tree=1),
        invoke(manifest_name="../manifest.json"),
        invoke(publication_files={}),
        invoke(parse_manifest=lambda data: cast(ClosedTreeDescriptor, object())),
        invoke(publication_files=InvalidMapping()),
        invoke(publication_files=extra),
        invoke(publication_files=undersized),
    )

    assert all(isinstance(error, ArtifactExportError) for error in cases)
    assert all("fixture-secret" not in str(error) for error in cases)
    assert not root.exists()


@pytest.mark.parametrize("marker_bytes", (b"not-json", b'{"x":NaN}'))
def test_publisher_rejects_a_corrupt_recovery_marker_without_touching_the_tree(
    tmp_path: Path,
    marker_bytes: bytes,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement_files = _fixture_files(generation=2)
    _publish(root, original)
    marker = tmp_path / ".artifact.replace.json"
    marker.write_bytes(marker_bytes)
    marker.chmod(0o600)

    error = _caught_publish_error(lambda: _publish(root, replacement_files, replace=True))

    assert isinstance(error, ArtifactExportError)
    assert "public-alpha" not in str(error)
    assert _tree(root) == original
    assert marker.read_bytes() == marker_bytes


def test_replacement_cleanup_failure_preserves_both_complete_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement = _fixture_files(generation=2)
    _publish(root, original)
    real_rmtree = tree_module.shutil.rmtree

    def fail_backup_cleanup(path: os.PathLike[str] | str, *args: object, **kwargs: object) -> None:
        if Path(path).name == ".artifact.backup":
            raise OSError("fixture-secret cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(tree_module.shutil, "rmtree", fail_backup_cleanup)
    error = _caught_publish_error(lambda: _publish(root, replacement, replace=True))

    assert "fixture-secret" not in str(error)
    assert "fixture-secret" not in repr(error)
    assert _tree(root) == replacement
    assert _tree(tmp_path / ".artifact.backup") == original
    assert (tmp_path / ".artifact.replace.json").is_file()


def _prepare_recovery_state(
    root: Path,
    replacement_source: Path,
    *,
    state: str,
) -> tuple[Path, Path, Path, dict[str, bytes], dict[str, bytes]]:
    original = _fixture_files(generation=1)
    replacement = _fixture_files(generation=2)
    _publish(root, original)
    _publish(replacement_source, replacement)
    backup, marker = tree_module._replacement_paths(root, root.parent)
    marker_bytes = tree_module._replacement_marker_bytes(
        root,
        root.lstat(),
        replacement_source.lstat(),
        replacement_key="stable-tree",
        original_manifest_digest=sha256(original[_MANIFEST_NAME]).hexdigest(),
        replacement_manifest_digest=sha256(replacement[_MANIFEST_NAME]).hexdigest(),
    )
    tree_module._write_file(root.parent, marker.name, marker_bytes)

    if state == "published-only":
        os.rename(root, root.parent / "saved-original")
        os.rename(replacement_source, root)
    elif state == "backup-only":
        os.rename(root, backup)
    elif state == "both":
        os.rename(root, backup)
        os.rename(replacement_source, root)
    elif state != "original-only":
        raise AssertionError("unknown recovery fixture state")
    return root, backup, marker, original, replacement


@pytest.mark.parametrize(
    ("state", "expected_generation"),
    (
        ("original-only", 1),
        ("published-only", 2),
        ("backup-only", 1),
        ("both", 2),
    ),
)
def test_interrupted_replacement_recovers_every_authentic_commit_state(
    tmp_path: Path,
    state: str,
    expected_generation: int,
) -> None:
    root, backup, marker, original, replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state=state,
    )

    tree_module._recover_interrupted_replacement(
        root,
        tmp_path,
        validate_tree=_validate_tree,
        manifest_name=_MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
    )

    assert _tree(root) == (original if expected_generation == 1 else replacement)
    assert not backup.exists()
    assert not marker.exists()


def test_tree_low_level_mode_read_write_and_marker_races_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = tmp_path / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o600)
    metadata = value.stat()

    with monkeypatch.context() as patch:
        patch.setattr(tree_module, "_current_owner", lambda item: False)
        assert not tree_module._safe_read_mode(metadata, required_posix_mode=0o600)
    with monkeypatch.context() as patch:
        patch.setattr(tree_module.os, "name", "nt")
        assert tree_module._safe_read_mode(metadata, required_posix_mode=0o400)

    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(tree_module.os, "read", lambda descriptor, maximum: b"")
            error = _caught_read_error(
                lambda: tree_module._read_regular_file(
                    directory_fd,
                    value.name,
                    maximum=16,
                )
            )
            assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY

        with monkeypatch.context() as patch:
            real_stat = tree_module.os.stat

            def mismatch_named(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == value.name:
                    return os.fstat(directory_fd)
                return real_stat(path, *args, **kwargs)

            patch.setattr(tree_module.os, "stat", mismatch_named)
            error = _caught_read_error(
                lambda: tree_module._read_regular_file(
                    directory_fd,
                    value.name,
                    maximum=16,
                )
            )
            assert error.kind is ClosedTreeReadErrorKind.UNSAFE_ENTRY
    finally:
        os.close(directory_fd)

    with monkeypatch.context() as patch:
        patch.setattr(tree_module.os, "write", lambda descriptor, data: 0)
        with pytest.raises(OSError):
            tree_module._write_file(tmp_path, "short.json", b"{}")

    with monkeypatch.context() as patch:
        patch.setattr(tree_module, "_current_owner", lambda item: False)
        with pytest.raises(OSError):
            tree_module._write_file(tmp_path, "unsafe.json", b"{}")

    with monkeypatch.context() as patch:
        real_lstat = Path.lstat

        def mismatch_lstat(path: Path) -> os.stat_result:
            if path.name == "raced.json":
                return tmp_path.lstat()
            return real_lstat(path)

        patch.setattr(Path, "lstat", mismatch_lstat)
        with pytest.raises(OSError):
            tree_module._write_file(tmp_path, "raced.json", b"{}")

    marker = tmp_path / ".artifact.replace.json"
    marker.write_bytes(canonical_json({"x": 1}))
    marker.chmod(0o600)
    with pytest.raises(ArtifactExportError):
        tree_module._read_replacement_marker(marker, tmp_path / "artifact")


def test_tree_removal_and_validation_helpers_cover_negative_edges(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    fake = tree_module._PathIdentity(0, 0, 0, 0, 0, 0, 0, 0)
    assert not tree_module._remove_owned_directory(missing, fake)
    assert tree_module._remove_owned_staging(missing, fake)

    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    assert not tree_module._remove_owned_staging(directory, fake)

    regular = tmp_path / "value.json"
    regular.write_bytes(b"{}")
    regular.chmod(0o600)
    assert not tree_module._unlink_owned_regular(regular, fake)

    root = tmp_path / "artifact"
    _publish(root, _fixture_files())
    descriptor = _parse_manifest(_fixture_files()[_MANIFEST_NAME])
    wrong_digest = "0" * 64
    assert (
        tree_module._validated_tree_descriptor(
            root,
            expected_digest=wrong_digest,
            validate_tree=lambda path, expected: descriptor,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        is None
    )
    assert (
        tree_module._validated_tree_descriptor(
            root,
            expected_digest=None,
            validate_tree=lambda path, expected: (_ for _ in ()).throw(
                RuntimeError("fixture-secret validator")
            ),
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        is None
    )


def test_destination_lock_rejects_parent_lock_and_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    identity = tree_module._PathIdentity.from_stat(tmp_path.stat())
    wrong_identity = replace(identity, inode=identity.inode + 1)
    with (
        pytest.raises(ArtifactDestinationError),
        tree_module._destination_lock(root, tmp_path, wrong_identity),
    ):
        raise AssertionError("unsafe parent unexpectedly locked")

    lock = tmp_path / ".artifact.lock"
    lock.write_bytes(b"")
    lock.chmod(0o640)
    with (
        pytest.raises(ArtifactDestinationError),
        tree_module._destination_lock(root, tmp_path, identity),
    ):
        raise AssertionError("unsafe lock unexpectedly opened")
    lock.chmod(0o600)

    with monkeypatch.context() as patch:
        real_lstat = Path.lstat

        def mismatch_lock(path: Path) -> os.stat_result:
            if path == lock:
                return tmp_path.lstat()
            return real_lstat(path)

        patch.setattr(Path, "lstat", mismatch_lock)
        with (
            pytest.raises(ArtifactDestinationError),
            tree_module._destination_lock(root, tmp_path, identity),
        ):
            raise AssertionError("raced lock unexpectedly opened")

    with monkeypatch.context() as patch:
        real_safe_parent = tree_module._safe_parent
        checks = 0

        def fail_second_parent_check(
            metadata: os.stat_result,
            expected: tree_module._PathIdentity,
        ) -> bool:
            nonlocal checks
            checks += 1
            return checks != 2 and real_safe_parent(metadata, expected)

        patch.setattr(tree_module, "_safe_parent", fail_second_parent_check)
        with (
            pytest.raises(ArtifactDestinationError),
            tree_module._destination_lock(root, tmp_path, identity),
        ):
            raise AssertionError("raced parent unexpectedly locked")

    with monkeypatch.context() as patch:
        patch.setattr(
            tree_module.os,
            "open",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fixture-secret lock open")),
        )
        with (
            pytest.raises(ArtifactDestinationError) as error,
            tree_module._destination_lock(root, tmp_path, identity),
        ):
            raise AssertionError("failed lock unexpectedly opened")
        assert "fixture-secret" not in str(error.value)


def test_exclusive_tree_adapter_maps_invalid_descriptor_and_validator_failures(
    tmp_path: Path,
) -> None:
    files = _fixture_files()

    with pytest.raises(ArtifactExportError):
        tree_module.publish_closed_tree_exclusive(
            tmp_path / "invalid-bound",
            files,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=True,
            parse_manifest=_parse_manifest,
            validate_tree=_validate_tree,
        )

    def invalid_descriptor(data: bytes) -> ClosedTreeDescriptor:
        return replace(_parse_manifest(data), manifest_digest="invalid")

    with pytest.raises(ArtifactExportError):
        tree_module.publish_closed_tree_exclusive(
            tmp_path / "invalid-descriptor",
            files,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
            parse_manifest=invalid_descriptor,
            validate_tree=_validate_tree,
        )

    root = tmp_path / "validator-failure"
    with pytest.raises(ArtifactDestinationError) as error:
        tree_module.publish_closed_tree_exclusive(
            root,
            files,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
            parse_manifest=_parse_manifest,
            validate_tree=lambda path, digest: (_ for _ in ()).throw(
                RuntimeError("fixture-secret validator")
            ),
        )
    assert "fixture-secret" not in str(error.value)
    assert _tree(root) == files


def test_publisher_rejects_a_well_formed_wrong_staging_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"

    def wrong_digest(path: Path, expected: str | None) -> ClosedTreeDescriptor:
        del expected
        return replace(_validate_tree(path, None), manifest_digest="0" * 64)

    error = _caught_publish_error(
        lambda: _publish(root, _fixture_files(), validate_tree=wrong_digest)
    )

    assert isinstance(error, ArtifactExportError)
    assert not root.exists()
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))


def test_replacement_authorization_race_preserves_the_original_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement_files = _fixture_files(generation=2)
    _publish(root, original)
    mutated = False

    def rewrite_original_during_staging(
        path: Path,
        expected: str | None,
    ) -> ClosedTreeDescriptor:
        nonlocal mutated
        descriptor = _validate_tree(path, expected)
        if path.name.startswith(".artifact.tmp-") and not mutated:
            manifest = root / _MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes())
            manifest.chmod(0o600)
            mutated = True
        return descriptor

    error = _caught_publish_error(
        lambda: _publish(
            root,
            replacement_files,
            replace=True,
            validate_tree=rewrite_original_during_staging,
        )
    )

    assert isinstance(error, ArtifactExistsError)
    assert _tree(root) == original
    assert not (tmp_path / ".artifact.backup").exists()
    assert not (tmp_path / ".artifact.replace.json").exists()
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))


def test_backup_validation_failure_is_rolled_back_and_cleans_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement_files = _fixture_files(generation=2)
    _publish(root, original)

    def fail_backup(path: Path, expected: str | None) -> ClosedTreeDescriptor:
        if path.name == ".artifact.backup":
            raise RuntimeError("fixture-secret backup validation")
        return _validate_tree(path, expected)

    error = _caught_publish_error(
        lambda: _publish(
            root,
            replacement_files,
            replace=True,
            validate_tree=fail_backup,
        )
    )

    assert isinstance(error, ArtifactExistsError)
    assert "fixture-secret" not in str(error)
    assert _tree(root) == original
    assert not (tmp_path / ".artifact.backup").exists()
    assert not (tmp_path / ".artifact.replace.json").exists()
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))


def test_missing_staging_after_backup_validation_restores_original(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    saved_staging = tmp_path / "saved-staging"
    original = _fixture_files(generation=1)
    replacement_files = _fixture_files(generation=2)
    _publish(root, original)
    moved = False

    def move_staging_after_backup(
        path: Path,
        expected: str | None,
    ) -> ClosedTreeDescriptor:
        nonlocal moved
        descriptor = _validate_tree(path, expected)
        if path.name == ".artifact.backup" and not moved:
            staging = next(tmp_path.glob(".artifact.tmp-*"))
            os.rename(staging, saved_staging)
            moved = True
        return descriptor

    error = _caught_publish_error(
        lambda: _publish(
            root,
            replacement_files,
            replace=True,
            validate_tree=move_staging_after_backup,
        )
    )

    assert isinstance(error, ArtifactExistsError)
    assert _tree(root) == original
    assert _tree(saved_staging) == replacement_files
    assert not (tmp_path / ".artifact.backup").exists()
    assert not (tmp_path / ".artifact.replace.json").exists()


def test_post_publish_validation_failure_keeps_a_complete_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    files = _fixture_files()

    def fail_destination(path: Path, expected: str | None) -> ClosedTreeDescriptor:
        if path == root:
            raise RuntimeError("fixture-secret post publish")
        return _validate_tree(path, expected)

    error = _caught_publish_error(lambda: _publish(root, files, validate_tree=fail_destination))

    assert isinstance(error, ArtifactDestinationError)
    assert "fixture-secret" not in str(error)
    assert _tree(root) == files
    assert not tuple(tmp_path.glob(".artifact.tmp-*"))


def test_marker_cleanup_failure_preserves_the_complete_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    replacement_files = _fixture_files(generation=2)
    _publish(root, original)
    real_unlink = tree_module._unlink_owned_regular

    def reject_marker(path: Path, identity: object) -> bool:
        if path.name == ".artifact.replace.json":
            return False
        return real_unlink(path, identity)

    monkeypatch.setattr(tree_module, "_unlink_owned_regular", reject_marker)
    error = _caught_publish_error(lambda: _publish(root, replacement_files, replace=True))

    assert isinstance(error, ArtifactDestinationError)
    assert _tree(root) == replacement_files
    assert not (tmp_path / ".artifact.backup").exists()
    assert (tmp_path / ".artifact.replace.json").is_file()


@pytest.mark.parametrize("failed_check", (3, 4))
def test_publisher_rechecks_parent_before_and_after_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_check: int,
) -> None:
    root = tmp_path / f"artifact-{failed_check}"
    files = _fixture_files()
    real_safe_parent = tree_module._safe_parent
    checks = 0

    def reject_selected_check(
        metadata: os.stat_result,
        expected: tree_module._PathIdentity,
    ) -> bool:
        nonlocal checks
        checks += 1
        return checks != failed_check and real_safe_parent(metadata, expected)

    monkeypatch.setattr(tree_module, "_safe_parent", reject_selected_check)
    error = _caught_publish_error(lambda: _publish(root, files))

    assert isinstance(error, ArtifactDestinationError)
    if failed_check == 3:
        assert not root.exists()
    else:
        assert _tree(root) == files
