"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import saliencegate.security.files as files
from saliencegate.security import (
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor tests require POSIX")


def _open_private_directory(path: Path) -> tuple[files._PrivateDirectoryAuthorization, int]:
    directory = files._authorize_private_directory(path, create=True)
    return directory, files._open_authorized_private_directory(directory)


def _publish(
    directory: files._PrivateDirectoryAuthorization,
    descriptor: int,
    name: str,
    data: bytes,
    *,
    validate_replacement: Callable[[bytes], bool] | None = None,
    validate_published: Callable[[bytes], bool] | None = None,
) -> files.StableFileRead:
    return files._publish_private_file_at_descriptor(
        directory,
        descriptor,
        name,
        data,
        maximum_bytes=128,
        validate_replacement=validate_replacement,
        validate_published=validate_published,
    )


def test_descriptor_replacement_rejection_restores_the_original_inode_contents(
    tmp_path: Path,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    target = Path(directory.path) / "result.json"
    try:
        _publish(directory, descriptor, target.name, b"old")

        with pytest.raises(SecureFileError):
            _publish(
                directory,
                descriptor,
                target.name,
                b"new",
                validate_replacement=lambda value: value == b"old",
                validate_published=lambda _value: False,
            )

        assert target.read_bytes() == b"old"
        assert not tuple(target.parent.glob(".saliencegate-atomic-*"))
    finally:
        os.close(descriptor)


def test_descriptor_replacement_rebuilds_a_lost_backup_before_rollback(
    tmp_path: Path,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    target = Path(directory.path) / "result.json"
    try:
        _publish(directory, descriptor, target.name, b"old")

        def remove_backup_and_reject(_value: bytes) -> bool:
            backups = tuple(target.parent.glob(".saliencegate-atomic-*"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == b"old"
            backups[0].unlink()
            return False

        with pytest.raises(SecureFileError):
            _publish(
                directory,
                descriptor,
                target.name,
                b"new",
                validate_replacement=lambda value: value == b"old",
                validate_published=remove_backup_and_reject,
            )

        assert target.read_bytes() == b"old"
        assert not tuple(target.parent.glob(".saliencegate-atomic-*"))
    finally:
        os.close(descriptor)


def test_descriptor_absent_publication_rejection_removes_the_new_name(
    tmp_path: Path,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    target = Path(directory.path) / "result.json"
    try:
        with pytest.raises(SecureFileError):
            _publish(
                directory,
                descriptor,
                target.name,
                b"new",
                validate_published=lambda _value: False,
            )
        assert not target.exists()
        assert not tuple(target.parent.glob(".saliencegate-atomic-*"))
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("maximum_bytes", "data", "replacement", "published", "error"),
    (
        (0, b"x", None, None, SecureFileBoundError),
        (1, "x", None, None, SecureFileError),
        (1, b"xx", None, None, SecureFileBoundError),
        (1, b"x", object(), None, SecureFileError),
        (1, b"x", None, object(), SecureFileError),
    ),
)
def test_descriptor_publication_rejects_every_public_argument_boundary(
    tmp_path: Path,
    maximum_bytes: int,
    data: object,
    replacement: object,
    published: object,
    error: type[Exception],
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    try:
        with pytest.raises(error):
            files._publish_private_file_at_descriptor(
                directory,
                descriptor,
                "result.json",
                data,  # type: ignore[arg-type]
                maximum_bytes=maximum_bytes,
                validate_replacement=replacement,  # type: ignore[arg-type]
                validate_published=published,  # type: ignore[arg-type]
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("callback_result", (False, RuntimeError("callback secret")))
def test_descriptor_replacement_callback_failures_are_sanitized(
    tmp_path: Path,
    callback_result: object,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    target = Path(directory.path) / "result.json"
    try:
        _publish(directory, descriptor, target.name, b"old")

        def validate(_value: bytes) -> bool:
            if isinstance(callback_result, BaseException):
                raise callback_result
            return bool(callback_result)

        with pytest.raises(SecureFileError) as captured:
            _publish(
                directory,
                descriptor,
                target.name,
                b"new",
                validate_replacement=validate,
            )
        assert "callback secret" not in repr(captured.value)
        assert target.read_bytes() == b"old"
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileBoundError(), SecureFileBoundError),
        (files._UnsupportedFileOperationError(), SecureFileUnsupportedError),
        (OSError(errno.ENOTSUP, "unsupported secret"), SecureFileUnsupportedError),
        (OSError(errno.EIO, "io secret"), SecureFileError),
        (RuntimeError("runtime secret"), SecureFileError),
    ),
)
def test_descriptor_publication_normalizes_internal_failure_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected: type[Exception],
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")

    def fail(*_args: object, **_kwargs: object) -> files.StableFileRead:
        raise failure

    monkeypatch.setattr(files, "_publish_private_file_at_descriptor_unchecked", fail)
    try:
        with pytest.raises(expected) as captured:
            _publish(directory, descriptor, "result.json", b"new")
        assert "secret" not in repr(captured.value)
    finally:
        os.close(descriptor)


def test_descriptor_publication_never_swallows_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")

    def interrupt(*_args: object, **_kwargs: object) -> files.StableFileRead:
        raise KeyboardInterrupt

    monkeypatch.setattr(files, "_publish_private_file_at_descriptor_unchecked", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            _publish(directory, descriptor, "result.json", b"new")
    finally:
        os.close(descriptor)


def test_descriptor_delete_validation_failure_restores_the_staged_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    target = Path(directory.path) / "entry"
    try:
        stable = _publish(directory, descriptor, target.name, b"keep")

        def reject(*_args: object, **_kwargs: object) -> None:
            raise files._UnsafeFilePathError

        monkeypatch.setattr(files, "_validate_staged_private_read_target_at", reject)
        with pytest.raises(files._UnsafeFilePathError):
            files._delete_authorized_private_file_at_descriptor(
                directory,
                descriptor,
                stable.authorization,
            )

        assert target.read_bytes() == b"keep"
        assert not tuple(target.parent.glob(".saliencegate-delete-*"))
    finally:
        os.close(descriptor)


def test_delete_stage_creation_cleans_a_directory_after_acl_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")

    def reject(_descriptor: int, *, deny_only_allowed: bool = False) -> None:
        del deny_only_allowed
        raise files._UnsafeFilePathError

    monkeypatch.setattr(files, "_require_safe_acl", reject)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._create_private_delete_stage(descriptor, forbidden_name="entry")
        assert not tuple(Path(directory.path).glob(".saliencegate-delete-*"))
    finally:
        os.close(descriptor)


def test_delete_stage_creation_exhausts_colliding_names_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    collision = Path(directory.path) / ".saliencegate-delete-collision"
    collision.mkdir(mode=0o700)
    monkeypatch.setattr(files.secrets, "token_hex", lambda _length: "collision")
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._create_private_delete_stage(descriptor, forbidden_name="entry")
        assert tuple(Path(directory.path).iterdir()) == (collision,)
    finally:
        os.close(descriptor)


def _dummy_authorization(
    kind: files._AuthorizationKind = files._AuthorizationKind.SQLITE,
) -> files.StableFileAuthorization:
    return files.StableFileAuthorization(
        path="/private/redacted",
        _parent_identity=None,
        _target_identity=None,
        _kind=kind,
    )


def _dummy_directory_authorization() -> files._PrivateDirectoryAuthorization:
    return files._PrivateDirectoryAuthorization(
        path="/private/redacted",
        _identity=files._StableIdentity(device=1, inode=2, mode=0, owner=0),
    )


def _atomic_publication() -> files.AtomicFilePublication:
    return files.AtomicFilePublication(
        _dummy_authorization(files._AuthorizationKind.PRIVATE_LOCATION),
        8,
        None,
        _token=files._ATOMIC_PUBLICATION_TOKEN,
    )


def test_authorization_kind_specific_accessors_fail_closed() -> None:
    with pytest.raises(SecureFileError):
        _ = _dummy_authorization().target_exists
    with pytest.raises(SecureFileError):
        _dummy_authorization(files._AuthorizationKind.STABLE_READ)._revalidate_mutable_sqlite()


@pytest.mark.parametrize(
    ("data", "authorization"),
    (
        (bytearray(), _dummy_authorization()),
        (b"", object()),
    ),
)
def test_stable_file_read_rejects_inexact_field_types(
    data: object,
    authorization: object,
) -> None:
    with pytest.raises(SecureFileError):
        files.StableFileRead(
            data=data,  # type: ignore[arg-type]
            authorization=authorization,  # type: ignore[arg-type]
        )


def test_stable_file_line_iterator_rejects_boolean_bounds() -> None:
    stable = files.StableFileRead(data=b"line", authorization=_dummy_authorization())
    with pytest.raises(SecureFileBoundError):
        tuple(stable.iter_lines(maximum_line_bytes=True, maximum_lines=1))


def test_atomic_publication_constructor_rejects_an_untrusted_token() -> None:
    with pytest.raises(SecureFileError):
        files.AtomicFilePublication(
            _dummy_authorization(files._AuthorizationKind.PRIVATE_LOCATION),
            8,
            None,
            _token=object(),
        )


def test_atomic_publication_rejects_inexact_publish_arguments() -> None:
    with pytest.raises(SecureFileError):
        _atomic_publication().publish(bytearray())  # type: ignore[arg-type]
    with pytest.raises(SecureFileError):
        _atomic_publication().publish(b"data", validate_published=object())  # type: ignore[arg-type]
    assert repr(_atomic_publication()) == "AtomicFilePublication(<redacted>)"


def test_preferred_failure_preserves_base_exception_priority() -> None:
    primary = KeyboardInterrupt()
    secondary = SystemExit()
    assert files._preferred_failure(primary, RuntimeError()) is primary
    assert files._preferred_failure(RuntimeError(), secondary) is secondary


@pytest.mark.parametrize(
    ("error_number", "expected"),
    (
        (errno.EINVAL, files._UnsupportedFileOperationError),
        (errno.EIO, OSError),
    ),
)
def test_directory_fsync_probe_classifies_os_failures(
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    expected: type[BaseException],
) -> None:
    def fail(_descriptor: int) -> None:
        raise OSError(error_number, "probe secret")

    monkeypatch.setattr(files.os, "fsync", fail)
    with pytest.raises(expected):
        files._probe_directory_fsync(7)


def test_atomic_platform_gate_normalizes_a_missing_secure_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject() -> None:
        raise files._UnsafeFilePathError

    monkeypatch.setattr(files, "_require_secure_platform", reject)
    with pytest.raises(files._UnsupportedFileOperationError):
        files._require_atomic_publication_platform()


@pytest.mark.parametrize("copier", (files._copy_path, files._copy_legacy_read_path))
def test_path_copiers_reject_a_root_without_a_leaf(
    copier: Callable[[str], Path],
) -> None:
    with pytest.raises(files._UnsafeFilePathError):
        copier(os.sep)


def test_opened_stable_read_rejects_a_short_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    descriptor = os.open(target, os.O_RDONLY)
    monkeypatch.setattr(files, "_read_descriptor_bounded", lambda *_args: b"")
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._read_opened_stable_file(
                descriptor,
                os.stat(target),
                lambda: os.stat(target),
                maximum_bytes=1,
                policy=files.StableReadPolicy.LEGACY_COMPATIBILITY,
                check_acl=False,
            )
    finally:
        os.close(descriptor)


def test_private_read_rejects_an_unknown_policy() -> None:
    with pytest.raises(files._UnsafeFilePathError):
        files._read_private_file(
            Path("/private/redacted"),
            1,
            policy=object(),  # type: ignore[arg-type]
        )


def test_private_directory_descriptor_rejects_invalid_and_regular_fds(
    tmp_path: Path,
) -> None:
    directory = _dummy_directory_authorization()
    with pytest.raises(files._UnsafeFilePathError):
        files._require_authorized_private_directory_descriptor(directory, -1)

    target = tmp_path / "regular"
    target.touch(mode=0o600)
    descriptor = os.open(target, os.O_RDONLY)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._require_authorized_private_directory_descriptor(directory, descriptor)
    finally:
        os.close(descriptor)


def test_private_directory_revalidation_reports_a_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "_open_authorized_private_directory", lambda _value: 7)

    def fail_close(_descriptor: int) -> None:
        raise OSError("close secret")

    monkeypatch.setattr(files.os, "close", fail_close)
    with pytest.raises(SecureFileError):
        _dummy_directory_authorization().revalidate()


def test_acl_probe_skips_darwin_apis_on_other_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files.sys, "platform", "linux")
    assert files._darwin_acl_is_unsafe(7, deny_only_allowed=False) is False


def test_private_directory_platform_requires_mkdirat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported = set(files.os.supports_dir_fd)
    supported.discard(files.os.mkdir)
    monkeypatch.setattr(files.os, "supports_dir_fd", supported)
    with pytest.raises(files._UnsafeFilePathError):
        files._require_private_directory_platform()


@pytest.mark.parametrize(
    "operation",
    (
        lambda: files._open_or_create_private_directory_chain(Path("relative")),
        lambda: files._open_safe_ancestor_directory(Path("relative")),
        lambda: files._inspect_private_directory_boundary(Path("relative")),
        lambda: files._open_directory_chain(Path("relative")),
    ),
)
def test_directory_walkers_reject_relative_paths(
    operation: Callable[[], object],
) -> None:
    with pytest.raises(files._UnsafeFilePathError):
        operation()


def test_private_directory_boundary_rejects_the_root() -> None:
    with pytest.raises(files._UnsafeFilePathError):
        files._inspect_private_directory_boundary(Path(os.sep))


def test_private_directory_helpers_reject_inexact_control_values() -> None:
    with pytest.raises(SecureFileError):
        files._authorize_private_directory("/private/redacted", create=1)  # type: ignore[arg-type]
    with pytest.raises(files._UnsafeFilePathError):
        files._open_authorized_private_directory(object())  # type: ignore[arg-type]
    with pytest.raises(SecureFileError):
        files._authorize_private_directory_child(
            _dummy_directory_authorization(),
            "child",
            create=1,  # type: ignore[arg-type]
        )
    with pytest.raises(files._UnsafeFilePathError):
        files._copy_private_child_name("")


def test_sqlite_sidecar_validation_rejects_malformed_authorizations() -> None:
    with pytest.raises(files._UnsafeFilePathError):
        files._validate_sqlite_sidecars(7, "capture.sqlite3", (), strict_transient=True)


@pytest.mark.parametrize(
    "validator",
    (files._validate_private_read_target_at, files._validate_staged_private_read_target_at),
)
def test_private_target_validators_require_complete_identity(
    validator: Callable[[int, str, object], None],
) -> None:
    with pytest.raises(files._UnsafeFilePathError):
        validator(7, "target", object())


class _CFunction:
    def __init__(self, implementation: Callable[..., object]) -> None:
        self.implementation = implementation
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        return self.implementation(*args)


@pytest.fixture
def cleared_darwin_acl_api_cache() -> Iterator[None]:
    files._darwin_acl_api.cache_clear()
    try:
        yield
    finally:
        files._darwin_acl_api.cache_clear()


@pytest.mark.parametrize(
    "scenario",
    ("entry_error", "entry_invalid", "tag_error", "deny_entry", "free_error"),
)
def test_darwin_acl_probe_covers_native_failure_and_deny_entry_paths(
    monkeypatch: pytest.MonkeyPatch,
    cleared_darwin_acl_api_cache: None,
    scenario: str,
) -> None:
    del cleared_darwin_acl_api_cache
    entry_calls = 0

    def get_entry(_acl: object, _entry_id: object, entry_pointer: object) -> int:
        nonlocal entry_calls
        entry_calls += 1
        if scenario == "entry_error":
            files.ctypes.set_errno(0)
            return -1
        if scenario == "entry_invalid":
            return 1
        if scenario == "deny_entry" and entry_calls > 1:
            files.ctypes.set_errno(errno.EINVAL)
            return -1
        entry_pointer._obj.value = 1  # type: ignore[attr-defined]
        return 0

    def get_tag(_entry: object, tag_pointer: object) -> int:
        if scenario == "tag_error":
            return 1
        tag_pointer._obj.value = files._DARWIN_ACL_EXTENDED_DENY  # type: ignore[attr-defined]
        return 0

    library = type(
        "FakeAclLibrary",
        (),
        {
            "acl_get_fd_np": _CFunction(lambda *_args: 1),
            "acl_get_entry": _CFunction(get_entry),
            "acl_get_tag_type": _CFunction(get_tag),
            "acl_free": _CFunction(lambda _acl: int(scenario == "free_error")),
        },
    )()
    monkeypatch.setattr(files.sys, "platform", "darwin")
    monkeypatch.setattr(files.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    assert files._darwin_acl_is_unsafe(
        7,
        deny_only_allowed=scenario != "free_error",
    ) is (scenario != "deny_entry")


def test_darwin_acl_api_binds_once_but_checks_and_frees_each_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    cleared_darwin_acl_api_cache: None,
) -> None:
    del cleared_darwin_acl_api_cache
    library_calls = 0
    checked_descriptors: list[int] = []
    freed_acls: list[object] = []

    def load_library(*_args: object, **_kwargs: object) -> object:
        nonlocal library_calls
        library_calls += 1
        return library

    def get_acl(descriptor: int, _acl_type: int) -> int:
        checked_descriptors.append(descriptor)
        return descriptor + 100

    def get_entry(_acl: object, _entry_id: object, _entry_pointer: object) -> int:
        files.ctypes.set_errno(errno.EINVAL)
        return -1

    library = type(
        "FakeAclLibrary",
        (),
        {
            "acl_get_fd_np": _CFunction(get_acl),
            "acl_get_entry": _CFunction(get_entry),
            "acl_get_tag_type": _CFunction(lambda *_args: 0),
            "acl_free": _CFunction(lambda acl: freed_acls.append(acl) or 0),
        },
    )()
    monkeypatch.setattr(files.sys, "platform", "darwin")
    monkeypatch.setattr(files.ctypes, "CDLL", load_library)

    assert files._darwin_acl_is_unsafe(7, deny_only_allowed=False) is True
    assert files._darwin_acl_is_unsafe(8, deny_only_allowed=True) is False
    assert library_calls == 1
    assert checked_descriptors == [7, 8]
    assert freed_acls == [107, 108]


def test_darwin_acl_api_binding_failure_is_fail_closed_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    cleared_darwin_acl_api_cache: None,
) -> None:
    del cleared_darwin_acl_api_cache
    attempts = 0

    def reject_library(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise OSError("native binding secret")

    monkeypatch.setattr(files.sys, "platform", "darwin")
    monkeypatch.setattr(files.ctypes, "CDLL", reject_library)

    assert files._darwin_acl_is_unsafe(7, deny_only_allowed=False) is True
    assert files._darwin_acl_is_unsafe(8, deny_only_allowed=True) is True
    assert attempts == 2


def test_private_directory_creation_rejects_unsafe_root_and_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_or_create_private_directory_chain(Path("/private"))

    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: True)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_or_create_private_directory_chain(Path("/private/../target"))


def test_private_directory_creation_rejects_identity_and_final_mode_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "leaf"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(files._StableIdentity, "matches", lambda *_args: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_or_create_private_directory_chain(target)

    checks = 0

    def private_until_final(_value: os.stat_result) -> bool:
        nonlocal checks
        checks += 1
        return checks < 3

    with monkeypatch.context() as patch:
        patch.setattr(files._StableIdentity, "matches", lambda *_args: True)
        patch.setattr(files, "_safe_private_directory", private_until_final)
        with pytest.raises(files._UnsafeFilePathError):
            files._open_or_create_private_directory_chain(target)


def test_safe_ancestor_walk_rejects_root_component_and_identity_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_safe_ancestor_directory(tmp_path)

    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: True)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_safe_ancestor_directory(Path("/private/../target"))

    checks = 0

    def reject_first_child(_value: os.stat_result) -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    with monkeypatch.context() as patch:
        patch.setattr(files, "_safe_ancestor", reject_first_child)
        patch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
        with pytest.raises(files._UnsafeFilePathError):
            files._open_safe_ancestor_directory(tmp_path)


def test_safe_ancestor_walk_rechecks_the_final_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    expected_calls = 2 * len(tmp_path.parts[1:]) + 2

    def reject_final(_value: os.stat_result) -> bool:
        nonlocal calls
        calls += 1
        return calls != expected_calls

    monkeypatch.setattr(files, "_safe_ancestor", reject_final)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_safe_ancestor_directory(tmp_path)


def test_private_boundary_inspection_rejects_root_and_leaf_identity_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._inspect_private_directory_boundary(tmp_path / "missing")

    target = tmp_path / "leaf"
    target.mkdir(mode=0o700)
    with monkeypatch.context() as patch:
        patch.setattr(files._StableIdentity, "matches", lambda *_args: False)
        patch.setattr(files, "_safe_ancestor", lambda _value: True)
        with pytest.raises(files._UnsafeFilePathError):
            files._inspect_private_directory_boundary(target)


def test_private_boundary_inspection_rejects_a_changed_absent_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_descriptors: list[int] = []

    def changed_parent(_path: Path) -> tuple[int, files._StableIdentity]:
        descriptor = os.open(os.sep, os.O_RDONLY)
        fresh_descriptors.append(descriptor)
        return descriptor, files._StableIdentity(device=-1, inode=-1, mode=0, owner=0)

    monkeypatch.setattr(files, "_open_safe_ancestor_directory", changed_parent)
    with pytest.raises(files._UnsafeFilePathError):
        files._inspect_private_directory_boundary(tmp_path / "missing")
    assert fresh_descriptors


def test_directory_chain_rejects_unsafe_root_component_and_final_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_directory_chain(tmp_path)

    monkeypatch.setattr(files, "_safe_ancestor", lambda _value: True)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_directory_chain(Path("/private/../target"))

    monkeypatch.setattr(files, "_safe_parent", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_directory_chain(tmp_path)


def test_private_directory_snapshot_rejects_first_and_second_walk_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "leaf"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(files, "_safe_private_directory", lambda _value: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_private_directory_snapshot(target)

    first = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    second = files._StableIdentity(device=2, inode=2, mode=0, owner=0)
    identities = iter((first, second))

    def open_walk(_path: Path) -> tuple[int, files._StableIdentity]:
        return os.open(target, os.O_RDONLY), next(identities)

    with monkeypatch.context() as patch:
        patch.setattr(files, "_open_directory_chain", open_walk)
        patch.setattr(files, "_safe_private_directory", lambda _value: True)
        patch.setattr(files._StableIdentity, "matches", lambda *_args: True)
        with pytest.raises(files._UnsafeFilePathError):
            files._open_private_directory_snapshot(target)


def test_private_directory_authorization_covers_close_identity_and_base_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    with monkeypatch.context() as patch:
        patch.setattr(files, "_require_private_directory_platform", lambda: None)
        patch.setattr(
            files,
            "_open_or_create_private_directory_chain",
            lambda _path: (7, identity),
        )
        patch.setattr(files, "_close_independent_descriptors", lambda *_args: OSError())
        with pytest.raises(SecureFileError):
            files._authorize_private_directory("/private/target", create=True)

    with monkeypatch.context() as patch:
        patch.setattr(files, "_require_private_directory_platform", lambda: None)
        patch.setattr(
            files,
            "_open_or_create_private_directory_chain",
            lambda _path: (7, identity),
        )
        patch.setattr(files, "_close_independent_descriptors", lambda *_args: None)
        patch.setattr(
            files,
            "_open_private_directory_snapshot",
            lambda _path: (8, files._StableIdentity(device=2, inode=2, mode=0, owner=0)),
        )
        with pytest.raises(SecureFileError):
            files._authorize_private_directory("/private/target", create=True)

    with monkeypatch.context() as patch:
        patch.setattr(
            files,
            "_require_secure_platform",
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            files._authorize_private_directory("/private/target", create=False)


def test_authorized_directory_rejects_noncanonical_copied_path() -> None:
    authorization = files._PrivateDirectoryAuthorization(
        path="/private/target/",
        _identity=files._StableIdentity(device=1, inode=1, mode=0, owner=0),
    )
    with pytest.raises(files._UnsafeFilePathError):
        files._open_authorized_private_directory(authorization)


def test_private_child_authorization_rejects_unsafe_metadata_and_base_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = files._authorize_private_directory(tmp_path / "parent", create=True)
    monkeypatch.setattr(files, "_safe_private_directory", lambda _value: False)
    with pytest.raises(SecureFileError):
        files._authorize_private_directory_child(parent, "child", create=True)

    with monkeypatch.context() as patch:
        patch.setattr(files, "_require_private_directory_platform", lambda: None)
        patch.setattr(
            files,
            "_open_authorized_private_directory",
            lambda _value: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            files._authorize_private_directory_child(parent, "other", create=False)


def test_parent_open_and_verification_reject_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    second = files._StableIdentity(device=2, inode=2, mode=0, owner=0)
    identities = iter((first, second))

    def open_walk(_path: Path) -> tuple[int, files._StableIdentity]:
        return os.open(os.sep, os.O_RDONLY), next(identities)

    monkeypatch.setattr(files, "_open_directory_chain", open_walk)
    with pytest.raises(files._UnsafeFilePathError):
        files._open_parent(tmp_path / "target")

    descriptor = os.open(os.sep, os.O_RDONLY)
    try:
        monkeypatch.setattr(
            files,
            "_open_directory_chain",
            lambda _path: (os.open(os.sep, os.O_RDONLY), first),
        )
        monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
        with pytest.raises(files._UnsafeFilePathError):
            files._verify_parent(tmp_path / "target", descriptor, second)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "validator",
    (files._validate_existing_target, files._validate_existing_mutable_target),
)
def test_existing_target_validators_reject_an_unsafe_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validator: Callable[..., object],
) -> None:
    target = tmp_path / "target"
    target.touch(mode=0o600)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    unsafe_values = list(os.stat(target))
    unsafe_values[0] = stat.S_IFREG | 0o644
    unsafe = os.stat_result(unsafe_values)
    monkeypatch.setattr(files.os, "fstat", lambda _descriptor: unsafe)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            validator(directory_fd, target.name)
    finally:
        os.close(directory_fd)


def test_mutable_target_rejects_a_post_open_metadata_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.touch(mode=0o600)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    calls = 0

    def safe_then_changed(_value: os.stat_result) -> bool:
        nonlocal calls
        calls += 1
        return calls < 3

    monkeypatch.setattr(files, "_safe_target", safe_then_changed)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._validate_existing_mutable_target(directory_fd, target.name)
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("failure_call", (1, 2))
def test_target_creation_cleans_up_unsafe_initial_and_final_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    calls = 0

    def safe_until_failure(_value: os.stat_result) -> bool:
        nonlocal calls
        calls += 1
        return calls != failure_call

    monkeypatch.setattr(files, "_safe_target", safe_until_failure)
    monkeypatch.setattr(files, "_require_safe_acl", lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._create_target(directory_fd, f"target-{failure_call}")
    finally:
        os.close(directory_fd)


def test_sqlite_sidecar_validation_rejects_inconsistent_cleanup_identity() -> None:
    identity = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    sidecars = tuple(
        files._SQLiteSidecarAuthorization(
            suffix=suffix,
            identity=identity,
            created=index == 0,
            transient=suffix in files._SQLITE_TRANSIENT_SIDECAR_SUFFIXES,
            cleanup_identity=None,
        )
        for index, suffix in enumerate(files._SQLITE_SIDECAR_SUFFIXES)
    )
    with pytest.raises(files._UnsafeFilePathError):
        files._validate_sqlite_sidecars(7, "capture.sqlite3", sidecars, strict_transient=True)


def test_sqlite_authorization_rejects_a_final_target_identity_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    changed = files._StableIdentity(device=2, inode=2, mode=0, owner=0)
    descriptor = os.open(os.sep, os.O_RDONLY)
    monkeypatch.setattr(files, "_open_parent", lambda _path: (descriptor, identity))
    monkeypatch.setattr(
        files,
        "_authorize_named_target",
        lambda *_args: (identity, False, None),
    )
    monkeypatch.setattr(files, "_verify_parent", lambda *_args: None)
    monkeypatch.setattr(files, "_validate_existing_target", lambda *_args: changed)
    monkeypatch.setattr(files, "_validate_sqlite_sidecars", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(files, "_matches_authorized_target", lambda *_args: False)
    with pytest.raises(files._UnsafeFilePathError):
        files._authorize(Path("/private/capture.sqlite3"))


def test_mutable_sidecar_and_sqlite_validation_reject_incomplete_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(files._UnsafeFilePathError):
        files._validate_mutable_sqlite_sidecars(7, "capture.sqlite3", ())

    identity = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    sidecars = tuple(
        files._SQLiteSidecarAuthorization(
            suffix=suffix,
            identity=identity,
            created=False,
            transient=suffix in files._SQLITE_TRANSIENT_SIDECAR_SUFFIXES,
            cleanup_identity=None,
        )
        for suffix in files._SQLITE_SIDECAR_SUFFIXES
    )
    monkeypatch.setattr(
        files,
        "_validate_existing_mutable_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    with pytest.raises(FileNotFoundError):
        files._validate_mutable_sqlite_sidecars(7, "capture.sqlite3", sidecars)

    with pytest.raises(files._UnsafeFilePathError):
        files._revalidate_mutable_sqlite(_dummy_authorization())


def test_mutable_sqlite_revalidation_rejects_parent_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = files._StableIdentity(device=1, inode=1, mode=0, owner=0)
    changed = files._StableIdentity(device=2, inode=2, mode=0, owner=0)
    authorization = files.StableFileAuthorization(
        path="/private/capture.sqlite3",
        _parent_identity=expected,
        _target_identity=expected,
    )
    monkeypatch.setattr(
        files,
        "_open_parent",
        lambda _path: (os.open(os.sep, os.O_RDONLY), changed),
    )
    with pytest.raises(files._UnsafeFilePathError):
        files._revalidate_mutable_sqlite(authorization)


def test_private_delete_platform_normalizes_secure_platform_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        files,
        "_require_secure_platform",
        lambda: (_ for _ in ()).throw(files._UnsafeFilePathError()),
    )
    with pytest.raises(files._UnsupportedFileOperationError):
        files._require_private_delete_platform()


def test_delete_stage_skips_its_forbidden_generated_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files.secrets, "token_hex", lambda _length: "forbidden")
    with pytest.raises(files._UnsafeFilePathError):
        files._create_private_delete_stage(
            7,
            forbidden_name=".saliencegate-delete-forbidden",
        )


def test_delete_stage_rejects_post_creation_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, descriptor = _open_private_directory(tmp_path / "private")
    monkeypatch.setattr(files, "_private_delete_stage_matches", lambda *_args: False)
    try:
        with pytest.raises(files._UnsafeFilePathError):
            files._create_private_delete_stage(descriptor, forbidden_name="entry")
    finally:
        os.close(descriptor)
        for child in Path(directory.path).iterdir():
            if child.is_dir():
                child.rmdir()
