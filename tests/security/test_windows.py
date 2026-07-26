from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import cast

import pytest

import saliencegate.security.windows as windows_module
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsFileIdentity,
    WindowsPathAuthorization,
    WindowsPathKind,
    WindowsPathSecurity,
    WindowsSecurityError,
    WindowsSecurityOperations,
    WindowsStableFileRead,
    authorize_windows_managed_path,
    authorize_windows_private_path,
    ensure_windows_private_directory,
)

_OWNER_SID = "S-1-5-21-1000"
_OTHER_SID = "S-1-5-21-2000"
_ADMINISTRATORS_SID = "S-1-5-32-544"
_PATH = PureWindowsPath(r"C:\Users\synthetic\capture.bin")


class _FakeWindowsOperations:
    def __init__(
        self,
        security: WindowsPathSecurity | None,
        *,
        owner_sid: str = _OWNER_SID,
    ) -> None:
        self.security = security
        self.owner_sid = owner_sid
        self.owner_calls = 0
        self.inspected: list[PureWindowsPath] = []
        self.ancestor_inspected: list[PureWindowsPath] = []
        self.ancestor_security: dict[PureWindowsPath, WindowsPathSecurity] = {}
        self.created: list[tuple[PureWindowsPath, WindowsPathKind]] = []

    def current_user_sid(self) -> str:
        self.owner_calls += 1
        return self.owner_sid

    def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        self.inspected.append(path)
        return self.security

    def inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        self.ancestor_inspected.append(path)
        return tuple(
            (
                ancestor,
                self.ancestor_security.get(
                    ancestor,
                    _security(
                        identity=_identity((index + 1).to_bytes(16, "little")),
                        kind=WindowsPathKind.DIRECTORY,
                        owner_private_dacl=False,
                    ),
                ),
            )
            for index, ancestor in enumerate(reversed(path.parents))
        )

    def create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        self.created.append((path, kind))


def _identity(file_id: bytes = b"a" * 16) -> WindowsFileIdentity:
    return WindowsFileIdentity(volume_serial_number=7, file_id=file_id)


def _security(
    *,
    identity: WindowsFileIdentity | None = None,
    kind: WindowsPathKind = WindowsPathKind.FILE,
    owner_sid: str = _OWNER_SID,
    owner_private_dacl: bool = True,
    owner_write_protected_dacl: bool = True,
    owner_traversal_protected_dacl: bool = True,
    reparse_tag: int | None = None,
) -> WindowsPathSecurity:
    return WindowsPathSecurity(
        identity=_identity() if identity is None else identity,
        kind=kind,
        owner_sid=owner_sid,
        owner_private_dacl=owner_private_dacl,
        owner_write_protected_dacl=owner_write_protected_dacl,
        owner_traversal_protected_dacl=owner_traversal_protected_dacl,
        hardlink_count=1,
        reparse_tag=reparse_tag,
    )


def _authorization(
    *,
    path: PureWindowsPath = _PATH,
) -> tuple[WindowsPathAuthorization, _FakeWindowsOperations]:
    operations = _FakeWindowsOperations(_security())
    authorization = authorize_windows_private_path(
        path,
        kind=WindowsPathKind.FILE,
        operations=operations,
    )
    return authorization, operations


def _assert_content_free(error: WindowsSecurityError, secret: str) -> None:
    assert str(error) == "Windows private path authorization failed"
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_windows_security_models_preserve_invariants_and_redact_representations() -> None:
    path = PureWindowsPath(r"C:\Users\provider-native-secret\capture.bin")
    identity = _identity(b"native-secret-id")
    security = _security(identity=identity)
    operations = _FakeWindowsOperations(security)
    authorization = authorize_windows_private_path(
        path,
        kind=WindowsPathKind.FILE,
        operations=operations,
    )
    snapshot = windows_module._WindowsFileSnapshot(
        security=security,
        size=23,
        last_write_time=101,
        change_time=102,
    )
    stable = WindowsStableFileRead(
        data=b"provider-native-stable-secret",
        authorization=authorization,
    )

    assert isinstance(operations, WindowsSecurityOperations)
    assert identity.volume_serial_number == 7
    assert identity.file_id == b"native-secret-id"
    assert security.identity is identity
    assert snapshot.security is security
    assert stable.authorization is authorization
    assert repr(identity) == "WindowsFileIdentity(<redacted>)"
    assert repr(security) == "WindowsPathSecurity(<redacted>)"
    assert repr(snapshot) == "_WindowsFileSnapshot(<redacted>)"
    assert repr(authorization) == "WindowsPathAuthorization(<redacted>)"
    assert repr(stable) == "WindowsStableFileRead(<redacted>)"
    rendered = "\n".join(
        repr(value) for value in (identity, security, snapshot, authorization, stable)
    )
    assert "provider-native" not in rendered
    assert "native-secret-id" not in rendered


@pytest.mark.parametrize(
    ("volume_serial_number", "file_id"),
    (
        (True, b"a" * 16),
        (-1, b"a" * 16),
        (1 << 64, b"a" * 16),
        (7, bytearray(b"a" * 16)),
        (7, b"a" * 15),
        (7, b"a" * 17),
    ),
)
def test_windows_file_identity_rejects_invalid_types_and_bounds(
    volume_serial_number: object,
    file_id: object,
) -> None:
    with pytest.raises(WindowsSecurityError) as captured:
        WindowsFileIdentity(
            volume_serial_number=cast(int, volume_serial_number),
            file_id=cast(bytes, file_id),
        )

    _assert_content_free(captured.value, "native-secret")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("identity", object()),
        ("kind", "file"),
        ("owner_sid", "not-a-sid"),
        ("owner_private_dacl", 1),
        ("owner_write_protected_dacl", 1),
        ("owner_traversal_protected_dacl", 1),
        ("hardlink_count", True),
        ("hardlink_count", 0),
        ("hardlink_count", 1 << 32),
        ("reparse_tag", True),
        ("reparse_tag", -1),
        ("reparse_tag", 1 << 32),
    ),
)
def test_windows_path_security_rejects_invalid_types_and_bounds(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "identity": _identity(),
        "kind": WindowsPathKind.FILE,
        "owner_sid": _OWNER_SID,
        "owner_private_dacl": True,
        "owner_write_protected_dacl": True,
        "owner_traversal_protected_dacl": True,
        "hardlink_count": 1,
        "reparse_tag": None,
    }
    values[field] = value

    with pytest.raises(WindowsSecurityError) as captured:
        WindowsPathSecurity(
            identity=cast(WindowsFileIdentity, values["identity"]),
            kind=cast(WindowsPathKind, values["kind"]),
            owner_sid=cast(str, values["owner_sid"]),
            owner_private_dacl=cast(bool, values["owner_private_dacl"]),
            owner_write_protected_dacl=cast(
                bool,
                values["owner_write_protected_dacl"],
            ),
            owner_traversal_protected_dacl=cast(
                bool,
                values["owner_traversal_protected_dacl"],
            ),
            hardlink_count=cast(int, values["hardlink_count"]),
            reparse_tag=cast(int | None, values["reparse_tag"]),
        )

    _assert_content_free(captured.value, "native-secret")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("security", object()),
        ("size", True),
        ("size", -1),
        ("last_write_time", True),
        ("last_write_time", -1),
        ("change_time", True),
        ("change_time", -1),
    ),
)
def test_windows_file_snapshot_rejects_invalid_types_and_bounds(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "security": _security(),
        "size": 0,
        "last_write_time": 1,
        "change_time": 2,
    }
    values[field] = value

    with pytest.raises(WindowsSecurityError) as captured:
        windows_module._WindowsFileSnapshot(
            security=cast(WindowsPathSecurity, values["security"]),
            size=cast(int, values["size"]),
            last_write_time=cast(int, values["last_write_time"]),
            change_time=cast(int, values["change_time"]),
        )

    _assert_content_free(captured.value, "native-secret")


@pytest.mark.parametrize("field", ("data", "authorization"))
def test_windows_stable_read_rejects_invalid_model_fields(field: str) -> None:
    authorization, _operations = _authorization()
    data: object = b"valid"
    supplied_authorization: object = authorization
    if field == "data":
        data = bytearray(b"provider-native-secret")
    else:
        supplied_authorization = object()

    with pytest.raises(WindowsSecurityError) as captured:
        WindowsStableFileRead(
            data=cast(bytes, data),
            authorization=cast(WindowsPathAuthorization, supplied_authorization),
        )

    _assert_content_free(captured.value, "provider-native-secret")


@pytest.mark.parametrize(
    "path",
    (
        PureWindowsPath(r"provider-native-secret\capture.bin"),
        PureWindowsPath(r"\\.\PhysicalDrive0"),
        PureWindowsPath(r"\\?\C:\provider-native-secret\capture.bin"),
        PureWindowsPath(r"C:\provider-native-secret\NUL.txt"),
    ),
    ids=("relative", "device", "extended_device", "reserved"),
)
def test_windows_authorizer_rejects_relative_device_and_reserved_paths(
    path: PureWindowsPath,
) -> None:
    operations = _FakeWindowsOperations(_security())

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_private_path(
            path,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    assert operations.owner_calls == 0
    assert operations.inspected == []
    assert operations.created == []
    _assert_content_free(captured.value, "provider-native-secret")


def test_windows_authorizer_rejects_wrong_argument_types_before_operations() -> None:
    operations = _FakeWindowsOperations(_security())
    invocations: tuple[Callable[[], object], ...] = (
        lambda: authorize_windows_private_path(
            cast(PureWindowsPath, r"C:\provider-native-secret\capture.bin"),
            kind=WindowsPathKind.FILE,
            operations=operations,
        ),
        lambda: authorize_windows_private_path(
            _PATH,
            kind=cast(WindowsPathKind, "file"),
            operations=operations,
        ),
        lambda: authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=cast(WindowsSecurityOperations, object()),
        ),
        lambda: authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
            create=cast(bool, 1),
        ),
    )

    for invoke in invocations:
        with pytest.raises(WindowsSecurityError) as captured:
            invoke()
        _assert_content_free(captured.value, "provider-native-secret")

    assert operations.owner_calls == 0
    assert operations.inspected == []
    assert operations.created == []


def test_windows_authorizer_rejects_invalid_current_user_sid_before_inspection() -> None:
    secret = "provider-native-invalid-sid"
    operations = _FakeWindowsOperations(_security(), owner_sid=secret)

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    assert operations.owner_calls == 1
    assert operations.inspected == []
    assert operations.created == []
    _assert_content_free(captured.value, secret)


def test_windows_authorizer_requires_create_for_a_missing_path() -> None:
    operations = _FakeWindowsOperations(None)

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
            create=False,
        )

    assert operations.owner_calls == 1
    assert operations.inspected == [_PATH]
    assert operations.created == []
    _assert_content_free(captured.value, "synthetic")


def test_windows_private_directory_creation_rejects_an_intermediate_reparse_before_write() -> None:
    existing = PureWindowsPath(r"C:\Users\synthetic")
    target = existing / "capture" / "provider"
    managed = _security(
        kind=WindowsPathKind.DIRECTORY,
        owner_private_dacl=False,
        owner_write_protected_dacl=True,
    )

    class MissingSuffixOperations(_FakeWindowsOperations):
        def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
            self.inspected.append(path)
            return managed if path == existing else None

    operations = MissingSuffixOperations(None)
    operations.ancestor_security[PureWindowsPath(r"C:\Users")] = _security(
        kind=WindowsPathKind.DIRECTORY,
        reparse_tag=0xA000000C,
    )

    with pytest.raises(WindowsSecurityError):
        ensure_windows_private_directory(target, operations=operations)

    assert operations.created == []


def test_windows_managed_authorizer_accepts_readable_but_write_protected_dacl() -> None:
    managed = _security(
        owner_private_dacl=False,
        owner_write_protected_dacl=True,
    )
    operations = _FakeWindowsOperations(managed)

    authorization = authorize_windows_managed_path(
        _PATH,
        kind=WindowsPathKind.FILE,
        operations=operations,
    )
    authorization.revalidate()

    with pytest.raises(WindowsSecurityError):
        authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )


def test_windows_managed_authorizer_accepts_protected_administrators_owner_only() -> None:
    operations = _FakeWindowsOperations(
        _security(
            owner_sid=_ADMINISTRATORS_SID,
            owner_private_dacl=True,
            owner_write_protected_dacl=True,
        )
    )

    authorization = authorize_windows_managed_path(
        _PATH,
        kind=WindowsPathKind.FILE,
        operations=operations,
    )
    authorization.revalidate()

    with pytest.raises(WindowsSecurityError):
        authorize_windows_private_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )


def test_windows_managed_authorizer_rejects_writable_administrators_owner() -> None:
    operations = _FakeWindowsOperations(
        _security(
            owner_sid=_ADMINISTRATORS_SID,
            owner_private_dacl=False,
            owner_write_protected_dacl=False,
        )
    )

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_managed_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    _assert_content_free(captured.value, "synthetic")


def test_windows_managed_authorizer_rejects_a_protected_untrusted_owner() -> None:
    operations = _FakeWindowsOperations(
        _security(
            owner_sid=_OTHER_SID,
            owner_private_dacl=False,
            owner_write_protected_dacl=True,
        )
    )

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_managed_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    _assert_content_free(captured.value, "synthetic")


def test_windows_managed_authorizer_rejects_an_untrusted_writable_dacl() -> None:
    operations = _FakeWindowsOperations(
        _security(
            owner_private_dacl=False,
            owner_write_protected_dacl=False,
        )
    )

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_managed_path(
            _PATH,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    _assert_content_free(captured.value, "synthetic")


def test_windows_private_dacl_cannot_claim_an_unprotected_write_policy() -> None:
    with pytest.raises(WindowsSecurityError):
        _security(
            owner_private_dacl=True,
            owner_write_protected_dacl=False,
        )


def test_windows_write_protected_dacl_cannot_claim_unsafe_traversal() -> None:
    with pytest.raises(WindowsSecurityError):
        _security(owner_traversal_protected_dacl=False)


def test_windows_path_authorization_rejects_a_forged_capability() -> None:
    security = _security()
    operations = _FakeWindowsOperations(security)

    with pytest.raises(WindowsSecurityError) as captured:
        WindowsPathAuthorization(
            path=_PATH,
            kind=WindowsPathKind.FILE,
            security=security,
            _owner_sid=_OWNER_SID,
            _operations=operations,
            _token=object(),
        )

    assert operations.owner_calls == 0
    assert operations.inspected == []
    _assert_content_free(captured.value, "synthetic")


@pytest.mark.parametrize("mutation", ("current_user", "path_owner"))
def test_windows_authorization_rejects_owner_sid_changes(mutation: str) -> None:
    authorization, operations = _authorization()
    if mutation == "current_user":
        operations.owner_sid = _OTHER_SID
    else:
        operations.security = _security(owner_sid=_OTHER_SID)

    with pytest.raises(WindowsSecurityError) as captured:
        authorization.revalidate()

    assert authorization.security.owner_sid == _OWNER_SID
    assert operations.created == []
    _assert_content_free(captured.value, "synthetic")


def test_windows_authorization_rejects_intermediate_ancestor_substitution() -> None:
    authorization, operations = _authorization()
    intermediate = PureWindowsPath(r"C:\Users\synthetic")
    original_target_identity = authorization.security.identity
    operations.ancestor_security[intermediate] = _security(
        identity=_identity(b"substituted-path"),
        kind=WindowsPathKind.DIRECTORY,
        owner_private_dacl=False,
    )

    with pytest.raises(WindowsSecurityError) as captured:
        authorization.revalidate()

    assert operations.security is not None
    assert operations.security.identity == original_target_identity
    _assert_content_free(captured.value, "synthetic")


@pytest.mark.skipif(
    os.name != "nt",
    reason="native Win32 atomic publication is the remote R01 gate",
)
def test_native_windows_private_file_publication_read_lock_and_delete(
    tmp_path: Path,
) -> None:
    operations = NativeWindowsSecurityOperations()
    directory = PureWindowsPath(str(tmp_path / "native-private-publication"))
    authorize_windows_private_path(
        directory,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
        create=True,
    ).revalidate()
    target = directory / "owned.bin"

    first = operations.publish_private_file(
        target,
        b"first",
        maximum_bytes=64,
        validate_published=lambda current: current == b"first",
    )
    assert first.data == b"first"
    first.authorization.revalidate()

    with pytest.raises(WindowsSecurityError):
        operations.publish_private_file(
            target,
            b"forged",
            maximum_bytes=64,
            validate_replacement=lambda current: current == b"other",
        )
    assert operations.read_private_file(target, maximum_bytes=64).data == b"first"

    replaced = operations.publish_private_file(
        target,
        b"second",
        maximum_bytes=64,
        validate_replacement=lambda current: current == b"first",
        validate_published=lambda current: current == b"second",
    )
    assert replaced.data == b"second"
    with operations.private_file_lock(directory / "owned.lock"):
        assert operations.read_private_file(target, maximum_bytes=64).data == b"second"

    current = operations.read_private_file(target, maximum_bytes=64)
    operations.delete_authorized_file(current.authorization)
    assert operations.inspect_path(target) is None


@pytest.mark.skipif(
    os.name != "nt",
    reason="native intermediate-junction substitution is the remote R01 gate",
)
def test_native_windows_authorization_rejects_intermediate_junction_substitution(
    tmp_path: Path,
) -> None:
    operations = NativeWindowsSecurityOperations()
    root = PureWindowsPath(str(tmp_path / "native-ancestor-chain"))
    first = root / "first"
    second = first / "second"
    for directory in (root, first, second):
        authorize_windows_private_path(
            directory,
            kind=WindowsPathKind.DIRECTORY,
            operations=operations,
            create=True,
        ).revalidate()
    target = second / "owned.bin"
    stable = operations.publish_private_file(
        target,
        b"stable",
        maximum_bytes=64,
        validate_published=lambda current: current == b"stable",
    )
    moved = PureWindowsPath(str(tmp_path / "moved-native-ancestor"))
    os.rename(first, moved)
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(first), str(moved)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        current = operations.inspect_path(target)
        assert current is not None
        assert current.identity == stable.authorization.security.identity
        with pytest.raises(WindowsSecurityError):
            stable.authorization.revalidate()
    finally:
        os.rmdir(first)
        os.rename(moved, first)
