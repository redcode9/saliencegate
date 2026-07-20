from __future__ import annotations

import hashlib
import inspect
import os
import stat
import subprocess
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

import saliencegate.integrations.bootstrap as bootstrap_module
import saliencegate.integrations.config_files as config_files_module
import saliencegate.integrations.installation as installation_module
import saliencegate.security.files as files_module
from saliencegate.capture import CaptureProfile, capture_capability_digest, capture_profile
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
from saliencegate.integrations.config_files import ConfigSyntax, OwnedConfigSpec
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationState,
    derive_installation_identity,
    git_tracked_project_files,
    inspect_provider_installation,
    install_provider,
    recover_provider_installation,
    uninstall_provider,
)
from saliencegate.integrations.registry import (
    BUILTIN_PROVIDER_REGISTRY,
    ProviderAlias,
    ProviderInstallationSpec,
    ProviderRegistryError,
)
from saliencegate.security import InstallationKey, StableFileRead, StableReadPolicy
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    authorize_windows_private_path,
)

KEY = InstallationKey(b"k" * 32)
PROFILE = CaptureProfile.CODEX_HOOKS_V1
CAPABILITY_DIGEST = capture_capability_digest(capture_profile(PROFILE))
MARKER = "saliencegate-owned:synthetic-v1"


def _bundle(relative_bootstrap: str = "./saliencegate.bootstrap.json") -> bytes:
    return (
        f'export const saliencegateBootstrap = new URL("{relative_bootstrap}", import.meta.url);\n'
    ).encode()


def _config_spec() -> OwnedConfigSpec:
    return OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=(
            b'"saliencegate":{'
            b'"marker":"saliencegate-owned:synthetic-v1",'
            b'"command":"saliencegate-capture-hook"}'
        ),
    )


def _make_private_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        windows_path = PureWindowsPath(os.fspath(path))
        authorization = authorize_windows_private_path(
            windows_path,
            kind=WindowsPathKind.DIRECTORY,
            operations=operations,
            create=True,
        )
        authorization.revalidate()
        return
    path.mkdir(mode=0o700, exist_ok=True)
    path.chmod(0o700)


def _write_new_private_file(path: Path, data: bytes) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        published = operations.publish_private_file(
            PureWindowsPath(os.fspath(path)),
            data,
            maximum_bytes=max(1, len(data)),
            validate_published=lambda current: current == data,
        )
        assert published.data == data
        return
    path.write_bytes(data)


def _make_spec(
    tmp_path: Path,
    *,
    generation: int = 1,
    bundle_name: str = "saliencegate-v1.js",
    bundle_bytes: bytes | None = None,
) -> ProviderInstallationSpec:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir(exist_ok=True)
    _make_private_directory(state)
    launcher = state / "synthetic-capture-hook"
    integration = project / ".synthetic"
    return ProviderInstallationSpec(
        provider_id="synthetic",
        profile=PROFILE,
        host_version="0.144.6",
        project_root=project,
        config_path=integration / "settings.json",
        bundle_path=integration / bundle_name,
        bootstrap_path=integration / "saliencegate.bootstrap.json",
        receipt_path=state / "synthetic.receipt.json",
        journal_path=state / "synthetic.journal.json",
        lock_path=state / "synthetic.lock",
        launcher_path=launcher,
        capability_digest=CAPABILITY_DIGEST,
        bundle_bytes=_bundle() if bundle_bytes is None else bundle_bytes,
        launcher_bytes=b"#!/bin/sh\nexit 0\n",
        bootstrap_relative_reference="./saliencegate.bootstrap.json",
        config=_config_spec(),
        generation=generation,
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_builtin_registry_is_closed_and_real_providers_remain_unavailable(
    tmp_path: Path,
) -> None:
    assert tuple(item.alias for item in BUILTIN_PROVIDER_REGISTRY.providers) == tuple(ProviderAlias)
    for alias in ProviderAlias:
        registration = BUILTIN_PROVIDER_REGISTRY.resolve(alias, require_available=False)
        assert registration.alias is alias
        assert registration.available is False
        with pytest.raises(ProviderRegistryError):
            BUILTIN_PROVIDER_REGISTRY.resolve(alias)

    with pytest.raises(ProviderRegistryError):
        BUILTIN_PROVIDER_REGISTRY.resolve("synthetic")
    payload = _make_spec(tmp_path).model_dump(mode="python")
    payload["project_root"] = str(payload["project_root"])
    with pytest.raises(ValidationError):
        ProviderInstallationSpec.model_validate(payload)

    separated = _make_spec(tmp_path).model_dump(mode="python")
    separated["bundle_path"] = separated["project_root"] / ".other" / "saliencegate.js"
    with pytest.raises(ValidationError):
        ProviderInstallationSpec.model_validate(separated)

    decoy = _make_spec(tmp_path).model_dump(mode="python")
    decoy["bundle_bytes"] = (
        b'const decoy = "./saliencegate.bootstrap.json";\n'
        b"const here = import.meta.url;\n"
        b'export default new URL("file:///private/tmp/wrong-bootstrap.json", here);\n'
    )
    with pytest.raises(ValidationError):
        ProviderInstallationSpec.model_validate(decoy)

    noncanonical_binding = _make_spec(tmp_path).model_dump(mode="python")
    noncanonical_binding["bundle_bytes"] = (
        b'const decoy = new URL("./saliencegate.bootstrap.json", import.meta.url);\n'
        b'const bootstrap = "file:///private/tmp/wrong-bootstrap.json";\n'
        b"export default bootstrap;\n"
    )
    with pytest.raises(ValidationError):
        ProviderInstallationSpec.model_validate(noncanonical_binding)


def test_pristine_missing_provider_directory_is_disabled_without_drift(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.DISABLED
    assert status.installed is False
    assert status.drift == ()
    assert not spec.config_path.parent.exists()


def test_malformed_journal_fails_before_creating_installation_paths(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _write_new_private_file(spec.journal_path, b"{}")

    with pytest.raises(InstallationError):
        install_provider(spec, KEY)

    assert not spec.config_path.parent.exists()
    assert not spec.lock_path.exists()
    assert tuple(path.name for path in spec.receipt_path.parent.iterdir()) == (
        spec.journal_path.name,
    )


def test_windows_installation_routes_every_owned_file_boundary_through_win32() -> None:
    installation_source = inspect.getsource(installation_module)
    bootstrap_source = inspect.getsource(bootstrap_module)
    config_source = inspect.getsource(config_files_module)

    assert "NativeWindowsSecurityOperations" in installation_source
    assert "operations.publish_private_file(" in installation_source
    assert "operations.read_private_file(" in installation_source
    assert "operations.delete_authorized_file(" in installation_source
    assert "operations.private_file_lock(" in installation_source
    assert "ensure_windows_private_directory(" in installation_source
    assert "ensure_private_installation_directory(" in installation_source
    assert "NativeWindowsSecurityOperations" in bootstrap_source
    assert "operations.publish_private_file_in_managed_directory(" in bootstrap_source
    assert "operations.read_private_file(" in bootstrap_source
    assert "NativeWindowsSecurityOperations" in config_source
    assert "operations.publish_managed_file(" in config_source
    assert "operations.read_managed_file(" in config_source
    assert "operations.delete_authorized_file(" in config_source


def test_dry_run_is_side_effect_free_and_reports_project_paths(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    before = {path: path.read_bytes() for path in spec.project_root.rglob("*") if path.is_file()}

    result = install_provider(spec, KEY, dry_run=True)

    assert result.disposition is InstallationDisposition.PLANNED
    assert result.state is InstallationState.PENDING
    assert result.would_write == (
        spec.launcher_path,
        spec.config_path,
        spec.bundle_path,
        spec.bootstrap_path,
        spec.receipt_path,
        spec.journal_path,
    )
    assert not spec.config_path.parent.exists()
    assert not spec.receipt_path.exists()
    assert not spec.journal_path.exists()
    assert not spec.lock_path.exists()
    assert before == {
        path: path.read_bytes() for path in spec.project_root.rglob("*") if path.is_file()
    }


def test_install_absent_is_private_authenticated_and_idempotent(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    identity = derive_installation_identity(spec, KEY)

    installed = install_provider(spec, KEY)

    assert installed.disposition is InstallationDisposition.INSTALLED
    assert installed.state is InstallationState.ENABLED
    assert installed.project_digest == CaptureDigestContext(KEY).workspace_identity(
        os.fsencode(spec.project_root)
    )
    assert installed.project_digest == identity.project_digest
    assert installed.connection_id == identity.connection_id
    assert spec.config_path.read_bytes() == b"{" + spec.config.owned_fragment + b"}"
    if os.name == "posix":
        assert _mode(spec.bundle_path) == 0o600
        assert _mode(spec.bootstrap_path) == 0o600
        assert _mode(spec.receipt_path) == 0o600
        assert _mode(spec.launcher_path) == 0o700
    else:  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        for path in (
            spec.bundle_path,
            spec.bootstrap_path,
            spec.receipt_path,
            spec.launcher_path,
        ):
            security = operations.inspect_path(PureWindowsPath(os.fspath(path)))
            assert security is not None
            assert security.owner_private_dacl is True
            assert security.hardlink_count == 1
            assert security.reparse_tag is None
    assert spec.launcher_path.read_bytes() == spec.launcher_bytes
    assert not spec.journal_path.exists()
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    assert bootstrap.profile is PROFILE
    assert bootstrap.launcher_path == spec.launcher_path
    assert bootstrap.capability_digest == CAPABILITY_DIGEST
    assert bootstrap.bundle_digest == hashlib.sha256(spec.bundle_bytes).hexdigest()

    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (
            spec.config_path,
            spec.bundle_path,
            spec.bootstrap_path,
            spec.receipt_path,
            spec.launcher_path,
        )
    }
    repeated = install_provider(spec, KEY)
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (
            spec.config_path,
            spec.bundle_path,
            spec.bootstrap_path,
            spec.receipt_path,
            spec.launcher_path,
        )
    }
    assert repeated.disposition is InstallationDisposition.NOOP
    assert after == before
    assert inspect_provider_installation(spec, KEY).drift == ()


@pytest.mark.skipif(os.name != "posix", reason="private mode bits require POSIX")
@pytest.mark.parametrize(
    ("attribute", "expected_drift"),
    (
        ("receipt_path", "receipt"),
        ("lock_path", "lock"),
        ("bundle_path", "bundle"),
        ("bootstrap_path", "bootstrap"),
    ),
)
def test_owned_private_asset_modes_are_exact_and_never_repaired(
    tmp_path: Path,
    attribute: str,
    expected_drift: str,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    changed = getattr(spec, attribute)
    changed.chmod(0o644)
    before = {
        path: path.read_bytes()
        for path in (
            spec.config_path,
            spec.bundle_path,
            spec.bootstrap_path,
            spec.receipt_path,
            spec.lock_path,
            spec.launcher_path,
        )
    }

    status = inspect_provider_installation(spec, KEY)

    assert status.installed is False
    assert status.drift == (expected_drift,)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert {path: path.read_bytes() for path in before} == before
    assert _mode(changed) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="private mode bits require POSIX")
@pytest.mark.parametrize("mode", (0o755, 0o1700))
def test_operational_directory_mode_is_exact_and_fails_before_writes(
    tmp_path: Path,
    mode: int,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    before = {
        path: path.read_bytes()
        for path in (
            spec.config_path,
            spec.bundle_path,
            spec.bootstrap_path,
            spec.receipt_path,
            spec.lock_path,
            spec.launcher_path,
        )
    }
    spec.receipt_path.parent.chmod(mode)

    status = inspect_provider_installation(spec, KEY)

    assert status.installed is False
    assert status.drift == ("receipt",)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert {path: path.read_bytes() for path in before} == before
    assert _mode(spec.receipt_path.parent) == mode


@pytest.mark.skipif(os.name != "posix", reason="private ACL inspection requires POSIX")
def test_unsafe_operational_acl_fails_read_only_preflight_before_project_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)

    def reject_acl(_descriptor: int, *, deny_only_allowed: bool = False) -> None:
        del deny_only_allowed
        raise OSError("simulated unsafe ACL")

    monkeypatch.setattr(files_module, "_require_safe_acl", reject_acl)

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.DISABLED
    assert status.drift == ("receipt",)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY, dry_run=True)
    assert not spec.config_path.parent.exists()
    assert tuple(spec.receipt_path.parent.iterdir()) == ()


@pytest.mark.skipif(os.name != "posix", reason="private boundary inspection requires POSIX")
def test_missing_operational_suffix_below_unsafe_prefix_fails_before_project_writes(
    tmp_path: Path,
) -> None:
    base = _make_spec(tmp_path)
    unsafe = tmp_path / "unsafe-operational-prefix"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    operational = unsafe / "missing-state"
    payload = base.model_dump(mode="python", warnings="error")
    payload.update(
        receipt_path=operational / "synthetic.receipt.json",
        journal_path=operational / "synthetic.journal.json",
        lock_path=operational / "synthetic.lock",
        launcher_path=operational / "synthetic-capture-hook",
    )
    spec = ProviderInstallationSpec.model_validate(payload)

    assert inspect_provider_installation(spec, KEY).drift == ("receipt",)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY, dry_run=True)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert not spec.config_path.parent.exists()
    assert not operational.exists()


@pytest.mark.skipif(os.name != "posix", reason="private mode bits require POSIX")
def test_unsafe_orphan_lock_is_reported_before_project_directory_creation(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    _write_new_private_file(spec.lock_path, b"")
    spec.lock_path.chmod(0o644)

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.DISABLED
    assert status.drift == ("lock",)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert not spec.config_path.parent.exists()
    assert _mode(spec.lock_path) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound locks require POSIX")
def test_operational_parent_swap_before_lock_never_reaches_decoy_or_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    moved = tmp_path / "moved-operational-state"
    decoy = tmp_path / "decoy-operational-state"
    _make_private_directory(decoy)
    real_ensure = installation_module.ensure_private_installation_directory
    swapped = False

    def swap_after_ensure(path: Path) -> None:
        nonlocal swapped
        real_ensure(path)
        path.rename(moved)
        path.symlink_to(decoy, target_is_directory=True)
        swapped = True

    monkeypatch.setattr(
        installation_module,
        "ensure_private_installation_directory",
        swap_after_ensure,
    )

    with pytest.raises(InstallationError):
        install_provider(spec, KEY)

    assert swapped is True
    assert not (decoy / spec.lock_path.name).exists()
    assert not spec.config_path.parent.exists()
    spec.receipt_path.parent.unlink()
    moved.rename(spec.receipt_path.parent)


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound locks require POSIX")
def test_lock_name_replacement_before_flock_fails_before_project_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    spec = _make_spec(tmp_path)
    displaced = spec.lock_path.with_name("displaced.lock")
    real_flock = fcntl.flock
    replaced = False

    def replace_name_before_lock(descriptor: int, operation: int) -> None:
        nonlocal replaced
        if operation & fcntl.LOCK_EX and not replaced:
            spec.lock_path.rename(displaced)
            _write_new_private_file(spec.lock_path, b"")
            replaced = True
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", replace_name_before_lock)

    with pytest.raises(InstallationError):
        install_provider(spec, KEY)

    assert replaced is True
    assert spec.lock_path.stat().st_ino != displaced.stat().st_ino
    assert not spec.config_path.parent.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="native Win32 installer publication is the remote R01 gate",
)
def test_native_windows_pristine_missing_operational_parent_is_clean_and_installable(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    spec.receipt_path.parent.rmdir()

    pristine = inspect_provider_installation(spec, KEY)
    planned = install_provider(spec, KEY, dry_run=True)

    assert pristine.state is InstallationState.DISABLED
    assert pristine.drift == ()
    assert planned.disposition is InstallationDisposition.PLANNED
    assert not spec.receipt_path.parent.exists()
    assert install_provider(spec, KEY).installed is True
    assert inspect_provider_installation(spec, KEY).drift == ()


@pytest.mark.skipif(
    os.name != "nt",
    reason="native intermediate-junction preflight is the remote R01 gate",
)
def test_native_windows_dry_run_rejects_missing_operational_suffix_below_junction(
    tmp_path: Path,
) -> None:
    base = _make_spec(tmp_path)
    outside = tmp_path / "junction-target"
    outside.mkdir()
    junction = tmp_path / "operational-junction"
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
        text=True,
    )
    operational = junction / "missing-state"
    payload = base.model_dump(mode="python", warnings="error")
    payload.update(
        receipt_path=operational / "synthetic.receipt.json",
        journal_path=operational / "synthetic.journal.json",
        lock_path=operational / "synthetic.lock",
        launcher_path=operational / "synthetic-capture-hook",
    )
    spec = ProviderInstallationSpec.model_validate(payload)
    try:
        with pytest.raises(InstallationError):
            install_provider(spec, KEY, dry_run=True)
        assert not spec.config_path.parent.exists()
        assert not operational.exists()
    finally:
        os.rmdir(junction)


@pytest.mark.skipif(
    os.name != "nt",
    reason="native Win32 installer publication is the remote R01 gate",
)
def test_native_windows_install_inspect_and_uninstall_round_trip(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    operations = NativeWindowsSecurityOperations()

    installed = install_provider(spec, KEY)

    assert installed.disposition is InstallationDisposition.INSTALLED
    assert inspect_provider_installation(spec, KEY).installed is True
    for path in (
        spec.config_path,
        spec.bundle_path,
        spec.bootstrap_path,
        spec.receipt_path,
        spec.lock_path,
        spec.launcher_path,
    ):
        security = operations.inspect_path(PureWindowsPath(os.fspath(path)))
        assert security is not None
        assert security.owner_private_dacl is True
        assert security.hardlink_count == 1
        assert security.reparse_tag is None

    removed = uninstall_provider(spec, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert inspect_provider_installation(spec, KEY).state is InstallationState.DISABLED
    assert not spec.config_path.exists()
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.exists()
    assert spec.lock_path.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="native inherited-DACL coexistence is the remote R01 gate",
)
def test_native_windows_existing_provider_directory_acl_is_not_rewritten(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    spec.config_path.parent.mkdir()
    original = b'{"foreign":"provider-owned"}'
    spec.config_path.write_bytes(original)
    operations = NativeWindowsSecurityOperations()
    windows_parent = PureWindowsPath(os.fspath(spec.config_path.parent))
    parent_before = operations.inspect_path(windows_parent)
    assert parent_before is not None
    assert parent_before.owner_write_protected_dacl is True

    install_provider(spec, KEY)
    parent_installed = operations.inspect_path(windows_parent)
    uninstall_provider(spec, KEY)
    parent_removed = operations.inspect_path(windows_parent)

    assert parent_installed == parent_before
    assert parent_removed == parent_before
    assert spec.config_path.read_bytes() == original


def test_foreign_json_bytes_are_preserved_and_untouched_uninstall_is_exact(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    original = b'{\r\n  "foreign": {"token": "foreign-secret"}\r\n}\r\n'
    _write_new_private_file(spec.config_path, original)

    install_provider(spec, KEY)
    installed = spec.config_path.read_bytes()
    assert b'\r\n  "foreign": {"token": "foreign-secret"}\r\n' in installed
    assert installed != original
    assert b"foreign-secret" not in spec.receipt_path.read_bytes()

    removed = uninstall_provider(spec, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert removed.state is InstallationState.DISABLED
    assert spec.config_path.read_bytes() == original
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.exists()
    assert not spec.journal_path.exists()


def test_existing_empty_opaque_config_is_restored_exactly(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=MARKER,
        owned_fragment=b"# saliencegate-owned:synthetic-v1\ncommand=saliencegate-capture-hook\n",
    )
    opaque = ProviderInstallationSpec.model_validate(payload)
    _make_private_directory(opaque.config_path.parent)
    _write_new_private_file(opaque.config_path, b"")

    install_provider(opaque, KEY)
    assert opaque.config_path.read_bytes() == opaque.config.owned_fragment

    removed = uninstall_provider(opaque, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert opaque.config_path.is_file()
    assert opaque.config_path.read_bytes() == b""


def test_malformed_foreign_config_fails_before_any_installer_write(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    malformed = b'{"foreign-secret":'
    _write_new_private_file(spec.config_path, malformed)

    with pytest.raises(InstallationError):
        install_provider(spec, KEY)

    assert spec.config_path.read_bytes() == malformed
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.receipt_path.exists()
    assert not spec.journal_path.exists()
    assert not spec.lock_path.exists()


def test_drifted_config_removes_only_the_exact_owned_span(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    original = b'{"foreign":"before-secret"}'
    _write_new_private_file(spec.config_path, original)
    install_provider(spec, KEY)
    current = spec.config_path.read_bytes()
    drifted = current[:-1] + b',"after":"after-secret"}'
    spec.config_path.write_bytes(drifted)

    removed = uninstall_provider(spec, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert spec.config_path.read_bytes() == b'{"foreign":"before-secret","after":"after-secret"}'


def test_drifted_config_from_empty_object_removes_the_new_separator(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    current = spec.config_path.read_bytes()
    drifted = current[:-1] + b',"foreign":"after-secret"}'
    spec.config_path.write_bytes(drifted)

    removed = uninstall_provider(spec, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert spec.config_path.read_bytes() == b'{"foreign":"after-secret"}'


def test_uninstall_accepts_the_authenticated_config_preimage_without_rewriting_it(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    original = b'{"foreign":"receipt-authenticated-preimage"}'
    _write_new_private_file(spec.config_path, original)
    install_provider(spec, KEY)
    spec.config_path.write_bytes(original)

    assert inspect_provider_installation(spec, KEY).drift == ("config",)

    removed = uninstall_provider(spec, KEY)

    assert removed.disposition is InstallationDisposition.UNINSTALLED
    assert spec.config_path.read_bytes() == original
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert inspect_provider_installation(spec, KEY).drift == ()


def test_uninstall_revalidates_a_no_write_config_plan_before_asset_deletion(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    original = b'{"foreign":"receipt-authenticated-preimage"}'
    _write_new_private_file(spec.config_path, original)
    install_provider(spec, KEY)
    installed = spec.config_path.read_bytes()
    spec.config_path.write_bytes(original)

    def restore_marker(stage: str) -> None:
        if stage == "after_draining_receipt_publish":
            spec.config_path.write_bytes(installed)

    with pytest.raises(InstallationError):
        uninstall_provider(spec, KEY, _fault_injector=restore_marker)

    assert spec.bundle_path.exists()
    assert spec.bootstrap_path.exists()
    assert spec.launcher_path.exists()
    assert spec.config.marker.encode("ascii") in spec.config_path.read_bytes()
    assert recover_provider_installation(spec, KEY).state is InstallationState.DISABLED


def test_ambiguous_or_mutated_owned_span_fails_closed_without_asset_deletion(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    config = spec.config_path.read_bytes()
    spec.config_path.write_bytes(config.replace(b"capture-hook", b"capture-HOOK"))

    with pytest.raises(InstallationError):
        uninstall_provider(spec, KEY)

    assert spec.bundle_path.exists()
    assert spec.bootstrap_path.exists()
    assert inspect_provider_installation(spec, KEY).drift == ("config",)


def test_bootstrap_or_bundle_drift_is_reported_and_never_repaired(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    spec.bootstrap_path.write_bytes(spec.bootstrap_path.read_bytes() + b"\n")
    before = spec.bootstrap_path.read_bytes()

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.ENABLED
    assert status.drift == ("bootstrap",)
    assert spec.bootstrap_path.read_bytes() == before
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert spec.bootstrap_path.read_bytes() == before


def test_inspection_reports_same_generation_config_contract_drift(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    payload = spec.model_dump(mode="python")
    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=(
            b'"saliencegate":{'
            b'"marker":"saliencegate-owned:synthetic-v1",'
            b'"command":"changed-capture-hook"}'
        ),
    )
    changed = ProviderInstallationSpec.model_validate(payload)

    status = inspect_provider_installation(changed, KEY)

    assert status.state is InstallationState.ENABLED
    assert status.drift == ("config",)
    with pytest.raises(InstallationError):
        install_provider(changed, KEY)


def test_same_generation_owned_fragment_must_match_exactly(tmp_path: Path) -> None:
    payload = _make_spec(tmp_path).model_dump(mode="python", warnings="error")
    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=MARKER,
        owned_fragment=b"command=old\n# saliencegate-owned:synthetic-v1",
    )
    installed = ProviderInstallationSpec.model_validate(payload)
    install_provider(installed, KEY)
    installed_bytes = installed.config_path.read_bytes()

    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=MARKER,
        owned_fragment=b"# saliencegate-owned:synthetic-v1",
    )
    changed = ProviderInstallationSpec.model_validate(payload)

    with pytest.raises(InstallationError):
        install_provider(changed, KEY)

    assert changed.config_path.read_bytes() == installed_bytes
    assert inspect_provider_installation(changed, KEY).drift == ("config",)


def test_pending_install_recovers_forward_from_content_free_journal(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    _write_new_private_file(spec.config_path, b'{"foreign":"journal-secret"}')

    def fail(stage: str) -> None:
        if stage == "after_config_publish":
            raise RuntimeError("crash-secret")

    with pytest.raises(RuntimeError, match="crash-secret"):
        install_provider(spec, KEY, _fault_injector=fail)

    assert spec.journal_path.exists()
    journal = spec.journal_path.read_bytes()
    assert b"journal-secret" not in journal
    assert b"crash-secret" not in journal

    recovered = recover_provider_installation(spec, KEY)

    assert recovered.disposition is InstallationDisposition.RECOVERED
    assert recovered.state is InstallationState.ENABLED
    assert not spec.journal_path.exists()
    assert inspect_provider_installation(spec, KEY).drift == ()


def test_inspection_reports_enabled_receipt_with_unfinished_install_journal(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)

    def fail(stage: str) -> None:
        if stage == "after_enabled_receipt_publish":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        install_provider(spec, KEY, _fault_injector=fail)

    receipt_before = spec.receipt_path.read_bytes()
    journal_before = spec.journal_path.read_bytes()

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.PENDING
    assert status.installed is False
    assert status.drift == ("receipt",)
    assert spec.receipt_path.read_bytes() == receipt_before
    assert spec.journal_path.read_bytes() == journal_before

    recovered = recover_provider_installation(spec, KEY)
    assert recovered.state is InstallationState.ENABLED
    assert inspect_provider_installation(spec, KEY).drift == ()


def test_inspection_reports_disabled_receipt_with_unfinished_uninstall_journal(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)

    def fail(stage: str) -> None:
        if stage == "after_asset_remove":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        uninstall_provider(spec, KEY, _fault_injector=fail)

    journal = spec.journal_path.read_bytes()
    assert recover_provider_installation(spec, KEY).state is InstallationState.DISABLED
    _write_new_private_file(spec.journal_path, journal)
    if os.name == "posix":
        spec.journal_path.chmod(0o600)
    receipt_before = spec.receipt_path.read_bytes()

    status = inspect_provider_installation(spec, KEY)

    assert status.state is InstallationState.DRAINING
    assert status.installed is False
    assert status.drift == ("receipt",)
    assert spec.receipt_path.read_bytes() == receipt_before
    assert spec.journal_path.read_bytes() == journal

    recovered = recover_provider_installation(spec, KEY)
    assert recovered.state is InstallationState.DISABLED
    assert inspect_provider_installation(spec, KEY).drift == ()


def test_inspection_fails_closed_for_corrupt_journal_with_valid_receipt(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)

    def fail(stage: str) -> None:
        if stage == "after_enabled_receipt_publish":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        install_provider(spec, KEY, _fault_injector=fail)

    receipt_before = spec.receipt_path.read_bytes()
    journal = spec.journal_path.read_bytes()
    marker = b'"journal_mac":"'
    mac_offset = journal.index(marker) + len(marker)
    replacement = b"0" if journal[mac_offset : mac_offset + 1] != b"0" else b"1"
    corrupted = journal[:mac_offset] + replacement + journal[mac_offset + 1 :]
    spec.journal_path.write_bytes(corrupted)

    with pytest.raises(InstallationError):
        inspect_provider_installation(spec, KEY)

    assert spec.receipt_path.read_bytes() == receipt_before
    assert spec.journal_path.read_bytes() == corrupted


def test_journal_only_install_uninstall_and_reinstall_recover_forward(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)

    def fail(stage: str) -> None:
        if stage == "after_journal_publish":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        install_provider(spec, KEY, _fault_injector=fail)
    assert recover_provider_installation(spec, KEY).state is InstallationState.ENABLED

    with pytest.raises(RuntimeError, match="simulated interruption"):
        uninstall_provider(spec, KEY, _fault_injector=fail)
    assert recover_provider_installation(spec, KEY).state is InstallationState.DISABLED
    assert not spec.launcher_path.exists()

    with pytest.raises(RuntimeError, match="simulated interruption"):
        install_provider(spec, KEY, _fault_injector=fail)
    assert recover_provider_installation(spec, KEY).state is InstallationState.ENABLED
    assert inspect_provider_installation(spec, KEY).drift == ()


@pytest.mark.parametrize(
    "stage",
    (
        "after_pending_receipt_publish",
        "after_launcher_publish",
        "after_bundle_publish",
        "after_bootstrap_publish",
        "after_config_publish",
        "after_enabled_receipt_publish",
    ),
)
def test_each_install_publication_phase_recovers_forward(tmp_path: Path, stage: str) -> None:
    spec = _make_spec(tmp_path)

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        install_provider(spec, KEY, _fault_injector=fail)

    assert recover_provider_installation(spec, KEY).state is InstallationState.ENABLED
    assert inspect_provider_installation(spec, KEY).drift == ()


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound path swap is POSIX-specific")
def test_project_directory_creation_stays_on_pinned_root_during_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    moved = tmp_path / "moved-project"
    decoy = tmp_path / "decoy-project"
    real_mkdir = os.mkdir
    swapped = False

    def swap_after_mkdir(
        path: str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        real_mkdir(path, mode, dir_fd=dir_fd)
        if path == ".synthetic" and not swapped:
            swapped = True
            spec.project_root.rename(moved)
            real_mkdir(decoy, 0o700)
            spec.project_root.symlink_to(decoy, target_is_directory=True)

    monkeypatch.setattr(os, "mkdir", swap_after_mkdir)

    with pytest.raises(InstallationError):
        install_provider(spec, KEY)

    assert swapped is True
    assert tuple(decoy.iterdir()) == ()
    assert (moved / ".synthetic").is_dir()
    assert not spec.receipt_path.exists()
    spec.project_root.unlink()
    decoy.rmdir()
    moved.rename(spec.project_root)


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound path swap is POSIX-specific")
def test_project_root_symlink_swap_cannot_receive_config_or_enable_install(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    moved = tmp_path / "moved-project"

    def swap_root(stage: str) -> None:
        if stage == "after_bootstrap_publish":
            spec.project_root.rename(moved)
            spec.project_root.symlink_to(moved, target_is_directory=True)

    with pytest.raises(InstallationError):
        install_provider(spec, KEY, _fault_injector=swap_root)

    assert not (moved / spec.config_path.relative_to(spec.project_root)).exists()
    assert spec.journal_path.exists()
    assert spec.receipt_path.exists()
    spec.project_root.unlink()
    moved.rename(spec.project_root)
    assert recover_provider_installation(spec, KEY).state is InstallationState.ENABLED


@pytest.mark.parametrize(
    "stage",
    (
        "after_draining_receipt_publish",
        "after_config_remove",
        "after_asset_remove",
    ),
)
def test_each_uninstall_publication_phase_recovers_forward(tmp_path: Path, stage: str) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        uninstall_provider(spec, KEY, _fault_injector=fail)

    assert recover_provider_installation(spec, KEY).state is InstallationState.DISABLED
    assert inspect_provider_installation(spec, KEY).drift == ()


def test_launcher_drift_blocks_uninstall_before_any_owned_write(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    config_before = spec.config_path.read_bytes()
    bundle_before = spec.bundle_path.read_bytes()
    bootstrap_before = spec.bootstrap_path.read_bytes()
    spec.launcher_path.write_bytes(b"#!/bin/sh\nexit 7\n")

    with pytest.raises(InstallationError):
        uninstall_provider(spec, KEY)

    assert spec.config_path.read_bytes() == config_before
    assert spec.bundle_path.read_bytes() == bundle_before
    assert spec.bootstrap_path.read_bytes() == bootstrap_before
    assert inspect_provider_installation(spec, KEY).drift == ("launcher",)


@pytest.mark.skipif(os.name != "posix", reason="launcher modes require POSIX")
def test_launcher_mode_change_at_stable_read_boundary_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    real_read = installation_module.read_stable_file

    def change_mode_before_read(
        path: Path,
        *,
        maximum_bytes: int,
        policy: StableReadPolicy,
    ) -> StableFileRead:
        if path == spec.launcher_path:
            path.chmod(0o644)
        return real_read(path, maximum_bytes=maximum_bytes, policy=policy)

    monkeypatch.setattr(installation_module, "read_stable_file", change_mode_before_read)

    status = inspect_provider_installation(spec, KEY)

    assert status.installed is False
    assert status.drift == ("launcher",)
    assert _mode(spec.launcher_path) == 0o644


def test_same_generation_contract_or_operational_lock_changes_fail_closed(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    original_launcher = spec.launcher_path.read_bytes()

    payload = spec.model_dump(mode="python")
    payload["launcher_bytes"] = b"#!/bin/sh\nexit 9\n"
    changed_launcher = ProviderInstallationSpec.model_validate(payload)
    with pytest.raises(InstallationError):
        install_provider(changed_launcher, KEY)
    assert spec.launcher_path.read_bytes() == original_launcher

    alternate_lock = spec.lock_path.with_name("alternate.lock")
    payload = spec.model_dump(mode="python")
    payload["lock_path"] = alternate_lock
    changed_lock = ProviderInstallationSpec.model_validate(payload)
    with pytest.raises(InstallationError):
        install_provider(changed_lock, KEY)
    assert not alternate_lock.exists()


def test_same_generation_config_syntax_change_is_reported_as_drift(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    install_provider(spec, KEY)
    payload = spec.model_dump(mode="python", warnings="error")
    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=spec.config.marker,
        owned_fragment=spec.config.owned_fragment,
    )
    changed = ProviderInstallationSpec.model_validate(payload)

    status = inspect_provider_installation(changed, KEY)

    assert status.installed is False
    assert status.drift == ("config",)


@pytest.mark.skipif(os.name != "posix", reason="symlink traversal is platform-specific")
def test_project_integration_paths_reject_intermediate_symlink_escape(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    spec.config_path.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(InstallationError):
        install_provider(spec, KEY, dry_run=True)

    assert tuple(outside.iterdir()) == ()


def test_controlled_upgrade_uses_a_new_immutable_bundle_and_keeps_config_bytes(
    tmp_path: Path,
) -> None:
    first = _make_spec(tmp_path)
    install_provider(first, KEY)
    config_before = first.config_path.read_bytes()
    second = _make_spec(
        tmp_path,
        generation=2,
        bundle_name="saliencegate-v2.js",
        bundle_bytes=_bundle() + b"// v2\n",
    )

    upgraded = install_provider(second, KEY)

    assert upgraded.disposition is InstallationDisposition.UPGRADED
    assert upgraded.state is InstallationState.ENABLED
    assert second.config_path.read_bytes() == config_before
    assert second.bundle_path.exists()
    assert not first.bundle_path.exists()
    assert inspect_provider_installation(second, KEY).drift == ()


def test_upgrade_rejects_a_changed_config_syntax_without_writing(tmp_path: Path) -> None:
    first = _make_spec(tmp_path)
    install_provider(first, KEY)
    before = {
        path: path.read_bytes()
        for path in (first.config_path, first.bundle_path, first.receipt_path)
    }
    payload = _make_spec(
        tmp_path,
        generation=2,
        bundle_name="saliencegate-v2.js",
        bundle_bytes=_bundle() + b"// v2\n",
    ).model_dump(mode="python", warnings="error")
    payload["config"] = OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=first.config.marker,
        owned_fragment=first.config.owned_fragment,
    )
    changed = ProviderInstallationSpec.model_validate(payload)

    with pytest.raises(InstallationError):
        install_provider(changed, KEY)

    assert {path: path.read_bytes() for path in before} == before
    assert not changed.bundle_path.exists()
    assert inspect_provider_installation(first, KEY).drift == ()


@pytest.mark.skipif(os.name != "posix", reason="private mode bits require POSIX")
def test_upgrade_rejects_nonprivate_bootstrap_before_transaction_publication(
    tmp_path: Path,
) -> None:
    first = _make_spec(tmp_path)
    install_provider(first, KEY)
    first.bootstrap_path.chmod(0o644)
    second = _make_spec(
        tmp_path,
        generation=2,
        bundle_name="saliencegate-v2.js",
        bundle_bytes=_bundle() + b"// v2\n",
    )
    before = {
        path: path.read_bytes()
        for path in (
            first.config_path,
            first.bundle_path,
            first.bootstrap_path,
            first.receipt_path,
            first.lock_path,
            first.launcher_path,
        )
    }

    with pytest.raises(InstallationError):
        install_provider(second, KEY)

    assert {path: path.read_bytes() for path in before} == before
    assert not second.bundle_path.exists()
    assert not second.journal_path.exists()
    assert _mode(first.bootstrap_path) == 0o644


def test_git_probe_is_read_only_and_reports_only_managed_project_files(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    if subprocess.run(("git", "--version"), capture_output=True, check=False).returncode != 0:
        pytest.skip("git is unavailable")
    subprocess.run(("git", "init", "--quiet"), cwd=spec.project_root, check=True)
    _make_private_directory(spec.config_path.parent)
    _write_new_private_file(spec.config_path, b"{}")
    subprocess.run(
        ("git", "add", spec.config_path.relative_to(spec.project_root).as_posix()),
        cwd=spec.project_root,
        check=True,
    )
    before = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=spec.project_root,
        capture_output=True,
        check=True,
    ).stdout

    tracked = git_tracked_project_files(spec)

    after = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=spec.project_root,
        capture_output=True,
        check=True,
    ).stdout
    assert tracked == (spec.config_path,)
    assert after == before
    assert not (spec.project_root / ".gitignore").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are platform-specific")
def test_receipt_permissions_and_launcher_contract_fail_closed(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    spec.launcher_path.write_bytes(b"foreign launcher")
    spec.launcher_path.chmod(0o600)
    with pytest.raises(InstallationError):
        install_provider(spec, KEY)
    assert not spec.receipt_path.exists()
