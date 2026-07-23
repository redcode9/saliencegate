"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.integrations.test_installation import (
    KEY,
    _make_command_hook_spec,
    _make_configless_bridge_spec,
    _make_private_directory,
    _make_spec,
)

import saliencegate.integrations.installation as installation
from saliencegate.integrations.config_files import plan_owned_config_install
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationJournal,
    InstallationReceipt,
    InstallationState,
    InstallationStatus,
)
from saliencegate.integrations.registry import ProviderInstallationSpec
from saliencegate.security import InstallationKey
from saliencegate.security.windows import (
    WindowsFileIdentity,
    WindowsPathKind,
    WindowsPathSecurity,
)


def _receipt(spec: ProviderInstallationSpec) -> InstallationReceipt:
    config_edit = (
        None if spec.config is None else plan_owned_config_install(None, spec.config).reverse_edit
    )
    return installation._make_receipt(
        spec,
        KEY,
        state=InstallationState.ENABLED,
        launcher_digest=spec.launcher_digest,
        config_edit=config_edit,
    )


def _replace_model_field(value: object, name: str, replacement: object) -> dict[str, object]:
    payload = value.model_dump(mode="python", warnings="error")  # type: ignore[attr-defined]
    payload[name] = replacement
    return payload


def _case_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


@pytest.mark.parametrize(
    "path",
    (
        Path("relative"),
        Path("/tmp/../escape"),
        Path("/"),
        Path("/tmp/bad\x00name"),
        "not-a-path",
    ),
)
def test_absolute_installation_paths_reject_ambiguous_values(path: object) -> None:
    with pytest.raises(ValueError, match="installation path is invalid"):
        installation._absolute_path(path)  # type: ignore[arg-type]


def test_receipt_models_reject_incomplete_config_bridge_and_alias_bindings(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    receipt = _receipt(spec)
    assert repr(receipt) == "InstallationReceipt(<redacted>)"

    invalid_payloads = [
        _replace_model_field(receipt, "config_edit", None),
        _replace_model_field(receipt, "bundle_digest", None),
        _replace_model_field(receipt, "launcher_path", receipt.receipt_path),
        _replace_model_field(
            receipt,
            "bootstrap_path",
            receipt.bootstrap_path.parent.parent / "elsewhere" / "bootstrap.json",
        ),
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            InstallationReceipt.model_validate(payload)

    command = _receipt(_make_command_hook_spec(_case_root(tmp_path, "command")))
    command_payload = command.model_dump(mode="python", warnings="error")
    command_payload.update(config_path=None, config_edit=None)
    with pytest.raises(ValidationError):
        InstallationReceipt.model_validate(command_payload)


def test_journal_and_status_models_reject_noncanonical_lifecycle_bindings(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    receipt = _receipt(spec)
    journal = installation._journal_for_install(receipt, KEY, prior=None)
    invalid_journals = [
        _replace_model_field(journal, "prior_bundle_path", receipt.bundle_path),
        _replace_model_field(journal, "prior_launcher_path", receipt.launcher_path),
        _replace_model_field(journal, "target_bootstrap_digest", None),
        _replace_model_field(journal, "prior_bootstrap_digest", "0" * 64),
        _replace_model_field(journal, "operation", "uninstall"),
    ]
    for payload in invalid_journals:
        with pytest.raises(ValidationError):
            InstallationJournal.model_validate(payload)

    command_receipt = _receipt(_make_command_hook_spec(_case_root(tmp_path, "command")))
    command_journal = installation._journal_for_install(command_receipt, KEY, prior=None)
    command_payload = command_journal.model_dump(mode="python", warnings="error")
    command_payload.update(
        prior_bundle_path=spec.bundle_path,
        prior_bundle_digest=spec.bundle_digest,
        prior_bootstrap_digest="0" * 64,
    )
    with pytest.raises(ValidationError):
        InstallationJournal.model_validate(command_payload)

    valid_status = installation._status(
        spec,
        KEY,
        disposition=InstallationDisposition.NOOP,
        state=InstallationState.ENABLED,
    )
    for payload in (
        _replace_model_field(valid_status, "drift", ("launcher", "receipt")),
        _replace_model_field(valid_status, "installed", False),
    ):
        with pytest.raises(ValidationError):
            InstallationStatus.model_validate(payload)


def test_seals_decoders_and_identity_helpers_fail_closed_on_wrong_types_and_macs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    receipt = _receipt(spec)
    journal = installation._journal_for_install(receipt, KEY, prior=None)
    receipt_data = installation._encode_model(
        receipt,
        installation.MAX_INSTALLATION_RECEIPT_BYTES,
    )
    journal_data = installation._encode_model(
        journal,
        installation.MAX_INSTALLATION_JOURNAL_BYTES,
    )
    assert installation._decode_receipt(receipt_data, KEY) == receipt
    assert installation._decode_journal(journal_data, KEY) == journal

    for data in (None, b"", b"{", receipt_data + b"\n"):
        with pytest.raises(InstallationError):
            installation._decode_receipt(data, KEY)  # type: ignore[arg-type]
    for data in (None, b"", b"{", journal_data + b"\n"):
        with pytest.raises(InstallationError):
            installation._decode_journal(data, KEY)  # type: ignore[arg-type]

    forged_receipt = receipt.model_copy(update={"receipt_mac": "0" * 64})
    forged_journal = journal.model_copy(update={"journal_mac": "0" * 64})
    with pytest.raises(InstallationError):
        installation._verify_receipt(forged_receipt, KEY)
    with pytest.raises(InstallationError):
        installation._verify_journal(forged_journal, KEY)
    with pytest.raises(InstallationError):
        installation._encode_model(receipt, 1)
    with pytest.raises(InstallationError):
        installation._encode_model(object(), 100)  # type: ignore[arg-type]

    for key, payload in ((object(), b"payload"), (KEY, "payload")):
        with pytest.raises(InstallationError):
            installation._keyed_digest(key, payload, b"domain")  # type: ignore[arg-type]
    with pytest.raises(InstallationError):
        installation.derive_installation_identity(spec, object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        installation.CaptureDigestContext, "workspace_identity", lambda *args: 1 / 0
    )
    with pytest.raises(InstallationError):
        installation._project_digest(spec, KEY)


def test_receipt_inspection_authenticates_exact_path_and_key(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    assert installation._load_receipt_optional(spec, KEY) is None
    assert installation._load_journal_optional(spec, KEY) is None
    with pytest.raises(InstallationError):
        installation.inspect_installation_receipt(spec.receipt_path, KEY)
    with pytest.raises(InstallationError):
        installation.inspect_installation_receipt(spec.receipt_path, object())  # type: ignore[arg-type]

    installation.install_provider(spec, KEY)
    assert installation.inspect_installation_receipt(spec.receipt_path, KEY).receipt_path == (
        spec.receipt_path
    )
    copied = spec.receipt_path.with_name("copied.receipt.json")
    copied.write_bytes(spec.receipt_path.read_bytes())
    copied.chmod(0o600)
    with pytest.raises(InstallationError):
        installation.inspect_installation_receipt(copied, KEY)


def test_directory_authorization_rejects_unsafe_existing_paths_and_builds_nested_private_paths(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "safe" / "nested"
    installation._ensure_directory(nested, private=True)
    assert installation._safe_directory(nested, private=True)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    assert not installation._safe_directory(unsafe, private=False)
    with pytest.raises(InstallationError):
        installation._ensure_directory(unsafe, private=False)
    assert not installation._safe_directory(tmp_path / "absent", private=True)

    for invalid in (Path("relative"), Path("/"), Path("/tmp/../escape"), "not-path"):
        with pytest.raises(InstallationError):
            installation.ensure_private_installation_directory(invalid)  # type: ignore[arg-type]


def test_launcher_publication_is_exclusive_compare_and_swap_and_cleans_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    _make_private_directory(state)
    path = state / "launcher"
    first = b"#!/bin/sh\nexit 0\n"
    second = b"#!/bin/sh\nexit 1\n"
    first_digest = hashlib.sha256(first).hexdigest()
    installation._publish_launcher(path, first)
    assert installation._read_launcher_digest(path) == first_digest
    assert installation._launcher_digest_optional(state / "missing") is None

    with pytest.raises(InstallationError):
        installation._publish_launcher(
            state / "missing-replace", second, replace_digest=first_digest
        )
    with pytest.raises(InstallationError):
        installation._publish_launcher(path, second)
    with pytest.raises(InstallationError):
        installation._publish_launcher(path, second, replace_digest="0" * 64)
    installation._publish_launcher(path, second, replace_digest=first_digest)
    assert path.read_bytes() == second

    path.chmod(0o600)
    with pytest.raises(InstallationError):
        installation._read_launcher_digest(path)
    path.chmod(0o700)

    real_write = installation.os.write
    monkeypatch.setattr(installation.os, "write", lambda _descriptor, _data: 0)
    with pytest.raises(InstallationError):
        installation._publish_launcher(state / "write-failure", first)
    monkeypatch.setattr(installation.os, "write", real_write)
    assert not tuple(state.glob(".*.tmp"))

    with pytest.raises(InstallationError):
        installation._publish_launcher(path, b"")
    with pytest.raises(InstallationError):
        installation._publish_launcher(tmp_path / "unsafe" / "launcher", first)


def test_private_publication_and_deletion_normalize_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state" / "private.json"
    _make_private_directory(path.parent)

    class BadPublication:
        def publish(self, data: bytes, **_kwargs: object) -> object:
            return SimpleNamespace(data=data + b"bad")

    monkeypatch.setattr(
        installation,
        "authorize_atomic_file_publication",
        lambda *args, **kwargs: BadPublication(),
    )
    with pytest.raises(InstallationError):
        installation._publish_private(path, b"data", maximum=100)
    with pytest.raises(InstallationError):
        installation._publish_private(
            path,
            b"data",
            maximum=100,
            managed_parent=1,  # type: ignore[arg-type]
        )

    stable = SimpleNamespace(data=b"different", authorization=object())
    monkeypatch.setattr(installation, "read_stable_file", lambda *args, **kwargs: stable)
    with pytest.raises(InstallationError):
        installation._delete_private_exact(
            path,
            hashlib.sha256(b"expected").hexdigest(),
            maximum=100,
        )
    installation._delete_private_if_present(path.parent / "absent", "0" * 64, maximum=100)


def test_windows_security_predicate_and_stable_absence_are_closed_over_every_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security = WindowsPathSecurity(
        identity=WindowsFileIdentity(volume_serial_number=1, file_id=b"i" * 16),
        kind=WindowsPathKind.DIRECTORY,
        owner_sid="S-1-5-21-1",
        owner_private_dacl=True,
        owner_write_protected_dacl=True,
        owner_traversal_protected_dacl=True,
        hardlink_count=1,
        reparse_tag=None,
    )
    assert installation._safe_windows_directory_security(
        security,
        owner_sid="S-1-5-21-1",
        private=True,
    )
    assert not installation._safe_windows_directory_security(
        security,
        owner_sid="S-1-5-21-2",
        private=True,
    )
    assert not installation._safe_windows_directory_security(
        None,
        owner_sid="S-1-5-21-1",
        private=False,
    )

    leaf = PureWindowsPath("C:/project/missing/receipt.json")
    existing = PureWindowsPath("C:/project")

    class Operations:
        def __init__(self) -> None:
            self.calls: dict[PureWindowsPath, int] = {}

        def inspect_path(self, path: PureWindowsPath) -> object | None:
            self.calls[path] = self.calls.get(path, 0) + 1
            return security if path == existing else None

    authorization = SimpleNamespace(revalidate=lambda: None)
    operations = Operations()
    monkeypatch.setattr(
        installation,
        "authorize_windows_managed_path",
        lambda *args, **kwargs: authorization,
    )
    assert installation._windows_path_is_stably_absent(leaf, operations) is True  # type: ignore[arg-type]
    assert operations.calls[leaf] >= 3

    class RacingOperations(Operations):
        def inspect_path(self, path: PureWindowsPath) -> object | None:
            value = super().inspect_path(path)
            if path == leaf and self.calls[path] >= 2:
                return security
            return value

    with pytest.raises(InstallationError):
        installation._windows_path_is_stably_absent(leaf, RacingOperations())  # type: ignore[arg-type]


def test_project_boundary_capture_revalidates_identity_and_rejects_empty_or_changed_sets(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    identity = installation._managed_directory_identity(spec.project_root)
    boundary = installation._ManagedProjectBoundary.capture(
        spec,
        expected_project_identity=identity,
    )
    assert repr(boundary) == "_ManagedProjectBoundary(<redacted>)"
    boundary.revalidate()
    with pytest.raises(InstallationError):
        installation._ManagedProjectBoundary.capture(
            spec,
            expected_project_identity=("wrong",),
        )
    with pytest.raises(InstallationError):
        installation._ManagedProjectBoundary(identities=()).revalidate()

    old_mode = stat.S_IMODE(spec.project_root.stat().st_mode)
    spec.project_root.chmod(0o755 if old_mode == 0o700 else 0o700)
    with pytest.raises(InstallationError):
        boundary.revalidate()
    spec.project_root.chmod(old_mode)


def test_posix_project_directory_creation_rejects_bad_identity_and_unsafe_existing_leaf(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    identity = installation._managed_directory_identity(spec.project_root)
    with pytest.raises(InstallationError):
        installation._ensure_posix_project_directory(
            spec.project_root,
            spec.config_path.parent,
            expected_project_identity=("bad",),
        )

    unsafe = spec.config_path.parent
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with pytest.raises(InstallationError):
        installation._ensure_posix_project_directory(
            spec.project_root,
            unsafe,
            expected_project_identity=identity,
        )


def test_inspection_reports_orphaned_assets_and_normalizes_config_and_launcher_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    spec.config_path.write_bytes(b'{"marker":"' + spec.config.marker.encode() + b'"}')
    spec.bundle_path.write_bytes(spec.bundle_bytes)
    spec.bundle_path.chmod(0o600)
    spec.bootstrap_path.write_bytes(b"orphan bootstrap")
    spec.bootstrap_path.chmod(0o600)
    spec.launcher_path.write_bytes(spec.launcher_bytes)
    spec.launcher_path.chmod(0o700)

    status = installation.inspect_provider_installation(spec, KEY)
    assert status.state is InstallationState.DISABLED
    assert set(status.drift) >= {"config", "bundle", "bootstrap", "launcher"}

    monkeypatch.setattr(
        installation, "read_config_bytes", lambda _path: (_ for _ in ()).throw(InstallationError())
    )
    monkeypatch.setattr(
        installation,
        "_launcher_digest_optional",
        lambda _path: (_ for _ in ()).throw(InstallationError()),
    )
    failed = installation.inspect_provider_installation(spec, KEY)
    assert "config" in failed.drift
    assert "launcher" in failed.drift


def test_config_removal_plans_cover_absence_drift_noop_and_invalid_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    _make_private_directory(spec.config_path.parent)
    spec.config_path.write_bytes(b'{"foreign":true}')
    installation.install_provider(spec, KEY)
    receipt = installation._load_receipt_optional(spec, KEY)
    assert receipt is not None and receipt.config_edit is not None
    installed = spec.config_path.read_bytes()

    plan = installation._plan_config_removal(receipt)
    assert plan is not None and plan[2] is True
    spec.config_path.unlink()
    with pytest.raises(InstallationError):
        installation._plan_config_removal(receipt)

    spec.config_path.write_bytes(b"{}")
    spec.config_path.chmod(0o600)
    if receipt.config_edit.target_existed:
        with pytest.raises(InstallationError):
            installation._plan_config_removal(receipt)
    else:
        assert installation._plan_config_removal(receipt) == (b"{}", b"{}", False)

    with pytest.raises(InstallationError):
        installation._apply_config_removal(receipt, None)
    with pytest.raises(InstallationError):
        installation._apply_config_removal(receipt, (None, None, True))

    configless = _receipt(_make_configless_bridge_spec(_case_root(tmp_path, "configless")))
    assert installation._plan_config_removal(configless) is None
    installation._apply_config_removal(configless, None)
    with pytest.raises(InstallationError):
        installation._apply_config_removal(configless, (b"old", b"new", True))

    monkeypatch.setattr(installation, "read_config_bytes", lambda _path: b"raced")
    with pytest.raises(InstallationError):
        installation._apply_config_removal(receipt, (installed, installed, False))


def test_public_install_recover_and_uninstall_wrappers_normalize_boundary_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    with pytest.raises(InstallationError):
        installation.install_provider(spec, object())  # type: ignore[arg-type]
    with pytest.raises(InstallationError):
        installation.install_provider(spec, KEY, dry_run=1)  # type: ignore[arg-type]
    with pytest.raises(InstallationError):
        installation.recover_provider_installation(spec, object())  # type: ignore[arg-type]
    with pytest.raises(InstallationError):
        installation.uninstall_provider(spec, object())  # type: ignore[arg-type]

    monkeypatch.setattr(installation, "_require_private_operational_directory", lambda _spec: False)
    with pytest.raises(InstallationError):
        installation.recover_provider_installation(spec, KEY)
    with pytest.raises(InstallationError):
        installation.uninstall_provider(spec, KEY)

    def runtime_failure(_spec: ProviderInstallationSpec) -> bool:
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(installation, "_require_private_operational_directory", runtime_failure)
    with pytest.raises(InstallationError):
        installation.install_provider(spec, KEY)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        installation.install_provider(spec, KEY, _fault_injector=lambda _stage: None)
    with pytest.raises(InstallationError):
        installation.recover_provider_installation(spec, KEY)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        installation.recover_provider_installation(
            spec,
            KEY,
            _fault_injector=lambda _stage: None,
        )
    with pytest.raises(InstallationError):
        installation.uninstall_provider(spec, KEY)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        installation.uninstall_provider(spec, KEY, _fault_injector=lambda _stage: None)


def test_install_noop_upgrade_and_recovery_paths_remain_durable_across_stage_boundaries(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    installed = installation.install_provider(spec, KEY)
    assert installed.disposition is InstallationDisposition.INSTALLED
    assert installation.install_provider(spec, KEY).disposition is InstallationDisposition.NOOP

    next_spec = _make_spec(
        tmp_path,
        generation=2,
        bundle_name="saliencegate-v2.js",
        bundle_bytes=spec.bundle_bytes + b"// v2\n",
    )
    upgraded = installation.install_provider(next_spec, KEY)
    assert upgraded.disposition is InstallationDisposition.UPGRADED
    assert installation.uninstall_provider(next_spec, KEY).state is InstallationState.DISABLED
    assert installation.install_provider(next_spec, KEY).state is InstallationState.ENABLED


@pytest.mark.parametrize(
    "stage",
    (
        "after_journal_publish",
        "after_pending_receipt_publish",
        "after_launcher_publish",
        "after_bundle_publish",
        "after_bootstrap_publish",
        "after_config_publish",
        "after_enabled_receipt_publish",
    ),
)
def test_every_bridge_install_stage_recovery_is_idempotent(
    tmp_path: Path,
    stage: str,
) -> None:
    spec = _make_spec(_case_root(tmp_path, stage))

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        installation.install_provider(spec, KEY, _fault_injector=fail)
    assert installation.recover_provider_installation(spec, KEY).state is InstallationState.ENABLED


@pytest.mark.parametrize(
    "stage",
    (
        "after_journal_publish",
        "after_draining_receipt_publish",
        "after_config_remove",
        "after_asset_remove",
    ),
)
def test_every_bridge_uninstall_stage_recovery_is_idempotent(
    tmp_path: Path,
    stage: str,
) -> None:
    spec = _make_spec(_case_root(tmp_path, stage))
    installation.install_provider(spec, KEY)

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        installation.uninstall_provider(spec, KEY, _fault_injector=fail)
    assert installation.recover_provider_installation(spec, KEY).state is InstallationState.DISABLED


def test_git_probe_handles_process_failures_nonzero_and_oversize_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    monkeypatch.setenv("COMSPEC", "synthetic-command")
    assert "COMSPEC" not in installation._git_environment()

    monkeypatch.setattr(
        installation.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    review = installation.git_project_file_review(spec)
    assert review.disposition is installation.GitProjectFileDisposition.UNAVAILABLE
    assert installation.git_tracked_project_files(spec) == ()

    for completed in (
        subprocess.CompletedProcess(("git",), 1, stdout=b""),
        subprocess.CompletedProcess(("git",), 0, stdout=b"x" * (64 * 1_024 + 1)),
    ):
        monkeypatch.setattr(
            installation.subprocess,
            "run",
            lambda *args, _completed=completed, **kwargs: _completed,
        )
        assert installation.git_tracked_project_files(spec) == ()


def test_git_probe_rejects_relative_cwd_and_project_path_executables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    project_bin = spec.project_root / "bin"
    project_bin.mkdir()
    executable = project_bin / "git"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.chdir(spec.project_root)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((".", "bin", str(project_bin))),
    )
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        raise AssertionError("an untrusted Git executable must never run")

    monkeypatch.setattr(installation.subprocess, "run", run)
    review = installation.git_project_file_review(spec)

    assert review.disposition is installation.GitProjectFileDisposition.UNAVAILABLE
    assert calls == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX case-alias regression")
def test_git_probe_rejects_case_alias_of_project_path_on_case_insensitive_filesystem(
    tmp_path: Path,
) -> None:
    project = tmp_path / "CaseProject"
    project_bin = project / "bin"
    current = tmp_path / "current"
    project_bin.mkdir(parents=True)
    current.mkdir()
    executable = project_bin / "git"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    aliased_project = tmp_path / "caseproject"
    try:
        same_project = os.path.samefile(project, aliased_project)
    except OSError:
        same_project = False
    if not same_project:
        pytest.skip("filesystem is case-sensitive")

    assert (
        installation._find_git_executable(
            str(aliased_project / "bin"),
            project_root=project,
            current_directory=current,
            native_windows=False,
            windows_pathext=None,
            candidate_is_trusted=lambda _candidate: True,
        )
        is None
    )


def test_git_path_identity_failure_rejects_candidate_before_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _case_root(tmp_path, "project")
    current = _case_root(tmp_path, "current")
    candidate_directory = _case_root(tmp_path, "candidate")
    executable = candidate_directory / "git"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    checked_project = project.resolve()
    original_stat = Path.stat

    def failed_project_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == checked_project:
            raise OSError("identity unavailable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failed_project_stat)
    observed: list[Path] = []

    assert installation._path_may_be_within(executable.resolve(), checked_project)
    assert (
        installation._find_git_executable(
            str(candidate_directory),
            project_root=project,
            current_directory=current,
            native_windows=False,
            windows_pathext=None,
            candidate_is_trusted=lambda candidate: observed.append(candidate) is None,
        )
        is None
    )
    assert observed == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode boundary")
def test_git_probe_rejects_an_absolute_executable_below_world_writable_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    with tempfile.TemporaryDirectory(prefix="saliencegate-hostile-git-", dir="/tmp") as raw:
        hostile_directory = Path(raw)
        executable = hostile_directory / "git"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        monkeypatch.setenv("PATH", str(hostile_directory))

        assert installation._resolved_git_executable(spec.project_root) is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX link-count boundary")
def test_git_probe_rejects_a_trusted_path_hard_linked_into_the_project() -> None:
    with tempfile.TemporaryDirectory(
        prefix="saliencegate-hardlink-git-",
        dir=Path.cwd(),
    ) as raw:
        boundary = Path(raw)
        project = boundary / "project"
        current = boundary / "current"
        trusted_bin = boundary / "trusted-bin"
        project.mkdir()
        current.mkdir()
        trusted_bin.mkdir()
        executable = trusted_bin / "git"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)

        assert installation._posix_executable_boundary_is_trusted(executable.resolve())

        os.link(executable, project / "project-git")

        assert executable.stat().st_nlink == 2
        assert not installation._posix_executable_boundary_is_trusted(executable.resolve())
        assert (
            installation._find_git_executable(
                str(trusted_bin),
                project_root=project,
                current_directory=current,
                native_windows=False,
                windows_pathext=None,
                candidate_is_trusted=installation._posix_executable_boundary_is_trusted,
            )
            is None
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and link-count boundary")
def test_posix_git_boundary_allows_privileged_owned_multiply_linked_system_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="saliencegate-system-git-",
        dir=Path.cwd(),
    ) as raw:
        executable = (Path(raw) / "git").resolve()
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        original_stat = Path.stat

        def system_owned_stat(path: Path, *args: object, **kwargs: object) -> object:
            metadata = original_stat(path, *args, **kwargs)
            if path == executable:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_nlink=2,
                    st_uid=0,
                )
            return metadata

        monkeypatch.setattr(Path, "stat", system_owned_stat)

        assert installation._posix_executable_boundary_is_trusted(executable)


def test_git_search_manually_enumerates_safe_windows_executables_without_cwd(
    tmp_path: Path,
) -> None:
    project = _case_root(tmp_path, "project")
    current = _case_root(tmp_path, "current")
    safe = _case_root(tmp_path, "safe")
    for directory in (project, current, safe):
        (directory / "git.exe").write_bytes(b"synthetic executable")
        (directory / "git.cmd").write_bytes(b"synthetic command wrapper")

    trusted = (safe / "git.exe").resolve()
    observed: list[Path] = []

    def trust(candidate: Path) -> bool:
        observed.append(candidate)
        return candidate == trusted

    selected = installation._find_git_executable(
        ";".join(("", ".", "relative", str(project), str(current), str(safe))),
        project_root=project,
        current_directory=current,
        native_windows=True,
        windows_pathext=".CMD;.EXE;.COM",
        candidate_is_trusted=trust,
    )

    assert selected == str(trusted)
    assert observed == [trusted]
    assert (
        installation._find_git_executable(
            str(safe),
            project_root=project,
            current_directory=current,
            native_windows=True,
            windows_pathext=".CMD",
            candidate_is_trusted=lambda _candidate: True,
        )
        is None
    )
    assert (
        installation._find_git_executable(
            str(safe),
            project_root=project,
            current_directory=current,
            native_windows=True,
            windows_pathext=".EXE",
            candidate_is_trusted=lambda _candidate: False,
        )
        is None
    )


def test_windows_git_boundary_requires_current_user_managed_path_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "git.exe"
    executable.write_bytes(b"synthetic executable")
    native_operations = object()
    revalidations: list[bool] = []
    authorization = SimpleNamespace(revalidate=lambda: revalidations.append(True))
    monkeypatch.setattr(
        installation,
        "NativeWindowsSecurityOperations",
        lambda: native_operations,
    )

    def authorize(
        path: PureWindowsPath,
        *,
        kind: WindowsPathKind,
        operations: object,
    ) -> object:
        assert path.name == "git.exe"
        assert kind is WindowsPathKind.FILE
        assert operations is native_operations
        return authorization

    monkeypatch.setattr(installation, "authorize_windows_managed_path", authorize)

    assert installation._windows_executable_boundary_is_trusted(executable.resolve())
    assert revalidations == [True]

    monkeypatch.setattr(
        installation,
        "authorize_windows_managed_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("system-owned or unsafe boundary is intentionally unavailable")
        ),
    )
    assert not installation._windows_executable_boundary_is_trusted(executable.resolve())


def test_spec_validation_and_mac_helpers_normalize_backend_and_project_type_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    spec.project_root.rmdir()
    spec.project_root.write_bytes(b"not a directory")
    with pytest.raises(InstallationError):
        installation._validated_spec(spec)

    spec.project_root.unlink()
    spec.project_root.mkdir()
    receipt = _receipt(spec)
    journal = installation._journal_for_install(receipt, KEY, prior=None)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            InstallationKey,
            "_hmac_sha256",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("hmac")),
        )
        with pytest.raises(InstallationError):
            installation._keyed_digest(KEY, b"payload", b"domain")

    for function, value in (
        (installation._seal_receipt, receipt),
        (installation._verify_receipt, receipt),
        (installation._seal_journal, journal),
        (installation._verify_journal, journal),
    ):
        for error in (InstallationError(), ValueError("backend")):
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    installation,
                    "_keyed_digest",
                    lambda *_args, _error=error, **_kwargs: (_ for _ in ()).throw(_error),
                )
                with pytest.raises(InstallationError):
                    function(value, KEY)


def test_stable_absence_optional_reads_and_receipt_inspection_cover_races_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = PureWindowsPath("C:/project/receipt.json")

    class Present:
        def inspect_path(self, _path: PureWindowsPath) -> object:
            return object()

    assert installation._windows_path_is_stably_absent(leaf, Present()) is False  # type: ignore[arg-type]

    class NoExistingPrefix:
        def inspect_path(self, _path: PureWindowsPath) -> None:
            return None

    with pytest.raises(InstallationError):
        installation._windows_path_is_stably_absent(leaf, NoExistingPrefix())  # type: ignore[arg-type]

    target = tmp_path / "unreadable"
    real_lstat = Path.lstat
    monkeypatch.setattr(
        installation,
        "read_stable_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("read")),
    )

    def denied_lstat(path: Path) -> os.stat_result:
        if path == target:
            raise PermissionError("lstat")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    with pytest.raises(InstallationError):
        installation._read_private_optional(target, maximum=100)
    with pytest.raises(InstallationError):
        installation.inspect_installation_receipt(Path("relative"), KEY)


def test_directory_creation_covers_existing_unsafe_ancestor_created_race_and_generic_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation._ensure_directory(tmp_path, private=True)
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with pytest.raises(InstallationError):
        installation._ensure_directory(unsafe / "child", private=True)

    real_safe = installation._safe_directory
    checks = 0

    def unsafe_after_creation(path: Path, *, private: bool) -> bool:
        nonlocal checks
        checks += 1
        if path.name == "raced":
            return False
        return real_safe(path, private=private)

    monkeypatch.setattr(installation, "_safe_directory", unsafe_after_creation)
    with pytest.raises(InstallationError):
        installation._ensure_directory(tmp_path / "raced", private=True)
    assert checks >= 2

    with monkeypatch.context() as scoped:
        scoped.setattr(installation.os, "name", "generic")
        generic = tmp_path / "generic-private"
        installation.ensure_private_installation_directory(generic)
        assert generic.is_dir()

    with monkeypatch.context() as scoped:
        scoped.setattr(installation.os, "name", "generic")
        scoped.setattr(
            installation,
            "_ensure_directory",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("mkdir")),
        )
        with pytest.raises(InstallationError):
            installation.ensure_private_installation_directory(tmp_path / "failure")


def test_launcher_revalidation_and_backend_faults_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    _make_private_directory(state)
    first = b"#!/bin/sh\nexit 0\n"
    second = b"#!/bin/sh\nexit 1\n"
    path = state / "launcher"
    installation._publish_launcher(path, first)
    first_digest = hashlib.sha256(first).hexdigest()
    real_digest = installation._read_launcher_digest

    calls = 0

    def changed_before_replace(current: Path) -> str:
        nonlocal calls
        calls += 1
        return first_digest if calls == 1 else "0" * 64

    monkeypatch.setattr(installation, "_read_launcher_digest", changed_before_replace)
    with pytest.raises(InstallationError):
        installation._publish_launcher(path, second, replace_digest=first_digest)
    monkeypatch.setattr(installation, "_read_launcher_digest", real_digest)

    final = state / "bad-final"

    def wrong_after_publication(current: Path) -> str:
        if current == final and current.exists():
            return "0" * 64
        return real_digest(current)

    monkeypatch.setattr(installation, "_read_launcher_digest", wrong_after_publication)
    with pytest.raises(InstallationError):
        installation._publish_launcher(final, first)
    monkeypatch.setattr(installation, "_read_launcher_digest", real_digest)

    directory_target = state / "directory-launcher"
    directory_target.mkdir()
    with pytest.raises(InstallationError):
        installation._publish_launcher(directory_target, first)

    backend = state / "backend"
    monkeypatch.setattr(
        installation,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync")),
    )
    with pytest.raises(InstallationError):
        installation._publish_launcher(backend, first)


def test_private_file_and_lock_faults_cover_backend_cleanup_and_post_yield_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    path = spec.receipt_path.parent / "private"
    with monkeypatch.context() as scoped:
        scoped.setattr(
            installation,
            "authorize_atomic_file_publication",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish")),
        )
        with pytest.raises(InstallationError):
            installation._publish_private(path, b"data", maximum=100)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            installation,
            "read_stable_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read")),
        )
        with pytest.raises(InstallationError):
            installation._delete_private_exact(path, "0" * 64, maximum=100)

    real_lstat = Path.lstat

    def denied(path_value: Path) -> os.stat_result:
        if path_value == path:
            raise PermissionError("lstat")
        return real_lstat(path_value)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(InstallationError):
        installation._delete_private_if_present(path, "0" * 64, maximum=100)
    monkeypatch.setattr(Path, "lstat", real_lstat)

    _make_private_directory(spec.lock_path.parent)
    with pytest.raises(InstallationError), installation._installation_lock(spec):
        spec.lock_path.chmod(0o644)


def test_bridge_asset_helpers_reject_constructed_incomplete_models_and_command_bootstrap(
    tmp_path: Path,
) -> None:
    bridge = _make_spec(tmp_path)
    command = _make_command_hook_spec(_case_root(tmp_path, "command"))
    command_payload = command.model_dump(mode="python", warnings="error")
    command_payload.update(
        bundle_path=bridge.bundle_path,
        bootstrap_path=bridge.bootstrap_path,
        bundle_bytes=bridge.bundle_bytes,
        bundle_digest=bridge.bundle_digest,
    )
    forged_command = ProviderInstallationSpec.model_construct(**command_payload)
    with pytest.raises(InstallationError):
        installation._spec_bridge_assets(forged_command)

    bridge_payload = bridge.model_dump(mode="python", warnings="error")
    bridge_payload["bundle_bytes"] = None
    forged_bridge = ProviderInstallationSpec.model_construct(**bridge_payload)
    with pytest.raises(InstallationError):
        installation._spec_bridge_assets(forged_bridge)

    command_receipt = _receipt(command)
    receipt_payload = _receipt(bridge).model_dump(mode="python", warnings="error")
    receipt_payload["bundle_digest"] = None
    incomplete_bridge_receipt = InstallationReceipt.model_construct(**receipt_payload)
    with pytest.raises(InstallationError):
        installation._receipt_bridge_assets(incomplete_bridge_receipt)
    with pytest.raises(InstallationError):
        installation._bootstrap_for(command_receipt)


def test_inspection_error_partition_covers_invalid_key_boundary_receipt_orphan_and_disabled_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    with pytest.raises(InstallationError):
        installation.inspect_provider_installation(spec, object())  # type: ignore[arg-type]

    with monkeypatch.context() as scoped:
        scoped.setattr(
            installation,
            "_require_private_operational_directory",
            lambda _spec: (_ for _ in ()).throw(InstallationError()),
        )
        status = installation.inspect_provider_installation(spec, KEY)
        assert status.drift == ("receipt",)

    installation.install_provider(spec, KEY)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            installation,
            "_load_receipt_optional",
            lambda *_args: (_ for _ in ()).throw(InstallationError()),
        )
        scoped.setattr(installation, "_lock_digest", lambda _spec: "invalid")
        status = installation.inspect_provider_installation(spec, KEY)
        assert status.drift == ("receipt", "lock")

    with monkeypatch.context() as scoped:
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: False)
        status = installation.inspect_provider_installation(spec, KEY)
        assert "receipt" in status.drift

    installation.uninstall_provider(spec, KEY)
    disabled = installation._load_receipt_optional(spec, KEY)
    assert disabled is not None
    spec.config_path.write_bytes(b'{"marker":"' + disabled.config_edit.marker.encode() + b'"}')
    spec.config_path.chmod(0o600)
    spec.bundle_path.write_bytes(spec.bundle_bytes)
    spec.bundle_path.chmod(0o600)
    spec.bootstrap_path.write_bytes(b"disabled-bootstrap")
    spec.bootstrap_path.chmod(0o600)
    spec.launcher_path.write_bytes(spec.launcher_bytes)
    spec.launcher_path.chmod(0o700)
    drifted = installation.inspect_provider_installation(spec, KEY)
    assert set(drifted.drift) >= {"config", "bundle", "bootstrap", "launcher"}


def test_orphan_journal_and_current_config_boundaries_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)

    def fail(stage: str) -> None:
        if stage == "after_journal_publish":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="after_journal_publish"):
        installation.install_provider(spec, KEY, _fault_injector=fail)
    status = installation.inspect_provider_installation(spec, KEY)
    assert status.state is InstallationState.PENDING
    assert "receipt" in status.drift
    installation.recover_provider_installation(spec, KEY)

    configless = _make_configless_bridge_spec(_case_root(tmp_path, "configless"))
    payload = configless.model_dump(mode="python", warnings="error")
    payload["config"] = spec.config
    inconsistent = ProviderInstallationSpec.model_construct(**payload)
    with pytest.raises(InstallationError):
        installation._current_config_for_plan(inconsistent)

    parent = spec.config_path.parent
    moved = parent.with_name("moved-config")
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(InstallationError):
        installation._current_config_for_plan(spec)


def test_install_state_machine_rejects_prior_and_collision_invariants_before_writes(
    tmp_path: Path,
) -> None:
    command_root = _case_root(tmp_path, "command")
    generation_two = _make_command_hook_spec(command_root, generation=2)
    installation.install_provider(generation_two, KEY)
    generation_one = _make_command_hook_spec(command_root, generation=1)
    with pytest.raises(InstallationError):
        installation.install_provider(generation_one, KEY)

    installation.uninstall_provider(generation_two, KEY)
    generation_two.launcher_path.write_bytes(generation_two.launcher_bytes)
    generation_two.launcher_path.chmod(0o700)
    with pytest.raises(InstallationError):
        installation.install_provider(generation_two, KEY)

    fresh = _make_spec(_case_root(tmp_path, "fresh"))
    _make_private_directory(fresh.config_path.parent)
    fresh.bundle_path.write_bytes(b"foreign bundle")
    fresh.bundle_path.chmod(0o600)
    with pytest.raises(InstallationError):
        installation.install_provider(fresh, KEY)


def test_recover_and_uninstall_prerequisites_reject_missing_lock_journal_receipt_and_drift(
    tmp_path: Path,
) -> None:
    fresh = _make_spec(tmp_path)
    with pytest.raises(InstallationError):
        installation.recover_provider_installation(fresh, KEY)

    installation.install_provider(fresh, KEY)
    with pytest.raises(InstallationError):
        installation.recover_provider_installation(fresh, KEY)

    fresh.lock_path.unlink()
    with pytest.raises(InstallationError):
        installation.uninstall_provider(fresh, KEY)

    drift_root = _case_root(tmp_path, "drift")
    drift = _make_spec(drift_root)
    installation.install_provider(drift, KEY)
    drift.bundle_path.write_bytes(b"drift")
    with pytest.raises(InstallationError):
        installation.uninstall_provider(drift, KEY)


def test_remaining_receipt_journal_and_bridge_model_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _receipt(_make_spec(tmp_path))
    command = _receipt(_make_command_hook_spec(_case_root(tmp_path, "command-model")))
    command_payload = command.model_dump(mode="python", warnings="error")
    command_payload.update(
        bundle_path=bridge.bundle_path,
        bootstrap_path=bridge.bootstrap_path,
        bundle_digest=bridge.bundle_digest,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            InstallationReceipt,
            "installation_kind",
            property(lambda _self: installation.ProviderInstallationKind.COMMAND_HOOK),
        )
        with pytest.raises(ValidationError, match="command-hook receipt declares bridge assets"):
            InstallationReceipt.model_validate(command_payload)

    journal = installation._journal_for_install(bridge, KEY, prior=None)
    lifecycle = journal.model_dump(mode="python", warnings="error")
    lifecycle["operation"] = installation._JournalOperation.UNINSTALL
    with pytest.raises(ValidationError, match="lifecycle binding"):
        InstallationJournal.model_validate(lifecycle)

    forged_command = command.model_copy(
        update={
            "bundle_path": bridge.bundle_path,
            "bootstrap_path": bridge.bootstrap_path,
            "bundle_digest": bridge.bundle_digest,
        }
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            InstallationReceipt,
            "installation_kind",
            property(lambda _self: installation.ProviderInstallationKind.COMMAND_HOOK),
        )
        with pytest.raises(InstallationError):
            installation._receipt_bridge_assets(forged_command)


def test_directory_root_lock_and_current_config_error_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "exists", lambda _path: False)
        with pytest.raises(InstallationError):
            installation._ensure_directory(tmp_path / "never-present", private=True)

    spec = _make_spec(_case_root(tmp_path, "unsafe-lock"))
    _make_private_directory(spec.lock_path.parent)
    with monkeypatch.context() as scoped:
        scoped.setattr(installation, "_safe_lock_metadata", lambda _metadata: False)
        with pytest.raises(InstallationError), installation._installation_lock(spec):
            pass

    with monkeypatch.context() as scoped:
        real_lstat = Path.lstat

        def failed_parent(path: Path):
            if path == spec.config_path.parent:
                raise OSError("parent inspection")
            return real_lstat(path)

        scoped.setattr(Path, "lstat", failed_parent)
        with pytest.raises(InstallationError):
            installation._current_config_for_plan(spec)


def test_inspection_rejects_config_pair_and_reports_bundle_bootstrap_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    receipt = _receipt(spec)

    def common(scoped: pytest.MonkeyPatch, observed: InstallationReceipt) -> None:
        scoped.setattr(installation, "_require_private_operational_directory", lambda _spec: True)
        scoped.setattr(installation, "_load_journal_optional", lambda *_args: None)
        scoped.setattr(installation, "_lock_digest", lambda _spec: "a" * 64)
        scoped.setattr(installation, "_load_receipt_optional", lambda *_args: observed)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        scoped.setattr(installation, "read_config_bytes", lambda _path: b"current")
        scoped.setattr(
            installation,
            "_read_launcher_digest",
            lambda _path: observed.launcher_digest,
        )

    incomplete = receipt.model_copy(update={"config_edit": None})
    with monkeypatch.context() as scoped:
        common(scoped, incomplete)
        with pytest.raises(InstallationError):
            installation.inspect_provider_installation(spec, KEY)

    assert receipt.bundle_path is not None and receipt.bootstrap_path is not None
    moved = receipt.model_copy(update={"bundle_path": receipt.bundle_path.with_name("moved.js")})
    with monkeypatch.context() as scoped:
        common(scoped, moved)
        scoped.setattr(installation, "_path_digest", lambda *_args, **_kwargs: moved.bundle_digest)
        scoped.setattr(
            installation,
            "inspect_integration_bootstrap",
            lambda _path: installation._bootstrap_for(moved),
        )
        status = installation.inspect_provider_installation(spec, KEY)
        assert "bundle" in status.drift

    with monkeypatch.context() as scoped:
        common(scoped, receipt)
        scoped.setattr(
            installation, "_path_digest", lambda *_args, **_kwargs: receipt.bundle_digest
        )
        scoped.setattr(installation, "inspect_integration_bootstrap", lambda _path: object())
        status = installation.inspect_provider_installation(spec, KEY)
        assert "bootstrap" in status.drift


def test_posix_and_generic_project_directory_identity_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    identity = installation._managed_directory_identity(spec.project_root)
    with pytest.raises(InstallationError):
        installation._ensure_posix_project_directory(
            spec.project_root,
            spec.project_root / ".." / "escape",
            expected_project_identity=identity,
        )
    with pytest.raises(InstallationError):
        installation._ensure_posix_project_directory(
            spec.project_root,
            spec.config_path.parent,
            expected_project_identity=(0, 0, 0, 0),
        )

    with monkeypatch.context() as scoped:
        values = iter((identity, (0, 0, 0, 0)))
        scoped.setattr(installation, "_managed_directory_identity", lambda _path: next(values))
        with pytest.raises(InstallationError):
            installation._ensure_posix_project_directory(
                spec.project_root,
                spec.config_path.parent,
                expected_project_identity=identity,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(installation.os, "name", "generic")
        scoped.setattr(installation, "_managed_directory_identity", lambda _path: (0, 0, 0, 0))
        with pytest.raises(InstallationError):
            installation._ensure_project_directory(
                spec,
                spec.config_path.parent,
                expected_project_identity=identity,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(installation.os, "name", "generic")
        values = iter((identity, (0, 0, 0, 0)))
        scoped.setattr(installation, "_managed_directory_identity", lambda _path: next(values))
        scoped.setattr(installation, "_ensure_directory", lambda *_args, **_kwargs: None)
        with pytest.raises(InstallationError):
            installation._ensure_project_directory(
                spec,
                spec.config_path.parent,
                expected_project_identity=identity,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(installation, "_ensure_project_directory", lambda *_args, **_kwargs: None)
        scoped.setattr(installation, "_managed_directory_identity", lambda _path: (0, 0, 0, 0))
        with pytest.raises(InstallationError):
            installation._ensure_install_directories(
                spec,
                expected_project_identity=identity,
            )


def _install_guard_defaults(
    scoped: pytest.MonkeyPatch,
    *,
    prior: InstallationReceipt | None,
    current: bytes | None = None,
    launcher_digest: str | None = None,
) -> None:
    scoped.setattr(installation, "_load_journal_optional", lambda *_args: None)
    scoped.setattr(installation, "_load_receipt_optional", lambda *_args: prior)
    scoped.setattr(installation, "_current_config_for_plan", lambda _spec: current)
    scoped.setattr(installation, "_launcher_digest_optional", lambda _path: launcher_digest)


def test_install_locked_recovery_config_prior_and_generation_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    boundary = SimpleNamespace(revalidate=lambda: None)

    with monkeypatch.context() as scoped:
        marker = object()
        scoped.setattr(installation, "_load_journal_optional", lambda *_args: marker)
        scoped.setattr(installation, "_recover_locked", lambda *_args, **_kwargs: "recovered")
        assert installation._install_locked(spec, KEY, boundary, fault_injector=None) == "recovered"

    forged_spec = spec.model_copy(update={"config_path": None})
    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=None)
        with pytest.raises(InstallationError):
            installation._install_locked(forged_spec, KEY, boundary, fault_injector=None)

    prior = _receipt(spec)
    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=prior, launcher_digest=prior.launcher_digest)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: False)
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)

    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=prior, launcher_digest="f" * 64)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)

    pending = prior.model_copy(update={"state": InstallationState.PENDING})
    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=pending)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)

    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=None, launcher_digest="f" * 64)
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)


def test_install_locked_bridge_upgrade_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    prior = _receipt(spec)
    boundary = SimpleNamespace(revalidate=lambda: None)

    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=prior, launcher_digest=prior.launcher_digest)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        scoped.setattr(installation, "_config_edit_matches_spec", lambda *_args: True)
        scoped.setattr(
            installation,
            "inspect_provider_installation",
            lambda *_args: SimpleNamespace(drift=(), model_copy=lambda **_kwargs: object()),
        )
        scoped.setattr(
            installation,
            "_receipt_bridge_assets",
            lambda _receipt: (
                prior.bundle_path.with_name("other.js"),
                prior.bootstrap_path,
                prior.bundle_digest,
            ),
        )
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)

    next_spec = _make_spec(
        _case_root(tmp_path, "upgrade-none"),
        generation=2,
        bundle_name="saliencegate-v2.js",
    )
    first_spec = _make_spec(next_spec.project_root, generation=1)
    first = _receipt(first_spec)
    with monkeypatch.context() as scoped:
        _install_guard_defaults(scoped, prior=first, launcher_digest=first.launcher_digest)
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        scoped.setattr(installation, "_receipt_bridge_assets", lambda _receipt: None)
        with pytest.raises(InstallationError):
            installation._install_locked(next_spec, KEY, boundary, fault_injector=None)

    changed = _make_spec(
        _case_root(tmp_path, "same-path"),
        generation=2,
        bundle_bytes=spec.bundle_bytes + b"changed",
    )
    changed_prior = _receipt(_make_spec(changed.project_root, generation=1))
    with monkeypatch.context() as scoped:
        _install_guard_defaults(
            scoped,
            prior=changed_prior,
            launcher_digest=changed_prior.launcher_digest,
        )
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        with pytest.raises(InstallationError):
            installation._install_locked(changed, KEY, boundary, fault_injector=None)

    no_config = prior.model_copy(update={"config_path": None, "config_edit": None})
    plan = plan_owned_config_install(None, spec.config)
    assert spec.config is not None
    with monkeypatch.context() as scoped:
        _install_guard_defaults(
            scoped,
            prior=no_config,
            launcher_digest=no_config.launcher_digest,
            current=plan.installed_bytes,
        )
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        scoped.setattr(installation, "_path_digest", lambda *_args, **_kwargs: None)
        with pytest.raises(InstallationError):
            installation._install_locked(spec, KEY, boundary, fault_injector=None)


def test_install_locked_upgrade_rejects_existing_new_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _case_root(tmp_path, "upgrade-collision")
    first_spec = _make_spec(root, generation=1)
    next_spec = _make_spec(root, generation=2, bundle_name="saliencegate-v2.js")
    prior = _receipt(first_spec)
    assert first_spec.config is not None
    current = plan_owned_config_install(None, first_spec.config).installed_bytes
    boundary = SimpleNamespace(revalidate=lambda: None)
    prior_bootstrap_digest = hashlib.sha256(
        installation.encode_integration_bootstrap(installation._bootstrap_for(prior))
    ).hexdigest()
    values = iter((prior.bundle_digest, prior_bootstrap_digest, "f" * 64, None))

    with monkeypatch.context() as scoped:
        _install_guard_defaults(
            scoped,
            prior=prior,
            current=current,
            launcher_digest=prior.launcher_digest,
        )
        scoped.setattr(installation, "_receipt_matches_spec", lambda *_args, **_kwargs: True)
        scoped.setattr(installation, "_path_digest", lambda *_args, **_kwargs: next(values))
        with pytest.raises(InstallationError):
            installation._install_locked(next_spec, KEY, boundary, fault_injector=None)
