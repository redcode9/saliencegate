from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.commands.global_capture import run_global_connect
from saliencegate.domain import canonical_json
from saliencegate.integrations import global_runtime as runtime_module
from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
from saliencegate.integrations.codex import CODEX_PROFILE
from saliencegate.integrations.global_installation import global_provider_installation_spec
from saliencegate.integrations.global_runtime import (
    GlobalCaptureRuntimeError,
    resolve_global_project_root,
    try_build_global_capture_hook_dependencies,
)
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import ProviderAlias
from saliencegate.security import (
    InstallationKey,
    default_installation_key_path,
    load_installation_key,
)


def _environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    (home / ".config" / "opencode").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    return {
        "HOME": os.fspath(home),
        "XDG_STATE_HOME": os.fspath(state),
        "PATH": os.environ.get("PATH", ""),
    }


@pytest.mark.parametrize("workspace", ("relative", "", "bad\x00path"))
def test_global_project_root_rejects_invalid_workspace_paths(workspace: str) -> None:
    with pytest.raises(GlobalCaptureRuntimeError):
        resolve_global_project_root(workspace)


def test_global_project_root_handles_standalone_and_git_file_workspaces(tmp_path: Path) -> None:
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    assert resolve_global_project_root(os.fspath(standalone)) == standalone

    repository = tmp_path / "repository"
    nested = repository / "nested"
    nested.mkdir(parents=True)
    (repository / ".git").write_text("gitdir: ../metadata\n", encoding="utf-8")
    assert resolve_global_project_root(os.fspath(nested)) == repository


def test_global_project_root_rejects_non_directories_and_symlinked_git_markers(
    tmp_path: Path,
) -> None:
    regular_file = tmp_path / "workspace"
    regular_file.write_bytes(b"not a directory")
    with pytest.raises(GlobalCaptureRuntimeError):
        resolve_global_project_root(os.fspath(regular_file))

    repository = tmp_path / "repository"
    repository.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (repository / ".git").symlink_to(metadata, target_is_directory=True)
    with pytest.raises(GlobalCaptureRuntimeError):
        resolve_global_project_root(os.fspath(repository))


@dataclass
class _Adapter:
    adaptation: object = ()

    def capabilities(self) -> tuple[str, ...]:
        return ("capture",)

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> object:
        assert source == b"rebound"
        assert isinstance(context, CaptureDigestContext)
        return self.adaptation

    def transport_chunk(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[bytes, CaptureDigestContext]:
        return source, context


def test_rebound_adapter_forwards_only_the_authenticated_source() -> None:
    context = CaptureDigestContext(InstallationKey(b"k" * 32))
    adapter = runtime_module._ReboundBridgeAdapter(
        _Adapter(),
        source=b"parent",
        rebound=b"rebound",
    )

    assert repr(adapter) == str(adapter) == "_ReboundBridgeAdapter(<redacted>)"
    assert adapter.capabilities() == ("capture",)
    assert adapter.adapt_bytes(b"parent", context=context) == ()
    assert adapter.transport_chunk(b"parent", context=context) == (b"rebound", context)

    with pytest.raises(GlobalCaptureRuntimeError):
        adapter.adapt_bytes(b"different", context=context)
    with pytest.raises(GlobalCaptureRuntimeError):
        adapter.transport_chunk(bytearray(b"parent"), context=context)  # type: ignore[arg-type]


def test_rebound_adapter_rejects_incomplete_or_invalid_adapters() -> None:
    context = CaptureDigestContext(InstallationKey(b"k" * 32))
    incomplete = runtime_module._ReboundBridgeAdapter(
        object(),
        source=b"parent",
        rebound=b"rebound",
    )
    with pytest.raises(GlobalCaptureRuntimeError):
        incomplete.capabilities()
    with pytest.raises(GlobalCaptureRuntimeError):
        incomplete.adapt_bytes(b"parent", context=context)
    with pytest.raises(GlobalCaptureRuntimeError):
        incomplete.transport_chunk(b"parent", context=context)

    invalid_result = runtime_module._ReboundBridgeAdapter(
        _Adapter(adaptation=[]),
        source=b"parent",
        rebound=b"rebound",
    )
    with pytest.raises(GlobalCaptureRuntimeError):
        invalid_result.adapt_bytes(b"parent", context=context)


def test_global_runtime_returns_none_when_no_global_installation_exists(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    assert (
        try_build_global_capture_hook_dependencies(
            ProviderAlias.OPENCODE,
            b"{}",
            connection_id="sgg-" + ("a" * 64),
            environ=environment,
        )
        is None
    )
    with pytest.raises(GlobalCaptureRuntimeError):
        try_build_global_capture_hook_dependencies(
            "opencode",  # type: ignore[arg-type]
            b"{}",
            connection_id="sgg-" + ("a" * 64),
            environ=environment,
        )


def test_global_runtime_dependencies_reject_mismatched_pipeline_identities(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    run_global_connect(provider="opencode", environ=environment)

    key = load_installation_key(default_installation_key_path(environ=environment))
    spec = materialize_provider_launcher(
        global_provider_installation_spec("opencode", environ=environment),
        key,
    )
    assert spec.bootstrap_path is not None
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    source = canonical_json(
        {
            "schema_version": "capture-batch/v1",
            "bootstrap": bootstrap.model_dump(mode="json", warnings="error"),
            "batch_id": "a" * 64,
            "session_id": "runtime-boundary-session",
            "workspace_path": os.fspath(project),
            "chunk_index": 0,
            "chunk_count": 1,
            "events": [],
        }
    )
    connection_id = derive_installation_identity(spec, key).connection_id
    dependencies = try_build_global_capture_hook_dependencies(
        ProviderAlias.OPENCODE,
        source,
        connection_id=connection_id,
        environ=environment,
    )
    assert dependencies is not None

    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.validate_registry(CODEX_PROFILE)
    registration = dependencies.validate_registry(spec.profile)
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.validate_receipt(spec.profile, "sgg-" + ("b" * 64), registration)
    receipt = dependencies.validate_receipt(spec.profile, connection_id, registration)
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.validate_connection(spec.profile, connection_id, registration, object())
    runtime = dependencies.validate_connection(spec.profile, connection_id, registration, receipt)

    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.load_context(object())
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.resolve_connection_id(runtime, "sgg-" + ("c" * 64))
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.resolve_adapter(object())
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.open_store(object())
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.open_spool(object())
    with pytest.raises(GlobalCaptureRuntimeError):
        dependencies.mark_health(object(), object())  # type: ignore[arg-type]
