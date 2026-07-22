from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from saliencegate.integrations import claude_code, codex, opencode, pi
from saliencegate.integrations.environment import (
    PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS,
    CaptureEnvironmentError,
    environment_without_provider_credentials,
)


class _UnreadableCredentialEnvironment(Mapping[str, str]):
    def __init__(self) -> None:
        self.read_keys: list[str] = []
        self._keys = ("PATH", *sorted(PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS))

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> str:
        self.read_keys.append(key)
        if key.upper() in PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS:
            raise AssertionError("provider credential value was read")
        if key == "PATH":
            return "/trusted/bin"
        raise KeyError(key)


def test_environment_projection_never_reads_provider_credentials() -> None:
    environment = _UnreadableCredentialEnvironment()

    assert environment_without_provider_credentials(environment) == {"PATH": "/trusted/bin"}
    assert environment.read_keys == ["PATH"]


def test_environment_projection_filters_provider_keys_case_insensitively() -> None:
    environment = _UnreadableCredentialEnvironment()
    environment._keys = ("PATH", "openai_api_key", "Anthropic_Api_Key")

    assert environment_without_provider_credentials(environment) == {"PATH": "/trusted/bin"}
    assert environment.read_keys == ["PATH"]


@pytest.mark.parametrize(
    ("builder", "error_type"),
    (
        (codex.build_capture_hook_dependencies, codex.CodexIntegrationError),
        (claude_code.build_capture_hook_dependencies, claude_code.ClaudeCodeIntegrationError),
        (opencode.build_capture_hook_dependencies, opencode.OpenCodeIntegrationError),
        (pi.build_capture_hook_dependencies, pi.PiIntegrationError),
    ),
)
def test_hook_dependency_builders_never_read_provider_credentials(
    builder: Callable[..., object],
    error_type: type[ValueError],
) -> None:
    environment = _UnreadableCredentialEnvironment()

    with pytest.raises(error_type):
        builder(
            b"{}",
            connection_id="capture-environment-test",
            environ=environment,
        )

    assert environment.read_keys == ["PATH"]


@pytest.mark.parametrize(
    ("probe", "error_type"),
    (
        (codex.probe_codex_environment, codex.CodexIntegrationError),
        (
            claude_code.probe_claude_code_environment,
            claude_code.ClaudeCodeIntegrationError,
        ),
    ),
)
def test_environment_probes_never_read_provider_credentials(
    probe: Callable[..., object],
    error_type: type[ValueError],
) -> None:
    environment = _UnreadableCredentialEnvironment()

    with pytest.raises(error_type):
        probe(environ=environment)

    assert environment.read_keys == ["PATH"]


@pytest.mark.parametrize(
    "builder",
    (
        codex.provider_installation_spec,
        claude_code.provider_installation_spec,
        opencode.provider_installation_spec,
        pi.provider_installation_spec,
    ),
)
def test_provider_specs_never_read_provider_credentials(
    builder: Callable[..., object],
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = _UnreadableCredentialEnvironment()

    builder(project, environ=environment)

    assert environment.read_keys == ["PATH"]


@pytest.mark.parametrize(
    "environment",
    (
        {1: "not-a-string-key"},
        {"PATH": object()},
    ),
)
def test_environment_projection_rejects_non_string_boundaries(
    environment: Mapping[str, str],
) -> None:
    with pytest.raises(CaptureEnvironmentError, match=r"^capture environment is invalid$"):
        environment_without_provider_credentials(environment)
