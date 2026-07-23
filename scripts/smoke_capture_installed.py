from __future__ import annotations

import _socket
import json
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module, resources
from io import BytesIO
from pathlib import Path

_PROVIDER_CREDENTIAL_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    }
)
_POISONED_CREDENTIAL = "provider-credential-read-must-fail"
_PROVIDERS = ("codex", "claude-code", "opencode", "pi")
_EMIT_CODEX_FIXTURE = "emit-codex-fixture"
_VALIDATE_CODEX_REPORT = "validate-codex-report"
_INSTALLED_E2E = "installed-e2e"
_SOCKET_DENIAL_FLAG = "SALIENCEGATE_ARTIFACT_SOCKET_DENIAL"
_SOCKET_STARTUP_LOG = "SALIENCEGATE_ARTIFACT_SOCKET_STARTUP_LOG"
_SOCKET_STARTUP_RECORD = b"installed-artifact-socket-denial-active\n"

_BRIDGE_CALLBACK_SOURCE = r"""
import { fileURLToPath } from "node:url";
const fail = (message) => { throw new Error(message); };
const [provider, assetHref, bootstrapHref] = process.argv.slice(1);
if (
  (provider !== "opencode" && provider !== "pi") ||
  typeof assetHref !== "string" ||
  typeof bootstrapHref !== "string"
) fail("invalid installed bridge proof arguments");
const inputChunks = [];
let inputBytes = 0;
for await (const chunk of process.stdin) {
  inputBytes += chunk.byteLength;
  if (inputBytes > 65_536) fail("installed bridge proof input is oversized");
  inputChunks.push(chunk);
}
let callbackInput;
try { callbackInput = JSON.parse(Buffer.concat(inputChunks).toString("utf8")); }
catch { fail("installed bridge proof input is invalid"); }
const nativeSession = callbackInput?.native_session;
const contentSentinel = callbackInput?.content_sentinel;
if (typeof nativeSession !== "string" || typeof contentSentinel !== "string") {
  fail("installed bridge proof input is invalid");
}
if (globalThis[Symbol.for("saliencegate.network-denial/v1")] !== true) {
  fail("installed bridge proof is missing network denial");
}
let fetchError;
try { await globalThis.fetch("http://127.0.0.1:9/"); } catch (error) { fetchError = error; }
if (fetchError?.code !== "ERR_SALIENCEGATE_NETWORK_DISABLED") {
  fail("installed bridge proof has the wrong network denial");
}
const credentialKeys = new Set([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID",
]);
const presentCredentials = Object.keys(process.env).filter((key) =>
  credentialKeys.has(key.toUpperCase()),
);
if (presentCredentials.length !== credentialKeys.size) {
  fail("installed bridge proof is missing poisoned credentials");
}
const inheritedGuard = process.env.SALIENCEGATE_NETWORK_GUARD;
if (typeof inheritedGuard !== "string" || !inheritedGuard.startsWith("file:")) {
  fail("installed bridge proof is missing its child network guard");
}
process.env.NODE_OPTIONS = `--import=${inheritedGuard}`;

const imported = await import(assetHref);
if (!(imported.saliencegateBootstrap instanceof URL)) {
  fail("installed bridge proof has no bootstrap reference");
}
if (
  fileURLToPath(imported.saliencegateBootstrap) !== fileURLToPath(bootstrapHref)
) {
  fail("installed bridge proof selected the wrong bootstrap");
}

if (provider === "opencode") {
  if (imported.default?.id !== "saliencegate") fail("invalid installed OpenCode id");
  if (typeof imported.default?.server !== "function") {
    fail("invalid installed OpenCode callback shape");
  }
  const hooks = await imported.default.server(Object.freeze({}));
  if (typeof hooks?.event !== "function" || typeof hooks?.dispose !== "function") {
    fail("invalid installed OpenCode hooks");
  }
  const part = {
    id: "artifact-installed-part",
    sessionID: nativeSession,
    messageID: "artifact-installed-message",
    type: "tool",
    callID: contentSentinel,
    tool: "read",
  };
  await hooks.event({
    event: {
      type: "message.part.updated",
      properties: {
        part: {
          ...part,
          state: { status: "pending", input: { path: contentSentinel } },
        },
      },
    },
  });
  await hooks.event({
    event: {
      type: "message.part.updated",
      properties: {
        part: {
          ...part,
          state: { status: "completed", input: { path: contentSentinel } },
        },
      },
    },
  });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: nativeSession } },
  });
  await hooks.event({
    event: {
      type: "session.deleted",
      properties: { info: { id: nativeSession } },
    },
  });
  await hooks.dispose();
} else {
  if (typeof imported.default !== "function") {
    fail("invalid installed Pi callback shape");
  }
  const handlers = new Map();
  const registered = await imported.default({
    on(name, handler) {
      if (handlers.has(name) || typeof handler !== "function") {
        fail("invalid installed Pi handler registration");
      }
      handlers.set(name, handler);
    },
  });
  if (registered !== undefined) fail("installed Pi extension returned a value");
  const expectedHandlers = [
    "session_start",
    "before_agent_start",
    "tool_execution_start",
    "tool_execution_end",
    "agent_settled",
    "session_compact",
    "session_tree",
    "session_shutdown",
  ];
  if (
    handlers.size !== expectedHandlers.length ||
    expectedHandlers.some((name) => !handlers.has(name))
  ) fail("installed Pi extension registered the wrong handlers");
  const context = Object.freeze({
    sessionManager: Object.freeze({ getSessionId() { return nativeSession; } }),
  });
  const invoke = async (name, value) => {
    const result = await handlers.get(name)(value, context);
    if (result !== undefined) fail(`installed Pi callback returned a value: ${name}`);
  };
  await invoke("session_start", { type: "session_start", reason: "startup" });
  await invoke("before_agent_start", {
    type: "before_agent_start",
    prompt: contentSentinel,
    images: [],
    systemPrompt: contentSentinel,
    systemPromptOptions: {},
  });
  await invoke("tool_execution_start", {
    type: "tool_execution_start",
    toolCallId: contentSentinel,
    toolName: "read",
    args: { path: contentSentinel },
  });
  await invoke("tool_execution_end", {
    type: "tool_execution_end",
    toolCallId: contentSentinel,
    toolName: "read",
    result: { content: contentSentinel },
    isError: false,
  });
  await invoke("agent_settled", { type: "agent_settled" });
  await invoke("session_shutdown", { type: "session_shutdown", reason: "quit" });
}
process.stdout.write("capture-installed-bridge-callbacks-ok\n");
""".strip()


@dataclass(frozen=True, slots=True)
class _ProviderCase:
    alias: str
    profile: str
    host_version: str
    module: str
    fixture: str | None
    expected_state: str
    expected_headline: str
    expected_action_identities: int
    expected_structured_results: int
    expected_limits: frozenset[str]
    expected_coverage_degraded: bool
    evidence_detector: str | None


@dataclass(frozen=True, slots=True)
class _InstalledCommandCallback:
    command: str | tuple[str, ...]
    executable: str | None = None
    launcher_environment: str | None = None


_CASES = (
    _ProviderCase(
        "codex",
        "codex-hooks/v1",
        "0.144.6",
        "saliencegate.integrations.codex",
        "codex-hooks-v1.json",
        "open",
        "insufficient_evidence",
        1,
        0,
        frozenset({"detector_minimum_not_met", "session_open"}),
        True,
        "repeated_action",
    ),
    _ProviderCase(
        "claude-code",
        "claude-code-hooks/v1",
        "2.1.204",
        "saliencegate.integrations.claude_code",
        "claude-code-hooks-v1.json",
        "open",
        "insufficient_evidence",
        1,
        1,
        frozenset({"session_open"}),
        True,
        "tool_error",
    ),
    _ProviderCase(
        "opencode",
        "opencode-plugin/v1",
        "1.18.3",
        "saliencegate.integrations.opencode",
        None,
        "closed",
        "no_current_evidence",
        1,
        1,
        frozenset(),
        False,
        "tool_error",
    ),
    _ProviderCase(
        "pi",
        "pi-extension/v1",
        "0.80.10",
        "saliencegate.integrations.pi",
        None,
        "closed",
        "insufficient_evidence",
        1,
        1,
        frozenset({"no_applicable_detector"}),
        False,
        None,
    ),
)


class _UnreadableCredentialEnvironment(Mapping[str, str]):
    """Expose names but make every provider credential value unreadable."""

    __slots__ = ("_source",)

    def __init__(self, source: Mapping[str, str]) -> None:
        self._source = source

    def __iter__(self) -> Iterator[str]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, key: str) -> str:
        if key.upper() in _PROVIDER_CREDENTIAL_KEYS:
            raise RuntimeError("provider credential value read during capture proof")
        return self._source[key]


def _require_isolated_offline_runtime() -> None:
    if sys.flags.isolated != 1 or os.environ.get("PYTHONPATH", ""):
        raise RuntimeError("capture artifact proof requires isolated Python without PYTHONPATH")
    for constructor in (socket.socket, _socket.socket):
        opened = None
        try:
            opened = constructor(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            continue
        finally:
            if opened is not None:
                opened.close()
        raise RuntimeError("capture artifact proof requires the socket guard")
    package = import_module("saliencegate")
    package_file = getattr(package, "__file__", None)
    if type(package_file) is not str or not Path(package_file).resolve().is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise RuntimeError("capture artifact proof did not import the installed package")
    if shutil.which("node") is not None or shutil.which("npm") is not None:
        raise RuntimeError("capture artifact runtime unexpectedly exposes Node or npm")


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fixture_payload(case: _ProviderCase, event_name: str) -> dict[str, object]:
    if case.fixture is None:
        raise RuntimeError("capture artifact fixture selection is invalid")
    source = (
        resources.files("saliencegate.integrations")
        .joinpath("fixtures")
        .joinpath(case.fixture)
        .read_bytes()
    )
    document = json.loads(source)
    if type(document) is not dict or type(document.get("events")) is not list:
        raise RuntimeError("capture artifact fixture is invalid")
    for event in document["events"]:
        if type(event) is dict and event.get("event_name") == event_name:
            payload = event.get("payload")
            if type(payload) is dict:
                return payload
    raise RuntimeError("capture artifact fixture event is missing")


def _hook_payloads(
    case: _ProviderCase,
    *,
    project: Path,
    bootstrap: dict[str, object] | None,
    native_session: str,
    content_sentinel: str,
) -> tuple[bytes, ...]:
    from saliencegate.domain import canonical_json

    if case.fixture is not None:
        payloads: list[bytes] = []
        for event_name in ("SessionStart", "PreToolUse", "PostToolUse"):
            payload = _fixture_payload(case, event_name)
            payload["cwd"] = str(project)
            payload["session_id"] = native_session
            if event_name != "SessionStart":
                payload["tool_input"] = {"path": content_sentinel}
            payloads.append(canonical_json(payload))
        return tuple(payloads)

    if bootstrap is None:
        raise RuntimeError("capture artifact bridge bootstrap is missing")
    events: list[dict[str, object]] = [
        {
            "kind": "tool_started",
            "session_id": native_session,
            "event_id": "artifact-event-start",
            "call_id": content_sentinel,
            "tool": "read",
            "identity_authority": "exact" if case.alias == "opencode" else "coarse",
        },
        {
            "kind": "tool_finished",
            "session_id": native_session,
            "event_id": "artifact-event-finish",
            "call_id": content_sentinel,
            "outcome": "succeeded",
        },
        {
            "kind": "session_finished",
            "session_id": native_session,
            "event_id": "artifact-event-close",
        },
    ]
    if case.alias == "opencode":
        events[0]["input"] = {"path": content_sentinel}
    if case.alias == "pi":
        for event_id, event in enumerate(events, start=1):
            event["window_discriminator"] = "7" * 64
            event["event_id"] = str(event_id)
        events[-1]["reason"] = "quit"
    batch: dict[str, object] = {
        "schema_version": "capture-batch/v1",
        "bootstrap": bootstrap,
        "batch_id": "8" * 64 if case.alias == "pi" else "2" * 64,
        "session_id": native_session,
        "chunk_index": 0,
        "chunk_count": 1,
        "events": events,
    }
    if case.alias == "pi":
        batch["window_discriminator"] = "7" * 64
    return (canonical_json(batch),)


def _validate_report(
    report: object,
    *,
    case: _ProviderCase,
) -> bytes:
    from saliencegate.capture.report import CaptureSessionReport, encode_capture_session_report

    checked = CaptureSessionReport.model_validate(report)
    if (
        checked.schema_version != "capture-session-report/v1"
        or checked.profile_id.value != case.profile
        or checked.host_version != case.host_version
        or checked.compatibility_status.value != "verified"
        or checked.model_calls != 0
        or checked.decision_authority is not False
        or checked.confirmatory is not False
        or checked.raw_content_persisted is not False
        or checked.transcript_read is not False
        or checked.source_authentication != "none_same_user_untrusted"
        or checked.at_rest_integrity != "hmac_sha256_local_mutation_detection"
        or checked.report_integrity != "sha256_canonical_body"
    ):
        raise RuntimeError("capture artifact report contract is invalid")
    detector_rows = {row.signal_type.value: row for row in checked.detectors}
    evidence_detector = (
        None if case.evidence_detector is None else detector_rows.get(case.evidence_detector)
    )
    limits = {item.value for item in checked.coverage.limits}
    if (
        checked.session_state.value != case.expected_state
        or checked.headline.value != case.expected_headline
        or checked.counts.captured_events < 3
        or checked.counts.projected_events < 2
        or checked.counts.action_identities != case.expected_action_identities
        or checked.counts.structured_results != case.expected_structured_results
        or checked.counts.detected_signals != 0
        or checked.coverage.spool_status.value != "verified_clean_drained"
        or checked.coverage.coverage_degraded is not case.expected_coverage_degraded
        or checked.coverage.gap_count != 0
        or checked.coverage.drop_count != 0
        or checked.coverage.overflow_count != 0
        or checked.coverage.queued_spool_events != 0
        or checked.coverage.dropped_spool_events != 0
        or limits != case.expected_limits
        or (case.evidence_detector is not None and evidence_detector is None)
        or (
            evidence_detector is not None
            and (
                evidence_detector.authorized_observation_count != 1
                or evidence_detector.unresolved_observation_count != 0
                or evidence_detector.detected_count != 0
            )
        )
    ):
        raise RuntimeError("capture artifact report evidence is invalid")
    return encode_capture_session_report(checked)


def _retained_report_contract(report: object) -> dict[str, object]:
    from saliencegate.capture.report import CaptureSessionReport

    body = CaptureSessionReport.model_validate(report).model_dump(mode="json", warnings="error")
    for field in ("normalization_digest", "report_digest", "snapshot_digest"):
        body.pop(field)
    coverage = body.get("coverage")
    if type(coverage) is not dict:
        raise RuntimeError("capture artifact report coverage is invalid")
    coverage.pop("spool_observation_tag")
    return body


def _capture_executable() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(sys.executable).parent / f"saliencegate-capture-hook{suffix}"
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError("capture artifact hook executable is invalid")
    return path


def _scan_for_sentinels(root: Path, sentinels: tuple[bytes, ...]) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("capture artifact proof created a symbolic link")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if any(sentinel in payload for sentinel in sentinels):
            raise RuntimeError("capture artifact proof persisted a raw sentinel")


def _provider_sentinels(case: _ProviderCase) -> tuple[bytes, bytes]:
    return (
        f"artifact-{case.alias}-native-session-sentinel".encode(),
        f"artifact-{case.alias}-raw-content-sentinel".encode(),
    )


def _artifact_environment() -> _UnreadableCredentialEnvironment:
    available_credentials = {
        key.upper() for key in os.environ if key.upper() in _PROVIDER_CREDENTIAL_KEYS
    }
    if available_credentials != _PROVIDER_CREDENTIAL_KEYS:
        raise RuntimeError("capture artifact proof requires poisoned provider credentials")
    source: dict[str, str] = {}
    for key in os.environ:
        if key.upper() not in _PROVIDER_CREDENTIAL_KEYS:
            source[key] = os.environ[key]
    source.update({key: _POISONED_CREDENTIAL for key in _PROVIDER_CREDENTIAL_KEYS})
    return _UnreadableCredentialEnvironment(source)


def _subprocess_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Materialize only the process boundary; credential values remain poison sentinels."""

    result: dict[str, str] = {}
    for key in environment:
        if key.upper() in _PROVIDER_CREDENTIAL_KEYS or key.upper() == "NODE_OPTIONS":
            continue
        value = environment[key]
        if type(key) is not str or type(value) is not str:
            raise RuntimeError("capture artifact subprocess environment is invalid")
        result[key] = value
    result.update({key: _POISONED_CREDENTIAL for key in _PROVIDER_CREDENTIAL_KEYS})
    result[_SOCKET_DENIAL_FLAG] = "1"
    return result


def _socket_guard_start_count(environment: Mapping[str, str]) -> int:
    raw = environment.get(_SOCKET_STARTUP_LOG)
    if type(raw) is not str or not raw:
        raise RuntimeError("capture artifact child socket guard log is invalid")
    path = Path(raw)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return 0
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("capture artifact child socket guard log is invalid")
    payload = path.read_bytes()
    if payload.replace(_SOCKET_STARTUP_RECORD, b""):
        raise RuntimeError("capture artifact child socket guard log is invalid")
    return payload.count(_SOCKET_STARTUP_RECORD)


def _require_regular_file(path: Path, *, executable: bool = False) -> Path:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise RuntimeError("capture artifact installed callback path is invalid") from None
    if (
        path.is_symlink()
        or resolved.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or (executable and os.name == "posix" and not os.access(resolved, os.X_OK))
    ):
        raise RuntimeError("capture artifact installed callback path is invalid")
    return resolved


def _launcher_command(
    launcher: Path,
    *,
    environment: Mapping[str, str],
    platform: str | None = None,
) -> _InstalledCommandCallback:
    selected_platform = os.name if platform is None else platform
    if selected_platform not in {"nt", "posix"}:
        raise RuntimeError("capture artifact callback platform is invalid")
    resolved = _require_regular_file(launcher, executable=True)
    if selected_platform == "posix":
        return _InstalledCommandCallback(command=(os.fspath(resolved),))
    system_root = environment.get("SystemRoot") or environment.get("SYSTEMROOT")
    if type(system_root) is not str:
        raise RuntimeError("capture artifact Windows command interpreter is invalid")
    command = _require_regular_file(Path(system_root) / "System32" / "cmd.exe")
    resolved_command = os.fspath(command)
    resolved_launcher = os.fspath(resolved)
    if '"' in resolved_command or '"' in resolved_launcher:
        raise RuntimeError("capture artifact installed callback path is invalid")
    return _InstalledCommandCallback(
        command=(f'"{resolved_command}" /d /v:off /s /c ""%SALIENCEGATE_LAUNCHER%""'),
        executable=resolved_command,
        launcher_environment=resolved_launcher,
    )


def _invoke_installed_command_callback(
    *,
    callback: _InstalledCommandCallback,
    payload: bytes,
    project: Path,
    environment: Mapping[str, str],
) -> None:
    process_environment = _subprocess_environment(environment)
    starts_before = _socket_guard_start_count(environment)
    if callback.launcher_environment is not None:
        process_environment["SALIENCEGATE_LAUNCHER"] = callback.launcher_environment
    try:
        completed = subprocess.run(
            callback.command,
            cwd=project,
            env=process_environment,
            executable=callback.executable,
            shell=False,
            check=False,
            capture_output=True,
            input=payload,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("capture artifact installed callback timed out") from None
    combined = completed.stdout + completed.stderr
    if any(
        sentinel in combined
        for sentinel in (_POISONED_CREDENTIAL.encode(), *_provider_sentinels_for_scan())
    ):
        raise RuntimeError("capture artifact installed callback exposed sensitive data")
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise RuntimeError("capture artifact installed callback failed")
    if _socket_guard_start_count(environment) != starts_before + 1:
        raise RuntimeError("capture artifact installed callback lacked child socket denial")


def _provider_sentinels_for_scan() -> tuple[bytes, ...]:
    return tuple(sentinel for case in _CASES for sentinel in _provider_sentinels(case))


def _invoke_installed_bridge_callbacks(
    case: _ProviderCase,
    *,
    spec: object,
    node: Path,
    network_guard: Path,
    project: Path,
    environment: Mapping[str, str],
    native_session: str,
    content_sentinel: str,
) -> None:
    bundle_path = getattr(spec, "bundle_path", None)
    bootstrap_path = getattr(spec, "bootstrap_path", None)
    if not isinstance(bundle_path, Path) or not isinstance(bootstrap_path, Path):
        raise RuntimeError("capture artifact installed bridge binding is invalid")
    bundle = _require_regular_file(bundle_path)
    bootstrap = _require_regular_file(bootstrap_path)
    resolved_node = _require_regular_file(node, executable=True)
    guard = _require_regular_file(network_guard)
    process_environment = _subprocess_environment(environment)
    process_environment["SALIENCEGATE_NETWORK_GUARD"] = guard.as_uri()
    starts_before = _socket_guard_start_count(environment)
    try:
        completed = subprocess.run(
            (
                resolved_node,
                "--no-warnings",
                "--import",
                guard.as_uri(),
                "--input-type=module",
                "--eval",
                _BRIDGE_CALLBACK_SOURCE,
                case.alias,
                bundle.as_uri(),
                bootstrap.as_uri(),
            ),
            cwd=project,
            env=process_environment,
            check=False,
            capture_output=True,
            input=json.dumps(
                {
                    "content_sentinel": content_sentinel,
                    "native_session": native_session,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("capture artifact installed bridge callback timed out") from None
    combined = completed.stdout + completed.stderr
    forbidden = (
        _POISONED_CREDENTIAL.encode(),
        native_session.encode(),
        content_sentinel.encode(),
    )
    if any(sentinel in combined for sentinel in forbidden):
        raise RuntimeError("capture artifact installed bridge callback exposed sensitive data")
    if (
        completed.returncode != 0
        or completed.stdout != b"capture-installed-bridge-callbacks-ok\n"
        or completed.stderr
    ):
        raise RuntimeError("capture artifact installed bridge callback failed")
    if _socket_guard_start_count(environment) <= starts_before:
        raise RuntimeError("capture artifact installed bridge lacked child socket denial")


def _configured_hook_handler(document: object, event_name: str) -> Mapping[str, object]:
    if not isinstance(document, Mapping):
        raise RuntimeError("capture artifact installed config binding is invalid")
    hooks = document.get("hooks")
    groups = hooks.get(event_name) if isinstance(hooks, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise RuntimeError("capture artifact installed config binding is invalid")
    handlers = groups[0].get("hooks")
    if not isinstance(handlers, list) or len(handlers) != 1 or not isinstance(handlers[0], Mapping):
        raise RuntimeError("capture artifact installed config binding is invalid")
    handler = handlers[0]
    if handler.get("type") != "command" or handler.get("timeout") != 3:
        raise RuntimeError("capture artifact installed config binding is invalid")
    return handler


def _installed_command_callbacks(
    case: _ProviderCase,
    spec: object,
    *,
    environment: Mapping[str, str],
) -> dict[str, _InstalledCommandCallback]:
    launcher = getattr(spec, "launcher_path", None)
    if not isinstance(launcher, Path):
        raise RuntimeError("capture artifact installed launcher binding is invalid")
    installed_launcher = _require_regular_file(launcher, executable=True)
    if os.name == "nt" and (
        " " not in os.fspath(installed_launcher) or "&" not in os.fspath(installed_launcher)
    ):
        raise RuntimeError("capture artifact Windows launcher metacharacter proof is incomplete")
    config_path = getattr(spec, "config_path", None)
    bundle_path = getattr(spec, "bundle_path", None)
    bootstrap_path = getattr(spec, "bootstrap_path", None)
    if case.fixture is not None:
        config = getattr(spec, "config", None)
        marker = getattr(config, "marker", None)
        if not isinstance(config_path, Path) or type(marker) is not str:
            raise RuntimeError("capture artifact installed config binding is invalid")
        configured = _require_regular_file(config_path).read_bytes()
        if configured.count(marker.encode("ascii")) != 1:
            raise RuntimeError("capture artifact installed config binding is invalid")
        try:
            document = (
                tomllib.loads(configured.decode("utf-8", errors="strict"))
                if case.alias == "codex"
                else json.loads(configured)
            )
        except (UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError):
            raise RuntimeError("capture artifact installed config binding is invalid") from None
        callbacks: dict[str, _InstalledCommandCallback] = {}
        for event_name in ("SessionStart", "PreToolUse", "PostToolUse"):
            handler = _configured_hook_handler(document, event_name)
            command = handler.get("command")
            if type(command) is not str or not command or "\0" in command:
                raise RuntimeError("capture artifact installed config binding is invalid")
            if case.alias == "codex":
                expected = (
                    subprocess.list2cmdline((os.fspath(installed_launcher),))
                    if os.name == "nt"
                    else shlex.quote(os.fspath(installed_launcher))
                )
                if command != expected:
                    raise RuntimeError("capture artifact installed config binding is invalid")
                callbacks[event_name] = _launcher_command(
                    installed_launcher,
                    environment=environment,
                )
                continue
            arguments = handler.get("args")
            if (
                not isinstance(arguments, list)
                or not arguments
                or any(type(argument) is not str or "\0" in argument for argument in arguments)
            ):
                raise RuntimeError("capture artifact installed config binding is invalid")
            configured_executable = _require_regular_file(Path(command), executable=True)
            callbacks[event_name] = _InstalledCommandCallback(
                command=(os.fspath(configured_executable), *arguments)
            )
        return callbacks
    if not isinstance(bundle_path, Path) or not isinstance(bootstrap_path, Path):
        raise RuntimeError("capture artifact installed bridge binding is invalid")
    _require_regular_file(bundle_path)
    _require_regular_file(bootstrap_path)
    return {}


def _exercise_provider(
    case: _ProviderCase,
    *,
    root: Path,
    environment: Mapping[str, str],
    capture_executable: Path,
    bridge_node: Path | None = None,
    network_guard: Path | None = None,
) -> tuple[bytes, ...]:
    from saliencegate.capture import CaptureSessionState
    from saliencegate.commands.capture.connect import render_connect_json, run_connect
    from saliencegate.commands.capture.disconnect import render_disconnect_json, run_disconnect
    from saliencegate.commands.capture.report import run_capture_report
    from saliencegate.commands.capture.sessions import render_sessions_json, run_sessions
    from saliencegate.commands.capture.status import render_status_json, run_status
    from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
    from saliencegate.integrations.hook import run_capture_hook
    from saliencegate.integrations.installation import derive_installation_identity
    from saliencegate.security import load_installation_key

    project = root / "projects" / case.alias
    project.mkdir(mode=0o700, parents=True)
    output = root / "outputs" / case.alias
    output.mkdir(mode=0o700, parents=True)
    module = import_module(case.module)

    def resolve_spec(alias: object, candidate: Path) -> object:
        if getattr(alias, "value", None) != case.alias or candidate != project:
            raise RuntimeError("capture artifact provider resolution is invalid")
        if case.alias in {"codex", "claude-code"}:
            return module.provider_installation_spec(
                candidate,
                environ=environment,
                probe_host=True,
            )
        return module.provider_installation_spec(candidate, environ=environment)

    connected = run_connect(
        provider=case.alias,
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    )
    if not connected.capture_enabled or connected.provider.value != case.alias:
        raise RuntimeError("capture artifact provider did not connect")
    _write_private(output / "connect.json", render_connect_json(connected).encode())

    if case.alias in {"codex", "claude-code"}:
        spec = module.provider_installation_spec(
            project,
            environ=environment,
            probe_host=True,
        )
    else:
        spec = module.provider_installation_spec(project, environ=environment)
    if spec.host_version != case.host_version:
        raise RuntimeError("capture artifact host contract is invalid")
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    bootstrap = (
        None
        if spec.bootstrap_path is None
        else inspect_integration_bootstrap(spec.bootstrap_path).model_dump(
            mode="json", warnings="error"
        )
    )
    native_sentinel, content_sentinel_bytes = _provider_sentinels(case)
    native_session = native_sentinel.decode()
    content_sentinel = content_sentinel_bytes.decode()
    installed_e2e = bridge_node is not None or network_guard is not None
    if installed_e2e and (bridge_node is None or network_guard is None):
        raise RuntimeError("capture artifact installed callback proof is incomplete")
    if installed_e2e:
        command_callbacks = _installed_command_callbacks(case, spec, environment=environment)
        if case.fixture is not None:
            for payload in _hook_payloads(
                case,
                project=project,
                bootstrap=bootstrap,
                native_session=native_session,
                content_sentinel=content_sentinel,
            ):
                try:
                    event_name = json.loads(payload).get("hook_event_name")
                except (UnicodeError, json.JSONDecodeError, AttributeError):
                    raise RuntimeError(
                        "capture artifact installed callback payload is invalid"
                    ) from None
                if type(event_name) is not str:
                    raise RuntimeError("capture artifact installed callback payload is invalid")
                callback = command_callbacks.get(event_name)
                if callback is None:
                    raise RuntimeError("capture artifact installed callback binding is missing")
                _invoke_installed_command_callback(
                    callback=callback,
                    payload=payload,
                    project=project,
                    environment=environment,
                )
        else:
            _invoke_installed_bridge_callbacks(
                case,
                spec=spec,
                node=bridge_node,
                network_guard=network_guard,
                project=project,
                environment=environment,
                native_session=native_session,
                content_sentinel=content_sentinel,
            )
    else:
        for payload in _hook_payloads(
            case,
            project=project,
            bootstrap=bootstrap,
            native_session=native_session,
            content_sentinel=content_sentinel,
        ):
            result = run_capture_hook(
                ("--profile", case.profile, "--connection", identity.connection_id),
                BytesIO(payload),
                environ=environment,
                capture_executable=capture_executable,
            )
            if result != 0:
                raise RuntimeError("capture artifact hook failed")

    status = run_status(
        provider=case.alias,
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    )
    if (
        len(status.providers) != 1
        or status.providers[0].status.value != "active_observed"
        or status.providers[0].drift
        or status.providers[0].session_count != 1
    ):
        raise RuntimeError("capture artifact status did not observe one session")
    _write_private(output / "status.json", render_status_json(status).encode())

    sessions = run_sessions(project=project, provider=case.alias, environ=environment)
    if len(sessions.sessions) != 1 or sessions.sessions[0].provider != case.alias:
        raise RuntimeError("capture artifact session listing is invalid")
    _write_private(output / "sessions.json", render_sessions_json(sessions).encode())
    _scan_for_sentinels(
        root.parent,
        (_POISONED_CREDENTIAL.encode(), native_session.encode(), content_sentinel.encode()),
    )

    before_path = output / "report-before-disconnect.json"
    before = run_capture_report(
        latest=True,
        project=project,
        output_path=before_path,
        environ=environment,
    )
    before_bytes = _validate_report(
        before,
        case=case,
    )
    if before_path.read_bytes() != before_bytes:
        raise RuntimeError("capture artifact report publication is invalid")

    disconnected = run_disconnect(
        provider=case.alias,
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    )
    if disconnected.disposition != "uninstalled":
        raise RuntimeError("capture artifact provider did not disconnect")
    _write_private(output / "disconnect.json", render_disconnect_json(disconnected).encode())

    after_status = run_status(
        provider=case.alias,
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    )
    if (
        len(after_status.providers) != 1
        or after_status.providers[0].status.value != "not_installed"
        or after_status.providers[0].session_count != 1
    ):
        raise RuntimeError("capture artifact disconnect status is invalid")
    listed = sessions.sessions[0]
    after_path = output / "report-after-disconnect.json"
    after = run_capture_report(
        latest=False,
        session_id=listed.session_id,
        project=project,
        output_path=after_path,
        environ=environment,
    )
    after_bytes = _validate_report(
        after,
        case=case,
    )
    if (
        _retained_report_contract(after) != _retained_report_contract(before)
        or after_path.read_bytes() != after_bytes
    ):
        raise RuntimeError("capture artifact report was not retained after disconnect")
    expected_state = CaptureSessionState(case.expected_state)
    if after.session_state is not expected_state:
        raise RuntimeError("capture artifact session lifecycle is invalid")
    return (native_sentinel, content_sentinel_bytes)


def _emit_codex_fixture(
    *,
    project: Path,
    scan_root: Path,
    environment: Mapping[str, str],
) -> None:
    from saliencegate.integrations.hook import run_capture_hook
    from saliencegate.integrations.installation import derive_installation_identity
    from saliencegate.security import load_installation_key

    case = _CASES[0]
    if (
        case.alias != "codex"
        or not project.is_dir()
        or not scan_root.is_dir()
        or not project.is_relative_to(scan_root)
    ):
        raise RuntimeError("capture artifact Codex fixture scope is invalid")
    module = import_module(case.module)
    spec = module.provider_installation_spec(
        project,
        environ=environment,
        probe_host=True,
    )
    if spec.profile.value != case.profile or spec.host_version != case.host_version:
        raise RuntimeError("capture artifact Codex fixture contract is invalid")
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    native_sentinel, content_sentinel = _provider_sentinels(case)
    for payload in _hook_payloads(
        case,
        project=project,
        bootstrap=None,
        native_session=native_sentinel.decode(),
        content_sentinel=content_sentinel.decode(),
    ):
        result = run_capture_hook(
            ("--profile", case.profile, "--connection", identity.connection_id),
            BytesIO(payload),
            environ=environment,
            capture_executable=_capture_executable(),
        )
        if result != 0:
            raise RuntimeError("capture artifact Codex fixture hook failed")
    _scan_for_sentinels(
        scan_root,
        (_POISONED_CREDENTIAL.encode(), native_sentinel, content_sentinel),
    )


def _validate_codex_report(*, report_path: Path, scan_root: Path) -> None:
    from saliencegate.capture.report import decode_capture_session_report
    from saliencegate.security import inspect_private_file_location

    case = _CASES[0]
    if case.alias != "codex" or not scan_root.is_dir() or not report_path.is_relative_to(scan_root):
        raise RuntimeError("capture artifact Codex report scope is invalid")
    authorization = inspect_private_file_location(report_path)
    payload = report_path.read_bytes()
    report = decode_capture_session_report(payload)
    if payload != _validate_report(report, case=case):
        raise RuntimeError("capture artifact Codex report publication is invalid")
    authorization.revalidate()
    native_sentinel, content_sentinel = _provider_sentinels(case)
    _scan_for_sentinels(
        scan_root,
        (_POISONED_CREDENTIAL.encode(), native_sentinel, content_sentinel),
    )


def exercise(
    root: Path,
    *,
    bridge_node: Path | None = None,
    network_guard: Path | None = None,
) -> None:
    _require_isolated_offline_runtime()
    if root.exists() or root.is_symlink():
        raise RuntimeError("capture artifact workspace must not already exist")
    root.mkdir(mode=0o700, parents=False)
    os.chmod(root, 0o700)
    environment = _artifact_environment()
    executable = _capture_executable()
    sentinels: list[bytes] = [_POISONED_CREDENTIAL.encode()]
    for case in _CASES:
        sentinels.extend(
            _exercise_provider(
                case,
                root=root,
                environment=environment,
                capture_executable=executable,
                bridge_node=bridge_node,
                network_guard=network_guard,
            )
        )
        _scan_for_sentinels(root.parent, tuple(sentinels))
    if tuple(case.alias for case in _CASES) != _PROVIDERS:
        raise RuntimeError("capture artifact provider matrix is incomplete")


def main() -> int:
    if len(sys.argv) == 2:
        exercise(Path(sys.argv[1]).resolve())
        print("capture-installed-artifact-ok")
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == _INSTALLED_E2E:
        exercise(
            Path(sys.argv[2]).resolve(),
            bridge_node=Path(sys.argv[3]).resolve(strict=True),
            network_guard=Path(sys.argv[4]).resolve(strict=True),
        )
        print("capture-installed-connectors-e2e-ok")
        return 0
    if len(sys.argv) != 4:
        return 2
    _require_isolated_offline_runtime()
    environment = _artifact_environment()
    subject = Path(sys.argv[2]).resolve(strict=True)
    scan_root = Path(sys.argv[3]).resolve(strict=True)
    if sys.argv[1] == _EMIT_CODEX_FIXTURE:
        _emit_codex_fixture(
            project=subject,
            scan_root=scan_root,
            environment=environment,
        )
        print("capture-codex-fixture-ok")
        return 0
    if sys.argv[1] == _VALIDATE_CODEX_REPORT:
        _validate_codex_report(report_path=subject, scan_root=scan_root)
        print("capture-codex-report-ok")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
