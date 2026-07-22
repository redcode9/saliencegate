from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath

EXPECTED_NODE_VERSION = "v22.19.0"
EXPECTED_NPM_VERSION = "10.9.3"
NETWORK_GUARD = "connectors/scripts/deny-network.mjs"
BUNDLES = {
    "opencode": (
        "saliencegate/integrations/assets/opencode-plugin.js",
        "src/saliencegate/integrations/assets/opencode-plugin.js",
    ),
    "pi": (
        "saliencegate/integrations/assets/pi-extension.js",
        "src/saliencegate/integrations/assets/pi-extension.js",
    ),
}
PROVIDER_CREDENTIAL_KEYS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
)
POISONED_CREDENTIAL = "provider-credential-read-must-fail"
PROVIDER_CREDENTIAL_KEY_SET = frozenset(PROVIDER_CREDENTIAL_KEYS)
RUNTIME_ASSET_NAMES = {
    "opencode": "saliencegate.js",
    "pi": "saliencegate.ts",
}
RUNTIME_PROFILES = {
    "opencode": "opencode-plugin/v1",
    "pi": "pi-extension/v1",
}

LAUNCHER_SMOKE_SOURCE = r"""#!/usr/bin/env node
import { appendFile } from "node:fs/promises";

const credentialKeys = new Set([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID",
]);
const present = Object.keys(process.env).filter((key) => credentialKeys.has(key.toUpperCase()));
if (present.length !== 0) throw new Error("provider credential reached connector launcher");
if (globalThis[Symbol.for("saliencegate.network-denial/v1")] !== true) {
  throw new Error("network denial was not inherited by connector launcher");
}
let bytes = 0;
for await (const chunk of process.stdin) bytes += chunk.length;
if (bytes === 0) throw new Error("connector launcher received no capture chunk");
const output = process.env.SALIENCEGATE_ARTIFACT_LAUNCH_LOG;
if (typeof output !== "string" || output.length === 0) throw new Error("missing proof log");
await appendFile(
  output,
  `${JSON.stringify({
    network_denial: true,
    provider_credential_keys_present: present,
    schema_version: "connector-artifact-launch/v1",
    stdin_bytes: bytes,
  })}\n`,
  { encoding: "utf8", mode: 0o600 },
);
"""

POSIX_LAUNCHER_SOURCE = (
    b'#!/bin/sh\nexec "$SALIENCEGATE_ARTIFACT_NODE" "$SALIENCEGATE_ARTIFACT_LAUNCH_SOURCE"\n'
)
WINDOWS_LAUNCHER_SOURCE = (
    b'@echo off\r\n"%SALIENCEGATE_ARTIFACT_NODE%" "%SALIENCEGATE_ARTIFACT_LAUNCH_SOURCE%"\r\n'
)

NODE_SMOKE_SOURCE = r"""
const fail = (message) => { throw new Error(message); };
const opencode = await import(process.env.SALIENCEGATE_OPENCODE_ASSET);
const pi = await import(process.env.SALIENCEGATE_PI_ASSET);
if (opencode.default?.id !== "saliencegate") fail("invalid OpenCode connector id");
if (typeof opencode.default?.server !== "function") fail("invalid OpenCode connector shape");
if (typeof pi.default !== "function") fail("invalid Pi connector shape");
for (const module of [opencode, pi]) {
  if (!(module.saliencegateBootstrap instanceof URL)) fail("invalid bootstrap reference");
  if (module.saliencegateBootstrap.protocol !== "file:") fail("non-local bootstrap reference");
  if (!module.saliencegateBootstrap.pathname.endsWith("/saliencegate.bootstrap.json")) {
    fail("invalid bootstrap filename");
  }
}
for (const specifier of [
  "node:_http_agent", "_http_agent", "node:_http_client", "_http_client",
  "node:_http_common", "_http_common", "node:_http_incoming", "_http_incoming",
  "node:_http_outgoing", "_http_outgoing", "node:_http_server", "_http_server",
  "node:_tls_common", "_tls_common", "node:_tls_wrap", "_tls_wrap",
  "node:net", "net", "node:tls", "tls", "node:http", "http",
  "node:https", "https", "node:dgram", "dgram", "node:dns", "dns",
]) {
  let importError;
  try { await import(specifier); } catch (error) { importError = error; }
  if (importError?.code !== "ERR_SALIENCEGATE_NETWORK_DISABLED") {
    fail(`network import returned the wrong denial: ${specifier}`);
  }
  let builtinError;
  try { process.getBuiltinModule(specifier); } catch (error) { builtinError = error; }
  if (builtinError?.code !== "ERR_SALIENCEGATE_NETWORK_DISABLED") {
    fail(`builtin lookup returned the wrong denial: ${specifier}`);
  }
}
let fetchError;
try { await globalThis.fetch("http://127.0.0.1:9/"); } catch (error) { fetchError = error; }
if (fetchError?.code !== "ERR_SALIENCEGATE_NETWORK_DISABLED") fail("wrong fetch denial");
let websocketError;
try { new globalThis.WebSocket("ws://127.0.0.1:9/"); } catch (error) { websocketError = error; }
if (websocketError?.code !== "ERR_SALIENCEGATE_NETWORK_DISABLED") {
  fail("wrong WebSocket denial");
}
const inheritedGuard = process.env.SALIENCEGATE_NETWORK_GUARD;
if (typeof inheritedGuard !== "string" || !inheritedGuard.startsWith("file:")) {
  fail("missing child network guard");
}
process.env.NODE_OPTIONS = `--import=${inheritedGuard}`;

const openCodeHooks = await opencode.default.server(Object.freeze({}));
const openCodePart = {
  id: "artifact-part",
  sessionID: "artifact-opencode-session",
  messageID: "artifact-message",
  type: "tool",
  callID: "artifact-call",
  tool: "read",
};
await openCodeHooks.event({
  event: {
    type: "message.part.updated",
    properties: {
      part: { ...openCodePart, state: { status: "pending", input: { path: "synthetic" } } },
    },
  },
});
await openCodeHooks.event({
  event: {
    type: "message.part.updated",
    properties: {
      part: { ...openCodePart, state: { status: "completed", input: { path: "synthetic" } } },
    },
  },
});
await openCodeHooks.event({
  event: {
    type: "session.idle",
    properties: { sessionID: "artifact-opencode-session" },
  },
});
await openCodeHooks.dispose();

const piHandlers = new Map();
await pi.default({
  on(name, handler) {
    if (piHandlers.has(name)) fail(`duplicate Pi handler: ${name}`);
    piHandlers.set(name, handler);
  },
});
const piContext = {
  sessionManager: {
    getSessionId() { return "019c0eaf-7b31-7000-8000-000000000001"; },
  },
};
await piHandlers.get("session_start")(
  { type: "session_start", reason: "startup" },
  piContext,
);
await piHandlers.get("session_shutdown")(
  { type: "session_shutdown", reason: "quit" },
  piContext,
);
process.stdout.write("connector-artifacts-ok\n");
""".strip()


class VerificationError(RuntimeError):
    pass


def _locate_regular_command(command: str, *, label: str) -> tuple[Path, Path]:
    if type(command) is not str or not command or "\0" in command:
        raise VerificationError(f"connector artifact smoke {label} command is invalid")
    located = shutil.which(command)
    if located is None:
        raise VerificationError(f"connector artifact smoke could not resolve {label}")
    try:
        resolved = Path(located).resolve(strict=True)
        metadata = resolved.stat(follow_symlinks=False)
    except OSError:
        raise VerificationError(f"connector artifact smoke could not resolve {label}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise VerificationError(f"connector artifact smoke could not resolve {label}")
    return Path(located), resolved


def _resolve_regular_command(command: str, *, label: str) -> Path:
    _located, resolved = _locate_regular_command(command, label=label)
    return resolved


def _validated_npm_cli(npm: str) -> Path:
    launcher, resolved = _locate_regular_command(npm, label="npm")
    candidates = (
        resolved,
        launcher.parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
        resolved.parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
        launcher.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js",
    )
    checked: set[Path] = set()
    for candidate in candidates:
        try:
            npm_cli = candidate.resolve(strict=True)
        except OSError:
            continue
        if npm_cli in checked:
            continue
        checked.add(npm_cli)
        if (
            npm_cli.name != "npm-cli.js"
            or npm_cli.parent.name != "bin"
            or npm_cli.parent.parent.name != "npm"
        ):
            continue
        try:
            cli_metadata = npm_cli.stat(follow_symlinks=False)
            package_path = (npm_cli.parent.parent / "package.json").resolve(strict=True)
            package_metadata = package_path.stat(follow_symlinks=False)
            package_bytes = package_path.read_bytes()
        except OSError:
            continue
        if (
            not stat.S_ISREG(cli_metadata.st_mode)
            or not stat.S_ISREG(package_metadata.st_mode)
            or package_path.parent != npm_cli.parent.parent
            or not 1 <= len(package_bytes) <= (1 << 20)
        ):
            continue
        try:
            package = json.loads(package_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            type(package) is dict
            and package.get("name") == "npm"
            and package.get("version") == EXPECTED_NPM_VERSION
        ):
            return npm_cli
    raise VerificationError("connector artifact smoke could not validate npm-cli.js")


def _canonical_archive_names(names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        raise VerificationError("distribution archive is empty")
    if len(names) != len(set(names)):
        raise VerificationError("distribution archive contains duplicate names")
    folded: set[str] = set()
    for name in names:
        if name != unicodedata.normalize("NFC", name):
            raise VerificationError("distribution archive contains a non-canonical name")
        if "\\" in name or "\0" in name or name.startswith("/"):
            raise VerificationError("distribution archive contains an unsafe name")
        path = PurePosixPath(name)
        if not path.parts or any(part in ("", ".", "..") for part in path.parts):
            raise VerificationError("distribution archive contains an unsafe name")
        if path.as_posix() != name:
            raise VerificationError("distribution archive contains a non-canonical name")
        canonical = name.casefold()
        if canonical in folded:
            raise VerificationError("distribution archive contains case-colliding names")
        folded.add(canonical)
    return names


def _distribution_pair(dist_dir: Path) -> tuple[Path, Path]:
    resolved = dist_dir.resolve(strict=True)
    children = tuple(resolved.iterdir())
    if len(children) != 2 or any(not child.is_file() or child.is_symlink() for child in children):
        raise VerificationError("dist must contain exactly one wheel and one sdist")
    wheels = tuple(child for child in children if child.suffix == ".whl")
    sdists = tuple(child for child in children if child.name.endswith(".tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise VerificationError("dist must contain exactly one wheel and one sdist")
    return wheels[0], sdists[0]


def _wheel_payloads(wheel: Path) -> dict[str, bytes]:
    required = {wheel_name for wheel_name, _sdist_name in BUNDLES.values()}
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        entries = tuple(archive.infolist())
        names = _canonical_archive_names(tuple(entry.filename for entry in entries))
        for entry, name in zip(entries, names, strict=True):
            if entry.is_dir() or entry.flag_bits & 0x1:
                raise VerificationError("wheel contains an unsupported archive entry")
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise VerificationError("wheel contains a non-regular archive entry")
            if name in required:
                payloads[name] = archive.read(entry)
    if payloads.keys() != required:
        raise VerificationError("wheel is missing an embedded connector bundle")
    return payloads


def _sdist_payloads(sdist: Path) -> dict[str, bytes]:
    required = {sdist_name for _wheel_name, sdist_name in BUNDLES.values()} | {NETWORK_GUARD}
    payloads: dict[str, bytes] = {}
    with tarfile.open(sdist, mode="r:gz") as archive:
        entries = tuple(archive.getmembers())
        names = _canonical_archive_names(tuple(entry.name for entry in entries))
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1:
            raise VerificationError("sdist must have one canonical archive root")
        root = roots.pop()
        normalized = tuple(name.removeprefix(f"{root}/") for name in names)
        _canonical_archive_names(normalized)
        for entry, name in zip(entries, normalized, strict=True):
            if not entry.isfile():
                raise VerificationError("sdist contains a non-regular archive entry")
            if name in required:
                extracted = archive.extractfile(entry)
                if extracted is None:
                    raise VerificationError("sdist connector payload could not be read")
                payloads[name] = extracted.read()
    if payloads.keys() != required:
        raise VerificationError("sdist is missing a connector proof resource")
    return payloads


def _write_private(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


@contextlib.contextmanager
def _workspace(requested: Path | None) -> Iterator[Path]:
    if requested is None:
        with tempfile.TemporaryDirectory(prefix="saliencegate-connector-artifacts-") as temporary:
            workspace = Path(temporary).resolve(strict=True)
            os.chmod(workspace, 0o700)
            yield workspace
        return

    if requested.exists() or requested.is_symlink():
        raise VerificationError("connector proof workspace must not already exist")
    requested.parent.resolve(strict=True)
    requested.mkdir(mode=0o700)
    workspace = requested.resolve(strict=True)
    os.chmod(workspace, 0o700)
    yield workspace


def _redacted(value: str) -> str:
    return value.replace(POISONED_CREDENTIAL, "<redacted>")


def _assert_poison_absent(workspace: Path) -> None:
    poisoned = POISONED_CREDENTIAL.encode("utf-8")
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise VerificationError("connector artifact smoke created a symbolic link")
        if path.is_file() and poisoned in path.read_bytes():
            raise VerificationError("connector artifact smoke persisted a poisoned credential")


def _project_environment_without_provider_values(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    ambient = os.environ if source is None else source
    if not isinstance(ambient, Mapping):
        raise VerificationError("connector artifact environment is invalid")
    environment: dict[str, str] = {}
    try:
        for key in ambient:
            if type(key) is not str:
                raise VerificationError("connector artifact environment is invalid")
            if key.upper() in PROVIDER_CREDENTIAL_KEY_SET or key.upper() == "NODE_OPTIONS":
                continue
            try:
                value = ambient[key]
            except KeyError:
                continue
            if type(value) is not str:
                raise VerificationError("connector artifact environment is invalid")
            environment[key] = value
    except VerificationError:
        raise
    except Exception:
        raise VerificationError("connector artifact environment is invalid") from None
    return environment


def _bootstrap_bytes(*, connector: str, bundle: bytes, launcher: Path) -> bytes:
    value = {
        "bundle_digest": hashlib.sha256(bundle).hexdigest(),
        "capability_digest": "1" * 64,
        "connection_id": f"sg-{'2' * 48}",
        "launcher_path": str(launcher),
        "profile": RUNTIME_PROFILES[connector],
        "receipt_mac": "3" * 64,
        "schema_version": "integration-bootstrap/v1",
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _materialize_launcher(workspace: Path, *, platform: str | None = None) -> Path:
    selected_platform = os.name if platform is None else platform
    if selected_platform not in {"nt", "posix"}:
        raise VerificationError("connector artifact launcher platform is unsupported")
    source = workspace / "proof" / "capture-launcher.mjs"
    _write_private(source, LAUNCHER_SMOKE_SOURCE.encode("utf-8"))
    launcher = (
        workspace
        / "proof"
        / ("capture-launcher.cmd" if selected_platform == "nt" else "capture-launcher")
    )
    launcher_source = (
        WINDOWS_LAUNCHER_SOURCE if selected_platform == "nt" else POSIX_LAUNCHER_SOURCE
    )
    _write_private(launcher, launcher_source, mode=0o700)
    return launcher


def _validate_launch_log(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise VerificationError("built connector runtime did not invoke the sanitized launcher")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise VerificationError("built connector runtime did not exercise both launch paths")
    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise VerificationError("connector launcher proof log is invalid") from error
        if not isinstance(record, dict):
            raise VerificationError("connector launcher proof log is invalid")
        if record.get("schema_version") != "connector-artifact-launch/v1":
            raise VerificationError("connector launcher proof schema is invalid")
        if record.get("network_denial") is not True:
            raise VerificationError("connector launcher did not inherit network denial")
        if record.get("provider_credential_keys_present") != []:
            raise VerificationError("connector launcher received a provider credential key")
        stdin_bytes = record.get("stdin_bytes")
        if not isinstance(stdin_bytes, int) or isinstance(stdin_bytes, bool) or stdin_bytes <= 0:
            raise VerificationError("connector launcher did not receive a capture payload")


def _run_node_smoke(
    *,
    node: str,
    npm: str,
    guard: Path,
    assets: dict[str, Path],
    launch_log: Path,
    workspace: Path,
) -> None:
    resolved_node = _resolve_regular_command(node, label="Node.js")
    npm_cli = _validated_npm_cli(npm)
    home = workspace / "home"
    for directory in (
        home,
        home / "config",
        home / "cache",
        home / "data",
        home / "state",
        home / "appdata",
        home / "localappdata",
        workspace / "tmp",
    ):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = _project_environment_without_provider_values()
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_STATE_HOME": str(home / "state"),
            "APPDATA": str(home / "appdata"),
            "LOCALAPPDATA": str(home / "localappdata"),
            "TEMP": str(workspace / "tmp"),
            "TMP": str(workspace / "tmp"),
            "TMPDIR": str(workspace / "tmp"),
            "SALIENCEGATE_OPENCODE_ASSET": assets["opencode"].as_uri(),
            "SALIENCEGATE_PI_ASSET": assets["pi"].as_uri(),
            "SALIENCEGATE_NETWORK_GUARD": guard.as_uri(),
            "SALIENCEGATE_ARTIFACT_LAUNCH_LOG": str(launch_log),
            "SALIENCEGATE_ARTIFACT_NODE": str(resolved_node),
            "SALIENCEGATE_ARTIFACT_LAUNCH_SOURCE": str(
                workspace / "proof" / "capture-launcher.mjs"
            ),
        }
    )
    for key in PROVIDER_CREDENTIAL_KEYS:
        environment[key] = POISONED_CREDENTIAL

    version = subprocess.run(
        (resolved_node, "--version"),
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if version.returncode != 0 or version.stdout.strip() != EXPECTED_NODE_VERSION:
        raise VerificationError("connector artifact smoke requires exact Node.js 22.19.0")
    npm_version = subprocess.run(
        (resolved_node, npm_cli, "--version"),
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if npm_version.returncode != 0 or npm_version.stdout.strip() != EXPECTED_NPM_VERSION:
        raise VerificationError("connector artifact smoke requires exact npm 10.9.3")

    completed = subprocess.run(
        (
            resolved_node,
            "--no-warnings",
            "--import",
            guard.as_uri(),
            "--input-type=module",
            "--eval",
            NODE_SMOKE_SOURCE,
        ),
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    if POISONED_CREDENTIAL in output:
        raise VerificationError("connector artifact smoke exposed a poisoned credential")
    if completed.returncode != 0:
        detail = _redacted(completed.stderr.strip() or completed.stdout.strip())
        raise VerificationError(f"connector artifact smoke failed: {detail}")
    if completed.stdout != "connector-artifacts-ok\n":
        raise VerificationError("connector artifact smoke returned unexpected output")
    _validate_launch_log(launch_log)
    _assert_poison_absent(workspace)


def verify(*, dist_dir: Path, work_dir: Path | None, node: str, npm: str) -> None:
    wheel, sdist = _distribution_pair(dist_dir)
    wheel_payloads = _wheel_payloads(wheel)
    sdist_payloads = _sdist_payloads(sdist)
    for wheel_name, sdist_name in BUNDLES.values():
        if wheel_payloads[wheel_name] != sdist_payloads[sdist_name]:
            raise VerificationError("wheel and sdist connector bundles are not byte-identical")

    with _workspace(work_dir) as workspace:
        guard = workspace / "proof" / "deny-network.mjs"
        _write_private(guard, sdist_payloads[NETWORK_GUARD])
        launcher = _materialize_launcher(workspace)
        launch_log = workspace / "proof" / "launches.ndjson"
        assets: dict[str, Path] = {}
        for connector, (wheel_name, _sdist_name) in BUNDLES.items():
            runtime = workspace / "proof" / connector
            runtime.mkdir(mode=0o700)
            _write_private(runtime / "package.json", b'{"type":"module"}')
            asset = runtime / RUNTIME_ASSET_NAMES[connector]
            bundle = wheel_payloads[wheel_name]
            _write_private(asset, bundle)
            _write_private(
                runtime / "saliencegate.bootstrap.json",
                _bootstrap_bytes(connector=connector, bundle=bundle, launcher=launcher),
            )
            assets[connector] = asset
        _run_node_smoke(
            node=node,
            npm=npm,
            guard=guard,
            assets=assets,
            launch_log=launch_log,
            workspace=workspace,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify connector bundles using only built wheel and sdist artifacts."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--node", default="node")
    parser.add_argument("--npm", default="npm")
    arguments = parser.parse_args()
    verify(
        dist_dir=arguments.dist_dir,
        work_dir=arguments.work_dir,
        node=arguments.node,
        npm=arguments.npm,
    )
    print("connector-artifact-proof-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
