from __future__ import annotations

import argparse
import importlib
import math
import os
import platform
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Final, Never

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.sessions import verify_capture_session_snapshot
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import CaptureStore, CaptureStoreMode
from saliencegate.commands.capture.connect import run_connect
from saliencegate.domain import canonical_json
from saliencegate.integrations.codex import CODEX_HOST_VERSION, provider_installation_spec
from saliencegate.integrations.hook import (
    CaptureHookDependencies,
    CaptureHookSpool,
    CaptureHookStore,
    _default_dependencies,
    run_capture_hook,
)
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.registry import ProviderAlias
from saliencegate.security import InstallationKey, load_installation_key

if __package__:
    from scripts import benchmark_capture_hook as benchmark
else:  # pragma: no cover - direct CLI execution
    import benchmark_capture_hook as benchmark

REGISTERED_BENCHMARK_SCHEMA_VERSION: Final = "registered-capture-hook-benchmark/v1"
CONCURRENT_HOOK_INVOCATIONS: Final = 64
PROVIDER_TIMEOUT_BUDGET_MS: Final = 2_000.0
_POISON_VALUE: Final = "provider-credential-read-must-fail"
_PROVIDER_CREDENTIAL_NAMES: Final = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
)
_FORBIDDEN_RUNTIME_MODULE_ROOTS: Final = frozenset(
    {"anthropic", "httpx", "openai", "openai_harmony", "requests", "urllib3"}
)


class RegisteredCaptureHookBenchmarkError(RuntimeError):
    """A content-free failure of the registered capture benchmark fixture."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("registered capture hook benchmark failed")


def _deny_network(*_args: object, **_kwargs: object) -> Never:
    raise RegisteredCaptureHookBenchmarkError()


def _install_socket_denial() -> None:
    """Deny network constructors and resolvers before exercising capture code."""

    public = importlib.import_module("socket")
    private = importlib.import_module("_socket")
    low_level_socket = private.socket

    class NetworkBlockedSocket(public.socket):
        __slots__ = ()

        def __new__(
            cls,
            family: int = public.AF_INET,
            type: int = public.SOCK_STREAM,
            proto: int = 0,
            fileno: int | None = None,
        ):
            if family != public.AF_UNIX or fileno is None:
                raise RegisteredCaptureHookBenchmarkError()
            return super().__new__(cls, family, type, proto, fileno)

    class LowLevelNetworkBlockedSocket(low_level_socket):
        __slots__ = ()

        def __new__(
            cls,
            family: int = public.AF_INET,
            type: int = public.SOCK_STREAM,
            proto: int = 0,
            fileno: int | None = None,
        ):
            if family != public.AF_UNIX or fileno is None:
                raise RegisteredCaptureHookBenchmarkError()
            return low_level_socket.__new__(cls, family, type, proto, fileno)

        def __init__(
            self,
            family: int = public.AF_INET,
            type: int = public.SOCK_STREAM,
            proto: int = 0,
            fileno: int | None = None,
        ) -> None:
            low_level_socket.__init__(self, family, type, proto, fileno)

    for module in (public, private):
        for attribute in (
            "create_connection",
            "create_server",
            "getaddrinfo",
            "gethostbyaddr",
            "gethostbyname",
            "gethostbyname_ex",
            "getnameinfo",
        ):
            if hasattr(module, attribute):
                setattr(module, attribute, _deny_network)
    public.socket = NetworkBlockedSocket
    private.socket = LowLevelNetworkBlockedSocket
    public.SocketType = NetworkBlockedSocket
    private.SocketType = LowLevelNetworkBlockedSocket


def _network_denial_is_active() -> bool:
    module = importlib.import_module("socket")
    try:
        module.socket(module.AF_INET, module.SOCK_STREAM)
    except RegisteredCaptureHookBenchmarkError:
        pass
    else:
        return False
    try:
        module.getaddrinfo("capture-benchmark.invalid", 443)
    except RegisteredCaptureHookBenchmarkError:
        return True
    return False


def _capture_executable() -> Path:
    name = "saliencegate-capture-hook.exe" if os.name == "nt" else "saliencegate-capture-hook"
    try:
        executable = (Path(sys.executable).parent / name).resolve(strict=True)
        if not executable.is_file() or executable.is_symlink():
            raise OSError
        return executable
    except OSError:
        raise RegisteredCaptureHookBenchmarkError() from None


def _write_child_socket_denial(directory: Path) -> None:
    source = """\
import _socket
import socket

class CaptureBenchmarkNetworkError(RuntimeError):
    pass

def _deny(*_args, **_kwargs):
    raise CaptureBenchmarkNetworkError("network disabled")

_low_level_socket = _socket.socket

class _NetworkBlockedSocket(socket.socket):
    __slots__ = ()
    def __new__(cls, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        if family != socket.AF_UNIX or fileno is None:
            raise CaptureBenchmarkNetworkError("network disabled")
        return super().__new__(cls, family, type, proto, fileno)

class _LowLevelNetworkBlockedSocket(_low_level_socket):
    __slots__ = ()
    def __new__(cls, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        if family != socket.AF_UNIX or fileno is None:
            raise CaptureBenchmarkNetworkError("network disabled")
        return _low_level_socket.__new__(cls, family, type, proto, fileno)
    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        _low_level_socket.__init__(self, family, type, proto, fileno)

for _module in (socket, _socket):
    for _name in (
        "create_connection", "create_server", "getaddrinfo", "gethostbyaddr",
        "gethostbyname", "gethostbyname_ex", "getnameinfo",
    ):
        if hasattr(_module, _name):
            setattr(_module, _name, _deny)
socket.socket = _NetworkBlockedSocket
socket.SocketType = _NetworkBlockedSocket
_socket.socket = _LowLevelNetworkBlockedSocket
_socket.SocketType = _LowLevelNetworkBlockedSocket
"""
    path = directory / "sitecustomize.py"
    path.write_text(source, encoding="utf-8", errors="strict")
    if os.name != "nt":
        path.chmod(0o600)


def _environment_without_provider_credentials(source: Mapping[str, str]) -> dict[str, str]:
    """Copy ambient state without ever reading provider credential values."""

    if not isinstance(source, Mapping):
        raise RegisteredCaptureHookBenchmarkError()
    environment: dict[str, str] = {}
    try:
        for key in source:
            if type(key) is not str:
                raise RegisteredCaptureHookBenchmarkError()
            if key.upper() in _PROVIDER_CREDENTIAL_NAMES:
                continue
            value = source[key]
            if type(value) is not str:
                raise RegisteredCaptureHookBenchmarkError()
            environment[key] = value
    except RegisteredCaptureHookBenchmarkError:
        raise
    except Exception:
        raise RegisteredCaptureHookBenchmarkError() from None
    return environment


def _isolated_environment(root: Path) -> dict[str, str]:
    denial = root / "python-network-denial"
    denial.mkdir(mode=0o700)
    _write_child_socket_denial(denial)
    home = root / "home"
    home.mkdir(mode=0o700)
    environment = _environment_without_provider_credentials(os.environ)
    environment.update(
        {
            "HOME": str(home),
            "LOCALAPPDATA": str(root / "local-app-data"),
            "XDG_CONFIG_HOME": str(root / "configuration"),
            "XDG_STATE_HOME": str(root / "state"),
            **{name: _POISON_VALUE for name in _PROVIDER_CREDENTIAL_NAMES},
        }
    )
    # The subprocess boundary receives this owned dict, so any provider values it
    # materializes below are benchmark sentinels rather than ambient credentials.
    inherited_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(denial) + (
        "" if not inherited_python_path else os.pathsep + inherited_python_path
    )
    return environment


def _codex_payload(
    project: Path,
    *,
    event_name: str,
    session_native: str,
    ordinal: int,
) -> bytes:
    document: dict[str, object] = {
        "cwd": str(project),
        "hook_event_name": event_name,
        "model": "synthetic-no-model-runtime",
        "permission_mode": "default",
        "session_id": session_native,
        "transcript_path": str(project / "not-read.jsonl"),
    }
    if event_name == "SessionStart":
        document["source"] = "startup"
    elif event_name == "PreToolUse":
        document.update(
            {
                "tool_input": {"path": f"synthetic-{ordinal}.txt"},
                "tool_name": "Read",
                "tool_use_id": f"capture-benchmark-call-{ordinal}",
                "turn_id": f"capture-benchmark-turn-{ordinal}",
            }
        )
    else:
        raise RegisteredCaptureHookBenchmarkError()
    return canonical_json(document)


def _verify_snapshot(
    store: CaptureStore,
    *,
    connection_id: str,
    session_native: str,
    expected_events: int,
    context: CaptureDigestContext,
    installation_key: InstallationKey,
) -> None:
    session_id = context.session_id(session_native.encode("utf-8", errors="strict"))
    snapshot = verify_capture_session_snapshot(
        store.snapshot_session(connection_id, session_id),
        installation_key=installation_key,
    )
    if (
        snapshot.event_count != expected_events
        or len(snapshot.events) != expected_events
        or tuple(item.receipt_ordinal for item in snapshot.events)
        != tuple(range(1, expected_events + 1))
    ):
        raise RegisteredCaptureHookBenchmarkError()


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        raise RegisteredCaptureHookBenchmarkError()
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


class _NonClosingStoreProxy:
    """Share one real store across pressure calls; the harness owns its lifetime."""

    __slots__ = ("_store",)

    def __init__(self, store: CaptureStore) -> None:
        self._store = store

    def append(self, intake: object) -> object:
        return self._store.append(intake)  # type: ignore[arg-type]

    def close(self) -> None:
        return None


class _NonClosingSpoolProxy:
    """Use one real spool boundary while keeping cleanup outside worker threads."""

    __slots__ = ("_lock", "_spool")

    def __init__(self, spool: CaptureSpool) -> None:
        self._lock = threading.Lock()
        self._spool = spool

    def admit(self, store: object, intake: object) -> object:
        with self._lock:
            return self._spool.admit(store, intake)  # type: ignore[arg-type]

    def enqueue(self, intake: object) -> object:
        with self._lock:
            return self._spool.enqueue(intake)  # type: ignore[arg-type]

    def admit_transport(
        self,
        store: object,
        chunk: object,
        intakes: tuple[object, ...],
        fallback: tuple[object, ...],
    ) -> tuple[object, ...]:
        with self._lock:
            return self._spool.admit_transport(  # type: ignore[arg-type,return-value]
                store,
                chunk,
                intakes,
                fallback,
            )

    def close(self) -> None:
        return None


def _concurrency_measurements(
    *,
    seed_returncode: int,
    seed_duration_ms: float,
    samples: tuple[tuple[int, float], ...],
) -> dict[str, object]:
    if not samples:
        raise RegisteredCaptureHookBenchmarkError()
    durations = tuple(duration for _returncode, duration in samples)
    failed_invocations = sum(returncode != 0 for returncode, _duration in samples)
    maximum_duration_ms = max(durations)
    provider_timeout_passed = (
        seed_duration_ms <= PROVIDER_TIMEOUT_BUDGET_MS
        and maximum_duration_ms <= PROVIDER_TIMEOUT_BUDGET_MS
    )
    return {
        "seed_returncode": seed_returncode,
        "seed_duration_ms": seed_duration_ms,
        "p95_ms": _nearest_rank(durations, 0.95),
        "max_ms": maximum_duration_ms,
        "failed_invocations": failed_invocations,
        "provider_timeout_budget_ms": PROVIDER_TIMEOUT_BUDGET_MS,
        "provider_timeout_passed": provider_timeout_passed,
        "passed": (seed_returncode == 0 and failed_invocations == 0 and provider_timeout_passed),
    }


def _concurrency_report(
    project: Path,
    environment: dict[str, str],
    *,
    connection_id: str,
    capture_executable: Path,
    session_native: str,
) -> dict[str, object]:
    arguments = (
        "--profile",
        "codex-hooks/v1",
        "--connection",
        connection_id,
    )

    def resolve_dependencies(source: bytes) -> CaptureHookDependencies:
        dependencies = _default_dependencies(
            profile="codex-hooks/v1",
            connection_id=connection_id,
            source=source,
            environ=environment,
            capture_executable=capture_executable,
        )
        if type(dependencies) is not CaptureHookDependencies:
            raise RegisteredCaptureHookBenchmarkError()
        return dependencies

    def resolve_runtime(
        source: bytes,
    ) -> tuple[CaptureHookDependencies, object]:
        dependencies = resolve_dependencies(source)
        profile = CaptureProfile.CODEX_HOOKS_V1
        registry = dependencies.validate_registry(profile)
        receipt = dependencies.validate_receipt(profile, connection_id, registry)
        connection = dependencies.validate_connection(
            profile,
            connection_id,
            registry,
            receipt,
        )
        return dependencies, connection

    seed_payload = _codex_payload(
        project,
        event_name="SessionStart",
        session_native=session_native,
        ordinal=0,
    )
    seed_dependencies = resolve_dependencies(seed_payload)
    seed_started = time.perf_counter_ns()
    seed_returncode = run_capture_hook(
        arguments,
        BytesIO(seed_payload),
        dependencies=seed_dependencies,
    )
    seed_duration_ms = max((time.perf_counter_ns() - seed_started) / 1_000_000.0, 0.001)
    prepared = tuple(
        (
            _codex_payload(
                project,
                event_name="PreToolUse",
                session_native=session_native,
                ordinal=ordinal,
            ),
            ordinal,
        )
        for ordinal in range(1, CONCURRENT_HOOK_INVOCATIONS + 1)
    )
    resolved = tuple((source, *resolve_runtime(source), ordinal) for source, ordinal in prepared)
    first_dependencies = resolved[0][1]
    first_connection = resolved[0][2]
    shared_store = first_dependencies.open_store(first_connection)
    shared_spool = first_dependencies.open_spool(first_connection)
    if type(shared_store) is not CaptureStore or type(shared_spool) is not CaptureSpool:
        raise RegisteredCaptureHookBenchmarkError()
    store_proxy = _NonClosingStoreProxy(shared_store)
    spool_proxy = _NonClosingSpoolProxy(shared_spool)

    def bind_resources(
        dependencies: CaptureHookDependencies,
        connection: object,
    ) -> CaptureHookDependencies:
        if dependencies is not first_dependencies:
            unused_store = dependencies.open_store(connection)
            if type(unused_store) is not CaptureStore:
                raise RegisteredCaptureHookBenchmarkError()
            unused_store.close()

        def open_store(candidate: object) -> CaptureHookStore:
            if candidate is not connection:
                raise RegisteredCaptureHookBenchmarkError()
            return store_proxy

        def open_spool(candidate: object) -> CaptureHookSpool:
            if candidate is not connection:
                raise RegisteredCaptureHookBenchmarkError()
            return spool_proxy

        return CaptureHookDependencies(
            validate_registry=dependencies.validate_registry,
            validate_receipt=dependencies.validate_receipt,
            validate_connection=dependencies.validate_connection,
            load_context=dependencies.load_context,
            resolve_adapter=dependencies.resolve_adapter,
            open_store=open_store,
            open_spool=open_spool,
            mark_health=dependencies.mark_health,
        )

    invocations = tuple(
        (source, bind_resources(dependencies, connection), ordinal)
        for source, dependencies, connection, ordinal in resolved
    )
    barrier = threading.Barrier(CONCURRENT_HOOK_INVOCATIONS)

    def invoke(
        invocation: tuple[bytes, CaptureHookDependencies, int],
    ) -> tuple[int, float]:
        source, dependencies, _ordinal = invocation
        try:
            barrier.wait(timeout=30.0)
        except threading.BrokenBarrierError:
            raise RegisteredCaptureHookBenchmarkError() from None
        started = time.perf_counter_ns()
        returncode = run_capture_hook(
            arguments,
            BytesIO(source),
            dependencies=dependencies,
        )
        return returncode, max((time.perf_counter_ns() - started) / 1_000_000.0, 0.001)

    try:
        with ThreadPoolExecutor(
            max_workers=CONCURRENT_HOOK_INVOCATIONS,
            thread_name_prefix="capture-hook-pressure",
        ) as executor:
            samples = tuple(executor.map(invoke, invocations))
    finally:
        shared_store.close()
    return {
        "mode": "in_process_hook_invocations",
        "invocations": CONCURRENT_HOOK_INVOCATIONS,
        "concurrent_workers": CONCURRENT_HOOK_INVOCATIONS,
        "simultaneous_barrier_release": True,
        "dependency_resolution": "real_default_resolver_before_barrier",
        "resource_lifecycle": (
            "one_real_store_and_process_local_fenced_spool_shared_by_pressure_invocations"
        ),
        "spool_process_local_fence": True,
        "admission_path": "run_capture_hook",
        "distinct_payloads": CONCURRENT_HOOK_INVOCATIONS,
        "launcher_processes_started": 0,
        **_concurrency_measurements(
            seed_returncode=seed_returncode,
            seed_duration_ms=seed_duration_ms,
            samples=samples,
        ),
        "standard_output_contract": "no_output_api",
    }


def _assert_poison_absent(root: Path) -> None:
    poison = _POISON_VALUE.encode("utf-8")
    for path in root.rglob("*"):
        try:
            if (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= 16 * 1_024 * 1_024
                and poison in path.read_bytes()
            ):
                raise RegisteredCaptureHookBenchmarkError()
        except RegisteredCaptureHookBenchmarkError:
            raise
        except OSError:
            raise RegisteredCaptureHookBenchmarkError() from None


def run_registered_capture_hook_benchmark(root: Path) -> dict[str, object]:
    """Run the exact benchmark against an installed, authenticated Codex fixture."""

    if not isinstance(root, Path) or not root.is_absolute() or not root.is_dir():
        raise RegisteredCaptureHookBenchmarkError()
    environment = _isolated_environment(root)
    project = root / "project"
    project.mkdir(mode=0o700)
    _install_socket_denial()
    if not _network_denial_is_active():
        raise RegisteredCaptureHookBenchmarkError()

    fake_host_spec = provider_installation_spec(
        project,
        environ=environment,
        host_version=CODEX_HOST_VERSION,
    )

    def resolve_fake_host(alias: ProviderAlias, candidate: Path):
        if alias is not ProviderAlias.CODEX or candidate != project:
            raise RegisteredCaptureHookBenchmarkError()
        return fake_host_spec

    connected = run_connect(
        provider="codex",
        project=project,
        environ=environment,
        spec_resolver=resolve_fake_host,
        capture_executable=_capture_executable(),
    )
    if not connected.capture_enabled or not fake_host_spec.launcher_path.is_file():
        raise RegisteredCaptureHookBenchmarkError()
    key = load_installation_key(environ=environment)
    context = CaptureDigestContext(key)
    identity = derive_installation_identity(fake_host_spec, key)
    launcher = fake_host_spec.launcher_path.resolve(strict=True)
    cold_sessions = tuple(
        f"registered-cold-session-{ordinal}"
        for ordinal in range(1, benchmark.COLD_MEASUREMENTS + 1)
    )
    warm_session = "registered-warm-session"

    def run_sample(
        phase: benchmark.BenchmarkPhase,
        ordinal: int,
    ) -> benchmark.CaptureHookBenchmarkSample:
        if phase == "cold":
            payload = _codex_payload(
                project,
                event_name="SessionStart",
                session_native=cold_sessions[ordinal - 1],
                ordinal=ordinal,
            )
        else:
            payload = _codex_payload(
                project,
                event_name="SessionStart" if ordinal == 1 else "PreToolUse",
                session_native=warm_session,
                ordinal=ordinal,
            )
        return benchmark.invoke_launcher(launcher, payload, environment=environment)

    prepared: list[int] = []
    result = benchmark.run_benchmark(
        run_sample,
        metadata_path=launcher.parent,
        cold_preparer=prepared.append,
    )
    if prepared != list(range(1, benchmark.COLD_MEASUREMENTS + 1)):
        raise RegisteredCaptureHookBenchmarkError()

    concurrency_session = "registered-concurrency-session"
    capture_executable = _capture_executable()
    concurrency = _concurrency_report(
        project,
        environment,
        connection_id=identity.connection_id,
        capture_executable=capture_executable,
        session_native=concurrency_session,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    spool = CaptureSpool.open(locations, key)
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        busy_timeout_ms=60_000,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        drained = spool.drain(store)
        if drained.remaining_events != 0:
            raise RegisteredCaptureHookBenchmarkError()
        for session_native in cold_sessions:
            _verify_snapshot(
                store,
                connection_id=identity.connection_id,
                session_native=session_native,
                expected_events=1,
                context=context,
                installation_key=key,
            )
        _verify_snapshot(
            store,
            connection_id=identity.connection_id,
            session_native=warm_session,
            expected_events=benchmark.WARM_MEASUREMENTS,
            context=context,
            installation_key=key,
        )
        _verify_snapshot(
            store,
            connection_id=identity.connection_id,
            session_native=concurrency_session,
            expected_events=CONCURRENT_HOOK_INVOCATIONS + 1,
            context=context,
            installation_key=key,
        )
        concurrency["authenticated_event_count_verified"] = CONCURRENT_HOOK_INVOCATIONS + 1
        concurrency["capture_admission_verified"] = True
    health = spool.health()
    if health.queued_events != 0 or health.dropped_events != 0:
        raise RegisteredCaptureHookBenchmarkError()
    _assert_poison_absent(root)
    loaded_forbidden = sorted(
        root_name for root_name in _FORBIDDEN_RUNTIME_MODULE_ROOTS if root_name in sys.modules
    )
    if loaded_forbidden:
        raise RegisteredCaptureHookBenchmarkError()

    protocol = result.get("protocol")
    if type(protocol) is not dict:
        raise RegisteredCaptureHookBenchmarkError()
    protocol.update(
        {
            "workload": "registered_rendered_codex_launcher",
            "fake_host_fixture": "audited_codex_spec_resolver",
            "provider_host_processes_started": 0,
            "rendered_launcher_processes_started": (
                benchmark.COLD_MEASUREMENTS + benchmark.WARM_MEASUREMENTS
            ),
            "concurrency_workload": "in_process_hook_invocations",
            "fresh_launcher_process_per_sample": True,
            "cold_distinct_authenticated_sessions": benchmark.COLD_MEASUREMENTS,
            "cold_distinct_authenticated_session_state_verified": True,
            "os_page_cache_reset_claimed": False,
            "warm_authenticated_session_events": benchmark.WARM_MEASUREMENTS,
            "capture_admission_verified": True,
            "snapshot_authentication": "hmac_sha256_verified",
            "spool_state": "verified_clean_drained",
            "network_access": "socket_and_resolver_denied",
            "provider_credentials": "poisoned_and_absent_from_fixture_files",
            "pass_scope": "launcher_performance_and_authenticated_capture_admission",
        }
    )
    result["schema_version"] = REGISTERED_BENCHMARK_SCHEMA_VERSION
    result["concurrency"] = concurrency
    result["runtime"] = {
        "network_denial_verified": True,
        "provider_credential_names_poisoned": list(_PROVIDER_CREDENTIAL_NAMES),
        "forbidden_runtime_modules_loaded": loaded_forbidden,
        "system": platform.system() or "unknown",
    }
    result["artifacts"] = {
        "launcher": benchmark._artifact_evidence(launcher, maximum_bytes=1 * 1_024 * 1_024),
        "capture_executable": benchmark._artifact_evidence(
            _capture_executable(),
            maximum_bytes=16 * 1_024 * 1_024,
        ),
    }
    result["passed"] = result.get("passed") is True and concurrency["passed"] is True
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real local capture admission through a rendered Codex launcher.",
        allow_abbrev=False,
    )
    parser.add_argument("--assert-budgets", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        with tempfile.TemporaryDirectory(prefix="saliencegate-capture-hook-benchmark-") as raw:
            root = Path(raw).resolve(strict=True)
            report = run_registered_capture_hook_benchmark(root)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return 2
    sys.stdout.write(canonical_json(report).decode("utf-8") + "\n")
    return 0 if not parsed.assert_budgets or report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
