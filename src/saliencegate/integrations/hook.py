"""Silent, bounded provider-to-capture transport.

The provider-facing entrypoint deliberately owns no installation discovery.  Its
dependencies form the narrow seam through which the provider registry, signed
installation receipt, and enabled connection are validated before native bytes
can reach an adapter.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Final, Never, Protocol

if TYPE_CHECKING:
    from saliencegate.capture.capabilities import CaptureProfile
    from saliencegate.capture.health import CaptureHealthCode
    from saliencegate.capture.identities import CaptureDigestContext
    from saliencegate.capture.schema import CaptureIntake

_CONNECTION_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")
_CAPTURE_PROFILE_VALUES: Final = frozenset(
    (
        "codex-hooks/v1",
        "claude-code-hooks/v1",
        "opencode-plugin/v1",
        "pi-extension/v1",
    )
)
_CODEX_HOOK_EVENT_VALUES: Final = frozenset(
    (
        "SessionStart",
        "PreToolUse",
        "PermissionRequest",
        "PostToolUse",
        "PreCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    )
)
_CLAUDE_CODE_HOOK_EVENT_VALUES: Final = frozenset(
    (
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PostToolBatch",
        "PermissionDenied",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "StopFailure",
        "SessionEnd",
    )
)
_MAX_CAPTURE_NATIVE_BYTES: Final = 2 * 1_024 * 1_024
_MAX_CAPTURE_JSON_DEPTH: Final = 32
_MAX_CAPTURE_JSON_ITEMS: Final = 10_000
_MAX_CAPTURE_JSON_STRING_BYTES: Final = 1 * 1_024 * 1_024


class CaptureHookError(ValueError):
    """A content-free transport failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture hook failed")


@dataclass(frozen=True, slots=True)
class CaptureHookArguments:
    """The complete non-sensitive command-line surface."""

    profile: CaptureProfile
    connection_id: str


class CaptureHookStore(Protocol):
    """Minimum store surface used by the passive hook."""

    def append(self, intake: CaptureIntake) -> object: ...


class CaptureHookSpool(Protocol):
    """Minimum spool surface for ordered configured admission."""

    def admit(self, store: CaptureHookStore, intake: CaptureIntake) -> object: ...


RegistryValidator = Callable[["CaptureProfile"], object]
ReceiptValidator = Callable[["CaptureProfile", str, object], object]
ConnectionValidator = Callable[["CaptureProfile", str, object, object], object]
ContextLoader = Callable[[object], "CaptureDigestContext"]
AdapterResolver = Callable[[object], object]
StoreFactory = Callable[[object], CaptureHookStore]
SpoolFactory = Callable[[object], CaptureHookSpool]
HealthMarker = Callable[[object, "CaptureHealthCode"], None]


@dataclass(frozen=True, slots=True)
class CaptureHookDependencies:
    """Ordered validation and runtime dependencies supplied by integration code."""

    validate_registry: RegistryValidator
    validate_receipt: ReceiptValidator
    validate_connection: ConnectionValidator
    load_context: ContextLoader
    resolve_adapter: AdapterResolver
    open_store: StoreFactory
    open_spool: SpoolFactory
    mark_health: HealthMarker

    def __post_init__(self) -> None:
        if any(
            not callable(value)
            for value in (
                self.validate_registry,
                self.validate_receipt,
                self.validate_connection,
                self.load_context,
                self.resolve_adapter,
                self.open_store,
                self.open_spool,
                self.mark_health,
            )
        ):
            raise CaptureHookError()


def _unavailable_registry(_profile: CaptureProfile) -> Never:
    raise CaptureHookError()


def _unavailable_receipt(
    _profile: CaptureProfile,
    _connection_id: str,
    _registry: object,
) -> Never:
    raise CaptureHookError()


def _unavailable_connection(
    _profile: CaptureProfile,
    _connection_id: str,
    _registry: object,
    _receipt: object,
) -> Never:
    raise CaptureHookError()


def _unavailable_context(_connection: object) -> Never:
    raise CaptureHookError()


def _unavailable_adapter(_connection: object) -> Never:
    raise CaptureHookError()


def _unavailable_store(_connection: object) -> Never:
    raise CaptureHookError()


def _unavailable_spool(_connection: object) -> Never:
    raise CaptureHookError()


def _unavailable_health(_connection: object, _code: CaptureHealthCode) -> None:
    return None


# The common installation layer replaces these resolvers when its authenticated
# receipt API is wired.  Until then the installed hook is deliberately fail-open.
_UNAVAILABLE_CAPTURE_HOOK_DEPENDENCIES: Final = CaptureHookDependencies(
    validate_registry=_unavailable_registry,
    validate_receipt=_unavailable_receipt,
    validate_connection=_unavailable_connection,
    load_context=_unavailable_context,
    resolve_adapter=_unavailable_adapter,
    open_store=_unavailable_store,
    open_spool=_unavailable_spool,
    mark_health=_unavailable_health,
)
DEFAULT_CAPTURE_HOOK_DEPENDENCIES: Final = _UNAVAILABLE_CAPTURE_HOOK_DEPENDENCIES


def _default_dependencies(
    *,
    profile: str,
    connection_id: str,
    source: bytes,
    environ: Mapping[str, str] | None,
    capture_executable: str | os.PathLike[str] | None,
) -> CaptureHookDependencies | None:
    """Resolve an installed provider runtime only after a plausible native event."""

    try:
        document = json.loads(
            source.decode("utf-8", errors="strict"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        providers = {
            "codex-hooks/v1": (
                _CODEX_HOOK_EVENT_VALUES,
                "saliencegate.integrations.codex",
            ),
            "claude-code-hooks/v1": (
                _CLAUDE_CODE_HOOK_EVENT_VALUES,
                "saliencegate.integrations.claude_code",
            ),
        }
        selected = providers.get(profile)
        if type(document) is not dict or selected is None:
            return None
        event_values, module_name = selected
        event_name = document.get("hook_event_name")
        session_id = document.get("session_id")
        cwd = document.get("cwd")
        if (
            type(event_name) is not str
            or event_name not in event_values
            or type(session_id) is not str
            or not 1 <= len(session_id.encode("utf-8")) <= _MAX_CAPTURE_JSON_STRING_BYTES
            or type(cwd) is not str
            or not 1 <= len(cwd.encode("utf-8")) <= _MAX_CAPTURE_JSON_STRING_BYTES
            or not os.path.isabs(cwd)
            or "\x00" in cwd
        ):
            return None
        import importlib

        module = importlib.import_module(module_name)
        builder = getattr(module, "build_capture_hook_dependencies", None)
        if not callable(builder):
            return None
        result = builder(
            source,
            connection_id=connection_id,
            environ=environ,
            capture_executable=capture_executable,
        )
        return result if type(result) is CaptureHookDependencies else None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return None


def _parse_capture_hook_argument_values(arguments: Sequence[str]) -> tuple[str, str]:
    try:
        if isinstance(arguments, (str, bytes)) or len(arguments) != 4:
            raise CaptureHookError()
        values: dict[str, str] = {}
        for offset in (0, 2):
            name = arguments[offset]
            value = arguments[offset + 1]
            if (
                type(name) is not str
                or type(value) is not str
                or name not in ("--profile", "--connection")
                or name in values
            ):
                raise CaptureHookError()
            values[name] = value
        if set(values) != {"--profile", "--connection"}:
            raise CaptureHookError()
        profile = values["--profile"]
        if profile not in _CAPTURE_PROFILE_VALUES:
            raise CaptureHookError()
        connection_id = values["--connection"]
        if _CONNECTION_ID.fullmatch(connection_id) is None:
            raise CaptureHookError()
        return profile, connection_id
    except CaptureHookError:
        raise
    except Exception:
        raise CaptureHookError() from None


def parse_capture_hook_arguments(arguments: Sequence[str]) -> CaptureHookArguments:
    """Parse exactly one profile and one connection argument without diagnostics."""

    try:
        from saliencegate.capture.capabilities import CaptureProfile

        profile, connection_id = _parse_capture_hook_argument_values(arguments)
        return CaptureHookArguments(
            profile=CaptureProfile(profile),
            connection_id=connection_id,
        )
    except CaptureHookError:
        raise
    except Exception:
        raise CaptureHookError() from None


def _bounded_read(stream: BinaryIO) -> bytes:
    read = getattr(stream, "read", None)
    if not callable(read):
        raise CaptureHookError()
    remaining = _MAX_CAPTURE_NATIVE_BYTES + 1
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = read(remaining)
        if type(chunk) is not bytes or len(chunk) > remaining:
            raise CaptureHookError()
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    source = b"".join(chunks)
    if not 1 <= len(source) <= _MAX_CAPTURE_NATIVE_BYTES:
        raise CaptureHookError()
    return source


def _reject_json_constant(_value: str) -> Never:
    raise ValueError("capture JSON is invalid")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("capture JSON is invalid")
        result[key] = value
    return result


def _json_within_capture_bounds(value: object) -> bool:
    """Apply the native JSON depth, item, and aggregate string limits."""

    try:
        items = 0
        string_bytes = 0
        stack: list[tuple[object, int]] = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            items += 1
            if items > _MAX_CAPTURE_JSON_ITEMS or depth > _MAX_CAPTURE_JSON_DEPTH:
                return False
            if type(item) is dict:
                assert isinstance(item, dict)
                if len(item) > _MAX_CAPTURE_JSON_ITEMS - items - len(stack):
                    return False
                for key, nested in item.items():
                    if type(key) is not str:
                        return False
                    string_bytes += len(key.encode("utf-8", errors="strict"))
                    stack.append((nested, depth + 1))
            elif type(item) is list:
                assert isinstance(item, list)
                if len(item) > _MAX_CAPTURE_JSON_ITEMS - items - len(stack):
                    return False
                stack.extend((nested, depth + 1) for nested in item)
            elif type(item) is str:
                string_bytes += len(item.encode("utf-8", errors="strict"))
            elif item is None or type(item) in (bool, int):
                pass
            elif type(item) is float:
                if not math.isfinite(item):
                    return False
            else:
                return False
            if string_bytes > _MAX_CAPTURE_JSON_STRING_BYTES:
                return False
        return True
    except Exception:
        return False


def read_capture_hook_document(stream: BinaryIO) -> bytes:
    """Read one strict-UTF-8, duplicate-safe JSON object through EOF."""

    try:
        source = _bounded_read(stream)
        document = json.loads(
            source.decode("utf-8", errors="strict"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        if type(document) is not dict or not _json_within_capture_bounds(document):
            raise CaptureHookError()
        return source
    except CaptureHookError:
        raise
    except Exception:
        raise CaptureHookError() from None


def _require_evidence(value: object) -> object:
    if value is None:
        raise CaptureHookError()
    return value


def _adapter_intakes(
    adapter: object,
    source: bytes,
    *,
    profile: CaptureProfile,
    connection_id: str,
    context: CaptureDigestContext,
) -> tuple[CaptureIntake, ...]:
    from saliencegate.capture.adapters import validated_capture_adapter
    from saliencegate.capture.publication import verify_capture_intake_authentication

    declaration = validated_capture_adapter(adapter)
    if declaration.profile_id is not profile:
        raise CaptureHookError()
    adapt_bytes = getattr(adapter, "adapt_bytes", None)
    if not callable(adapt_bytes):
        raise CaptureHookError()
    result = adapt_bytes(source, context=context)
    if type(result) is not tuple or len(result) > _MAX_CAPTURE_JSON_ITEMS:
        raise CaptureHookError()
    checked: list[CaptureIntake] = []
    for candidate in result:
        intake = verify_capture_intake_authentication(candidate, context=context)
        if (
            intake.adapter_profile != profile.value
            or intake.connection_id != connection_id
            or intake.capability_manifest_digest != declaration.capability_digest
        ):
            raise CaptureHookError()
        checked.append(intake)
    return tuple(checked)


def _mark_health(
    dependencies: CaptureHookDependencies,
    connection: object | None,
    code: CaptureHealthCode,
) -> None:
    if connection is None:
        return
    try:
        dependencies.mark_health(connection, code)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return


def _mark_coverage_degraded(
    dependencies: CaptureHookDependencies,
    connection: object | None,
) -> None:
    if connection is None:
        return
    from saliencegate.capture.health import CaptureHealthCode

    _mark_health(dependencies, connection, CaptureHealthCode.COVERAGE_DEGRADED)


def _close_resource(value: object | None) -> bool:
    if value is None:
        return True
    try:
        close = getattr(value, "close", None)
        if callable(close):
            close()
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return False


def _spool_disposition(receipt: object) -> str | None:
    try:
        value = getattr(receipt, "disposition", None)
        return value if type(value) is str else None
    except Exception:
        return None


def run_capture_hook(
    arguments: Sequence[str],
    stream: BinaryIO,
    *,
    dependencies: CaptureHookDependencies | None = None,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | None = None,
) -> int:
    """Run one passive admission attempt and absorb all ordinary failures."""

    selected = DEFAULT_CAPTURE_HOOK_DEPENDENCIES if dependencies is None else dependencies
    connection: object | None = None
    store: CaptureHookStore | None = None
    spool: CaptureHookSpool | None = None
    try:
        if type(selected) is not CaptureHookDependencies:
            raise CaptureHookError()
        source: bytes | None = None
        if selected is _UNAVAILABLE_CAPTURE_HOOK_DEPENDENCIES:
            profile_value, connection_id = _parse_capture_hook_argument_values(arguments)
            source = read_capture_hook_document(stream)
            if environ is not None and not isinstance(environ, Mapping):
                return 0
            resolved = _default_dependencies(
                profile=profile_value,
                connection_id=connection_id,
                source=source,
                environ=environ,
                capture_executable=capture_executable,
            )
            if resolved is None:
                return 0
            selected = resolved

        from saliencegate.capture.health import CaptureHealthCode
        from saliencegate.capture.identities import CaptureDigestContext
        from saliencegate.capture.spool import CaptureSpoolError

        parsed = parse_capture_hook_arguments(arguments)
        if source is None:
            source = read_capture_hook_document(stream)
        registry = _require_evidence(selected.validate_registry(parsed.profile))
        receipt = _require_evidence(
            selected.validate_receipt(parsed.profile, parsed.connection_id, registry)
        )
        connection = _require_evidence(
            selected.validate_connection(
                parsed.profile,
                parsed.connection_id,
                registry,
                receipt,
            )
        )
        context = selected.load_context(connection)
        if type(context) is not CaptureDigestContext:
            raise CaptureHookError()
        adapter = selected.resolve_adapter(connection)
        intakes = _adapter_intakes(
            adapter,
            source,
            profile=parsed.profile,
            connection_id=parsed.connection_id,
            context=context,
        )
        store = selected.open_store(connection)
        if not callable(getattr(store, "append", None)):
            raise CaptureHookError()
        if intakes:
            try:
                spool = selected.open_spool(connection)
                if not callable(getattr(spool, "admit", None)):
                    raise CaptureHookError()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _mark_health(selected, connection, CaptureHealthCode.SPOOL_UNAVAILABLE)
                return 0
        for intake in intakes:
            try:
                if spool is None:
                    raise CaptureHookError()
                spool_receipt = spool.admit(store, intake)
            except (KeyboardInterrupt, SystemExit):
                raise
            except CaptureSpoolError:
                _mark_health(selected, connection, CaptureHealthCode.SPOOL_UNAVAILABLE)
                return 0
            if _spool_disposition(spool_receipt) == "dropped_quota":
                _mark_health(selected, connection, CaptureHealthCode.SPOOL_QUOTA)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _mark_coverage_degraded(selected, connection)
        return 0
    finally:
        spool_closed = _close_resource(spool)
        store_closed = _close_resource(store)
        if not spool_closed or not store_closed:
            _mark_coverage_degraded(selected, connection)


def _silence_standard_streams() -> bool:
    descriptor: int | None = None
    try:
        descriptor = os.open(os.devnull, os.O_WRONLY)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        return True
    except Exception:
        return False
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def entrypoint(arguments: Sequence[str] | None = None) -> int:
    """Provider-facing console entrypoint: silent and fail-open by construction."""

    if not _silence_standard_streams():
        return 0
    try:
        selected_arguments = sys.argv[1:] if arguments is None else arguments
        return run_capture_hook(
            selected_arguments,
            sys.stdin.buffer,
            capture_executable=sys.argv[0],
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return 0


__all__ = [
    "DEFAULT_CAPTURE_HOOK_DEPENDENCIES",
    "CaptureHookArguments",
    "CaptureHookDependencies",
    "CaptureHookError",
    "CaptureHookSpool",
    "CaptureHookStore",
    "entrypoint",
    "parse_capture_hook_arguments",
    "read_capture_hook_document",
    "run_capture_hook",
]
