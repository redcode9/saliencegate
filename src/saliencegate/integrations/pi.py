"""Passive, content-free Pi extension capture adapter."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final

from saliencegate.capture.adapters import (
    CAPTURE_ADAPTER_PROTOCOL_VERSION,
    CaptureAdapterCapabilities,
)
from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.publication import authenticate_capture_intake
from saliencegate.capture.schema import (
    CAPTURE_NATIVE_JSON_LIMITS,
    CaptureIntake,
    read_bounded_json,
    validate_capture_intake,
)
from saliencegate.capture.transport import (
    MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION,
    CaptureTransportChunk,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import (
    IntegrationBootstrap,
    decode_integration_bootstrap,
)
from saliencegate.integrations.registry import (
    ProviderInstallationKind,
    ProviderInstallationSpec,
)

if TYPE_CHECKING:
    from saliencegate.integrations.hook import CaptureHookDependencies

PI_HOST_VERSION: Final = "0.80.10"
PI_PROFILE: Final = CaptureProfile.PI_EXTENSION_V1
PI_BOOTSTRAP_REFERENCE: Final = "./saliencegate.bootstrap.json"

_CONNECTION_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_SESSION_ID: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_DECIMAL_EVENT_ID: Final = re.compile(r"^[1-9][0-9]*$")
_ZERO_TAG: Final = "0" * 64
_MAX_REDUCED_EVENTS_PER_CHUNK: Final = 999
_MAX_TEXT_BYTES: Final = CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes
_MAX_SESSION_ID_BYTES: Final = 16 * 1_024
_MAX_REDUCED_EVENT_ID: Final = 997
_MAX_EVENT_ID_BYTES: Final = len(str(_MAX_REDUCED_EVENT_ID))
_MAX_CALL_ID_BYTES: Final = 16 * 1_024
_MAX_LINEAGE_ID_BYTES: Final = 16 * 1_024
_MAX_TOOL_NAME_BYTES: Final = 1 * 1_024
_HOOK_STORE_BUSY_TIMEOUT_MS: Final = 100


class PiIntegrationError(ValueError):
    """A Pi boundary failed without disclosing provider-owned values."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("Pi capture integration is invalid")


def _exact_text(value: object, *, maximum: int = _MAX_TEXT_BYTES) -> str | None:
    if type(value) is not str:
        return None
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError:
        return None
    return value if 1 <= size <= maximum else None


def _exact_keys(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PiIntegrationError()
    keys = frozenset(value)
    if not required <= keys or keys - required - optional:
        raise PiIntegrationError()
    if any(type(key) is not str for key in value):
        raise PiIntegrationError()
    return value


def _canonical_batch(source: bytes) -> Mapping[str, object]:
    try:
        return read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
    except PiIntegrationError:
        raise
    except Exception:
        raise PiIntegrationError() from None


def _bootstrap_from_document(value: object) -> IntegrationBootstrap:
    try:
        mapping = _exact_keys(
            value,
            required=frozenset(
                {
                    "schema_version",
                    "profile",
                    "connection_id",
                    "launcher_path",
                    "capability_digest",
                    "bundle_digest",
                    "receipt_mac",
                }
            ),
        )
        return decode_integration_bootstrap(canonical_json(mapping))
    except PiIntegrationError:
        raise
    except Exception:
        raise PiIntegrationError() from None


def _event_id(value: object) -> str | None:
    candidate = _exact_text(value, maximum=_MAX_EVENT_ID_BYTES)
    if (
        candidate is None
        or _DECIMAL_EVENT_ID.fullmatch(candidate) is None
        or int(candidate) > _MAX_REDUCED_EVENT_ID
    ):
        return None
    return candidate


def _event_order(value: str) -> tuple[int, str]:
    return (len(value), value)


@dataclass(frozen=True, slots=True)
class _PiBatch:
    source: bytes
    document: Mapping[str, object]
    bootstrap: IntegrationBootstrap
    batch_id: str
    session_id: str
    window_discriminator: str
    chunk_index: int
    chunk_count: int
    events: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _PiHookRuntime:
    key: object
    locations: object
    spec: ProviderInstallationSpec
    bootstrap: IntegrationBootstrap
    registration: object
    installation: object
    connection: object


def _parse_batch(source: bytes) -> _PiBatch:
    document = _canonical_batch(source)
    _exact_keys(
        document,
        required=frozenset(
            {
                "schema_version",
                "bootstrap",
                "batch_id",
                "session_id",
                "window_discriminator",
                "chunk_index",
                "chunk_count",
                "events",
            }
        ),
    )
    if document["schema_version"] != "capture-batch/v1":
        raise PiIntegrationError()
    batch_id = _exact_text(document["batch_id"], maximum=64)
    session_id = _exact_text(document["session_id"], maximum=_MAX_SESSION_ID_BYTES)
    window_discriminator = _exact_text(document["window_discriminator"], maximum=64)
    chunk_index = document["chunk_index"]
    chunk_count = document["chunk_count"]
    events = document["events"]
    if (
        batch_id is None
        or _SHA256.fullmatch(batch_id) is None
        or session_id is None
        or _NATIVE_SESSION_ID.fullmatch(session_id) is None
        or window_discriminator is None
        or _SHA256.fullmatch(window_discriminator) is None
        or type(chunk_index) is not int
        or type(chunk_count) is not int
        or not 0 <= chunk_index < chunk_count <= MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION
        or type(events) is not tuple
        or len(events) > _MAX_REDUCED_EVENTS_PER_CHUNK
    ):
        raise PiIntegrationError()
    return _PiBatch(
        source=source,
        document=document,
        bootstrap=_bootstrap_from_document(document["bootstrap"]),
        batch_id=batch_id,
        session_id=session_id,
        window_discriminator=window_discriminator,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        events=events,
    )


def _window_preimage(*, session_id: str, window_discriminator: str) -> bytes:
    return canonical_json(
        {
            "schema_version": "pi-observed-window/v1",
            "native_session_id": session_id,
            "window_discriminator": window_discriminator,
        }
    )


def _correlation_preimage(
    *,
    kind: str,
    session_id: str,
    window_discriminator: str,
    identifier: str | None = None,
    batch_id: str | None = None,
    chunk_index: int | None = None,
    event_index: int | None = None,
    old_leaf_id: str | None = None,
    new_leaf_id: str | None = None,
    compaction_reason: str | None = None,
    from_extension: bool | None = None,
    will_retry: bool | None = None,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": "pi-capture-correlation/v1",
        "kind": kind,
        "native_session_id": session_id,
        "window_discriminator": window_discriminator,
    }
    if identifier is not None:
        body["identifier"] = identifier
    if batch_id is not None:
        body["batch_id"] = batch_id
    if chunk_index is not None:
        body["chunk_index"] = chunk_index
    if event_index is not None:
        body["event_index"] = event_index
    if old_leaf_id is not None:
        body["old_leaf_id"] = old_leaf_id
    if new_leaf_id is not None:
        body["new_leaf_id"] = new_leaf_id
    if compaction_reason is not None:
        body["compaction_reason"] = compaction_reason
    if from_extension is not None:
        body["from_extension"] = from_extension
    if will_retry is not None:
        body["will_retry"] = will_retry
    return canonical_json(body)


def _tool_class(tool_name: str) -> str:
    normalized = tool_name.casefold()
    if normalized in {"bash", "shell", "terminal"}:
        return "shell"
    if normalized in {"apply_patch", "edit", "write", "multiedit"}:
        return "file_write"
    if normalized in {"read", "view_image"}:
        return "file_read"
    if normalized in {"grep", "glob", "search", "codesearch"}:
        return "search"
    if normalized in {"fetch", "webfetch", "websearch"}:
        return "network"
    if normalized in {"agent", "task", "subagent"}:
        return "subagent"
    return "other"


class PiCaptureAdapter:
    """Reduce one runtime-produced Pi window chunk into authenticated intake."""

    __slots__ = (
        "_bootstrap",
        "_capability_digest",
        "_connection_id",
        "_host_version",
        "_project_root",
    )

    def __init__(
        self,
        *,
        connection_id: str,
        bootstrap: IntegrationBootstrap,
        project_root: Path,
        host_version: str = PI_HOST_VERSION,
    ) -> None:
        try:
            if (
                type(connection_id) is not str
                or _CONNECTION_ID.fullmatch(connection_id) is None
                or type(bootstrap) is not IntegrationBootstrap
                or bootstrap.profile is not PI_PROFILE
                or bootstrap.connection_id != connection_id
                or not isinstance(project_root, Path)
                or not project_root.is_absolute()
                or ".." in project_root.parts
                or type(host_version) is not str
                or host_version != PI_HOST_VERSION
            ):
                raise PiIntegrationError()
            capability_digest = capture_capability_digest(capture_profile(PI_PROFILE))
            if bootstrap.capability_digest != capability_digest:
                raise PiIntegrationError()
            self._connection_id = connection_id
            self._bootstrap = bootstrap
            self._project_root = project_root
            self._host_version = host_version
            self._capability_digest = capability_digest
        except PiIntegrationError:
            raise
        except Exception:
            raise PiIntegrationError() from None

    def __repr__(self) -> str:
        return "PiCaptureAdapter(<redacted>)"

    __str__ = __repr__

    def capabilities(self) -> CaptureAdapterCapabilities:
        try:
            return CaptureAdapterCapabilities(
                protocol_version=CAPTURE_ADAPTER_PROTOCOL_VERSION,
                profile_id=PI_PROFILE,
                capability_digest=self._capability_digest,
                host_version=self._host_version,
            )
        except Exception:
            raise PiIntegrationError() from None

    @staticmethod
    def _session_digest(batch: _PiBatch, *, context: CaptureDigestContext) -> str:
        return context.session_id(
            _window_preimage(
                session_id=batch.session_id,
                window_discriminator=batch.window_discriminator,
            )
        )

    def _common(
        self,
        *,
        context: CaptureDigestContext,
        batch: _PiBatch,
        kind: str,
        event_index: int | None = None,
        event_id: str | None = None,
        identifier: str | None = None,
        disposition: str = "captured",
        session_stable: bool = False,
    ) -> dict[str, object]:
        correlation_kind = "event" if event_id is not None else kind
        correlation_identifier = event_id if event_id is not None else identifier
        return {
            "schema_version": "capture-intake/v1",
            "adapter_profile": PI_PROFILE.value,
            "capability_manifest_digest": self._capability_digest,
            "connection_id": self._connection_id,
            "session_id": self._session_digest(batch, context=context),
            "producer_event_digest": context.producer_event(
                _correlation_preimage(
                    kind=correlation_kind,
                    session_id=batch.session_id,
                    window_discriminator=batch.window_discriminator,
                    identifier=correlation_identifier,
                    batch_id=(None if event_id is not None or session_stable else batch.batch_id),
                    chunk_index=(
                        None if event_id is not None or session_stable else batch.chunk_index
                    ),
                    event_index=(None if event_id is not None or session_stable else event_index),
                )
            ),
            "intake_tag": _ZERO_TAG,
            "occurred_at": None,
            "timestamp_authority": "unavailable",
            "producer_sequence": None,
            "sequence_authority": "unavailable",
            "capture_disposition": disposition,
        }

    @staticmethod
    def _authenticated(
        values: Mapping[str, object],
        *,
        context: CaptureDigestContext,
    ) -> CaptureIntake:
        return authenticate_capture_intake(
            validate_capture_intake(dict(values)),
            context=context,
        )

    @staticmethod
    def _required_event_id(event: Mapping[str, object]) -> str:
        candidate = _event_id(event.get("event_id"))
        if candidate is None:
            raise PiIntegrationError()
        return candidate

    def _call_material(self, *, batch: _PiBatch, call_id: str) -> bytes:
        return _correlation_preimage(
            kind="tool_call",
            session_id=batch.session_id,
            window_discriminator=batch.window_discriminator,
            identifier=call_id,
        )

    def _event_intake(
        self,
        value: object,
        *,
        batch: _PiBatch,
        event_index: int,
        context: CaptureDigestContext,
    ) -> CaptureIntake:
        if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
            raise PiIntegrationError()
        event = value
        kind = _exact_text(event.get("kind"), maximum=64)
        if kind is None:
            raise PiIntegrationError()

        coordinates = frozenset({"kind", "session_id", "window_discriminator"})
        if kind == "oversize":
            _exact_keys(event, required=coordinates | {"reason"})
            if event["reason"] != "event_limit":
                raise PiIntegrationError()
            common = self._common(
                context=context,
                batch=batch,
                kind="oversize:event_limit",
                event_index=event_index,
                disposition="degraded",
            )
            common.update(kind="controller_failed", error_code="overflow", failure_signature=None)
            return self._authenticated(common, context=context)

        if kind == "tool_started":
            event = _exact_keys(
                event,
                required=coordinates | {"event_id", "call_id", "tool", "identity_authority"},
            )
            event_id = self._required_event_id(event)
            call_id = _exact_text(event["call_id"], maximum=_MAX_CALL_ID_BYTES)
            tool = _exact_text(event["tool"], maximum=_MAX_TOOL_NAME_BYTES)
            if call_id is None or tool is None or event["identity_authority"] != "coarse":
                raise PiIntegrationError()
            call_material = self._call_material(batch=batch, call_id=call_id)
            common = self._common(
                context=context,
                batch=batch,
                kind="tool_started",
                event_index=event_index,
                event_id=event_id,
            )
            common.update(
                kind="action_started",
                call_ref=context.call_ref(call_material),
                action_digest=context.action_identity(
                    canonical_json(
                        {
                            "schema_version": "pi-coarse-action-identity/v1",
                            "tool": tool,
                        }
                    )
                ),
                workspace_digest=context.workspace_identity(os.fsencode(self._project_root)),
                environment_digest=context.environment_identity(
                    canonical_json(
                        {
                            "schema_version": "pi-capture-environment/v1",
                            "profile": PI_PROFILE.value,
                            "host_version": self._host_version,
                        }
                    )
                ),
                tool_class=_tool_class(tool),
                identity_authority="coarse",
            )
            return self._authenticated(common, context=context)

        if kind == "tool_finished":
            event = _exact_keys(
                event,
                required=coordinates | {"event_id", "call_id", "outcome"},
            )
            event_id = self._required_event_id(event)
            call_id = _exact_text(event["call_id"], maximum=_MAX_CALL_ID_BYTES)
            outcome = event["outcome"]
            if call_id is None or outcome != "succeeded":
                raise PiIntegrationError()
            common = self._common(
                context=context,
                batch=batch,
                kind="tool_finished",
                event_index=event_index,
                event_id=event_id,
            )
            common.update(
                kind="action_finished",
                call_ref=context.call_ref(self._call_material(batch=batch, call_id=call_id)),
                outcome_status="succeeded",
                outcome_authority="producer_claimed_structured",
                exit_status=None,
                error_code=None,
                failure_signature=None,
            )
            return self._authenticated(common, context=context)

        if kind == "coverage_degraded":
            reason = event.get("reason")
            degraded_event_id: str | None
            if reason == "transport_gap":
                event = _exact_keys(event, required=coordinates | {"reason"})
                degraded_event_id = None
            else:
                event = _exact_keys(
                    event,
                    required=coordinates | {"event_id", "reason"},
                )
                degraded_event_id = self._required_event_id(event)
            if reason not in {
                "invalid_transition",
                "missing_field",
                "overflow",
                "transport_gap",
                "ambiguous_error",
                "unmatched_start",
            }:
                raise PiIntegrationError()
            common = self._common(
                context=context,
                batch=batch,
                kind=f"coverage_degraded:{reason}",
                event_index=event_index,
                event_id=degraded_event_id,
                identifier="transport_gap" if reason == "transport_gap" else None,
                disposition="degraded",
                session_stable=reason == "transport_gap",
            )
            common.update(
                kind="controller_failed",
                error_code=(
                    "overflow"
                    if reason == "overflow"
                    else "gap_detected"
                    if reason == "transport_gap"
                    else "invalid_transition"
                ),
                failure_signature=None,
            )
            return self._authenticated(common, context=context)

        if kind == "coverage_boundary":
            reason = event.get("reason")
            compaction_reason: str | None = None
            from_extension: bool | None = None
            will_retry: bool | None = None
            if reason == "compaction":
                event = _exact_keys(
                    event,
                    required=coordinates
                    | {
                        "event_id",
                        "reason",
                        "compaction_reason",
                        "from_extension",
                        "will_retry",
                    },
                )
                candidate_reason = event["compaction_reason"]
                candidate_from_extension = event["from_extension"]
                candidate_will_retry = event["will_retry"]
                if (
                    candidate_reason not in {"manual", "threshold", "overflow"}
                    or type(candidate_from_extension) is not bool
                    or type(candidate_will_retry) is not bool
                ):
                    raise PiIntegrationError()
                assert isinstance(candidate_reason, str)
                assert isinstance(candidate_from_extension, bool)
                assert isinstance(candidate_will_retry, bool)
                compaction_reason = candidate_reason
                from_extension = candidate_from_extension
                will_retry = candidate_will_retry
                old_leaf_id = new_leaf_id = None
            elif reason == "tree":
                event = _exact_keys(
                    event,
                    required=coordinates | {"event_id", "reason", "old_leaf_id", "new_leaf_id"},
                )
                old_leaf = event["old_leaf_id"]
                new_leaf = event["new_leaf_id"]
                old_leaf_id = (
                    None
                    if old_leaf is None
                    else _exact_text(old_leaf, maximum=_MAX_LINEAGE_ID_BYTES)
                )
                new_leaf_id = (
                    None
                    if new_leaf is None
                    else _exact_text(new_leaf, maximum=_MAX_LINEAGE_ID_BYTES)
                )
                if (old_leaf is not None and old_leaf_id is None) or (
                    new_leaf is not None and new_leaf_id is None
                ):
                    raise PiIntegrationError()
            else:
                raise PiIntegrationError()
            event_id = self._required_event_id(event)
            common = self._common(
                context=context,
                batch=batch,
                kind="coverage_boundary",
                event_index=event_index,
                event_id=event_id,
                disposition="coverage_boundary",
            )
            common.update(
                kind="turn_finished",
                turn_id=context.turn_id(
                    _correlation_preimage(
                        kind=f"coverage_boundary:{reason}",
                        session_id=batch.session_id,
                        window_discriminator=batch.window_discriminator,
                        identifier=event_id,
                        old_leaf_id=old_leaf_id,
                        new_leaf_id=new_leaf_id,
                        compaction_reason=compaction_reason,
                        from_extension=from_extension,
                        will_retry=will_retry,
                    )
                ),
            )
            return self._authenticated(common, context=context)

        event = _exact_keys(
            event,
            required=coordinates | {"event_id"},
            optional=frozenset({"reason"}) if kind == "session_finished" else frozenset(),
        )
        if kind not in {"turn_finished", "session_finished"}:
            raise PiIntegrationError()
        event_id = self._required_event_id(event)
        if kind == "session_finished" and event.get("reason") not in {
            "quit",
            "reload",
            "new",
            "resume",
            "fork",
        }:
            raise PiIntegrationError()
        common = self._common(
            context=context,
            batch=batch,
            kind=kind,
            event_index=event_index,
            event_id=event_id,
        )
        if kind == "turn_finished":
            common.update(
                kind="turn_finished",
                turn_id=context.turn_id(
                    _correlation_preimage(
                        kind="turn",
                        session_id=batch.session_id,
                        window_discriminator=batch.window_discriminator,
                        identifier=event_id,
                    )
                ),
            )
        else:
            common["kind"] = "session_finished"
        return self._authenticated(common, context=context)

    @staticmethod
    def _validate_event_coordinates(value: object, batch: _PiBatch) -> str | None:
        if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
            raise PiIntegrationError()
        if (
            _exact_text(value.get("session_id"), maximum=_MAX_SESSION_ID_BYTES) != batch.session_id
            or _exact_text(value.get("window_discriminator"), maximum=64)
            != batch.window_discriminator
        ):
            raise PiIntegrationError()
        kind = _exact_text(value.get("kind"), maximum=64)
        if kind is None:
            raise PiIntegrationError()
        event_id = value.get("event_id")
        if event_id is None:
            if kind == "oversize" or (
                kind == "coverage_degraded" and value.get("reason") == "transport_gap"
            ):
                return None
            raise PiIntegrationError()
        candidate = _event_id(event_id)
        if candidate is None:
            raise PiIntegrationError()
        return candidate

    def _validated_batch(self, source: bytes) -> _PiBatch:
        batch = _parse_batch(source)
        if batch.bootstrap != self._bootstrap:
            raise PiIntegrationError()
        prior: tuple[int, str] | None = None
        for index, event in enumerate(batch.events):
            candidate = self._validate_event_coordinates(event, batch)
            if (
                isinstance(event, Mapping)
                and event.get("kind") == "session_finished"
                and (index != len(batch.events) - 1 or batch.chunk_index != batch.chunk_count - 1)
            ):
                raise PiIntegrationError()
            if candidate is None:
                continue
            order = _event_order(candidate)
            if prior is not None and order <= prior:
                raise PiIntegrationError()
            prior = order
            if not isinstance(event, Mapping):
                raise PiIntegrationError()
            kind = event.get("kind")
            if kind == "tool_started":
                if index + 1 >= len(batch.events):
                    raise PiIntegrationError()
                finished = batch.events[index + 1]
                if (
                    not isinstance(finished, Mapping)
                    or finished.get("kind") != "tool_finished"
                    or _exact_text(event.get("call_id"), maximum=_MAX_CALL_ID_BYTES)
                    != _exact_text(finished.get("call_id"), maximum=_MAX_CALL_ID_BYTES)
                ):
                    raise PiIntegrationError()
                finished_event_id = _event_id(finished.get("event_id"))
                if (
                    candidate is None
                    or finished_event_id is None
                    or int(finished_event_id) != int(candidate) + 1
                ):
                    raise PiIntegrationError()
            elif kind == "tool_finished":
                if index == 0:
                    raise PiIntegrationError()
                started = batch.events[index - 1]
                if not isinstance(started, Mapping) or started.get("kind") != "tool_started":
                    raise PiIntegrationError()
        return batch

    def transport_chunk(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> CaptureTransportChunk:
        """Commit only receiver-keyed coordinates for one canonical Pi chunk."""

        try:
            if type(context) is not CaptureDigestContext:
                raise PiIntegrationError()
            batch = self._validated_batch(source)
            return CaptureTransportChunk(
                connection_id=self._connection_id,
                session_id=self._session_digest(batch, context=context),
                batch_ref=context.transport_batch_ref(batch.batch_id.encode("ascii")),
                chunk_index=batch.chunk_index,
                chunk_count=batch.chunk_count,
                chunk_digest=context.transport_chunk_digest(source),
            )
        except PiIntegrationError:
            raise
        except Exception:
            raise PiIntegrationError() from None

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[CaptureIntake, ...]:
        """Reduce one canonical bridge chunk; raw bytes never leave this call."""

        try:
            if type(context) is not CaptureDigestContext:
                raise PiIntegrationError()
            batch = self._validated_batch(source)
            started = self._common(
                context=context,
                batch=batch,
                kind="session_started",
                identifier="window",
                session_stable=True,
            )
            started["kind"] = "session_started"
            intakes = [self._authenticated(started, context=context)]
            intakes.extend(
                self._event_intake(
                    event,
                    batch=batch,
                    event_index=index,
                    context=context,
                )
                for index, event in enumerate(batch.events)
            )
            if len(intakes) > MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION:
                raise PiIntegrationError()
            return tuple(intakes)
        except PiIntegrationError:
            raise
        except Exception:
            raise PiIntegrationError() from None


def _bundle_bytes() -> bytes:
    try:
        data = (
            resources.files("saliencegate.integrations")
            .joinpath("assets")
            .joinpath("pi-extension.js")
            .read_bytes()
        )
        if not data:
            raise PiIntegrationError()
        return data
    except PiIntegrationError:
        raise
    except Exception:
        raise PiIntegrationError() from None


def provider_installation_spec(
    project: Path,
    *,
    environ: Mapping[str, str] | None = None,
    host_version: str = PI_HOST_VERSION,
) -> ProviderInstallationSpec:
    """Describe one configless project-local Pi extension installation."""

    try:
        if (
            not isinstance(project, Path)
            or not project.is_absolute()
            or ".." in project.parts
            or not project.is_dir()
            or project.is_symlink()
            or host_version != PI_HOST_VERSION
        ):
            raise PiIntegrationError()
        environment = os.environ if environ is None else environ
        if not isinstance(environment, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise PiIntegrationError()
        configured_home = environment.get("HOME")
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(environ=environment, home=home)
        project_locator = hashlib.sha256(
            canonical_json(
                {
                    "schema_version": "pi-installation-location/v1",
                    "project_root": os.fspath(project),
                }
            )
        ).hexdigest()
        operational = locations.state_directory / "integrations" / project_locator / "pi"
        extension_directory = project / ".pi" / "extensions"
        launcher = operational / ("capture-hook.cmd" if os.name == "nt" else "capture-hook")
        placeholder = (
            b"@exit /b 0\r\n"
            if os.name == "nt"  # pragma: no cover - exercised by native Windows R01
            else b"#!/bin/sh\nexit 0\n"
        )
        return ProviderInstallationSpec(
            installation_kind=ProviderInstallationKind.BRIDGE,
            provider_id="pi",
            profile=PI_PROFILE,
            host_version=host_version,
            project_root=project,
            config_path=None,
            config=None,
            bundle_path=extension_directory / "saliencegate.ts",
            bootstrap_path=extension_directory / "saliencegate.bootstrap.json",
            receipt_path=operational / "receipt.json",
            journal_path=operational / "journal.json",
            lock_path=operational / "install.lock",
            launcher_path=launcher,
            capability_digest=capture_capability_digest(capture_profile(PI_PROFILE)),
            bundle_bytes=_bundle_bytes(),
            launcher_bytes=placeholder,
            bootstrap_relative_reference=PI_BOOTSTRAP_REFERENCE,
            generation=1,
        )
    except PiIntegrationError:
        raise
    except Exception:
        raise PiIntegrationError() from None


def build_capture_hook_dependencies(
    source: bytes,
    *,
    connection_id: str,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureHookDependencies:
    """Authenticate an installed Pi runtime before admitting one batch."""

    try:
        from saliencegate.capture.connections import CaptureConnectionSummary
        from saliencegate.capture.health import CaptureHealthCode
        from saliencegate.capture.locations import CaptureStoreLocations
        from saliencegate.capture.spool import CaptureSpool
        from saliencegate.capture.store import (
            CaptureConnectionState,
            CaptureStore,
            CaptureStoreMode,
        )
        from saliencegate.commands.capture.connect import materialize_provider_launcher
        from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
        from saliencegate.integrations.hook import CaptureHookDependencies
        from saliencegate.integrations.installation import (
            InstallationReceipt,
            InstallationState,
            InstallationStatus,
            derive_installation_identity,
            inspect_installation_receipt,
            inspect_provider_installation,
        )
        from saliencegate.integrations.registry import (
            BUILTIN_PROVIDER_REGISTRY,
            ProviderAlias,
            ProviderRegistration,
        )
        from saliencegate.security import InstallationKey, load_installation_key

        if (
            type(source) is not bytes
            or type(connection_id) is not str
            or _CONNECTION_ID.fullmatch(connection_id) is None
            or (environ is not None and not isinstance(environ, Mapping))
        ):
            raise PiIntegrationError()
        environment = dict(os.environ if environ is None else environ)
        if any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise PiIntegrationError()
        batch = _parse_batch(source)
        if (
            batch.bootstrap.profile is not PI_PROFILE
            or batch.bootstrap.connection_id != connection_id
        ):
            raise PiIntegrationError()
        key = load_installation_key(environ=environment)
        receipt_path = batch.bootstrap.launcher_path.parent / "receipt.json"
        receipt = inspect_installation_receipt(receipt_path, key)
        bundle_path = receipt.bundle_path
        bootstrap_path = receipt.bootstrap_path
        if (
            receipt.state is not InstallationState.ENABLED
            or receipt.provider_id != "pi"
            or receipt.profile is not PI_PROFILE
            or receipt.host_version != PI_HOST_VERSION
            or receipt.connection_id != connection_id
            or receipt.launcher_path != batch.bootstrap.launcher_path
            or receipt.receipt_mac != batch.bootstrap.receipt_mac
            or receipt.capability_digest != batch.bootstrap.capability_digest
            or receipt.bundle_digest != batch.bootstrap.bundle_digest
            or bundle_path is None
            or bootstrap_path is None
            or bundle_path.name != "saliencegate.ts"
            or bundle_path.parent.name != "extensions"
            or bundle_path.parent.parent.name != ".pi"
        ):
            raise PiIntegrationError()
        project = bundle_path.parent.parent.parent
        if bootstrap_path != bundle_path.parent / "saliencegate.bootstrap.json":
            raise PiIntegrationError()
        spec = provider_installation_spec(
            project,
            environ=environment,
            host_version=receipt.host_version,
        )
        spec = materialize_provider_launcher(
            spec,
            key,
            capture_executable=capture_executable,
        )
        identity = derive_installation_identity(spec, key)
        if (
            spec.receipt_path != receipt_path
            or identity.connection_id != connection_id
            or identity.project_digest != receipt.project_digest
        ):
            raise PiIntegrationError()
        registration = BUILTIN_PROVIDER_REGISTRY.resolve(
            ProviderAlias.PI,
            require_available=True,
        )
        if registration.profile is not PI_PROFILE or registration.host_version != PI_HOST_VERSION:
            raise PiIntegrationError()
        configured_home = environment.get("HOME")
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(environ=environment, home=home)
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            busy_timeout_ms=_HOOK_STORE_BUSY_TIMEOUT_MS,
            mode=CaptureStoreMode.HOOK,
        ) as store:
            connection = store.get_connection(connection_id)
        installation = inspect_provider_installation(spec, key)
        installed_bootstrap = inspect_integration_bootstrap(bootstrap_path)
        if (
            installation.state is not InstallationState.ENABLED
            or not installation.installed
            or installation.drift
            or installation.connection_id != connection_id
            or installed_bootstrap != batch.bootstrap
            or connection.state is not CaptureConnectionState.ENABLED
            or connection.project_digest != identity.project_digest
            or connection.profile_id is not PI_PROFILE
            or connection.capability_manifest_digest != spec.capability_digest
            or connection.host_version != spec.host_version
        ):
            raise PiIntegrationError()
        runtime = _PiHookRuntime(
            key=key,
            locations=locations,
            spec=spec,
            bootstrap=installed_bootstrap,
            registration=registration,
            installation=installation,
            connection=connection,
        )

        def checked_runtime(value: object) -> _PiHookRuntime:
            if value is not runtime:
                raise PiIntegrationError()
            return runtime

        def validate_registry(profile: CaptureProfile) -> object:
            if profile is not PI_PROFILE:
                raise PiIntegrationError()
            return registration

        def validate_receipt(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
        ) -> object:
            if (
                profile is not PI_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
            ):
                raise PiIntegrationError()
            return installation

        def validate_connection(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
            candidate_receipt: object,
        ) -> object:
            if (
                profile is not PI_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
                or candidate_receipt is not installation
            ):
                raise PiIntegrationError()
            return runtime

        def load_context(candidate: object) -> CaptureDigestContext:
            selected = checked_runtime(candidate)
            if type(selected.key) is not InstallationKey:
                raise PiIntegrationError()
            return CaptureDigestContext(selected.key)

        def resolve_adapter(candidate: object) -> PiCaptureAdapter:
            selected = checked_runtime(candidate)
            if (
                type(selected.connection) is not CaptureConnectionSummary
                or type(selected.bootstrap) is not IntegrationBootstrap
            ):
                raise PiIntegrationError()
            return PiCaptureAdapter(
                connection_id=selected.connection.connection_id,
                bootstrap=selected.bootstrap,
                project_root=selected.spec.project_root,
                host_version=selected.connection.host_version,
            )

        def open_store(candidate: object) -> CaptureStore:
            selected = checked_runtime(candidate)
            if (
                type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
            ):
                raise PiIntegrationError()
            return CaptureStore.open(
                selected.locations.database_path,
                installation_key=selected.key,
                busy_timeout_ms=_HOOK_STORE_BUSY_TIMEOUT_MS,
                mode=CaptureStoreMode.HOOK,
            )

        def open_spool(candidate: object) -> CaptureSpool:
            selected = checked_runtime(candidate)
            if (
                type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
            ):
                raise PiIntegrationError()
            return CaptureSpool.open(selected.locations, selected.key)

        pseudonymous_session_id = CaptureDigestContext(key).session_id(
            _window_preimage(
                session_id=batch.session_id,
                window_discriminator=batch.window_discriminator,
            )
        )

        def mark_health(candidate: object, code: CaptureHealthCode) -> None:
            selected = checked_runtime(candidate)
            if (
                type(code) is not CaptureHealthCode
                or type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
                or type(selected.connection) is not CaptureConnectionSummary
            ):
                raise PiIntegrationError()
            with CaptureStore.open(
                selected.locations.database_path,
                installation_key=selected.key,
                busy_timeout_ms=_HOOK_STORE_BUSY_TIMEOUT_MS,
                mode=CaptureStoreMode.HOOK,
            ) as health_store:
                health_store.mark_session_health(
                    selected.connection.connection_id,
                    pseudonymous_session_id,
                    code,
                )

        if (
            type(registration) is not ProviderRegistration
            or type(receipt) is not InstallationReceipt
            or type(installation) is not InstallationStatus
            or type(connection) is not CaptureConnectionSummary
        ):
            raise PiIntegrationError()
        return CaptureHookDependencies(
            validate_registry=validate_registry,
            validate_receipt=validate_receipt,
            validate_connection=validate_connection,
            load_context=load_context,
            resolve_adapter=resolve_adapter,
            open_store=open_store,
            open_spool=open_spool,
            mark_health=mark_health,
        )
    except PiIntegrationError:
        raise
    except Exception:
        raise PiIntegrationError() from None


__all__ = [
    "PI_BOOTSTRAP_REFERENCE",
    "PI_HOST_VERSION",
    "PI_PROFILE",
    "PiCaptureAdapter",
    "PiIntegrationError",
    "build_capture_hook_dependencies",
    "provider_installation_spec",
]
