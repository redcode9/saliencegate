from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Literal, TypeAlias, cast
from uuid import UUID

from pydantic import BaseModel, ValidationError

from saliencegate.domain import (
    EventType,
    LedgerRecord,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    Signal,
    TraceEvent,
    canonical_json,
    trace_event_payload_is_bounded,
)
from saliencegate.ports.repository import (
    AppendDisposition,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    LedgerEntry,
    LedgerHead,
    LedgerHeadConflictError,
    RecordCollisionError,
    RepositoryError,
    RunNotFoundError,
)
from saliencegate.repository import MemoryRunRepository, SQLiteRunRepository
from saliencegate.security.files import (
    SecureFileError,
    StableFileAuthorization,
    _claim_private_sqlite_location,
    authorize_private_sqlite_path,
    inspect_private_file_location,
)
from saliencegate.security.keys import (
    InstallationKey,
    generate_installation_key,
    load_or_create_installation_key,
)
from saliencegate.security.redaction import RedactionPolicy, Redactor
from saliencegate.shadow.config import (
    ShadowConfig,
    build_shadow_extractor,
    validate_shadow_config,
)
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
)
from saliencegate.shadow.evaluation import evaluate_shadow_heuristic
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowActionIdentityInput,
    ShadowActionInput,
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowInputRecord,
    ShadowObservationInput,
    ShadowObservationSource,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
    project_shadow_input,
)
from saliencegate.shadow.observation import (
    ShadowEventResult,
    _build_shadow_observation_from_selection,
    _select_detection_context,
    derive_shadow_feature_snapshot_digest,
    select_detection_context,
)
from saliencegate.shadow.trace import ShadowTraceBinding
from saliencegate.signals import (
    DetectionContext,
    DeterministicSignalExtractor,
    ExtractionReport,
    OpaqueActionEvidence,
    ShellActionEvidence,
    TestFailureEvidence,
    TestReportEvidence,
    ToolOutcomeEvidence,
)

_DEFAULT_SOURCE_ADAPTER = "saliencegate-shadow/v1"
_MAX_SHADOW_EVENTS = 10_000
_MAX_CAS_ATTEMPTS = 8
_REDACTION_POLICY_DOMAIN = b"saliencegate:shadow:redaction-policy:v1"
_MARKER_PREFLIGHT_EVENT_ID = UUID("00000000-0000-4000-8000-000000000001")
_COMPONENT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAPTURE_SCOPES = frozenset(
    {"unknown", "selected_events", "bounded_window", "complete_run_declared"}
)
_RESERVED_OBSERVATION_KEYS = frozenset(
    {
        "shadow_run",
        "shadow_run_end",
        "action",
        "action_identity",
        "tool_outcome",
        "test_report",
        "controller_error",
    }
)

CaptureScope: TypeAlias = Literal[
    "unknown",
    "selected_events",
    "bounded_window",
    "complete_run_declared",
]
_Repository: TypeAlias = MemoryRunRepository | SQLiteRunRepository


class _RetryableSnapshotRaceError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _SessionOptions:
    run_id: UUID
    config: ShadowConfig
    installation_key: InstallationKey
    redaction_policy: RedactionPolicy
    redaction_policy_tag: PayloadDigest
    capture_scope: CaptureScope
    task_scope_digest: str | None
    lineage_scope_digest: str | None
    capture_manifest_digest: str | None
    source_adapter: str


@dataclass(frozen=True, slots=True)
class _RunState:
    entries: tuple[LedgerEntry, ...]
    head: LedgerHead
    events: tuple[TraceEvent, ...]
    signals: tuple[Signal, ...]
    event_positions: Mapping[UUID, int]
    signal_positions: Mapping[UUID, int]
    start: TraceEvent
    finish: TraceEvent | None

    @property
    def events_by_source(self) -> Mapping[str, TraceEvent]:
        return {event.source_event_id: event for event in self.events}

    @property
    def events_by_id(self) -> Mapping[UUID, TraceEvent]:
        return {event.event_id: event for event in self.events}


_INPUT_TYPES = (
    ShadowStartInput,
    ShadowActionInput,
    ShadowActionIdentityInput,
    ShadowToolResultInput,
    ShadowTestResultInput,
    ShadowObservationInput,
    ShadowControllerErrorInput,
    ShadowFinishInput,
)


def _is_uuid4(value: object) -> bool:
    return type(value) is UUID and value.version == 4


def _copy_optional_digest(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError("digest is invalid")
    return value


def _copy_installation_key(value: object) -> InstallationKey:
    if type(value) is not InstallationKey:
        raise ValueError("installation key is invalid")
    return value._copy()


def _copy_redaction_policy(value: object) -> RedactionPolicy:
    if (
        type(value) is not RedactionPolicy
        or type(value.literal_secrets) is not tuple
        or type(value.structured_field_names) is not tuple
        or any(type(item) is not str for item in value.literal_secrets)
        or any(type(item) is not str for item in value.structured_field_names)
    ):
        raise ValueError("redaction policy is invalid")
    copied = RedactionPolicy(
        literal_secrets=value.literal_secrets,
        structured_field_names=value.structured_field_names,
    )
    if copied != value:
        raise ValueError("redaction policy is invalid")
    return copied


def _redaction_policy_tag(
    key: InstallationKey,
    policy: RedactionPolicy,
) -> PayloadDigest:
    configuration = canonical_json(
        {
            "literal_secrets": policy.literal_secrets,
            "structured_field_names": policy.structured_field_names,
        }
    )
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=key._hmac_sha256(configuration, domain=_REDACTION_POLICY_DOMAIN),
    )


def _start_marker_payload(options: _SessionOptions) -> dict[str, object]:
    return {
        "schema_version": "shadow-run/v1",
        "detector_profile_digest": options.config.detector_profile_digest,
        "evaluator_configuration_digest": options.config.evaluator_configuration_digest,
        "redaction_policy_tag": options.redaction_policy_tag.model_dump(mode="json"),
        "source_adapter": options.source_adapter,
        "capture_scope": options.capture_scope,
        "task_scope_digest": options.task_scope_digest,
        "lineage_scope_digest": options.lineage_scope_digest,
        "capture_manifest_digest": options.capture_manifest_digest,
        "split_metadata_complete": options.capture_manifest_digest is not None
        or (options.task_scope_digest is not None and options.lineage_scope_digest is not None),
    }


def _finish_marker_payload(
    options: _SessionOptions,
    start_event_id: UUID,
) -> dict[str, object]:
    payload = _start_marker_payload(options)
    payload["schema_version"] = "shadow-run-end/v1"
    payload["start_event_id"] = str(start_event_id)
    return payload


def _marker_is_redaction_identity(
    policy: RedactionPolicy,
    payload: Mapping[str, object],
) -> bool:
    try:
        redactor = Redactor(
            literal_secrets=policy.literal_secrets,
            structured_field_names=policy.structured_field_names,
        )
        redacted = redactor.redact_payload(payload)
        return canonical_json(redacted.payload.root) == canonical_json(payload)
    except Exception:
        return False


def _signal_probe(
    options: _SessionOptions,
    *,
    detector_index: int,
    created_at: datetime,
    evidence_event_id: UUID,
) -> Signal:
    detector = options.config.detectors[detector_index]
    return Signal(
        signal_id=UUID(int=_MARKER_PREFLIGHT_EVENT_ID.int + detector_index),
        run_id=options.run_id,
        created_at=created_at,
        signal_type=detector.signal_type,
        strength=1.0,
        evidence_event_ids=(evidence_event_id,),
        detector_version=detector.detector_version,
        reason_code=ReasonCode(detector.signal_type.value),
    )


def _literal_can_match_a_uuid(value: str) -> bool:
    without_controls = "".join(
        character for character in value if unicodedata.category(character) != "Cf"
    )
    normalized = unicodedata.normalize("NFKC", without_controls)
    return bool(normalized) and all(character in "0123456789abcdef-" for character in normalized)


def _require_static_policy_compatibility(options: _SessionOptions) -> None:
    if not _marker_is_redaction_identity(
        options.redaction_policy,
        _start_marker_payload(options),
    ) or not _marker_is_redaction_identity(
        options.redaction_policy,
        _finish_marker_payload(options, _MARKER_PREFLIGHT_EVENT_ID),
    ):
        raise ValueError("redaction policy conflicts with shadow markers")
    if any(_literal_can_match_a_uuid(value) for value in options.redaction_policy.literal_secrets):
        raise ValueError("redaction policy conflicts with generated identities")
    metadata = {
        "source_event_id": "shadow-preflight",
        "source_adapter": options.source_adapter,
    }
    if not _marker_is_redaction_identity(options.redaction_policy, metadata):
        raise ValueError("redaction policy conflicts with shadow metadata")
    for detector_index in range(len(options.config.detectors)):
        probe = _signal_probe(
            options,
            detector_index=detector_index,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
            evidence_event_id=_MARKER_PREFLIGHT_EVENT_ID,
        )
        if not _marker_is_redaction_identity(
            options.redaction_policy,
            probe.model_dump(mode="json", warnings=False),
        ):
            raise ValueError("redaction policy conflicts with shadow signals")


def _prepare_options(
    *,
    run_id: object,
    config: object,
    installation_key: object,
    redaction_policy: object,
    capture_scope: object,
    task_scope_digest: object,
    lineage_scope_digest: object,
    capture_manifest_digest: object,
    source_adapter: object,
) -> _SessionOptions:
    if not _is_uuid4(run_id):
        raise ValueError("run identity is invalid")
    assert isinstance(run_id, UUID)
    copied_config = ShadowConfig.reference() if config is None else validate_shadow_config(config)
    key = (
        load_or_create_installation_key()
        if installation_key is None
        else _copy_installation_key(installation_key)
    )
    policy = (
        RedactionPolicy() if redaction_policy is None else _copy_redaction_policy(redaction_policy)
    )
    if type(capture_scope) is not str or capture_scope not in _CAPTURE_SCOPES:
        raise ValueError("capture scope is invalid")
    if type(source_adapter) is not str or _COMPONENT_IDENTIFIER.fullmatch(source_adapter) is None:
        raise ValueError("source adapter is invalid")
    if source_adapter.casefold() == "saliencegate.repository":
        raise ValueError("source adapter is reserved")
    task_digest = _copy_optional_digest(task_scope_digest)
    lineage_digest = _copy_optional_digest(lineage_scope_digest)
    manifest_digest = _copy_optional_digest(capture_manifest_digest)
    options = _SessionOptions(
        run_id=UUID(int=run_id.int),
        config=copied_config,
        installation_key=key,
        redaction_policy=policy,
        redaction_policy_tag=_redaction_policy_tag(key, policy),
        capture_scope=cast(CaptureScope, capture_scope),
        task_scope_digest=task_digest,
        lineage_scope_digest=lineage_digest,
        capture_manifest_digest=manifest_digest,
        source_adapter=source_adapter,
    )
    _require_static_policy_compatibility(options)
    return options


def _copy_input(value: object) -> ShadowInputRecord:
    model_type = type(value)
    if model_type not in _INPUT_TYPES:
        raise ShadowInputError()
    try:
        serialized = model_type.__pydantic_serializer__.to_python(
            value,
            mode="python",
            warnings=False,
        )
        copied = model_type.model_validate(serialized)
        if copied != value:
            raise ValueError("shadow input copy differs")
    except Exception:
        copied = None
    if copied is None:
        raise ShadowInputError()
    return copied


class ShadowSession:
    """Owned, provider-free orchestration for one authenticated Shadow run."""

    __slots__ = (
        "_closed",
        "_config",
        "_extractor",
        "_installation_key",
        "_lazy_sqlite_location",
        "_lazy_sqlite_sidecars",
        "_lineage_scope_digest",
        "_lock",
        "_manifest_digest",
        "_options",
        "_redaction_policy",
        "_redaction_policy_tag",
        "_redactor",
        "_repository",
        "_run_id",
        "_source_adapter",
        "_task_scope_digest",
        "_trace_binding",
    )

    _closed: bool
    _config: ShadowConfig
    _extractor: DeterministicSignalExtractor
    _installation_key: InstallationKey
    _lineage_scope_digest: str | None
    _lock: asyncio.Lock
    _manifest_digest: str | None
    _options: _SessionOptions
    _redaction_policy: RedactionPolicy
    _redaction_policy_tag: PayloadDigest
    _redactor: Redactor
    _repository: _Repository | None
    _run_id: UUID
    _source_adapter: str
    _task_scope_digest: str | None
    _trace_binding: ShadowTraceBinding | None
    _lazy_sqlite_location: StableFileAuthorization | None
    _lazy_sqlite_sidecars: tuple[StableFileAuthorization, ...]

    def __init__(self) -> None:
        raise ShadowConfigurationError()

    @classmethod
    def _bind(
        cls,
        repository: _Repository | None,
        options: _SessionOptions,
        *,
        trace_binding: ShadowTraceBinding | None = None,
        lazy_sqlite_location: StableFileAuthorization | None = None,
        lazy_sqlite_sidecars: tuple[StableFileAuthorization, ...] = (),
    ) -> ShadowSession:
        self = object.__new__(cls)
        self._repository = repository
        self._options = options
        self._run_id = options.run_id
        self._config = options.config
        self._installation_key = options.installation_key
        self._redaction_policy = options.redaction_policy
        self._redaction_policy_tag = options.redaction_policy_tag
        self._redactor = Redactor(
            literal_secrets=options.redaction_policy.literal_secrets,
            structured_field_names=options.redaction_policy.structured_field_names,
        )
        self._source_adapter = options.source_adapter
        self._task_scope_digest = options.task_scope_digest
        self._lineage_scope_digest = options.lineage_scope_digest
        self._manifest_digest = options.capture_manifest_digest
        self._trace_binding = trace_binding
        self._lazy_sqlite_location = lazy_sqlite_location
        self._lazy_sqlite_sidecars = lazy_sqlite_sidecars
        self._extractor = build_shadow_extractor(options.config)
        self._lock = asyncio.Lock()
        self._closed = False
        return self

    @staticmethod
    def _copy_trace_binding(value: object) -> ShadowTraceBinding:
        if (
            type(value) is not ShadowTraceBinding
            or type(value.__dict__) is not dict
            or set(value.__dict__) != set(ShadowTraceBinding.model_fields)
            or value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
        ):
            raise ValueError("trace binding is invalid")
        serialized = ShadowTraceBinding.__pydantic_serializer__.to_json(
            value,
            warnings=False,
        )
        copied = ShadowTraceBinding.model_validate_json(serialized)
        copied_serialized = ShadowTraceBinding.__pydantic_serializer__.to_json(
            copied,
            warnings=False,
        )
        if copied != value or copied_serialized != serialized:
            raise ValueError("trace binding is invalid")
        return copied

    @classmethod
    def in_memory_for_trace(
        cls,
        *,
        run_id: UUID,
        trace_binding: ShadowTraceBinding,
        config: ShadowConfig | None = None,
        installation_key: InstallationKey | None = None,
        redaction_policy: RedactionPolicy | None = None,
    ) -> ShadowSession:
        """Create a trace-bound memory session without persistent key lookup."""

        result: ShadowSession | None = None
        try:
            binding = cls._copy_trace_binding(trace_binding)
            options = _prepare_options(
                run_id=run_id,
                config=config,
                installation_key=(
                    generate_installation_key() if installation_key is None else installation_key
                ),
                redaction_policy=redaction_policy,
                capture_scope=binding.capture_scope,
                task_scope_digest=binding.task_scope_digest,
                lineage_scope_digest=binding.lineage_scope_digest,
                capture_manifest_digest=binding.capture_manifest_digest,
                source_adapter=binding.source_adapter,
            )
            repository = MemoryRunRepository(
                installation_key=options.installation_key,
                redaction_policy=options.redaction_policy,
            )
            result = cls._bind(repository, options, trace_binding=binding)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        if result is None:
            raise ShadowConfigurationError()
        return result

    @classmethod
    def sqlite_for_trace(
        cls,
        path: str | Path,
        *,
        run_id: UUID,
        trace_binding: ShadowTraceBinding,
        installation_key: InstallationKey,
        config: ShadowConfig | None = None,
        redaction_policy: RedactionPolicy | None = None,
    ) -> ShadowSession:
        """Create a trace-bound SQLite session that opens storage on first analysis."""

        if type(installation_key) is not InstallationKey:
            raise ShadowConfigurationError()
        locations: tuple[StableFileAuthorization, ...] | None = None
        try:
            database = inspect_private_file_location(path)
            locations = (
                database,
                *(
                    inspect_private_file_location(f"{database.path}{suffix}")
                    for suffix in ("-wal", "-shm", "-journal")
                ),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        if locations is None:
            raise ShadowConfigurationError()
        return cls._from_sqlite_authorization_for_trace(
            locations[0],
            sidecar_authorizations=locations[1:],
            run_id=run_id,
            trace_binding=trace_binding,
            installation_key=installation_key,
            config=config,
            redaction_policy=redaction_policy,
        )

    @classmethod
    def _from_sqlite_authorization_for_trace(
        cls,
        authorization: StableFileAuthorization,
        *,
        sidecar_authorizations: tuple[StableFileAuthorization, ...] = (),
        run_id: UUID,
        trace_binding: ShadowTraceBinding,
        installation_key: InstallationKey,
        config: ShadowConfig | None = None,
        redaction_policy: RedactionPolicy | None = None,
    ) -> ShadowSession:
        """Bind the exact pre-inspected SQLite namespace without inspecting it again."""

        result: ShadowSession | None = None
        try:
            if (
                type(authorization) is not StableFileAuthorization
                or type(sidecar_authorizations) is not tuple
                or len(sidecar_authorizations) != 3
                or any(
                    type(sidecar) is not StableFileAuthorization
                    for sidecar in sidecar_authorizations
                )
                or type(installation_key) is not InstallationKey
            ):
                raise ValueError("trace SQLite configuration is invalid")
            binding = cls._copy_trace_binding(trace_binding)
            options = _prepare_options(
                run_id=run_id,
                config=config,
                installation_key=installation_key,
                redaction_policy=redaction_policy,
                capture_scope=binding.capture_scope,
                task_scope_digest=binding.task_scope_digest,
                lineage_scope_digest=binding.lineage_scope_digest,
                capture_manifest_digest=binding.capture_manifest_digest,
                source_adapter=binding.source_adapter,
            )
            result = cls._bind(
                None,
                options,
                trace_binding=binding,
                lazy_sqlite_location=authorization,
                lazy_sqlite_sidecars=sidecar_authorizations,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        if result is None:
            raise ShadowConfigurationError()
        return result

    @classmethod
    def in_memory(
        cls,
        *,
        run_id: UUID,
        config: ShadowConfig | None = None,
        installation_key: InstallationKey | None = None,
        redaction_policy: RedactionPolicy | None = None,
        capture_scope: CaptureScope = "unknown",
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
        source_adapter: str = _DEFAULT_SOURCE_ADAPTER,
    ) -> ShadowSession:
        result: ShadowSession | None = None
        failure: Exception | None = None
        try:
            options = _prepare_options(
                run_id=run_id,
                config=config,
                installation_key=installation_key,
                redaction_policy=redaction_policy,
                capture_scope=capture_scope,
                task_scope_digest=task_scope_digest,
                lineage_scope_digest=lineage_scope_digest,
                capture_manifest_digest=capture_manifest_digest,
                source_adapter=source_adapter,
            )
            repository = MemoryRunRepository(
                installation_key=options.installation_key,
                redaction_policy=options.redaction_policy,
            )
            result = cls._bind(repository, options)
        except Exception as error:
            failure = error
        if result is None:
            if isinstance(failure, ShadowStateError):
                raise ShadowStateError()
            raise ShadowConfigurationError()
        return result

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        run_id: UUID,
        config: ShadowConfig | None = None,
        installation_key: InstallationKey | None = None,
        redaction_policy: RedactionPolicy | None = None,
        capture_scope: CaptureScope = "unknown",
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
        source_adapter: str = _DEFAULT_SOURCE_ADAPTER,
    ) -> ShadowSession:
        options: _SessionOptions | None = None
        authorization: StableFileAuthorization | None = None
        try:
            options = _prepare_options(
                run_id=run_id,
                config=config,
                installation_key=installation_key,
                redaction_policy=redaction_policy,
                capture_scope=capture_scope,
                task_scope_digest=task_scope_digest,
                lineage_scope_digest=lineage_scope_digest,
                capture_manifest_digest=capture_manifest_digest,
                source_adapter=source_adapter,
            )
            authorization = authorize_private_sqlite_path(path)
        except Exception:
            pass
        if options is None or authorization is None:
            raise ShadowConfigurationError()
        return cls._bind_sqlite_authorization(authorization, options)

    @classmethod
    def _from_sqlite_authorization(
        cls,
        authorization: StableFileAuthorization,
        *,
        run_id: UUID,
        config: ShadowConfig | None = None,
        installation_key: InstallationKey | None = None,
        redaction_policy: RedactionPolicy | None = None,
        capture_scope: CaptureScope = "unknown",
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
        source_adapter: str = _DEFAULT_SOURCE_ADAPTER,
    ) -> ShadowSession:
        """Own an already authorized SQLite boundary without authorizing it again."""

        options: _SessionOptions | None = None
        try:
            if type(authorization) is not StableFileAuthorization:
                raise TypeError("authorization must be exactly StableFileAuthorization")
            options = _prepare_options(
                run_id=run_id,
                config=config,
                installation_key=installation_key,
                redaction_policy=redaction_policy,
                capture_scope=capture_scope,
                task_scope_digest=task_scope_digest,
                lineage_scope_digest=lineage_scope_digest,
                capture_manifest_digest=capture_manifest_digest,
                source_adapter=source_adapter,
            )
        except Exception:
            pass
        if options is None:
            raise ShadowConfigurationError()
        return cls._bind_sqlite_authorization(authorization, options)

    @classmethod
    def _bind_sqlite_authorization(
        cls,
        authorization: StableFileAuthorization,
        options: _SessionOptions,
    ) -> ShadowSession:
        result: ShadowSession | None = None
        repository: SQLiteRunRepository | None = None
        failure: Exception | None = None
        try:
            if type(authorization) is not StableFileAuthorization:
                raise TypeError("authorization must be exactly StableFileAuthorization")
            repository = SQLiteRunRepository._from_file_authorization(
                authorization,
                installation_key=options.installation_key,
                redaction_policy=options.redaction_policy,
            )
            result = cls._bind(repository, options)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            if repository is not None:
                with suppress(Exception):
                    repository.close()
            raise
        except Exception as error:
            failure = error
        if result is None:
            if repository is not None:
                with suppress(Exception):
                    repository.close()
            if isinstance(failure, RepositoryError) and not isinstance(failure, SecureFileError):
                raise ShadowStateError()
            raise ShadowConfigurationError()
        return result

    def __repr__(self) -> str:
        return f"ShadowSession(closed={self._closed})"

    async def __aenter__(self) -> ShadowSession:
        if self._closed:
            raise ShadowInputError()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            failed = False
            try:
                repository = self._repository
                if isinstance(repository, SQLiteRunRepository):
                    await repository.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                failed = True
            if failed:
                raise ShadowStateError()
            self._closed = True

    def _repository_for_operation(self) -> _Repository:
        """Return storage, materializing a trace SQLite boundary exactly once."""

        repository = self._repository
        if repository is not None:
            return repository
        if self._closed or self._trace_binding is None:
            raise ShadowStateError()
        location = self._lazy_sqlite_location
        if type(location) is not StableFileAuthorization:
            raise ShadowStateError()
        claimed: StableFileAuthorization | None = None
        claim_failed = False
        try:
            claimed = _claim_private_sqlite_location(
                location,
                sidecar_locations=self._lazy_sqlite_sidecars,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            claim_failed = True
        if claim_failed or claimed is None:
            raise ShadowConfigurationError()

        self._lazy_sqlite_location = None
        self._lazy_sqlite_sidecars = ()
        created: SQLiteRunRepository | None = None
        creation_failed = False
        unsafe_configuration = False
        try:
            created = SQLiteRunRepository._from_file_authorization(
                claimed,
                installation_key=self._installation_key,
                redaction_policy=self._redaction_policy,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            if created is not None:
                with suppress(Exception):
                    created.close()
            else:
                with suppress(Exception):
                    claimed._cleanup_created_sqlite_sidecars()
            raise
        except Exception as error:
            creation_failed = True
            unsafe_configuration = isinstance(error, SecureFileError)
            if created is not None:
                with suppress(Exception):
                    created.close()
            else:
                with suppress(Exception):
                    claimed._cleanup_created_sqlite_sidecars()
        if unsafe_configuration:
            raise ShadowConfigurationError()
        if creation_failed or created is None:
            raise ShadowStateError()
        self._repository = created
        return created

    async def _snapshot_for_trace(self) -> _RunState | None:
        """Return one authenticated trace state under the session ownership lock."""

        async with self._lock:
            if self._closed or self._trace_binding is None:
                raise ShadowInputError()
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                try:
                    return await self._load_state()
                except _RetryableSnapshotRaceError:
                    continue
        raise ShadowStateError()

    async def _append_trace_batch(
        self,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        """Append one whole-trace suffix while retaining session ownership."""

        async with self._lock:
            if self._closed or self._trace_binding is None:
                raise ShadowInputError()
            return await self._append_trace_batch_locked(
                operations,
                expected_head=expected_head,
            )

    async def _append_trace_batch_locked(
        self,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        repository = self._repository_for_operation()
        return await repository.append_records_if_head(
            operations,
            expected_head=expected_head,
        )

    async def _snapshot_for_cli(
        self,
    ) -> tuple[LedgerHead | None, tuple[TraceEvent, ...]]:
        """Return one authenticated initial head and its ordered event prefix."""

        async with self._lock:
            if self._closed:
                raise ShadowInputError()
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                try:
                    state = await self._load_state()
                except _RetryableSnapshotRaceError:
                    continue
                if state is None:
                    return None, ()
                return state.head, state.events
        raise ShadowStateError()

    def _start_payload(self) -> dict[str, object]:
        return _start_marker_payload(self._options)

    def _finish_payload(self, start: TraceEvent) -> dict[str, object]:
        return _finish_marker_payload(self._options, start.event_id)

    async def start(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowStartInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def action(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        working_directory: str,
        environment_digest: str,
        command: str | None = None,
        argv: tuple[str, ...] | None = None,
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowActionInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            working_directory=working_directory,
            environment_digest=environment_digest,
            command=command,
            argv=argv,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def action_identity(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        action_digest: str,
        workspace_digest: str,
        environment_digest: str,
        identity_authority: Literal["exact", "coarse", "unavailable"],
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowActionIdentityInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            action_digest=action_digest,
            workspace_digest=workspace_digest,
            environment_digest=environment_digest,
            identity_authority=identity_authority,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def tool_result(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        action: ShadowEventRef,
        status: Literal["succeeded", "failed"] | None = None,
        exit_status: int | None = None,
        exception_type: str | None = None,
        error_code: str | None = None,
        failure_signature: str | None = None,
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowToolResultInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            action=action,
            status=status,
            exit_status=exit_status,
            exception_type=exception_type,
            error_code=error_code,
            failure_signature=failure_signature,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def test_result(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        action: ShadowEventRef,
        framework: str,
        status: Literal["passed", "failed"],
        failures: tuple[TestFailureEvidence, ...],
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowTestResultInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            action=action,
            framework=framework,
            status=status,
            failures=failures,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def observation(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        source: ShadowObservationSource,
        payload: Mapping[str, object],
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowObservationInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            source=source,
            payload=payload,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def controller_error(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
        error_code: str,
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowControllerErrorInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
            error_code=error_code,
        )
        return await self._submit(value, cli_input_ordinal=None)

    async def finish(
        self,
        *,
        source_event_id: str,
        occurred_at: datetime,
    ) -> ShadowEventResult:
        value = self._public_input(
            ShadowFinishInput,
            source_event_id=source_event_id,
            occurred_at=occurred_at,
        )
        return await self._submit(value, cli_input_ordinal=None)

    @staticmethod
    def _public_input(model_type: type[BaseModel], **values: object) -> ShadowInputRecord:
        result: ShadowInputRecord | None = None
        try:
            candidate = model_type.model_validate(values)
            if type(candidate) in _INPUT_TYPES:
                result = cast(ShadowInputRecord, candidate)
        except Exception:
            result = None
        if result is None:
            raise ShadowInputError()
        return result

    async def _submit(
        self,
        input_record: ShadowInputRecord,
        *,
        cli_input_ordinal: int | None,
    ) -> ShadowEventResult:
        result: ShadowEventResult | None = None
        failure: Exception | None = None
        try:
            async with self._lock:
                if self._closed:
                    raise ShadowInputError()
                result = await self._submit_locked(
                    _copy_input(input_record),
                    cli_input_ordinal=cli_input_ordinal,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = error
        if result is not None:
            return result
        if isinstance(failure, ShadowInputError | RecordCollisionError | ValidationError):
            raise ShadowInputError()
        if isinstance(failure, ShadowConfigurationError):
            raise ShadowConfigurationError()
        if isinstance(failure, ShadowInvariantError):
            raise ShadowInvariantError()
        if isinstance(failure, ShadowStateError | RepositoryError | _RetryableSnapshotRaceError):
            raise ShadowStateError()
        raise ShadowInvariantError()

    async def _submit_locked(
        self,
        input_record: ShadowInputRecord,
        *,
        cli_input_ordinal: int | None,
    ) -> ShadowEventResult:
        if cli_input_ordinal is not None and (
            type(cli_input_ordinal) is not int or cli_input_ordinal < 1
        ):
            raise ShadowInputError()
        receipt = None
        candidate_event: TraceEvent | None = None
        attempt_state: _RunState | None = None
        submitted_kind = ShadowInputKind(input_record.kind)
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            try:
                state = await self._load_state()
            except _RetryableSnapshotRaceError:
                continue
            attempt_state = state
            self._authorize_input(input_record, submitted_kind, state)
            start_payload = (
                self._start_payload() if submitted_kind is ShadowInputKind.START else None
            )
            finish_payload = (
                self._finish_payload(state.start)
                if submitted_kind is ShadowInputKind.FINISH and state is not None
                else None
            )
            event_id = derive_shadow_event_id(self._run_id, input_record.source_event_id)
            if submitted_kind is ShadowInputKind.START:
                if (
                    start_payload is None
                    or not _marker_is_redaction_identity(
                        self._redaction_policy,
                        start_payload,
                    )
                    or not _marker_is_redaction_identity(
                        self._redaction_policy,
                        _finish_marker_payload(self._options, event_id),
                    )
                ):
                    raise ShadowConfigurationError()
            elif submitted_kind is ShadowInputKind.FINISH and (
                finish_payload is None
                or not _marker_is_redaction_identity(self._redaction_policy, finish_payload)
            ):
                raise ShadowConfigurationError()
            draft = project_shadow_input(
                input_record,
                run_id=self._run_id,
                source_adapter=self._source_adapter,
                start_payload=start_payload,
                finish_payload=finish_payload,
            )
            existing = (
                None if state is None else state.events_by_source.get(input_record.source_event_id)
            )
            sequence = (
                existing.sequence
                if existing is not None
                else 1
                if state is None
                else len(state.events) + 1
            )
            candidate_event = self._preflight_event(
                draft,
                kind=submitted_kind,
                event_id=event_id,
                sequence=sequence,
            )
            if existing is not None and candidate_event != existing:
                raise ShadowInputError()
            expected_head = None if state is None else state.head
            try:
                receipt = await self._repository_for_operation().append_event_if_head(
                    draft,
                    event_id=event_id,
                    expected_head=expected_head,
                )
            except LedgerHeadConflictError:
                continue
            break
        if receipt is None or candidate_event is None:
            raise ShadowStateError()
        if receipt.disposition not in (AppendDisposition.APPENDED, AppendDisposition.DUPLICATE):
            raise ShadowInputError()
        if receipt.event != candidate_event:
            raise ShadowStateError()

        state = (
            attempt_state
            if receipt.disposition is AppendDisposition.DUPLICATE and attempt_state is not None
            else await self._load_state_required()
        )
        prefix = self._prefix_for_receipt(state, receipt.event)
        selection = _select_detection_context(prefix)
        context = selection.context
        report = self._extract_report(context)
        for signal in report.signals:
            await self._persist_signal(signal, expected_prefix=prefix)

        final_state = await self._load_state_required() if report.signals else state
        final_prefix = self._prefix_for_receipt(final_state, receipt.event)
        if final_prefix != prefix:
            raise ShadowStateError()
        persisted = {signal.signal_id: signal for signal in final_state.signals}
        if any(persisted.get(signal.signal_id) != signal for signal in report.signals):
            raise ShadowStateError()
        feature_digest = derive_shadow_feature_snapshot_digest(
            prefix=prefix,
            context=context,
            report=report,
            config=self._config,
        )
        heuristic = evaluate_shadow_heuristic(
            report,
            input_kind=submitted_kind,
            config=self._config,
            feature_snapshot_digest=feature_digest,
        )
        observation = _build_shadow_observation_from_selection(
            selection=selection,
            report=report,
            config=self._config,
            input_kind=submitted_kind,
            heuristic=heuristic,
            source_event_digest=derive_shadow_source_event_digest(
                self._run_id,
                input_record.source_event_id,
            ),
            redaction_policy_tag=self._redaction_policy_tag,
            cli_input_ordinal=cli_input_ordinal,
        )
        result: ShadowEventResult | None = None
        try:
            result = ShadowEventResult(
                ref=ShadowEventRef(
                    run_id=self._run_id,
                    event_id=receipt.event.event_id,
                    sequence=receipt.event.sequence,
                ),
                observation=observation,
            )
        except Exception:
            result = None
        if result is None:
            raise ShadowInvariantError()
        return result

    def _preflight_event(
        self,
        draft: NormalizedTraceEventDraft,
        *,
        kind: ShadowInputKind,
        event_id: UUID,
        sequence: int,
    ) -> TraceEvent:
        candidate: TraceEvent | None = None
        try:
            metadata = {
                "source_event_id": draft.source_event_id,
                "source_adapter": draft.source_adapter,
            }
            redacted_metadata = self._redactor.redact_payload(metadata)
            if canonical_json(redacted_metadata.payload.root) != canonical_json(metadata):
                raise ValueError("shadow metadata is not redaction-stable")
            redacted = self._redactor.redact_event(
                draft,
                key=self._installation_key,
            )
            values = redacted.event.model_dump(mode="python", warnings=False)
            values.update(
                record_type="trace_event",
                event_id=event_id,
                sequence=sequence,
            )
            candidate = TraceEvent.model_validate(values)
        except Exception:
            candidate = None
        if candidate is None or not self._payload_is_valid(candidate, kind):
            raise ShadowInputError()

        applicability_kind = (
            ShadowInputKind.ACTION if kind is ShadowInputKind.ACTION_IDENTITY else kind
        )
        applicability = next(
            item for item in self._config.applicability if item.input_kind is applicability_kind
        )
        for detector_index, detector in enumerate(self._config.detectors):
            if detector.signal_type not in applicability.applicable_signal_types:
                continue
            probe = _signal_probe(
                self._options,
                detector_index=detector_index,
                created_at=candidate.timestamp,
                evidence_event_id=candidate.event_id,
            )
            if not _marker_is_redaction_identity(
                self._redaction_policy,
                probe.model_dump(mode="json", warnings=False),
            ):
                raise ShadowConfigurationError()
        return candidate

    def _authorize_input(
        self,
        input_record: ShadowInputRecord,
        kind: ShadowInputKind,
        state: _RunState | None,
    ) -> None:
        if state is None:
            if kind is not ShadowInputKind.START:
                raise ShadowInputError()
            return
        existing = state.events_by_source.get(input_record.source_event_id)
        if kind is ShadowInputKind.START and existing is None:
            raise ShadowInputError()
        if state.finish is not None and existing is None:
            raise ShadowInputError()
        if existing is None:
            if len(state.events) >= _MAX_SHADOW_EVENTS:
                raise ShadowInputError()
            if input_record.occurred_at < state.events[-1].timestamp:
                raise ShadowInputError()
        if type(input_record) in (ShadowToolResultInput, ShadowTestResultInput):
            assert isinstance(input_record, (ShadowToolResultInput, ShadowTestResultInput))
            parent = input_record.action
            known = state.events_by_id.get(parent.event_id)
            if (
                parent.run_id != self._run_id
                or known is None
                or known.event_type is not EventType.ACTION_PROPOSAL
                or known.sequence != parent.sequence
                or (existing is not None and known.sequence >= existing.sequence)
            ):
                raise ShadowInputError()

    async def _load_state_required(self) -> _RunState:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            try:
                state = await self._load_state()
            except _RetryableSnapshotRaceError:
                continue
            if state is None:
                raise ShadowStateError()
            return state
        raise ShadowStateError()

    async def _load_state(self) -> _RunState | None:
        try:
            repository = self._repository_for_operation()
            entries = await repository.ledger(self._run_id)
        except RunNotFoundError:
            return None
        head = await repository.ledger_head(self._run_id)
        if (
            type(entries) is not tuple
            or type(head) is not LedgerHead
            or len(entries) != head.entry_count
            or not entries
            or entries[-1].chain_tag != head.chain_tag
        ):
            raise _RetryableSnapshotRaceError()
        state: _RunState | None = None
        failed = False
        try:
            state = self._validate_run_state(entries, head)
        except ShadowInvariantError:
            raise
        except Exception:
            failed = True
        if failed or state is None:
            raise ShadowStateError()
        return state

    def _validate_run_state(
        self,
        entries: tuple[LedgerEntry, ...],
        head: LedgerHead,
    ) -> _RunState:
        if head.run_id != self._run_id or len(entries) != head.entry_count:
            raise ShadowStateError()
        events: list[TraceEvent] = []
        signals: list[Signal] = []
        event_positions: dict[UUID, int] = {}
        signal_positions: dict[UUID, int] = {}
        for expected_position, entry in enumerate(entries, start=1):
            if (
                type(entry) is not LedgerEntry
                or entry.run_id != self._run_id
                or entry.position != expected_position
            ):
                raise ShadowStateError()
            record: LedgerRecord = entry.record
            if type(record) is TraceEvent:
                events.append(record)
                event_positions[record.event_id] = entry.position
            elif type(record) is Signal:
                signals.append(record)
                signal_positions[record.signal_id] = entry.position
            else:
                raise ShadowStateError()
        if not events or type(entries[0].record) is not TraceEvent:
            raise ShadowStateError()
        if len(events) > _MAX_SHADOW_EVENTS:
            raise ShadowStateError()
        if len(event_positions) != len(events) or len(signal_positions) != len(signals):
            raise ShadowStateError()
        if any(event.sequence != sequence for sequence, event in enumerate(events, start=1)):
            raise ShadowStateError()
        if len({event.source_event_id for event in events}) != len(events):
            raise ShadowStateError()
        if any(
            event.run_id != self._run_id
            or event.source_adapter != self._source_adapter
            or event.event_id != derive_shadow_event_id(self._run_id, event.source_event_id)
            for event in events
        ):
            raise ShadowStateError()
        if any(right.timestamp < left.timestamp for left, right in pairwise(events)):
            raise ShadowStateError()
        kinds = tuple(self._event_kind(event) for event in events)
        if any(kind is None for kind in kinds) or kinds[0] is not ShadowInputKind.START:
            raise ShadowStateError()
        if sum(kind is ShadowInputKind.START for kind in kinds) != 1:
            raise ShadowStateError()
        finish_indexes = tuple(
            index for index, kind in enumerate(kinds) if kind is ShadowInputKind.FINISH
        )
        if len(finish_indexes) > 1 or (finish_indexes and finish_indexes[0] != len(events) - 1):
            raise ShadowStateError()
        start = events[0]
        if canonical_json(start.payload["shadow_run"]) != canonical_json(self._start_payload()):
            raise ShadowStateError()
        finish = events[-1] if finish_indexes else None
        if finish is not None and canonical_json(
            finish.payload["shadow_run_end"]
        ) != canonical_json(self._finish_payload(start)):
            raise ShadowStateError()
        by_id = {event.event_id: event for event in events}
        for event, kind in zip(events, kinds, strict=True):
            assert kind is not None
            if not self._payload_is_valid(event, kind):
                raise ShadowStateError()
            if kind in (ShadowInputKind.TOOL_RESULT, ShadowInputKind.TEST_RESULT):
                if len(event.parent_ids) != 1:
                    raise ShadowStateError()
                parent = by_id.get(event.parent_ids[0])
                if (
                    parent is None
                    or parent.event_type is not EventType.ACTION_PROPOSAL
                    or parent.sequence >= event.sequence
                ):
                    raise ShadowStateError()
            elif event.parent_ids:
                raise ShadowStateError()
        state = _RunState(
            entries=entries,
            head=head,
            events=tuple(events),
            signals=tuple(signals),
            event_positions=event_positions,
            signal_positions=signal_positions,
            start=start,
            finish=finish,
        )
        self._validate_existing_signals(state)
        return state

    def _event_kind(self, event: TraceEvent) -> ShadowInputKind | None:
        try:
            namespace = tuple(event.payload)
            for kind, spec in SHADOW_PROJECTION_MATRIX.items():
                if (
                    event.event_type is spec.event_type
                    and event.phase is spec.phase
                    and namespace == (spec.payload_namespace,)
                ):
                    if kind is ShadowInputKind.OBSERVATION:
                        if event.trust_label.value not in {
                            "untrusted_task_input",
                            "untrusted_tool_output",
                            "untrusted_model_output",
                            "untrusted_external_memory",
                        }:
                            return None
                    elif event.trust_label is not spec.trust_label:
                        return None
                    return kind
        except Exception:
            return None
        return None

    def _payload_is_valid(self, event: TraceEvent, kind: ShadowInputKind) -> bool:
        try:
            namespace = SHADOW_PROJECTION_MATRIX[kind].payload_namespace
            value = event.payload[namespace]
            if kind is ShadowInputKind.START:
                return canonical_json(value) == canonical_json(self._start_payload())
            if kind is ShadowInputKind.FINISH:
                return isinstance(value, Mapping)
            if kind is ShadowInputKind.ACTION:
                action_evidence = ShellActionEvidence.model_validate_json(canonical_json(value))
                return canonical_json(action_evidence) == canonical_json(value)
            if kind is ShadowInputKind.ACTION_IDENTITY:
                opaque_action_evidence = OpaqueActionEvidence.model_validate_json(
                    canonical_json(value)
                )
                return canonical_json(opaque_action_evidence) == canonical_json(value)
            if kind is ShadowInputKind.TOOL_RESULT:
                tool_evidence = ToolOutcomeEvidence.model_validate_json(canonical_json(value))
                return canonical_json(tool_evidence) == canonical_json(value)
            if kind is ShadowInputKind.TEST_RESULT:
                if not isinstance(value, Mapping) or set(value) != {
                    "schema_version",
                    "framework",
                    "status",
                    "failures",
                }:
                    return False
                raw_failures = value["failures"]
                if type(raw_failures) is not tuple:
                    return False
                failures = tuple(
                    TestFailureEvidence.model_validate_json(canonical_json(item))
                    for item in raw_failures
                )
                test_evidence = TestReportEvidence(
                    schema_version=cast(Literal["1.0"], value["schema_version"]),
                    framework=cast(str, value["framework"]),
                    status=cast(Literal["passed", "failed"], value["status"]),
                    failures=failures,
                )
                return canonical_json(test_evidence) == canonical_json(value)
            if kind is ShadowInputKind.OBSERVATION:
                return (
                    isinstance(value, Mapping)
                    and trace_event_payload_is_bounded(value)
                    and not any(key in _RESERVED_OBSERVATION_KEYS for key in value)
                )
            if kind is ShadowInputKind.CONTROLLER_ERROR:
                return (
                    isinstance(value, Mapping)
                    and set(value) == {"schema_version", "error_code"}
                    and value.get("schema_version") == "controller_error/v1"
                    and type(value.get("error_code")) is str
                    and _COMPONENT_IDENTIFIER.fullmatch(value["error_code"]) is not None
                )
        except Exception:
            return False
        return False

    def _validate_existing_signals(self, state: _RunState) -> None:
        by_id = state.events_by_id
        reports: dict[int, tuple[Signal, ...]] = {}
        for signal in state.signals:
            if signal.run_id != self._run_id or not signal.evidence_event_ids:
                raise ShadowStateError()
            evidence = tuple(by_id.get(event_id) for event_id in signal.evidence_event_ids)
            if any(event is None for event in evidence):
                raise ShadowStateError()
            current = by_id[signal.evidence_event_ids[-1]]
            if state.signal_positions[signal.signal_id] <= state.event_positions[current.event_id]:
                raise ShadowStateError()
            cutoff = current.sequence
            expected = reports.get(cutoff)
            if expected is None:
                prefix = state.events[:cutoff]
                context = select_detection_context(prefix)
                expected = self._extract_report(context).signals
                reports[cutoff] = expected
            matching = tuple(item for item in expected if item.signal_id == signal.signal_id)
            if len(matching) != 1 or matching[0] != signal:
                raise ShadowStateError()

    def _extract_report(self, context: DetectionContext) -> ExtractionReport:
        report: ExtractionReport | None = None
        try:
            report = self._extractor.extract_report(context)
        except asyncio.CancelledError:
            raise
        except Exception:
            report = None
        if report is None:
            raise ShadowInvariantError()
        return report

    @staticmethod
    def _prefix_for_receipt(state: _RunState, event: TraceEvent) -> tuple[TraceEvent, ...]:
        if (
            event.run_id != state.start.run_id
            or not 1 <= event.sequence <= len(state.events)
            or state.events[event.sequence - 1] != event
        ):
            raise ShadowStateError()
        return state.events[: event.sequence]

    async def _persist_signal(
        self,
        signal: Signal,
        *,
        expected_prefix: tuple[TraceEvent, ...],
    ) -> None:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            try:
                state = await self._load_state()
            except _RetryableSnapshotRaceError:
                continue
            if state is None or state.events[: len(expected_prefix)] != expected_prefix:
                raise ShadowStateError()
            context = select_detection_context(expected_prefix)
            report = self._extract_report(context)
            if signal not in report.signals:
                raise ShadowStateError()
            existing = {item.signal_id: item for item in state.signals}.get(signal.signal_id)
            if existing is not None:
                if existing != signal:
                    raise ShadowStateError()
                return
            try:
                await self._repository_for_operation().record_signal_if_head(
                    signal,
                    expected_head=state.head,
                )
            except LedgerHeadConflictError:
                continue
            return
        raise ShadowStateError()


__all__ = ["ShadowSession"]
