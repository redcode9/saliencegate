from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from uuid import UUID

from saliencegate.adapters.generic import GenericHarnessAdapter
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN,
    OutcomeFreeTaskSkeleton,
    OutcomeFreeTraceFixture,
    PublicAssertionFixture,
    PublicBindingFixture,
    PublicConstraintReferenceFixture,
    PublicDetectorMemoryFixture,
    PublicExpectedAssertionEvidence,
    PublicExpectedDetectorEvidence,
    PublicExpectedMemoryEvidence,
    PublicExpectedSignal,
    PublicFixtureEvent,
    PublicImpactClass,
    PublicSignalFixtureVariant,
    PublicSlotProfile,
    public_assertion_fixture_digest,
    trace_fixture_digest,
)
from saliencegate.domain import (
    ClaimKind,
    DeliveryTarget,
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    SignalType,
    TraceEvent,
    TrustLabel,
    ValidityState,
)
from saliencegate.ports.adapters import DeliveryEnvelope
from saliencegate.repository import MemoryRunRepository
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionStatus,
    SignalDetector,
)
from saliencegate.signals.fingerprints import ShellActionEvidence
from saliencegate.signals.repetition import (
    RepeatedActionDetector,
    RepeatedFailureDetector,
    RepetitionConfig,
)
from saliencegate.signals.test_failures import TestFailureDetector
from saliencegate.signals.tool_errors import ToolErrorDetector

_LEGACY_REPETITION_WINDOW_EVENTS: Final = 8
_FIXTURE_ADAPTER_ID: Final = "state-decay-v2-public-fixture-adapter"
_REFERENCE_TYPES: Final = (
    SignalType.CONFLICT,
    SignalType.CONTEXT_SHIFT,
    SignalType.IRREVERSIBLE_ACTION,
    SignalType.STAGNATION,
    SignalType.STALE_CONSTRAINT,
)
_LEGACY_TYPES: Final = (
    SignalType.REPEATED_ACTION,
    SignalType.REPEATED_FAILURE,
    SignalType.TEST_FAILURE,
    SignalType.TOOL_ERROR,
)
_VARIANT_BY_SLOT = MappingProxyType(
    {
        0: PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT,
        1: PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE,
        2: PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS,
        3: PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT,
        4: PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION,
    }
)
_OPEN_MEMORY_KINDS: Final = frozenset((ClaimKind.REQUIREMENT, ClaimKind.OPEN_SUBGOAL))
_REFERENCE_RESULT_ISSUER = object()
_LEGACY_RESULT_ISSUER = object()


class SignalFixtureInputError(ValueError):
    """A value-free failure at the public signal-fixture boundary."""

    def __init__(self) -> None:
        super().__init__("public signal fixture input failed validation")


class ReferencePredicateStatus(StrEnum):
    DETECTED = "detected"
    NO_MATCH = "no_match"
    ABSTAINED = "abstained"


class ReferencePredicateAbstentionReason(StrEnum):
    REQUIRED_CAPABILITY_UNAVAILABLE = "required_capability_unavailable"
    OPERAND_UNRESOLVED = "operand_unresolved"
    OPERAND_INVALID = "operand_invalid"


@dataclass(frozen=True, slots=True)
class ReferencePredicateResult:
    """A non-runtime result produced by one frozen model-runtime reference predicate."""

    _issuer: InitVar[object]
    signal_type: SignalType
    status: ReferencePredicateStatus
    strength_ppm: int | None = None
    evidence: PublicExpectedDetectorEvidence | None = None
    abstention_reason: ReferencePredicateAbstentionReason | None = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _REFERENCE_RESULT_ISSUER:
            raise ValueError("reference result issuer is invalid")
        if type(self.signal_type) is not SignalType or self.signal_type not in _REFERENCE_TYPES:
            raise ValueError("reference result type is not reserved")
        if type(self.status) is not ReferencePredicateStatus:
            raise ValueError("reference result status is invalid")
        if self.evidence is not None and type(self.evidence) is not PublicExpectedDetectorEvidence:
            raise ValueError("reference result evidence is invalid")
        if (
            self.abstention_reason is not None
            and type(self.abstention_reason) is not ReferencePredicateAbstentionReason
        ):
            raise ValueError("reference result abstention reason is invalid")
        detected = self.status is ReferencePredicateStatus.DETECTED
        abstained = self.status is ReferencePredicateStatus.ABSTAINED
        if detected and (self.strength_ppm is None or self.evidence is None):
            raise ValueError("reference result detection fields are inconsistent")
        if not detected and (self.strength_ppm is not None or self.evidence is not None):
            raise ValueError("reference result detection fields are inconsistent")
        if detected and (
            type(self.strength_ppm) is not int or not 1 <= self.strength_ppm <= 1_000_000
        ):
            raise ValueError("reference result strength is invalid")
        if abstained != (self.abstention_reason is not None):
            raise ValueError("reference result abstention fields are inconsistent")


@dataclass(frozen=True, slots=True)
class LegacyDetectorResult:
    """The public event-index projection of one real legacy detector evaluation."""

    _issuer: InitVar[object]
    signal_type: SignalType
    detector_version: str
    status: DetectionStatus
    strength_ppm: int | None
    evidence_event_pool_indices: tuple[int, ...]
    related_event_pool_indices: tuple[int, ...]
    abstention_reason: AbstentionReason | None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _LEGACY_RESULT_ISSUER:
            raise ValueError("legacy detector result issuer is invalid")
        if type(self.signal_type) is not SignalType or self.signal_type not in _LEGACY_TYPES:
            raise ValueError("legacy detector result type is invalid")
        if (
            type(self.detector_version) is not str
            or not self.detector_version
            or type(self.status) is not DetectionStatus
        ):
            raise ValueError("legacy detector result metadata is invalid")
        for indices in (self.evidence_event_pool_indices, self.related_event_pool_indices):
            if (
                type(indices) is not tuple
                or len(set(indices)) != len(indices)
                or any(type(index) is not int or not 0 <= index <= 7 for index in indices)
            ):
                raise ValueError("legacy detector result indices are invalid")
        detected = self.status is DetectionStatus.DETECTED
        abstained = self.status is DetectionStatus.ABSTAINED
        if detected:
            if (
                type(self.strength_ppm) is not int
                or not 1 <= self.strength_ppm <= 1_000_000
                or not self.evidence_event_pool_indices
                or self.related_event_pool_indices
                or self.abstention_reason is not None
            ):
                raise ValueError("legacy detected result fields are inconsistent")
        elif (
            self.strength_ppm is not None
            or self.evidence_event_pool_indices
            or not self.related_event_pool_indices
            or (abstained != (self.abstention_reason is not None))
        ):
            raise ValueError("legacy non-detected result fields are inconsistent")
        if (
            self.abstention_reason is not None
            and type(self.abstention_reason) is not AbstentionReason
        ):
            raise ValueError("legacy detector abstention reason is invalid")


@dataclass(frozen=True, slots=True)
class LegacyFixtureEvaluation:
    """Repository-created events and the four final-boundary detector projections."""

    _issuer: InitVar[object]
    events: tuple[TraceEvent, ...]
    results: tuple[LegacyDetectorResult, ...]

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _LEGACY_RESULT_ISSUER:
            raise ValueError("legacy fixture evaluation issuer is invalid")
        if (
            type(self.events) is not tuple
            or not 3 <= len(self.events) <= 8
            or any(type(event) is not TraceEvent for event in self.events)
            or type(self.results) is not tuple
            or any(type(result) is not LegacyDetectorResult for result in self.results)
            or tuple(result.signal_type for result in self.results) != _LEGACY_TYPES
        ):
            raise ValueError("legacy fixture evaluation is invalid")
        if any(
            index >= len(self.events)
            for result in self.results
            for index in (
                *result.evidence_event_pool_indices,
                *result.related_event_pool_indices,
            )
        ):
            raise ValueError("legacy fixture evaluation evidence does not resolve")


@dataclass(frozen=True, slots=True)
class _NativeFixtureEvent:
    event: PublicFixtureEvent
    run_id: UUID
    event_id: UUID
    parent_ids: tuple[UUID, ...]
    timestamp: datetime


def _validated_materializer_inputs(
    slot_profile: PublicSlotProfile,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> tuple[PublicSlotProfile, OutcomeFreeTaskSkeleton]:
    try:
        if (
            type(slot_profile) is not PublicSlotProfile
            or type(task_skeleton) is not OutcomeFreeTaskSkeleton
        ):
            raise TypeError
        profile = PublicSlotProfile.model_validate_json(
            slot_profile.model_dump_json(warnings=False)
        )
        skeleton = OutcomeFreeTaskSkeleton.model_validate_json(
            task_skeleton.model_dump_json(warnings=False)
        )
        slot = profile.generator_slot
        if (
            profile.signals.fixture_variant is not _VARIANT_BY_SLOT[slot]
            or profile.structure.trajectory_event_count != slot + 3
            or len(skeleton.trajectory) != profile.structure.trajectory_event_count
            or len(skeleton.candidate_memories) != profile.structure.candidate_memory_count
        ):
            raise ValueError
        for index, memory in enumerate(skeleton.candidate_memories):
            if (
                memory.revision != profile.integers.memory_revision + index
                or memory.validity is not profile.counterbalance.memory_validity
            ):
                raise ValueError
    except Exception:
        raise SignalFixtureInputError() from None
    return profile, skeleton


def _action_payload(command: str) -> dict[str, object]:
    return {
        "action": {
            "schema_version": "1.0",
            "kind": "shell",
            "command": command,
            "working_directory": "/workspace",
            "environment_digest": "a" * 64,
        }
    }


def _test_failure_payload() -> dict[str, object]:
    return {
        "test_report": {
            "schema_version": "1.0",
            "framework": "pytest",
            "status": "failed",
            "failures": (
                {
                    "schema_version": "1.0",
                    "test_id": "tests/test_constraint.py::test_retained_constraint",
                    "failure_type": "AssertionError",
                    "signature": "retained-constraint-mismatch",
                },
            ),
        }
    }


def _tool_failure_payload() -> dict[str, object]:
    return {
        "tool_outcome": {
            "schema_version": "1.0",
            "status": "failed",
            "exit_status": 1,
            "failure_signature": "fixture-tool-failure",
        }
    }


def _event_shape(slot: int, index: int) -> tuple[EventType, EventPhase, dict[str, object]]:
    if index == 0:
        return EventType.RUN_START, EventPhase.INITIALIZATION, {}
    if slot == 0:
        if index == 1:
            return EventType.ACTION_PROPOSAL, EventPhase.PRE_ACTION, _action_payload("pytest -q")
        return EventType.TOOL_COMPLETION, EventPhase.POST_ACTION, _test_failure_payload()
    if slot == 1 and index in (2, 3):
        return EventType.ACTION_PROPOSAL, EventPhase.PRE_ACTION, _action_payload("pytest -q")
    if slot == 2 and 1 <= index <= 4:
        stagnation_commands = (
            "pytest -q",
            "python -m compileall src",
            "ruff check src",
            "mypy src",
        )
        return (
            EventType.ACTION_PROPOSAL,
            EventPhase.PRE_ACTION,
            _action_payload(stagnation_commands[index - 1]),
        )
    if slot == 3:
        if index in (2, 4):
            return EventType.ACTION_PROPOSAL, EventPhase.PRE_ACTION, _action_payload("pytest -q")
        if index in (3, 5):
            return EventType.TOOL_COMPLETION, EventPhase.POST_ACTION, _tool_failure_payload()
    if slot == 4 and 2 <= index <= 6:
        repeated_action_commands = (
            "pytest -q",
            "python -m compileall src",
            "ruff check src",
            "mypy src",
            "pytest -q",
        )
        return (
            EventType.ACTION_PROPOSAL,
            EventPhase.PRE_ACTION,
            _action_payload(repeated_action_commands[index - 2]),
        )
    return EventType.OBSERVATION, EventPhase.POST_ACTION, {}


def _progress_marker(slot: int, index: int) -> str:
    if (slot == 2 and 1 <= index <= 4) or (slot == 4 and 2 <= index <= 6):
        return "b" * 64
    return f"{index:x}" * 64


def _assertions_for(slot: int, index: int) -> tuple[PublicAssertionFixture, ...]:
    if (slot, index) not in ((0, 2), (2, 4)):
        return ()
    return (
        PublicAssertionFixture(
            subject_id="build",
            predicate_id="status",
            value_digest="c" * 64,
            precedence=1,
            revision=1,
            supersedes_assertion_digest=None,
        ),
        PublicAssertionFixture(
            subject_id="build",
            predicate_id="status",
            value_digest="d" * 64,
            precedence=1,
            revision=1,
            supersedes_assertion_digest=None,
        ),
    )


def _constraint_references_for(
    slot: int,
    index: int,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> tuple[PublicConstraintReferenceFixture, ...]:
    if (slot, index) == (0, 2):
        return (PublicConstraintReferenceFixture(memory_pool_index=1, revision=1),)
    if (slot, index) == (3, 5):
        return (
            PublicConstraintReferenceFixture(
                memory_pool_index=0,
                revision=task_skeleton.candidate_memories[0].revision,
            ),
        )
    return ()


def materialize_public_trace_fixture(
    slot_profile: PublicSlotProfile,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> OutcomeFreeTraceFixture:
    """Materialize one closed raw recipe without consulting its published signal vector."""

    profile, skeleton = _validated_materializer_inputs(slot_profile, task_skeleton)
    slot = profile.generator_slot
    final_index = len(skeleton.trajectory) - 1

    events: list[PublicFixtureEvent] = []
    bindings: list[PublicBindingFixture] = []
    for index, policy_event in enumerate(skeleton.trajectory):
        event_type, phase, payload = _event_shape(slot, index)
        events.append(
            PublicFixtureEvent(
                event_pool_index=index,
                event_type=event_type,
                phase=phase,
                payload=payload,
                parent_event_pool_indices=(() if index == 0 else (index - 1,)),
            )
        )
        if index == 0:
            bindings.append(
                PublicBindingFixture(
                    event_pool_index=0,
                    action_step=None,
                    scope_id=None,
                    progress_marker_digest=None,
                    constraint_references=None,
                    impact=None,
                    authorization_event_pool_indices=None,
                    safeguard_event_pool_indices=None,
                    assertions=None,
                )
            )
            continue
        scope_id = "scope-secondary" if (slot, index) in ((1, 3), (4, 6)) else "scope-primary"
        bindings.append(
            PublicBindingFixture(
                event_pool_index=index,
                action_step=policy_event.action_step,
                scope_id=scope_id,
                progress_marker_digest=_progress_marker(slot, index),
                constraint_references=_constraint_references_for(slot, index, skeleton),
                impact=(
                    PublicImpactClass.IRREVERSIBLE
                    if (slot, index) == (1, final_index)
                    else PublicImpactClass.REVERSIBLE
                ),
                authorization_event_pool_indices=(),
                safeguard_event_pool_indices=(),
                assertions=_assertions_for(slot, index),
            )
        )

    provenance_index = 2 if slot == 1 else 5 if slot == 4 else 0
    memories = tuple(
        PublicDetectorMemoryFixture(
            memory_pool_index=index,
            kind=(ClaimKind.REQUIREMENT if index == 0 else ClaimKind.ENVIRONMENT_FACT),
            current_revision=memory.revision,
            validity=memory.validity,
            provenance_event_pool_indices=((provenance_index,) if index == 0 else (0,)),
            expires_at_event_pool_index=None,
        )
        for index, memory in enumerate(skeleton.candidate_memories)
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-outcome-free-trace-fixture/v1",
        "events": tuple(events),
        "bindings": tuple(bindings),
        "memories": memories,
    }
    values["trace_fixture_digest"] = trace_fixture_digest(values)
    try:
        return OutcomeFreeTraceFixture.model_validate(values)
    except Exception:
        raise SignalFixtureInputError() from None


def _evidence(
    event_indices: tuple[int, ...],
    *,
    binding_indices: tuple[int, ...] = (),
    memory_references: tuple[tuple[int, int], ...] = (),
    assertion_references: tuple[tuple[int, int], ...] = (),
) -> PublicExpectedDetectorEvidence:
    return PublicExpectedDetectorEvidence(
        event_pool_indices=event_indices,
        binding_event_pool_indices=binding_indices,
        memory_references=tuple(
            PublicExpectedMemoryEvidence(memory_pool_index=index, revision=revision)
            for index, revision in memory_references
        ),
        assertion_references=tuple(
            PublicExpectedAssertionEvidence(
                binding_event_pool_index=event_index,
                assertion_index=assertion_index,
            )
            for event_index, assertion_index in assertion_references
        ),
    )


def _detected(
    signal_type: SignalType,
    strength_ppm: int,
    evidence: PublicExpectedDetectorEvidence,
) -> ReferencePredicateResult:
    return ReferencePredicateResult(
        _issuer=_REFERENCE_RESULT_ISSUER,
        signal_type=signal_type,
        status=ReferencePredicateStatus.DETECTED,
        strength_ppm=strength_ppm,
        evidence=evidence,
    )


def _no_match(signal_type: SignalType) -> ReferencePredicateResult:
    return ReferencePredicateResult(
        _issuer=_REFERENCE_RESULT_ISSUER,
        signal_type=signal_type,
        status=ReferencePredicateStatus.NO_MATCH,
    )


def _abstained(
    signal_type: SignalType,
    reason: ReferencePredicateAbstentionReason,
) -> ReferencePredicateResult:
    return ReferencePredicateResult(
        _issuer=_REFERENCE_RESULT_ISSUER,
        signal_type=signal_type,
        status=ReferencePredicateStatus.ABSTAINED,
        abstention_reason=reason,
    )


def _binding_resolves_to_skeleton(
    binding: PublicBindingFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> bool:
    return (
        binding.action_step is not None
        and binding.event_pool_index < len(task_skeleton.trajectory)
        and binding.action_step == task_skeleton.trajectory[binding.event_pool_index].action_step
    )


def _is_structured_action(event: PublicFixtureEvent) -> bool:
    if not _is_action_candidate(event):
        return False
    action_payload = event.payload.get("action")
    try:
        if not isinstance(action_payload, MappingProxyType):
            raise TypeError
        ShellActionEvidence.model_validate(dict(action_payload))
    except Exception:
        return False
    return True


def _is_action_candidate(event: PublicFixtureEvent) -> bool:
    return event.event_type is EventType.ACTION_PROPOSAL and event.phase is EventPhase.PRE_ACTION


def _evaluate_context_shift(
    fixture: OutcomeFreeTraceFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> ReferencePredicateResult:
    signal_type = SignalType.CONTEXT_SHIFT
    current = fixture.bindings[-1]
    if current.action_step is None or current.scope_id is None:
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    prior = next(
        (
            binding
            for binding in reversed(fixture.bindings[:-1])
            if binding.action_step is not None and binding.action_step != current.action_step
        ),
        None,
    )
    if prior is None or prior.scope_id is None:
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    if not _binding_resolves_to_skeleton(prior, task_skeleton) or not _binding_resolves_to_skeleton(
        current, task_skeleton
    ):
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    if prior.scope_id == current.scope_id:
        return _no_match(signal_type)

    qualifying: list[PublicDetectorMemoryFixture] = []
    for memory in fixture.memories:
        if memory.validity is not ValidityState.ACTIVE or memory.kind not in _OPEN_MEMORY_KINDS:
            continue
        provenance_scopes = tuple(
            fixture.bindings[index].scope_id for index in memory.provenance_event_pool_indices
        )
        if any(scope is None for scope in provenance_scopes):
            return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
        if prior.scope_id in provenance_scopes:
            qualifying.append(memory)
    if not qualifying:
        return _no_match(signal_type)
    strength = min(1_000_000, 500_000 + 100_000 * (len(qualifying) - 1))
    event_indices = (prior.event_pool_index, current.event_pool_index)
    return _detected(
        signal_type,
        strength,
        _evidence(
            event_indices,
            binding_indices=event_indices,
            memory_references=tuple(
                (memory.memory_pool_index, memory.current_revision) for memory in qualifying
            ),
        ),
    )


def _evaluate_stale_constraint(
    fixture: OutcomeFreeTraceFixture,
) -> ReferencePredicateResult:
    signal_type = SignalType.STALE_CONSTRAINT
    current = fixture.bindings[-1]
    references = current.constraint_references
    if references is None:
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    if not references:
        return _no_match(signal_type)
    memories = {memory.memory_pool_index: memory for memory in fixture.memories}
    matches: list[tuple[int, int, int]] = []
    for reference in references:
        memory = memories.get(reference.memory_pool_index)
        if (
            memory is None
            or reference.revision != memory.current_revision
            or memory.validity in (ValidityState.INVALIDATED, ValidityState.EXPIRED)
            or (
                memory.expires_at_event_pool_index is not None
                and memory.expires_at_event_pool_index <= current.event_pool_index
            )
        ):
            strength = 1_000_000
        elif memory.validity is ValidityState.SUPERSEDED:
            strength = 750_000
        else:
            continue
        matches.append((strength, reference.memory_pool_index, reference.revision))
    if not matches:
        return _no_match(signal_type)
    strength = max(item[0] for item in matches)
    references_at_max = tuple(
        sorted(
            (pool, revision)
            for item_strength, pool, revision in matches
            if item_strength == strength
        )
    )
    index = current.event_pool_index
    return _detected(
        signal_type,
        strength,
        _evidence(
            (index,),
            binding_indices=(index,),
            memory_references=references_at_max,
        ),
    )


def _evaluate_stagnation(
    fixture: OutcomeFreeTraceFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> ReferencePredicateResult:
    signal_type = SignalType.STAGNATION
    current = fixture.bindings[-1]
    if not _is_action_candidate(fixture.events[-1]):
        return _no_match(signal_type)
    if not _is_structured_action(fixture.events[-1]):
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    if current.action_step is None or current.progress_marker_digest is None:
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    marker = current.progress_marker_digest
    selected: list[int] = []
    action_steps: set[int] = set()
    for event, binding in zip(reversed(fixture.events), reversed(fixture.bindings), strict=True):
        if event.event_type is EventType.RUN_END or event.phase is EventPhase.TERMINAL:
            break
        if not _is_action_candidate(event):
            break
        if not _is_structured_action(event):
            return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
        if binding.action_step is None or binding.progress_marker_digest is None:
            if event.event_type is not EventType.RUN_START:
                return _abstained(
                    signal_type,
                    ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
                )
            break
        if binding.progress_marker_digest != marker or binding.action_step in action_steps:
            break
        if not _binding_resolves_to_skeleton(binding, task_skeleton):
            return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
        action_steps.add(binding.action_step)
        selected.append(binding.event_pool_index)
        if len(selected) == 8:
            break
    if len(selected) < 4:
        return _no_match(signal_type)
    indices = tuple(reversed(selected))
    strength = min(1_000_000, 500_000 + 125_000 * (len(indices) - 4))
    return _detected(
        signal_type,
        strength,
        _evidence(indices, binding_indices=indices),
    )


def _evaluate_irreversible_action(
    fixture: OutcomeFreeTraceFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> ReferencePredicateResult:
    signal_type = SignalType.IRREVERSIBLE_ACTION
    current = fixture.bindings[-1]
    if (
        current.impact is None
        or current.authorization_event_pool_indices is None
        or current.safeguard_event_pool_indices is None
    ):
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    if current.impact is PublicImpactClass.UNKNOWN:
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    if current.impact is PublicImpactClass.REVERSIBLE:
        return _no_match(signal_type)
    if not _binding_resolves_to_skeleton(current, task_skeleton):
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    if not _is_structured_action(fixture.events[-1]):
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    current_index = current.event_pool_index
    controls = (
        *current.authorization_event_pool_indices,
        *current.safeguard_event_pool_indices,
    )
    if controls:
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)
    return _detected(
        signal_type,
        1_000_000,
        _evidence((current_index,), binding_indices=(current_index,)),
    )


def _evaluate_conflict(fixture: OutcomeFreeTraceFixture) -> ReferencePredicateResult:
    signal_type = SignalType.CONFLICT
    current = fixture.bindings[-1]
    assertions = current.assertions
    if assertions is None:
        return _abstained(
            signal_type,
            ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE,
        )
    if len(assertions) < 2:
        return _no_match(signal_type)
    assertion_digests = tuple(
        public_assertion_fixture_digest(assertion) for assertion in assertions
    )
    superseded = {
        assertion.supersedes_assertion_digest
        for assertion in assertions
        if assertion.supersedes_assertion_digest is not None
    }
    if not superseded.issubset(assertion_digests):
        return _abstained(signal_type, ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED)

    groups: dict[tuple[str, str], list[tuple[int, PublicAssertionFixture]]] = {}
    for index, (assertion, assertion_digest) in enumerate(
        zip(assertions, assertion_digests, strict=True)
    ):
        if assertion_digest in superseded:
            continue
        groups.setdefault((assertion.subject_id, assertion.predicate_id), []).append(
            (index, assertion)
        )
    for key in sorted(groups):
        group = groups[key]
        highest = max(assertion.precedence for _, assertion in group)
        highest_items = tuple(
            (index, assertion) for index, assertion in group if assertion.precedence == highest
        )
        for left_position, (left_index, left) in enumerate(highest_items):
            for right_index, right in highest_items[left_position + 1 :]:
                if left.value_digest == right.value_digest:
                    continue
                event_index = current.event_pool_index
                return _detected(
                    signal_type,
                    1_000_000,
                    _evidence(
                        (event_index,),
                        binding_indices=(event_index,),
                        assertion_references=(
                            (event_index, left_index),
                            (event_index, right_index),
                        ),
                    ),
                )
    return _no_match(signal_type)


def _validated_evaluator_inputs(
    fixture: OutcomeFreeTraceFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> tuple[OutcomeFreeTraceFixture, OutcomeFreeTaskSkeleton]:
    try:
        if (
            type(fixture) is not OutcomeFreeTraceFixture
            or type(task_skeleton) is not OutcomeFreeTaskSkeleton
        ):
            raise TypeError
        checked_fixture = OutcomeFreeTraceFixture.model_validate_json(
            fixture.model_dump_json(warnings=False)
        )
        checked_skeleton = OutcomeFreeTaskSkeleton.model_validate_json(
            task_skeleton.model_dump_json(warnings=False)
        )
        if len(checked_fixture.events) != len(checked_skeleton.trajectory) or tuple(
            memory.memory_pool_index for memory in checked_fixture.memories
        ) != tuple(range(len(checked_skeleton.candidate_memories))):
            raise ValueError
        for binding, event in zip(
            checked_fixture.bindings,
            checked_skeleton.trajectory,
            strict=True,
        ):
            if binding.action_step is not None and binding.action_step != event.action_step:
                raise ValueError
        for fixture_memory, candidate_memory in zip(
            checked_fixture.memories,
            checked_skeleton.candidate_memories,
            strict=True,
        ):
            if (
                fixture_memory.current_revision != candidate_memory.revision
                or fixture_memory.validity is not candidate_memory.validity
            ):
                raise ValueError
    except Exception:
        raise SignalFixtureInputError() from None
    return checked_fixture, checked_skeleton


def evaluate_reference_predicates(
    fixture: OutcomeFreeTraceFixture,
    task_skeleton: OutcomeFreeTaskSkeleton,
) -> tuple[ReferencePredicateResult, ...]:
    """Evaluate the five frozen non-runtime predicates from raw structured operands only."""

    raw, skeleton = _validated_evaluator_inputs(fixture, task_skeleton)
    return (
        _evaluate_conflict(raw),
        _evaluate_context_shift(raw, skeleton),
        _evaluate_irreversible_action(raw, skeleton),
        _evaluate_stagnation(raw, skeleton),
        _evaluate_stale_constraint(raw),
    )


def _deterministic_uuid(scenario_id: str, label: str) -> UUID:
    digest = bytearray(
        hashlib.sha256(
            f"saliencegate:state-decay-v2:public-fixture-runtime:v1\x00{scenario_id}\x00{label}".encode()
        ).digest()[:16]
    )
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(digest))


def _validated_scenario_id(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SignalFixtureInputError()
    return value


def _normalize_native_event(value: object) -> NormalizedTraceEventDraft:
    if type(value) is not _NativeFixtureEvent:
        raise SignalFixtureInputError()
    native = value
    return NormalizedTraceEventDraft(
        run_id=native.run_id,
        source_event_id=f"public-fixture-event-{native.event.event_pool_index}",
        timestamp=native.timestamp,
        event_type=native.event.event_type,
        phase=native.event.phase,
        payload=native.event.payload,
        parent_ids=native.parent_ids,
        source_adapter=_FIXTURE_ADAPTER_ID,
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


def _unused_capabilities() -> object:
    raise RuntimeError("fixture adapter delivery is unavailable")


def _unused_target_request_id(value: object, target: DeliveryTarget) -> object:
    del value, target
    return None


async def _unused_delivery(delivery: DeliveryEnvelope) -> object:
    del delivery
    raise RuntimeError("fixture adapter delivery is unavailable")


def _event_id_callback(value: object, ordinal: int) -> object:
    if type(value) is not _NativeFixtureEvent or ordinal != value.event.event_pool_index + 1:
        raise SignalFixtureInputError()
    return value.event_id


def _fixture_adapter() -> GenericHarnessAdapter:
    return GenericHarnessAdapter(
        normalize_callback=_normalize_native_event,
        capabilities_callback=_unused_capabilities,
        target_request_id_callback=_unused_target_request_id,
        delivery_callback=_unused_delivery,
        event_id_callback=_event_id_callback,
    )


def _legacy_detectors() -> tuple[SignalDetector, ...]:
    detectors: tuple[SignalDetector, ...] = (
        RepeatedActionDetector(RepetitionConfig(window_events=_LEGACY_REPETITION_WINDOW_EVENTS)),
        RepeatedFailureDetector(RepetitionConfig(window_events=_LEGACY_REPETITION_WINDOW_EVENTS)),
        TestFailureDetector(),
        ToolErrorDetector(),
    )
    return tuple(sorted(detectors, key=lambda detector: detector.signal_type.value))


async def evaluate_legacy_signal_fixture(
    fixture: OutcomeFreeTraceFixture,
    *,
    scenario_id: str,
) -> LegacyFixtureEvaluation:
    """Run the four real detectors once at the final boundary over repository-created events."""

    try:
        if type(fixture) is not OutcomeFreeTraceFixture:
            raise TypeError
        raw = OutcomeFreeTraceFixture.model_validate_json(fixture.model_dump_json(warnings=False))
    except Exception:
        raise SignalFixtureInputError() from None
    coordinate = _validated_scenario_id(scenario_id)
    run_id = _deterministic_uuid(coordinate, "run")
    event_ids = tuple(
        _deterministic_uuid(coordinate, f"event:{index}") for index in range(len(raw.events))
    )
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    native_events = tuple(
        _NativeFixtureEvent(
            event=event,
            run_id=run_id,
            event_id=event_ids[event.event_pool_index],
            parent_ids=tuple(event_ids[index] for index in event.parent_event_pool_indices),
            timestamp=base_time + timedelta(microseconds=event.event_pool_index),
        )
        for event in raw.events
    )
    adapter = _fixture_adapter()
    repository = MemoryRunRepository(synthetic_benchmark=True)
    events: list[TraceEvent] = []
    for ordinal, native in enumerate(native_events, start=1):
        draft = adapter.normalize(native)
        event_id = adapter.resolve_event_id(native, ordinal)
        if event_id is None:
            raise SignalFixtureInputError()
        receipt = await repository.append(draft, event_id=event_id)
        events.append(receipt.event)

    context = DetectionContext(run_id=run_id, events=tuple(events))
    indices_by_id = {event.event_id: index for index, event in enumerate(events)}
    results: list[LegacyDetectorResult] = []
    for detector in _legacy_detectors():
        outcome = detector.evaluate(context)
        try:
            evidence = tuple(indices_by_id[event_id] for event_id in outcome.evidence_event_ids)
            related = tuple(indices_by_id[event_id] for event_id in outcome.related_event_ids)
        except KeyError:
            raise SignalFixtureInputError() from None
        strength_ppm = None if outcome.strength is None else round(outcome.strength * 1_000_000)
        results.append(
            LegacyDetectorResult(
                _issuer=_LEGACY_RESULT_ISSUER,
                signal_type=detector.signal_type,
                detector_version=detector.detector_version,
                status=outcome.status,
                strength_ppm=strength_ppm,
                evidence_event_pool_indices=evidence,
                related_event_pool_indices=related,
                abstention_reason=outcome.abstention_reason,
            )
        )
    return LegacyFixtureEvaluation(
        _issuer=_LEGACY_RESULT_ISSUER,
        events=tuple(events),
        results=tuple(results),
    )


def detected_signal_projection(
    *,
    legacy_results: tuple[LegacyDetectorResult, ...] = (),
    reference_results: tuple[ReferencePredicateResult, ...] = (),
) -> tuple[PublicExpectedSignal, ...]:
    """Project detected real/reference results into the shared public comparison shape."""

    if (
        type(legacy_results) is not tuple
        or type(reference_results) is not tuple
        or any(type(result) is not LegacyDetectorResult for result in legacy_results)
        or any(type(result) is not ReferencePredicateResult for result in reference_results)
        or (
            bool(legacy_results)
            and tuple(result.signal_type for result in legacy_results) != _LEGACY_TYPES
        )
        or (
            bool(reference_results)
            and tuple(result.signal_type for result in reference_results) != _REFERENCE_TYPES
        )
    ):
        raise SignalFixtureInputError()
    projected: list[PublicExpectedSignal] = []
    for legacy_result in legacy_results:
        if legacy_result.status is not DetectionStatus.DETECTED:
            continue
        if legacy_result.strength_ppm is None:
            raise SignalFixtureInputError()
        projected.append(
            PublicExpectedSignal(
                signal_type=legacy_result.signal_type,
                strength_ppm=legacy_result.strength_ppm,
                evidence=_evidence(legacy_result.evidence_event_pool_indices),
            )
        )
    for reference_result in reference_results:
        if reference_result.status is not ReferencePredicateStatus.DETECTED:
            continue
        if reference_result.strength_ppm is None or reference_result.evidence is None:
            raise SignalFixtureInputError()
        projected.append(
            PublicExpectedSignal(
                signal_type=reference_result.signal_type,
                strength_ppm=reference_result.strength_ppm,
                evidence=reference_result.evidence,
            )
        )
    signal_types = tuple(item.signal_type for item in projected)
    if len(set(signal_types)) != len(signal_types):
        raise SignalFixtureInputError()
    return tuple(sorted(projected, key=lambda item: item.signal_type.value))


__all__ = [
    "PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN",
    "LegacyDetectorResult",
    "LegacyFixtureEvaluation",
    "ReferencePredicateAbstentionReason",
    "ReferencePredicateResult",
    "ReferencePredicateStatus",
    "SignalFixtureInputError",
    "detected_signal_projection",
    "evaluate_legacy_signal_fixture",
    "evaluate_reference_predicates",
    "materialize_public_trace_fixture",
    "public_assertion_fixture_digest",
]
