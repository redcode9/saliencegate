from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError

import saliencegate.intervention.claims as claims_module
import saliencegate.repository.projector as projector_module
from saliencegate.domain import (
    ClaimKind,
    CycleRecord,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionDecision,
    MemoryKind,
    MemoryRecord,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    cycle_id,
)
from saliencegate.intervention import (
    DeterministicSelectorProvenance,
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingState,
    GroundingVerificationError,
    InterventionProposal,
    ProposedClaim,
    RenderingConfig,
    resolve_grounding_configuration,
    verify_grounded_intervention,
)
from saliencegate.ports import BeginCycle
from saliencegate.ports.repository import ProjectionInvariantError
from saliencegate.repository.projector import Projection, empty_projection
from saliencegate.runtime.token_counting import APPROXIMATE_TOKEN_ALGORITHM_VERSION

RUN_ID = UUID("00000000-0000-4000-8000-000000009001")
CURRENT_EVENT_ID = UUID("00000000-0000-4000-8000-000000009002")
MISSING_EVENT_ID = UUID("00000000-0000-4000-8000-000000009003")
MEMORY_ID = UUID("00000000-0000-4000-8000-000000009004")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000009005")
NOW = datetime(2026, 7, 11, 20, 0, tzinfo=UTC)
MODEL_CALL_INDEX = 0
MODEL_CALL_DIGEST = "d" * 64
POLICY_CONFIGURATION_DIGEST = "a" * 64
POLICY_VERSION = "receipt-tests/1"


def rendering_config() -> RenderingConfig:
    return RenderingConfig(
        schema_version="1.0",
        renderer_version="fixed-ascii/v1",
        token_counter_version=APPROXIMATE_TOKEN_ALGORITHM_VERSION,
        max_claims=2,
        max_evidence_bytes=1_024,
        max_output_bytes=4_096,
        max_token_equivalents=1_024,
        include_provenance=False,
    )


def grounding_config(
    *,
    max_claims: int = 2,
    max_pointer_segments: int = 32,
    allowed_delivery_targets: tuple[DeliveryTarget, ...] = (
        DeliveryTarget.NEXT_MODEL_CALL,
        DeliveryTarget.PRE_ACTION_REPLAN,
    ),
) -> GroundingConfig:
    return GroundingConfig(
        schema_version="1.0",
        pipeline_version="grounding-pipeline/v1",
        claim_schema_version="citation-only-claims/v1",
        max_claims=max_claims,
        max_evidence_per_claim=1,
        max_pointer_segments=max_pointer_segments,
        max_pointer_utf8_bytes=1_024,
        duplicate_window_events=0,
        cooldown_events=0,
        ttl_steps=1,
        allowed_delivery_targets=allowed_delivery_targets,
        rendering=rendering_config(),
    )


def trace_event(
    *,
    sequence: int = 1,
    event_id: UUID = CURRENT_EVENT_ID,
    message: str = "Run the repository test suite.",
) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        run_id=RUN_ID,
        sequence=sequence,
        source_event_id=f"receipt-event-{sequence}",
        timestamp=NOW - timedelta(seconds=1),
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": message},
        payload_digest=PayloadDigest(
            algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
            value="e" * 64,
        ),
        source_adapter="receipt-tests/1",
        trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
    )


def event_reference(event_id: UUID = CURRENT_EVENT_ID) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=event_id,
        field_path="/payload/message",
    )


def proposal(
    *,
    action: InterventionAction = InterventionAction.REMIND,
    event_id: UUID = CURRENT_EVENT_ID,
    model_free_text: str | None = None,
) -> InterventionProposal:
    claims = (
        ()
        if action is InterventionAction.SILENCE
        else (
            ProposedClaim(
                kind=ClaimKind.ENVIRONMENT_FACT,
                evidence=event_reference(event_id),
            ),
        )
    )
    return InterventionProposal(
        action=action,
        claims=claims,
        confidence=0.8,
        model_free_text=model_free_text,
    )


def grounding_context(
    *,
    sequence: int = 1,
    target: DeliveryTarget | None = DeliveryTarget.NEXT_MODEL_CALL,
    cycle: str = "c" * 64,
) -> GroundingContext:
    return GroundingContext(
        schema_version="1.0",
        intervention_id=INTERVENTION_ID,
        run_id=RUN_ID,
        cycle_id=cycle,
        current_event_sequence=sequence,
        created_at=NOW,
        requested_delivery_target=target,
        model_call_index=MODEL_CALL_INDEX,
        model_call_digest=MODEL_CALL_DIGEST,
    )


def selector_provenance() -> DeterministicSelectorProvenance:
    return DeterministicSelectorProvenance(
        selector_id="candidate-bank-ascii-token-top-k/v1",
        configuration_digest="1" * 64,
        request_digest="2" * 64,
        result_digest="3" * 64,
    )


def selector_grounding_context(
    *,
    sequence: int = 1,
    target: DeliveryTarget | None = DeliveryTarget.NEXT_MODEL_CALL,
    cycle: str = "c" * 64,
) -> GroundingContext:
    return GroundingContext(
        schema_version="2.0",
        intervention_id=INTERVENTION_ID,
        run_id=RUN_ID,
        cycle_id=cycle,
        current_event_sequence=sequence,
        created_at=NOW,
        requested_delivery_target=target,
        selector_provenance=selector_provenance(),
    )


def grounding_state(current: TraceEvent) -> GroundingState:
    return GroundingState(
        schema_version="1.0",
        events=(current,),
        memories=(),
        reminder_history=(),
    )


def ground(
    selected_proposal: InterventionProposal,
    *,
    configuration: GroundingConfig,
    target: DeliveryTarget | None = DeliveryTarget.NEXT_MODEL_CALL,
    current: TraceEvent | None = None,
    cycle: str = "c" * 64,
) -> tuple[InterventionDecision, GroundingContext, GroundingState]:
    selected_event = trace_event() if current is None else current
    context = grounding_context(
        sequence=selected_event.sequence,
        target=target,
        cycle=cycle,
    )
    state = grounding_state(selected_event)
    decision = GroundingPipeline(configuration).ground(
        selected_proposal,
        context=context,
        state=state,
    )
    return decision, context, state


def verify(
    decision: InterventionDecision,
    *,
    context: GroundingContext,
    state: GroundingState,
    configuration: GroundingConfig,
) -> None:
    verify_grounded_intervention(
        decision,
        context=context,
        state=state,
        expected_configuration=resolve_grounding_configuration(configuration),
    )


def receipt_payload(decision: InterventionDecision) -> Mapping[str, object]:
    return decision.grounding_receipt


@dataclass(frozen=True, slots=True)
class PinnedCycle:
    cycle_id: str
    last_event_sequence: int
    grounding_version: str
    grounding_configuration: Mapping[str, object]
    grounding_configuration_digest: str
    requested_delivery_target: DeliveryTarget | None
    model_call_digests: tuple[str, ...]
    selector_provenance: Mapping[str, object] | None


def pinned_cycle(
    decision: InterventionDecision,
    *,
    configuration: GroundingConfig,
    target: DeliveryTarget | None,
    sequence: int,
) -> CycleRecord:
    resolved = resolve_grounding_configuration(configuration)
    selected = decision.grounding_receipt.get("selector_provenance")
    selector = cast(Mapping[str, object], selected) if isinstance(selected, Mapping) else None
    return cast(
        CycleRecord,
        PinnedCycle(
            cycle_id=decision.cycle_id,
            last_event_sequence=sequence,
            grounding_version=resolved.pipeline_version,
            grounding_configuration=resolved.configuration,
            grounding_configuration_digest=resolved.configuration_digest,
            requested_delivery_target=target,
            model_call_digests=(MODEL_CALL_DIGEST,),
            selector_provenance=selector,
        ),
    )


def projected_state(current: TraceEvent) -> Projection:
    return Projection(
        run_id=RUN_ID,
        events_by_id={current.event_id: current},
        events_by_sequence={current.sequence: current},
    )


def test_decision_persists_a_typed_citation_only_grounding_receipt() -> None:
    secret = "raw-model-free-text-must-not-persist"
    configuration = grounding_config()
    decision, _context, _state = ground(
        proposal(event_id=MISSING_EVENT_ID, model_free_text=secret),
        configuration=configuration,
    )

    receipt = receipt_payload(decision)
    receipt_json = canonical_json(receipt).decode("utf-8")

    assert set(receipt) == {
        "receipt_version",
        "parse_status",
        "proposal_action",
        "claims",
        "confidence",
        "requested_delivery_target",
        "model_call_index",
        "model_call_digest",
    }
    assert receipt["receipt_version"] == "grounding-receipt/v1"
    assert receipt["parse_status"] == "valid"
    assert receipt["proposal_action"] == InterventionAction.REMIND.value
    assert receipt["requested_delivery_target"] == DeliveryTarget.NEXT_MODEL_CALL.value
    assert receipt["model_call_index"] == MODEL_CALL_INDEX
    assert receipt["model_call_digest"] == MODEL_CALL_DIGEST
    assert str(MISSING_EVENT_ID) in receipt_json
    assert secret not in receipt_json

    receipt_type = getattr(claims_module, "GroundingReceipt", None)
    assert isinstance(receipt_type, type) and issubclass(receipt_type, BaseModel)
    typed_receipt = receipt_type.model_validate_json(receipt_json)
    assert canonical_json(typed_receipt) == canonical_json(receipt)


def test_deterministic_selector_receipt_replays_without_a_fake_model_call() -> None:
    configuration = grounding_config()
    current = trace_event()
    context = selector_grounding_context()
    state = grounding_state(current)
    decision = GroundingPipeline(configuration).ground(
        proposal(),
        context=context,
        state=state,
    )

    receipt = receipt_payload(decision)
    assert receipt["receipt_version"] == "grounding-receipt/v2"
    assert "model_call_index" not in receipt
    assert "model_call_digest" not in receipt
    assert receipt["selector_provenance"] == selector_provenance().model_dump(mode="json")
    verify(
        decision,
        context=context,
        state=state,
        configuration=configuration,
    )
    projector_module._authoritatively_verify_intervention(
        projected_state(current),
        pinned_cycle(
            decision,
            configuration=configuration,
            target=DeliveryTarget.NEXT_MODEL_CALL,
            sequence=current.sequence,
        ),
        decision,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("selector_id", "candidate-bank-ascii-token-top-k/v2"),
        ("configuration_digest", "4" * 64),
        ("request_digest", "5" * 64),
        ("result_digest", "6" * 64),
        ("provenance_digest", "7" * 64),
    ),
)
def test_selector_provenance_rejects_every_tampered_binding(
    field_name: str,
    replacement: object,
) -> None:
    payload = selector_provenance().model_dump(mode="json")
    payload[field_name] = replacement

    with pytest.raises(ValidationError, match="provenance digest"):
        DeterministicSelectorProvenance.model_validate(payload)


def test_grounding_provenance_versions_are_an_exact_xor() -> None:
    base = selector_grounding_context().model_dump(mode="python")

    with pytest.raises(ValidationError, match="provenance"):
        GroundingContext.model_validate(
            {
                **base,
                "schema_version": "1.0",
            }
        )
    with pytest.raises(ValidationError, match="provenance"):
        GroundingContext.model_validate(
            {
                **base,
                "model_call_index": 0,
                "model_call_digest": MODEL_CALL_DIGEST,
            }
        )


def test_forged_silence_reason_is_rejected_by_receipt_replay() -> None:
    configuration = grounding_config()
    decision, context, state = ground(
        proposal(event_id=MISSING_EVENT_ID),
        configuration=configuration,
    )
    assert decision.action is InterventionAction.SILENCE
    assert decision.reason_code is ReasonCode.CITATION_MISSING

    forged = decision.model_copy(update={"reason_code": ReasonCode.INVALID_PROVENANCE})

    with pytest.raises(GroundingVerificationError):
        verify(
            forged,
            context=context,
            state=state,
            configuration=configuration,
        )


def test_valid_two_claim_receipt_replays_against_pinned_one_claim_limit() -> None:
    configuration = grounding_config(max_claims=1)
    selected_proposal = InterventionProposal(
        action=InterventionAction.REMIND,
        claims=(
            ProposedClaim(
                kind=ClaimKind.REQUIREMENT,
                evidence=event_reference(),
            ),
            ProposedClaim(
                kind=ClaimKind.ENVIRONMENT_FACT,
                evidence=event_reference(),
            ),
        ),
        confidence=0.7,
        model_free_text="untrusted prose is excluded from the receipt",
    )
    decision, context, state = ground(
        selected_proposal,
        configuration=configuration,
    )

    receipt = receipt_payload(decision)
    assert receipt["parse_status"] == "valid"
    assert len(cast(tuple[object, ...], receipt["claims"])) == 2
    assert decision.action is InterventionAction.SILENCE
    assert decision.reason_code is ReasonCode.CLAIM_OVER_LIMIT

    verify(
        decision,
        context=context,
        state=state,
        configuration=configuration,
    )
    forged = decision.model_copy(update={"reason_code": ReasonCode.SCHEMA_INVALID})
    with pytest.raises(GroundingVerificationError):
        verify(
            forged,
            context=context,
            state=state,
            configuration=configuration,
        )


def test_non_valid_receipt_statuses_replay_without_constructing_invalid_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = grounding_config()
    valid_claim = ProposedClaim(
        kind=ClaimKind.REQUIREMENT,
        evidence=event_reference(),
    )
    valid_silence = proposal(action=InterventionAction.SILENCE)
    malformed_inputs = (
        (
            valid_silence.model_copy(update={"action": InterventionAction.REMIND}),
            "empty_reminder",
            ReasonCode.NO_GROUNDED_CLAIMS,
        ),
        (
            proposal().model_copy(update={"claims": (valid_claim, valid_claim, valid_claim)}),
            "claim_over_limit",
            ReasonCode.CLAIM_OVER_LIMIT,
        ),
        (
            valid_silence.model_copy(update={"action": "not-an-action"}),
            "schema_invalid",
            ReasonCode.SCHEMA_INVALID,
        ),
    )
    grounded = tuple(
        (
            *ground(selected, configuration=configuration),
            parse_status,
            expected_reason,
        )
        for selected, parse_status, expected_reason in malformed_inputs
    )

    def reject_model_construct(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError("receipt replay must not construct an invalid proposal model")

    monkeypatch.setattr(
        InterventionProposal,
        "model_construct",
        classmethod(reject_model_construct),
    )

    for decision, context, state, parse_status, expected_reason in grounded:
        assert receipt_payload(decision)["parse_status"] == parse_status
        assert decision.reason_code is expected_reason
        verify(
            decision,
            context=context,
            state=state,
            configuration=configuration,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("model_call_index", 1),
        ("model_call_digest", "f" * 64),
    ),
)
def test_receipt_is_bound_to_exact_model_call(
    field_name: str,
    replacement: object,
) -> None:
    configuration = grounding_config()
    decision, context, state = ground(
        proposal(action=InterventionAction.SILENCE),
        configuration=configuration,
    )
    forged_receipt = dict(receipt_payload(decision))
    forged_receipt[field_name] = replacement
    forged = decision.model_copy(update={"grounding_receipt": forged_receipt})

    with pytest.raises(GroundingVerificationError):
        verify(
            forged,
            context=context,
            state=state,
            configuration=configuration,
        )


def test_self_consistent_alternate_configuration_cannot_bypass_cycle_pin() -> None:
    trusted = grounding_config(max_pointer_segments=32)
    alternate = grounding_config(max_pointer_segments=31)
    decision, _context, _state = ground(
        proposal(action=InterventionAction.SILENCE),
        configuration=alternate,
    )
    current = trace_event()
    cycle = pinned_cycle(
        decision,
        configuration=trusted,
        target=DeliveryTarget.NEXT_MODEL_CALL,
        sequence=current.sequence,
    )

    assert decision.grounding_configuration_digest != cycle.grounding_configuration_digest
    with pytest.raises(ProjectionInvariantError, match="grounded intervention"):
        projector_module._authoritatively_verify_intervention(
            projected_state(current),
            cycle,
            decision,
        )


@pytest.mark.parametrize(
    ("selected_proposal", "target", "allowed_targets", "expected_reason"),
    (
        (
            proposal(action=InterventionAction.SILENCE),
            DeliveryTarget.NEXT_MODEL_CALL,
            (DeliveryTarget.NEXT_MODEL_CALL, DeliveryTarget.PRE_ACTION_REPLAN),
            ReasonCode.SILENCE_SELECTED,
        ),
        (
            proposal(),
            DeliveryTarget.PRE_ACTION_REPLAN,
            (DeliveryTarget.NEXT_MODEL_CALL,),
            ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
        ),
    ),
)
def test_projector_preserves_requested_target_when_replaying_silence(
    monkeypatch: pytest.MonkeyPatch,
    selected_proposal: InterventionProposal,
    target: DeliveryTarget,
    allowed_targets: tuple[DeliveryTarget, ...],
    expected_reason: ReasonCode,
) -> None:
    configuration = grounding_config(allowed_delivery_targets=allowed_targets)
    decision, _context, _state = ground(
        selected_proposal,
        configuration=configuration,
        target=target,
    )
    current = trace_event()
    captured: list[GroundingContext] = []

    def capture(
        _decision: object,
        *,
        context: GroundingContext,
        state: GroundingState,
        **_arguments: object,
    ) -> None:
        assert state.events == (current,)
        captured.append(context)

    monkeypatch.setattr(projector_module, "verify_grounded_intervention", capture)
    projector_module._authoritatively_verify_intervention(
        projected_state(current),
        pinned_cycle(
            decision,
            configuration=configuration,
            target=target,
            sequence=current.sequence,
        ),
        decision,
    )

    assert decision.action is InterventionAction.SILENCE
    assert decision.reason_code is expected_reason
    assert decision.delivery_target is None
    assert receipt_payload(decision)["requested_delivery_target"] == target.value
    assert len(captured) == 1
    assert captured[0].requested_delivery_target is target


def test_unsupported_target_in_silence_receipt_cannot_be_rewritten() -> None:
    configuration = grounding_config(allowed_delivery_targets=(DeliveryTarget.NEXT_MODEL_CALL,))
    decision, context, state = ground(
        proposal(),
        configuration=configuration,
        target=DeliveryTarget.PRE_ACTION_REPLAN,
    )
    assert decision.reason_code is ReasonCode.UNSUPPORTED_DELIVERY_TARGET
    forged_receipt = dict(receipt_payload(decision))
    forged_receipt["requested_delivery_target"] = DeliveryTarget.NEXT_MODEL_CALL.value
    forged = decision.model_copy(update={"grounding_receipt": forged_receipt})

    with pytest.raises(GroundingVerificationError):
        verify(
            forged,
            context=context,
            state=state,
            configuration=configuration,
        )


class LargeNonIterableEventMap(Mapping[int, TraceEvent]):
    def __init__(self, current: TraceEvent, *, size: int) -> None:
        self._current = current
        self._size = size

    def __getitem__(self, key: int) -> TraceEvent:
        if key != self._current.sequence:
            raise KeyError(key)
        return self._current

    def __iter__(self) -> Iterator[int]:
        raise AssertionError("projector must not scan or serialize the whole event state")

    def __len__(self) -> int:
        return self._size


def test_projector_selects_bounded_grounding_state_above_ten_thousand_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = 10_001
    current = trace_event(sequence=sequence)
    configuration = grounding_config()
    decision, _context, _state = ground(
        proposal(action=InterventionAction.SILENCE),
        configuration=configuration,
        current=current,
    )
    projection = empty_projection(RUN_ID)
    object.__setattr__(
        projection,
        "events_by_sequence",
        LargeNonIterableEventMap(current, size=sequence),
    )
    object.__setattr__(projection, "events_by_id", {current.event_id: current})
    captured: list[GroundingState] = []

    def capture(
        _decision: object,
        *,
        context: GroundingContext,
        state: GroundingState,
        **_arguments: object,
    ) -> None:
        assert context.current_event_sequence == sequence
        captured.append(state)

    monkeypatch.setattr(projector_module, "verify_grounded_intervention", capture)

    projector_module._authoritatively_verify_intervention(
        projection,
        pinned_cycle(
            decision,
            configuration=configuration,
            target=DeliveryTarget.NEXT_MODEL_CALL,
            sequence=sequence,
        ),
        decision,
    )

    assert len(projection.events_by_sequence) == 10_001
    assert len(captured) == 1
    assert captured[0].events == (current,)


def test_rendering_configuration_pins_token_counter_in_grounding_digest() -> None:
    counter_field = RenderingConfig.model_fields.get("token_counter_version")
    assert counter_field is not None
    assert counter_field.is_required()

    configuration = grounding_config()
    resolved = resolve_grounding_configuration(configuration)
    rendered = resolved.configuration["rendering"]

    assert isinstance(rendered, Mapping)
    assert rendered["token_counter_version"] == APPROXIMATE_TOKEN_ALGORITHM_VERSION
    incomplete = configuration.rendering.model_dump(mode="python")
    incomplete.pop("token_counter_version")
    with pytest.raises(ValidationError):
        RenderingConfig.model_validate(incomplete)


def test_delivery_target_set_has_one_canonical_order_and_digest() -> None:
    canonical = grounding_config(
        allowed_delivery_targets=(
            DeliveryTarget.NEXT_MODEL_CALL,
            DeliveryTarget.PRE_ACTION_REPLAN,
        )
    )
    reversed_input = grounding_config(
        allowed_delivery_targets=(
            DeliveryTarget.PRE_ACTION_REPLAN,
            DeliveryTarget.NEXT_MODEL_CALL,
        )
    )

    assert reversed_input.allowed_delivery_targets == canonical.allowed_delivery_targets
    resolved = resolve_grounding_configuration(canonical)
    assert resolve_grounding_configuration(reversed_input) == resolved
    assert (
        resolved.configuration_digest
        == "94bf38e6dbc7d53ac6416a14e9a1d5a4da77c2e2571ffc4386d34be519510fc0"
    )


def test_pipeline_builds_one_validated_pre_model_pin() -> None:
    pipeline = GroundingPipeline(grounding_config())

    pin = pipeline.pin(DeliveryTarget.NEXT_MODEL_CALL)

    assert pin.grounding_version == pipeline.pipeline_version
    assert pin.grounding_configuration == pipeline.resolved_configuration.configuration
    assert pin.grounding_configuration_digest == pipeline.configuration_digest
    assert pin.requested_delivery_target is DeliveryTarget.NEXT_MODEL_CALL


def test_cycle_and_begin_command_require_flat_grounding_pins() -> None:
    required_fields = {
        "grounding_version",
        "grounding_configuration",
        "grounding_configuration_digest",
        "requested_delivery_target",
    }

    assert required_fields <= CycleRecord.model_fields.keys()
    assert required_fields <= BeginCycle.model_fields.keys()
    assert CycleRecord.model_fields["grounding_configuration"].repr is False
    assert BeginCycle.model_fields["grounding_configuration"].repr is False


def test_cycle_identity_v2_commits_grounding_pin_and_requested_target() -> None:
    derive = cast(Callable[..., str], cycle_id)
    base = {
        "run_id": RUN_ID,
        "first_event_sequence": 1,
        "last_event_sequence": 1,
        "policy_version": POLICY_VERSION,
        "configuration_digest": POLICY_CONFIGURATION_DIGEST,
        "grounding_version": "grounding-pipeline/v1",
        "grounding_configuration_digest": "b" * 64,
        "requested_delivery_target": DeliveryTarget.NEXT_MODEL_CALL,
    }

    selected = derive(**base)
    changed_version = derive(**{**base, "grounding_version": "grounding-pipeline/v2"})
    changed_digest = derive(**{**base, "grounding_configuration_digest": "c" * 64})
    changed_target = derive(
        **{**base, "requested_delivery_target": DeliveryTarget.PRE_ACTION_REPLAN}
    )

    assert len({selected, changed_version, changed_digest, changed_target}) == 4


def test_grounding_state_repr_hides_event_and_memory_source_text() -> None:
    event_secret = "event-payload-sensitive-sentinel"
    memory_secret = "memory-content-sensitive-sentinel"
    current = trace_event(message=event_secret)
    selected_memory = MemoryRecord(
        memory_id=MEMORY_ID,
        run_id=RUN_ID,
        kind=MemoryKind.KNOWLEDGE,
        content=memory_secret,
        provenance=(event_reference(),),
        confidence=0.9,
        validity=ValidityState.ACTIVE,
        revision=1,
        created_at=NOW - timedelta(seconds=2),
        updated_at=NOW - timedelta(seconds=1),
        trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
    )
    selected_state = GroundingState(
        schema_version="1.0",
        events=(current,),
        memories=(selected_memory,),
        reminder_history=(),
    )

    rendered = repr(selected_state)

    assert event_secret not in rendered
    assert memory_secret not in rendered
