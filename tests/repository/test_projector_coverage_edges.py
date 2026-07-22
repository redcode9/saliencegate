"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from tests.repository.test_projector import (
    ATTEMPT_1_ID,
    CLAIM_1_ID,
    CLAIM_2_ID,
    DECISION_1_ID,
    DECISION_2_ID,
    DELIVERY_ID,
    EVENT_1_ID,
    EVENT_2_ID,
    INTERVENTION_1_ID,
    MEMORY_1_ID,
    MEMORY_3_ID,
    MEMORY_4_ID,
    NOW,
    RUN_ID,
    apply_records,
    committed_with_intervention,
    cycle_revisions,
    delta_preview_fixture,
    entry,
    event_reference,
    first_committed_projection,
    first_reminder_projection,
    grounded_reminder,
    invocation,
    trace_event,
)

from saliencegate.domain import (
    BudgetAmounts,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    ExpirationAction,
    ExpirationPatch,
    MemoryDelta,
    MemoryInvalidation,
    ReasonCode,
    canonical_digest,
)
from saliencegate.ports.repository import ProjectionInvariantError
from saliencegate.repository.projector import (
    _authoritatively_verify_intervention,
    _resolved_cycle_grounding,
    _validate_cycle_budget,
    apply_entry,
    empty_projection,
    preview_memory_delta,
    validate_complete_projection,
)


def test_preview_rejects_replacement_overlapping_an_invalidation() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    overlapping = delta.model_copy(
        update={
            "invalidations": (
                *delta.invalidations,
                MemoryInvalidation(
                    memory_id=MEMORY_3_ID,
                    expected_revision=1,
                    reason_code=ReasonCode.CONFLICT,
                ),
            )
        }
    )

    with pytest.raises(ProjectionInvariantError, match="cannot overlap"):
        preview_memory_delta(
            projection,
            overlapping,
            assignments,
            last_event_sequence=committed.last_event_sequence,
        )


def test_preview_rejects_existing_assigned_create_id() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    duplicate = (
        assignments[0].model_copy(update={"memory_id": MEMORY_1_ID}),
        assignments[1],
    )

    with pytest.raises(ProjectionInvariantError, match="assigned memory ID already exists"):
        preview_memory_delta(
            projection,
            delta,
            duplicate,
            last_event_sequence=committed.last_event_sequence,
        )


@pytest.mark.parametrize(
    ("expiration", "expected"),
    [
        (ExpirationPatch(action=ExpirationAction.CLEAR), None),
        (
            ExpirationPatch(
                action=ExpirationAction.SET,
                value=NOW + timedelta(days=1),
            ),
            NOW + timedelta(days=1),
        ),
    ],
)
def test_preview_applies_non_default_expiration_patches(
    expiration: ExpirationPatch,
    expected: datetime | None,
) -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    update = delta.updates[0].model_copy(update={"expiration": expiration})
    changed = delta.model_copy(update={"updates": (update,)})

    preview = preview_memory_delta(
        projection,
        changed,
        assignments,
        last_event_sequence=committed.last_event_sequence,
    )

    assert preview.memories[MEMORY_1_ID].expires_at == expected


def test_preview_rejects_backward_invalidation_timestamp() -> None:
    projection, delta, _assignments, committed = delta_preview_fixture()
    changed = MemoryDelta(
        delta_id=delta.delta_id,
        run_id=RUN_ID,
        invalidations=(
            MemoryInvalidation(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        created_at=NOW,
    )

    with pytest.raises(ProjectionInvariantError, match="invalidation timestamp"):
        preview_memory_delta(
            projection,
            changed,
            (),
            last_event_sequence=committed.last_event_sequence,
        )


def test_preview_invalidation_clears_current_private_status() -> None:
    projection, delta, _assignments, committed = delta_preview_fixture()
    invalidation_only = MemoryDelta(
        delta_id=delta.delta_id,
        run_id=RUN_ID,
        invalidations=(
            MemoryInvalidation(
                memory_id=MEMORY_3_ID,
                expected_revision=1,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        created_at=delta.created_at,
    )

    preview = preview_memory_delta(
        projection,
        invalidation_only,
        (),
        last_event_sequence=committed.last_event_sequence,
    )

    assert preview.current_private_status_id is None


def test_preview_rejects_unconditional_replacement_when_private_status_exists() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    replacement = delta.private_status_replacement
    assert replacement is not None
    changed = delta.model_copy(
        update={
            "creates": (),
            "updates": (),
            "invalidations": (),
            "private_status_replacement": replacement.model_copy(
                update={"expected_memory_id": None, "expected_revision": None}
            ),
        }
    )

    with pytest.raises(ProjectionInvariantError, match="already exists"):
        preview_memory_delta(
            projection,
            changed,
            (assignments[1],),
            last_event_sequence=committed.last_event_sequence,
        )


def test_preview_rejects_replacement_target_that_is_not_current_private_status() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    replacement = delta.private_status_replacement
    assert replacement is not None
    changed = delta.model_copy(
        update={
            "creates": (),
            "updates": (),
            "invalidations": (),
            "private_status_replacement": replacement.model_copy(
                update={"expected_memory_id": MEMORY_1_ID, "expected_revision": 1}
            ),
        }
    )

    with pytest.raises(ProjectionInvariantError, match="target is not current"):
        preview_memory_delta(
            projection,
            changed,
            (assignments[1],),
            last_event_sequence=committed.last_event_sequence,
        )


def test_preview_rejects_backward_private_status_timestamp() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    changed = delta.model_copy(
        update={
            "creates": (),
            "updates": (),
            "invalidations": (),
            "created_at": NOW,
        }
    )

    with pytest.raises(ProjectionInvariantError, match="private-status timestamp"):
        preview_memory_delta(
            projection,
            changed,
            (assignments[1],),
            last_event_sequence=committed.last_event_sequence,
        )


def test_preview_rejects_existing_assigned_private_status_id() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    duplicate = (
        assignments[0],
        assignments[1].model_copy(update={"memory_id": MEMORY_3_ID}),
    )

    with pytest.raises(ProjectionInvariantError, match="assigned private-status ID already exists"):
        preview_memory_delta(
            projection,
            delta,
            duplicate,
            last_event_sequence=committed.last_event_sequence,
        )


def _cycle_fixture() -> tuple[object, tuple[object, ...]]:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=MEMORY_4_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    return apply_records((event, decision)), cycles


def _forged_entry(position: int, valid: object, forged: object) -> object:
    return entry(position, valid).model_copy(update={"record": forged})


def test_cycle_must_end_at_its_invocation_event() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, trace_event(2, EVENT_2_ID)))
    changed = cycles[0].model_copy(update={"last_event_sequence": 2})

    with pytest.raises(ProjectionInvariantError, match="must end"):
        apply_entry(projection, _forged_entry(4, cycles[0], changed))


def test_new_cycle_must_continue_memory_cursor() -> None:
    projection, cycles = _cycle_fixture()
    changed = cycles[0].model_copy(update={"first_event_sequence": 2})

    with pytest.raises(ProjectionInvariantError, match="memory cursor"):
        apply_entry(projection, _forged_entry(3, cycles[0], changed))


def test_invocation_decision_cannot_have_two_cycles() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, cycles[0]))
    changed = cycles[0].model_copy(update={"cycle_id": "1" * 64})

    with pytest.raises(ProjectionInvariantError, match="already has a cycle"):
        apply_entry(projection, _forged_entry(4, cycles[0], changed))


def test_run_cannot_have_two_active_cycles() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, cycles[0]))
    projection = apply_entry(projection, entry(4, trace_event(2, EVENT_2_ID)))
    projection = apply_entry(projection, entry(5, invocation(2, DECISION_2_ID)))
    second = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=MemoryDelta(
            delta_id=MEMORY_3_ID,
            run_id=RUN_ID,
            created_at=NOW + timedelta(seconds=23),
        ),
        assignments=(),
        intervention_id=MEMORY_4_ID,
    )[0]
    projection = replace(projection, memory_cursor=1)

    with pytest.raises(ProjectionInvariantError, match="active cycle"):
        apply_entry(projection, entry(6, second))


def test_cycle_revision_rejects_changed_grounding_configuration() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, cycles[0]))
    projection = replace(
        projection,
        cycles={cycles[0].cycle_id: cycles[0].model_copy(update={"grounding_configuration": {}})},
    )

    with pytest.raises(ProjectionInvariantError, match="identity fields"):
        apply_entry(projection, entry(4, cycles[1]))


def test_cycle_revision_rejects_backward_timestamp() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, cycles[0]))
    projection = replace(
        projection,
        cycles={
            cycles[0].cycle_id: cycles[0].model_copy(
                update={"updated_at": cycles[1].updated_at + timedelta(seconds=1)}
            )
        },
    )

    with pytest.raises(ProjectionInvariantError, match="timestamp moved backwards"):
        apply_entry(projection, entry(4, cycles[1]))


@pytest.mark.parametrize(
    ("target", "updates", "message"),
    [
        (2, {"budget_reservation": BudgetAmounts(model_calls=2)}, "budget reservation changed"),
        (3, {"batch_digest": "e" * 64}, "batch digest changed"),
    ],
)
def test_cycle_revision_rejects_changed_reserved_state(
    target: int,
    updates: dict[str, object],
    message: str,
) -> None:
    projection, cycles = _cycle_fixture()
    for offset, cycle in enumerate(cycles[:target], start=3):
        projection = apply_entry(projection, entry(offset, cycle))
    changed = cycles[target].model_copy(update=updates)

    with pytest.raises(ProjectionInvariantError, match=message):
        apply_entry(
            projection,
            _forged_entry(3 + target, cycles[target], changed),
        )


def test_cycle_history_rejects_duplicate_revision_key() -> None:
    projection, cycles = _cycle_fixture()
    projection = apply_entry(projection, entry(3, cycles[0]))
    without_latest = replace(projection, cycles={})

    with pytest.raises(ProjectionInvariantError, match="revision already exists"):
        apply_entry(without_latest, entry(4, cycles[0]))


def test_committed_cycle_rechecks_memory_cursor() -> None:
    projection, cycles = _cycle_fixture()
    for offset, cycle in enumerate(cycles[:-1], start=3):
        projection = apply_entry(projection, entry(offset, cycle))
    projection = replace(projection, memory_cursor=1)

    with pytest.raises(ProjectionInvariantError, match="committed cycle"):
        apply_entry(projection, entry(6, cycles[-1]))


def test_committed_cycle_rejects_duplicate_intervention_id() -> None:
    projection, cycles = _cycle_fixture()
    for offset, cycle in enumerate(cycles[:-1], start=3):
        projection = apply_entry(projection, entry(offset, cycle))
    intervention = cycles[-1].intervention
    assert intervention is not None
    projection = replace(
        projection,
        interventions={intervention.intervention_id: intervention},
    )

    with pytest.raises(ProjectionInvariantError, match="intervention ID already exists"):
        apply_entry(projection, entry(6, cycles[-1]))


def test_committed_reminder_rejects_duplicate_event_sequence() -> None:
    projection, cycles = _cycle_fixture()
    for offset, cycle in enumerate(cycles[:-1], start=3):
        projection = apply_entry(projection, entry(offset, cycle))
    event = projection.events_by_sequence[1]
    reminder = grounded_reminder(
        cycle=cycles[-1].cycle_id,
        intervention_id=INTERVENTION_1_ID,
        current_event=event,
        evidence=event_reference(EVENT_1_ID),
        created_at=cycles[-1].updated_at,
    )
    committed = committed_with_intervention(cycles[-1], reminder)
    projection = replace(projection, reminder_interventions_by_sequence={1: reminder})

    with pytest.raises(ProjectionInvariantError, match="already has a reminder"):
        apply_entry(projection, entry(6, committed))


def test_complete_projection_requires_reminder_delivery_outbox() -> None:
    with pytest.raises(ProjectionInvariantError, match="missing its delivery outbox"):
        validate_complete_projection(first_reminder_projection())


def test_cycle_grounding_rejects_a_pin_mismatch() -> None:
    _projection, cycles = _cycle_fixture()
    changed = cycles[0].model_copy(update={"grounding_version": "wrong/1"})

    with pytest.raises(ProjectionInvariantError, match="grounding pin"):
        _resolved_cycle_grounding(changed)


def test_authoritative_grounding_requires_current_event() -> None:
    _projection, cycles = _cycle_fixture()
    intervention = cycles[-1].intervention
    assert intervention is not None

    with pytest.raises(ProjectionInvariantError, match="authoritative verification"):
        _authoritatively_verify_intervention(
            empty_projection(RUN_ID),
            cycles[-1],
            intervention,
        )


def test_event_projection_rejects_preexisting_sequence() -> None:
    event = trace_event(1, EVENT_1_ID)
    projection = replace(empty_projection(RUN_ID), events_by_sequence={1: event})

    with pytest.raises(ProjectionInvariantError, match="sequence already exists"):
        apply_entry(projection, entry(1, event))


def _validate_budget(changed: object, previous: object) -> None:
    projection, _cycles = _cycle_fixture()
    decision = projection.decisions[DECISION_1_ID]
    _validate_cycle_budget(changed, decision, projection, previous)  # type: ignore[arg-type]


def test_cycle_budget_reservation_requires_model_call() -> None:
    _projection, cycles = _cycle_fixture()
    reservation = cycles[1].budget_reservation
    assert reservation is not None
    changed = cycles[1].model_copy(
        update={"budget_reservation": reservation.model_copy(update={"model_calls": 0})}
    )

    with pytest.raises(ProjectionInvariantError, match="requires a model call"):
        _validate_budget(changed, cycles[0])


def test_cycle_budget_requires_paired_call_receipts() -> None:
    _projection, cycles = _cycle_fixture()
    changed = cycles[-1].model_copy(update={"model_call_latencies_us": ()})

    with pytest.raises(ProjectionInvariantError, match="receipts are incomplete"):
        _validate_budget(changed, cycles[-2])


def test_cycle_budget_rejects_latency_above_settlement() -> None:
    _projection, cycles = _cycle_fixture()
    changed = cycles[-1].model_copy(update={"model_call_latencies_us": (2_000,)})

    with pytest.raises(ProjectionInvariantError, match="exceeds settled"):
        _validate_budget(changed, cycles[-2])


@pytest.mark.parametrize(
    ("updates", "previous_index", "message"),
    [
        (
            {
                "state": CycleState.FAILED,
                "failure_reason": ReasonCode.MODEL_ERROR,
                "intervention": None,
                "budget_settlement": BudgetAmounts(
                    model_calls=1,
                    input_tokens=50,
                    output_tokens=10,
                    canonical_token_equivalents=60,
                    latency_us=1_000,
                    interventions=1,
                ),
            },
            2,
            "cannot consume an intervention",
        ),
        (
            {
                "state": CycleState.FAILED,
                "failure_reason": ReasonCode.MODEL_ERROR,
                "intervention": None,
                "budget_settlement": BudgetAmounts(
                    model_calls=1,
                    latency_us=1_000,
                ),
                "model_call_digests": (),
                "model_call_latencies_us": (),
            },
            2,
            "digests do not match",
        ),
        (
            {
                "state": CycleState.FAILED,
                "failure_reason": ReasonCode.MODEL_ERROR,
                "intervention": None,
                "budget_settlement": BudgetAmounts(
                    model_calls=1,
                    input_tokens=1,
                    latency_us=1_000,
                ),
            },
            1,
            "before running cannot consume",
        ),
    ],
)
def test_failed_cycle_budget_rejects_invalid_usage(
    updates: dict[str, object],
    previous_index: int,
    message: str,
) -> None:
    _projection, cycles = _cycle_fixture()
    changed = cycles[-1].model_copy(update=updates)

    with pytest.raises(ProjectionInvariantError, match=message):
        _validate_budget(changed, cycles[previous_index])


def _pending_delivery_fixture() -> tuple[object, DeliveryRecord]:
    projection = first_reminder_projection()
    cycle = next(iter(projection.cycles.values()))
    intervention = next(iter(projection.interventions.values()))
    assert intervention.delivery_target is not None
    assert intervention.rendered_text is not None
    pending = DeliveryRecord(
        delivery_id=DELIVERY_ID,
        run_id=RUN_ID,
        revision=1,
        cycle_id=cycle.cycle_id,
        intervention_id=intervention.intervention_id,
        rendered_text_digest=canonical_digest(intervention.rendered_text),
        target_request_id="projector-coverage",
        target=intervention.delivery_target,
        state=DeliveryState.PENDING,
        attempt_count=0,
        adapter_id="fixture",
        adapter_deduplicates=True,
        adapter_deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        adapter_supports_pre_action=True,
        adapter_contract_version="adapter-contract/v1",
        adapter_capabilities_digest="8" * 64,
        created_at=cycle.updated_at,
        updated_at=cycle.updated_at,
    )
    return projection, pending


def _delivery_projection(projection: object, previous: DeliveryRecord) -> object:
    return replace(
        projection,
        deliveries={previous.delivery_id: previous},
        delivery_history={},
    )


def _apply_forged_delivery(
    projection: object,
    valid: DeliveryRecord,
    forged: DeliveryRecord,
) -> None:
    apply_entry(
        projection,  # type: ignore[arg-type]
        entry(7, valid).model_copy(update={"record": forged}),
    )


def test_delivery_requires_matching_committed_cycle() -> None:
    projection, pending = _pending_delivery_fixture()
    cycle = projection.cycles[pending.cycle_id]  # type: ignore[attr-defined]
    projection = replace(
        projection,
        cycles={pending.cycle_id: cycle.model_copy(update={"state": CycleState.RUNNING})},
    )

    with pytest.raises(ProjectionInvariantError, match="committed cycle"):
        apply_entry(projection, entry(7, pending))


def test_unknown_delivery_retry_requires_new_claim() -> None:
    projection, pending = _pending_delivery_fixture()
    attempting = pending.model_copy(
        update={
            "revision": 3,
            "state": DeliveryState.ATTEMPTING,
            "claim_id": CLAIM_1_ID,
            "attempt_id": ATTEMPT_1_ID,
            "attempt_count": 1,
        }
    )
    unknown = attempting.model_copy(
        update={
            "revision": 4,
            "state": DeliveryState.UNKNOWN,
            "outcome": DeliveryOutcome.UNKNOWN,
            "reason_code": ReasonCode.DELIVERY_UNKNOWN,
        }
    )
    retry = unknown.model_copy(
        update={
            "revision": 5,
            "state": DeliveryState.CLAIMED,
            "attempt_id": None,
        }
    )

    with pytest.raises(ProjectionInvariantError, match="new claim owner"):
        _apply_forged_delivery(_delivery_projection(projection, unknown), pending, retry)


@pytest.mark.parametrize(
    ("previous_state", "candidate_state", "updates", "message"),
    [
        (
            DeliveryState.PENDING,
            DeliveryState.CLAIMED,
            {"claim_id": None, "attempt_id": ATTEMPT_1_ID},
            "invalid delivery claim ownership",
        ),
        (
            DeliveryState.CLAIMED,
            DeliveryState.ATTEMPTING,
            {"claim_id": CLAIM_2_ID, "attempt_id": ATTEMPT_1_ID, "attempt_count": 1},
            "invalid delivery attempt ownership",
        ),
        (
            DeliveryState.ATTEMPTING,
            DeliveryState.DELIVERED,
            {
                "claim_id": CLAIM_2_ID,
                "attempt_id": ATTEMPT_1_ID,
                "attempt_count": 1,
                "outcome": DeliveryOutcome.DELIVERED,
            },
            "completion ownership changed",
        ),
        (
            DeliveryState.PENDING,
            DeliveryState.REJECTED,
            {
                "claim_id": CLAIM_1_ID,
                "outcome": DeliveryOutcome.REFUSED,
                "reason_code": ReasonCode.TARGET_UNAVAILABLE,
            },
            "rejection gained an owner",
        ),
        (
            DeliveryState.CLAIMED,
            DeliveryState.REJECTED,
            {
                "claim_id": CLAIM_2_ID,
                "outcome": DeliveryOutcome.REFUSED,
                "reason_code": ReasonCode.TARGET_UNAVAILABLE,
            },
            "rejection changed owner",
        ),
    ],
)
def test_delivery_transition_rejects_invalid_ownership(
    previous_state: DeliveryState,
    candidate_state: DeliveryState,
    updates: dict[str, object],
    message: str,
) -> None:
    projection, pending = _pending_delivery_fixture()
    previous_updates: dict[str, object] = {"state": previous_state}
    if previous_state in (DeliveryState.CLAIMED, DeliveryState.ATTEMPTING):
        previous_updates["claim_id"] = CLAIM_1_ID
    if previous_state is DeliveryState.ATTEMPTING:
        previous_updates.update(attempt_id=ATTEMPT_1_ID, attempt_count=1)
    previous = pending.model_copy(update=previous_updates)
    candidate = previous.model_copy(
        update={
            "revision": previous.revision + 1,
            "state": candidate_state,
            **updates,
        }
    )

    with pytest.raises(ProjectionInvariantError, match=message):
        _apply_forged_delivery(_delivery_projection(projection, previous), pending, candidate)


def test_complete_projection_rejects_ambiguous_initial_delivery() -> None:
    projection, pending = _pending_delivery_fixture()
    projection = replace(
        projection,
        delivery_history={(CLAIM_1_ID, 1): pending},
    )

    with pytest.raises(ProjectionInvariantError, match="ambiguous initial delivery"):
        validate_complete_projection(projection)


def test_complete_projection_rejects_silent_cycle_delivery() -> None:
    silent = first_committed_projection()
    _reminder_projection, pending = _pending_delivery_fixture()
    cycle = next(iter(silent.cycles.values()))
    changed = pending.model_copy(update={"cycle_id": cycle.cycle_id})
    silent = replace(
        silent,
        delivery_history={(changed.delivery_id, 1): changed},
    )

    with pytest.raises(ProjectionInvariantError, match="silent cycle"):
        validate_complete_projection(silent)


def test_complete_projection_rejects_orphan_delivery_outbox() -> None:
    projection, pending = _pending_delivery_fixture()
    projection = replace(
        projection,
        cycles={},
        delivery_history={(pending.delivery_id, 1): pending},
    )

    with pytest.raises(ProjectionInvariantError, match="does not match"):
        validate_complete_projection(projection)
