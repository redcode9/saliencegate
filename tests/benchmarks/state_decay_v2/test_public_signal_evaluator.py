from __future__ import annotations

import ast
import inspect

import pytest

from saliencegate.benchmarks.state_decay_v2.public_contract import (
    OutcomeFreeAllowedAction,
    OutcomeFreeCandidateMemory,
    OutcomeFreeEvent,
    OutcomeFreeEvidenceReference,
    OutcomeFreePivot,
    OutcomeFreeTaskSkeleton,
    PublicCounterbalanceProfile,
    PublicEvidenceProfile,
    PublicExpectedAssertionEvidence,
    PublicExpectedDetectorEvidence,
    PublicExpectedMemoryEvidence,
    PublicExpectedSignal,
    PublicGeneratorSlot,
    PublicIntegerProfile,
    PublicParameterProfile,
    PublicParameterValue,
    PublicSignalFixtureVariant,
    PublicSignalProfile,
    PublicSlotProfile,
    PublicStructuralProfile,
    PublicTextLengthProfile,
    signal_profile_digest,
    trace_fixture_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import AdapterMetadata
from saliencegate.benchmarks.state_decay_v2.signal_fixtures import (
    PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN,
    LegacyDetectorResult,
    LegacyFixtureEvaluation,
    ReferencePredicateAbstentionReason,
    ReferencePredicateStatus,
    SignalFixtureInputError,
    detected_signal_projection,
    evaluate_legacy_signal_fixture,
    evaluate_reference_predicates,
    materialize_public_trace_fixture,
    public_assertion_fixture_digest,
)
from saliencegate.domain import PayloadDigestAlgorithm, SignalType, ValidityState, canonical_json
from saliencegate.signals.base import DetectionStatus


def _evidence(
    *events: int,
    bindings: tuple[int, ...] = (),
    memories: tuple[tuple[int, int], ...] = (),
    assertions: tuple[tuple[int, int], ...] = (),
) -> PublicExpectedDetectorEvidence:
    return PublicExpectedDetectorEvidence(
        event_pool_indices=events,
        binding_event_pool_indices=bindings,
        memory_references=tuple(
            PublicExpectedMemoryEvidence(memory_pool_index=index, revision=revision)
            for index, revision in memories
        ),
        assertion_references=tuple(
            PublicExpectedAssertionEvidence(
                binding_event_pool_index=event_index,
                assertion_index=assertion_index,
            )
            for event_index, assertion_index in assertions
        ),
    )


def _signal_profile(
    slot: int,
    variant: PublicSignalFixtureVariant,
    expected: tuple[PublicExpectedSignal, ...],
) -> PublicSignalProfile:
    values: dict[str, object] = {
        "profile_id": f"signals-slot-{slot}",
        "fixture_variant": variant,
        "expected_signals": expected,
    }
    values["profile_digest"] = signal_profile_digest(values)
    return PublicSignalProfile.model_validate(values)


def _signal_profiles() -> tuple[PublicSignalProfile, ...]:
    return (
        _signal_profile(
            0,
            PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, bindings=(2,), assertions=((2, 0), (2, 1))),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, bindings=(2,), memories=((1, 1),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TEST_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2),
                ),
            ),
        ),
        _signal_profile(
            1,
            PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(2, 3, bindings=(2, 3), memories=((0, 2),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.IRREVERSIBLE_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(3, bindings=(3,)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 3),
                ),
            ),
        ),
        _signal_profile(
            2,
            PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(4, bindings=(4,), assertions=((4, 0), (4, 1))),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=500_000,
                    evidence=_evidence(1, 2, 3, 4, bindings=(1, 2, 3, 4)),
                ),
            ),
        ),
        _signal_profile(
            3,
            PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 3, 4, 5),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=750_000,
                    evidence=_evidence(5, bindings=(5,), memories=((0, 4),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TOOL_ERROR,
                    strength_ppm=1_000_000,
                    evidence=_evidence(5),
                ),
            ),
        ),
        _signal_profile(
            4,
            PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(5, 6, bindings=(5, 6), memories=((0, 5),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 6),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=625_000,
                    evidence=_evidence(2, 3, 4, 5, 6, bindings=(2, 3, 4, 5, 6)),
                ),
            ),
        ),
    )


def _slot_profile(slot: PublicGeneratorSlot) -> PublicSlotProfile:
    inactive = slot == 3
    return PublicSlotProfile(
        generator_slot=slot,
        counterbalance=PublicCounterbalanceProfile(
            profile_id=f"counterbalance-slot-{slot}",
            allowed_action_order=((0, 1) if slot % 2 == 0 else (1, 0)),
            decisive_action_position=(0 if slot % 2 == 0 else 1),
            memory_validity=(ValidityState.SUPERSEDED if inactive else ValidityState.ACTIVE),
            include_validity_transition=inactive,
        ),
        parameters=PublicParameterProfile(
            profile_id=f"parameters-slot-{slot}",
            allowed_values=(PublicParameterValue(parameter_id=f"parameter-{slot}", value=slot),),
        ),
        structure=PublicStructuralProfile(
            profile_id=f"structure-slot-{slot}",
            trajectory_event_count=3 + slot,
            candidate_memory_count=1,
        ),
        integers=PublicIntegerProfile(
            profile_id=f"integers-slot-{slot}",
            sequence_start=1,
            sequence_stride=1,
            action_step_start=0,
            action_step_stride=1,
            memory_revision=1 + slot,
        ),
        evidence=PublicEvidenceProfile(
            profile_id=f"evidence-slot-{slot}",
            evidence_reference_count=1,
            decisive_event_count=1,
            decisive_memory_count=1,
        ),
        text_lengths=PublicTextLengthProfile(
            profile_id=f"text-slot-{slot}",
            event_padding_spaces=slot,
            memory_padding_spaces=slot,
            pivot_padding_spaces=slot,
            action_padding_spaces=slot,
        ),
        signals=_signal_profiles()[slot],
    )


def _skeleton(profile: PublicSlotProfile) -> OutcomeFreeTaskSkeleton:
    trajectory = tuple(
        OutcomeFreeEvent(
            event_id=f"event-{index}",
            sequence=profile.integers.sequence_start + index,
            action_step=profile.integers.action_step_start + index,
            statement=f"Repository event statement {index}.",
        )
        for index in range(profile.structure.trajectory_event_count)
    )
    final = trajectory[-1]
    validity = profile.counterbalance.memory_validity
    return OutcomeFreeTaskSkeleton(
        trajectory=trajectory,
        candidate_memories=(
            OutcomeFreeCandidateMemory(
                memory_id="memory-0",
                revision=profile.integers.memory_revision,
                statement="The retained constraint remains available.",
                evidence_refs=(
                    OutcomeFreeEvidenceReference(
                        event_id=trajectory[0].event_id,
                        event_sequence=trajectory[0].sequence,
                    ),
                ),
                recorded_sequence=trajectory[0].sequence,
                recorded_action_step=trajectory[0].action_step,
                validity=validity,
                validity_sequence=(None if validity is ValidityState.ACTIVE else final.sequence),
                validity_action_step=(
                    None if validity is ValidityState.ACTIVE else final.action_step
                ),
            ),
        ),
        pivot=OutcomeFreePivot(
            event_id="pivot",
            sequence=final.sequence + 1,
            action_step=final.action_step,
            statement="Choose exactly one next action.",
        ),
        allowed_actions=(
            OutcomeFreeAllowedAction(action_id="action-a", statement="Apply action alpha."),
            OutcomeFreeAllowedAction(action_id="action-b", statement="Apply action bravo."),
        ),
        adapter=AdapterMetadata(
            adapter_id="public-fixture-adapter",
            adapter_version="v1",
            response_profile_id="two-action-choice",
            response_profile_digest="9" * 64,
        ),
    )


def _inputs() -> tuple[tuple[PublicSlotProfile, OutcomeFreeTaskSkeleton], ...]:
    profiles = tuple(_slot_profile(slot) for slot in range(5))
    return tuple((profile, _skeleton(profile)) for profile in profiles)


def _redigest_fixture(fixture: object) -> object:
    assert isinstance(fixture, dict)
    fixture["trace_fixture_digest"] = trace_fixture_digest(fixture)
    return fixture


def test_materializer_builds_five_self_attesting_raw_recipes_without_shortcuts() -> None:
    fixtures = tuple(
        materialize_public_trace_fixture(profile, skeleton) for profile, skeleton in _inputs()
    )

    assert tuple(len(fixture.events) for fixture in fixtures) == (3, 4, 5, 6, 7)
    assert fixtures[0].events[-1].payload.keys() == {"test_report"}
    assert fixtures[3].events[-1].payload.keys() == {"tool_outcome"}
    assert tuple(fixtures[3].events[index].event_type.value for index in range(2, 6)) == (
        "action_proposal",
        "tool_completion",
        "action_proposal",
        "tool_completion",
    )
    assert all(
        event.event_type.value == "action_proposal"
        and event.phase.value == "pre_action"
        and set(event.payload) == {"action"}
        for event in fixtures[2].events[1:]
    )
    assert fixtures[0].bindings[0].constraint_references is None
    assert fixtures[0].bindings[1].constraint_references == ()
    assert fixtures[0].trace_fixture_digest == trace_fixture_digest(fixtures[0])
    assert tuple(fixture.trace_fixture_digest for fixture in fixtures) == (
        "7acbc3a3aa08f08d94688159f9b23cd17ed996d7d9b36034ad0742edd7c9cf1f",
        "c77f66be98d4628b98535a6aa48e4c028bd69cf7de8c63a44dbef8a3917e1493",
        "e3961039ef4e3cda5346ae5f282fdf3da2a5ef190ae3a61b1edeff2f8664b471",
        "c0522c8b7cdded2c844614e46f15620fce974ab67977c146ea7b80ddc566fb70",
        "c86792ef8c7409b25deeebfe44e8164e8f410c174d32905c56d44befe722bfc3",
    )
    assert fixtures == tuple(
        materialize_public_trace_fixture(profile, skeleton) for profile, skeleton in _inputs()
    )
    for fixture in fixtures:
        encoded = canonical_json(fixture)
        for forbidden in (
            b'"profile_id"',
            b'"fixture_variant"',
            b'"expected_signals"',
            b'"signal_type"',
            b'"strength_ppm"',
            b'"helpful"',
            b'"harmful"',
            b'"redundant"',
            b'"unresolved"',
        ):
            assert forbidden not in encoded


def test_reference_evaluator_reproduces_five_exact_raw_only_projections() -> None:
    for profile, skeleton in _inputs():
        fixture = materialize_public_trace_fixture(profile, skeleton)
        results = evaluate_reference_predicates(fixture, skeleton)
        projection = tuple(
            signal
            for signal in detected_signal_projection(reference_results=results)
            if signal.signal_type
            in {
                SignalType.CONFLICT,
                SignalType.CONTEXT_SHIFT,
                SignalType.IRREVERSIBLE_ACTION,
                SignalType.STAGNATION,
                SignalType.STALE_CONSTRAINT,
            }
        )
        expected = tuple(
            signal
            for signal in profile.signals.expected_signals
            if signal.signal_type
            in {
                SignalType.CONFLICT,
                SignalType.CONTEXT_SHIFT,
                SignalType.IRREVERSIBLE_ACTION,
                SignalType.STAGNATION,
                SignalType.STALE_CONSTRAINT,
            }
        )
        assert projection == expected
        assert all(
            result.status is ReferencePredicateStatus.NO_MATCH
            for result in results
            if result.signal_type not in {signal.signal_type for signal in expected}
        )

    parameters = tuple(inspect.signature(evaluate_reference_predicates).parameters)
    assert parameters == ("fixture", "task_skeleton")
    assert "expected" not in inspect.getsource(evaluate_reference_predicates)


@pytest.mark.parametrize(
    ("slot", "signal_type", "mutate", "expected_status"),
    (
        (
            0,
            SignalType.CONFLICT,
            lambda payload: payload["bindings"][2]["assertions"][1].__setitem__("precedence", 0),
            ReferencePredicateStatus.NO_MATCH,
        ),
        (
            0,
            SignalType.STALE_CONSTRAINT,
            lambda payload: payload["bindings"][2].__setitem__("constraint_references", ()),
            ReferencePredicateStatus.NO_MATCH,
        ),
        (
            1,
            SignalType.CONTEXT_SHIFT,
            lambda payload: payload["memories"][0].__setitem__(
                "provenance_event_pool_indices", (3,)
            ),
            ReferencePredicateStatus.NO_MATCH,
        ),
        (
            1,
            SignalType.IRREVERSIBLE_ACTION,
            lambda payload: payload["bindings"][3].__setitem__(
                "authorization_event_pool_indices", (3,)
            ),
            ReferencePredicateStatus.ABSTAINED,
        ),
        (
            2,
            SignalType.STAGNATION,
            lambda payload: payload["bindings"][3].__setitem__("progress_marker_digest", "e" * 64),
            ReferencePredicateStatus.NO_MATCH,
        ),
    ),
)
def test_reference_predicates_require_their_structured_operands(
    slot: int,
    signal_type: SignalType,
    mutate: object,
    expected_status: ReferencePredicateStatus,
) -> None:
    profile, skeleton = _inputs()[slot]
    fixture = materialize_public_trace_fixture(profile, skeleton)
    payload = fixture.model_dump(mode="python")
    assert callable(mutate)
    mutate(payload)
    mutated = type(fixture).model_validate(_redigest_fixture(payload))

    result = next(
        item
        for item in evaluate_reference_predicates(mutated, skeleton)
        if item.signal_type is signal_type
    )
    assert result.status is expected_status


@pytest.mark.parametrize(("slot", "revision"), ((1, 1), (3, 3)))
def test_noncurrent_constraint_coordinate_is_missing_at_full_strength(
    slot: int,
    revision: int,
) -> None:
    profile, skeleton = _inputs()[slot]
    fixture = materialize_public_trace_fixture(profile, skeleton)
    payload = fixture.model_dump(mode="python")
    payload["bindings"][-1]["constraint_references"] = (
        {"memory_pool_index": 0, "revision": revision},
    )
    mutated = type(fixture).model_validate(_redigest_fixture(payload))

    result = next(
        item
        for item in evaluate_reference_predicates(mutated, skeleton)
        if item.signal_type is SignalType.STALE_CONSTRAINT
    )
    assert result.status is ReferencePredicateStatus.DETECTED
    assert result.strength_ppm == 1_000_000
    assert result.evidence == _evidence(
        len(fixture.events) - 1,
        bindings=(len(fixture.events) - 1,),
        memories=((0, revision),),
    )


def test_assertion_supersession_uses_the_frozen_assertion_identity() -> None:
    profile, skeleton = _inputs()[0]
    fixture = materialize_public_trace_fixture(profile, skeleton)
    predecessor = fixture.bindings[-1].assertions
    assert predecessor is not None
    predecessor_digest = public_assertion_fixture_digest(predecessor[0])
    assert PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN.endswith(":assertion-fixture:v1")
    assert predecessor_digest == "dd26276ead0c34e1017c36bb6929e15b68ff92eac82ca187a6fc72f69e3fb089"

    payload = fixture.model_dump(mode="python")
    payload["bindings"][-1]["assertions"][1]["supersedes_assertion_digest"] = predecessor_digest
    superseded = type(fixture).model_validate(_redigest_fixture(payload))
    result = next(
        item
        for item in evaluate_reference_predicates(superseded, skeleton)
        if item.signal_type is SignalType.CONFLICT
    )
    assert result.status is ReferencePredicateStatus.NO_MATCH

    payload = fixture.model_dump(mode="python")
    payload["bindings"][-1]["assertions"][1]["supersedes_assertion_digest"] = "f" * 64
    unresolved = type(fixture).model_validate(_redigest_fixture(payload))
    result = next(
        item
        for item in evaluate_reference_predicates(unresolved, skeleton)
        if item.signal_type is SignalType.CONFLICT
    )
    assert result.status is ReferencePredicateStatus.ABSTAINED
    assert result.abstention_reason is ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED


@pytest.mark.parametrize(
    ("slot", "signal_type", "field", "value"),
    (
        (1, SignalType.CONTEXT_SHIFT, "scope_id", None),
        (0, SignalType.STALE_CONSTRAINT, "constraint_references", None),
        (2, SignalType.STAGNATION, "progress_marker_digest", None),
        (1, SignalType.IRREVERSIBLE_ACTION, "impact", None),
        (0, SignalType.CONFLICT, "assertions", None),
    ),
)
def test_reference_predicates_abstain_when_required_capability_is_unavailable(
    slot: int,
    signal_type: SignalType,
    field: str,
    value: object,
) -> None:
    profile, skeleton = _inputs()[slot]
    fixture = materialize_public_trace_fixture(profile, skeleton)
    payload = fixture.model_dump(mode="python")
    payload["bindings"][-1][field] = value
    mutated = type(fixture).model_validate(_redigest_fixture(payload))

    result = next(
        item
        for item in evaluate_reference_predicates(mutated, skeleton)
        if item.signal_type is signal_type
    )
    assert result.status is ReferencePredicateStatus.ABSTAINED
    assert (
        result.abstention_reason
        is ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE
    )


def test_reference_evaluator_rejects_a_skeleton_fixture_state_mismatch() -> None:
    profile, skeleton = _inputs()[1]
    fixture = materialize_public_trace_fixture(profile, skeleton)
    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"][0]["revision"] = 99
    mismatched = OutcomeFreeTaskSkeleton.model_validate(payload)

    with pytest.raises(SignalFixtureInputError):
        evaluate_reference_predicates(fixture, mismatched)


@pytest.mark.asyncio
async def test_repository_adapter_executes_only_four_real_detectors_at_final_boundary() -> None:
    expected_detected = (
        (SignalType.TEST_FAILURE,),
        (SignalType.REPEATED_ACTION,),
        (),
        (SignalType.REPEATED_FAILURE, SignalType.TOOL_ERROR),
        (SignalType.REPEATED_ACTION,),
    )
    expected_statuses = (
        ("abstained", "no_match", "detected", "abstained"),
        ("detected", "abstained", "abstained", "abstained"),
        ("no_match", "abstained", "abstained", "abstained"),
        ("abstained", "detected", "abstained", "detected"),
        ("detected", "abstained", "abstained", "abstained"),
    )
    expected_abstentions = (
        ("event_not_applicable", None, None, "structured_evidence_invalid"),
        (None, "event_not_applicable", "event_not_applicable", "event_not_applicable"),
        (None, "event_not_applicable", "event_not_applicable", "event_not_applicable"),
        ("event_not_applicable", None, "structured_evidence_invalid", None),
        (None, "event_not_applicable", "event_not_applicable", "event_not_applicable"),
    )

    for slot, (profile, skeleton) in enumerate(_inputs()):
        fixture = materialize_public_trace_fixture(profile, skeleton)
        evaluation = await evaluate_legacy_signal_fixture(
            fixture,
            scenario_id=f"{slot + 1:x}" * 64,
        )
        detected = tuple(
            result.signal_type for result in evaluation.results if result.status.value == "detected"
        )
        assert detected == expected_detected[slot]
        assert (
            tuple(result.status.value for result in evaluation.results) == expected_statuses[slot]
        )
        assert (
            tuple(
                None if result.abstention_reason is None else result.abstention_reason.value
                for result in evaluation.results
            )
            == expected_abstentions[slot]
        )
        assert tuple(event.sequence for event in evaluation.events) == tuple(
            range(1, len(fixture.events) + 1)
        )
        assert all(
            event.payload_digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
            for event in evaluation.events
        )
        assert evaluation.events[-1].event_type is fixture.events[-1].event_type
        assert set(evaluation.events[-1].payload) == set(fixture.events[-1].payload)

        repeated = await evaluate_legacy_signal_fixture(
            fixture,
            scenario_id=f"{slot + 1:x}" * 64,
        )
        assert tuple(event.event_id for event in repeated.events) == tuple(
            event.event_id for event in evaluation.events
        )
        assert repeated.results == evaluation.results
        if slot == 0:
            with pytest.raises(SignalFixtureInputError):
                detected_signal_projection(legacy_results=evaluation.results[:1])
            with pytest.raises(ValueError, match="issuer"):
                LegacyDetectorResult(
                    _issuer=object(),
                    signal_type=SignalType.TEST_FAILURE,
                    detector_version="forged/v1",
                    status=DetectionStatus.DETECTED,
                    strength_ppm=1_000_000,
                    evidence_event_pool_indices=(2,),
                    related_event_pool_indices=(),
                    abstention_reason=None,
                )
            with pytest.raises(ValueError, match="issuer"):
                LegacyFixtureEvaluation(
                    _issuer=object(),
                    events=evaluation.events,
                    results=evaluation.results,
                )


@pytest.mark.asyncio
async def test_combined_real_and_reference_projection_matches_each_frozen_profile() -> None:
    for slot, (profile, skeleton) in enumerate(_inputs()):
        fixture = materialize_public_trace_fixture(profile, skeleton)
        legacy = await evaluate_legacy_signal_fixture(
            fixture,
            scenario_id=f"{slot + 10:x}" * 64,
        )
        reference = evaluate_reference_predicates(fixture, skeleton)

        assert (
            detected_signal_projection(
                legacy_results=legacy.results,
                reference_results=reference,
            )
            == profile.signals.expected_signals
        )


def test_signal_fixture_module_never_constructs_trace_events_directly() -> None:
    import saliencegate.benchmarks.state_decay_v2.signal_fixtures as module

    tree = ast.parse(inspect.getsource(module))
    forbidden_calls = {
        "TraceEvent",
        "TraceEvent.model_validate",
        "TraceEvent.model_validate_json",
    }
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert calls.isdisjoint(forbidden_calls)
    assert "DetectionOutcome(" not in inspect.getsource(module)
