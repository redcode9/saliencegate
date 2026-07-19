from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

import saliencegate.benchmarks.state_decay_v2.signal_fixtures as signal_fixtures
from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATION_CONTRACT,
    CounterbalanceAxis,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    SIGNAL_PROFILE_DIGEST_DOMAIN,
    TRACE_FIXTURE_DIGEST_DOMAIN,
    OutcomeFreeTraceFixture,
    PublicAssertionFixture,
    PublicBindingFixture,
    PublicConstraintReferenceFixture,
    PublicCounterbalanceProfile,
    PublicDetectorMemoryFixture,
    PublicEvidenceProfile,
    PublicExpectedAssertionEvidence,
    PublicExpectedDetectorEvidence,
    PublicExpectedMemoryEvidence,
    PublicExpectedSignal,
    PublicFixtureEvent,
    PublicImpactClass,
    PublicIntegerProfile,
    PublicParameterProfile,
    PublicParameterValue,
    PublicProfileCatalog,
    PublicSignalFixtureVariant,
    PublicSignalProfile,
    PublicSlotProfile,
    PublicStructuralProfile,
    PublicTextLengthProfile,
    profile_catalog_digest,
    signal_profile_digest,
    trace_fixture_digest,
)
from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY
from saliencegate.domain import (
    ClaimKind,
    EventPhase,
    EventType,
    SignalType,
    TraceEvent,
    ValidityState,
    canonical_json,
)
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionOutcome,
    DetectionStatus,
)


def _evidence(
    *event_indices: int,
    binding_indices: tuple[int, ...] = (),
    memory_references: tuple[PublicExpectedMemoryEvidence, ...] = (),
    assertion_references: tuple[PublicExpectedAssertionEvidence, ...] = (),
) -> PublicExpectedDetectorEvidence:
    return PublicExpectedDetectorEvidence(
        event_pool_indices=event_indices,
        binding_event_pool_indices=binding_indices,
        memory_references=memory_references,
        assertion_references=assertion_references,
    )


def _profile(
    *,
    slot: int,
    variant: PublicSignalFixtureVariant,
    signals: tuple[PublicExpectedSignal, ...],
) -> PublicSignalProfile:
    values: dict[str, object] = {
        "profile_id": f"signals-slot-{slot}",
        "fixture_variant": variant,
        "expected_signals": signals,
    }
    values["profile_digest"] = signal_profile_digest(values)
    return PublicSignalProfile.model_validate(values)


def _five_profiles() -> tuple[PublicSignalProfile, ...]:
    return (
        _profile(
            slot=0,
            variant=PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(
                        2,
                        binding_indices=(2,),
                        assertion_references=(
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=2,
                                assertion_index=0,
                            ),
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=2,
                                assertion_index=1,
                            ),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(
                        2,
                        binding_indices=(2,),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=1, revision=1),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TEST_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2),
                ),
            ),
        ),
        _profile(
            slot=1,
            variant=PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(
                        2,
                        3,
                        binding_indices=(2, 3),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=2),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.IRREVERSIBLE_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(3, binding_indices=(3,)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 3),
                ),
            ),
        ),
        _profile(
            slot=2,
            variant=PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(
                        4,
                        binding_indices=(4,),
                        assertion_references=(
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=4,
                                assertion_index=0,
                            ),
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=4,
                                assertion_index=1,
                            ),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=500_000,
                    evidence=_evidence(1, 2, 3, 4, binding_indices=(1, 2, 3, 4)),
                ),
            ),
        ),
        _profile(
            slot=3,
            variant=PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 3, 4, 5),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=750_000,
                    evidence=_evidence(
                        5,
                        binding_indices=(5,),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=4),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TOOL_ERROR,
                    strength_ppm=1_000_000,
                    evidence=_evidence(5),
                ),
            ),
        ),
        _profile(
            slot=4,
            variant=PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(
                        5,
                        6,
                        binding_indices=(5, 6),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=5),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(2, 6),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=625_000,
                    evidence=_evidence(
                        2,
                        3,
                        4,
                        5,
                        6,
                        binding_indices=(2, 3, 4, 5, 6),
                    ),
                ),
            ),
        ),
    )


def _trace_fixture() -> OutcomeFreeTraceFixture:
    events = (
        PublicFixtureEvent(
            event_pool_index=0,
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            payload={},
            parent_event_pool_indices=(),
        ),
        PublicFixtureEvent(
            event_pool_index=1,
            event_type=EventType.ACTION_PROPOSAL,
            phase=EventPhase.PRE_ACTION,
            payload={
                "action": {
                    "schema_version": "1.0",
                    "kind": "shell",
                    "command": "pytest -q",
                    "working_directory": "/workspace",
                    "environment_digest": "a" * 64,
                }
            },
            parent_event_pool_indices=(),
        ),
        PublicFixtureEvent(
            event_pool_index=2,
            event_type=EventType.TOOL_COMPLETION,
            phase=EventPhase.POST_ACTION,
            payload={
                "test_report": {
                    "schema_version": "1.0",
                    "framework": "pytest",
                    "status": "failed",
                    "failures": (),
                },
            },
            parent_event_pool_indices=(1,),
        ),
    )
    bindings = (
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
        ),
        PublicBindingFixture(
            event_pool_index=1,
            action_step=1,
            scope_id="scope-a",
            progress_marker_digest="b" * 64,
            constraint_references=(),
            impact=PublicImpactClass.REVERSIBLE,
            authorization_event_pool_indices=(),
            safeguard_event_pool_indices=(),
            assertions=(),
        ),
        PublicBindingFixture(
            event_pool_index=2,
            action_step=1,
            scope_id="scope-a",
            progress_marker_digest="b" * 64,
            constraint_references=(
                PublicConstraintReferenceFixture(memory_pool_index=1, revision=1),
            ),
            impact=PublicImpactClass.REVERSIBLE,
            authorization_event_pool_indices=(),
            safeguard_event_pool_indices=(),
            assertions=(
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
            ),
        ),
    )
    memories = (
        PublicDetectorMemoryFixture(
            memory_pool_index=0,
            kind=ClaimKind.REQUIREMENT,
            current_revision=1,
            validity=ValidityState.ACTIVE,
            provenance_event_pool_indices=(0,),
            expires_at_event_pool_index=None,
        ),
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-outcome-free-trace-fixture/v1",
        "events": events,
        "bindings": bindings,
        "memories": memories,
    }
    values["trace_fixture_digest"] = trace_fixture_digest(values)
    return OutcomeFreeTraceFixture.model_validate(values)


def _slot_profile(slot: int) -> PublicSlotProfile:
    return PublicSlotProfile(
        generator_slot=slot,
        counterbalance=PublicCounterbalanceProfile(
            profile_id=f"signal-counterbalance-{slot}",
            allowed_action_order=((0, 1) if slot % 2 == 0 else (1, 0)),
            decisive_action_position=(0 if slot % 2 == 0 else 1),
            memory_validity=(ValidityState.SUPERSEDED if slot == 3 else ValidityState.ACTIVE),
            include_validity_transition=slot == 3,
        ),
        parameters=PublicParameterProfile(
            profile_id=f"signal-parameters-{slot}",
            allowed_values=(
                PublicParameterValue(parameter_id=f"signal-parameter-{slot}", value=slot),
            ),
        ),
        structure=PublicStructuralProfile(
            profile_id=f"signal-structure-{slot}",
            trajectory_event_count=3 + slot,
            candidate_memory_count=1,
        ),
        integers=PublicIntegerProfile(
            profile_id=f"signal-integers-{slot}",
            sequence_start=1,
            sequence_stride=1,
            action_step_start=0,
            action_step_stride=1,
            memory_revision=1 + slot,
        ),
        evidence=PublicEvidenceProfile(
            profile_id=f"signal-evidence-{slot}",
            evidence_reference_count=1,
            decisive_event_count=1,
            decisive_memory_count=1,
        ),
        text_lengths=PublicTextLengthProfile(
            profile_id=f"signal-text-{slot}",
            event_padding_spaces=slot,
            memory_padding_spaces=slot,
            pivot_padding_spaces=slot,
            action_padding_spaces=slot,
        ),
        signals=_five_profiles()[slot],
    )


def _catalog() -> PublicProfileCatalog:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-profile-catalog/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": "1" * 64,
        "counterbalance_axes": tuple(CounterbalanceAxis),
        "slot_profiles": tuple(_slot_profile(slot) for slot in range(5)),
    }
    values["catalog_digest"] = profile_catalog_digest(values)
    return PublicProfileCatalog.model_validate(values)


def _legacy_result(
    *,
    signal_type: SignalType = SignalType.REPEATED_ACTION,
    detector_version: str = "coverage-probe/v1",
    status: DetectionStatus = DetectionStatus.NO_MATCH,
    strength_ppm: int | None = None,
    evidence_indices: tuple[int, ...] = (),
    related_indices: tuple[int, ...] = (0,),
    abstention_reason: AbstentionReason | None = None,
    issuer: object = signal_fixtures._LEGACY_RESULT_ISSUER,
) -> signal_fixtures.LegacyDetectorResult:
    return signal_fixtures.LegacyDetectorResult(
        _issuer=issuer,
        signal_type=signal_type,
        detector_version=detector_version,
        status=status,
        strength_ppm=strength_ppm,
        evidence_event_pool_indices=evidence_indices,
        related_event_pool_indices=related_indices,
        abstention_reason=abstention_reason,
    )


def _replace_binding(
    fixture: OutcomeFreeTraceFixture,
    index: int,
    **updates: object,
) -> OutcomeFreeTraceFixture:
    bindings = list(fixture.bindings)
    bindings[index] = bindings[index].model_copy(update=updates)
    return fixture.model_copy(update={"bindings": tuple(bindings)})


def _replace_event(
    fixture: OutcomeFreeTraceFixture,
    index: int,
    **updates: object,
) -> OutcomeFreeTraceFixture:
    events = list(fixture.events)
    events[index] = events[index].model_copy(update=updates)
    return fixture.model_copy(update={"events": tuple(events)})


def test_signal_profiles_cover_nine_types_with_distinct_masks_and_strengths() -> None:
    profiles = _five_profiles()

    assert SIGNAL_PROFILE_DIGEST_DOMAIN.endswith(":signal-profile:v1")
    assert tuple(profile.fixture_variant for profile in profiles) == tuple(
        PublicSignalFixtureVariant
    )
    masks = tuple(
        tuple(signal.signal_type for signal in profile.expected_signals) for profile in profiles
    )
    assert len(set(masks)) == 5
    assert {signal for mask in masks for signal in mask} == set(SignalType)
    assert {signal.strength_ppm for profile in profiles for signal in profile.expected_signals} == {
        500_000,
        625_000,
        750_000,
        1_000_000,
    }
    for profile in profiles:
        assert tuple(signal.signal_type.value for signal in profile.expected_signals) == tuple(
            sorted(signal.signal_type.value for signal in profile.expected_signals)
        )
        assert PublicSignalProfile.model_validate_json(canonical_json(profile)) == profile


def test_signal_profile_rejects_noncanonical_or_duplicate_signals_and_digest_tamper() -> None:
    profile = _five_profiles()[0]
    payload = profile.model_dump(mode="python")
    payload["expected_signals"] = tuple(reversed(payload["expected_signals"]))
    payload["profile_digest"] = signal_profile_digest(payload)
    with pytest.raises(ValidationError, match="canonical"):
        PublicSignalProfile.model_validate(payload)

    payload = profile.model_dump(mode="python")
    payload["expected_signals"] = (
        payload["expected_signals"][0],
        payload["expected_signals"][0],
    )
    payload["profile_digest"] = signal_profile_digest(payload)
    with pytest.raises(ValidationError, match=r"unique|canonical"):
        PublicSignalProfile.model_validate(payload)

    payload = profile.model_dump(mode="python")
    payload["profile_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="profile digest"):
        PublicSignalProfile.model_validate(payload)

    payload = profile.model_dump(mode="python")
    payload["fixture_variant"] = PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE
    payload["profile_digest"] = signal_profile_digest(payload)
    with pytest.raises(ValidationError, match=r"fixture|variant|mask"):
        PublicSignalProfile.model_validate(payload)

    invalid_strength = profile.expected_signals[0].model_copy(update={"strength_ppm": 500_000})
    with pytest.raises(ValidationError, match="strength"):
        PublicExpectedSignal.model_validate(invalid_strength)


def test_slot_and_catalog_bind_complete_signal_coverage_without_selectors() -> None:
    catalog = _catalog()

    assert tuple(PublicSlotProfile.model_fields)[-1] == "signals"
    assert tuple(profile.signals.fixture_variant for profile in catalog.slot_profiles) == tuple(
        PublicSignalFixtureVariant
    )

    payload = catalog.model_dump(mode="python")
    payload["slot_profiles"][4]["signals"]["fixture_variant"] = payload["slot_profiles"][3][
        "signals"
    ]["fixture_variant"]
    payload["slot_profiles"][4]["signals"]["expected_signals"] = payload["slot_profiles"][3][
        "signals"
    ]["expected_signals"]
    payload["slot_profiles"][4]["signals"]["profile_digest"] = signal_profile_digest(
        payload["slot_profiles"][4]["signals"]
    )
    payload["catalog_digest"] = profile_catalog_digest(payload)
    with pytest.raises(ValidationError, match=r"signal|mask|variant|coverage"):
        PublicProfileCatalog.model_validate(payload)

    profile_payload = catalog.slot_profiles[0].model_dump(mode="python")
    profile_payload["signals"]["expected_signals"][0]["evidence"]["event_pool_indices"] = (7,)
    profile_payload["signals"]["expected_signals"][0]["evidence"]["binding_event_pool_indices"] = ()
    profile_payload["signals"]["profile_digest"] = signal_profile_digest(profile_payload["signals"])
    with pytest.raises(ValidationError, match=r"signal.*trajectory|evidence"):
        PublicSlotProfile.model_validate(profile_payload)


def test_expected_evidence_is_structured_nonempty_and_canonical() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        PublicExpectedDetectorEvidence(
            event_pool_indices=(),
            binding_event_pool_indices=(),
            memory_references=(),
            assertion_references=(),
        )

    with pytest.raises(ValidationError, match=r"canonical|unique"):
        _evidence(2, 1)

    with pytest.raises(ValidationError):
        PublicExpectedMemoryEvidence(memory_pool_index=4, revision=1)

    for signal_type, evidence in (
        (SignalType.CONTEXT_SHIFT, _evidence(1, 2, binding_indices=(1, 2))),
        (SignalType.CONFLICT, _evidence(2, binding_indices=(2,))),
        (SignalType.STALE_CONSTRAINT, _evidence(2, binding_indices=(2,))),
        (SignalType.STAGNATION, _evidence(1, 2, 3, binding_indices=(1, 2, 3))),
    ):
        with pytest.raises(ValidationError, match="evidence"):
            PublicExpectedSignal(
                signal_type=signal_type,
                strength_ppm=(
                    500_000
                    if signal_type in (SignalType.CONTEXT_SHIFT, SignalType.STAGNATION)
                    else 1_000_000
                ),
                evidence=evidence,
            )


def test_raw_trace_fixture_round_trips_without_profile_or_expected_labels() -> None:
    fixture = _trace_fixture()
    encoded = canonical_json(fixture)

    assert TRACE_FIXTURE_DIGEST_DOMAIN.endswith(":trace-fixture:v1")
    assert OutcomeFreeTraceFixture.model_validate_json(encoded) == fixture
    assert fixture.bindings[0].constraint_references is None
    assert fixture.bindings[1].constraint_references == ()
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


def test_raw_trace_fixture_rejects_bad_bindings_parents_and_digest() -> None:
    fixture = _trace_fixture()

    payload = fixture.model_dump(mode="python")
    payload["bindings"] = tuple(reversed(payload["bindings"]))
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="binding"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["events"][1]["parent_event_pool_indices"] = (2,)
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="parent"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["trace_fixture_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="trace fixture digest"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["events"][1]["payload"]["action"]["command"] = "run-helpful-path"
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="outcome label"):
        OutcomeFreeTraceFixture.model_validate(payload)


def test_reference_result_constructor_rejects_unissued_and_incoherent_values() -> None:
    issuer = signal_fixtures._REFERENCE_RESULT_ISSUER
    evidence = _five_profiles()[0].expected_signals[0].evidence
    no_match = signal_fixtures.ReferencePredicateStatus.NO_MATCH
    detected = signal_fixtures.ReferencePredicateStatus.DETECTED
    abstained = signal_fixtures.ReferencePredicateStatus.ABSTAINED

    with pytest.raises(ValueError, match="issuer"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=object(), signal_type=SignalType.CONFLICT, status=no_match
        )
    with pytest.raises(ValueError, match="reserved"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer, signal_type=SignalType.REPEATED_ACTION, status=no_match
        )
    with pytest.raises(ValueError, match="status"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer,
            signal_type=SignalType.CONFLICT,
            status="no_match",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="evidence"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer,
            signal_type=SignalType.CONFLICT,
            status=no_match,
            evidence=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="abstention reason"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer,
            signal_type=SignalType.CONFLICT,
            status=no_match,
            abstention_reason=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="detection fields"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer, signal_type=SignalType.CONFLICT, status=detected
        )
    with pytest.raises(ValueError, match="detection fields"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer,
            signal_type=SignalType.CONFLICT,
            status=no_match,
            strength_ppm=1,
            evidence=evidence,
        )
    with pytest.raises(ValueError, match="strength"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer,
            signal_type=SignalType.CONFLICT,
            status=detected,
            strength_ppm=0,
            evidence=evidence,
        )
    with pytest.raises(ValueError, match="abstention fields"):
        signal_fixtures.ReferencePredicateResult(
            _issuer=issuer, signal_type=SignalType.CONFLICT, status=abstained
        )


def test_legacy_result_and_evaluation_constructors_reject_incoherent_values() -> None:
    with pytest.raises(ValueError, match="issuer"):
        _legacy_result(issuer=object())
    with pytest.raises(ValueError, match="type"):
        _legacy_result(signal_type=SignalType.CONFLICT)
    with pytest.raises(ValueError, match="metadata"):
        _legacy_result(detector_version="")
    with pytest.raises(ValueError, match="indices"):
        _legacy_result(related_indices=(0, 0))
    with pytest.raises(ValueError, match="detected result"):
        _legacy_result(
            status=DetectionStatus.DETECTED,
            strength_ppm=1_000_000,
            related_indices=(),
        )
    with pytest.raises(ValueError, match="non-detected"):
        _legacy_result(related_indices=())
    with pytest.raises(ValueError, match="abstention reason"):
        _legacy_result(
            status=DetectionStatus.ABSTAINED,
            abstention_reason=object(),  # type: ignore[arg-type]
        )

    events = tuple(object.__new__(TraceEvent) for _ in range(3))
    with pytest.raises(ValueError, match="evaluation is invalid"):
        signal_fixtures.LegacyFixtureEvaluation(
            _issuer=signal_fixtures._LEGACY_RESULT_ISSUER,
            events=events,
            results=(),
        )
    unresolved_results = tuple(
        _legacy_result(signal_type=signal_type, related_indices=(3,))
        for signal_type in signal_fixtures._LEGACY_TYPES
    )
    with pytest.raises(ValueError, match="does not resolve"):
        signal_fixtures.LegacyFixtureEvaluation(
            _issuer=signal_fixtures._LEGACY_RESULT_ISSUER,
            events=events,
            results=unresolved_results,
        )


def test_fixture_materializer_rejects_cross_profile_and_wrapped_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previews = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews

    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.materialize_public_trace_fixture(
            object(),  # type: ignore[arg-type]
            previews[0].task_skeleton,
        )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.materialize_public_trace_fixture(
            previews[0].slot_profile,
            previews[1].task_skeleton,
        )

    integers = previews[0].slot_profile.integers.model_copy(update={"memory_revision": 99})
    mismatched_profile = previews[0].slot_profile.model_copy(update={"integers": integers})
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.materialize_public_trace_fixture(
            mismatched_profile,
            previews[0].task_skeleton,
        )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fault-injection detail")

    monkeypatch.setattr(
        OutcomeFreeTraceFixture,
        "model_validate",
        classmethod(fail),
    )
    with pytest.raises(signal_fixtures.SignalFixtureInputError, match="failed validation"):
        signal_fixtures.materialize_public_trace_fixture(
            previews[0].slot_profile,
            previews[0].task_skeleton,
        )


def test_reference_evaluator_rejects_cross_fixture_and_binding_mismatches() -> None:
    previews = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews

    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.evaluate_reference_predicates(
            object(),  # type: ignore[arg-type]
            previews[0].task_skeleton,
        )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.evaluate_reference_predicates(
            previews[0].trace_fixture,
            previews[1].task_skeleton,
        )

    fixture = previews[0].trace_fixture
    mismatched = _replace_binding(
        fixture,
        -1,
        action_step=fixture.bindings[-1].action_step + 1,  # type: ignore[operator]
    )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.evaluate_reference_predicates(
            mismatched,
            previews[0].task_skeleton,
        )


def test_reference_predicate_defensive_branches_are_typed_and_deterministic() -> None:
    previews = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews
    operand_unresolved = signal_fixtures.ReferencePredicateAbstentionReason.OPERAND_UNRESOLVED
    capability_missing = (
        signal_fixtures.ReferencePredicateAbstentionReason.REQUIRED_CAPABILITY_UNAVAILABLE
    )

    slot_one = previews[1]
    no_prior = slot_one.trace_fixture.model_copy(
        update={
            "bindings": (
                *(
                    binding.model_copy(update={"action_step": None})
                    for binding in slot_one.trace_fixture.bindings[:-1]
                ),
                slot_one.trace_fixture.bindings[-1],
            )
        }
    )
    assert (
        signal_fixtures._evaluate_context_shift(no_prior, slot_one.task_skeleton).abstention_reason
        is operand_unresolved
    )

    bad_current = _replace_binding(
        slot_one.trace_fixture,
        -1,
        action_step=slot_one.trace_fixture.bindings[-1].action_step + 1,  # type: ignore[operator]
    )
    assert (
        signal_fixtures._evaluate_context_shift(
            bad_current, slot_one.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )

    missing_provenance_scope = _replace_binding(slot_one.trace_fixture, 1, scope_id=None)
    memories = list(missing_provenance_scope.memories)
    memories[0] = memories[0].model_copy(update={"provenance_event_pool_indices": (1,)})
    missing_provenance_scope = missing_provenance_scope.model_copy(
        update={"memories": tuple(memories)}
    )
    assert (
        signal_fixtures._evaluate_context_shift(
            missing_provenance_scope, slot_one.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )

    slot_three = previews[3]
    active_memories = list(slot_three.trace_fixture.memories)
    active_memories[0] = active_memories[0].model_copy(update={"validity": ValidityState.ACTIVE})
    active_constraint = slot_three.trace_fixture.model_copy(
        update={"memories": tuple(active_memories)}
    )
    assert (
        signal_fixtures._evaluate_stale_constraint(active_constraint).status
        is signal_fixtures.ReferencePredicateStatus.NO_MATCH
    )

    slot_two = previews[2]
    unstructured_current = _replace_event(slot_two.trace_fixture, -1, payload={})
    assert (
        signal_fixtures._evaluate_stagnation(
            unstructured_current, slot_two.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )

    slot_four = previews[4]
    prior_terminal = _replace_event(slot_four.trace_fixture, -2, phase=EventPhase.TERMINAL)
    assert (
        signal_fixtures._evaluate_stagnation(prior_terminal, slot_four.task_skeleton).status
        is signal_fixtures.ReferencePredicateStatus.NO_MATCH
    )
    prior_unstructured = _replace_event(slot_four.trace_fixture, -2, payload={})
    assert (
        signal_fixtures._evaluate_stagnation(
            prior_unstructured, slot_four.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )
    prior_capability_missing = _replace_binding(
        slot_four.trace_fixture,
        -2,
        action_step=None,
        progress_marker_digest=None,
    )
    assert (
        signal_fixtures._evaluate_stagnation(
            prior_capability_missing, slot_four.task_skeleton
        ).abstention_reason
        is capability_missing
    )
    prior_binding_mismatch = _replace_binding(
        slot_four.trace_fixture,
        -2,
        action_step=slot_four.trace_fixture.bindings[-2].action_step + 1,  # type: ignore[operator]
    )
    assert (
        signal_fixtures._evaluate_stagnation(
            prior_binding_mismatch, slot_four.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )

    unknown_impact = _replace_binding(
        slot_one.trace_fixture,
        -1,
        impact=PublicImpactClass.UNKNOWN,
    )
    assert (
        signal_fixtures._evaluate_irreversible_action(
            unknown_impact, slot_one.task_skeleton
        ).abstention_reason
        is capability_missing
    )
    bad_irreversible_binding = _replace_binding(
        slot_one.trace_fixture,
        -1,
        action_step=slot_one.trace_fixture.bindings[-1].action_step + 1,  # type: ignore[operator]
    )
    assert (
        signal_fixtures._evaluate_irreversible_action(
            bad_irreversible_binding, slot_one.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )
    unstructured_irreversible = _replace_event(slot_one.trace_fixture, -1, payload={})
    assert (
        signal_fixtures._evaluate_irreversible_action(
            unstructured_irreversible, slot_one.task_skeleton
        ).abstention_reason
        is operand_unresolved
    )

    conflict_fixture = previews[0].trace_fixture
    assertions = conflict_fixture.bindings[-1].assertions
    assert assertions is not None
    equal_assertions = (
        assertions[0],
        assertions[1].model_copy(update={"value_digest": assertions[0].value_digest}),
    )
    equal_conflict = _replace_binding(conflict_fixture, -1, assertions=equal_assertions)
    assert (
        signal_fixtures._evaluate_conflict(equal_conflict).status
        is signal_fixtures.ReferencePredicateStatus.NO_MATCH
    )
    assert not signal_fixtures._is_structured_action(conflict_fixture.events[-1])


@pytest.mark.asyncio
async def test_legacy_fixture_runtime_sanitizes_adapter_and_detector_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0].trace_fixture
    scenario_id = "0" * 64

    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        await signal_fixtures.evaluate_legacy_signal_fixture(
            object(),  # type: ignore[arg-type]
            scenario_id=scenario_id,
        )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        await signal_fixtures.evaluate_legacy_signal_fixture(fixture, scenario_id="invalid")
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures._normalize_native_event(object())
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures._event_id_callback(object(), 1)
    with pytest.raises(RuntimeError, match="delivery is unavailable"):
        signal_fixtures._unused_capabilities()
    assert signal_fixtures._unused_target_request_id(object(), object()) is None  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="delivery is unavailable"):
        await signal_fixtures._unused_delivery(object())  # type: ignore[arg-type]

    class _NullEventIdAdapter:
        def normalize(self, value: object) -> object:
            return signal_fixtures._normalize_native_event(value)

        def resolve_event_id(self, value: object, ordinal: int) -> None:
            del value, ordinal
            return None

    monkeypatch.setattr(signal_fixtures, "_fixture_adapter", _NullEventIdAdapter)
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        await signal_fixtures.evaluate_legacy_signal_fixture(fixture, scenario_id=scenario_id)
    monkeypatch.undo()

    class _UnknownEvidenceDetector:
        signal_type = SignalType.REPEATED_ACTION
        detector_version = "coverage-probe/v1"

        def evaluate(self, context: object) -> DetectionOutcome:
            del context
            return DetectionOutcome.detected(self.signal_type, (uuid4(),))

    monkeypatch.setattr(
        signal_fixtures,
        "_legacy_detectors",
        lambda: (_UnknownEvidenceDetector(),),
    )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        await signal_fixtures.evaluate_legacy_signal_fixture(fixture, scenario_id=scenario_id)


def test_detected_projection_rejects_mutated_internal_result_invariants() -> None:
    legacy_results = tuple(
        _legacy_result(signal_type=signal_type) for signal_type in signal_fixtures._LEGACY_TYPES
    )
    object.__setattr__(legacy_results[0], "status", DetectionStatus.DETECTED)
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.detected_signal_projection(legacy_results=legacy_results)

    reference_results = tuple(
        signal_fixtures.ReferencePredicateResult(
            _issuer=signal_fixtures._REFERENCE_RESULT_ISSUER,
            signal_type=signal_type,
            status=signal_fixtures.ReferencePredicateStatus.NO_MATCH,
        )
        for signal_type in signal_fixtures._REFERENCE_TYPES
    )
    object.__setattr__(
        reference_results[0],
        "status",
        signal_fixtures.ReferencePredicateStatus.DETECTED,
    )
    with pytest.raises(signal_fixtures.SignalFixtureInputError):
        signal_fixtures.detected_signal_projection(reference_results=reference_results)
