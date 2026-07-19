from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import EventType, SignalType
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionInputError,
    DetectionOutcome,
    DetectionStatus,
    DetectorContractError,
    DetectorEvaluation,
    DeterministicSignalExtractor,
    ExtractionReport,
    validate_detection_context,
)


class StubDetector:
    def __init__(
        self,
        signal_type: SignalType,
        detector_version: str,
        outcome: object,
    ) -> None:
        self._signal_type = signal_type
        self._detector_version = detector_version
        self.outcome = outcome

    @property
    def signal_type(self) -> SignalType:
        return self._signal_type

    @property
    def detector_version(self) -> str:
        return self._detector_version

    def evaluate(self, _context: DetectionContext) -> DetectionOutcome:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(DetectionOutcome, self.outcome)


class SingleReadMetadataDetector(StubDetector):
    def __init__(self, outcome: DetectionOutcome) -> None:
        super().__init__(SignalType.TOOL_ERROR, "single-read/v1", outcome)
        self.signal_type_reads = 0
        self.version_reads = 0

    @property
    def signal_type(self) -> SignalType:
        self.signal_type_reads += 1
        if self.signal_type_reads != 1:
            raise RuntimeError("metadata was read again")
        return SignalType.TOOL_ERROR

    @property
    def detector_version(self) -> str:
        self.version_reads += 1
        if self.version_reads != 1:
            raise RuntimeError("metadata was read again")
        return "single-read/v1"


class CoincidentalPrivateMethodDetector(StubDetector):
    def _evaluate_validated(self, _context: DetectionContext) -> DetectionOutcome:
        raise RuntimeError("the extractor must not call a third-party private method")


class FailingMetadataDetector:
    def evaluate(self, _context: DetectionContext) -> DetectionOutcome:
        raise AssertionError("unreachable")

    @property
    def signal_type(self) -> SignalType:
        raise RuntimeError("metadata-secret")

    @property
    def detector_version(self) -> str:
        return "metadata/v1"


def test_context_canonicalizes_one_contiguous_run_window(event_factory: Any) -> None:
    first = event_factory(1)
    second = event_factory(2)

    context = DetectionContext(run_id=first.run_id, events=(second, first))

    assert context.events == (first, second)
    assert context.current == second
    assert validate_detection_context(context) == context
    assert "payload" not in repr(context)


def test_context_rejects_gap_mixed_run_duplicates_and_forged_models(event_factory: Any) -> None:
    first = event_factory(1)
    with pytest.raises(ValidationError, match="contiguous"):
        DetectionContext(run_id=first.run_id, events=(first, event_factory(3)))
    with pytest.raises(ValidationError, match="one run"):
        DetectionContext(
            run_id=first.run_id,
            events=(
                first,
                event_factory(
                    2,
                    run_id=UUID("00000000-0000-4000-8000-000000002002"),
                ),
            ),
        )
    duplicate = event_factory(2).model_copy(update={"event_id": first.event_id})
    with pytest.raises(ValidationError, match="identities"):
        DetectionContext(run_id=first.run_id, events=(first, duplicate))
    forged = DetectionContext.model_construct(run_id=first.run_id, events=(first, event_factory(3)))
    with pytest.raises(DetectionInputError) as error:
        validate_detection_context(forged)
    assert "signal-source" not in str(error.value)
    with pytest.raises(DetectionInputError):
        validate_detection_context(cast(Any, {"run_id": first.run_id, "events": (first,)}))


def test_detection_outcome_enforces_its_tagged_union() -> None:
    event_id = UUID("00000000-0000-4000-8000-000000002001")
    with pytest.raises(ValidationError, match="inconsistent"):
        DetectionOutcome(signal_type=SignalType.TOOL_ERROR, status=DetectionStatus.DETECTED)
    with pytest.raises(ValidationError, match="inconsistent"):
        DetectionOutcome(
            signal_type=SignalType.TOOL_ERROR,
            status=DetectionStatus.NO_MATCH,
            abstention_reason=AbstentionReason.INSUFFICIENT_HISTORY,
        )
    with pytest.raises(ValidationError, match="inconsistent"):
        DetectionOutcome(
            signal_type=SignalType.TEST_FAILURE,
            status=DetectionStatus.NO_MATCH,
        )
    with pytest.raises(ValidationError, match="unique"):
        DetectionOutcome.abstained(
            SignalType.TOOL_ERROR,
            AbstentionReason.INSUFFICIENT_HISTORY,
            (event_id, event_id),
        )
    abstained = DetectionOutcome.abstained(
        SignalType.TOOL_ERROR,
        AbstentionReason.INSUFFICIENT_HISTORY,
        (event_id,),
    )
    assert abstained.status is DetectionStatus.ABSTAINED


def test_extractor_orders_detectors_and_materializes_only_detected_outcomes(
    event_factory: Any,
) -> None:
    event = event_factory(1, event_type=EventType.CONTROLLER_ERROR)
    context = DetectionContext(run_id=event.run_id, events=(event,))
    detected = DetectionOutcome.detected(SignalType.TOOL_ERROR, (event.event_id,))
    no_match = DetectionOutcome.no_match(SignalType.TEST_FAILURE, (event.event_id,))
    extractor = DeterministicSignalExtractor(
        (
            StubDetector(SignalType.TOOL_ERROR, "z-detector/v1", detected),
            StubDetector(SignalType.TEST_FAILURE, "a-detector/v1", no_match),
        )
    )

    evaluations = extractor.evaluate(context)
    report = extractor.extract_report(context)
    signals = report.signals

    assert tuple(evaluation.signal_type for evaluation in evaluations) == (
        SignalType.TEST_FAILURE,
        SignalType.TOOL_ERROR,
    )
    assert tuple(evaluation.detector_version for evaluation in evaluations) == (
        "a-detector/v1",
        "z-detector/v1",
    )
    assert tuple(evaluation.outcome.status for evaluation in report.evaluations) == (
        DetectionStatus.NO_MATCH,
        DetectionStatus.DETECTED,
    )
    assert report.run_id == event.run_id
    assert report.current_event_id == event.event_id
    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.TOOL_ERROR
    assert extractor.extract(context) == signals


def test_extractor_rejects_bad_plugins_and_evidence(
    event_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = event_factory(1)
    context = DetectionContext(run_id=event.run_id, events=(event,))
    valid = DetectionOutcome.detected(SignalType.TOOL_ERROR, (event.event_id,))
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(cast(Any, []))
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(cast(Any, (object(),)))
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(cast(Any, (object(),) * (len(SignalType) + 1)))
    with pytest.raises(DetectorContractError) as metadata_error:
        DeterministicSignalExtractor(cast(Any, (FailingMetadataDetector(),)))
    assert metadata_error.value.__context__ is None
    assert metadata_error.value.__cause__ is None
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor((StubDetector(SignalType.TOOL_ERROR, "bad version", valid),))
    duplicate = StubDetector(SignalType.TOOL_ERROR, "same/v1", valid)
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor((duplicate, duplicate))
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(
            (
                duplicate,
                StubDetector(SignalType.TOOL_ERROR, "different/v2", valid),
            )
        )

    wrong_type = StubDetector(SignalType.TEST_FAILURE, "wrong/v1", valid)
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor((wrong_type,)).evaluate(context)
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(
            (StubDetector(SignalType.TOOL_ERROR, "object/v1", object()),)
        ).evaluate(context)
    with pytest.raises(DetectorContractError) as plugin_error:
        DeterministicSignalExtractor(
            (StubDetector(SignalType.TOOL_ERROR, "failure/v1", RuntimeError("fixture-secret")),)
        ).evaluate(context)
    assert plugin_error.value.__context__ is None
    assert plugin_error.value.__cause__ is None
    tainted = DetectorContractError.__new__(DetectorContractError)
    ValueError.__init__(tainted, "fixture-secret")
    extractor = DeterministicSignalExtractor(
        (StubDetector(SignalType.TOOL_ERROR, "tainted/v1", tainted),)
    )
    with pytest.raises(DetectorContractError) as error:
        extractor.evaluate(context)
    assert "fixture-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None

    forged_outcome = DetectionOutcome.no_match(
        SignalType.TOOL_ERROR,
        (event.event_id,),
    ).model_copy(update={"strength": float("nan")})
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(
            (StubDetector(SignalType.TOOL_ERROR, "forged/v1", forged_outcome),)
        ).evaluate(context)
    forged_type = valid.model_copy(update={"signal_type": "fixture-secret"})
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(
            (StubDetector(SignalType.TOOL_ERROR, "forged-type/v1", forged_type),)
        ).evaluate(context)
    assert "fixture-secret" not in capsys.readouterr().err


def test_extractor_requires_current_ordered_in_window_evidence(event_factory: Any) -> None:
    first = event_factory(1)
    current = event_factory(2)
    context = DetectionContext(run_id=first.run_id, events=(first, current))

    missing_current = DetectionOutcome.detected(SignalType.REPEATED_ACTION, (first.event_id,))
    reversed_evidence = DetectionOutcome.detected(
        SignalType.REPEATED_ACTION,
        (current.event_id, first.event_id),
    )
    for version, outcome in (
        ("missing/v1", missing_current),
        ("reversed/v1", reversed_evidence),
    ):
        detector = StubDetector(SignalType.REPEATED_ACTION, version, outcome)
        with pytest.raises(DetectorContractError):
            DeterministicSignalExtractor((detector,)).evaluate(context)

    outside = UUID("00000000-0000-4000-8000-00000000eeee")
    invalid_no_match = DetectionOutcome.no_match(
        SignalType.REPEATED_ACTION,
        (outside, current.event_id),
    )
    with pytest.raises(DetectorContractError):
        DeterministicSignalExtractor(
            (StubDetector(SignalType.REPEATED_ACTION, "outside/v1", invalid_no_match),)
        ).evaluate(context)


def test_extractor_snapshots_plugin_metadata_exactly_once(event_factory: Any) -> None:
    event = event_factory(1)
    detector = SingleReadMetadataDetector(
        DetectionOutcome.detected(SignalType.TOOL_ERROR, (event.event_id,))
    )
    extractor = DeterministicSignalExtractor((detector,))

    assert len(extractor.extract(DetectionContext(run_id=event.run_id, events=(event,)))) == 1
    assert detector.signal_type_reads == 1
    assert detector.version_reads == 1
    with pytest.raises(AttributeError):
        extractor._entries = ()

    public_only = CoincidentalPrivateMethodDetector(
        SignalType.TEST_FAILURE,
        "public-only/v1",
        DetectionOutcome.no_match(SignalType.TEST_FAILURE, (event.event_id,)),
    )
    evaluations = DeterministicSignalExtractor((public_only,)).evaluate(
        DetectionContext(run_id=event.run_id, events=(event,))
    )
    assert evaluations[0].outcome.status is DetectionStatus.NO_MATCH


def test_public_evaluation_invariants_reject_mismatches(event_factory: Any) -> None:
    event = event_factory(1)
    no_match = DetectionOutcome.no_match(SignalType.TEST_FAILURE, (event.event_id,))
    with pytest.raises(ValidationError, match="signal types disagree"):
        DetectorEvaluation(
            signal_type=SignalType.TOOL_ERROR,
            detector_version="mismatch/v1",
            outcome=no_match,
        )

    class StringSubclass(str):
        pass

    with pytest.raises(ValidationError, match="exact string"):
        DetectorEvaluation(
            signal_type=SignalType.TEST_FAILURE,
            detector_version=StringSubclass("subclass/v1"),
            outcome=no_match,
        )


def test_extraction_report_rejects_inconsistent_public_construction(
    event_factory: Any,
) -> None:
    event = event_factory(1, event_type=EventType.CONTROLLER_ERROR)
    context = DetectionContext(run_id=event.run_id, events=(event,))
    report = DeterministicSignalExtractor(
        (
            StubDetector(
                SignalType.TOOL_ERROR,
                "tool/v1",
                DetectionOutcome.detected(SignalType.TOOL_ERROR, (event.event_id,)),
            ),
        )
    ).extract_report(context)

    wrong_current = UUID("00000000-0000-4000-8000-00000000eeee")
    wrong_current_evaluation = DetectorEvaluation(
        signal_type=SignalType.TEST_FAILURE,
        detector_version="no-match/v1",
        outcome=DetectionOutcome.no_match(SignalType.TEST_FAILURE, (wrong_current,)),
    )
    with pytest.raises(ValidationError, match="current event"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=(wrong_current_evaluation,),
            signals=(),
        )
    with pytest.raises(ValidationError, match="unique"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=(report.evaluations[0], report.evaluations[0]),
            signals=report.signals,
        )
    with pytest.raises(ValidationError, match="disagree"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=report.evaluations,
            signals=(),
        )

    with pytest.raises(ValidationError, match="attribution"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=report.evaluations,
            signals=(report.signals[0].model_copy(update={"detector_version": "other/v1"}),),
        )
    with pytest.raises(ValidationError, match="attribution"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=report.evaluations,
            signals=(
                report.signals[0].model_copy(
                    update={"signal_id": UUID("00000000-0000-4000-8000-00000000ffff")}
                ),
            ),
        )
    with pytest.raises(ValidationError, match="unvalidated records"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=report.evaluations,
            signals=(report.signals[0].model_copy(update={"schema_version": "9.9"}),),
        )
    with pytest.raises(ValidationError, match="attribution"):
        ExtractionReport(
            run_id=report.run_id,
            current_event_id=report.current_event_id,
            current_event_timestamp=report.current_event_timestamp,
            evaluations=report.evaluations,
            signals=(
                report.signals[0].model_copy(
                    update={"created_at": report.current_event_timestamp.replace(year=2025)}
                ),
            ),
        )


def test_context_rejects_an_oversized_local_payload(event_factory: Any) -> None:
    oversized = event_factory(1).model_copy(update={"payload": {"value": "x" * 1_700_000}})

    with pytest.raises(ValidationError, match="local size bound"):
        DetectionContext(run_id=oversized.run_id, events=(oversized,))


def test_context_bound_counts_parent_ids_and_sanitizes_forged_events(
    event_factory: Any,
) -> None:
    shared_empty_tuples = event_factory(1, payload={"left": [], "right": []})
    assert (
        DetectionContext(
            run_id=shared_empty_tuples.run_id,
            events=(shared_empty_tuples,),
        ).current
        == shared_empty_tuples
    )
    scalar_payload = event_factory(
        1,
        payload={"none": None, "bool": True, "int": -42, "float": 1.25},
    )
    assert (
        validate_detection_context(
            DetectionContext(run_id=scalar_payload.run_id, events=(scalar_payload,))
        ).current
        == scalar_payload
    )

    parents = tuple(UUID(f"00000000-0000-4000-8000-{value:012x}") for value in range(1, 258))
    event = event_factory(1).model_copy(update={"parent_ids": parents})
    with pytest.raises(ValidationError, match="local size bound"):
        DetectionContext(run_id=event.run_id, events=(event,))

    forged = DetectionContext.model_construct(
        run_id=event.run_id,
        events=(RuntimeError("fixture-secret"),),
    )
    with pytest.raises(DetectionInputError) as error:
        validate_detection_context(forged)
    assert "fixture-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None

    cyclic_payload: dict[str, object] = {}
    cyclic_payload["self"] = cyclic_payload
    cyclic_event = event.model_copy(update={"payload": cyclic_payload})
    cyclic_context = DetectionContext.model_construct(
        run_id=event.run_id,
        events=(cyclic_event,),
    )
    with pytest.raises(DetectionInputError):
        validate_detection_context(cyclic_context)

    forged_digest = event.payload_digest.model_construct(
        algorithm=event.payload_digest.algorithm,
        value="x" * 100_000,
    )
    forged_digest_event = event.model_copy(update={"payload_digest": forged_digest})
    forged_digest_context = DetectionContext.model_construct(
        run_id=event.run_id,
        events=(forged_digest_event,),
    )
    with pytest.raises(DetectionInputError):
        validate_detection_context(forged_digest_context)

    non_finite_event = event.model_copy(
        update={"payload": MappingProxyType({"value": float("nan")})}
    )
    non_finite_context = DetectionContext.model_construct(
        run_id=event.run_id,
        events=(non_finite_event,),
    )
    with pytest.raises(DetectionInputError):
        validate_detection_context(non_finite_context)
