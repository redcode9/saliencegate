from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.signals.conftest import EventFactory

import saliencegate.signals.base as base_module
from saliencegate.domain import SignalType, canonical_json
from saliencegate.signals.base import (
    DetectionContext,
    DetectionInputError,
    DetectionOutcome,
    DetectorContractError,
    DeterministicSignalExtractor,
    _admit_detection_sequence,
    _extract_trusted_report,
    _longest_trusted_detection_context,
    _trusted_detection_context_is_exact,
    _trusted_extraction_is_exact,
    _ValidatedSignalDetector,
)


class _TrustedToolDetector(_ValidatedSignalDetector):
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TOOL_ERROR

    @property
    def detector_version(self) -> str:
        return "trusted-tool/v1"

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        return self._evaluate_validated(context)

    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome:
        return DetectionOutcome.detected(
            SignalType.TOOL_ERROR,
            (context.current.event_id,),
        )


class _PublicOnlyDetector:
    def __init__(self) -> None:
        self.executed = False

    @property
    def signal_type(self) -> SignalType:
        return SignalType.TOOL_ERROR

    @property
    def detector_version(self) -> str:
        return "public-only/v1"

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        self.executed = True
        return DetectionOutcome.no_match(
            SignalType.TOOL_ERROR,
            (context.current.event_id,),
        )


def test_sequence_proof_snapshots_and_canonicalizes_one_contiguous_run(
    event_factory: EventFactory,
) -> None:
    first = event_factory(1)
    second = event_factory(2)

    proof = _admit_detection_sequence((second, first))

    assert proof.events == (first, second)
    assert proof.events[0] is not first
    assert proof.event_bytes == tuple(canonical_json(event) for event in proof.events)
    assert repr(proof) == "_DetectionSequenceProof(event_count=2)"
    assert "signal-source" not in repr(proof)
    for end_ordinal in (1, 2):
        trusted = _longest_trusted_detection_context(proof, end_ordinal)
        expected = DetectionContext(
            run_id=proof.run_id,
            events=proof.events[:end_ordinal],
        )
        assert trusted.context == expected
        assert trusted.context.events is trusted._events


def test_longest_window_proves_exact_bound_truncation_and_singleton_failure(
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tuple(event_factory(sequence) for sequence in range(1, 4))
    baseline = _admit_detection_sequence(events)
    exact_two_event_bound = sum(baseline._event_costs[-2:])
    monkeypatch.setattr(
        base_module,
        "_MAX_CONTEXT_SIZE_UPPER_BOUND",
        exact_two_event_bound,
    )

    exact = _admit_detection_sequence(events)
    trusted = _longest_trusted_detection_context(exact, 3)

    assert trusted.start_index == 1
    assert trusted.context.events == exact.events[-2:]
    assert (
        exact._prefix_costs[trusted.end_ordinal] - exact._prefix_costs[trusted.start_index]
        == exact_two_event_bound
    )
    with pytest.raises(ValidationError, match="local size bound"):
        DetectionContext(run_id=exact.run_id, events=exact.events)
    assert DetectionContext(run_id=exact.run_id, events=exact.events[-2:]) == trusted.context

    monkeypatch.setattr(
        base_module,
        "_MAX_CONTEXT_SIZE_UPPER_BOUND",
        exact_two_event_bound - 1,
    )
    truncated = _admit_detection_sequence(events)
    shorter = _longest_trusted_detection_context(truncated, 3)
    assert shorter.start_index == 2
    assert shorter.context.events == truncated.events[-1:]

    monkeypatch.setattr(
        base_module,
        "_MAX_CONTEXT_SIZE_UPPER_BOUND",
        baseline._event_costs[-1] - 1,
    )
    with pytest.raises(DetectionInputError):
        _admit_detection_sequence(events[-1:])


def test_sequence_proof_rejects_graph_and_coordinate_tampering_value_free(
    event_factory: EventFactory,
) -> None:
    first = event_factory(1)
    second = event_factory(2)
    proof = _admit_detection_sequence((first, second))
    mixed_run = event_factory(
        2,
        run_id=UUID("00000000-0000-4000-8000-000000002002"),
    )
    duplicate = second.model_copy(update={"event_id": first.event_id})

    for event_candidate in (
        (first, event_factory(3)),
        (first, mixed_run),
        (first, duplicate),
    ):
        with pytest.raises(DetectionInputError) as captured:
            _admit_detection_sequence(event_candidate)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

    for proof_candidate, end_ordinal in (
        (replace(proof, _token=object()), 2),
        (replace(proof, _context_starts=(1, 1)), 2),
        (proof, 0),
        (proof, 3),
    ):
        with pytest.raises(DetectionInputError):
            _longest_trusted_detection_context(proof_candidate, end_ordinal)


def test_per_instance_seals_reject_middle_preimage_and_context_replacement(
    event_factory: EventFactory,
) -> None:
    events = tuple(event_factory(sequence) for sequence in range(1, 4))
    proof = _admit_detection_sequence(events)
    foreign_middle = event_factory(
        2,
        run_id=UUID("00000000-0000-4000-8000-000000002002"),
    )
    changed_events = (proof.events[0], foreign_middle, proof.events[2])
    changed_bytes = list(proof.event_bytes)
    changed_bytes[1] = b"{}"
    changed_costs = list(proof._prefix_costs)
    changed_costs[2] += 1
    changed_starts = list(proof._context_starts)
    changed_starts[1] = 1

    for damaged in (
        replace(proof, events=changed_events),
        replace(proof, event_bytes=tuple(changed_bytes)),
        replace(proof, _prefix_costs=tuple(changed_costs)),
        replace(proof, _context_starts=tuple(changed_starts)),
    ):
        with pytest.raises(DetectionInputError):
            _longest_trusted_detection_context(damaged, 3)

    trusted_context = _longest_trusted_detection_context(proof, 3)
    changed_context = trusted_context.context.model_copy(update={"events": changed_events})
    damaged_context = replace(
        trusted_context,
        context=changed_context,
        _events=changed_events,
    )
    assert not _trusted_detection_context_is_exact(damaged_context)

    extractor = DeterministicSignalExtractor((_TrustedToolDetector(),))
    extraction = _extract_trusted_report(extractor, trusted_context)
    copied_report = base_module.ExtractionReport.model_validate_json(
        extraction.report.model_dump_json(warnings=False)
    )
    assert copied_report == extraction.report
    assert copied_report is not extraction.report
    assert not _trusted_extraction_is_exact(replace(extraction, report=copied_report))


def test_trusted_extraction_matches_public_report_and_shares_materialization(
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tuple(event_factory(sequence) for sequence in range(1, 4))
    extractor = DeterministicSignalExtractor((_TrustedToolDetector(),))
    proof = _admit_detection_sequence(events)
    trusted_context = _longest_trusted_detection_context(proof, 3)
    public_context = DetectionContext(run_id=proof.run_id, events=proof.events)
    real_materialize = base_module._materialize_extraction_report
    calls: list[DetectionContext] = []

    def recording_materialize(
        context: DetectionContext,
        evaluations: tuple[base_module.DetectorEvaluation, ...],
    ) -> base_module.ExtractionReport:
        calls.append(context)
        return real_materialize(context, evaluations)

    monkeypatch.setattr(
        base_module,
        "_materialize_extraction_report",
        recording_materialize,
    )

    public_report = extractor.extract_report(public_context)
    trusted = _extract_trusted_report(extractor, trusted_context)

    assert len(calls) == 2
    assert trusted.report == public_report
    assert canonical_json(trusted.report) == canonical_json(public_report)
    assert trusted.context == public_context
    assert _trusted_extraction_is_exact(trusted)
    assert repr(trusted) == "_TrustedExtraction(<validated>)"


def test_trusted_extraction_rejects_unsealed_contexts_extractors_and_results(
    event_factory: EventFactory,
) -> None:
    event = event_factory(1)
    proof = _admit_detection_sequence((event,))
    trusted_context = _longest_trusted_detection_context(proof, 1)
    trusted_extractor = DeterministicSignalExtractor((_TrustedToolDetector(),))
    forged_context = replace(trusted_context, _token=object())

    with pytest.raises(DetectorContractError):
        _extract_trusted_report(trusted_extractor, forged_context)

    public_only = _PublicOnlyDetector()
    untrusted_extractor = DeterministicSignalExtractor((public_only,))
    with pytest.raises(DetectorContractError) as captured:
        _extract_trusted_report(untrusted_extractor, trusted_context)
    assert not public_only.executed
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None

    trusted = _extract_trusted_report(trusted_extractor, trusted_context)
    assert not _trusted_extraction_is_exact(replace(trusted, _token=object()))
