from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    authenticated_intake,
    capture_context,
    initialized_store,
    register_connection,
)

import saliencegate.capture.spool as spool_module
from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.capture.normalization import (
    CaptureNormalization,
    normalize_capture_session_snapshot,
)
from saliencegate.capture.report import (
    CaptureReportCounts,
    CaptureReportCoverage,
    CaptureReportDetector,
    CaptureReportError,
    CaptureReportHeadline,
    CaptureReportHealthCount,
    CaptureReportInterval,
    CaptureReportLimit,
    CaptureSessionReport,
    CaptureSpoolReportStatus,
    _seal_capture_session_report,
    build_capture_session_report,
    decode_capture_session_report,
    encode_capture_session_report,
    render_capture_session_report_human,
    render_capture_session_report_json,
)
from saliencegate.capture.sessions import (
    CaptureSessionSnapshot,
    CaptureSnapshotHealth,
    _authenticate_capture_session_snapshot,
)
from saliencegate.capture.spool import CaptureSpool, CaptureSpoolObservation
from saliencegate.capture.store import CaptureSessionState
from saliencegate.domain import SignalType
from saliencegate.shadow.evaluation import ShadowHeuristicDisposition

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def _spool(state_directory: Path) -> CaptureSpool:
    return CaptureSpool.open(
        CaptureStoreLocations(
            platform="windows" if os.name == "nt" else "posix",
            state_directory=state_directory,
            database_path=state_directory / "capture.sqlite3",
            spool_directory=state_directory / "capture-spool",
        ),
        INSTALLATION_KEY,
    )


def _real_capture(
    tmp_path: Path,
    name: str,
    *,
    action_count: int,
    repeated: bool,
    identity_authority: str = "exact",
    close: bool = True,
    controller_failure: str | None = None,
) -> tuple[CaptureSessionSnapshot, CaptureNormalization]:
    state_path = tmp_path / f"{name}.sqlite3"
    _spool(tmp_path)
    context = capture_context()
    shared_action = context.action_identity(b"synthetic-repeated-action")
    with initialized_store(state_path) as store:
        register_connection(store)
        started = authenticated_intake("session_started", producer_index=1)
        store.append(started)
        for offset in range(action_count):
            producer_index = offset + 2
            changes: dict[str, object] = {"identity_authority": identity_authority}
            if repeated:
                changes["action_digest"] = shared_action
            store.append(
                authenticated_intake(
                    "action_started",
                    producer_index=producer_index,
                    changes=changes,
                )
            )
        next_index = action_count + 2
        if controller_failure is not None:
            store.append(
                authenticated_intake(
                    "controller_failed",
                    producer_index=next_index,
                    changes={"error_code": controller_failure},
                )
            )
            next_index += 1
        if close:
            store.append(
                authenticated_intake(
                    "session_finished",
                    producer_index=next_index,
                )
            )
        snapshot = store.snapshot_session(CONNECTION_ID, started.session_id)
    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    return snapshot, normalization


def _report_real_capture(
    snapshot: CaptureSessionSnapshot,
    normalization: CaptureNormalization,
    *,
    tmp_path: Path,
    spool: CaptureSpool | None = None,
) -> CaptureSessionReport:
    return build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=_spool(tmp_path) if spool is None else spool,
    )


def _health() -> tuple[CaptureReportHealthCount, ...]:
    return tuple(
        CaptureReportHealthCount(code=code, count=0, lower_bound=0) for code in CaptureHealthCode
    )


def _detectors(
    *,
    authorized_actions: int,
    detected: bool = False,
    applicable: bool = True,
) -> tuple[CaptureReportDetector, ...]:
    manifest = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    result: list[CaptureReportDetector] = []
    for capability in manifest.detectors:
        signal_type = capability.signal_type
        if capability.support is not CapabilitySupport.UNSUPPORTED:
            evaluated_actions = authorized_actions if applicable else 0
            sufficient = evaluated_actions >= 2
            result.append(
                CaptureReportDetector(
                    signal_type=signal_type,
                    support=capability.support,
                    omissions=capability.omissions,
                    disposition=(
                        ShadowHeuristicDisposition.FLAGGED
                        if detected
                        else ShadowHeuristicDisposition.NOT_FLAGGED
                        if sufficient
                        else ShadowHeuristicDisposition.INDETERMINATE
                    ),
                    minimum_authorized_observations=2,
                    authorized_observation_count=evaluated_actions,
                    unresolved_observation_count=0,
                    detected_count=int(detected),
                    sufficient_for_absence=sufficient,
                )
            )
        else:
            result.append(
                CaptureReportDetector(
                    signal_type=signal_type,
                    support=capability.support,
                    omissions=capability.omissions,
                    disposition=ShadowHeuristicDisposition.NOT_APPLICABLE,
                    minimum_authorized_observations=0,
                    authorized_observation_count=0,
                    unresolved_observation_count=0,
                    detected_count=0,
                    sufficient_for_absence=False,
                )
            )
    return tuple(result)


def _report(
    *,
    headline: CaptureReportHeadline,
    disposition: ShadowHeuristicDisposition,
    authorized_actions: int,
    detected: bool = False,
    applicable: bool = True,
    state: CaptureSessionState = CaptureSessionState.CLOSED,
    limits: tuple[CaptureReportLimit, ...] = (),
    spool_status: CaptureSpoolReportStatus = (CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED),
    queued_spool_events: int = 0,
    dropped_spool_events: int = 0,
) -> CaptureSessionReport:
    manifest = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    detectors = _detectors(
        authorized_actions=authorized_actions,
        detected=detected,
        applicable=applicable,
    )
    ordered_limits = tuple(sorted(limits, key=lambda item: item.value))
    evidence_only = {
        CaptureReportLimit.NO_APPLICABLE_DETECTOR,
        CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,
    }
    captured_events = max(authorized_actions, 0)
    body: dict[str, object] = {
        "schema_version": "capture-session-report/v1",
        "session_id": "abcdefghijkl",
        "session_state": state,
        "interval": CaptureReportInterval(
            opened_at=NOW,
            updated_at=NOW,
            closed_at=NOW if state is CaptureSessionState.CLOSED else None,
        ),
        "profile_id": CaptureProfile.CODEX_HOOKS_V1,
        "host_version": "0.144.6",
        "compatibility_status": CompatibilityStatus.VERIFIED,
        "headline": headline,
        "shadow_disposition": disposition,
        "counts": CaptureReportCounts(
            captured_events=captured_events,
            projected_events=captured_events,
            action_identities=captured_events,
            structured_results=0,
            detected_signals=int(detected),
            ignored_records=0,
        ),
        "coverage": CaptureReportCoverage(
            spool_integrity=(
                "unavailable"
                if spool_status is CaptureSpoolReportStatus.UNAVAILABLE
                else "hmac_verified_snapshot_bound"
            ),
            spool_observation_tag=(
                None if spool_status is CaptureSpoolReportStatus.UNAVAILABLE else "d" * 64
            ),
            spool_status=spool_status,
            coverage_degraded=bool(set(ordered_limits) - evidence_only),
            gap_count=0,
            drop_count=0,
            overflow_count=0,
            queued_spool_events=queued_spool_events,
            dropped_spool_events=dropped_spool_events,
            health=_health(),
            capability_exclusions=manifest.coverage_exclusions,
            limits=ordered_limits,
        ),
        "detectors": detectors,
        "capability_manifest_digest": capture_capability_digest(manifest),
        "snapshot_digest": "b" * 64,
        "normalization_digest": "c" * 64,
    }
    return _seal_capture_session_report(body)


def test_capture_report_headline_vocabulary_is_closed() -> None:
    assert tuple(item.value for item in CaptureReportHeadline) == (
        "memory_review_suggested",
        "no_current_evidence",
        "insufficient_evidence",
    )


def test_three_headlines_bind_to_shadow_dispositions_and_detector_minima() -> None:
    suggested = _report(
        headline=CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
        disposition=ShadowHeuristicDisposition.FLAGGED,
        authorized_actions=2,
        detected=True,
    )
    absent = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    insufficient = _report(
        headline=CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.INDETERMINATE,
        authorized_actions=1,
        limits=(CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,),
    )

    assert suggested.counts.detected_signals == 1
    assert absent.detectors[2].sufficient_for_absence is True
    assert insufficient.detectors[2].sufficient_for_absence is False


def test_positive_evidence_is_independent_of_the_negative_absence_minimum() -> None:
    detector = CaptureReportDetector(
        signal_type=SignalType.TOOL_ERROR,
        support=CapabilitySupport.SUPPORTED,
        omissions=(),
        disposition=ShadowHeuristicDisposition.FLAGGED,
        minimum_authorized_observations=1,
        authorized_observation_count=0,
        unresolved_observation_count=0,
        detected_count=1,
        sufficient_for_absence=False,
    )

    assert detector.disposition is ShadowHeuristicDisposition.FLAGGED
    assert detector.sufficient_for_absence is False


def test_positive_signal_precedes_open_and_degraded_coverage() -> None:
    report = _report(
        headline=CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
        disposition=ShadowHeuristicDisposition.FLAGGED,
        authorized_actions=2,
        detected=True,
        state=CaptureSessionState.OPEN,
        limits=(
            CaptureReportLimit.SESSION_OPEN,
            CaptureReportLimit.SPOOL_PENDING,
        ),
        spool_status=CaptureSpoolReportStatus.VERIFIED_PENDING,
        queued_spool_events=1,
    )

    assert report.headline is CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED
    assert report.coverage.coverage_degraded is True


def test_quarantined_empty_window_is_not_applicable_and_insufficient() -> None:
    report = _report(
        headline=CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_APPLICABLE,
        authorized_actions=0,
        applicable=False,
        state=CaptureSessionState.QUARANTINED,
        limits=(
            CaptureReportLimit.NO_APPLICABLE_DETECTOR,
            CaptureReportLimit.SESSION_QUARANTINED,
        ),
    )

    assert report.counts.captured_events == 0
    repeated = next(
        item for item in report.detectors if item.signal_type is SignalType.REPEATED_ACTION
    )
    assert repeated.disposition is ShadowHeuristicDisposition.INDETERMINATE
    assert all(
        item.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
        for item in report.detectors
        if item.signal_type is not SignalType.REPEATED_ACTION
    )


@pytest.mark.parametrize(
    ("spool_status", "queued", "dropped"),
    (
        (CaptureSpoolReportStatus.UNAVAILABLE, 0, 0),
        (CaptureSpoolReportStatus.VERIFIED_PENDING, 1, 0),
        (CaptureSpoolReportStatus.VERIFIED_DEGRADED, 0, 1),
    ),
)
def test_non_clean_spool_cannot_back_a_negative_headline(
    spool_status: CaptureSpoolReportStatus,
    queued: int,
    dropped: int,
) -> None:
    limit = (
        CaptureReportLimit.SPOOL_UNAVAILABLE
        if spool_status is CaptureSpoolReportStatus.UNAVAILABLE
        else CaptureReportLimit.SPOOL_PENDING
        if spool_status is CaptureSpoolReportStatus.VERIFIED_PENDING
        else CaptureReportLimit.SPOOL_DROP
    )
    report = _report(
        headline=CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.INDETERMINATE,
        authorized_actions=2,
        limits=(limit,),
        spool_status=spool_status,
        queued_spool_events=queued,
        dropped_spool_events=dropped,
    )

    assert report.coverage.spool_status is spool_status


def test_report_codec_is_deterministic_canonical_and_one_line() -> None:
    first = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    second = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )

    encoded = encode_capture_session_report(first)

    assert encoded == encode_capture_session_report(second)
    assert decode_capture_session_report(encoded) == first
    assert render_capture_session_report_json(first) == encoded.decode("utf-8") + "\n"
    assert (
        encoded
        == json.dumps(
            json.loads(encoded),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


@pytest.mark.parametrize(
    "invalid",
    (
        b"{}\n",
        b'{"schema_version":"capture-session-report/v1",'
        b'"schema_version":"capture-session-report/v1"}',
        b"{" + (b'"x":' + b"[" * 20 + b"0" + b"]" * 20) + b"}",
        b"x" * (512 * 1_024 + 1),
    ),
)
def test_report_codec_rejects_noncanonical_duplicate_or_unbounded_json(invalid: bytes) -> None:
    with pytest.raises(CaptureReportError, match="capture session report is invalid"):
        decode_capture_session_report(invalid)


def test_report_codec_rejects_digest_tampering() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = json.loads(encode_capture_session_report(report))
    body["report_digest"] = "0" * 64
    tampered = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(CaptureReportError):
        decode_capture_session_report(tampered)


def test_human_report_declares_authority_and_limits_without_digests_or_paths() -> None:
    report = _report(
        headline=CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.INDETERMINATE,
        authorized_actions=1,
        limits=(CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,),
    )

    rendered = render_capture_session_report_human(report)

    assert "source=none_same_user_untrusted" in rendered
    assert "decision authority: false; model calls: 0; confirmatory: false" in rendered
    assert "limits: detector_minimum_not_met" in rendered
    assert "digest" not in rendered
    assert "a" * 64 not in rendered
    assert "b" * 64 not in rendered
    assert "c" * 64 not in rendered
    assert report.source_authentication == "none_same_user_untrusted"
    assert report.raw_content_persisted is False
    assert report.transcript_read is False
    assert report.complete_execution_session_coverage is False
    assert report.at_rest_integrity == "hmac_sha256_local_mutation_detection"
    assert report.report_integrity == "sha256_canonical_body"
    assert report.rollback_detection == "none"
    assert report.timestamp_authority == "local_observation"
    assert report.sequence_authority == "local_receipt_order"
    assert report.evidence_level == "descriptive_observational"
    assert report.decision_authority is False
    assert report.model_calls == 0
    assert report.confirmatory is False


@pytest.mark.parametrize(
    ("headline", "disposition", "authorized", "detected", "limits", "opening"),
    (
        (
            CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
            ShadowHeuristicDisposition.FLAGGED,
            2,
            True,
            (),
            "Memory review suggested\n",
        ),
        (
            CaptureReportHeadline.NO_CURRENT_EVIDENCE,
            ShadowHeuristicDisposition.NOT_FLAGGED,
            2,
            False,
            (),
            "No current evidence\n",
        ),
        (
            CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
            ShadowHeuristicDisposition.INDETERMINATE,
            1,
            False,
            (CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,),
            "Insufficient evidence\n",
        ),
    ),
)
def test_human_report_opens_with_the_exact_conclusion(
    headline: CaptureReportHeadline,
    disposition: ShadowHeuristicDisposition,
    authorized: int,
    detected: bool,
    limits: tuple[CaptureReportLimit, ...],
    opening: str,
) -> None:
    report = _report(
        headline=headline,
        disposition=disposition,
        authorized_actions=authorized,
        detected=detected,
        limits=limits,
    )

    assert render_capture_session_report_human(report).startswith(opening)


def test_report_rejects_invented_detector_minima_and_denominators() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    repeated_index = tuple(SignalType).index(SignalType.REPEATED_ACTION)
    repeated = report.detectors[repeated_index]

    forged_minimum = repeated.model_copy(
        update={
            "minimum_authorized_observations": 1,
            "authorized_observation_count": 1,
        }
    )
    body["detectors"] = (
        *report.detectors[:repeated_index],
        forged_minimum,
        *report.detectors[repeated_index + 1 :],
    )
    body["counts"] = report.counts.model_copy(
        update={
            "captured_events": 1,
            "projected_events": 1,
            "action_identities": 1,
        }
    )
    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)

    body = report.model_dump(mode="python")
    body.pop("report_digest")
    over_denominator = repeated.model_copy(update={"authorized_observation_count": 3})
    body["detectors"] = (
        *report.detectors[:repeated_index],
        over_denominator,
        *report.detectors[repeated_index + 1 :],
    )
    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


@pytest.mark.parametrize(
    "change",
    ("manifest_digest", "capability_exclusions", "detector_contract"),
)
def test_report_codec_is_bound_to_the_embedded_capability_manifest(change: str) -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    if change == "manifest_digest":
        body["capability_manifest_digest"] = "f" * 64
    elif change == "capability_exclusions":
        body["coverage"] = report.coverage.model_copy(
            update={"capability_exclusions": ("C:/provider/secret/path",)}
        )
    else:
        repeated_index = tuple(SignalType).index(SignalType.REPEATED_ACTION)
        row = report.detectors[repeated_index]
        body["detectors"] = (
            *report.detectors[:repeated_index],
            row.model_copy(update={"omissions": ("invented_omission",)}),
            *report.detectors[repeated_index + 1 :],
        )

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


def test_detector_matrix_is_exact_and_cannot_be_reordered() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    body["detectors"] = tuple(reversed(report.detectors))

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


def test_report_seal_rejects_coverage_counters_without_matching_limits() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    body["coverage"] = report.coverage.model_copy(update={"gap_count": 1})

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


def test_report_seal_binds_session_state_to_its_exact_boundary_limit() -> None:
    open_report = _report(
        headline=CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
        disposition=ShadowHeuristicDisposition.FLAGGED,
        authorized_actions=2,
        detected=True,
        state=CaptureSessionState.OPEN,
        limits=(CaptureReportLimit.SESSION_OPEN, CaptureReportLimit.SPOOL_PENDING),
        spool_status=CaptureSpoolReportStatus.VERIFIED_PENDING,
        queued_spool_events=1,
    )
    open_body = open_report.model_dump(mode="python")
    open_body.pop("report_digest")
    open_body["coverage"] = open_report.coverage.model_copy(
        update={"limits": (CaptureReportLimit.SPOOL_PENDING,)}
    )

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(open_body)

    closed_report = _report(
        headline=CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.INDETERMINATE,
        authorized_actions=1,
        limits=(CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,),
    )
    closed_body = closed_report.model_dump(mode="python")
    closed_body.pop("report_digest")
    closed_body["coverage"] = closed_report.coverage.model_copy(
        update={
            "coverage_degraded": True,
            "limits": (
                CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,
                CaptureReportLimit.SESSION_OPEN,
            ),
        }
    )

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(closed_body)


def test_report_seal_cannot_hide_unverified_compatibility_from_limits() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    body["host_version"] = "0.0.0"
    body["compatibility_status"] = CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


def test_report_seal_rejects_a_flagged_quarantined_conclusion() -> None:
    report = _report(
        headline=CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
        disposition=ShadowHeuristicDisposition.FLAGGED,
        authorized_actions=2,
        detected=True,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    body["session_state"] = CaptureSessionState.QUARANTINED
    body["interval"] = report.interval.model_copy(update={"closed_at": None})

    with pytest.raises(CaptureReportError):
        _seal_capture_session_report(body)


def test_report_preserves_backward_wall_clock_observations() -> None:
    report = _report(
        headline=CaptureReportHeadline.NO_CURRENT_EVIDENCE,
        disposition=ShadowHeuristicDisposition.NOT_FLAGGED,
        authorized_actions=2,
    )
    body = report.model_dump(mode="python")
    body.pop("report_digest")
    earlier = NOW - timedelta(hours=1)
    body["interval"] = CaptureReportInterval(
        opened_at=NOW,
        updated_at=earlier,
        closed_at=earlier,
    )

    backward_clock_report = _seal_capture_session_report(body)

    assert backward_clock_report.interval.updated_at < backward_clock_report.interval.opened_at
    assert (
        decode_capture_session_report(encode_capture_session_report(backward_clock_report))
        == backward_clock_report
    )


def test_real_store_normalization_and_report_cover_all_three_headlines(
    tmp_path: Path,
) -> None:
    repeated_snapshot, repeated_normalization = _real_capture(
        tmp_path,
        "repeated",
        action_count=2,
        repeated=True,
    )
    clean_snapshot, clean_normalization = _real_capture(
        tmp_path,
        "clean",
        action_count=2,
        repeated=False,
    )
    short_snapshot, short_normalization = _real_capture(
        tmp_path,
        "short",
        action_count=1,
        repeated=False,
    )

    repeated_report = _report_real_capture(
        repeated_snapshot,
        repeated_normalization,
        tmp_path=tmp_path,
    )
    clean_report = _report_real_capture(
        clean_snapshot,
        clean_normalization,
        tmp_path=tmp_path,
    )
    short_report = _report_real_capture(
        short_snapshot,
        short_normalization,
        tmp_path=tmp_path,
    )

    assert repeated_report.headline is CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED
    assert repeated_report.shadow_disposition is ShadowHeuristicDisposition.FLAGGED
    assert clean_report.headline is CaptureReportHeadline.NO_CURRENT_EVIDENCE
    assert clean_report.shadow_disposition is ShadowHeuristicDisposition.NOT_FLAGGED
    assert short_report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert short_report.shadow_disposition is ShadowHeuristicDisposition.INDETERMINATE
    assert CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET in short_report.coverage.limits


def test_real_report_uses_the_exact_profile_detector_matrix(tmp_path: Path) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "matrix",
        action_count=2,
        repeated=False,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)

    assert tuple(item.signal_type for item in report.detectors) == tuple(SignalType)
    assert tuple(item.support for item in report.detectors) == (
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.CONDITIONAL,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
        CapabilitySupport.UNSUPPORTED,
    )
    assert report.detectors[2].minimum_authorized_observations == 2
    assert report.detectors[2].authorized_observation_count == 2


def test_open_real_window_is_bounded_and_insufficient_without_a_flag(
    tmp_path: Path,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "open",
        action_count=2,
        repeated=False,
        close=False,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert report.coverage.capture_scope == "bounded_window"
    assert CaptureReportLimit.SESSION_OPEN in report.coverage.limits


@pytest.mark.parametrize(
    ("action_count", "identity_authority"),
    ((0, "exact"), (2, "coarse")),
)
def test_marker_only_or_coarse_only_window_has_no_applicable_detector(
    tmp_path: Path,
    action_count: int,
    identity_authority: str,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        f"not-applicable-{action_count}-{identity_authority}",
        action_count=action_count,
        repeated=False,
        identity_authority=identity_authority,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert report.shadow_disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
    assert CaptureReportLimit.NO_APPLICABLE_DETECTOR in report.coverage.limits
    assert CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET not in report.coverage.limits


def test_abstained_mixed_authority_window_cannot_claim_signal_absence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed-authority.sqlite3"
    _spool(tmp_path)
    with initialized_store(path) as store:
        register_connection(store)
        started = authenticated_intake("session_started", producer_index=1)
        store.append(started)
        store.append(authenticated_intake("action_started", producer_index=2))
        store.append(
            authenticated_intake(
                "action_started",
                producer_index=3,
                changes={"identity_authority": "coarse"},
            )
        )
        store.append(authenticated_intake("action_started", producer_index=4))
        store.append(authenticated_intake("session_finished", producer_index=5))
        snapshot = store.snapshot_session(CONNECTION_ID, started.session_id)
    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)
    repeated = next(
        item for item in report.detectors if item.signal_type is SignalType.REPEATED_ACTION
    )

    assert repeated.authorized_observation_count == 1
    assert repeated.unresolved_observation_count == 1
    assert repeated.sufficient_for_absence is False
    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET in report.coverage.limits


def test_later_unresolved_action_blocks_an_earlier_met_report_minimum(
    tmp_path: Path,
) -> None:
    context = capture_context()
    target = context.action_identity(b"report-later-unresolved-target")
    _spool(tmp_path)
    with initialized_store(tmp_path / "later-unresolved-report.sqlite3") as store:
        register_connection(store)
        started = authenticated_intake("session_started", producer_index=1)
        store.append(started)
        store.append(authenticated_intake("action_started", producer_index=2))
        store.append(authenticated_intake("action_started", producer_index=3))
        store.append(
            authenticated_intake(
                "action_started",
                producer_index=4,
                changes={"action_digest": target, "identity_authority": "coarse"},
            )
        )
        store.append(
            authenticated_intake(
                "action_started",
                producer_index=5,
                changes={"action_digest": target},
            )
        )
        store.append(authenticated_intake("session_finished", producer_index=6))
        snapshot = store.snapshot_session(CONNECTION_ID, started.session_id)
    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)
    repeated = next(
        item for item in report.detectors if item.signal_type is SignalType.REPEATED_ACTION
    )

    assert repeated.authorized_observation_count == 2
    assert repeated.unresolved_observation_count == 1
    assert repeated.sufficient_for_absence is False
    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET in report.coverage.limits


@pytest.mark.parametrize(
    ("code", "unattributed", "expected_limit"),
    (
        (
            CaptureHealthCode.GAP_DETECTED,
            False,
            CaptureReportLimit.GAP_DETECTED,
        ),
        (
            CaptureHealthCode.UNATTRIBUTED_DROP,
            True,
            CaptureReportLimit.UNATTRIBUTED_DROP,
        ),
        (
            CaptureHealthCode.SESSION_OVERFLOW,
            False,
            CaptureReportLimit.SESSION_OVERFLOW,
        ),
    ),
)
def test_authenticated_gap_drop_and_overflow_block_a_negative_conclusion(
    tmp_path: Path,
    code: CaptureHealthCode,
    unattributed: bool,
    expected_limit: CaptureReportLimit,
) -> None:
    snapshot, _ = _real_capture(
        tmp_path,
        code.value,
        action_count=2,
        repeated=False,
    )
    bounded = snapshot.model_copy(
        update={
            "coverage_degraded": True,
            "unattributed_drop": unattributed,
            "health": (
                CaptureSnapshotHealth(
                    code=code,
                    count=1,
                    lower_bound=1,
                    created_at=snapshot.updated_at,
                    updated_at=snapshot.updated_at,
                ),
            ),
        }
    )
    authenticated = _authenticate_capture_session_snapshot(
        bounded,
        context=capture_context(),
    )
    normalization = normalize_capture_session_snapshot(
        authenticated,
        installation_key=INSTALLATION_KEY,
    )

    report = _report_real_capture(authenticated, normalization, tmp_path=tmp_path)

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert expected_limit in report.coverage.limits


@pytest.mark.parametrize(
    ("error_code", "expected_limit", "counter_name"),
    (
        ("gap_detected", CaptureReportLimit.GAP_DETECTED, "gap_count"),
        ("overflow", CaptureReportLimit.SESSION_OVERFLOW, "overflow_count"),
        (
            "provider_callback_failed",
            CaptureReportLimit.CAPTURE_DEGRADED,
            None,
        ),
    ),
)
def test_controller_failure_records_are_coverage_not_invented_outcomes(
    tmp_path: Path,
    error_code: str,
    expected_limit: CaptureReportLimit,
    counter_name: str | None,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        f"controller-{error_code}",
        action_count=2,
        repeated=False,
        controller_failure=error_code,
    )

    report = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert expected_limit in report.coverage.limits
    if counter_name is not None:
        assert getattr(report.coverage, counter_name) == 1
    assert report.counts.structured_results == 0


def test_none_spool_state_blocks_an_otherwise_sufficient_negative(
    tmp_path: Path,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "no-spool-observation",
        action_count=2,
        repeated=False,
    )

    report = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=None,
    )

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert report.coverage.spool_status is CaptureSpoolReportStatus.UNAVAILABLE
    assert CaptureReportLimit.SPOOL_UNAVAILABLE in report.coverage.limits


@pytest.mark.parametrize(
    ("spool_state", "expected_status", "expected_limit"),
    (
        (
            "pending",
            CaptureSpoolReportStatus.VERIFIED_PENDING,
            CaptureReportLimit.SPOOL_PENDING,
        ),
        (
            "dropped",
            CaptureSpoolReportStatus.VERIFIED_DEGRADED,
            CaptureReportLimit.SPOOL_DROP,
        ),
    ),
)
def test_authenticated_pending_and_dropped_spool_states_block_negative_headlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spool_state: str,
    expected_status: CaptureSpoolReportStatus,
    expected_limit: CaptureReportLimit,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        expected_status.value,
        action_count=2,
        repeated=False,
    )
    spool = _spool(tmp_path)
    if spool_state == "dropped":
        monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    receipt = spool.enqueue(
        authenticated_intake(
            "session_started",
            session_native=f"report-spool-{spool_state}".encode(),
            producer_index=900,
        )
    )
    assert receipt.disposition == ("queued" if spool_state == "pending" else "dropped_quota")

    report = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=spool,
    )

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert report.coverage.spool_status is expected_status
    assert expected_limit in report.coverage.limits


def test_builder_rejects_detached_stale_spool_health(tmp_path: Path) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "invalid-spool",
        action_count=2,
        repeated=False,
    )
    spool = _spool(tmp_path)
    stale = spool.health()
    spool.enqueue(
        authenticated_intake(
            "session_started",
            session_native=b"report-stale-spool",
            producer_index=901,
        )
    )

    with pytest.raises(CaptureReportError):
        build_capture_session_report(
            snapshot,
            normalization,
            installation_key=INSTALLATION_KEY,
            spool=stale,  # type: ignore[arg-type]
        )

    current = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=spool,
    )
    assert current.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert current.coverage.spool_status is CaptureSpoolReportStatus.VERIFIED_PENDING


def test_builder_rejects_a_clean_same_key_spool_from_another_boundary(
    tmp_path: Path,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "wrong-spool-boundary",
        action_count=2,
        repeated=False,
    )
    actual = _spool(tmp_path)
    actual.enqueue(
        authenticated_intake(
            "session_started",
            session_native=b"report-actual-pending-spool",
            producer_index=902,
        )
    )
    alternate = _spool(tmp_path / "alternate-boundary")

    with pytest.raises(CaptureReportError):
        build_capture_session_report(
            snapshot,
            normalization,
            installation_key=INSTALLATION_KEY,
            spool=alternate,
        )

    current = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=actual,
    )
    assert current.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert current.coverage.spool_status is CaptureSpoolReportStatus.VERIFIED_PENDING


def test_builder_rejects_a_same_path_replacement_spool_boundary(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live-state"
    live.mkdir(mode=0o700)
    snapshot, normalization = _real_capture(
        live,
        "replaced-spool-boundary",
        action_count=2,
        repeated=False,
    )
    original = _spool(live)
    original.enqueue(
        authenticated_intake(
            "session_started",
            session_native=b"report-replaced-pending-spool",
            producer_index=904,
        )
    )
    live.rename(tmp_path / "moved-original-state")
    replacement = _spool(live)
    replacement_observation = replacement.observe_health(snapshot.snapshot_digest)

    assert replacement_observation.spool_boundary_digest != snapshot.spool_boundary_digest
    with pytest.raises(CaptureReportError):
        build_capture_session_report(
            snapshot,
            normalization,
            installation_key=INSTALLATION_KEY,
            spool=replacement,
        )


def test_builder_rejects_replacement_of_only_the_spool_directory(
    tmp_path: Path,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "replaced-spool-child",
        action_count=2,
        repeated=False,
    )
    original = _spool(tmp_path)
    original.enqueue(
        authenticated_intake(
            "session_started",
            session_native=b"report-replaced-spool-child",
            producer_index=905,
        )
    )
    (tmp_path / "capture-spool").rename(tmp_path / "moved-original-spool")
    replacement = _spool(tmp_path)
    replacement_observation = replacement.observe_health(snapshot.snapshot_digest)

    assert replacement_observation.spool_boundary_digest != snapshot.spool_boundary_digest
    with pytest.raises(CaptureReportError):
        build_capture_session_report(
            snapshot,
            normalization,
            installation_key=INSTALLATION_KEY,
            spool=replacement,
        )


def test_builder_revalidates_spool_state_after_report_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "spool-revalidation",
        action_count=2,
        repeated=False,
    )
    spool = _spool(tmp_path)
    original_observe = CaptureSpool.observe_health
    mutated = False

    def observe_then_enqueue(
        self: CaptureSpool,
        snapshot_digest: str,
    ) -> CaptureSpoolObservation:
        nonlocal mutated
        observation = original_observe(self, snapshot_digest)
        if self is spool and not mutated:
            mutated = True
            self.enqueue(
                authenticated_intake(
                    "session_started",
                    session_native=b"report-racing-spool-entry",
                    producer_index=903,
                )
            )
        return observation

    monkeypatch.setattr(CaptureSpool, "observe_health", observe_then_enqueue)

    with pytest.raises(CaptureReportError):
        build_capture_session_report(
            snapshot,
            normalization,
            installation_key=INSTALLATION_KEY,
            spool=spool,
        )

    assert mutated is True
    assert spool.health().queued_events == 1


def test_same_verified_snapshot_normalization_key_and_spool_render_identically(
    tmp_path: Path,
) -> None:
    snapshot, normalization = _real_capture(
        tmp_path,
        "deterministic",
        action_count=2,
        repeated=False,
    )

    first = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)
    second = _report_real_capture(snapshot, normalization, tmp_path=tmp_path)

    assert encode_capture_session_report(first) == encode_capture_session_report(second)
    assert first.report_digest == second.report_digest


def test_builder_rejects_snapshot_mismatch_and_digest_forgery(tmp_path: Path) -> None:
    first_snapshot, first_normalization = _real_capture(
        tmp_path,
        "binding-one",
        action_count=2,
        repeated=False,
    )
    second_snapshot, _ = _real_capture(
        tmp_path,
        "binding-two",
        action_count=1,
        repeated=False,
    )
    forged_normalization = first_normalization.model_copy(update={"normalization_digest": "0" * 64})
    forged_snapshot = first_snapshot.model_copy(update={"snapshot_digest": "0" * 64})

    with pytest.raises(CaptureReportError):
        _report_real_capture(second_snapshot, first_normalization, tmp_path=tmp_path)
    with pytest.raises(CaptureReportError):
        _report_real_capture(first_snapshot, forged_normalization, tmp_path=tmp_path)
    with pytest.raises(CaptureReportError):
        _report_real_capture(forged_snapshot, first_normalization, tmp_path=tmp_path)
