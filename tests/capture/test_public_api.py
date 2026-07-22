from __future__ import annotations

import json
import subprocess
import sys

import saliencegate.capture as public_capture
from saliencegate.capture import (
    locations,
    migrations,
    normalization,
    publication,
    report,
    sessions,
    spool,
    store,
    transport,
)


def test_capture_root_defers_submodules_until_a_public_name_is_accessed() -> None:
    program = """
import json
import sys

import saliencegate.capture as capture

loaded_before = sorted(
    name for name in sys.modules if name.startswith("saliencegate.capture.")
)
first = capture.CaptureProfile
second = capture.CaptureProfile
loaded_after = sorted(
    name for name in sys.modules if name.startswith("saliencegate.capture.")
)
print(json.dumps({
    "cached": first is second and capture.__dict__["CaptureProfile"] is first,
    "declared": "CaptureProfile" in capture.__all__,
    "discoverable": "CaptureProfile" in dir(capture),
    "loaded_before": loaded_before,
    "loaded_after": loaded_after,
}))
"""

    completed = subprocess.run(
        (sys.executable, "-c", program),
        capture_output=True,
        check=False,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    observation = json.loads(completed.stdout)
    assert observation["loaded_before"] == []
    assert observation["cached"] is True
    assert observation["declared"] is True
    assert observation["discoverable"] is True
    assert "saliencegate.capture.capabilities" in observation["loaded_after"]
    assert "saliencegate.capture.normalization" not in observation["loaded_after"]
    assert "saliencegate.capture.report" not in observation["loaded_after"]


def test_capture_root_star_import_resolves_every_declared_export() -> None:
    program = """
import json
import saliencegate.capture as capture
from saliencegate.capture import *

scope = locals()
print(json.dumps({
    "missing": sorted(name for name in capture.__all__ if name not in scope),
    "mismatched": sorted(
        name for name in capture.__all__
        if name in scope and scope[name] is not getattr(capture, name)
    ),
}))
"""

    completed = subprocess.run(
        (sys.executable, "-c", program),
        capture_output=True,
        check=False,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {"mismatched": [], "missing": []}


def test_capture_root_exports_the_stable_store_contract() -> None:
    expected = {
        "CAPTURE_SESSION_REPORT_SCHEMA_VERSION": (report.CAPTURE_SESSION_REPORT_SCHEMA_VERSION),
        "MAX_CAPTURE_EVENTS_PER_SESSION": store.MAX_CAPTURE_EVENTS_PER_SESSION,
        "MAX_CAPTURE_SPOOL_BYTES": spool.MAX_CAPTURE_SPOOL_BYTES,
        "MAX_CAPTURE_SPOOL_EVENTS": spool.MAX_CAPTURE_SPOOL_EVENTS,
        "MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION": (
            transport.MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION
        ),
        "CaptureAdmissionSource": store.CaptureAdmissionSource,
        "CaptureAppendDisposition": store.CaptureAppendDisposition,
        "CaptureAppendReceipt": store.CaptureAppendReceipt,
        "CaptureConnectionRegistration": store.CaptureConnectionRegistration,
        "CaptureConnectionState": store.CaptureConnectionState,
        "CaptureConnectionTransition": store.CaptureConnectionTransition,
        "CaptureDetectorEvidence": normalization.CaptureDetectorEvidence,
        "CaptureHumanSessionId": sessions.CaptureHumanSessionId,
        "CaptureIntakeAuthenticationError": publication.CaptureIntakeAuthenticationError,
        "CaptureLocationError": locations.CaptureLocationError,
        "CaptureMigrationError": migrations.CaptureMigrationError,
        "CaptureMigrationIntegrityError": migrations.CaptureMigrationIntegrityError,
        "CaptureMigrationReceipt": migrations.CaptureMigrationReceipt,
        "CaptureNormalization": normalization.CaptureNormalization,
        "CaptureNormalizationCounts": normalization.CaptureNormalizationCounts,
        "CaptureNormalizationDiagnostic": normalization.CaptureNormalizationDiagnostic,
        "CaptureNormalizationDiagnosticCode": (normalization.CaptureNormalizationDiagnosticCode),
        "CaptureNormalizationError": normalization.CaptureNormalizationError,
        "CaptureReportCounts": report.CaptureReportCounts,
        "CaptureReportCoverage": report.CaptureReportCoverage,
        "CaptureReportDetector": report.CaptureReportDetector,
        "CaptureReportError": report.CaptureReportError,
        "CaptureReportHeadline": report.CaptureReportHeadline,
        "CaptureReportHealthCount": report.CaptureReportHealthCount,
        "CaptureReportInterval": report.CaptureReportInterval,
        "CaptureReportLimit": report.CaptureReportLimit,
        "CaptureSchemaTooNewError": migrations.CaptureSchemaTooNewError,
        "CaptureSessionState": store.CaptureSessionState,
        "CaptureSessionReport": report.CaptureSessionReport,
        "CaptureSessionSnapshot": sessions.CaptureSessionSnapshot,
        "CaptureSessionSnapshotError": sessions.CaptureSessionSnapshotError,
        "CaptureSessionVerification": store.CaptureSessionVerification,
        "CaptureSpool": spool.CaptureSpool,
        "CaptureSpoolDrainReceipt": spool.CaptureSpoolDrainReceipt,
        "CaptureSpoolEnqueueReceipt": spool.CaptureSpoolEnqueueReceipt,
        "CaptureSpoolError": spool.CaptureSpoolError,
        "CaptureSpoolHealth": spool.CaptureSpoolHealth,
        "CaptureSpoolIntegrityError": spool.CaptureSpoolIntegrityError,
        "CaptureSpoolObservation": spool.CaptureSpoolObservation,
        "CaptureSpoolObservationError": spool.CaptureSpoolObservationError,
        "CaptureSpoolReportStatus": report.CaptureSpoolReportStatus,
        "CaptureSnapshotEvent": sessions.CaptureSnapshotEvent,
        "CaptureSnapshotHealth": sessions.CaptureSnapshotHealth,
        "CaptureStore": store.CaptureStore,
        "CaptureStoreBusyError": store.CaptureStoreBusyError,
        "CaptureStoreClosedError": store.CaptureStoreClosedError,
        "CaptureStoreError": store.CaptureStoreError,
        "CaptureStoreIntegrityError": store.CaptureStoreIntegrityError,
        "CaptureStoreLocations": locations.CaptureStoreLocations,
        "CaptureStoreMode": store.CaptureStoreMode,
        "CaptureStoreStateError": store.CaptureStoreStateError,
        "CaptureTransportChunk": transport.CaptureTransportChunk,
        "CaptureTransportDisposition": transport.CaptureTransportDisposition,
        "CaptureTransportError": transport.CaptureTransportError,
        "CaptureTransportReceipt": transport.CaptureTransportReceipt,
        "admit_capture_intake": spool.admit_capture_intake,
        "authenticate_capture_intake": publication.authenticate_capture_intake,
        "build_capture_session_report": report.build_capture_session_report,
        "decode_capture_session_report": report.decode_capture_session_report,
        "encode_capture_session_report": report.encode_capture_session_report,
        "initialize_capture_store": migrations.initialize_capture_store,
        "normalize_capture_session_snapshot": (normalization.normalize_capture_session_snapshot),
        "render_capture_session_report_human": (report.render_capture_session_report_human),
        "render_capture_session_report_json": report.render_capture_session_report_json,
        "resolve_capture_store_locations": locations.resolve_capture_store_locations,
        "verify_capture_intake_authentication": (publication.verify_capture_intake_authentication),
        "verify_capture_normalization": normalization.verify_capture_normalization,
        "verify_capture_session_snapshot": sessions.verify_capture_session_snapshot,
        "verify_capture_spool_observation": spool.verify_capture_spool_observation,
        "validate_capture_transport_chunk": transport.validate_capture_transport_chunk,
    }

    assert len(public_capture.__all__) == len(set(public_capture.__all__))
    assert expected.keys() <= set(public_capture.__all__)
    for name, target in expected.items():
        assert getattr(public_capture, name) is target


def test_capture_root_does_not_publish_migration_or_integrity_internals() -> None:
    assert {
        "APPLICATION_ID",
        "LATEST_SCHEMA_VERSION",
        "CaptureHealthIdentityMaterial",
        "CaptureHealthIntegrityMaterial",
        "CaptureMigration",
        "apply_capture_migrations",
        "capture_health_identity_material",
        "capture_health_integrity_material",
        "discover_capture_migrations",
        "validate_capture_store_schema",
    }.isdisjoint(public_capture.__all__)
