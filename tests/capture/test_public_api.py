from __future__ import annotations

import saliencegate.capture as public_capture
from saliencegate.capture import locations, migrations, publication, spool, store


def test_capture_root_exports_the_stable_store_contract() -> None:
    expected = {
        "MAX_CAPTURE_EVENTS_PER_SESSION": store.MAX_CAPTURE_EVENTS_PER_SESSION,
        "MAX_CAPTURE_SPOOL_BYTES": spool.MAX_CAPTURE_SPOOL_BYTES,
        "MAX_CAPTURE_SPOOL_EVENTS": spool.MAX_CAPTURE_SPOOL_EVENTS,
        "CaptureAdmissionSource": store.CaptureAdmissionSource,
        "CaptureAppendDisposition": store.CaptureAppendDisposition,
        "CaptureAppendReceipt": store.CaptureAppendReceipt,
        "CaptureConnectionRegistration": store.CaptureConnectionRegistration,
        "CaptureConnectionState": store.CaptureConnectionState,
        "CaptureConnectionTransition": store.CaptureConnectionTransition,
        "CaptureIntakeAuthenticationError": publication.CaptureIntakeAuthenticationError,
        "CaptureLocationError": locations.CaptureLocationError,
        "CaptureMigrationError": migrations.CaptureMigrationError,
        "CaptureMigrationIntegrityError": migrations.CaptureMigrationIntegrityError,
        "CaptureMigrationReceipt": migrations.CaptureMigrationReceipt,
        "CaptureSchemaTooNewError": migrations.CaptureSchemaTooNewError,
        "CaptureSessionState": store.CaptureSessionState,
        "CaptureSessionVerification": store.CaptureSessionVerification,
        "CaptureSpool": spool.CaptureSpool,
        "CaptureSpoolDrainReceipt": spool.CaptureSpoolDrainReceipt,
        "CaptureSpoolEnqueueReceipt": spool.CaptureSpoolEnqueueReceipt,
        "CaptureSpoolError": spool.CaptureSpoolError,
        "CaptureSpoolHealth": spool.CaptureSpoolHealth,
        "CaptureSpoolIntegrityError": spool.CaptureSpoolIntegrityError,
        "CaptureStore": store.CaptureStore,
        "CaptureStoreBusyError": store.CaptureStoreBusyError,
        "CaptureStoreClosedError": store.CaptureStoreClosedError,
        "CaptureStoreError": store.CaptureStoreError,
        "CaptureStoreIntegrityError": store.CaptureStoreIntegrityError,
        "CaptureStoreLocations": locations.CaptureStoreLocations,
        "CaptureStoreMode": store.CaptureStoreMode,
        "CaptureStoreStateError": store.CaptureStoreStateError,
        "admit_capture_intake": spool.admit_capture_intake,
        "authenticate_capture_intake": publication.authenticate_capture_intake,
        "initialize_capture_store": migrations.initialize_capture_store,
        "resolve_capture_store_locations": locations.resolve_capture_store_locations,
        "verify_capture_intake_authentication": (publication.verify_capture_intake_authentication),
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
