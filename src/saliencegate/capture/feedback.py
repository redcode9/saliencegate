"""Local, content-free feedback and descriptive classification evaluation."""

from __future__ import annotations

import hmac
from collections import defaultdict
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.schema import CaptureJSONLimits, read_bounded_json
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import Sha256Digest, UtcDatetime
from saliencegate.security import InstallationKey

if TYPE_CHECKING:
    from saliencegate.capture.normalization import CaptureNormalization
    from saliencegate.capture.report import CaptureSessionReport
    from saliencegate.capture.sessions import CaptureSessionSnapshot
    from saliencegate.capture.spool import CaptureSpool


MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION: Final = 1_000
MAX_CAPTURE_FEEDBACK_EXAMPLES: Final = 1_000
CAPTURE_E01_MIN_FINAL_TEST_SESSIONS: Final = 200
CAPTURE_E01_MIN_PROJECTS: Final = 3
CAPTURE_E01_MIN_PROVIDERS: Final = 2
CAPTURE_E01_MIN_MEMORY_NEEDED: Final = 30
CAPTURE_E01_MIN_NOT_MEMORY_NEEDED: Final = 30
CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES: Final = 2_000
CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR: Final = 30
CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT: Final = 30
CAPTURE_CALIBRATION_BOOTSTRAP_SEED: Final = (
    "9f4c8dc1d7f87c2bf08bfc24f9cb6bb4de27c57fa3466a7a63d7f01e13961e7e"
)
CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT: Final[
    Literal["0ba351d0bc979e0f7280c10f3f36c6d4da3b9df59b4b8b0c664d504ecd89ddc3"]
] = "0ba351d0bc979e0f7280c10f3f36c6d4da3b9df59b4b8b0c664d504ecd89ddc3"
CAPTURE_FEEDBACK_EVALUATOR_VERSION: Final = "capture-headline-three-way/v1"

_PPM: Final = 1_000_000
_BOOTSTRAP_LOWER_RANK: Final = 50
_BOOTSTRAP_UPPER_RANK: Final = 1_950
_BOOTSTRAP_REJECTION_ATTEMPTS: Final = 16
_CONFUSION_CELL_NAMES: Final = (
    "true_positive",
    "false_negative",
    "abstained_memory_needed",
    "false_positive",
    "true_negative",
    "abstained_not_memory_needed",
    "uncertain_predicted_memory_needed",
    "uncertain_predicted_not_memory_needed",
    "uncertain_prediction_abstained",
)
_METRIC_NAMES: Final = (
    "prevalence",
    "precision",
    "recall",
    "false_positive_rate",
    "reference_abstention_rate",
    "prediction_abstention_rate",
    "joint_abstention_rate",
)
_DATASET_DIGEST_DOMAIN: Final = "saliencegate:capture:feedback-dataset:v1"
_DATASET_TAG_DOMAIN: Final = b"saliencegate:capture:feedback-dataset-tag:v1"
_REPORT_DIGEST_DOMAIN: Final = "saliencegate:capture:calibration-report:v1"
_REPORT_TAG_DOMAIN: Final = b"saliencegate:capture:calibration-report-tag:v1"
_EXPORT_ID_DOMAIN: Final = "saliencegate:capture:feedback-export:v1"
_EXAMPLE_ID_DOMAIN: Final = b"saliencegate:capture:feedback-example:v1"
_PROJECT_ID_DOMAIN: Final = b"saliencegate:capture:feedback-project:v1"
_REPORT_BINDING_DOMAIN: Final = b"saliencegate:capture:feedback-report-binding:v1"
_EXPORT_RECORD_DOMAIN: Final = b"saliencegate:capture:feedback-export-record:v1"
_FEEDBACK_RECORD_DOMAIN: Final = b"saliencegate:capture:feedback-record:v1"

_FEEDBACK_JSON_LIMITS = CaptureJSONLimits(
    max_bytes=4 * 1_024 * 1_024,
    max_depth=12,
    max_items=50_000,
    max_string_bytes=1_024 * 1_024,
)


class CaptureFeedbackError(ValueError):
    """A stable, content-free feedback or evaluation failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture feedback data is invalid")


class CaptureFeedbackLabel(StrEnum):
    MEMORY_NEEDED = "memory-needed"
    NOT_MEMORY_NEEDED = "not-memory-needed"
    UNCERTAIN = "uncertain"


class CaptureFeedbackWriteDisposition(StrEnum):
    RECORDED = "recorded"
    UNCHANGED = "unchanged"
    CHANGED = "changed"


class CaptureFeedbackPrediction(StrEnum):
    MEMORY_NEEDED = "memory-needed"
    NOT_MEMORY_NEEDED = "not-memory-needed"
    ABSTAIN = "abstain"


class CaptureFeedbackPartition(StrEnum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    FINAL_TEST = "final_test"
    UNASSIGNED = "unassigned"


class CaptureFeedbackEvidenceSource(StrEnum):
    SYNTHETIC = "synthetic"
    LOCAL_FEEDBACK = "local_feedback"
    DECLARED_E01 = "declared_e01_external_review"


class CaptureFeedbackRecordOrigin(StrEnum):
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    LOCAL_CAPTURE_REPORT = "local_capture_report"


class CaptureFeedbackBlindingStatus(StrEnum):
    EXTERNALLY_ATTESTED = "externally_attested"
    NOT_BLINDED = "not_blinded"
    UNKNOWN = "unknown"


class CaptureCalibrationEvidenceStatus(StrEnum):
    INSUFFICIENT_REAL_WORLD_EVIDENCE = "insufficient_real_world_evidence"


class CaptureCalibrationInsufficiency(StrEnum):
    NON_E01_EVIDENCE = "non_e01_evidence"
    DEVELOPMENT_COHORT = "development_or_calibration_cohort_missing"
    FINAL_TEST_SUPPORT = "final_test_support_below_200"
    PROJECT_SUPPORT = "final_test_project_support_below_3"
    PROVIDER_SUPPORT = "final_test_provider_support_below_2"
    MEMORY_NEEDED_SUPPORT = "final_test_memory_needed_support_below_30"
    NOT_MEMORY_NEEDED_SUPPORT = "final_test_not_memory_needed_support_below_30"
    PROCESS_ATTESTATION = "external_process_attestation_missing"
    PROVIDER_DISCLOSURE = "provider_stratum_below_disclosure_floor_30"
    METRIC_INTERVAL = "descriptive_metric_interval_unavailable"
    EXTERNAL_REVIEW = "external_review_required"


class CaptureCalibrationIntervalStatus(StrEnum):
    ESTIMATED = "estimated"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    UNDEFINED_BOOTSTRAP_REPLICATE = "undefined_bootstrap_replicate"


class CaptureCalibrationBoundaryStatus(StrEnum):
    INTERIOR = "interior"
    DEGENERATE_ZERO_WIDTH = "degenerate_zero_width"


def _require_exact_text(value: str) -> str:
    if type(value) is not str:
        raise ValueError("feedback text is invalid")
    return value


_HumanSessionId: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=12, max_length=52, pattern=r"^[a-z2-7]+$"),
    AfterValidator(_require_exact_text),
]
_BoundedCount: TypeAlias = Annotated[int, Field(ge=0, le=MAX_CAPTURE_FEEDBACK_EXAMPLES)]
_PositiveRevision: TypeAlias = Annotated[
    int,
    Field(ge=1, le=MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION),
]
_PartsPerMillion: TypeAlias = Annotated[int, Field(ge=0, le=_PPM)]


class _FeedbackModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class CaptureFeedbackRevision(_FeedbackModel):
    schema_version: Literal["capture-feedback-revision/v1"] = "capture-feedback-revision/v1"
    session_id: Annotated[_HumanSessionId, Field(repr=False)]
    label: CaptureFeedbackLabel
    revision: _PositiveRevision
    created_at: Annotated[UtcDatetime, Field(repr=False)]


class CaptureFeedbackRecord(_FeedbackModel):
    schema_version: Literal["capture-feedback-record/v1"] = "capture-feedback-record/v1"
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    human_id: Annotated[_HumanSessionId, Field(repr=False)]
    label: CaptureFeedbackLabel
    revision_count: _PositiveRevision
    labeled_at: Annotated[UtcDatetime, Field(repr=False)]
    record_tag: Annotated[Sha256Digest, Field(repr=False)]


class CaptureFeedbackReceipt(_FeedbackModel):
    schema_version: Literal["capture-feedback-receipt/v1"] = "capture-feedback-receipt/v1"
    session_id: Annotated[_HumanSessionId, Field(repr=False)]
    label: CaptureFeedbackLabel
    disposition: CaptureFeedbackWriteDisposition
    revision_count: _PositiveRevision
    labeled_at: Annotated[UtcDatetime, Field(repr=False)]

    @model_validator(mode="after")
    def disposition_matches_revision(self) -> Self:
        if (
            self.disposition is CaptureFeedbackWriteDisposition.RECORDED
            and self.revision_count != 1
        ) or (
            self.disposition is CaptureFeedbackWriteDisposition.CHANGED and self.revision_count < 2
        ):
            raise ValueError("feedback disposition is inconsistent")
        return self


class CaptureFeedbackStudyAttestation(_FeedbackModel):
    """Opaque commitments supplied by an external, consented study process."""

    schema_version: Literal["capture-feedback-study-attestation/v1"] = (
        "capture-feedback-study-attestation/v1"
    )
    consent_protocol_digest: Sha256Digest
    preregistration_digest: Sha256Digest
    temporal_split_digest: Sha256Digest
    evaluator_freeze_digest: Sha256Digest
    label_freeze_digest: Sha256Digest
    report_selection_policy_digest: Sha256Digest
    no_test_tuning_digest: Sha256Digest
    blinding_status: CaptureFeedbackBlindingStatus


class CaptureFeedbackExportRecord(_FeedbackModel):
    """Internal report-bound input; raw local identifiers are never serialized."""

    schema_version: Literal["capture-feedback-export-record/v1"] = (
        "capture-feedback-export-record/v1"
    )
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    label: CaptureFeedbackLabel
    prediction: CaptureFeedbackPrediction
    partition: CaptureFeedbackPartition
    origin: CaptureFeedbackRecordOrigin
    source_report_digest: Annotated[Sha256Digest, Field(repr=False)]
    evaluator_version: Literal["capture-headline-three-way/v1"] = CAPTURE_FEEDBACK_EVALUATOR_VERSION
    record_tag: Annotated[Sha256Digest, Field(repr=False)]


class CaptureFeedbackDatasetExample(_FeedbackModel):
    example_id: Annotated[Sha256Digest, Field(repr=False)]
    project_id: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    label: CaptureFeedbackLabel
    prediction: CaptureFeedbackPrediction
    partition: CaptureFeedbackPartition
    report_binding: Annotated[Sha256Digest, Field(repr=False)]


class CaptureFeedbackDataset(_FeedbackModel):
    schema_version: Literal["capture-feedback-dataset/v1"] = "capture-feedback-dataset/v1"
    evidence_source: CaptureFeedbackEvidenceSource
    study_attestation: CaptureFeedbackStudyAttestation | None
    explicit_opt_in: Literal[True] = True
    raw_content_included: Literal[False] = False
    direct_identifiers_included: Literal[False] = False
    identifier_scope: Literal["export_specific_hmac"] = "export_specific_hmac"
    export_id: Annotated[Sha256Digest, Field(repr=False)]
    examples: Annotated[
        tuple[CaptureFeedbackDatasetExample, ...],
        Field(max_length=MAX_CAPTURE_FEEDBACK_EXAMPLES, repr=False),
    ]
    dataset_digest: Annotated[Sha256Digest, Field(repr=False)]
    dataset_tag: Annotated[Sha256Digest, Field(repr=False)]

    @model_validator(mode="after")
    def dataset_is_canonical_and_bound(self) -> Self:
        identities = tuple(item.example_id for item in self.examples)
        if (
            identities != tuple(sorted(set(identities)))
            or any(item.partition is CaptureFeedbackPartition.UNASSIGNED for item in self.examples)
            or (self.evidence_source is CaptureFeedbackEvidenceSource.DECLARED_E01)
            is not (self.study_attestation is not None)
            or not hmac.compare_digest(_dataset_body_digest(self), self.dataset_digest)
        ):
            raise ValueError("feedback dataset is inconsistent")
        return self


def _model_is_exact(model: type[BaseModel], value: object) -> bool:
    try:
        return (
            type(value) is model
            and type(value.__dict__) is dict
            and set(value.__dict__) == set(model.model_fields)
            and value.__pydantic_extra__ is None
            and value.__pydantic_private__ is None
        )
    except Exception:
        return False


def _dataset_body(value: CaptureFeedbackDataset) -> dict[str, object]:
    return {
        key: item
        for key, item in value.model_dump(mode="json", warnings=False).items()
        if key not in {"dataset_digest", "dataset_tag"}
    }


def _dataset_body_digest(value: CaptureFeedbackDataset) -> str:
    return length_prefixed_sha256(
        canonical_json(_dataset_body(value)),
        domain=_DATASET_DIGEST_DOMAIN,
    )


def _dataset_tag_material(value: CaptureFeedbackDataset) -> dict[str, object]:
    return {
        key: item
        for key, item in value.model_dump(mode="json", warnings=False).items()
        if key != "dataset_tag"
    }


def _verified_dataset(
    value: object,
    *,
    installation_key: InstallationKey,
) -> CaptureFeedbackDataset:
    if type(installation_key) is not InstallationKey or not _model_is_exact(
        CaptureFeedbackDataset, value
    ):
        raise CaptureFeedbackError()
    assert isinstance(value, CaptureFeedbackDataset)
    expected = installation_key._hmac_sha256(
        canonical_json(_dataset_tag_material(value)),
        domain=_DATASET_TAG_DOMAIN,
    )
    if not hmac.compare_digest(value.dataset_tag, expected):
        raise CaptureFeedbackError()
    return value


def _export_record_material(record: CaptureFeedbackExportRecord) -> dict[str, object]:
    return {
        key: value
        for key, value in record.model_dump(mode="json", warnings=False).items()
        if key != "record_tag"
    }


def _feedback_record_material(record: CaptureFeedbackRecord) -> dict[str, object]:
    return {
        key: value
        for key, value in record.model_dump(mode="json", warnings=False).items()
        if key != "record_tag"
    }


def _verified_feedback_record(
    value: object,
    *,
    installation_key: InstallationKey,
) -> CaptureFeedbackRecord:
    if not _model_is_exact(CaptureFeedbackRecord, value):
        raise CaptureFeedbackError()
    assert isinstance(value, CaptureFeedbackRecord)
    expected = installation_key._hmac_sha256(
        canonical_json(_feedback_record_material(value)),
        domain=_FEEDBACK_RECORD_DOMAIN,
    )
    if not hmac.compare_digest(value.record_tag, expected):
        raise CaptureFeedbackError()
    return value


def _seal_export_record(
    body: Mapping[str, object],
    *,
    installation_key: InstallationKey,
) -> CaptureFeedbackExportRecord:
    if type(installation_key) is not InstallationKey or "record_tag" in body:
        raise CaptureFeedbackError()
    try:
        tag = installation_key._hmac_sha256(
            canonical_json(dict(body)),
            domain=_EXPORT_RECORD_DOMAIN,
        )
        return CaptureFeedbackExportRecord.model_validate({**body, "record_tag": tag})
    except Exception:
        raise CaptureFeedbackError() from None


def build_synthetic_capture_feedback_export_record(
    *,
    project_digest: str,
    profile_id: CaptureProfile,
    session_id: str,
    label: CaptureFeedbackLabel,
    prediction: CaptureFeedbackPrediction,
    partition: CaptureFeedbackPartition,
    installation_key: InstallationKey,
) -> CaptureFeedbackExportRecord:
    """Create an explicitly synthetic input that cannot be relabeled as E01 evidence."""

    try:
        if (
            type(profile_id) is not CaptureProfile
            or type(label) is not CaptureFeedbackLabel
            or type(prediction) is not CaptureFeedbackPrediction
            or type(partition) is not CaptureFeedbackPartition
            or partition is CaptureFeedbackPartition.UNASSIGNED
            or type(installation_key) is not InstallationKey
        ):
            raise ValueError
        source_digest = length_prefixed_sha256(
            project_digest,
            session_id,
            profile_id.value,
            label.value,
            prediction.value,
            partition.value,
            domain="saliencegate:capture:synthetic-feedback-source:v1",
        )
        return _seal_export_record(
            {
                "schema_version": "capture-feedback-export-record/v1",
                "project_digest": project_digest,
                "profile_id": profile_id,
                "session_id": session_id,
                "label": label,
                "prediction": prediction,
                "partition": partition,
                "origin": CaptureFeedbackRecordOrigin.SYNTHETIC_FIXTURE,
                "source_report_digest": source_digest,
                "evaluator_version": CAPTURE_FEEDBACK_EVALUATOR_VERSION,
            },
            installation_key=installation_key,
        )
    except CaptureFeedbackError:
        raise
    except Exception:
        raise CaptureFeedbackError() from None


def build_capture_feedback_export_record(
    record: CaptureFeedbackRecord,
    report: CaptureSessionReport,
    *,
    snapshot: CaptureSessionSnapshot,
    normalization: CaptureNormalization,
    spool: CaptureSpool,
    partition: CaptureFeedbackPartition,
    label_freeze: datetime,
    installation_key: InstallationKey,
) -> CaptureFeedbackExportRecord:
    """Bind one frozen local label to one closed, structured capture report."""

    try:
        from saliencegate.capture.normalization import verify_capture_normalization
        from saliencegate.capture.report import (
            CaptureReportHeadline,
            CaptureSessionReport,
            build_capture_session_report,
            decode_capture_session_report,
            encode_capture_session_report,
        )
        from saliencegate.capture.sessions import verify_capture_session_snapshot
        from saliencegate.capture.spool import CaptureSpool
        from saliencegate.capture.store import CaptureSessionState

        checked_record = _verified_feedback_record(
            record,
            installation_key=installation_key,
        )
        if (
            type(report) is not CaptureSessionReport
            or type(spool) is not CaptureSpool
            or type(partition) is not CaptureFeedbackPartition
            or partition is CaptureFeedbackPartition.UNASSIGNED
            or type(label_freeze) is not datetime
            or label_freeze.tzinfo is None
            or label_freeze.utcoffset() != timedelta(0)
            or label_freeze != label_freeze.astimezone(UTC)
            or type(installation_key) is not InstallationKey
        ):
            raise ValueError
        checked = decode_capture_session_report(encode_capture_session_report(report))
        checked_snapshot = verify_capture_session_snapshot(
            snapshot,
            installation_key=installation_key,
        )
        checked_normalization = verify_capture_normalization(
            normalization,
            snapshot=checked_snapshot,
            installation_key=installation_key,
        )
        expected_report = build_capture_session_report(
            checked_snapshot,
            checked_normalization,
            installation_key=installation_key,
            spool=spool,
        )
        if (
            not hmac.compare_digest(
                encode_capture_session_report(checked),
                encode_capture_session_report(expected_report),
            )
            or checked_snapshot.session_id != checked_record.session_id
            or checked_snapshot.human_id != checked_record.human_id
            or checked_snapshot.profile_id is not checked_record.profile_id
            or checked.session_state is not CaptureSessionState.CLOSED
            or checked.interval.closed_at is None
            or checked.session_id != checked_record.human_id
            or checked.profile_id is not checked_record.profile_id
            or checked_record.labeled_at < checked.interval.closed_at
            or checked_record.labeled_at >= label_freeze
        ):
            raise ValueError
        prediction = {
            CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED: (
                CaptureFeedbackPrediction.MEMORY_NEEDED
            ),
            CaptureReportHeadline.NO_CURRENT_EVIDENCE: (
                CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
            ),
            CaptureReportHeadline.INSUFFICIENT_EVIDENCE: CaptureFeedbackPrediction.ABSTAIN,
        }[checked.headline]
        return _seal_export_record(
            {
                "schema_version": "capture-feedback-export-record/v1",
                "project_digest": checked_record.project_digest,
                "profile_id": checked_record.profile_id,
                "session_id": checked_record.session_id,
                "label": checked_record.label,
                "prediction": prediction,
                "partition": partition,
                "origin": CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT,
                "source_report_digest": checked.report_digest,
                "evaluator_version": CAPTURE_FEEDBACK_EVALUATOR_VERSION,
            },
            installation_key=installation_key,
        )
    except CaptureFeedbackError:
        raise
    except Exception:
        raise CaptureFeedbackError() from None


def _verified_export_record(
    value: object,
    *,
    installation_key: InstallationKey,
) -> CaptureFeedbackExportRecord:
    if not _model_is_exact(CaptureFeedbackExportRecord, value):
        raise CaptureFeedbackError()
    assert isinstance(value, CaptureFeedbackExportRecord)
    expected = installation_key._hmac_sha256(
        canonical_json(_export_record_material(value)),
        domain=_EXPORT_RECORD_DOMAIN,
    )
    if not hmac.compare_digest(value.record_tag, expected):
        raise CaptureFeedbackError()
    return value


def build_capture_feedback_dataset(
    records: tuple[CaptureFeedbackExportRecord, ...],
    *,
    installation_key: InstallationKey,
    export_nonce: bytes,
    evidence_source: CaptureFeedbackEvidenceSource,
    opt_in: bool,
    study_attestation: CaptureFeedbackStudyAttestation | None = None,
) -> CaptureFeedbackDataset:
    """Build one bounded, unlinkable export only after an exact opt-in."""

    try:
        if (
            type(records) is not tuple
            or len(records) > MAX_CAPTURE_FEEDBACK_EXAMPLES
            or type(installation_key) is not InstallationKey
            or type(export_nonce) is not bytes
            or len(export_nonce) != 32
            or type(evidence_source) is not CaptureFeedbackEvidenceSource
            or type(opt_in) is not bool
            or not opt_in
            or (
                study_attestation is not None
                and not _model_is_exact(CaptureFeedbackStudyAttestation, study_attestation)
            )
        ):
            raise ValueError
        checked = tuple(
            _verified_export_record(item, installation_key=installation_key) for item in records
        )
        internal_sessions = tuple(item.session_id for item in checked)
        if len(set(internal_sessions)) != len(internal_sessions):
            raise ValueError
        origins = {item.origin for item in checked}
        if evidence_source is CaptureFeedbackEvidenceSource.SYNTHETIC:
            if origins - {CaptureFeedbackRecordOrigin.SYNTHETIC_FIXTURE} or study_attestation:
                raise ValueError
        elif evidence_source is CaptureFeedbackEvidenceSource.LOCAL_FEEDBACK:
            if origins - {CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT} or study_attestation:
                raise ValueError
        elif (
            origins - {CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT}
            or study_attestation is None
        ):
            raise ValueError

        export_id = length_prefixed_sha256(export_nonce, domain=_EXPORT_ID_DOMAIN)
        examples: list[CaptureFeedbackDatasetExample] = []
        for item in checked:
            example_id = installation_key._hmac_sha256(
                canonical_json(
                    {
                        "schema_version": "capture-feedback-example-id/v1",
                        "export_id": export_id,
                        "session_id": item.session_id,
                    }
                ),
                domain=_EXAMPLE_ID_DOMAIN,
            )
            project_id = installation_key._hmac_sha256(
                canonical_json(
                    {
                        "schema_version": "capture-feedback-project-id/v1",
                        "export_id": export_id,
                        "project_digest": item.project_digest,
                    }
                ),
                domain=_PROJECT_ID_DOMAIN,
            )
            report_binding = installation_key._hmac_sha256(
                canonical_json(
                    {
                        "schema_version": "capture-feedback-report-binding/v1",
                        "export_id": export_id,
                        "report_digest": item.source_report_digest,
                        "evaluator_version": item.evaluator_version,
                    }
                ),
                domain=_REPORT_BINDING_DOMAIN,
            )
            examples.append(
                CaptureFeedbackDatasetExample(
                    example_id=example_id,
                    project_id=project_id,
                    profile_id=item.profile_id,
                    label=item.label,
                    prediction=item.prediction,
                    partition=item.partition,
                    report_binding=report_binding,
                )
            )
        examples.sort(key=lambda item: item.example_id)
        body: dict[str, object] = {
            "schema_version": "capture-feedback-dataset/v1",
            "evidence_source": evidence_source.value,
            "study_attestation": (
                None
                if study_attestation is None
                else study_attestation.model_dump(mode="json", warnings="error")
            ),
            "explicit_opt_in": True,
            "raw_content_included": False,
            "direct_identifiers_included": False,
            "identifier_scope": "export_specific_hmac",
            "export_id": export_id,
            "examples": [item.model_dump(mode="json", warnings="error") for item in examples],
        }
        digest = length_prefixed_sha256(canonical_json(body), domain=_DATASET_DIGEST_DOMAIN)
        tagged_body = {**body, "dataset_digest": digest}
        dataset_tag = installation_key._hmac_sha256(
            canonical_json(tagged_body),
            domain=_DATASET_TAG_DOMAIN,
        )
        dataset = CaptureFeedbackDataset.model_validate_json(
            canonical_json({**tagged_body, "dataset_tag": dataset_tag})
        )
        if not _model_is_exact(CaptureFeedbackDataset, dataset):
            raise ValueError
        return dataset
    except CaptureFeedbackError:
        raise
    except Exception:
        raise CaptureFeedbackError() from None


def _decode_dataset(data: bytes) -> CaptureFeedbackDataset:
    if type(data) is not bytes:
        raise ValueError
    parsed = read_bounded_json(data, limits=_FEEDBACK_JSON_LIMITS)
    if not hmac.compare_digest(canonical_json(parsed), data):
        raise ValueError
    result = CaptureFeedbackDataset.model_validate_json(data)
    if not _model_is_exact(CaptureFeedbackDataset, result):
        raise ValueError
    return result


def encode_capture_feedback_dataset(dataset: CaptureFeedbackDataset) -> bytes:
    result: bytes | None = None
    with suppress(Exception):
        if _model_is_exact(CaptureFeedbackDataset, dataset):
            candidate = canonical_json(dataset.model_dump(mode="json", warnings=False))
            if _decode_dataset(candidate) == dataset:
                result = candidate
    if result is None:
        raise CaptureFeedbackError()
    return result


def decode_capture_feedback_dataset(
    data: bytes,
    *,
    installation_key: InstallationKey,
) -> CaptureFeedbackDataset:
    result: CaptureFeedbackDataset | None = None
    with suppress(Exception):
        result = _verified_dataset(
            _decode_dataset(data),
            installation_key=installation_key,
        )
    if result is None:
        raise CaptureFeedbackError()
    return result


class CaptureCalibrationBootstrapInterval(_FeedbackModel):
    lower_ppm: _PartsPerMillion
    upper_ppm: _PartsPerMillion
    confidence_ppm: Literal[950_000] = 950_000
    bootstrap_replicates: Literal[2_000] = CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES
    percentile_rule: Literal["nearest-rank-50-1950-of-2000/v1"] = "nearest-rank-50-1950-of-2000/v1"
    boundary_status: CaptureCalibrationBoundaryStatus
    finite_sample_safety_bound: Literal[False] = False

    @model_validator(mode="after")
    def interval_is_ordered(self) -> Self:
        if self.lower_ppm > self.upper_ppm:
            raise ValueError("calibration interval is reversed")
        if (self.boundary_status is CaptureCalibrationBoundaryStatus.DEGENERATE_ZERO_WIDTH) != (
            self.lower_ppm == self.upper_ppm
        ):
            raise ValueError("calibration interval boundary status is inconsistent")
        return self


class CaptureCalibrationEstimate(_FeedbackModel):
    numerator: _BoundedCount
    denominator: Annotated[int, Field(ge=1, le=MAX_CAPTURE_FEEDBACK_EXAMPLES)]
    value_ppm: _PartsPerMillion
    interval: CaptureCalibrationBootstrapInterval | None
    interval_status: CaptureCalibrationIntervalStatus
    undefined_bootstrap_replicates: Annotated[
        int,
        Field(ge=0, le=CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES),
    ]

    @model_validator(mode="after")
    def estimate_is_exact(self) -> Self:
        if self.numerator > self.denominator or self.value_ppm != (
            self.numerator * _PPM // self.denominator
        ):
            raise ValueError("calibration estimate is inconsistent")
        if self.interval_status is CaptureCalibrationIntervalStatus.ESTIMATED:
            valid = self.interval is not None and self.undefined_bootstrap_replicates == 0
        elif self.interval_status is CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT:
            valid = self.interval is None and self.undefined_bootstrap_replicates == 0
        else:
            valid = self.interval is None and self.undefined_bootstrap_replicates > 0
        if not valid:
            raise ValueError("calibration estimate interval status is inconsistent")
        if (
            self.denominator < CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR
            and self.interval_status is not CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT
        ):
            raise ValueError("calibration estimate interval support is inconsistent")
        return self


class CaptureCalibrationConfusion(_FeedbackModel):
    true_positive: _BoundedCount
    false_negative: _BoundedCount
    abstained_memory_needed: _BoundedCount
    false_positive: _BoundedCount
    true_negative: _BoundedCount
    abstained_not_memory_needed: _BoundedCount
    uncertain_predicted_memory_needed: _BoundedCount
    uncertain_predicted_not_memory_needed: _BoundedCount
    uncertain_prediction_abstained: _BoundedCount


class CaptureCalibrationMetrics(_FeedbackModel):
    prevalence: CaptureCalibrationEstimate | None
    precision: CaptureCalibrationEstimate | None
    recall: CaptureCalibrationEstimate | None
    false_positive_rate: CaptureCalibrationEstimate | None
    reference_abstention_rate: CaptureCalibrationEstimate | None
    prediction_abstention_rate: CaptureCalibrationEstimate | None
    joint_abstention_rate: CaptureCalibrationEstimate | None


class CaptureCalibrationStratum(_FeedbackModel):
    scope: Literal["pooled", "provider"]
    profile_id: CaptureProfile | None
    project_count: _BoundedCount
    total_support: _BoundedCount
    memory_needed_support: _BoundedCount
    not_memory_needed_support: _BoundedCount
    uncertain_support: _BoundedCount
    confusion: CaptureCalibrationConfusion
    metrics: CaptureCalibrationMetrics

    @model_validator(mode="after")
    def stratum_counts_are_disjoint(self) -> Self:
        cells = self.confusion
        counts = {name: getattr(cells, name) for name in _CONFUSION_CELL_NAMES}
        positive = cells.true_positive + cells.false_negative + cells.abstained_memory_needed
        negative = cells.false_positive + cells.true_negative + cells.abstained_not_memory_needed
        uncertain = (
            cells.uncertain_predicted_memory_needed
            + cells.uncertain_predicted_not_memory_needed
            + cells.uncertain_prediction_abstained
        )
        metrics_are_exact = all(
            ((parts := _metric_parts(counts, name)) is None and getattr(self.metrics, name) is None)
            or (
                parts is not None
                and (estimate := getattr(self.metrics, name)) is not None
                and (estimate.numerator, estimate.denominator) == parts
            )
            for name in _METRIC_NAMES
        )
        if (
            (self.scope == "pooled") != (self.profile_id is None)
            or positive != self.memory_needed_support
            or negative != self.not_memory_needed_support
            or uncertain != self.uncertain_support
            or self.total_support != positive + negative + uncertain
            or self.project_count > self.total_support
            or not metrics_are_exact
        ):
            raise ValueError("calibration stratum counts are inconsistent")
        return self


class CaptureCalibrationReport(_FeedbackModel):
    schema_version: Literal["capture-calibration-report/v1"] = "capture-calibration-report/v1"
    dataset_digest: Annotated[Sha256Digest, Field(repr=False)]
    evidence_source: CaptureFeedbackEvidenceSource
    study_attestation_digest: Annotated[Sha256Digest | None, Field(repr=False)]
    blinding_status: CaptureFeedbackBlindingStatus | None
    evidence_status: CaptureCalibrationEvidenceStatus = (
        CaptureCalibrationEvidenceStatus.INSUFFICIENT_REAL_WORLD_EVIDENCE
    )
    insufficiency_reasons: Annotated[
        tuple[CaptureCalibrationInsufficiency, ...],
        Field(min_length=1, max_length=len(CaptureCalibrationInsufficiency)),
    ]
    dataset_support: _BoundedCount
    development_support: _BoundedCount
    calibration_support: _BoundedCount
    final_test_support: _BoundedCount
    provider_count: Annotated[int, Field(ge=0, le=len(CaptureProfile))]
    suppressed_provider_count: Annotated[int, Field(ge=0, le=len(CaptureProfile))]
    suppressed_provider_support: _BoundedCount
    pooled: CaptureCalibrationStratum
    providers: Annotated[
        tuple[CaptureCalibrationStratum, ...],
        Field(max_length=len(CaptureProfile)),
    ]
    bootstrap_seed_commitment: Literal[
        "0ba351d0bc979e0f7280c10f3f36c6d4da3b9df59b4b8b0c664d504ecd89ddc3"
    ] = CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT
    bootstrap_stratification: Literal["provider_project"] = "provider_project"
    classification_evaluation: Literal[True] = True
    calibrated: Literal[False] = False
    calibration_eligible: Literal[False] = False
    raw_content_used: Literal[False] = False
    project_identifiers_disclosed: Literal[False] = False
    confirmatory: Literal[False] = False
    decision_authority: Literal[False] = False
    dataset_authentication_verified: Literal[True] = True
    report_digest: Annotated[Sha256Digest, Field(repr=False)]
    report_tag: Annotated[Sha256Digest, Field(repr=False)]

    @model_validator(mode="after")
    def report_is_canonical_and_non_authoritative(self) -> Self:
        reasons = tuple(item.value for item in self.insufficiency_reasons)
        provider_profiles = tuple(item.profile_id for item in self.providers)
        provider_profile_values = tuple(
            "" if item is None else item.value for item in provider_profiles
        )
        provider_cell_totals = {
            name: sum(getattr(item.confusion, name) for item in self.providers)
            for name in _CONFUSION_CELL_NAMES
        }
        expected_reasons: set[CaptureCalibrationInsufficiency] = {
            CaptureCalibrationInsufficiency.EXTERNAL_REVIEW
        }
        bootstrap_enabled = self.final_test_support >= CAPTURE_E01_MIN_FINAL_TEST_SESSIONS
        interval_statuses_are_exact = all(
            estimate is None
            or (estimate.interval_status is CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT)
            is (
                not bootstrap_enabled
                or estimate.denominator < CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR
            )
            for stratum in (self.pooled, *self.providers)
            for estimate in (
                stratum.metrics.prevalence,
                stratum.metrics.precision,
                stratum.metrics.recall,
                stratum.metrics.false_positive_rate,
                stratum.metrics.reference_abstention_rate,
                stratum.metrics.prediction_abstention_rate,
                stratum.metrics.joint_abstention_rate,
            )
        )
        if self.evidence_source is not CaptureFeedbackEvidenceSource.DECLARED_E01:
            expected_reasons.add(CaptureCalibrationInsufficiency.NON_E01_EVIDENCE)
        if self.development_support + self.calibration_support == 0:
            expected_reasons.add(CaptureCalibrationInsufficiency.DEVELOPMENT_COHORT)
        if self.final_test_support < CAPTURE_E01_MIN_FINAL_TEST_SESSIONS:
            expected_reasons.add(CaptureCalibrationInsufficiency.FINAL_TEST_SUPPORT)
        if self.pooled.project_count < CAPTURE_E01_MIN_PROJECTS:
            expected_reasons.add(CaptureCalibrationInsufficiency.PROJECT_SUPPORT)
        if self.provider_count < CAPTURE_E01_MIN_PROVIDERS:
            expected_reasons.add(CaptureCalibrationInsufficiency.PROVIDER_SUPPORT)
        if self.pooled.memory_needed_support < CAPTURE_E01_MIN_MEMORY_NEEDED:
            expected_reasons.add(CaptureCalibrationInsufficiency.MEMORY_NEEDED_SUPPORT)
        if self.pooled.not_memory_needed_support < CAPTURE_E01_MIN_NOT_MEMORY_NEEDED:
            expected_reasons.add(CaptureCalibrationInsufficiency.NOT_MEMORY_NEEDED_SUPPORT)
        if self.suppressed_provider_count:
            expected_reasons.add(CaptureCalibrationInsufficiency.PROVIDER_DISCLOSURE)
        declared_e01 = self.evidence_source is CaptureFeedbackEvidenceSource.DECLARED_E01
        if (
            declared_e01
            and self.blinding_status is not CaptureFeedbackBlindingStatus.EXTERNALLY_ATTESTED
        ):
            expected_reasons.add(CaptureCalibrationInsufficiency.PROCESS_ATTESTATION)
        if any(
            estimate is None or estimate.interval is None
            for stratum in (self.pooled, *self.providers)
            for estimate in (
                stratum.metrics.prevalence,
                stratum.metrics.precision,
                stratum.metrics.recall,
                stratum.metrics.false_positive_rate,
                stratum.metrics.reference_abstention_rate,
                stratum.metrics.prediction_abstention_rate,
                stratum.metrics.joint_abstention_rate,
            )
        ):
            expected_reasons.add(CaptureCalibrationInsufficiency.METRIC_INTERVAL)
        if (
            reasons != tuple(sorted(set(reasons)))
            or self.insufficiency_reasons
            != tuple(sorted(expected_reasons, key=lambda item: item.value))
            or (self.study_attestation_digest is not None) is not declared_e01
            or (self.blinding_status is not None) is not declared_e01
            or self.dataset_support
            != self.development_support + self.calibration_support + self.final_test_support
            or self.pooled.scope != "pooled"
            or self.pooled.total_support != self.final_test_support
            or self.provider_count != len(self.providers) + self.suppressed_provider_count
            or self.suppressed_provider_support > self.final_test_support
            or (self.suppressed_provider_count == 0) is not (self.suppressed_provider_support == 0)
            or any(item is None for item in provider_profiles)
            or provider_profile_values != tuple(sorted(set(provider_profile_values)))
            or any(item.scope != "provider" for item in self.providers)
            or any(
                item.total_support < CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT
                for item in self.providers
            )
            or not interval_statuses_are_exact
            or sum(item.total_support for item in self.providers) + self.suppressed_provider_support
            != self.final_test_support
            or (
                self.suppressed_provider_support == 0
                and (
                    sum(item.memory_needed_support for item in self.providers)
                    != self.pooled.memory_needed_support
                    or sum(item.not_memory_needed_support for item in self.providers)
                    != self.pooled.not_memory_needed_support
                    or sum(item.uncertain_support for item in self.providers)
                    != self.pooled.uncertain_support
                    or any(
                        getattr(self.pooled.confusion, name) != provider_cell_totals[name]
                        for name in _CONFUSION_CELL_NAMES
                    )
                )
            )
            or not hmac.compare_digest(_calibration_report_body_digest(self), self.report_digest)
        ):
            raise ValueError("calibration report is inconsistent")
        return self


def _empty_cells() -> dict[str, int]:
    return dict.fromkeys(_CONFUSION_CELL_NAMES, 0)


def _cell_name(example: CaptureFeedbackDatasetExample) -> str:
    if example.label is CaptureFeedbackLabel.MEMORY_NEEDED:
        return {
            CaptureFeedbackPrediction.MEMORY_NEEDED: "true_positive",
            CaptureFeedbackPrediction.NOT_MEMORY_NEEDED: "false_negative",
            CaptureFeedbackPrediction.ABSTAIN: "abstained_memory_needed",
        }[example.prediction]
    if example.label is CaptureFeedbackLabel.NOT_MEMORY_NEEDED:
        return {
            CaptureFeedbackPrediction.MEMORY_NEEDED: "false_positive",
            CaptureFeedbackPrediction.NOT_MEMORY_NEEDED: "true_negative",
            CaptureFeedbackPrediction.ABSTAIN: "abstained_not_memory_needed",
        }[example.prediction]
    return {
        CaptureFeedbackPrediction.MEMORY_NEEDED: "uncertain_predicted_memory_needed",
        CaptureFeedbackPrediction.NOT_MEMORY_NEEDED: "uncertain_predicted_not_memory_needed",
        CaptureFeedbackPrediction.ABSTAIN: "uncertain_prediction_abstained",
    }[example.prediction]


def _counts(examples: tuple[CaptureFeedbackDatasetExample, ...]) -> dict[str, int]:
    result = _empty_cells()
    for example in examples:
        result[_cell_name(example)] += 1
    return result


def _metric_fractions(counts: Mapping[str, int]) -> dict[str, Fraction | None]:
    positive = (
        counts["true_positive"] + counts["false_negative"] + counts["abstained_memory_needed"]
    )
    negative = (
        counts["false_positive"] + counts["true_negative"] + counts["abstained_not_memory_needed"]
    )
    uncertain = (
        counts["uncertain_predicted_memory_needed"]
        + counts["uncertain_predicted_not_memory_needed"]
        + counts["uncertain_prediction_abstained"]
    )
    total = positive + negative + uncertain
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    predicted_abstain = (
        counts["abstained_memory_needed"]
        + counts["abstained_not_memory_needed"]
        + counts["uncertain_prediction_abstained"]
    )
    return {
        "prevalence": None if positive + negative == 0 else Fraction(positive, positive + negative),
        "precision": (
            None
            if predicted_positive == 0
            else Fraction(counts["true_positive"], predicted_positive)
        ),
        "recall": None if positive == 0 else Fraction(counts["true_positive"], positive),
        "false_positive_rate": (
            None if negative == 0 else Fraction(counts["false_positive"], negative)
        ),
        "reference_abstention_rate": None if total == 0 else Fraction(uncertain, total),
        "prediction_abstention_rate": (None if total == 0 else Fraction(predicted_abstain, total)),
        "joint_abstention_rate": (
            None
            if total == 0
            else Fraction(
                uncertain
                + counts["abstained_memory_needed"]
                + counts["abstained_not_memory_needed"],
                total,
            )
        ),
    }


def _metric_parts(counts: Mapping[str, int], name: str) -> tuple[int, int] | None:
    positive = (
        counts["true_positive"] + counts["false_negative"] + counts["abstained_memory_needed"]
    )
    negative = (
        counts["false_positive"] + counts["true_negative"] + counts["abstained_not_memory_needed"]
    )
    uncertain = (
        counts["uncertain_predicted_memory_needed"]
        + counts["uncertain_predicted_not_memory_needed"]
        + counts["uncertain_prediction_abstained"]
    )
    total = positive + negative + uncertain
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    predicted_abstain = (
        counts["abstained_memory_needed"]
        + counts["abstained_not_memory_needed"]
        + counts["uncertain_prediction_abstained"]
    )
    parts = {
        "prevalence": (positive, positive + negative),
        "precision": (counts["true_positive"], predicted_positive),
        "recall": (counts["true_positive"], positive),
        "false_positive_rate": (counts["false_positive"], negative),
        "reference_abstention_rate": (uncertain, total),
        "prediction_abstention_rate": (predicted_abstain, total),
        "joint_abstention_rate": (
            uncertain + counts["abstained_memory_needed"] + counts["abstained_not_memory_needed"],
            total,
        ),
    }[name]
    return None if parts[1] == 0 else parts


def _fraction_ppm(value: Fraction) -> int:
    return value.numerator * _PPM // value.denominator


def _bootstrap_draw_index(
    *,
    size: int,
    replicate: int,
    profile: str,
    project_id: str,
    draw: int,
) -> int:
    if (
        type(size) is not int
        or not 1 <= size <= MAX_CAPTURE_FEEDBACK_EXAMPLES
        or type(replicate) is not int
        or not 0 <= replicate < CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES
        or type(profile) is not str
        or type(project_id) is not str
        or type(draw) is not int
        or not 0 <= draw < size
    ):
        raise CaptureFeedbackError()
    space = 1 << 64
    ceiling = space - (space % size)
    for attempt in range(_BOOTSTRAP_REJECTION_ATTEMPTS):
        digest = length_prefixed_sha256(
            CAPTURE_CALIBRATION_BOOTSTRAP_SEED,
            str(replicate),
            profile,
            project_id,
            str(draw),
            str(attempt),
            domain="saliencegate:capture:feedback-bootstrap-index:v1",
        )
        candidate = int(digest[:16], 16)
        if candidate < ceiling:
            return candidate % size
    raise CaptureFeedbackError()


def _bootstrap_samples(
    examples: tuple[CaptureFeedbackDatasetExample, ...],
    replicate: int,
) -> tuple[CaptureFeedbackDatasetExample, ...]:
    by_cell: dict[tuple[str, str], list[CaptureFeedbackDatasetExample]] = defaultdict(list)
    for example in examples:
        by_cell[(example.profile_id.value, example.project_id)].append(example)

    groups: list[tuple[str, str, str, tuple[CaptureFeedbackDatasetExample, ...]]] = []
    for (profile, project_id), rows in by_cell.items():
        ordered = tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.label.value,
                    item.prediction.value,
                    item.partition.value,
                    item.example_id,
                ),
            )
        )
        signature = length_prefixed_sha256(
            canonical_json(
                {
                    "schema_version": "capture-feedback-bootstrap-cell/v1",
                    "counts": _counts(ordered),
                }
            ),
            domain="saliencegate:capture:feedback-bootstrap-cell:v1",
        )
        groups.append((profile, signature, project_id, ordered))
    groups.sort(key=lambda item: (item[0], item[1], item[2]))

    duplicate_ordinals: dict[tuple[str, str], int] = defaultdict(int)
    sampled: list[CaptureFeedbackDatasetExample] = []
    for profile, signature, _project_id, ordered in groups:
        duplicate_key = (profile, signature)
        duplicate_ordinal = duplicate_ordinals[duplicate_key]
        duplicate_ordinals[duplicate_key] += 1
        project_coordinate = f"{signature}:{duplicate_ordinal}"
        sampled.extend(
            ordered[
                _bootstrap_draw_index(
                    size=len(ordered),
                    replicate=replicate,
                    profile=profile,
                    project_id=project_coordinate,
                    draw=draw,
                )
            ]
            for draw in range(len(ordered))
        )
    return tuple(sampled)


def _interval(
    values: list[Fraction],
    *,
    undefined: int,
) -> CaptureCalibrationBootstrapInterval | None:
    if undefined or len(values) != CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES:
        return None
    ordered = sorted(values)
    lower_value = ordered[_BOOTSTRAP_LOWER_RANK - 1]
    upper_value = ordered[_BOOTSTRAP_UPPER_RANK - 1]
    lower = _fraction_ppm(lower_value)
    upper = (upper_value.numerator * _PPM + upper_value.denominator - 1) // upper_value.denominator
    return CaptureCalibrationBootstrapInterval(
        lower_ppm=lower,
        upper_ppm=upper,
        boundary_status=(
            CaptureCalibrationBoundaryStatus.DEGENERATE_ZERO_WIDTH
            if lower == upper
            else CaptureCalibrationBoundaryStatus.INTERIOR
        ),
    )


def _estimate(
    counts: Mapping[str, int],
    name: str,
    *,
    values: list[Fraction],
    undefined: int,
    bootstrap_enabled: bool,
) -> CaptureCalibrationEstimate | None:
    parts = _metric_parts(counts, name)
    if parts is None:
        return None
    numerator, denominator = parts
    interval_enabled = (
        bootstrap_enabled and denominator >= CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR
    )
    interval = _interval(values, undefined=undefined) if interval_enabled else None
    if not interval_enabled:
        status = CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT
        undefined_replicates = 0
    elif undefined:
        status = CaptureCalibrationIntervalStatus.UNDEFINED_BOOTSTRAP_REPLICATE
        undefined_replicates = undefined
    else:
        status = CaptureCalibrationIntervalStatus.ESTIMATED
        undefined_replicates = 0
    return CaptureCalibrationEstimate(
        numerator=numerator,
        denominator=denominator,
        value_ppm=numerator * _PPM // denominator,
        interval=interval,
        interval_status=status,
        undefined_bootstrap_replicates=undefined_replicates,
    )


def _build_stratum(
    examples: tuple[CaptureFeedbackDatasetExample, ...],
    *,
    profile_id: CaptureProfile | None,
    bootstrap_values: Mapping[str, list[Fraction]],
    bootstrap_undefined: Mapping[str, int],
    bootstrap_enabled: bool,
) -> CaptureCalibrationStratum:
    counts = _counts(examples)
    positive = (
        counts["true_positive"] + counts["false_negative"] + counts["abstained_memory_needed"]
    )
    negative = (
        counts["false_positive"] + counts["true_negative"] + counts["abstained_not_memory_needed"]
    )
    uncertain = len(examples) - positive - negative
    estimates = {
        name: _estimate(
            counts,
            name,
            values=bootstrap_values[name],
            undefined=bootstrap_undefined[name],
            bootstrap_enabled=bootstrap_enabled,
        )
        for name in _METRIC_NAMES
    }
    return CaptureCalibrationStratum(
        scope="pooled" if profile_id is None else "provider",
        profile_id=profile_id,
        project_count=len({item.project_id for item in examples}),
        total_support=len(examples),
        memory_needed_support=positive,
        not_memory_needed_support=negative,
        uncertain_support=uncertain,
        confusion=CaptureCalibrationConfusion(**counts),
        metrics=CaptureCalibrationMetrics(**estimates),
    )


def _calibration_report_body(value: CaptureCalibrationReport) -> dict[str, object]:
    return {
        key: item
        for key, item in value.model_dump(mode="json", warnings=False).items()
        if key not in {"report_digest", "report_tag"}
    }


def _calibration_report_body_digest(value: CaptureCalibrationReport) -> str:
    return length_prefixed_sha256(
        canonical_json(_calibration_report_body(value)),
        domain=_REPORT_DIGEST_DOMAIN,
    )


def _calibration_report_tag_material(
    value: CaptureCalibrationReport,
) -> dict[str, object]:
    return {
        key: item
        for key, item in value.model_dump(mode="json", warnings=False).items()
        if key != "report_tag"
    }


def _verified_calibration_report(
    value: object,
    *,
    installation_key: InstallationKey,
) -> CaptureCalibrationReport:
    if type(installation_key) is not InstallationKey or not _model_is_exact(
        CaptureCalibrationReport, value
    ):
        raise CaptureFeedbackError()
    assert isinstance(value, CaptureCalibrationReport)
    expected = installation_key._hmac_sha256(
        canonical_json(_calibration_report_tag_material(value)),
        domain=_REPORT_TAG_DOMAIN,
    )
    if not hmac.compare_digest(value.report_tag, expected):
        raise CaptureFeedbackError()
    return value


def _attestation_digest(value: CaptureFeedbackStudyAttestation | None) -> str | None:
    if value is None:
        return None
    return length_prefixed_sha256(
        canonical_json(value.model_dump(mode="json", warnings="error")),
        domain="saliencegate:capture:feedback-study-attestation:v1",
    )


def evaluate_capture_feedback_dataset(
    dataset: CaptureFeedbackDataset,
    *,
    installation_key: InstallationKey,
) -> CaptureCalibrationReport:
    """Evaluate a frozen final-test cohort without producing an activation decision."""

    try:
        checked = decode_capture_feedback_dataset(
            encode_capture_feedback_dataset(dataset),
            installation_key=installation_key,
        )
        development = tuple(
            item
            for item in checked.examples
            if item.partition is CaptureFeedbackPartition.DEVELOPMENT
        )
        calibration = tuple(
            item
            for item in checked.examples
            if item.partition is CaptureFeedbackPartition.CALIBRATION
        )
        final_test = tuple(
            item
            for item in checked.examples
            if item.partition is CaptureFeedbackPartition.FINAL_TEST
        )
        profiles = tuple(
            sorted({item.profile_id for item in final_test}, key=lambda item: item.value)
        )
        profile_rows = {
            profile: tuple(item for item in final_test if item.profile_id is profile)
            for profile in profiles
        }
        disclosed_profiles = tuple(
            profile
            for profile in profiles
            if len(profile_rows[profile]) >= CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT
        )
        suppressed_profiles = tuple(
            profile for profile in profiles if profile not in disclosed_profiles
        )

        scope_examples: dict[str, tuple[CaptureFeedbackDatasetExample, ...]] = {
            "pooled": final_test,
            **{profile.value: profile_rows[profile] for profile in disclosed_profiles},
        }
        values: dict[str, dict[str, list[Fraction]]] = {
            scope: {name: [] for name in _METRIC_NAMES} for scope in scope_examples
        }
        undefined: dict[str, dict[str, int]] = {
            scope: {name: 0 for name in _METRIC_NAMES} for scope in scope_examples
        }
        bootstrap_enabled = len(final_test) >= CAPTURE_E01_MIN_FINAL_TEST_SESSIONS
        if bootstrap_enabled:
            for replicate in range(CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES):
                sampled = _bootstrap_samples(final_test, replicate)
                sampled_scopes = {
                    "pooled": sampled,
                    **{
                        profile.value: tuple(item for item in sampled if item.profile_id is profile)
                        for profile in disclosed_profiles
                    },
                }
                for scope, examples in sampled_scopes.items():
                    metrics = _metric_fractions(_counts(examples))
                    for name, metric in metrics.items():
                        if metric is None:
                            undefined[scope][name] += 1
                        else:
                            values[scope][name].append(metric)

        pooled = _build_stratum(
            final_test,
            profile_id=None,
            bootstrap_values=values["pooled"],
            bootstrap_undefined=undefined["pooled"],
            bootstrap_enabled=bootstrap_enabled,
        )
        providers = tuple(
            _build_stratum(
                profile_rows[profile],
                profile_id=profile,
                bootstrap_values=values[profile.value],
                bootstrap_undefined=undefined[profile.value],
                bootstrap_enabled=bootstrap_enabled,
            )
            for profile in disclosed_profiles
        )

        reasons: set[CaptureCalibrationInsufficiency] = {
            CaptureCalibrationInsufficiency.EXTERNAL_REVIEW
        }
        if checked.evidence_source is not CaptureFeedbackEvidenceSource.DECLARED_E01:
            reasons.add(CaptureCalibrationInsufficiency.NON_E01_EVIDENCE)
        if not development and not calibration:
            reasons.add(CaptureCalibrationInsufficiency.DEVELOPMENT_COHORT)
        if len(final_test) < CAPTURE_E01_MIN_FINAL_TEST_SESSIONS:
            reasons.add(CaptureCalibrationInsufficiency.FINAL_TEST_SUPPORT)
        if pooled.project_count < CAPTURE_E01_MIN_PROJECTS:
            reasons.add(CaptureCalibrationInsufficiency.PROJECT_SUPPORT)
        if len(profiles) < CAPTURE_E01_MIN_PROVIDERS:
            reasons.add(CaptureCalibrationInsufficiency.PROVIDER_SUPPORT)
        if suppressed_profiles:
            reasons.add(CaptureCalibrationInsufficiency.PROVIDER_DISCLOSURE)
        if pooled.memory_needed_support < CAPTURE_E01_MIN_MEMORY_NEEDED:
            reasons.add(CaptureCalibrationInsufficiency.MEMORY_NEEDED_SUPPORT)
        if pooled.not_memory_needed_support < CAPTURE_E01_MIN_NOT_MEMORY_NEEDED:
            reasons.add(CaptureCalibrationInsufficiency.NOT_MEMORY_NEEDED_SUPPORT)
        if checked.evidence_source is CaptureFeedbackEvidenceSource.DECLARED_E01:
            attestation = checked.study_attestation
            if (
                attestation is None
                or attestation.blinding_status
                is not CaptureFeedbackBlindingStatus.EXTERNALLY_ATTESTED
            ):
                reasons.add(CaptureCalibrationInsufficiency.PROCESS_ATTESTATION)
        required_strata = (pooled, *providers)
        if any(
            estimate is None or estimate.interval is None
            for stratum in required_strata
            for estimate in (
                stratum.metrics.prevalence,
                stratum.metrics.precision,
                stratum.metrics.recall,
                stratum.metrics.false_positive_rate,
                stratum.metrics.reference_abstention_rate,
                stratum.metrics.prediction_abstention_rate,
                stratum.metrics.joint_abstention_rate,
            )
        ):
            reasons.add(CaptureCalibrationInsufficiency.METRIC_INTERVAL)

        body: dict[str, object] = {
            "schema_version": "capture-calibration-report/v1",
            "dataset_digest": checked.dataset_digest,
            "evidence_source": checked.evidence_source.value,
            "study_attestation_digest": _attestation_digest(checked.study_attestation),
            "blinding_status": (
                None
                if checked.study_attestation is None
                else checked.study_attestation.blinding_status.value
            ),
            "evidence_status": "insufficient_real_world_evidence",
            "insufficiency_reasons": sorted(item.value for item in reasons),
            "dataset_support": len(checked.examples),
            "development_support": len(development),
            "calibration_support": len(calibration),
            "final_test_support": len(final_test),
            "provider_count": len(profiles),
            "suppressed_provider_count": len(suppressed_profiles),
            "suppressed_provider_support": sum(
                len(profile_rows[profile]) for profile in suppressed_profiles
            ),
            "pooled": pooled.model_dump(mode="json", warnings="error"),
            "providers": [item.model_dump(mode="json", warnings="error") for item in providers],
            "bootstrap_seed_commitment": CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT,
            "bootstrap_stratification": "provider_project",
            "classification_evaluation": True,
            "calibrated": False,
            "calibration_eligible": False,
            "raw_content_used": False,
            "project_identifiers_disclosed": False,
            "confirmatory": False,
            "decision_authority": False,
            "dataset_authentication_verified": True,
        }
        digest = length_prefixed_sha256(canonical_json(body), domain=_REPORT_DIGEST_DOMAIN)
        tagged_body = {**body, "report_digest": digest}
        report_tag = installation_key._hmac_sha256(
            canonical_json(tagged_body),
            domain=_REPORT_TAG_DOMAIN,
        )
        result = CaptureCalibrationReport.model_validate_json(
            canonical_json({**tagged_body, "report_tag": report_tag})
        )
        if not _model_is_exact(CaptureCalibrationReport, result):
            raise ValueError
        return result
    except CaptureFeedbackError:
        raise
    except Exception:
        raise CaptureFeedbackError() from None


def _decode_calibration_report(data: bytes) -> CaptureCalibrationReport:
    if type(data) is not bytes:
        raise ValueError
    parsed = read_bounded_json(data, limits=_FEEDBACK_JSON_LIMITS)
    if not hmac.compare_digest(canonical_json(parsed), data):
        raise ValueError
    result = CaptureCalibrationReport.model_validate_json(data)
    if not _model_is_exact(CaptureCalibrationReport, result):
        raise ValueError
    return result


def encode_capture_calibration_report(report: CaptureCalibrationReport) -> bytes:
    result: bytes | None = None
    with suppress(Exception):
        if _model_is_exact(CaptureCalibrationReport, report):
            candidate = canonical_json(report.model_dump(mode="json", warnings=False))
            if _decode_calibration_report(candidate) == report:
                result = candidate
    if result is None:
        raise CaptureFeedbackError()
    return result


def decode_capture_calibration_report(
    data: bytes,
    *,
    installation_key: InstallationKey,
) -> CaptureCalibrationReport:
    result: CaptureCalibrationReport | None = None
    with suppress(Exception):
        result = _verified_calibration_report(
            _decode_calibration_report(data),
            installation_key=installation_key,
        )
    if result is None:
        raise CaptureFeedbackError()
    return result


__all__ = [
    "CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES",
    "CAPTURE_CALIBRATION_BOOTSTRAP_SEED",
    "CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT",
    "CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR",
    "CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT",
    "CAPTURE_E01_MIN_FINAL_TEST_SESSIONS",
    "CAPTURE_E01_MIN_MEMORY_NEEDED",
    "CAPTURE_E01_MIN_NOT_MEMORY_NEEDED",
    "CAPTURE_E01_MIN_PROJECTS",
    "CAPTURE_E01_MIN_PROVIDERS",
    "CAPTURE_FEEDBACK_EVALUATOR_VERSION",
    "MAX_CAPTURE_FEEDBACK_EXAMPLES",
    "MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION",
    "CaptureCalibrationBootstrapInterval",
    "CaptureCalibrationBoundaryStatus",
    "CaptureCalibrationConfusion",
    "CaptureCalibrationEstimate",
    "CaptureCalibrationEvidenceStatus",
    "CaptureCalibrationInsufficiency",
    "CaptureCalibrationIntervalStatus",
    "CaptureCalibrationMetrics",
    "CaptureCalibrationReport",
    "CaptureCalibrationStratum",
    "CaptureFeedbackBlindingStatus",
    "CaptureFeedbackDataset",
    "CaptureFeedbackDatasetExample",
    "CaptureFeedbackError",
    "CaptureFeedbackEvidenceSource",
    "CaptureFeedbackExportRecord",
    "CaptureFeedbackLabel",
    "CaptureFeedbackPartition",
    "CaptureFeedbackPrediction",
    "CaptureFeedbackReceipt",
    "CaptureFeedbackRecord",
    "CaptureFeedbackRecordOrigin",
    "CaptureFeedbackRevision",
    "CaptureFeedbackStudyAttestation",
    "CaptureFeedbackWriteDisposition",
    "build_capture_feedback_dataset",
    "build_capture_feedback_export_record",
    "build_synthetic_capture_feedback_export_record",
    "decode_capture_calibration_report",
    "decode_capture_feedback_dataset",
    "encode_capture_calibration_report",
    "encode_capture_feedback_dataset",
    "evaluate_capture_feedback_dataset",
]
