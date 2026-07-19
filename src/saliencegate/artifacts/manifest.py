from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    ConstraintStatus,
    CycleState,
    DeliveryOutcome,
    DeliveryState,
    DeliveryTarget,
    InterventionAction,
    InterventionOutcome,
    InvocationDecision,
    JsonObject,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    NonNegativeInt,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.intervention import ProposalParseStatus
from saliencegate.ports.adapters import DeduplicationGuarantee
from saliencegate.ports.repository import LedgerHead, ProjectionDigests
from saliencegate.runtime.engine import ReplayEngineConfig, ReplayRoutingBinding

ARTIFACT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
MAX_ARTIFACT_COMPONENT_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_RECORDS = 100_000
MAX_SIGNED_64 = (1 << 63) - 1

_SCHEMA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_PACKAGE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}$")
_GIT_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_COMPONENT_DIGEST_DOMAIN = "saliencegate:artifact:component:v1"
_CONTENT_SET_DIGEST_DOMAIN = "saliencegate:artifact:content-set:v1"
_MANIFEST_DIGEST_DOMAIN = "saliencegate:artifact:manifest:v1"
_DELIVERY_BINDING_DIGEST_DOMAIN = "saliencegate:artifact:delivery-binding:v1"


class ArtifactClassification(StrEnum):
    USER_REDACTED = "user_redacted"
    SYNTHETIC_DIGEST_ONLY = "synthetic_digest_only"
    SYNTHETIC_RAW = "synthetic_raw"


class ArtifactEvidenceLevel(StrEnum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class RevisionSource(StrEnum):
    GIT = "git"
    DISTRIBUTION = "distribution"
    UNATTESTED = "unattested"


class ArtifactComponentName(StrEnum):
    ATTESTATIONS = "attestations"
    BUDGETS = "budgets"
    DECISIONS = "decisions"
    DELIVERIES = "deliveries"
    OUTCOMES = "outcomes"
    RUN = "run"
    SYNTHETIC = "synthetic"


_COMPONENT_PATHS: dict[ArtifactComponentName, str] = {
    name: f"{name.value}.json" for name in ArtifactComponentName
}
_REQUIRED_COMPONENTS = frozenset(ArtifactComponentName) - {ArtifactComponentName.SYNTHETIC}


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class RevisionEvidence(_ArtifactModel):
    schema_version: Literal["revision-evidence/v1"] = "revision-evidence/v1"
    source: RevisionSource
    package_version: Annotated[str, Field(min_length=1, max_length=128)]
    commit: str | None = None
    dirty_worktree: bool | None = None
    distribution_digest: Sha256Digest | None = None

    @field_validator("package_version")
    @classmethod
    def bounded_package_version(cls, value: str) -> str:
        if _PACKAGE_VERSION.fullmatch(value) is None:
            raise ValueError("package version is not a stable identifier")
        return value

    @field_validator("commit")
    @classmethod
    def valid_git_revision(cls, value: str | None) -> str | None:
        if value is not None and _GIT_REVISION.fullmatch(value) is None:
            raise ValueError("Git revision must be a lowercase full commit digest")
        return value

    @model_validator(mode="after")
    def source_has_exact_evidence(self) -> Self:
        if self.source is RevisionSource.GIT:
            if (
                self.commit is None
                or self.dirty_worktree is None
                or self.distribution_digest is not None
            ):
                raise ValueError("Git revision evidence is incomplete")
        elif self.source is RevisionSource.DISTRIBUTION:
            if (
                self.commit is not None
                or self.dirty_worktree is not None
                or self.distribution_digest is None
            ):
                raise ValueError("distribution revision evidence is incomplete")
        elif (
            self.commit is not None
            or self.dirty_worktree is not None
            or self.distribution_digest is not None
        ):
            raise ValueError("unattested revision cannot carry provenance claims")
        return self

    @property
    def confirmatory_eligible(self) -> bool:
        return (
            self.source is RevisionSource.GIT
            and self.commit is not None
            and self.dirty_worktree is False
        ) or (self.source is RevisionSource.DISTRIBUTION and self.distribution_digest is not None)


class ArtifactComponent(_ArtifactModel):
    schema_version: Literal["artifact-component/v1"] = "artifact-component/v1"
    name: ArtifactComponentName
    path: Annotated[str, Field(min_length=1, max_length=64)]
    byte_count: Annotated[int, Field(ge=2, le=MAX_ARTIFACT_COMPONENT_BYTES)]
    record_count: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    content_digest: Sha256Digest

    @model_validator(mode="after")
    def path_is_fixed_for_component(self) -> Self:
        if self.path != _COMPONENT_PATHS[self.name]:
            raise ValueError("artifact component path is not the fixed v1 path")
        return self


class ArtifactCounters(_ArtifactModel):
    schema_version: Literal["artifact-counters/v1"] = "artifact-counters/v1"
    events: Annotated[int, Field(ge=1, le=MAX_ARTIFACT_RECORDS)]
    decisions: Annotated[int, Field(ge=1, le=MAX_ARTIFACT_RECORDS)]
    invoked: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    cycles: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    model_calls: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    deliveries: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    delivered: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]
    outcomes: Annotated[int, Field(ge=0, le=MAX_ARTIFACT_RECORDS)]

    @model_validator(mode="after")
    def totals_are_possible(self) -> Self:
        if self.decisions != self.events:
            raise ValueError("artifact decision count must equal event count")
        if (
            self.invoked != self.cycles
            or self.cycles > self.events
            or self.model_calls > self.cycles
            or self.deliveries > self.cycles
            or self.delivered > self.deliveries
            or self.outcomes > self.cycles
        ):
            raise ValueError("artifact counters are internally inconsistent")
        return self


class PolicyConfigurationAttestation(_ArtifactModel):
    policy_version: ComponentIdentifier
    configuration_digest: Sha256Digest


class GroundingConfigurationAttestation(_ArtifactModel):
    grounding_version: ComponentIdentifier
    configuration_digest: Sha256Digest


class ArtifactRunComponent(_ArtifactModel):
    schema_version: Literal["artifact-run/v1"] = "artifact-run/v1"
    run_id: UUID4
    trace_digest: Sha256Digest
    trace_attestation_mode: Literal["adapter_manifest", "engine_normalized"]
    trace_event_count: Annotated[int, Field(ge=1, le=MAX_ARTIFACT_RECORDS)]
    model_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    engine_configuration: ReplayEngineConfig
    engine_configuration_digest: Sha256Digest
    policy_configurations: tuple[PolicyConfigurationAttestation, ...]
    grounding_configurations: tuple[GroundingConfigurationAttestation, ...]
    model_execution_mode: Literal["structured_model", "frozen_replay"]
    replay_id: ComponentIdentifier | None = None
    fixture_digest: Sha256Digest | None = None
    fixture_response_count: NonNegativeInt | None = None
    fixture_consumed_count: NonNegativeInt | None = None
    routing_digest: Sha256Digest
    projection_digests: ProjectionDigests
    ledger_entry_count: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    ledger_head: LedgerHead
    rebuild_equivalent: bool
    source_result_digest: Sha256Digest

    @model_validator(mode="after")
    def configuration_and_execution_are_attested(self) -> Self:
        if self.engine_configuration_digest != canonical_digest(
            self.engine_configuration.model_dump(mode="json", warnings=False)
        ):
            raise ValueError("artifact run configuration digest does not match")
        policy_keys = tuple(
            (binding.policy_version, binding.configuration_digest)
            for binding in self.policy_configurations
        )
        grounding_keys = tuple(
            (binding.grounding_version, binding.configuration_digest)
            for binding in self.grounding_configurations
        )
        if (
            policy_keys != tuple(sorted(set(policy_keys)))
            or grounding_keys != tuple(sorted(set(grounding_keys)))
            or self.ledger_head.run_id != self.run_id
            or self.ledger_head.entry_count != self.ledger_entry_count
        ):
            raise ValueError("artifact run attestations are non-canonical")
        fixture_values = (
            self.replay_id,
            self.fixture_digest,
            self.fixture_response_count,
            self.fixture_consumed_count,
        )
        if self.model_execution_mode == "structured_model":
            if any(value is not None for value in fixture_values):
                raise ValueError("structured model artifact cannot claim replay fixture evidence")
        elif (
            any(value is None for value in fixture_values)
            or self.fixture_response_count != self.fixture_consumed_count
        ):
            raise ValueError("frozen replay artifact requires a consumed fixture")
        return self


class ArtifactDecisionsComponent(_ArtifactModel):
    schema_version: Literal["artifact-decisions/v1"] = "artifact-decisions/v1"
    run_id: UUID4
    decisions: Annotated[
        tuple[InvocationDecision, ...],
        Field(min_length=1, max_length=MAX_ARTIFACT_RECORDS),
    ]
    decisions_digest: Sha256Digest

    @model_validator(mode="after")
    def decisions_are_one_ordered_run(self) -> Self:
        if (
            tuple(decision.event_sequence for decision in self.decisions)
            != tuple(range(1, len(self.decisions) + 1))
            or any(decision.run_id != self.run_id for decision in self.decisions)
            or len({decision.decision_id for decision in self.decisions}) != len(self.decisions)
            or self.decisions_digest
            != canonical_digest(
                tuple(decision.model_dump(mode="json") for decision in self.decisions)
            )
        ):
            raise ValueError("artifact decisions are not a canonical ordered run")
        return self


class InterventionAttestation(_ArtifactModel):
    intervention_id: UUID4
    intervention_digest: Sha256Digest
    action: InterventionAction
    delivery_target: DeliveryTarget | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    reason_code: ReasonCode
    ttl_steps: Annotated[int, Field(ge=0, le=1)]
    grounding_version: ComponentIdentifier
    grounding_configuration_digest: Sha256Digest
    grounding_receipt_digest: Sha256Digest
    receipt_parse_status: ProposalParseStatus
    receipt_proposal_action: InterventionAction | None = None
    receipt_requested_delivery_target: DeliveryTarget | None = None
    receipt_model_call_index: Annotated[int, Field(ge=0, le=MAX_SIGNED_64)]
    receipt_model_call_digest: Sha256Digest
    claim_fingerprints: Annotated[tuple[Sha256Digest, ...], Field(max_length=2)] = ()
    claim_evidence_counts: Annotated[
        tuple[Annotated[int, Field(ge=0, le=2)], ...],
        Field(max_length=2),
    ] = ()
    claim_set_digest: Sha256Digest
    cited_memory_ids: tuple[UUID4, ...] = ()
    cited_event_ids: tuple[UUID4, ...] = ()
    rendered_text_digest: Sha256Digest | None = None
    created_at: UtcDatetime

    @model_validator(mode="after")
    def action_has_digest_only_grounding_evidence(self) -> Self:
        if self.claim_set_digest != canonical_digest(self.claim_fingerprints):
            raise ValueError("intervention claim-set digest does not match")
        if len(set(self.claim_fingerprints)) != len(self.claim_fingerprints):
            raise ValueError("intervention claim fingerprints must be unique")
        if self.action is InterventionAction.SILENCE:
            if (
                self.delivery_target is not None
                or self.rendered_text_digest is not None
                or self.claim_fingerprints
                or self.claim_evidence_counts
                or self.cited_memory_ids
                or self.cited_event_ids
                or self.ttl_steps != 0
            ):
                raise ValueError("silent intervention attestation carries reminder evidence")
            return self
        if (
            self.delivery_target is None
            or self.rendered_text_digest is None
            or self.reason_code is not ReasonCode.GROUNDED_REMINDER
            or self.ttl_steps != 1
            or not 1 <= len(self.claim_fingerprints) <= 2
            or self.claim_evidence_counts != (1,) * len(self.claim_fingerprints)
            or self.receipt_parse_status is not ProposalParseStatus.VALID
            or self.receipt_proposal_action is not InterventionAction.REMIND
            or self.receipt_requested_delivery_target is not self.delivery_target
            or not (self.cited_memory_ids or self.cited_event_ids)
        ):
            raise ValueError("reminder intervention lacks producer grounding attestations")
        return self


class CycleAttestation(_ArtifactModel):
    cycle_id: Sha256Digest
    revision: Annotated[int, Field(ge=1, le=4)]
    invocation_decision_id: UUID4
    policy_version: ComponentIdentifier
    configuration_digest: Sha256Digest
    grounding_version: ComponentIdentifier
    grounding_configuration_digest: Sha256Digest
    requested_delivery_target: DeliveryTarget | None = None
    first_event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    last_event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    state: CycleState
    budget_reservation: BudgetAmounts | None = None
    budget_settlement: BudgetAmounts | None = None
    batch_digest: Sha256Digest | None = None
    model_request_digest: Sha256Digest | None = None
    model_call_digests: tuple[Sha256Digest, ...] = ()
    model_call_latencies_us: tuple[NonNegativeInt, ...] = ()
    memory_creates: NonNegativeInt = 0
    memory_updates: NonNegativeInt = 0
    memory_invalidations: NonNegativeInt = 0
    private_status_replaced: bool = False
    intervention: InterventionAttestation | None = None
    failure_reason: ReasonCode | None = None

    @model_validator(mode="after")
    def cycle_is_terminal_and_bounded(self) -> Self:
        if self.last_event_sequence < self.first_event_sequence:
            raise ValueError("artifact cycle range is reversed")
        if self.state not in (CycleState.COMMITTED, CycleState.FAILED):
            raise ValueError("artifact cycle is not terminal")
        if (self.batch_digest is not None) is not (self.model_request_digest is not None):
            raise ValueError("artifact cycle model request attestation is inconsistent")
        if len(self.model_call_digests) != len(self.model_call_latencies_us):
            raise ValueError("artifact cycle model receipts are inconsistent")
        if self.state is CycleState.COMMITTED:
            if (
                self.revision != 4
                or self.budget_reservation is None
                or self.budget_settlement is None
                or self.failure_reason is not None
            ):
                raise ValueError("committed artifact cycle is incomplete")
        elif (
            not 2 <= self.revision <= 4
            or self.intervention is not None
            or self.failure_reason is None
        ):
            raise ValueError("failed artifact cycle is inconsistent")
        if self.budget_settlement is not None and (
            self.budget_settlement.model_calls != len(self.model_call_digests)
            and self.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST
        ):
            raise ValueError("artifact cycle settled calls do not match receipts")
        return self


def _sum_budget_amounts(values: tuple[BudgetAmounts, ...]) -> BudgetAmounts:
    fields = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    )
    return BudgetAmounts(
        **{field_name: sum(getattr(value, field_name) for value in values) for field_name in fields}
    )


class ArtifactBudgetsComponent(_ArtifactModel):
    schema_version: Literal["artifact-budgets/v1"] = "artifact-budgets/v1"
    run_id: UUID4
    limits: BudgetLimits
    configured_reservation: BudgetAmounts
    cycles: Annotated[tuple[CycleAttestation, ...], Field(max_length=MAX_ARTIFACT_RECORDS)]
    consumed: BudgetAmounts
    budget_projection_digest: Sha256Digest

    @model_validator(mode="after")
    def consumed_budget_matches_terminal_cycles(self) -> Self:
        cycle_ids = tuple(cycle.cycle_id for cycle in self.cycles)
        if len(set(cycle_ids)) != len(cycle_ids):
            raise ValueError("artifact cycle identities are not unique")
        settlements = tuple(
            cycle.budget_settlement for cycle in self.cycles if cycle.budget_settlement is not None
        )
        if self.consumed != _sum_budget_amounts(settlements):
            raise ValueError("artifact consumed budget does not match cycle settlements")
        return self


def delivery_binding_digest(values: Mapping[str, object]) -> str:
    selected = {
        key: values[key]
        for key in (
            "delivery_id",
            "cycle_id",
            "intervention_id",
            "rendered_text_digest",
            "target_request_id_digest",
            "target",
            "adapter_id_digest",
            "adapter_capabilities_digest",
        )
    }
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(selected)),
        domain=_DELIVERY_BINDING_DIGEST_DOMAIN,
    )


class DeliveryAttestation(_ArtifactModel):
    delivery_id: UUID4
    event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    cycle_id: Sha256Digest
    intervention_id: UUID4
    rendered_text_digest: Sha256Digest
    target_request_id_digest: Sha256Digest
    target: DeliveryTarget
    state: DeliveryState
    attempt_count: NonNegativeInt
    adapter_id_digest: Sha256Digest
    adapter_deduplicates: bool
    adapter_deduplication_guarantee: DeduplicationGuarantee
    adapter_supports_pre_action: bool
    adapter_contract_version: ComponentIdentifier
    adapter_capabilities_digest: Sha256Digest
    claim_id: UUID4 | None = None
    attempt_id: UUID4 | None = None
    receipt_digest: Sha256Digest | None = None
    outcome: DeliveryOutcome | None = None
    reason_code: ReasonCode | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    source_delivery_digest: Sha256Digest
    binding_digest: Sha256Digest

    @model_validator(mode="after")
    def binding_and_terminal_state_are_attested(self) -> Self:
        expected_binding = delivery_binding_digest(
            self.model_dump(mode="python", exclude={"binding_digest"}, warnings=False)
        )
        if self.binding_digest != expected_binding:
            raise ValueError("artifact delivery binding digest does not match")
        if self.state in (DeliveryState.PENDING, DeliveryState.CLAIMED, DeliveryState.ATTEMPTING):
            raise ValueError("artifact delivery is not terminal")
        if self.updated_at < self.created_at:
            raise ValueError("artifact delivery update precedes creation")
        durable = self.adapter_deduplication_guarantee is DeduplicationGuarantee.DURABLE_DELIVERY_ID
        if (
            self.adapter_deduplicates is not durable
            or (not self.adapter_deduplicates and self.attempt_count > 1)
            or (
                self.target is DeliveryTarget.PRE_ACTION_REPLAN
                and self.attempt_count > 0
                and not self.adapter_supports_pre_action
            )
        ):
            raise ValueError("artifact delivery capability attestation is inconsistent")
        owns_attempt = (
            self.attempt_count >= 1 and self.claim_id is not None and self.attempt_id is not None
        )
        if self.state is DeliveryState.DELIVERED:
            valid_terminal = (
                owns_attempt
                and self.outcome is DeliveryOutcome.DELIVERED
                and self.reason_code is ReasonCode.DELIVERY_SUCCEEDED
                and self.receipt_digest is not None
            )
        elif self.state is DeliveryState.UNKNOWN:
            valid_terminal = (
                owns_attempt
                and self.outcome is DeliveryOutcome.UNKNOWN
                and self.reason_code is ReasonCode.DELIVERY_UNKNOWN
                and self.receipt_digest is None
            )
        elif self.state is DeliveryState.FAILED:
            valid_terminal = (
                owns_attempt
                and self.outcome is DeliveryOutcome.FAILED
                and self.reason_code is ReasonCode.DELIVERY_FAILED
                and self.receipt_digest is None
            )
        else:
            allowed_rejections = {
                ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
                ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL,
                ReasonCode.UNSAFE_ROLE_MAPPING,
                ReasonCode.TARGET_UNAVAILABLE,
            }
            valid_terminal = (
                self.state is DeliveryState.REJECTED
                and self.attempt_id is None
                and self.receipt_digest is None
                and self.outcome is DeliveryOutcome.REFUSED
                and self.reason_code in allowed_rejections
                and (self.claim_id is not None or self.attempt_count == 0)
                and (self.attempt_count == 0 or self.adapter_deduplicates)
            )
        if not valid_terminal:
            raise ValueError("artifact delivery terminal state is inconsistent")
        return self


class ArtifactDeliveriesComponent(_ArtifactModel):
    schema_version: Literal["artifact-deliveries/v1"] = "artifact-deliveries/v1"
    run_id: UUID4
    deliveries: Annotated[
        tuple[DeliveryAttestation, ...],
        Field(max_length=MAX_ARTIFACT_RECORDS),
    ]

    @model_validator(mode="after")
    def deliveries_are_unique_and_ordered(self) -> Self:
        ids = tuple(delivery.delivery_id for delivery in self.deliveries)
        sequences = tuple(delivery.event_sequence for delivery in self.deliveries)
        if (
            len(set(ids)) != len(ids)
            or len(set(sequences)) != len(sequences)
            or sequences != tuple(sorted(sequences))
        ):
            raise ValueError("artifact deliveries are not unique and ordered")
        return self


class ArtifactOutcomesComponent(_ArtifactModel):
    schema_version: Literal["artifact-outcomes/v1"] = "artifact-outcomes/v1"
    run_id: UUID4
    outcomes: Annotated[
        tuple[InterventionOutcome, ...],
        Field(max_length=MAX_ARTIFACT_RECORDS),
    ]

    @model_validator(mode="after")
    def outcomes_are_one_unique_run(self) -> Self:
        if (
            any(outcome.run_id != self.run_id for outcome in self.outcomes)
            or len({outcome.outcome_id for outcome in self.outcomes}) != len(self.outcomes)
            or any(
                outcome.evidence_mode is not OutcomeEvidenceMode.POLICY_REPLAY
                or outcome.next_action_fingerprint is not None
                or outcome.repeated_error_status is not RepeatedErrorStatus.UNKNOWN
                or outcome.constraint_status is not ConstraintStatus.UNKNOWN
                or outcome.utility is not None
                or outcome.action_changed is not None
                or outcome.task_reward is not None
                or outcome.task_passed is not None
                for outcome in self.outcomes
            )
        ):
            raise ValueError("artifact outcomes do not form one unique run")
        return self


class ArtifactAttestationsComponent(_ArtifactModel):
    schema_version: Literal["artifact-attestations/v1"] = "artifact-attestations/v1"
    run_id: UUID4
    normalized_trace_digest: Sha256Digest
    normalized_draft_digests: tuple[Sha256Digest, ...]
    persisted_event_draft_digests: tuple[Sha256Digest, ...]
    routing_bindings: tuple[ReplayRoutingBinding, ...]
    routing_digest: Sha256Digest
    trace_record_digests: tuple[Sha256Digest, ...]
    trace_expected_event_ids: tuple[UUID4, ...]
    events_digest: Sha256Digest
    model_request_digests: tuple[Sha256Digest, ...]
    decisions_digest: Sha256Digest
    projection_digests: ProjectionDigests
    ledger_head: LedgerHead
    source_result_digest: Sha256Digest

    @model_validator(mode="after")
    def trace_attestations_have_one_cardinality(self) -> Self:
        event_count = len(self.normalized_draft_digests)
        if (
            event_count < 1
            or len(self.persisted_event_draft_digests) != event_count
            or len(self.routing_bindings) != event_count
            or tuple(binding.ordinal for binding in self.routing_bindings)
            != tuple(range(1, event_count + 1))
            or len(self.trace_expected_event_ids) != event_count
            or len(set(self.model_request_digests)) != len(self.model_request_digests)
            or self.routing_digest
            != canonical_digest(
                tuple(binding.model_dump(mode="json") for binding in self.routing_bindings)
            )
            or self.normalized_trace_digest
            != canonical_digest(
                {
                    "schema_version": "engine-normalized-trace/v1",
                    "draft_digests": self.normalized_draft_digests,
                }
            )
        ):
            raise ValueError("artifact trace attestations have inconsistent cardinality")
        return self


class ArtifactSyntheticComponent(_ArtifactModel):
    schema_version: Literal["artifact-synthetic/v1"] = "artifact-synthetic/v1"
    run_id: UUID4
    trace_digest: Sha256Digest
    prompt_template_digest: Sha256Digest
    model_request_digests: tuple[Sha256Digest, ...]
    model_call_digests: tuple[Sha256Digest, ...]
    prompt: JsonObject = Field(repr=False)
    responses: tuple[JsonObject, ...] = Field(repr=False)


def component_content_digest(data: bytes) -> str:
    if type(data) is not bytes:
        raise TypeError("artifact component digest requires exact bytes")
    return length_prefixed_sha256(data, domain=_COMPONENT_DIGEST_DOMAIN)


def overall_content_digest(components: tuple[ArtifactComponent, ...]) -> str:
    ordered = tuple(sorted(components, key=lambda component: component.name.value))
    descriptor_set = tuple(
        {
            "name": component.name.value,
            "path": component.path,
            "byte_count": component.byte_count,
            "record_count": component.record_count,
            "content_digest": component.content_digest,
        }
        for component in ordered
    )
    return length_prefixed_sha256(
        canonical_json(descriptor_set),
        domain=_CONTENT_SET_DIGEST_DOMAIN,
    )


def artifact_manifest_digest(values: Mapping[str, object]) -> str:
    payload = {key: value for key, value in values.items() if key != "manifest_digest"}
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(payload)),
        domain=_MANIFEST_DIGEST_DOMAIN,
    )


class ArtifactManifest(_ArtifactModel):
    record_type: Literal["artifact_manifest"] = "artifact_manifest"
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    artifact_kind: Literal["replay_run"] = "replay_run"
    classification: ArtifactClassification
    evidence_level: ArtifactEvidenceLevel
    run_id: UUID4
    revision: RevisionEvidence
    confirmatory_eligible: bool
    engine_configuration_digest: Sha256Digest
    trace_digest: Sha256Digest
    model_id: ComponentIdentifier
    replay_id: ComponentIdentifier | None = None
    prompt_template_digest: Sha256Digest
    result_digest: Sha256Digest
    components: Annotated[
        tuple[ArtifactComponent, ...],
        Field(min_length=len(_REQUIRED_COMPONENTS), max_length=len(ArtifactComponentName)),
    ]
    counters: ArtifactCounters
    overall_content_digest: Sha256Digest
    manifest_digest: Sha256Digest

    @property
    def confirmatory(self) -> bool:
        return self.evidence_level is ArtifactEvidenceLevel.CONFIRMATORY

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        match = _SCHEMA_VERSION.fullmatch(value)
        if match is None:
            raise ValueError("artifact schema version is malformed")
        if match.group(1) != "1":
            raise ValueError("unsupported artifact schema major")
        if value != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema minor")
        return value

    @model_validator(mode="after")
    def manifest_is_self_attesting(self) -> Self:
        names = tuple(component.name for component in self.components)
        paths = tuple(component.path for component in self.components)
        required_names = set(_REQUIRED_COMPONENTS)
        expected_names = (
            required_names | {ArtifactComponentName.SYNTHETIC}
            if self.classification is ArtifactClassification.SYNTHETIC_RAW
            else required_names
        )
        if (
            len(set(names)) != len(names)
            or len(set(paths)) != len(paths)
            or set(names) != expected_names
            or names != tuple(sorted(names, key=lambda item: item.value))
        ):
            raise ValueError("artifact component set is incomplete or non-canonical")
        if self.confirmatory_eligible is not self.revision.confirmatory_eligible:
            raise ValueError("confirmatory eligibility does not match revision evidence")
        if self.confirmatory and not self.confirmatory_eligible:
            raise ValueError("confirmatory artifact requires eligible revision evidence")
        record_counts = {component.name: component.record_count for component in self.components}
        expected_record_counts = {
            ArtifactComponentName.RUN: 1,
            ArtifactComponentName.DECISIONS: self.counters.decisions,
            ArtifactComponentName.BUDGETS: self.counters.cycles,
            ArtifactComponentName.DELIVERIES: self.counters.deliveries,
            ArtifactComponentName.OUTCOMES: self.counters.outcomes,
            ArtifactComponentName.ATTESTATIONS: 1,
        }
        if any(record_counts[name] != count for name, count in expected_record_counts.items()):
            raise ValueError("artifact component record counts do not match counters")
        expected_content_digest = overall_content_digest(self.components)
        if self.overall_content_digest != expected_content_digest:
            raise ValueError("artifact overall content digest does not match components")
        expected_manifest_digest = artifact_manifest_digest(
            self.model_dump(mode="json", exclude={"manifest_digest"}, warnings=False)
        )
        if self.manifest_digest != expected_manifest_digest:
            raise ValueError("artifact manifest digest does not match")
        return self

    @classmethod
    def create(
        cls,
        *,
        classification: ArtifactClassification,
        evidence_level: ArtifactEvidenceLevel = ArtifactEvidenceLevel.EXPLORATORY,
        run_id: UUID4,
        revision: RevisionEvidence,
        engine_configuration_digest: Sha256Digest,
        trace_digest: Sha256Digest,
        model_id: ComponentIdentifier,
        replay_id: ComponentIdentifier | None,
        prompt_template_digest: Sha256Digest,
        result_digest: Sha256Digest,
        components: tuple[ArtifactComponent, ...],
        counters: ArtifactCounters,
    ) -> ArtifactManifest:
        ordered = tuple(sorted(components, key=lambda component: component.name.value))
        eligible = revision.confirmatory_eligible
        if evidence_level is ArtifactEvidenceLevel.CONFIRMATORY and not eligible:
            raise ValueError("confirmatory artifact requires eligible revision evidence")
        values: dict[str, object] = {
            "record_type": "artifact_manifest",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": "replay_run",
            "classification": classification,
            "evidence_level": evidence_level,
            "run_id": run_id,
            "revision": revision,
            "confirmatory_eligible": eligible,
            "engine_configuration_digest": engine_configuration_digest,
            "trace_digest": trace_digest,
            "model_id": model_id,
            "replay_id": replay_id,
            "prompt_template_digest": prompt_template_digest,
            "result_digest": result_digest,
            "components": ordered,
            "counters": counters,
            "overall_content_digest": overall_content_digest(ordered),
        }
        values["manifest_digest"] = artifact_manifest_digest(values)
        return cls.model_validate(values)


def expected_component_path(name: ArtifactComponentName) -> str:
    return _COMPONENT_PATHS[name]


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MAX_ARTIFACT_COMPONENT_BYTES",
    "ArtifactAttestationsComponent",
    "ArtifactBudgetsComponent",
    "ArtifactClassification",
    "ArtifactComponent",
    "ArtifactComponentName",
    "ArtifactCounters",
    "ArtifactDecisionsComponent",
    "ArtifactDeliveriesComponent",
    "ArtifactEvidenceLevel",
    "ArtifactManifest",
    "ArtifactOutcomesComponent",
    "ArtifactRunComponent",
    "ArtifactSyntheticComponent",
    "CycleAttestation",
    "DeliveryAttestation",
    "GroundingConfigurationAttestation",
    "InterventionAttestation",
    "PolicyConfigurationAttestation",
    "RevisionEvidence",
    "RevisionSource",
    "artifact_manifest_digest",
    "component_content_digest",
    "delivery_binding_digest",
    "expected_component_path",
    "overall_content_digest",
]
