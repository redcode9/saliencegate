from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from saliencegate.artifacts import (
    ArtifactClassification,
    ArtifactCounters,
    ArtifactEvidenceLevel,
    load_validated_artifact,
)
from saliencegate.artifacts.manifest import InterventionAttestation, RevisionEvidence
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ConstraintStatus,
    CycleState,
    DeliveryOutcome,
    DeliveryState,
    DeliveryTarget,
    InterventionAction,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    UtilityLabel,
    canonical_json,
)
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    Sha256Digest,
    UnitInterval,
    UtcDatetime,
)
from saliencegate.ports.repository import LedgerHead, ProjectionDigests

INSPECT_SCHEMA_VERSION: Literal["cli-inspect-report/v1"] = "cli-inspect-report/v1"


class InspectRunMismatchError(ValueError):
    """The requested run does not identify the validated artifact."""

    def __init__(self) -> None:
        super().__init__("requested run does not match artifact")


class InspectInputError(ValueError):
    """The inspect request cannot be interpreted without exposing its value."""

    def __init__(self) -> None:
        super().__init__("inspect input failed validation")


class _InspectModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class InspectManifest(_InspectModel):
    schema_version: Literal["cli-inspect-manifest/v1"] = "cli-inspect-manifest/v1"
    classification: ArtifactClassification
    evidence_level: ArtifactEvidenceLevel
    confirmatory: bool
    confirmatory_eligible: bool
    manifest_digest: Sha256Digest
    overall_content_digest: Sha256Digest
    result_digest: Sha256Digest
    counters: ArtifactCounters
    revision: RevisionEvidence


class InspectExecution(_InspectModel):
    schema_version: Literal["cli-inspect-execution/v1"] = "cli-inspect-execution/v1"
    trace_digest: Sha256Digest
    trace_event_count: PositiveInt
    model_id: ComponentIdentifier
    model_execution_mode: Literal["structured_model", "frozen_replay"]
    replay_id: ComponentIdentifier | None
    engine_configuration_digest: Sha256Digest
    rebuild_equivalent: bool
    source_result_digest: Sha256Digest


class InspectDecision(_InspectModel):
    schema_version: Literal["cli-inspect-decision/v1"] = "cli-inspect-decision/v1"
    decision_id: UUID4
    event_sequence: PositiveInt
    invoke: bool
    risk_score: UnitInterval | None
    reason_codes: tuple[ReasonCode, ...]
    policy_version: ComponentIdentifier
    budget_snapshot: BudgetSnapshot
    cooldown_active: bool


class InspectIntervention(_InspectModel):
    schema_version: Literal["cli-inspect-intervention/v1"] = "cli-inspect-intervention/v1"
    intervention_id: UUID4
    intervention_digest: Sha256Digest
    action: InterventionAction
    delivery_target: DeliveryTarget | None
    confidence: UnitInterval
    reason_code: ReasonCode
    ttl_steps: NonNegativeInt
    grounding_version: ComponentIdentifier
    grounding_configuration_digest: Sha256Digest
    claim_fingerprints: tuple[Sha256Digest, ...]
    claim_evidence_counts: tuple[NonNegativeInt, ...]
    cited_memory_ids: tuple[UUID4, ...]
    cited_event_ids: tuple[UUID4, ...]
    rendered_text_digest: Sha256Digest | None
    created_at: UtcDatetime


class InspectCycle(_InspectModel):
    schema_version: Literal["cli-inspect-cycle/v1"] = "cli-inspect-cycle/v1"
    cycle_id: Sha256Digest
    invocation_decision_id: UUID4
    state: CycleState
    first_event_sequence: PositiveInt
    last_event_sequence: PositiveInt
    requested_delivery_target: DeliveryTarget | None
    budget_reservation: BudgetAmounts | None
    budget_settlement: BudgetAmounts | None
    model_call_count: NonNegativeInt
    memory_creates: NonNegativeInt
    memory_updates: NonNegativeInt
    memory_invalidations: NonNegativeInt
    private_status_replaced: bool
    intervention: InspectIntervention | None
    failure_reason: ReasonCode | None


class InspectBudgets(_InspectModel):
    schema_version: Literal["cli-inspect-budgets/v1"] = "cli-inspect-budgets/v1"
    limits: BudgetLimits
    configured_reservation: BudgetAmounts
    consumed: BudgetAmounts
    budget_projection_digest: Sha256Digest


class InspectDelivery(_InspectModel):
    schema_version: Literal["cli-inspect-delivery/v1"] = "cli-inspect-delivery/v1"
    delivery_id: UUID4
    event_sequence: PositiveInt
    cycle_id: Sha256Digest
    intervention_id: UUID4
    target: DeliveryTarget
    state: DeliveryState
    attempt_count: NonNegativeInt
    adapter_id_digest: Sha256Digest
    adapter_capabilities_digest: Sha256Digest
    adapter_deduplicates: bool
    outcome: DeliveryOutcome | None
    reason_code: ReasonCode | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    binding_digest: Sha256Digest


class InspectOutcome(_InspectModel):
    schema_version: Literal["cli-inspect-outcome/v1"] = "cli-inspect-outcome/v1"
    outcome_id: UUID4
    intervention_id: UUID4
    evidence_mode: OutcomeEvidenceMode
    repeated_error_status: RepeatedErrorStatus
    constraint_status: ConstraintStatus
    utility: UtilityLabel | None
    action_changed: bool | None
    task_reward: FiniteFloat | None
    task_passed: bool | None
    steps: NonNegativeInt
    tool_calls: NonNegativeInt
    memory_calls: NonNegativeInt
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    canonical_token_equivalents: NonNegativeInt
    latency_us: NonNegativeInt
    unresolved: bool
    created_at: UtcDatetime


class InspectAttestations(_InspectModel):
    schema_version: Literal["cli-inspect-attestations/v1"] = "cli-inspect-attestations/v1"
    normalized_trace_digest: Sha256Digest
    normalized_draft_digests: tuple[Sha256Digest, ...]
    persisted_event_draft_digests: tuple[Sha256Digest, ...]
    trace_record_digests: tuple[Sha256Digest, ...]
    trace_expected_event_ids: tuple[UUID4, ...]
    events_digest: Sha256Digest
    model_request_digests: tuple[Sha256Digest, ...]
    decisions_digest: Sha256Digest
    projection_digests: ProjectionDigests
    ledger_head: LedgerHead
    source_result_digest: Sha256Digest


class InspectReport(_InspectModel):
    schema_version: Literal["cli-inspect-report/v1"] = INSPECT_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    run_id: UUID4
    manifest: InspectManifest
    execution: InspectExecution
    decisions: tuple[InspectDecision, ...]
    budgets: InspectBudgets
    cycles: tuple[InspectCycle, ...]
    deliveries: tuple[InspectDelivery, ...]
    outcomes: tuple[InspectOutcome, ...]
    attestations: InspectAttestations

    @model_validator(mode="after")
    def summaries_match_the_validated_run(self) -> Self:
        counters = self.manifest.counters
        invoked_ids = {decision.decision_id for decision in self.decisions if decision.invoke}
        cycle_decision_ids = {cycle.invocation_decision_id for cycle in self.cycles}
        committed_interventions = {
            cycle.intervention.intervention_id
            for cycle in self.cycles
            if cycle.intervention is not None
        }
        if (
            self.run_id != self.attestations.ledger_head.run_id
            or self.manifest.result_digest != self.execution.source_result_digest
            or self.execution.source_result_digest != self.attestations.source_result_digest
            or self.execution.trace_event_count != counters.events
            or len(self.decisions) != counters.decisions
            or sum(decision.invoke for decision in self.decisions) != counters.invoked
            or len(self.cycles) != counters.cycles
            or len(self.attestations.model_request_digests) != counters.model_calls
            or len(self.deliveries) != counters.deliveries
            or sum(delivery.state is DeliveryState.DELIVERED for delivery in self.deliveries)
            != counters.delivered
            or len(self.outcomes) != counters.outcomes
            or invoked_ids != cycle_decision_ids
            or any(
                outcome.intervention_id not in committed_interventions
                or outcome.unresolved is not (outcome.utility is None)
                for outcome in self.outcomes
            )
            or len(self.attestations.trace_expected_event_ids) != counters.events
        ):
            raise ValueError("inspection summary does not match the validated artifact")
        return self


def _manifest_path(artifact_path: os.PathLike[str] | str) -> Path:
    if isinstance(artifact_path, bytes):
        raise InspectInputError()
    try:
        path = Path(os.fspath(artifact_path))
        metadata = path.lstat()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        raise InspectInputError() from None
    if stat.S_ISDIR(metadata.st_mode):
        return path / "manifest.json"
    if stat.S_ISREG(metadata.st_mode) and path.name == "manifest.json":
        return path
    raise InspectInputError()


def _intervention_report(value: InterventionAttestation | None) -> InspectIntervention | None:
    if value is None:
        return None
    # Cycle components were type-validated by ``load_validated_artifact``.  The
    # attribute-based construction keeps receipt/request internals out of this view.
    return InspectIntervention(
        intervention_id=value.intervention_id,
        intervention_digest=value.intervention_digest,
        action=value.action,
        delivery_target=value.delivery_target,
        confidence=value.confidence,
        reason_code=value.reason_code,
        ttl_steps=value.ttl_steps,
        grounding_version=value.grounding_version,
        grounding_configuration_digest=value.grounding_configuration_digest,
        claim_fingerprints=value.claim_fingerprints,
        claim_evidence_counts=value.claim_evidence_counts,
        cited_memory_ids=value.cited_memory_ids,
        cited_event_ids=value.cited_event_ids,
        rendered_text_digest=value.rendered_text_digest,
        created_at=value.created_at,
    )


def run_inspect(
    run_id: UUID,
    *,
    artifact_path: os.PathLike[str] | str,
) -> InspectReport:
    """Validate and inspect one artifact without reopening files or exposing payloads."""

    if type(run_id) is not UUID or run_id.version != 4:
        raise InspectInputError()
    loaded = load_validated_artifact(_manifest_path(artifact_path))
    if loaded.manifest.run_id != run_id:
        raise InspectRunMismatchError()

    decisions = tuple(
        InspectDecision(
            decision_id=decision.decision_id,
            event_sequence=decision.event_sequence,
            invoke=decision.invoke,
            risk_score=decision.risk_score,
            reason_codes=decision.reason_codes,
            policy_version=decision.policy_version,
            budget_snapshot=decision.budget_snapshot,
            cooldown_active=decision.cooldown_active,
        )
        for decision in loaded.decisions.decisions
    )
    cycles = tuple(
        InspectCycle(
            cycle_id=cycle.cycle_id,
            invocation_decision_id=cycle.invocation_decision_id,
            state=cycle.state,
            first_event_sequence=cycle.first_event_sequence,
            last_event_sequence=cycle.last_event_sequence,
            requested_delivery_target=cycle.requested_delivery_target,
            budget_reservation=cycle.budget_reservation,
            budget_settlement=cycle.budget_settlement,
            model_call_count=len(cycle.model_call_digests),
            memory_creates=cycle.memory_creates,
            memory_updates=cycle.memory_updates,
            memory_invalidations=cycle.memory_invalidations,
            private_status_replaced=cycle.private_status_replaced,
            intervention=_intervention_report(cycle.intervention),
            failure_reason=cycle.failure_reason,
        )
        for cycle in loaded.budgets.cycles
    )
    deliveries = tuple(
        InspectDelivery(
            delivery_id=delivery.delivery_id,
            event_sequence=delivery.event_sequence,
            cycle_id=delivery.cycle_id,
            intervention_id=delivery.intervention_id,
            target=delivery.target,
            state=delivery.state,
            attempt_count=delivery.attempt_count,
            adapter_id_digest=delivery.adapter_id_digest,
            adapter_capabilities_digest=delivery.adapter_capabilities_digest,
            adapter_deduplicates=delivery.adapter_deduplicates,
            outcome=delivery.outcome,
            reason_code=delivery.reason_code,
            created_at=delivery.created_at,
            updated_at=delivery.updated_at,
            binding_digest=delivery.binding_digest,
        )
        for delivery in loaded.deliveries.deliveries
    )
    outcomes = tuple(
        InspectOutcome(
            outcome_id=outcome.outcome_id,
            intervention_id=outcome.intervention_id,
            evidence_mode=outcome.evidence_mode,
            repeated_error_status=outcome.repeated_error_status,
            constraint_status=outcome.constraint_status,
            utility=outcome.utility,
            action_changed=outcome.action_changed,
            task_reward=outcome.task_reward,
            task_passed=outcome.task_passed,
            steps=outcome.steps,
            tool_calls=outcome.tool_calls,
            memory_calls=outcome.memory_calls,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            canonical_token_equivalents=outcome.canonical_token_equivalents,
            latency_us=outcome.latency_us,
            unresolved=outcome.utility is None,
            created_at=outcome.created_at,
        )
        for outcome in loaded.outcomes.outcomes
    )
    manifest = loaded.manifest
    run = loaded.run
    attestations = loaded.attestations
    return InspectReport(
        run_id=manifest.run_id,
        manifest=InspectManifest(
            classification=manifest.classification,
            evidence_level=manifest.evidence_level,
            confirmatory=manifest.confirmatory,
            confirmatory_eligible=manifest.confirmatory_eligible,
            manifest_digest=manifest.manifest_digest,
            overall_content_digest=manifest.overall_content_digest,
            result_digest=manifest.result_digest,
            counters=manifest.counters,
            revision=manifest.revision,
        ),
        execution=InspectExecution(
            trace_digest=run.trace_digest,
            trace_event_count=run.trace_event_count,
            model_id=run.model_id,
            model_execution_mode=run.model_execution_mode,
            replay_id=run.replay_id,
            engine_configuration_digest=run.engine_configuration_digest,
            rebuild_equivalent=run.rebuild_equivalent,
            source_result_digest=run.source_result_digest,
        ),
        decisions=decisions,
        budgets=InspectBudgets(
            limits=loaded.budgets.limits,
            configured_reservation=loaded.budgets.configured_reservation,
            consumed=loaded.budgets.consumed,
            budget_projection_digest=loaded.budgets.budget_projection_digest,
        ),
        cycles=cycles,
        deliveries=deliveries,
        outcomes=outcomes,
        attestations=InspectAttestations(
            normalized_trace_digest=attestations.normalized_trace_digest,
            normalized_draft_digests=attestations.normalized_draft_digests,
            persisted_event_draft_digests=attestations.persisted_event_draft_digests,
            trace_record_digests=attestations.trace_record_digests,
            trace_expected_event_ids=attestations.trace_expected_event_ids,
            events_digest=attestations.events_digest,
            model_request_digests=attestations.model_request_digests,
            decisions_digest=attestations.decisions_digest,
            projection_digests=attestations.projection_digests,
            ledger_head=attestations.ledger_head,
            source_result_digest=attestations.source_result_digest,
        ),
    )


def render_inspect_json(report: InspectReport) -> str:
    """Render exactly one canonical machine record; the caller owns stdout."""

    validated = InspectReport.model_validate(report)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_inspect_human(report: InspectReport) -> str:
    """Render a concise digest-only inspection; the caller owns stdout."""

    validated = InspectReport.model_validate(report)
    invoked = sum(decision.invoke for decision in validated.decisions)
    committed = sum(cycle.state is CycleState.COMMITTED for cycle in validated.cycles)
    failed = len(validated.cycles) - committed
    delivered = sum(delivery.state is DeliveryState.DELIVERED for delivery in validated.deliveries)
    unresolved = sum(outcome.unresolved for outcome in validated.outcomes)
    consumed = validated.budgets.consumed
    lines = [
        f"Run {validated.run_id}",
        f"manifest digest: {validated.manifest.manifest_digest}",
        f"trace digest: {validated.execution.trace_digest}",
        f"decisions: {len(validated.decisions)} ({invoked} invoked)",
        f"cycles: {len(validated.cycles)} ({committed} committed, {failed} failed)",
        (
            "budget consumed: "
            f"{consumed.model_calls} model calls, "
            f"{consumed.input_tokens} input tokens, "
            f"{consumed.output_tokens} output tokens"
        ),
        f"deliveries: {len(validated.deliveries)} ({delivered} delivered)",
        f"outcomes: {len(validated.outcomes)} ({unresolved} unresolved)",
    ]
    for decision in validated.decisions:
        disposition = "invoke" if decision.invoke else "silence"
        reasons = ",".join(reason.value for reason in decision.reason_codes)
        lines.append(
            f"decision {decision.event_sequence}: {disposition}; "
            f"id={decision.decision_id}; reasons={reasons}"
        )
    for cycle in validated.cycles:
        intervention = cycle.intervention
        if intervention is None:
            evidence = "evidence=none"
        else:
            memory_ids = ",".join(str(value) for value in intervention.cited_memory_ids) or "none"
            event_ids = ",".join(str(value) for value in intervention.cited_event_ids) or "none"
            evidence = f"memory_ids={memory_ids}; event_ids={event_ids}"
        lines.append(f"cycle {cycle.cycle_id}: {cycle.state.value}; {evidence}")
    return "\n".join(lines) + "\n"


__all__ = [
    "INSPECT_SCHEMA_VERSION",
    "InspectAttestations",
    "InspectBudgets",
    "InspectCycle",
    "InspectDecision",
    "InspectDelivery",
    "InspectExecution",
    "InspectInputError",
    "InspectIntervention",
    "InspectManifest",
    "InspectOutcome",
    "InspectReport",
    "InspectRunMismatchError",
    "render_inspect_human",
    "render_inspect_json",
    "run_inspect",
]
