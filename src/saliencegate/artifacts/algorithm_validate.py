from __future__ import annotations

import hmac
import json
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal, Never, cast

from pydantic import BaseModel, ConfigDict

from saliencegate.artifacts import algorithm_manifest as _algorithm
from saliencegate.artifacts import algorithm_projection as _projection
from saliencegate.artifacts import tree as artifact_tree
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
)
from saliencegate.artifacts.validate import (
    ArtifactValidationCode,
    ArtifactValidationError,
)
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    ConstraintStatus,
    CycleState,
    DeduplicationGuarantee,
    DeliveryState,
    DeliveryTarget,
    InterventionAction,
    OutcomeEvidenceMode,
    PayloadDigestAlgorithm,
    ReasonCode,
    RepeatedErrorStatus,
    canonical_digest,
    canonical_json,
)
from saliencegate.domain import (
    cycle_id as derive_cycle_id,
)
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.experiments.conditions import (
    SelectionMode,
    Stage2ConditionId,
)
from saliencegate.models.replay_two_phase import (
    two_phase_replay_fixture_digest_from_receipts,
)
from saliencegate.ports.adapters import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapabilities,
    DeliveryChannel,
    DeliveryRole,
    InjectionMapping,
    adapter_capabilities_digest,
)
from saliencegate.ports.model_calls import (
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallStatus,
)
from saliencegate.ports.repository import LedgerHead, ProjectionDigests
from saliencegate.ports.two_phase import TwoPhaseUsage, call_policy_accepts_receipts
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.runtime.algorithm_result import algorithm_runtime_uuid
from saliencegate.runtime.scheduling import FixedStepReason, FixedStepSchedule

_MAX_MANIFEST_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE2_DELIVERY_ADAPTER_ID = "stage2-offline-delivery/v1"
_INTERVENTION_REJECTION_REASONS = {
    StructuredCallParseStatus.SCHEMA_INVALID: ReasonCode.SCHEMA_INVALID,
    StructuredCallParseStatus.EMPTY_REMINDER: ReasonCode.NO_GROUNDED_CLAIMS,
    StructuredCallParseStatus.CLAIM_OVER_LIMIT: ReasonCode.CLAIM_OVER_LIMIT,
}


class AlgorithmSourceResultAssurance(StrEnum):
    PRODUCER_ATTESTED = "producer_attested"
    RECOMPUTED_FROM_RAW = "recomputed_from_raw"


class AlgorithmArtifactValidationReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    schema_version: Literal["algorithm-validation-report/v1"] = "algorithm-validation-report/v1"
    valid: Literal[True] = True
    structurally_valid: Literal[True] = True
    self_consistent: Literal[True] = True
    expected_digest_matched: bool | None
    source_result_assurance: AlgorithmSourceResultAssurance
    confirmatory: Literal[False]
    manifest_digest: str
    overall_content_digest: str
    component_count: int


class ValidatedAlgorithmAttestations(BaseModel):
    """Safe view of attestations that deliberately omits the optional raw result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["validated-algorithm-attestations/v1"] = (
        "validated-algorithm-attestations/v1"
    )
    run_id: UUID4
    boundaries: tuple[_algorithm.AlgorithmBoundaryAttestation, ...]
    semantic_projection_digests: ProjectionDigests
    repository_projection_digests: ProjectionDigests
    ledger_entries: tuple[_algorithm.AlgorithmLedgerEntryAttestation, ...]
    ledger_entry_count: int
    ledger_head: LedgerHead
    rebuild_equivalent: Literal[True]
    source_result_digest: Sha256Digest
    attestations_component_digest: Sha256Digest


class ValidatedAlgorithmArtifact(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["validated-algorithm-artifact/v1"] = "validated-algorithm-artifact/v1"
    report: AlgorithmArtifactValidationReport
    manifest: _algorithm.AlgorithmArtifactManifest
    run: _algorithm.AlgorithmRunComponent
    trajectory: _algorithm.AlgorithmTrajectoryComponent
    calls: _algorithm.AlgorithmCallsComponent
    decisions: _algorithm.AlgorithmDecisionsComponent
    cycles: _algorithm.AlgorithmCyclesComponent
    deliveries: _algorithm.AlgorithmDeliveriesComponent
    outcomes: _algorithm.AlgorithmOutcomesComponent
    metrics: _algorithm.AlgorithmMetricsComponent
    attestations: ValidatedAlgorithmAttestations


def _raise(code: ArtifactValidationCode) -> Never:
    raise ArtifactValidationError(code)


_TREE_ERROR_CODES: dict[
    artifact_tree.ClosedTreeReadErrorKind,
    ArtifactValidationCode,
] = {
    artifact_tree.ClosedTreeReadErrorKind.UNSAFE_PATH: ArtifactValidationCode.UNSAFE_PATH,
    artifact_tree.ClosedTreeReadErrorKind.MISSING_ENTRY: (ArtifactValidationCode.MISSING_COMPONENT),
    artifact_tree.ClosedTreeReadErrorKind.UNSAFE_ENTRY: (ArtifactValidationCode.UNSAFE_COMPONENT),
    artifact_tree.ClosedTreeReadErrorKind.INVALID_DESCRIPTOR: (
        ArtifactValidationCode.INVALID_MANIFEST
    ),
}


def _raise_tree_error(error: artifact_tree.ClosedTreeReadError) -> Never:
    _raise(_TREE_ERROR_CODES[error.kind])


def _reject_nonfinite(value: str) -> Never:
    del value
    raise ValueError("non-finite JSON value")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_canonical_object(data: bytes, *, manifest: bool) -> dict[str, object]:
    code = (
        ArtifactValidationCode.INVALID_MANIFEST
        if manifest
        else ArtifactValidationCode.INVALID_COMPONENT
    )
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
        if type(parsed) is not dict or canonical_json(parsed) != data:
            raise ValueError
        return cast(dict[str, object], parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        _raise(code)


def _preflight_manifest(payload: Mapping[str, object]) -> None:
    version = payload.get("schema_version")
    if type(version) is not str:
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    if version != _algorithm.ALGORITHM_ARTIFACT_SCHEMA_VERSION:
        if version.startswith("algorithm-artifact/v"):
            _raise(ArtifactValidationCode.UNSUPPORTED_VERSION)
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    if payload.get("artifact_kind") != "algorithm_run":
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    components = payload.get("components")
    if type(components) is not list:
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    for item in components:
        if type(item) is not dict:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        raw_name = item.get("name")
        raw_path = item.get("path")
        if type(raw_name) is not str:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        try:
            name = _algorithm.AlgorithmArtifactComponentName(raw_name)
        except (TypeError, ValueError):
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if raw_path != _algorithm.expected_algorithm_component_path(name):
            _raise(ArtifactValidationCode.UNSAFE_PATH)


_COMPONENT_MODELS: dict[
    _algorithm.AlgorithmArtifactComponentName,
    type[BaseModel],
] = {
    _algorithm.AlgorithmArtifactComponentName.ATTESTATIONS: (
        _algorithm.AlgorithmAttestationsComponent
    ),
    _algorithm.AlgorithmArtifactComponentName.CALLS: _algorithm.AlgorithmCallsComponent,
    _algorithm.AlgorithmArtifactComponentName.CYCLES: _algorithm.AlgorithmCyclesComponent,
    _algorithm.AlgorithmArtifactComponentName.DECISIONS: (_algorithm.AlgorithmDecisionsComponent),
    _algorithm.AlgorithmArtifactComponentName.DELIVERIES: (_algorithm.AlgorithmDeliveriesComponent),
    _algorithm.AlgorithmArtifactComponentName.METRICS: _algorithm.AlgorithmMetricsComponent,
    _algorithm.AlgorithmArtifactComponentName.OUTCOMES: (_algorithm.AlgorithmOutcomesComponent),
    _algorithm.AlgorithmArtifactComponentName.RUN: _algorithm.AlgorithmRunComponent,
    _algorithm.AlgorithmArtifactComponentName.TRAJECTORY: (_algorithm.AlgorithmTrajectoryComponent),
}


def _parse_component(
    name: _algorithm.AlgorithmArtifactComponentName,
    data: bytes,
) -> BaseModel:
    _decode_canonical_object(data, manifest=False)
    try:
        return _COMPONENT_MODELS[name].model_validate_json(data)
    except Exception:
        _raise(ArtifactValidationCode.INVALID_COMPONENT)


def _budget_limits(
    reservation: BudgetAmounts,
    events: int,
    max_call_latency_us: int,
) -> BudgetLimits:
    return BudgetLimits(
        model_calls=reservation.model_calls * events,
        input_tokens=reservation.input_tokens * events,
        output_tokens=reservation.output_tokens * events,
        canonical_token_equivalents=reservation.canonical_token_equivalents * events,
        latency_us=reservation.latency_us * events,
        interventions=reservation.interventions * events,
        schema_repairs=reservation.schema_repairs * events,
        max_call_latency_us=max_call_latency_us,
    )


def _sum_budgets(values: tuple[BudgetAmounts, ...]) -> BudgetAmounts:
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
        **{field: sum(getattr(value, field) for value in values) for field in fields}
    )


def _same_run(
    manifest: _algorithm.AlgorithmArtifactManifest,
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    calls: _algorithm.AlgorithmCallsComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    outcomes: _algorithm.AlgorithmOutcomesComponent,
    metrics: _algorithm.AlgorithmMetricsComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> bool:
    return all(
        run_id == manifest.run_id
        for run_id in (
            run.run_id,
            trajectory.run_id,
            calls.run_id,
            decisions.run_id,
            cycles.run_id,
            deliveries.run_id,
            outcomes.run_id,
            metrics.run_id,
            attestations.run_id,
        )
    )


def _validate_manifest_bindings(
    manifest: _algorithm.AlgorithmArtifactManifest,
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    raw_present = attestations.raw_synthetic_result is not None
    if (
        manifest.condition_id != run.condition.condition_id
        or manifest.condition_digest != run.condition.condition_digest
        or manifest.cycle_mode is not run.cycle_mode
        or manifest.trace_digest != trajectory.trace_digest
        or manifest.schedule_digest != trajectory.schedule.schedule_digest
        or manifest.window_digests != tuple(item.window_digest for item in trajectory.windows)
        or manifest.window_set_digest != trajectory.window_set_digest
        or manifest.prompt_bundle_digest != run.prompt_bundle.bundle_digest
        or manifest.model_id != run.model_profile.model_id
        or manifest.model_profile_digest != run.model_profile.profile_digest
        or manifest.execution != run.execution
        or manifest.execution_digest != run.execution.execution_digest
        or manifest.result_digest != run.source_result_digest
        or manifest.result_digest != attestations.source_result_digest
        or any(
            item.version != run.condition.shared_controls.window_version
            for item in trajectory.windows
        )
        or raw_present is not (manifest.classification is ArtifactClassification.SYNTHETIC_RAW)
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_cardinality(
    manifest: _algorithm.AlgorithmArtifactManifest,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    calls: _algorithm.AlgorithmCallsComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    outcomes: _algorithm.AlgorithmOutcomesComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    counters = manifest.counters
    actual = (
        len(trajectory.records),
        trajectory.schedule.invocation_count,
        len(decisions.decisions),
        len(cycles.cycles),
        len(calls.groups),
        len(calls.calls),
        len(deliveries.deliveries),
        len(outcomes.outcomes),
        len(attestations.ledger_entries),
    )
    expected = (
        counters.events,
        counters.scheduled_invocations,
        counters.decisions,
        counters.cycles,
        counters.requests,
        counters.model_calls,
        counters.deliveries,
        counters.outcomes,
        counters.ledger_entries,
    )
    if (
        actual != expected
        or len(trajectory.windows) != counters.scheduled_invocations
        or len(attestations.boundaries) != counters.scheduled_invocations
        or tuple(item.boundary_event_id for item in trajectory.windows)
        != tuple(item.boundary_event_id for item in attestations.boundaries)
        or tuple(item.boundary_event_sequence for item in trajectory.windows)
        != tuple(item.boundary_event_sequence for item in attestations.boundaries)
        or tuple(item.window_digest for item in trajectory.windows)
        != tuple(item.window_digest for item in attestations.boundaries)
    ):
        _raise(ArtifactValidationCode.INCONSISTENT_COUNTERS)


def _validate_execution_binding(
    run: _algorithm.AlgorithmRunComponent,
    calls: _algorithm.AlgorithmCallsComponent,
) -> None:
    execution = run.execution
    fixture = execution.response_fixture
    if not _algorithm.algorithm_call_evidence_matches_execution_mode(
        execution.execution_mode,
        calls.calls,
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if execution.execution_mode is _algorithm.AlgorithmExecutionMode.FROZEN_REPLAY:
        if calls.calls:
            try:
                fixture_digest = (
                    None
                    if fixture is None
                    else two_phase_replay_fixture_digest_from_receipts(
                        calls.calls,
                        replay_id=fixture.replay_id,
                    )
                )
            except Exception:
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
            if (
                fixture is None
                or fixture.fixture_digest != fixture_digest
                or fixture.response_count != len(calls.calls)
                or fixture.consumed_count != len(calls.calls)
            ):
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        elif fixture is not None:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if not _algorithm.algorithm_call_evidence_matches_tokenizer(
        execution.tokenizer,
        calls.calls,
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_decisions(
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
) -> None:
    active = run.condition.condition_id is not Stage2ConditionId.NO_MEMORY
    cycle_sequences = tuple(item.boundary_event_sequence for item in cycles.cycles)
    if cycle_sequences != tuple(sorted(cycle_sequences)) or len(set(cycle_sequences)) != len(
        cycle_sequences
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    settled = BudgetAmounts()
    cycle_index = 0
    for record, scheduled, decision in zip(
        trajectory.records,
        trajectory.schedule.decisions,
        decisions.decisions,
        strict=True,
    ):
        while (
            cycle_index < len(cycles.cycles)
            and cycles.cycles[cycle_index].boundary_event_sequence < decision.event_sequence
        ):
            settled = _sum_budgets((settled, cycles.cycles[cycle_index].budget_settlement))
            cycle_index += 1
        invoke = scheduled.invoke and active
        reason = (
            ReasonCode.BOOTSTRAP
            if invoke and scheduled.reason is FixedStepReason.BOOTSTRAP
            else ReasonCode.SCRIPTED_INVOKE
            if invoke
            else ReasonCode.SCRIPTED_SILENCE
        )
        if (
            decision.decision_id
            != algorithm_runtime_uuid(
                trajectory.trace_digest,
                "stage2-decision",
                record.event_sequence,
            )
            or decision.run_id != run.run_id
            or decision.event_sequence != record.event_sequence
            or decision.invoke is not invoke
            or decision.risk_score is not None
            or decision.reason_codes != (reason,)
            or decision.policy_version != run.policy_version
            or decision.configuration_digest != run.condition.condition_digest
            or decision.budget_snapshot.limits != run.budget_limits
            or decision.budget_snapshot.reserved != BudgetAmounts()
            or decision.budget_snapshot.consumed != settled
            or decision.cooldown_active
            or decision.created_at != record.event_timestamp
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_calls_and_cycles(
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    calls: _algorithm.AlgorithmCallsComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    for boundary, window in zip(
        attestations.boundaries,
        trajectory.windows,
        strict=True,
    ):
        decision = decisions.decisions[boundary.boundary_event_sequence - 1]
        observed = boundary.observation.observed
        try:
            boundary_schedule_digest = FixedStepSchedule(
                schedule_version=trajectory.schedule.schedule_version,
                run_id=trajectory.schedule.run_id,
                boundary_event_sequence=boundary.boundary_event_sequence,
                trajectory_prefix_digest=window.trajectory_prefix_digest,
                decisions=trajectory.schedule.decisions[: boundary.boundary_event_sequence],
                invocation_count=boundary.invocation_ordinal,
            ).schedule_digest
        except Exception:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        if (
            boundary.invocation_decision_id != decision.decision_id
            or observed.run_id != run.run_id
            or observed.invocation_decision_digest != canonical_digest(decision)
            or observed.schedule_digest != boundary_schedule_digest
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    active_boundaries = tuple(
        boundary for boundary in attestations.boundaries if boundary.cycle_id is not None
    )
    expected_phases = run.condition.expected.call_phases
    templates = {item.phase: item for item in run.prompt_bundle.templates}
    first_event_sequence = 1
    if len(active_boundaries) != len(cycles.cycles):
        _raise(ArtifactValidationCode.INCONSISTENT_COUNTERS)
    for group, cycle, boundary in zip(
        calls.groups,
        cycles.cycles,
        active_boundaries,
        strict=True,
    ):
        grouped_calls = tuple(
            item for item in calls.calls if item.receipt_digest in group.call_receipt_digests
        )
        try:
            usage = TwoPhaseUsage.from_receipts(grouped_calls)
            policy_accepts = call_policy_accepts_receipts(run.call_policy, grouped_calls)
            expected_cycle_id = derive_cycle_id(
                run.run_id,
                first_event_sequence,
                cycle.boundary_event_sequence,
                run.policy_version,
                run.condition.condition_digest,
                run.condition.shared_controls.grounding.pipeline_version,
                run.condition.shared_controls.grounding.configuration_digest,
                run.condition.shared_controls.requested_delivery_target,
            )
        except Exception:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        provider_known = (
            usage.provider_input_tokens is not None and usage.provider_output_tokens is not None
        )
        expected_settlement = BudgetAmounts(
            model_calls=usage.model_calls,
            input_tokens=(
                cast(int, usage.provider_input_tokens)
                if provider_known
                else run.cycle_reservation.input_tokens
            ),
            output_tokens=(
                cast(int, usage.provider_output_tokens)
                if provider_known
                else run.cycle_reservation.output_tokens
            ),
            canonical_token_equivalents=(
                usage.canonical_token_equivalents
                if usage.canonical_token_equivalents is not None
                else run.cycle_reservation.canonical_token_equivalents
            ),
            latency_us=usage.latency_us,
            interventions=int(cycle.intervention_action is InterventionAction.REMIND),
            schema_repairs=usage.schema_repairs,
        )
        decision = decisions.decisions[cycle.boundary_event_sequence - 1]
        final_parse_status = grouped_calls[-1].parse_status
        parse_rejection_reason = _INTERVENTION_REJECTION_REASONS.get(final_parse_status)
        if (
            group.invocation_ordinal != cycle.invocation_ordinal
            or group.invocation_ordinal != boundary.invocation_ordinal
            or group.boundary_event_sequence != cycle.boundary_event_sequence
            or group.boundary_event_sequence != boundary.boundary_event_sequence
            or group.cycle_id != cycle.cycle_id
            or group.cycle_id != boundary.cycle_id
            or cycle.cycle_id != expected_cycle_id
            or group.cycle_request_digest != cycle.cycle_request_digest
            or group.call_receipt_digests != cycle.call_receipt_digests
            or cycle.revision != 4
            or cycle.invocation_decision_id != decision.decision_id
            or cycle.invocation_decision_id != boundary.invocation_decision_id
            or cycle.intervention_id
            != algorithm_runtime_uuid(
                trajectory.trace_digest,
                "stage2-intervention",
                cycle.cycle_id,
            )
            or cycle.window_digest != boundary.window_digest
            or cycle.batch_digest != cycle.window_digest
            or cycle.policy_version != run.policy_version
            or cycle.configuration_digest != run.condition.condition_digest
            or cycle.grounding_version != run.condition.shared_controls.grounding.pipeline_version
            or cycle.grounding_configuration_digest
            != run.condition.shared_controls.grounding.configuration_digest
            or cycle.budget_reservation != run.cycle_reservation
            or cycle.budget_settlement != expected_settlement
            or not policy_accepts
            or tuple(item.phase for item in grouped_calls) != expected_phases
            or any(item.status is not StructuredCallStatus.COMPLETED for item in grouped_calls)
            or grouped_calls[0].parse_status is not StructuredCallParseStatus.VALID
            or (
                run.condition.expected.selection_mode is not SelectionMode.LEXICAL_TOP_K
                and final_parse_status is not StructuredCallParseStatus.VALID
                and parse_rejection_reason is not cycle.reason_code
            )
            or tuple(item.call_digest for item in grouped_calls) != cycle.model_call_digests
            or tuple(item.usage.latency_us for item in grouped_calls)
            != cycle.model_call_latencies_us
            or any(item.window_digest != cycle.window_digest for item in grouped_calls)
            or any(item.model_id != run.model_profile.model_id for item in grouped_calls)
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        for call in grouped_calls:
            template = templates.get(call.phase)
            if (
                template is None
                or call.prompt_template_id != template.template_id
                or call.prompt_template_digest != template.template_digest
            ):
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        observed = boundary.observation.observed
        if (
            boundary.observation.condition_id != run.condition.condition_id
            or boundary.observation.condition_digest != run.condition.condition_digest
            or observed.call_receipt_digests != group.call_receipt_digests
            or observed.call_phases != expected_phases
            or grouped_calls[0].bank_view_digest != observed.current_bank_view_digest
            or (
                run.condition.expected.selection_mode is not SelectionMode.LEXICAL_TOP_K
                and grouped_calls[-1].bank_view_digest != observed.candidate_bank_view_digest
            )
            or cycle.observation_digest != boundary.observation.observation_digest
            or cycle.boundary_evidence_digest != boundary.source_evidence_digest
            or cycle.intervention_digest != observed.intervention_digest
            or cycle.intervention_action is not observed.intervention_action
            or observed.delivery_record_digests
            != (() if cycle.delivery_source_digest is None else (cycle.delivery_source_digest,))
            or (
                cycle.memory_create_count
                + cycle.memory_update_count
                + cycle.memory_invalidation_count
                + int(cycle.private_status_replaced)
                != observed.memory_mutation_count
            )
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        if run.condition.expected.selection_mode is SelectionMode.LEXICAL_TOP_K:
            selector = cycle.selector_provenance
            controls = run.condition.shared_controls.retrieval
            if (
                selector is None
                or cycle.grounding_model_call_index is not None
                or selector.selector_id != controls.retrieval_version
                or selector.configuration_digest != controls.configuration_digest
                or selector.request_digest != observed.retrieval_request_digest
                or selector.result_digest != observed.retrieval_result_digest
            ):
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        elif (
            cycle.selector_provenance is not None
            or cycle.grounding_model_call_index != len(grouped_calls) - 1
            or cycle.grounding_model_call_digest != grouped_calls[-1].call_digest
            or group.grounding_call_index != grouped_calls[-1].model_call_index
            or group.grounding_state_digest != grouped_calls[-1].grounding_state_digest
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        first_event_sequence = cycle.boundary_event_sequence + 1


def _validate_deliveries_and_outcomes(
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    outcomes: _algorithm.AlgorithmOutcomesComponent,
) -> None:
    capabilities = AdapterCapabilities(
        schema_version="1.0",
        adapter_id=_STAGE2_DELIVERY_ADAPTER_ID,
        pre_action_interception=False,
        deduplicates_delivery_id=True,
        deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        injection_mappings=(
            InjectionMapping(
                channel=DeliveryChannel.PROVIDER_DATA,
                role=DeliveryRole.DATA,
                provider_channel="context",
            ),
        ),
    )
    expected_capabilities_digest = adapter_capabilities_digest(capabilities)
    trace_digest = trajectory.trace_digest
    reminders = tuple(
        cycle for cycle in cycles.cycles if cycle.intervention_action is InterventionAction.REMIND
    )
    silences = tuple(
        cycle for cycle in cycles.cycles if cycle.intervention_action is InterventionAction.SILENCE
    )
    if (
        len(reminders) != len(deliveries.deliveries)
        or any(cycle.delivery_source_digest is not None for cycle in silences)
        or tuple(item.cycle_id for item in deliveries.deliveries)
        != tuple(item.cycle_id for item in reminders)
    ):
        _raise(ArtifactValidationCode.UNGROUNDED_DELIVERY)
    for cycle, delivery in zip(reminders, deliveries.deliveries, strict=True):
        boundary_timestamp = trajectory.records[delivery.event_sequence - 1].event_timestamp
        if (
            delivery.state is not DeliveryState.DELIVERED
            or delivery.intervention_id != cycle.intervention_id
            or delivery.rendered_text_digest != cycle.rendered_text_digest
            or delivery.source_delivery_digest != cycle.delivery_source_digest
            or delivery.event_sequence != cycle.boundary_event_sequence
            or delivery.target is not DeliveryTarget.NEXT_MODEL_CALL
            or delivery.attempt_count != 1
            or delivery.adapter_id_digest != canonical_digest(_STAGE2_DELIVERY_ADAPTER_ID)
            or not delivery.adapter_deduplicates
            or delivery.adapter_deduplication_guarantee
            is not DeduplicationGuarantee.DURABLE_DELIVERY_ID
            or delivery.adapter_supports_pre_action
            or delivery.adapter_contract_version != ADAPTER_CONTRACT_VERSION
            or delivery.adapter_capabilities_digest != expected_capabilities_digest
            or delivery.claim_id
            != algorithm_runtime_uuid(
                trace_digest,
                "stage2-delivery-worker",
                delivery.delivery_id,
                1,
            )
            or delivery.attempt_id
            != algorithm_runtime_uuid(
                trace_digest,
                "stage2-delivery-worker",
                delivery.delivery_id,
                2,
            )
            or delivery.receipt_digest
            != canonical_digest({"provider_receipt_id": "stage2-offline-1"})
            or delivery.created_at != boundary_timestamp
            or delivery.updated_at != boundary_timestamp
        ):
            _raise(ArtifactValidationCode.UNGROUNDED_DELIVERY)
    if tuple(item.intervention_id for item in outcomes.outcomes) != tuple(
        item.intervention_id for item in cycles.cycles
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    for cycle, outcome in zip(cycles.cycles, outcomes.outcomes, strict=True):
        boundary_timestamp = trajectory.records[cycle.boundary_event_sequence - 1].event_timestamp
        if (
            outcome.run_id != run.run_id
            or outcome.outcome_id
            != algorithm_runtime_uuid(
                trace_digest,
                "stage2-outcome",
                cycle.intervention_id,
            )
            or outcome.repeated_error_status is not RepeatedErrorStatus.UNKNOWN
            or outcome.constraint_status is not ConstraintStatus.UNKNOWN
            or outcome.evidence_mode is not OutcomeEvidenceMode.POLICY_REPLAY
            or outcome.next_action_fingerprint is not None
            or outcome.utility is not None
            or outcome.action_changed is not None
            or outcome.task_reward is not None
            or outcome.task_passed is not None
            or (
                outcome.steps,
                outcome.tool_calls,
                outcome.memory_calls,
                outcome.input_tokens,
                outcome.output_tokens,
                outcome.canonical_token_equivalents,
                outcome.latency_us,
            )
            != (0, 0, 0, 0, 0, 0, 0)
            or outcome.created_at != boundary_timestamp
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _metric_totals(calls: _algorithm.AlgorithmCallsComponent) -> dict[str, int | None]:
    provider_known = all(
        item.usage.provider_input_tokens is not None
        and item.usage.provider_output_tokens is not None
        for item in calls.calls
    )
    canonical_known = all(
        item.usage.canonical_input_tokens is not None
        and item.usage.canonical_output_tokens is not None
        for item in calls.calls
    )
    provider_input = (
        sum(cast(int, item.usage.provider_input_tokens) for item in calls.calls)
        if provider_known
        else None
    )
    provider_output = (
        sum(cast(int, item.usage.provider_output_tokens) for item in calls.calls)
        if provider_known
        else None
    )
    canonical_input = (
        sum(cast(int, item.usage.canonical_input_tokens) for item in calls.calls)
        if canonical_known
        else None
    )
    canonical_output = (
        sum(cast(int, item.usage.canonical_output_tokens) for item in calls.calls)
        if canonical_known
        else None
    )
    return {
        "provider_input": provider_input,
        "provider_output": provider_output,
        "canonical_input": canonical_input,
        "canonical_output": canonical_output,
        "canonical_total": (
            canonical_input + canonical_output
            if canonical_input is not None and canonical_output is not None
            else None
        ),
    }


def _replayed_memory_record_count(
    cycles: _algorithm.AlgorithmCyclesComponent,
) -> int | None:
    expected_record_count = 0
    feasible_active: dict[bool, int] = {False: 0}
    for cycle in cycles.cycles:
        creates = cycle.memory_create_count
        updates = cycle.memory_update_count
        if updates != 0:
            return None
        invalidations = cycle.memory_invalidation_count
        replaces_private = cycle.private_status_replaced
        targets = updates + invalidations
        next_active: dict[bool, int] = {}
        for has_private, active in feasible_active.items():
            if not replaces_private:
                if not has_private and targets <= active:
                    next_active[False] = max(
                        next_active.get(False, -1),
                        active - invalidations + creates,
                    )
                if has_private:
                    if targets <= active and (invalidations == 0 or invalidations < active):
                        next_active[True] = max(
                            next_active.get(True, -1),
                            active - invalidations + creates,
                        )
                    if invalidations >= 1 and targets <= active:
                        next_active[False] = max(
                            next_active.get(False, -1),
                            active - invalidations + creates,
                        )
            elif not has_private and targets <= active:
                next_active[True] = max(
                    next_active.get(True, -1),
                    active - invalidations + creates + 1,
                )
            elif has_private:
                if targets + 1 <= active:
                    next_active[True] = max(
                        next_active.get(True, -1),
                        active - invalidations + creates,
                    )
        if not next_active:
            return None
        feasible_active = next_active
        expected_record_count += creates + int(replaces_private)
    return expected_record_count


def _validate_metrics_and_budget(
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    calls: _algorithm.AlgorithmCallsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    metrics: _algorithm.AlgorithmMetricsComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    measured = metrics.metrics
    totals = _metric_totals(calls)
    expected_limits = _budget_limits(
        run.cycle_reservation,
        len(trajectory.records),
        run.call_policy.max_call_latency_us,
    )
    final_budget = metrics.final_budget_snapshot
    settlements = tuple(item.budget_settlement for item in cycles.cycles)
    expected_consumed = _sum_budgets(settlements)
    expected_interventions = sum(
        item.intervention_action is InterventionAction.REMIND for item in cycles.cycles
    )
    expected_rejections = sum(
        item.intervention_action is InterventionAction.SILENCE
        and item.reason_code is not ReasonCode.SILENCE_SELECTED
        for item in cycles.cycles
    )
    expected_mutations = sum(
        boundary.observation.observed.memory_mutation_count for boundary in attestations.boundaries
    )
    expected_violations = sum(
        boundary.observation.condition_violation for boundary in attestations.boundaries
    )
    expected_memory_cursor = cycles.cycles[-1].boundary_event_sequence if cycles.cycles else 0
    expected_record_count = _replayed_memory_record_count(cycles)
    if expected_record_count is None:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if (
        run.budget_limits != expected_limits
        or final_budget.limits != expected_limits
        or final_budget.reserved != BudgetAmounts()
        or final_budget.consumed != expected_consumed
        or measured.model_call_count != len(calls.calls)
        or measured.provider_input_tokens != totals["provider_input"]
        or measured.provider_output_tokens != totals["provider_output"]
        or measured.canonical_input_tokens != totals["canonical_input"]
        or measured.canonical_output_tokens != totals["canonical_output"]
        or measured.canonical_token_equivalents != totals["canonical_total"]
        or measured.memory_call_latency_us
        != sum(
            item.usage.latency_us
            for item in calls.calls
            if item.phase is StructuredCallPhase.MEMORY_EDIT
        )
        or measured.intervention_count != expected_interventions
        or measured.intervention_count != len(deliveries.deliveries)
        or measured.grounding_rejection_count != expected_rejections
        or measured.provenance_validated_boundary_count != len(cycles.cycles)
        or measured.memory_mutation_count != expected_mutations
        or measured.condition_violation_count != expected_violations
        or metrics.final_memory.ledger_position != attestations.ledger_entry_count
        or metrics.final_memory.ingestion_cursor != len(trajectory.records)
        or metrics.final_memory.memory_cursor != expected_memory_cursor
        or metrics.final_memory.record_count != expected_record_count
        or metrics.final_memory.projection_digest
        != attestations.repository_projection_digests.overall
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_ledger(
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    outcomes: _algorithm.AlgorithmOutcomesComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    entries = attestations.ledger_entries
    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    previous_chain_tag = None
    for entry in entries:
        if (
            entry.record_tag.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
            or entry.chain_tag.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
            or entry.previous_chain_tag != previous_chain_tag
            or not integrity.verify(
                {
                    "run_id": str(attestations.run_id),
                    "position": entry.position,
                    "record_key": entry.record_key,
                    "record_tag": entry.record_tag,
                    "previous_chain_tag": entry.previous_chain_tag,
                },
                entry.chain_tag,
                domain="saliencegate:ledger-chain:v1",
            )
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        previous_chain_tag = entry.chain_tag
    head = attestations.ledger_head
    if (
        head.chain_tag.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
        or head.projection_tag.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
        or head.head_tag.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
        or previous_chain_tag != head.chain_tag
        or not integrity.verify(
            {
                "run_id": str(head.run_id),
                "entry_count": head.entry_count,
                "chain_tag": head.chain_tag,
                "projection_tag": head.projection_tag,
            },
            head.head_tag,
            domain="saliencegate:ledger-head:v1",
        )
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    cycles_by_sequence = {item.boundary_event_sequence: item for item in cycles.cycles}
    deliveries_by_cycle = {item.cycle_id: item for item in deliveries.deliveries}
    outcomes_by_intervention = {item.intervention_id: item for item in outcomes.outcomes}
    complete_records = {
        **{f"invocation_decision:{item.decision_id}": item for item in decisions.decisions},
        **{f"intervention_outcome:{item.outcome_id}": item for item in outcomes.outcomes},
    }
    if (
        len(cycles_by_sequence) != len(cycles.cycles)
        or len(deliveries_by_cycle) != len(deliveries.deliveries)
        or len(outcomes_by_intervention) != len(outcomes.outcomes)
        or len(complete_records) != len(decisions.decisions) + len(outcomes.outcomes)
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)

    expected_layout: list[tuple[str, str, int | None, str | None, str | None]] = []
    for record, decision in zip(trajectory.records, decisions.decisions, strict=True):
        expected_layout.extend(
            (
                (
                    f"trace_event:{record.event_id}",
                    "trace_event",
                    None,
                    None,
                    record.event_digest,
                ),
                (
                    f"invocation_decision:{decision.decision_id}",
                    "invocation_decision",
                    None,
                    None,
                    canonical_digest(decision),
                ),
            )
        )
        cycle = cycles_by_sequence.get(record.event_sequence)
        if cycle is None:
            continue
        for cycle_revision, cycle_state in enumerate(
            (
                CycleState.PENDING,
                CycleState.RESERVED,
                CycleState.RUNNING,
                CycleState.COMMITTED,
            ),
            start=1,
        ):
            expected_layout.append(
                (
                    f"cycle:{cycle.cycle_id}:{cycle_revision}",
                    "cycle_record",
                    cycle_revision,
                    cycle_state.value,
                    cycle.source_cycle_digest if cycle_revision == 4 else None,
                )
            )
        delivery = deliveries_by_cycle.get(cycle.cycle_id)
        if delivery is not None:
            for delivery_revision, delivery_state in enumerate(
                (
                    DeliveryState.PENDING,
                    DeliveryState.CLAIMED,
                    DeliveryState.ATTEMPTING,
                    DeliveryState.DELIVERED,
                ),
                start=1,
            ):
                expected_layout.append(
                    (
                        f"delivery:{delivery.delivery_id}:{delivery_revision}",
                        "delivery_record",
                        delivery_revision,
                        delivery_state.value,
                        (delivery.source_delivery_digest if delivery_revision == 4 else None),
                    )
                )
        outcome = outcomes_by_intervention.get(cycle.intervention_id)
        if outcome is None:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        expected_layout.append(
            (
                f"intervention_outcome:{outcome.outcome_id}",
                "intervention_outcome",
                None,
                None,
                canonical_digest(outcome),
            )
        )
    if len(expected_layout) != len(entries):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    complete_entry_keys = {
        item.record_key
        for item in entries
        if item.record_type in {"invocation_decision", "intervention_outcome"}
    }
    if complete_entry_keys != set(complete_records):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    for entry, (key, record_type, expected_revision, expected_state, source_digest) in zip(
        entries,
        expected_layout,
        strict=True,
    ):
        complete_record = complete_records.get(key)
        if (
            entry.record_key != key
            or entry.record_type != record_type
            or entry.record_revision != expected_revision
            or entry.record_state != expected_state
            or (source_digest is not None and entry.source_record_digest != source_digest)
            or (
                complete_record is not None
                and entry.record_tag
                != integrity.tag(
                    complete_record,
                    domain="saliencegate:ledger-record:v1",
                )
            )
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    event_entries = tuple(item for item in entries if item.record_type == "trace_event")
    for record, entry in zip(trajectory.records, event_entries, strict=True):
        if (
            record.binding.ledger_position != entry.position
            or record.binding.record_tag != entry.record_tag
            or record.binding.chain_tag != entry.chain_tag
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    for window in trajectory.windows:
        boundary_entry = event_entries[window.boundary_event_sequence - 1]
        if (
            boundary_entry.position != window.boundary_ledger_position
            or boundary_entry.chain_tag != window.boundary_chain_tag
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_raw_projection(
    manifest: _algorithm.AlgorithmArtifactManifest,
    run: _algorithm.AlgorithmRunComponent,
    trajectory: _algorithm.AlgorithmTrajectoryComponent,
    calls: _algorithm.AlgorithmCallsComponent,
    decisions: _algorithm.AlgorithmDecisionsComponent,
    cycles: _algorithm.AlgorithmCyclesComponent,
    deliveries: _algorithm.AlgorithmDeliveriesComponent,
    outcomes: _algorithm.AlgorithmOutcomesComponent,
    metrics: _algorithm.AlgorithmMetricsComponent,
    attestations: _algorithm.AlgorithmAttestationsComponent,
) -> None:
    raw = attestations.raw_synthetic_result
    if raw is None:
        return
    if manifest.classification is not ArtifactClassification.SYNTHETIC_RAW:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    try:
        _projection._validate_source_execution_binding(raw, run.execution)
        expected, _ = _projection._project_algorithm_components(
            raw,
            run.execution,
            ArtifactClassification.SYNTHETIC_RAW,
        )
    except Exception:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    actual: dict[_algorithm.AlgorithmArtifactComponentName, BaseModel] = {
        _algorithm.AlgorithmArtifactComponentName.RUN: run,
        _algorithm.AlgorithmArtifactComponentName.TRAJECTORY: trajectory,
        _algorithm.AlgorithmArtifactComponentName.CALLS: calls,
        _algorithm.AlgorithmArtifactComponentName.DECISIONS: decisions,
        _algorithm.AlgorithmArtifactComponentName.CYCLES: cycles,
        _algorithm.AlgorithmArtifactComponentName.DELIVERIES: deliveries,
        _algorithm.AlgorithmArtifactComponentName.OUTCOMES: outcomes,
        _algorithm.AlgorithmArtifactComponentName.METRICS: metrics,
        _algorithm.AlgorithmArtifactComponentName.ATTESTATIONS: attestations,
    }
    if expected != actual:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_cross_component_invariants(
    manifest: _algorithm.AlgorithmArtifactManifest,
    parsed: Mapping[_algorithm.AlgorithmArtifactComponentName, BaseModel],
) -> tuple[
    _algorithm.AlgorithmRunComponent,
    _algorithm.AlgorithmTrajectoryComponent,
    _algorithm.AlgorithmCallsComponent,
    _algorithm.AlgorithmDecisionsComponent,
    _algorithm.AlgorithmCyclesComponent,
    _algorithm.AlgorithmDeliveriesComponent,
    _algorithm.AlgorithmOutcomesComponent,
    _algorithm.AlgorithmMetricsComponent,
    _algorithm.AlgorithmAttestationsComponent,
]:
    try:
        run = cast(
            _algorithm.AlgorithmRunComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.RUN],
        )
        trajectory = cast(
            _algorithm.AlgorithmTrajectoryComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.TRAJECTORY],
        )
        calls = cast(
            _algorithm.AlgorithmCallsComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.CALLS],
        )
        decisions = cast(
            _algorithm.AlgorithmDecisionsComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.DECISIONS],
        )
        cycles = cast(
            _algorithm.AlgorithmCyclesComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.CYCLES],
        )
        deliveries = cast(
            _algorithm.AlgorithmDeliveriesComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.DELIVERIES],
        )
        outcomes = cast(
            _algorithm.AlgorithmOutcomesComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.OUTCOMES],
        )
        metrics = cast(
            _algorithm.AlgorithmMetricsComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.METRICS],
        )
        attestations = cast(
            _algorithm.AlgorithmAttestationsComponent,
            parsed[_algorithm.AlgorithmArtifactComponentName.ATTESTATIONS],
        )
    except KeyError:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if not _same_run(
        manifest,
        run,
        trajectory,
        calls,
        decisions,
        cycles,
        deliveries,
        outcomes,
        metrics,
        attestations,
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    _validate_manifest_bindings(manifest, run, trajectory, attestations)
    _validate_cardinality(
        manifest,
        trajectory,
        calls,
        decisions,
        cycles,
        deliveries,
        outcomes,
        attestations,
    )
    _validate_execution_binding(run, calls)
    _validate_decisions(run, trajectory, decisions, cycles)
    _validate_calls_and_cycles(run, trajectory, calls, decisions, cycles, attestations)
    _validate_deliveries_and_outcomes(
        run,
        trajectory,
        cycles,
        deliveries,
        outcomes,
    )
    _validate_metrics_and_budget(
        run,
        trajectory,
        calls,
        cycles,
        deliveries,
        metrics,
        attestations,
    )
    _validate_ledger(trajectory, decisions, cycles, deliveries, outcomes, attestations)
    _validate_raw_projection(
        manifest,
        run,
        trajectory,
        calls,
        decisions,
        cycles,
        deliveries,
        outcomes,
        metrics,
        attestations,
    )
    return (
        run,
        trajectory,
        calls,
        decisions,
        cycles,
        deliveries,
        outcomes,
        metrics,
        attestations,
    )


def _safe_attestations(
    value: _algorithm.AlgorithmAttestationsComponent,
) -> ValidatedAlgorithmAttestations:
    return ValidatedAlgorithmAttestations(
        run_id=value.run_id,
        boundaries=value.boundaries,
        semantic_projection_digests=value.semantic_projection_digests,
        repository_projection_digests=value.repository_projection_digests,
        ledger_entries=value.ledger_entries,
        ledger_entry_count=value.ledger_entry_count,
        ledger_head=value.ledger_head,
        rebuild_equivalent=value.rebuild_equivalent,
        source_result_digest=value.source_result_digest,
        attestations_component_digest=value.attestations_component_digest,
    )


def _load_algorithm_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None,
) -> ValidatedAlgorithmArtifact:
    if isinstance(manifest_path, bytes):
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    try:
        path = Path(os.fspath(manifest_path))
    except (TypeError, ValueError, OSError):
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    if path.name != "manifest.json":
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    if expected_manifest_digest is not None and (
        type(expected_manifest_digest) is not str
        or _SHA256.fullmatch(expected_manifest_digest) is None
    ):
        _raise(ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)

    descriptors: dict[
        _algorithm.AlgorithmArtifactComponentName,
        _algorithm.AlgorithmArtifactComponent,
    ] = {}

    def parse_manifest(
        data: bytes,
    ) -> artifact_tree.ClosedTreeDescriptor[
        _algorithm.AlgorithmArtifactManifest,
        _algorithm.AlgorithmArtifactComponentName,
    ]:
        payload = _decode_canonical_object(data, manifest=True)
        _preflight_manifest(payload)
        try:
            manifest = _algorithm.AlgorithmArtifactManifest.model_validate_json(data)
        except Exception:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if expected_manifest_digest is not None and not hmac.compare_digest(
            manifest.manifest_digest,
            expected_manifest_digest,
        ):
            _raise(ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)
        descriptors.clear()
        descriptors.update((item.name, item) for item in manifest.components)
        return artifact_tree.ClosedTreeDescriptor(
            manifest=manifest,
            manifest_name="manifest.json",
            manifest_digest=manifest.manifest_digest,
            replacement_key=(f"algorithm_run:{manifest.run_id}:{manifest.condition_id.value}"),
            files=tuple(
                artifact_tree.ClosedTreeFileSpec(
                    key=item.name,
                    name=item.path,
                    maximum_bytes=_algorithm.MAX_ALGORITHM_COMPONENT_BYTES,
                )
                for item in sorted(manifest.components, key=lambda value: value.path)
            ),
        )

    def parse_file(
        name: _algorithm.AlgorithmArtifactComponentName,
        data: bytes,
    ) -> BaseModel:
        descriptor = descriptors.get(name)
        if descriptor is None:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if len(data) != descriptor.byte_count or not hmac.compare_digest(
            _algorithm.algorithm_component_content_digest(name, data),
            descriptor.content_digest,
        ):
            _raise(ArtifactValidationCode.CONTENT_MISMATCH)
        return _parse_component(name, data)

    def finish(
        manifest: _algorithm.AlgorithmArtifactManifest,
        parsed: Mapping[_algorithm.AlgorithmArtifactComponentName, BaseModel],
    ) -> ValidatedAlgorithmArtifact:
        (
            run,
            trajectory,
            calls,
            decisions,
            cycles,
            deliveries,
            outcomes,
            metrics,
            attestations,
        ) = _validate_cross_component_invariants(manifest, parsed)
        assurance = (
            AlgorithmSourceResultAssurance.RECOMPUTED_FROM_RAW
            if attestations.raw_synthetic_result is not None
            else AlgorithmSourceResultAssurance.PRODUCER_ATTESTED
        )
        report = AlgorithmArtifactValidationReport(
            expected_digest_matched=(None if expected_manifest_digest is None else True),
            source_result_assurance=assurance,
            confirmatory=manifest.confirmatory,
            manifest_digest=manifest.manifest_digest,
            overall_content_digest=manifest.overall_content_digest,
            component_count=len(manifest.components),
        )
        return ValidatedAlgorithmArtifact(
            report=report,
            manifest=manifest,
            run=run,
            trajectory=trajectory,
            calls=calls,
            decisions=decisions,
            cycles=cycles,
            deliveries=deliveries,
            outcomes=outcomes,
            metrics=metrics,
            attestations=_safe_attestations(attestations),
        )

    try:
        return artifact_tree.read_closed_tree(
            path,
            maximum_manifest_bytes=_MAX_MANIFEST_BYTES,
            parse_manifest=parse_manifest,
            parse_file=parse_file,
            finish=finish,
        ).value
    except artifact_tree.ClosedTreeReadError as error:
        _raise_tree_error(error)


def load_validated_algorithm_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None = None,
) -> ValidatedAlgorithmArtifact:
    """Load one closed algorithm artifact through a value-free validation boundary."""

    failure: ArtifactValidationCode | None = None
    try:
        return _load_algorithm_artifact(
            manifest_path,
            expected_manifest_digest=expected_manifest_digest,
        )
    except ArtifactValidationError as error:
        failure = error.code
    except Exception:
        failure = ArtifactValidationCode.INVALID_MANIFEST
    assert failure is not None
    raise ArtifactValidationError(failure)


def validate_algorithm_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None = None,
) -> AlgorithmArtifactValidationReport:
    """Validate one algorithm artifact and report only claims supported by its evidence."""

    return load_validated_algorithm_artifact(
        manifest_path,
        expected_manifest_digest=expected_manifest_digest,
    ).report


__all__ = [
    "AlgorithmArtifactValidationReport",
    "AlgorithmSourceResultAssurance",
    "ValidatedAlgorithmArtifact",
    "ValidatedAlgorithmAttestations",
    "load_validated_algorithm_artifact",
    "validate_algorithm_artifact",
]
