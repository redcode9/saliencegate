from __future__ import annotations

import hmac
import json
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal, Never, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from saliencegate.artifacts import tree as artifact_tree
from saliencegate.artifacts.manifest import (
    MAX_ARTIFACT_COMPONENT_BYTES,
    ArtifactAttestationsComponent,
    ArtifactBudgetsComponent,
    ArtifactComponent,
    ArtifactComponentName,
    ArtifactCounters,
    ArtifactDecisionsComponent,
    ArtifactDeliveriesComponent,
    ArtifactManifest,
    ArtifactOutcomesComponent,
    ArtifactRunComponent,
    ArtifactSyntheticComponent,
    InterventionAttestation,
    component_content_digest,
    expected_component_path,
)
from saliencegate.domain import (
    CycleState,
    DeliveryState,
    InterventionAction,
    canonical_digest,
    cycle_id,
    length_prefixed_sha256,
)
from saliencegate.domain.serde import canonical_json

_MAX_MANIFEST_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_COMPONENT_VERSION = re.compile(r"^artifact-[a-z]+/v(0|[1-9][0-9]*)$")


class ArtifactValidationCode(StrEnum):
    INVALID_MANIFEST = "invalid_manifest"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNSAFE_PATH = "unsafe_path"
    MISSING_COMPONENT = "missing_component"
    UNSAFE_COMPONENT = "unsafe_component"
    CONTENT_MISMATCH = "content_mismatch"
    INVALID_COMPONENT = "invalid_component"
    INCONSISTENT_COUNTERS = "inconsistent_counters"
    UNGROUNDED_DELIVERY = "ungrounded_delivery"
    CROSS_COMPONENT_INVARIANT = "cross_component_invariant"
    EXPECTED_DIGEST_MISMATCH = "expected_digest_mismatch"
    CONFIRMATORY_INELIGIBLE = "confirmatory_ineligible"


_ERROR_MESSAGES: dict[ArtifactValidationCode, str] = {
    ArtifactValidationCode.INVALID_MANIFEST: "artifact manifest failed validation",
    ArtifactValidationCode.UNSUPPORTED_VERSION: "artifact schema version is unsupported",
    ArtifactValidationCode.UNSAFE_PATH: "artifact contains an unsafe component path",
    ArtifactValidationCode.MISSING_COMPONENT: "artifact is missing a required component",
    ArtifactValidationCode.UNSAFE_COMPONENT: "artifact contains an unsafe filesystem entry",
    ArtifactValidationCode.CONTENT_MISMATCH: "artifact component content does not match its digest",
    ArtifactValidationCode.INVALID_COMPONENT: "artifact component failed schema validation",
    ArtifactValidationCode.INCONSISTENT_COUNTERS: "artifact counters are inconsistent",
    ArtifactValidationCode.UNGROUNDED_DELIVERY: (
        "delivered reminder lacks producer grounding attestations"
    ),
    ArtifactValidationCode.CROSS_COMPONENT_INVARIANT: (
        "artifact components violate a cross-component invariant"
    ),
    ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH: (
        "artifact does not match the expected out-of-band digest"
    ),
    ArtifactValidationCode.CONFIRMATORY_INELIGIBLE: (
        "artifact is not eligible for confirmatory evidence"
    ),
}


class ArtifactValidationError(ValueError):
    def __init__(self, code: ArtifactValidationCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


class ArtifactValidationReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    schema_version: Literal["artifact-validation-report/v1"] = "artifact-validation-report/v1"
    valid: Literal[True] = True
    integrity_valid: Literal[True] = True
    structurally_valid: Literal[True] = True
    expected_digest_matched: bool | None
    grounding_assurance: Literal["producer_attested_digest_only"] = "producer_attested_digest_only"
    confirmatory: bool
    manifest_digest: str
    overall_content_digest: str
    component_count: int


class ValidatedArtifact(BaseModel):
    """A fully validated, immutable view of the inspectable artifact components.

    The optional raw synthetic component is deliberately not exposed.  Every field
    in this view was parsed, schema-validated, digest-checked, and cross-checked in
    the same filesystem pass that produced ``report``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["validated-artifact/v1"] = "validated-artifact/v1"
    report: ArtifactValidationReport
    manifest: ArtifactManifest
    run: ArtifactRunComponent
    decisions: ArtifactDecisionsComponent
    budgets: ArtifactBudgetsComponent
    deliveries: ArtifactDeliveriesComponent
    outcomes: ArtifactOutcomesComponent
    attestations: ArtifactAttestationsComponent


def _raise(code: ArtifactValidationCode) -> Never:
    raise ArtifactValidationError(code)


_TREE_ERROR_CODES: dict[
    artifact_tree.ClosedTreeReadErrorKind,
    ArtifactValidationCode,
] = {
    artifact_tree.ClosedTreeReadErrorKind.UNSAFE_PATH: ArtifactValidationCode.UNSAFE_PATH,
    artifact_tree.ClosedTreeReadErrorKind.MISSING_ENTRY: ArtifactValidationCode.MISSING_COMPONENT,
    artifact_tree.ClosedTreeReadErrorKind.UNSAFE_ENTRY: ArtifactValidationCode.UNSAFE_COMPONENT,
    artifact_tree.ClosedTreeReadErrorKind.INVALID_DESCRIPTOR: (
        ArtifactValidationCode.INVALID_MANIFEST
    ),
}


def _raise_tree_error(error: artifact_tree.ClosedTreeReadError) -> Never:
    _raise(_TREE_ERROR_CODES[error.kind])


def _read_regular_file(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
    missing_code: ArtifactValidationCode,
) -> object:
    """Compatibility adapter for the former replay-specific reader."""

    missing_kind = artifact_tree.ClosedTreeReadErrorKind.MISSING_ENTRY
    try:
        return artifact_tree._read_regular_file(
            directory_fd,
            name,
            maximum=maximum,
            missing_kind=missing_kind,
        )
    except artifact_tree.ClosedTreeReadError as error:
        if error.kind is missing_kind:
            _raise(missing_code)
        _raise_tree_error(error)


def _reject_nonfinite(value: str) -> Never:
    del value
    _raise(ArtifactValidationCode.INVALID_COMPONENT)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _raise(ArtifactValidationCode.INVALID_COMPONENT)
        result[key] = value
    return result


def _decode_canonical_object(
    data: bytes,
    *,
    manifest: bool,
) -> dict[str, object]:
    code = (
        ArtifactValidationCode.INVALID_MANIFEST
        if manifest
        else ArtifactValidationCode.INVALID_COMPONENT
    )
    try:
        text = data.decode("utf-8")
        parsed = json.loads(
            text,
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except ArtifactValidationError:
        if manifest:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _raise(code)
    if not isinstance(parsed, dict):
        _raise(code)
    try:
        if canonical_json(parsed) != data:
            _raise(code)
    except ArtifactValidationError:
        raise
    except Exception:
        _raise(code)
    return parsed


def _preflight_manifest(payload: dict[str, object]) -> None:
    version = payload.get("schema_version")
    if not isinstance(version, str):
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    match = _MANIFEST_VERSION.fullmatch(version)
    if match is None:
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    if match.group(1) != "1" or version != "1.0":
        _raise(ArtifactValidationCode.UNSUPPORTED_VERSION)
    components = payload.get("components")
    if not isinstance(components, list):
        _raise(ArtifactValidationCode.INVALID_MANIFEST)
    for component in components:
        if not isinstance(component, dict):
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        raw_name = component.get("name")
        raw_path = component.get("path")
        if not isinstance(raw_name, str):
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        try:
            name = ArtifactComponentName(raw_name)
        except (TypeError, ValueError):
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if not isinstance(raw_path, str) or raw_path != expected_component_path(name):
            _raise(ArtifactValidationCode.UNSAFE_PATH)


_COMPONENT_MODELS: dict[ArtifactComponentName, type[BaseModel]] = {
    ArtifactComponentName.ATTESTATIONS: ArtifactAttestationsComponent,
    ArtifactComponentName.BUDGETS: ArtifactBudgetsComponent,
    ArtifactComponentName.DECISIONS: ArtifactDecisionsComponent,
    ArtifactComponentName.DELIVERIES: ArtifactDeliveriesComponent,
    ArtifactComponentName.OUTCOMES: ArtifactOutcomesComponent,
    ArtifactComponentName.RUN: ArtifactRunComponent,
    ArtifactComponentName.SYNTHETIC: ArtifactSyntheticComponent,
}
_COMPONENT_SCHEMA_VERSIONS: dict[ArtifactComponentName, str] = {
    name: f"artifact-{name.value}/v1" for name in ArtifactComponentName
}


def _parse_component(name: ArtifactComponentName, data: bytes) -> BaseModel:
    payload = _decode_canonical_object(data, manifest=False)
    version = payload.get("schema_version")
    if isinstance(version, str):
        match = _COMPONENT_VERSION.fullmatch(version)
        if match is not None and match.group(1) != "1":
            _raise(ArtifactValidationCode.UNSUPPORTED_VERSION)
    if version != _COMPONENT_SCHEMA_VERSIONS[name]:
        _raise(ArtifactValidationCode.INVALID_COMPONENT)
    try:
        return _COMPONENT_MODELS[name].model_validate_json(data)
    except Exception:
        _raise(ArtifactValidationCode.INVALID_COMPONENT)


def _same_run(
    manifest: ArtifactManifest,
    run: ArtifactRunComponent,
    decisions: ArtifactDecisionsComponent,
    budgets: ArtifactBudgetsComponent,
    deliveries: ArtifactDeliveriesComponent,
    outcomes: ArtifactOutcomesComponent,
    attestations: ArtifactAttestationsComponent,
) -> bool:
    return all(
        run_id == manifest.run_id
        for run_id in (
            run.run_id,
            decisions.run_id,
            budgets.run_id,
            deliveries.run_id,
            outcomes.run_id,
            attestations.run_id,
        )
    )


def _replay_semantic_uuid(trace_digest: str, label: str, *parts: object) -> UUID:
    digest = length_prefixed_sha256(
        trace_digest,
        label,
        *(str(part) for part in parts),
        domain="saliencegate:replay-engine:identity:v1",
    )
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _validate_cross_component_invariants(
    manifest: ArtifactManifest,
    parsed: dict[ArtifactComponentName, BaseModel],
) -> None:
    try:
        run = parsed[ArtifactComponentName.RUN]
        decisions = parsed[ArtifactComponentName.DECISIONS]
        budgets = parsed[ArtifactComponentName.BUDGETS]
        deliveries = parsed[ArtifactComponentName.DELIVERIES]
        outcomes = parsed[ArtifactComponentName.OUTCOMES]
        attestations = parsed[ArtifactComponentName.ATTESTATIONS]
        assert isinstance(run, ArtifactRunComponent)
        assert isinstance(decisions, ArtifactDecisionsComponent)
        assert isinstance(budgets, ArtifactBudgetsComponent)
        assert isinstance(deliveries, ArtifactDeliveriesComponent)
        assert isinstance(outcomes, ArtifactOutcomesComponent)
        assert isinstance(attestations, ArtifactAttestationsComponent)
    except (AssertionError, KeyError):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if not _same_run(
        manifest,
        run,
        decisions,
        budgets,
        deliveries,
        outcomes,
        attestations,
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if (
        manifest.engine_configuration_digest != run.engine_configuration_digest
        or manifest.trace_digest != run.trace_digest
        or manifest.model_id != run.model_id
        or manifest.replay_id != run.replay_id
        or manifest.prompt_template_digest != run.prompt_template_digest
        or manifest.result_digest != run.source_result_digest
        or run.source_result_digest != attestations.source_result_digest
        or run.routing_digest != attestations.routing_digest
        or run.projection_digests != attestations.projection_digests
        or run.ledger_head != attestations.ledger_head
        or decisions.decisions_digest != attestations.decisions_digest
        or budgets.limits != run.engine_configuration.budget_limits
        or budgets.configured_reservation != run.engine_configuration.reservation
        or budgets.budget_projection_digest != run.projection_digests.budgets.value
        or not run.rebuild_equivalent
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if (
        len(decisions.decisions) != run.trace_event_count
        or len(attestations.normalized_draft_digests) != run.trace_event_count
    ):
        _raise(ArtifactValidationCode.INCONSISTENT_COUNTERS)
    if run.trace_attestation_mode == "adapter_manifest":
        if len(attestations.trace_record_digests) != run.trace_event_count:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        expected_trace_digest = canonical_digest(
            {
                "schema_version": "1.0",
                "run_id": str(manifest.run_id),
                "record_digests": attestations.trace_record_digests,
            }
        )
    else:
        if attestations.trace_record_digests:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        expected_trace_digest = canonical_digest(
            {
                "schema_version": "engine-normalized-execution/v1",
                "normalized_trace_digest": attestations.normalized_trace_digest,
                "routing_digest": attestations.routing_digest,
                "expected_event_ids": tuple(
                    str(event_id) for event_id in attestations.trace_expected_event_ids
                ),
            }
        )
    if run.trace_digest != expected_trace_digest:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    policy_bindings = tuple(
        sorted(
            {
                (decision.policy_version, decision.configuration_digest)
                for decision in decisions.decisions
            }
        )
    )
    if policy_bindings != tuple(
        (binding.policy_version, binding.configuration_digest)
        for binding in run.policy_configurations
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    grounding_bindings = tuple(
        sorted(
            {
                (cycle.grounding_version, cycle.grounding_configuration_digest)
                for cycle in budgets.cycles
            }
        )
    )
    if grounding_bindings != tuple(
        (binding.grounding_version, binding.configuration_digest)
        for binding in run.grounding_configurations
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    cycles_by_decision = {cycle.invocation_decision_id: cycle for cycle in budgets.cycles}
    invoked_ids = {decision.decision_id for decision in decisions.decisions if decision.invoke}
    if set(cycles_by_decision) != invoked_ids or tuple(
        cycle.last_event_sequence for cycle in budgets.cycles
    ) != tuple(sorted(cycle.last_event_sequence for cycle in budgets.cycles)):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    decision_by_id = {decision.decision_id: decision for decision in decisions.decisions}
    model_request_digests: list[str] = []
    committed_interventions: dict[object, InterventionAttestation] = {}
    reminder_cycles: dict[str, InterventionAttestation] = {}
    for cycle in budgets.cycles:
        decision = decision_by_id[cycle.invocation_decision_id]
        if (
            cycle.policy_version != decision.policy_version
            or cycle.configuration_digest != decision.configuration_digest
            or cycle.last_event_sequence != decision.event_sequence
            or cycle.cycle_id
            != cycle_id(
                manifest.run_id,
                cycle.first_event_sequence,
                cycle.last_event_sequence,
                cycle.policy_version,
                cycle.configuration_digest,
                cycle.grounding_version,
                cycle.grounding_configuration_digest,
                cycle.requested_delivery_target,
            )
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        if cycle.model_request_digest is not None:
            model_request_digests.append(cycle.model_request_digest)
        intervention = cycle.intervention
        if cycle.state is CycleState.COMMITTED:
            if intervention is None:
                if any(delivery.cycle_id == cycle.cycle_id for delivery in deliveries.deliveries):
                    _raise(ArtifactValidationCode.UNGROUNDED_DELIVERY)
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
            committed_interventions[intervention.intervention_id] = intervention
            if (
                intervention.grounding_version != cycle.grounding_version
                or intervention.grounding_configuration_digest
                != cycle.grounding_configuration_digest
                or intervention.delivery_target is not cycle.requested_delivery_target
                or intervention.receipt_model_call_digest not in cycle.model_call_digests
            ):
                _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
            if intervention.action is InterventionAction.REMIND:
                reminder_cycles[cycle.cycle_id] = intervention
        elif intervention is not None:
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if tuple(model_request_digests) != attestations.model_request_digests:
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    cycles_by_id = {cycle.cycle_id: cycle for cycle in budgets.cycles}
    if any(
        delivery.cycle_id not in cycles_by_id
        or delivery.event_sequence != cycles_by_id[delivery.cycle_id].last_event_sequence
        for delivery in deliveries.deliveries
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    deliveries_by_cycle = {delivery.cycle_id: delivery for delivery in deliveries.deliveries}
    if len(deliveries_by_cycle) != len(deliveries.deliveries) or set(deliveries_by_cycle) != set(
        reminder_cycles
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    for cycle_identifier, intervention in reminder_cycles.items():
        delivery = deliveries_by_cycle[cycle_identifier]
        if (
            delivery.intervention_id != intervention.intervention_id
            or delivery.rendered_text_digest != intervention.rendered_text_digest
            or delivery.target is not intervention.delivery_target
        ):
            if delivery.state is DeliveryState.DELIVERED:
                _raise(ArtifactValidationCode.UNGROUNDED_DELIVERY)
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    cycles_by_sequence = {cycle.last_event_sequence: cycle for cycle in budgets.cycles}
    deliveries_by_sequence = {
        delivery.event_sequence: delivery for delivery in deliveries.deliveries
    }
    for binding in attestations.routing_bindings:
        routed_cycle = cycles_by_sequence.get(binding.ordinal)
        if (
            routed_cycle is not None
            and binding.target is not routed_cycle.requested_delivery_target
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
        routed_delivery = deliveries_by_sequence.get(binding.ordinal)
        if routed_delivery is not None and (
            binding.target is None
            or binding.target_request_id_digest != routed_delivery.target_request_id_digest
            or binding.adapter_id is None
            or canonical_digest(binding.adapter_id) != routed_delivery.adapter_id_digest
            or binding.adapter_capabilities_digest != routed_delivery.adapter_capabilities_digest
            or binding.target is not routed_delivery.target
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    expected_outcome_interventions = tuple(committed_interventions)
    if tuple(outcome.intervention_id for outcome in outcomes.outcomes) != (
        expected_outcome_interventions
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    if any(
        outcome.outcome_id
        != _replay_semantic_uuid(run.trace_digest, "outcome", outcome.intervention_id)
        or outcome.created_at != committed_interventions[outcome.intervention_id].created_at
        for outcome in outcomes.outcomes
    ):
        _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    calculated = ArtifactCounters(
        events=run.trace_event_count,
        decisions=len(decisions.decisions),
        invoked=sum(decision.invoke for decision in decisions.decisions),
        cycles=len(budgets.cycles),
        model_calls=len(model_request_digests),
        deliveries=len(deliveries.deliveries),
        delivered=sum(
            delivery.state is DeliveryState.DELIVERED for delivery in deliveries.deliveries
        ),
        outcomes=len(outcomes.outcomes),
    )
    if calculated != manifest.counters:
        _raise(ArtifactValidationCode.INCONSISTENT_COUNTERS)
    synthetic = parsed.get(ArtifactComponentName.SYNTHETIC)
    if synthetic is not None:
        assert isinstance(synthetic, ArtifactSyntheticComponent)
        if (
            synthetic.run_id != manifest.run_id
            or synthetic.trace_digest != manifest.trace_digest
            or synthetic.prompt_template_digest != manifest.prompt_template_digest
            or synthetic.model_request_digests != attestations.model_request_digests
            or synthetic.model_call_digests
            != tuple(digest for cycle in budgets.cycles for digest in cycle.model_call_digests)
        ):
            _raise(ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def _validate_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None = None,
    require_confirmatory: bool = False,
) -> ValidatedArtifact:
    """Validate one closed v1 artifact tree without following filesystem links."""

    if isinstance(manifest_path, bytes):
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    try:
        path = Path(os.fspath(manifest_path))
    except (TypeError, ValueError, OSError):
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    if path.name != "manifest.json" or type(require_confirmatory) is not bool:
        _raise(ArtifactValidationCode.UNSAFE_PATH)
    if expected_manifest_digest is not None and (
        type(expected_manifest_digest) is not str
        or _SHA256.fullmatch(expected_manifest_digest) is None
    ):
        _raise(ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)

    components: dict[ArtifactComponentName, ArtifactComponent] = {}

    def parse_manifest(
        data: bytes,
    ) -> artifact_tree.ClosedTreeDescriptor[ArtifactManifest, ArtifactComponentName]:
        raw_manifest = _decode_canonical_object(data, manifest=True)
        _preflight_manifest(raw_manifest)
        try:
            manifest = ArtifactManifest.model_validate_json(data)
        except Exception:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if expected_manifest_digest is not None and not hmac.compare_digest(
            manifest.manifest_digest,
            expected_manifest_digest,
        ):
            _raise(ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)
        if require_confirmatory and not manifest.confirmatory:
            _raise(ArtifactValidationCode.CONFIRMATORY_INELIGIBLE)
        components.clear()
        components.update((component.name, component) for component in manifest.components)
        # Replay byte-count drift remains a CONTENT_MISMATCH, not a filesystem error.
        files = tuple(
            artifact_tree.ClosedTreeFileSpec(
                key=component.name,
                name=component.path,
                maximum_bytes=MAX_ARTIFACT_COMPONENT_BYTES,
            )
            for component in sorted(manifest.components, key=lambda item: item.path)
        )
        return artifact_tree.ClosedTreeDescriptor(
            manifest=manifest,
            manifest_name="manifest.json",
            manifest_digest=manifest.manifest_digest,
            replacement_key=str(manifest.run_id),
            files=files,
        )

    def parse_file(name: ArtifactComponentName, data: bytes) -> BaseModel:
        component = components.get(name)
        if component is None:
            _raise(ArtifactValidationCode.INVALID_MANIFEST)
        if len(data) != component.byte_count or not hmac.compare_digest(
            component_content_digest(data),
            component.content_digest,
        ):
            _raise(ArtifactValidationCode.CONTENT_MISMATCH)
        return _parse_component(name, data)

    def finish(
        manifest: ArtifactManifest,
        parsed: Mapping[ArtifactComponentName, BaseModel],
    ) -> ValidatedArtifact:
        parsed_components = dict(parsed)
        _validate_cross_component_invariants(manifest, parsed_components)
        report = ArtifactValidationReport(
            expected_digest_matched=(None if expected_manifest_digest is None else True),
            confirmatory=manifest.confirmatory,
            manifest_digest=manifest.manifest_digest,
            overall_content_digest=manifest.overall_content_digest,
            component_count=len(manifest.components),
        )
        return ValidatedArtifact(
            report=report,
            manifest=manifest,
            run=cast(ArtifactRunComponent, parsed_components[ArtifactComponentName.RUN]),
            decisions=cast(
                ArtifactDecisionsComponent,
                parsed_components[ArtifactComponentName.DECISIONS],
            ),
            budgets=cast(
                ArtifactBudgetsComponent,
                parsed_components[ArtifactComponentName.BUDGETS],
            ),
            deliveries=cast(
                ArtifactDeliveriesComponent,
                parsed_components[ArtifactComponentName.DELIVERIES],
            ),
            outcomes=cast(
                ArtifactOutcomesComponent,
                parsed_components[ArtifactComponentName.OUTCOMES],
            ),
            attestations=cast(
                ArtifactAttestationsComponent,
                parsed_components[ArtifactComponentName.ATTESTATIONS],
            ),
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


def load_validated_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None = None,
    require_confirmatory: bool = False,
) -> ValidatedArtifact:
    """Load one safe inspectable view through a value-free validation boundary."""

    failure_code: ArtifactValidationCode | None = None
    try:
        return _validate_artifact(
            manifest_path,
            expected_manifest_digest=expected_manifest_digest,
            require_confirmatory=require_confirmatory,
        )
    except ArtifactValidationError as error:
        failure_code = error.code
    except Exception:
        failure_code = ArtifactValidationCode.INVALID_MANIFEST
    assert failure_code is not None
    raise ArtifactValidationError(failure_code)


def validate_artifact(
    manifest_path: os.PathLike[str] | str,
    *,
    expected_manifest_digest: str | None = None,
    require_confirmatory: bool = False,
) -> ArtifactValidationReport:
    """Validate an untrusted artifact tree and return its integrity report."""

    return load_validated_artifact(
        manifest_path,
        expected_manifest_digest=expected_manifest_digest,
        require_confirmatory=require_confirmatory,
    ).report


__all__ = [
    "ArtifactValidationCode",
    "ArtifactValidationError",
    "ArtifactValidationReport",
    "ValidatedArtifact",
    "load_validated_artifact",
    "validate_artifact",
]
