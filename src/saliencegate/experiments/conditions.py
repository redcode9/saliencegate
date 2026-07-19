from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    ClaimKind,
    DeliveryTarget,
    InterventionAction,
    JsonObject,
    MemoryKind,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    PositiveSigned64Offset,
    Sha256Digest,
)
from saliencegate.intervention import (
    CLAIM_SCHEMA_VERSION,
    FIXED_ASCII_RENDERER_VERSION,
    GROUNDING_PIPELINE_VERSION,
    TOKEN_COUNTER_VERSION,
    GroundingConfig,
    RenderingConfig,
    ResolvedGroundingConfiguration,
    resolve_grounding_configuration,
)
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
)
from saliencegate.ports.model_calls import StructuredCallPhase
from saliencegate.prompts.contracts import MAX_PROMPT_PAYLOAD_BYTES, PromptBundleIdentity
from saliencegate.prompts.paper_two_phase_v1 import (
    PAPER_TWO_PHASE_FORCED_REMINDER_V1,
    PAPER_TWO_PHASE_V1,
)
from saliencegate.runtime.message_window import (
    MAX_MESSAGE_WINDOW_CANONICAL_BYTES,
    MAX_MESSAGE_WINDOW_ITEMS,
    MAX_TASK_DESCRIPTION_UTF8_BYTES,
    MESSAGE_WINDOW_VERSION,
)
from saliencegate.runtime.scheduling import FIXED_STEP_SCHEDULE_VERSION

STAGE2_CONDITION_SCHEMA_VERSION: Literal["stage2-condition/v1"] = "stage2-condition/v1"
STAGE2_SHARED_CONTROLS_SCHEMA_VERSION: Literal["stage2-shared-controls/v1"] = (
    "stage2-shared-controls/v1"
)
STAGE2_EXPECTED_BEHAVIOR_SCHEMA_VERSION: Literal["stage2-expected-behavior/v1"] = (
    "stage2-expected-behavior/v1"
)
STAGE2_OBSERVED_BEHAVIOR_SCHEMA_VERSION: Literal["stage2-observed-behavior/v1"] = (
    "stage2-observed-behavior/v1"
)
STAGE2_CONDITION_OBSERVATION_SCHEMA_VERSION: Literal["stage2-condition-observation/v1"] = (
    "stage2-condition-observation/v1"
)
STAGE2_PHASE_TWO_SCHEMA_VERSION: Literal["stage2-phase-two-schema/v1"] = (
    "stage2-phase-two-schema/v1"
)
STAGE2_RETRIEVAL_CONTROLS_SCHEMA_VERSION: Literal["stage2-retrieval-controls/v1"] = (
    "stage2-retrieval-controls/v1"
)

OPTIONAL_INTERVENTION_SCHEMA_ID: Literal["paper-intervention-optional/v1"] = (
    "paper-intervention-optional/v1"
)
FORCED_REMINDER_SCHEMA_ID: Literal["paper-intervention-forced-reminder/v1"] = (
    "paper-intervention-forced-reminder/v1"
)

_CONDITION_DIGEST_DOMAIN = "saliencegate:experiments:stage2-condition:v1"
_OBSERVATION_DIGEST_DOMAIN = "saliencegate:experiments:stage2-condition-observation:v1"
_PHASE_TWO_SCHEMA_DIGEST_DOMAIN = "saliencegate:experiments:stage2-phase-two-schema:v1"
_RETRIEVAL_CONTROLS_DIGEST_DOMAIN = "saliencegate:experiments:stage2-retrieval-controls:v1"
_MAX_SIGNED_64 = (1 << 63) - 1
_QUERY_CONTEXT_UTF8_BYTES = MAX_TASK_DESCRIPTION_UTF8_BYTES + MAX_MESSAGE_WINDOW_CANONICAL_BYTES
_MAX_RETRIEVAL_QUERY_TERMS = 4_096


class Stage2ConditionError(ValueError):
    """A value-free failure at the closed experiment-condition boundary."""

    def __init__(self) -> None:
        super().__init__("offline experiment condition failed validation")


class Stage2ConditionId(StrEnum):
    NO_MEMORY = "no_memory"
    FIXED_STEP = "fixed_step"
    RETRIEVAL_ALWAYS = "retrieval_always"
    ALWAYS_INJECT = "always_inject"


class CandidateBankMode(StrEnum):
    DISABLED = "disabled"
    FULL_ACTIVE_POST_DELTA = "full_active_post_delta"


class BankMaintenanceMode(StrEnum):
    DISABLED = "disabled"
    MODEL_PHASE_ONE = "model_phase_one"


class SelectionMode(StrEnum):
    DISABLED = "disabled"
    MODEL_OPTIONAL = "model_optional"
    LEXICAL_TOP_K = "lexical_top_k"
    MODEL_REQUIRED = "model_required"


class InterventionRequirement(StrEnum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED_WITH_SAFE_SILENCE = "required_with_safe_silence"


class _Stage2Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _phase_two_schema_payload(schema_id: str) -> dict[str, object]:
    if schema_id == OPTIONAL_INTERVENTION_SCHEMA_ID:
        response_schema = PAPER_TWO_PHASE_V1.intervention_template.response_format.json_schema
    elif schema_id == FORCED_REMINDER_SCHEMA_ID:
        response_schema = (
            PAPER_TWO_PHASE_FORCED_REMINDER_V1.intervention_template.response_format.json_schema
        )
    else:
        raise Stage2ConditionError() from None
    value = json.loads(canonical_json(response_schema.schema_value))
    if type(value) is not dict:  # pragma: no cover - the reviewed schema is a root object
        raise TypeError("intervention schema is not an object")
    return {
        "schema_version": STAGE2_PHASE_TWO_SCHEMA_VERSION,
        "schema_id": schema_id,
        "output_schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION,
        "response_format_name": response_schema.name,
        "json_schema": cast(dict[str, object], value),
    }


def _phase_two_schema_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "schema_digest"}
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_PHASE_TWO_SCHEMA_DIGEST_DOMAIN,
    )


class Stage2PhaseTwoSchema(_Stage2Model):
    """A reviewed provider schema used by one closed Phase 2 selector."""

    schema_version: Literal["stage2-phase-two-schema/v1"] = STAGE2_PHASE_TWO_SCHEMA_VERSION
    schema_id: ComponentIdentifier
    output_schema_version: Literal["intervention-output/v1"]
    response_format_name: ComponentIdentifier
    json_schema: JsonObject = Field(repr=False)
    schema_digest: Sha256Digest = Field(default_factory=_phase_two_schema_digest)

    @model_validator(mode="after")
    def document_and_digest_are_reviewed(self) -> Self:
        expected = _phase_two_schema_payload(self.schema_id)
        actual = self.model_dump(mode="json", exclude={"schema_digest"}, warnings=False)
        if actual != expected or self.schema_digest != _phase_two_schema_digest(actual):
            raise ValueError("Phase 2 schema is not a reviewed document")
        return self


def _build_phase_two_schema(schema_id: str) -> Stage2PhaseTwoSchema:
    return Stage2PhaseTwoSchema.model_validate(_phase_two_schema_payload(schema_id))


_OPTIONAL_PHASE_TWO_SCHEMA = _build_phase_two_schema(OPTIONAL_INTERVENTION_SCHEMA_ID)
_FORCED_PHASE_TWO_SCHEMA = _build_phase_two_schema(FORCED_REMINDER_SCHEMA_ID)


class Stage2RetrievalClaimMapping(_Stage2Model):
    memory_kind: MemoryKind
    claim_kind: ClaimKind


def _retrieval_controls_payload() -> dict[str, object]:
    return {
        "schema_version": STAGE2_RETRIEVAL_CONTROLS_SCHEMA_VERSION,
        "retrieval_version": "candidate-bank-ascii-token-top-k/v1",
        "query_version": "task-latest-eight-ascii-tokens/v1",
        "ranker_version": "ascii-token-overlap/v1",
        "top_k": 2,
        "max_query_utf8_bytes": _QUERY_CONTEXT_UTF8_BYTES,
        "max_query_terms": _MAX_RETRIEVAL_QUERY_TERMS,
        "claim_kind_mapping": (
            {
                "memory_kind": MemoryKind.KNOWLEDGE.value,
                "claim_kind": ClaimKind.ENVIRONMENT_FACT.value,
            },
            {
                "memory_kind": MemoryKind.PROCEDURAL.value,
                "claim_kind": ClaimKind.DIAGNOSIS.value,
            },
            {
                "memory_kind": MemoryKind.PRIVATE_STATUS.value,
                "claim_kind": ClaimKind.OPEN_SUBGOAL.value,
            },
        ),
    }


def _retrieval_controls_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "configuration_digest"}
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_RETRIEVAL_CONTROLS_DIGEST_DOMAIN,
    )


class Stage2RetrievalControls(_Stage2Model):
    """The complete deterministic selector held fixed by the condition digest."""

    schema_version: Literal["stage2-retrieval-controls/v1"] = (
        STAGE2_RETRIEVAL_CONTROLS_SCHEMA_VERSION
    )
    retrieval_version: Literal["candidate-bank-ascii-token-top-k/v1"]
    query_version: Literal["task-latest-eight-ascii-tokens/v1"]
    ranker_version: Literal["ascii-token-overlap/v1"]
    top_k: Annotated[int, Field(ge=1, le=2)]
    max_query_utf8_bytes: Annotated[int, Field(ge=1, le=_QUERY_CONTEXT_UTF8_BYTES)]
    max_query_terms: Annotated[int, Field(ge=1, le=_MAX_RETRIEVAL_QUERY_TERMS)]
    claim_kind_mapping: Annotated[
        tuple[Stage2RetrievalClaimMapping, ...], Field(min_length=3, max_length=3)
    ]
    configuration_digest: Sha256Digest = Field(default_factory=_retrieval_controls_digest)

    @model_validator(mode="after")
    def controls_are_the_frozen_stage2_selector(self) -> Self:
        actual = self.model_dump(mode="json", exclude={"configuration_digest"}, warnings=False)
        if canonical_json(actual) != canonical_json(
            _retrieval_controls_payload()
        ) or self.configuration_digest != _retrieval_controls_digest(actual):
            raise ValueError("experiment retrieval controls are not the frozen selector")
        return self


def _build_retrieval_controls() -> Stage2RetrievalControls:
    return Stage2RetrievalControls.model_validate_json(
        canonical_json(_retrieval_controls_payload())
    )


_RETRIEVAL_CONTROLS = _build_retrieval_controls()


def _build_stage2_grounding_configuration() -> ResolvedGroundingConfiguration:
    rendering = RenderingConfig(
        schema_version="1.0",
        renderer_version=FIXED_ASCII_RENDERER_VERSION,
        token_counter_version=TOKEN_COUNTER_VERSION,
        max_claims=2,
        max_evidence_bytes=1_024,
        max_output_bytes=4_096,
        max_token_equivalents=1_024,
        include_provenance=False,
    )
    return resolve_grounding_configuration(
        GroundingConfig(
            schema_version="1.0",
            pipeline_version=GROUNDING_PIPELINE_VERSION,
            claim_schema_version=CLAIM_SCHEMA_VERSION,
            max_claims=2,
            max_evidence_per_claim=1,
            max_pointer_segments=32,
            max_pointer_utf8_bytes=1_024,
            duplicate_window_events=0,
            cooldown_events=0,
            ttl_steps=1,
            allowed_delivery_targets=(
                DeliveryTarget.NEXT_MODEL_CALL,
                DeliveryTarget.PRE_ACTION_REPLAN,
            ),
            rendering=rendering,
        )
    )


_STAGE2_GROUNDING_CONFIGURATION = _build_stage2_grounding_configuration()


class Stage2SharedControls(_Stage2Model):
    """Controls held fixed across every offline comparison condition."""

    schema_version: Literal["stage2-shared-controls/v1"] = STAGE2_SHARED_CONTROLS_SCHEMA_VERSION
    schedule_version: Literal["first-and-every-action-step/v1"]
    window_version: Literal["latest-eight-logical-messages/v1"]
    message_window_limit: Annotated[int, Field(ge=1, le=MAX_MESSAGE_WINDOW_ITEMS)]
    prompt_context_budget_utf8_bytes: Annotated[int, Field(ge=1, le=MAX_PROMPT_PAYLOAD_BYTES)]
    reference_prompt_bundle: PromptBundleIdentity
    candidate_bank_source: Literal["candidate-post-delta-active-bank/v1"]
    phase_one_schema_version: Literal["memory-edit-output/v1"]
    retrieval: Stage2RetrievalControls
    grounding: ResolvedGroundingConfiguration
    requested_delivery_target: DeliveryTarget
    optional_phase_two_schema_digest: Sha256Digest
    forced_phase_two_schema_digest: Sha256Digest

    @model_validator(mode="after")
    def controls_match_the_frozen_comparison(self) -> Self:
        if self.model_dump(mode="json", warnings=False) != _shared_controls_payload():
            raise ValueError("shared controls differ from the frozen offline comparison")
        return self


def _shared_controls_payload() -> dict[str, object]:
    return {
        "schema_version": STAGE2_SHARED_CONTROLS_SCHEMA_VERSION,
        "schedule_version": FIXED_STEP_SCHEDULE_VERSION,
        "window_version": MESSAGE_WINDOW_VERSION,
        "message_window_limit": MAX_MESSAGE_WINDOW_ITEMS,
        "prompt_context_budget_utf8_bytes": MAX_PROMPT_PAYLOAD_BYTES,
        "reference_prompt_bundle": PAPER_TWO_PHASE_V1.identity.model_dump(
            mode="json", warnings=False
        ),
        "candidate_bank_source": "candidate-post-delta-active-bank/v1",
        "phase_one_schema_version": MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        "retrieval": _RETRIEVAL_CONTROLS.model_dump(mode="json", warnings=False),
        "grounding": _STAGE2_GROUNDING_CONFIGURATION.model_dump(mode="json", warnings=False),
        "requested_delivery_target": DeliveryTarget.NEXT_MODEL_CALL.value,
        "optional_phase_two_schema_digest": _OPTIONAL_PHASE_TWO_SCHEMA.schema_digest,
        "forced_phase_two_schema_digest": _FORCED_PHASE_TWO_SCHEMA.schema_digest,
    }


class Stage2ExpectedBehavior(_Stage2Model):
    schema_version: Literal["stage2-expected-behavior/v1"] = STAGE2_EXPECTED_BEHAVIOR_SCHEMA_VERSION
    candidate_bank_mode: CandidateBankMode
    bank_maintenance_mode: BankMaintenanceMode
    selection_mode: SelectionMode
    intervention_requirement: InterventionRequirement
    call_phases: Annotated[tuple[StructuredCallPhase, ...], Field(max_length=2)]
    phase_two_schema_id: ComponentIdentifier | None
    max_memory_mutations: Annotated[int, Field(ge=0, le=MAX_MEMORY_DELTA_ITEMS)]
    max_delivery_records: Annotated[int, Field(ge=0, le=1)]
    safe_silence_is_condition_violation: bool


def _expected_payload(condition_id: Stage2ConditionId) -> dict[str, object]:
    common: dict[str, object] = {
        "schema_version": STAGE2_EXPECTED_BEHAVIOR_SCHEMA_VERSION,
        "candidate_bank_mode": CandidateBankMode.FULL_ACTIVE_POST_DELTA.value,
        "bank_maintenance_mode": BankMaintenanceMode.MODEL_PHASE_ONE.value,
        "max_memory_mutations": MAX_MEMORY_DELTA_ITEMS,
        "max_delivery_records": 1,
        "safe_silence_is_condition_violation": False,
    }
    if condition_id is Stage2ConditionId.NO_MEMORY:
        return common | {
            "candidate_bank_mode": CandidateBankMode.DISABLED.value,
            "bank_maintenance_mode": BankMaintenanceMode.DISABLED.value,
            "selection_mode": SelectionMode.DISABLED.value,
            "intervention_requirement": InterventionRequirement.DISABLED.value,
            "call_phases": (),
            "phase_two_schema_id": None,
            "max_memory_mutations": 0,
            "max_delivery_records": 0,
        }
    if condition_id is Stage2ConditionId.FIXED_STEP:
        return common | {
            "selection_mode": SelectionMode.MODEL_OPTIONAL.value,
            "intervention_requirement": InterventionRequirement.OPTIONAL.value,
            "call_phases": (
                StructuredCallPhase.MEMORY_EDIT.value,
                StructuredCallPhase.INTERVENTION.value,
            ),
            "phase_two_schema_id": OPTIONAL_INTERVENTION_SCHEMA_ID,
        }
    if condition_id is Stage2ConditionId.RETRIEVAL_ALWAYS:
        return common | {
            "selection_mode": SelectionMode.LEXICAL_TOP_K.value,
            "intervention_requirement": InterventionRequirement.OPTIONAL.value,
            "call_phases": (StructuredCallPhase.MEMORY_EDIT.value,),
            "phase_two_schema_id": None,
        }
    return common | {
        "selection_mode": SelectionMode.MODEL_REQUIRED.value,
        "intervention_requirement": InterventionRequirement.REQUIRED_WITH_SAFE_SILENCE.value,
        "call_phases": (
            StructuredCallPhase.MEMORY_EDIT.value,
            StructuredCallPhase.INTERVENTION.value,
        ),
        "phase_two_schema_id": FORCED_REMINDER_SCHEMA_ID,
        "safe_silence_is_condition_violation": True,
    }


def _condition_payload(condition_id: Stage2ConditionId) -> dict[str, object]:
    return {
        "schema_version": STAGE2_CONDITION_SCHEMA_VERSION,
        "condition_id": condition_id.value,
        "shared_controls": _shared_controls_payload(),
        "expected": _expected_payload(condition_id),
    }


def _condition_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "condition_digest"}
    return length_prefixed_sha256(canonical_json(material), domain=_CONDITION_DIGEST_DOMAIN)


class ResolvedStage2Condition(_Stage2Model):
    """One complete row from the closed offline experiment matrix."""

    schema_version: Literal["stage2-condition/v1"] = STAGE2_CONDITION_SCHEMA_VERSION
    condition_id: Stage2ConditionId
    shared_controls: Stage2SharedControls
    expected: Stage2ExpectedBehavior
    condition_digest: Sha256Digest = Field(default_factory=_condition_digest)

    @model_validator(mode="after")
    def row_and_digest_match_the_registry(self) -> Self:
        actual = self.model_dump(mode="json", exclude={"condition_digest"}, warnings=False)
        expected = _condition_payload(self.condition_id)
        if canonical_json(actual) != canonical_json(
            expected
        ) or self.condition_digest != _condition_digest(actual):
            raise ValueError("offline experiment condition is not a registered row")
        return self


def _resolved(condition_id: Stage2ConditionId) -> ResolvedStage2Condition:
    payload = canonical_json(_condition_payload(condition_id))
    return ResolvedStage2Condition.model_validate_json(payload)


_CONDITIONS = tuple(_resolved(condition_id) for condition_id in Stage2ConditionId)


def available_stage2_conditions() -> tuple[ResolvedStage2Condition, ...]:
    return tuple(
        ResolvedStage2Condition.model_validate_json(item.model_dump_json(warnings=False))
        for item in _CONDITIONS
    )


def resolve_stage2_condition(value: object) -> ResolvedStage2Condition:
    try:
        if type(value) is Stage2ConditionId:
            condition_id = value
        elif type(value) is str:
            condition_id = Stage2ConditionId(value)
        else:
            raise TypeError
        selected = next(item for item in _CONDITIONS if item.condition_id is condition_id)
        return ResolvedStage2Condition.model_validate_json(selected.model_dump_json(warnings=False))
    except Exception:
        raise Stage2ConditionError() from None


def available_stage2_phase_two_schemas() -> tuple[Stage2PhaseTwoSchema, ...]:
    return tuple(
        Stage2PhaseTwoSchema.model_validate_json(item.model_dump_json(warnings=False))
        for item in (_OPTIONAL_PHASE_TWO_SCHEMA, _FORCED_PHASE_TWO_SCHEMA)
    )


def resolve_stage2_phase_two_schema(value: object) -> Stage2PhaseTwoSchema:
    try:
        if type(value) is not str:
            raise TypeError
        selected = next(
            item
            for item in (_OPTIONAL_PHASE_TWO_SCHEMA, _FORCED_PHASE_TWO_SCHEMA)
            if item.schema_id == value
        )
        return Stage2PhaseTwoSchema.model_validate_json(selected.model_dump_json(warnings=False))
    except Exception:
        raise Stage2ConditionError() from None


class Stage2ObservedBehavior(_Stage2Model):
    """Compact projection derived from a runner-owned, closed execution set.

    This value is content-addressed but not independently authoritative. The
    offline experiment runner must derive every field from schedule, repository,
    call, selector, intervention, and delivery objects; the run-level validator
    must recompute it.
    """

    schema_version: Literal["stage2-observed-behavior/v1"] = STAGE2_OBSERVED_BEHAVIOR_SCHEMA_VERSION
    run_id: UUID4
    invocation_decision_id: UUID4
    invocation_decision_digest: Sha256Digest
    boundary_event_id: UUID4
    boundary_event_sequence: PositiveSigned64Offset
    invocation_ordinal: PositiveSigned64Offset
    schedule_digest: Sha256Digest
    window_digest: Sha256Digest
    cycle_id: Sha256Digest | None
    call_phases: Annotated[tuple[StructuredCallPhase, ...], Field(max_length=2)]
    call_receipt_digests: Annotated[tuple[Sha256Digest, ...], Field(max_length=2)]
    candidate_bank_mode: CandidateBankMode
    current_bank_view_digest: Sha256Digest | None
    candidate_bank_view_digest: Sha256Digest | None
    materialization_digest: Sha256Digest | None
    bank_maintenance_mode: BankMaintenanceMode
    selection_mode: SelectionMode
    phase_two_schema_digest: Sha256Digest | None
    retrieval_request_digest: Sha256Digest | None
    retrieval_result_digest: Sha256Digest | None
    memory_mutation_count: Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
    intervention_action: InterventionAction | None
    intervention_digest: Sha256Digest | None
    delivery_record_count: Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
    delivery_record_digests: Annotated[tuple[Sha256Digest, ...], Field(max_length=1)]

    @model_validator(mode="after")
    def evidence_cardinalities_are_exact(self) -> Self:
        if (
            len(self.call_receipt_digests) != len(self.call_phases)
            or len(set(self.call_receipt_digests)) != len(self.call_receipt_digests)
            or len(self.delivery_record_digests) != self.delivery_record_count
            or len(set(self.delivery_record_digests)) != len(self.delivery_record_digests)
        ):
            raise ValueError("observed experiment evidence cardinality does not match")
        return self


def _observation_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "observation_digest"}
    return length_prefixed_sha256(canonical_json(material), domain=_OBSERVATION_DIGEST_DOMAIN)


class Stage2ConditionObservation(_Stage2Model):
    """Expected and observed projection at one scheduled comparison boundary.

    Source authority and completeness belong to the offline run-level evidence
    validator; this schema only closes the serialized projection and ablation row.
    """

    schema_version: Literal["stage2-condition-observation/v1"] = (
        STAGE2_CONDITION_OBSERVATION_SCHEMA_VERSION
    )
    condition_id: Stage2ConditionId
    condition_digest: Sha256Digest
    expected: Stage2ExpectedBehavior
    observed: Stage2ObservedBehavior
    condition_violation: bool
    observation_digest: Sha256Digest = Field(default_factory=_observation_digest)

    @model_validator(mode="after")
    def observed_behavior_conforms_to_the_named_ablation(self) -> Self:
        condition = _resolved(self.condition_id)
        expected = condition.expected
        observed = self.observed
        if (
            self.condition_digest != condition.condition_digest
            or self.expected != expected
            or observed.call_phases != expected.call_phases
            or observed.candidate_bank_mode is not expected.candidate_bank_mode
            or observed.bank_maintenance_mode is not expected.bank_maintenance_mode
            or observed.selection_mode is not expected.selection_mode
            or observed.memory_mutation_count > expected.max_memory_mutations
            or observed.delivery_record_count > expected.max_delivery_records
        ):
            raise ValueError("observed behavior does not match the named experiment condition")
        if expected.candidate_bank_mode is CandidateBankMode.DISABLED:
            if any(
                value is not None
                for value in (
                    observed.cycle_id,
                    observed.current_bank_view_digest,
                    observed.candidate_bank_view_digest,
                    observed.materialization_digest,
                    observed.phase_two_schema_digest,
                    observed.retrieval_request_digest,
                    observed.retrieval_result_digest,
                    observed.intervention_action,
                    observed.intervention_digest,
                )
            ):
                raise ValueError("the control condition cannot claim cycle evidence")
        else:
            if any(
                value is None
                for value in (
                    observed.cycle_id,
                    observed.current_bank_view_digest,
                    observed.candidate_bank_view_digest,
                    observed.materialization_digest,
                    observed.intervention_action,
                    observed.intervention_digest,
                )
            ):
                raise ValueError("a memory condition requires exact cycle evidence")
        retrieval_evidence = (
            observed.retrieval_request_digest,
            observed.retrieval_result_digest,
        )
        if expected.selection_mode is SelectionMode.LEXICAL_TOP_K:
            if any(value is None for value in retrieval_evidence):
                raise ValueError("retrieval selection requires exact selector evidence")
        elif any(value is not None for value in retrieval_evidence):
            raise ValueError("only retrieval selection may carry selector evidence")
        expected_phase_two_digest: str | None = None
        if expected.selection_mode is SelectionMode.MODEL_OPTIONAL:
            expected_phase_two_digest = self._optional_schema_digest(condition)
        elif expected.selection_mode is SelectionMode.MODEL_REQUIRED:
            expected_phase_two_digest = self._forced_schema_digest(condition)
        if observed.phase_two_schema_digest != expected_phase_two_digest:
            raise ValueError("observed Phase 2 schema does not match the named condition")
        if observed.intervention_action is InterventionAction.SILENCE:
            if observed.delivery_record_count != 0:
                raise ValueError("safe silence cannot produce a delivery record")
        elif (
            observed.intervention_action is InterventionAction.REMIND
            and observed.delivery_record_count != 1
        ):
            raise ValueError("a reminder requires exactly one delivery record")
        required_silence = (
            expected.intervention_requirement is InterventionRequirement.REQUIRED_WITH_SAFE_SILENCE
            and observed.intervention_action is InterventionAction.SILENCE
        )
        if self.condition_violation is not required_silence:
            raise ValueError("condition violation does not match forced-reminder safe silence")
        actual = self.model_dump(mode="json", exclude={"observation_digest"}, warnings=False)
        if self.observation_digest != _observation_digest(actual):
            raise ValueError("experiment condition observation digest does not match")
        return self

    @staticmethod
    def _optional_schema_digest(condition: ResolvedStage2Condition) -> str:
        return condition.shared_controls.optional_phase_two_schema_digest

    @staticmethod
    def _forced_schema_digest(condition: ResolvedStage2Condition) -> str:
        return condition.shared_controls.forced_phase_two_schema_digest


__all__ = [
    "FORCED_REMINDER_SCHEMA_ID",
    "OPTIONAL_INTERVENTION_SCHEMA_ID",
    "STAGE2_CONDITION_OBSERVATION_SCHEMA_VERSION",
    "STAGE2_CONDITION_SCHEMA_VERSION",
    "STAGE2_EXPECTED_BEHAVIOR_SCHEMA_VERSION",
    "STAGE2_OBSERVED_BEHAVIOR_SCHEMA_VERSION",
    "STAGE2_PHASE_TWO_SCHEMA_VERSION",
    "STAGE2_RETRIEVAL_CONTROLS_SCHEMA_VERSION",
    "STAGE2_SHARED_CONTROLS_SCHEMA_VERSION",
    "BankMaintenanceMode",
    "CandidateBankMode",
    "InterventionRequirement",
    "ResolvedStage2Condition",
    "SelectionMode",
    "Stage2ConditionError",
    "Stage2ConditionId",
    "Stage2ConditionObservation",
    "Stage2ExpectedBehavior",
    "Stage2ObservedBehavior",
    "Stage2PhaseTwoSchema",
    "Stage2RetrievalClaimMapping",
    "Stage2RetrievalControls",
    "Stage2SharedControls",
    "available_stage2_conditions",
    "available_stage2_phase_two_schemas",
    "resolve_stage2_condition",
    "resolve_stage2_phase_two_schema",
]
