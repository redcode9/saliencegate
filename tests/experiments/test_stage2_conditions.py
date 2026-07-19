from __future__ import annotations

import json
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import InterventionAction, canonical_json
from saliencegate.experiments.conditions import (
    FORCED_REMINDER_SCHEMA_ID,
    OPTIONAL_INTERVENTION_SCHEMA_ID,
    BankMaintenanceMode,
    CandidateBankMode,
    InterventionRequirement,
    ResolvedStage2Condition,
    SelectionMode,
    Stage2ConditionError,
    Stage2ConditionId,
    Stage2ConditionObservation,
    Stage2ObservedBehavior,
    Stage2PhaseTwoSchema,
    Stage2SharedControls,
    available_stage2_conditions,
    available_stage2_phase_two_schemas,
    resolve_stage2_condition,
    resolve_stage2_phase_two_schema,
)
from saliencegate.intervention import GroundingConfig, resolve_grounding_configuration
from saliencegate.ports.model_calls import StructuredCallPhase
from saliencegate.prompts import PAPER_TWO_PHASE_FORCED_REMINDER_V1, PAPER_TWO_PHASE_V1
from saliencegate.prompts.contracts import PromptBundleIdentity

EXPECTED_DIGESTS = {
    Stage2ConditionId.NO_MEMORY: "0b3fcd4bb9d260b4fe8a560c20daa18c735848d9eb4215ae6b675cbccd9cbb44",
    Stage2ConditionId.FIXED_STEP: (
        "87a37f9c65fa8ce7ff8eb499a073357c5c858cea066be9d1123c7af098bb9dd7"
    ),
    Stage2ConditionId.RETRIEVAL_ALWAYS: (
        "3ba0335d5166d94e059a9091f2d83ca5be25f8470b02a752db574b2065d300da"
    ),
    Stage2ConditionId.ALWAYS_INJECT: (
        "fa21957aa68b1129ccfbee894fd1407872b70b8ec0039dca942035af69c83863"
    ),
}
BOUNDARY_EVENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
INVOCATION_DECISION_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")


def _observed(
    condition_id: Stage2ConditionId,
    *,
    action: InterventionAction | None = None,
    mutations: int = 0,
    deliveries: int = 0,
) -> Stage2ObservedBehavior:
    expected = resolve_stage2_condition(condition_id).expected
    condition = resolve_stage2_condition(condition_id)
    active = condition_id is not Stage2ConditionId.NO_MEMORY
    retrieval = condition_id is Stage2ConditionId.RETRIEVAL_ALWAYS
    if active and action is None:
        action = InterventionAction.SILENCE
    phase_two_schema_digest = None
    if expected.selection_mode is SelectionMode.MODEL_OPTIONAL:
        phase_two_schema_digest = condition.shared_controls.optional_phase_two_schema_digest
    elif expected.selection_mode is SelectionMode.MODEL_REQUIRED:
        phase_two_schema_digest = condition.shared_controls.forced_phase_two_schema_digest
    return Stage2ObservedBehavior(
        run_id=RUN_ID,
        invocation_decision_id=INVOCATION_DECISION_ID,
        invocation_decision_digest="1" * 64,
        boundary_event_id=BOUNDARY_EVENT_ID,
        boundary_event_sequence=1,
        invocation_ordinal=1,
        schedule_digest="2" * 64,
        window_digest="3" * 64,
        cycle_id="c" * 64 if active else None,
        call_phases=expected.call_phases,
        call_receipt_digests=tuple(
            character * 64 for character in ("a", "b")[: len(expected.call_phases)]
        ),
        candidate_bank_mode=expected.candidate_bank_mode,
        current_bank_view_digest="8" * 64 if active else None,
        candidate_bank_view_digest="d" * 64 if active else None,
        materialization_digest="e" * 64 if active else None,
        bank_maintenance_mode=expected.bank_maintenance_mode,
        selection_mode=expected.selection_mode,
        phase_two_schema_digest=phase_two_schema_digest,
        retrieval_request_digest="6" * 64 if retrieval else None,
        retrieval_result_digest="7" * 64 if retrieval else None,
        memory_mutation_count=mutations,
        intervention_action=action,
        intervention_digest="f" * 64 if active else None,
        delivery_record_count=deliveries,
        delivery_record_digests=("9" * 64,) if deliveries else (),
    )


def _condition_observation(
    condition_id: Stage2ConditionId,
    observed: Stage2ObservedBehavior,
) -> Stage2ConditionObservation:
    condition = resolve_stage2_condition(condition_id)
    violation = (
        condition.expected.intervention_requirement
        is InterventionRequirement.REQUIRED_WITH_SAFE_SILENCE
        and observed.intervention_action is InterventionAction.SILENCE
    )
    return Stage2ConditionObservation(
        condition_id=condition.condition_id,
        condition_digest=condition.condition_digest,
        expected=condition.expected,
        observed=observed,
        condition_violation=violation,
    )


def _json_payload(model: ResolvedStage2Condition | Stage2ConditionObservation) -> dict[str, object]:
    value = json.loads(model.model_dump_json(warnings=False))
    assert type(value) is dict
    return cast(dict[str, object], value)


def test_registry_is_closed_ordered_detached_and_content_addressed() -> None:
    first = available_stage2_conditions()
    second = available_stage2_conditions()

    assert tuple(item.condition_id for item in first) == tuple(Stage2ConditionId)
    assert {item.condition_id: item.condition_digest for item in first} == EXPECTED_DIGESTS
    assert first == second
    assert all(left is not right for left, right in zip(first, second, strict=True))
    assert tuple(resolve_stage2_condition(item.value) for item in Stage2ConditionId) == first


@pytest.mark.parametrize("value", ["undeclared", "", 1, True, object()])
def test_condition_resolution_rejects_unknown_or_ill_typed_ids_without_echo(value: object) -> None:
    with pytest.raises(Stage2ConditionError) as error:
        resolve_stage2_condition(value)
    assert "undeclared" not in str(error.value)
    assert error.value.__cause__ is None


def test_exact_configuration_exposes_only_the_declared_ablation_dimensions() -> None:
    no_memory, fixed, retrieval, forced = available_stage2_conditions()

    assert all(
        item.shared_controls == fixed.shared_controls for item in (no_memory, retrieval, forced)
    )
    retrieval_controls = fixed.shared_controls.retrieval
    assert retrieval_controls.configuration_digest == (
        "ecd22fb340646cba2c45f6759a81d1f976fcba81e47a4486713408f5918e8f98"
    )
    assert retrieval_controls.ranker_version == "ascii-token-overlap/v1"
    assert tuple(
        (item.memory_kind.value, item.claim_kind.value)
        for item in retrieval_controls.claim_kind_mapping
    ) == (
        ("knowledge", "environment_fact"),
        ("procedural", "diagnosis"),
        ("private_status", "open_subgoal"),
    )
    prompt_bundle = fixed.shared_controls.reference_prompt_bundle
    assert prompt_bundle == PAPER_TWO_PHASE_V1.identity
    assert prompt_bundle.bundle_id == "paper-two-phase/v1"
    assert prompt_bundle.bundle_digest == (
        "233f60e043d986f127a57a35c7ee2ba1bfcbf7cff4146b63e0aa3f7c55793bf9"
    )
    templates = {item.phase: item for item in prompt_bundle.templates}
    assert templates[StructuredCallPhase.MEMORY_EDIT].template_id == (
        "paper-two-phase/memory-edit-v1"
    )
    assert templates[StructuredCallPhase.MEMORY_EDIT].template_digest == (
        "9c2818e1aef2b3937e650efe15159d6033caeabe2c91d8720d74fcf497491e1c"
    )
    assert templates[StructuredCallPhase.INTERVENTION].template_id == (
        "paper-two-phase/intervention-v1"
    )
    assert templates[StructuredCallPhase.INTERVENTION].template_digest == (
        "74696d911953956a637ba74701129ddb11cb9fdc69453da9a5b1427becc0362a"
    )
    grounding = fixed.shared_controls.grounding
    assert grounding.configuration_digest == (
        "94bf38e6dbc7d53ac6416a14e9a1d5a4da77c2e2571ffc4386d34be519510fc0"
    )
    assert grounding.configuration == {
        "schema_version": "1.0",
        "pipeline_version": "grounding-pipeline/v1",
        "claim_schema_version": "citation-only-claims/v1",
        "max_claims": 2,
        "max_evidence_per_claim": 1,
        "max_pointer_segments": 32,
        "max_pointer_utf8_bytes": 1_024,
        "duplicate_window_events": 0,
        "cooldown_events": 0,
        "ttl_steps": 1,
        "allowed_delivery_targets": ("next_model_call", "pre_action_replan"),
        "rendering": {
            "schema_version": "1.0",
            "renderer_version": "fixed-ascii/v1",
            "token_counter_version": "utf8-bytes-ceil-div-4-v1",
            "max_claims": 2,
            "max_evidence_bytes": 1_024,
            "max_output_bytes": 4_096,
            "max_token_equivalents": 1_024,
            "include_provenance": False,
        },
    }
    assert fixed.shared_controls.requested_delivery_target.value == "next_model_call"
    assert no_memory.expected.call_phases == ()
    assert no_memory.expected.candidate_bank_mode is CandidateBankMode.DISABLED
    assert no_memory.expected.bank_maintenance_mode is BankMaintenanceMode.DISABLED
    assert no_memory.expected.selection_mode is SelectionMode.DISABLED
    assert no_memory.expected.max_memory_mutations == 0
    assert no_memory.expected.max_delivery_records == 0

    assert fixed.expected.call_phases == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.INTERVENTION,
    )
    assert retrieval.expected.call_phases == (StructuredCallPhase.MEMORY_EDIT,)
    assert forced.expected.call_phases == fixed.expected.call_phases
    for item in (fixed, retrieval, forced):
        assert item.expected.candidate_bank_mode is CandidateBankMode.FULL_ACTIVE_POST_DELTA
        assert item.expected.bank_maintenance_mode is BankMaintenanceMode.MODEL_PHASE_ONE
        assert item.expected.max_memory_mutations == 64
        assert item.expected.max_delivery_records == 1

    assert fixed.expected.selection_mode is SelectionMode.MODEL_OPTIONAL
    assert fixed.expected.intervention_requirement is InterventionRequirement.OPTIONAL
    assert fixed.expected.phase_two_schema_id == OPTIONAL_INTERVENTION_SCHEMA_ID
    assert retrieval.expected.selection_mode is SelectionMode.LEXICAL_TOP_K
    assert retrieval.expected.intervention_requirement is InterventionRequirement.OPTIONAL
    assert retrieval.expected.phase_two_schema_id is None
    assert forced.expected.selection_mode is SelectionMode.MODEL_REQUIRED
    assert (
        forced.expected.intervention_requirement
        is InterventionRequirement.REQUIRED_WITH_SAFE_SILENCE
    )
    assert forced.expected.phase_two_schema_id == FORCED_REMINDER_SCHEMA_ID
    assert forced.expected.safe_silence_is_condition_violation is True


def test_provider_schemas_are_distinct_reviewed_and_forced_schema_disallows_silence() -> None:
    optional, forced = available_stage2_phase_two_schemas()

    assert optional.schema_id == OPTIONAL_INTERVENTION_SCHEMA_ID
    assert forced.schema_id == FORCED_REMINDER_SCHEMA_ID
    assert optional.schema_digest != forced.schema_digest
    assert resolve_stage2_phase_two_schema(optional.schema_id) == optional
    assert resolve_stage2_phase_two_schema(forced.schema_id) == forced
    assert (
        optional.output_schema_version == forced.output_schema_version == "intervention-output/v1"
    )
    optional_response = PAPER_TWO_PHASE_V1.intervention_template.response_format.json_schema
    forced_response = (
        PAPER_TWO_PHASE_FORCED_REMINDER_V1.intervention_template.response_format.json_schema
    )
    assert optional.response_format_name == optional_response.name
    assert forced.response_format_name == forced_response.name
    assert canonical_json(optional.json_schema) == canonical_json(optional_response.schema_value)
    assert canonical_json(forced.json_schema) == canonical_json(forced_response.schema_value)

    optional_properties = optional.json_schema["properties"]
    forced_properties = forced.json_schema["properties"]
    assert isinstance(optional_properties, dict | type(optional.json_schema))
    assert isinstance(forced_properties, dict | type(forced.json_schema))
    optional_action = optional_properties["action"]
    forced_action = forced_properties["action"]
    forced_claims = forced_properties["claims"]
    assert isinstance(optional_action, dict | type(optional.json_schema))
    assert isinstance(forced_action, dict | type(forced.json_schema))
    assert isinstance(forced_claims, dict | type(forced.json_schema))
    assert optional_action.get("$ref") == "#/$defs/InterventionAction"
    assert forced_action.get("const") == "remind"
    assert forced_claims.get("minItems") == 1
    assert forced_claims.get("maxItems") == 2


@pytest.mark.parametrize("value", ["unknown-schema", "", 1, object()])
def test_phase_two_schema_resolution_rejects_unknown_or_ill_typed_ids(value: object) -> None:
    with pytest.raises(Stage2ConditionError):
        resolve_stage2_phase_two_schema(value)


def test_phase_two_schema_rejects_unknown_identity_or_modified_document() -> None:
    schema = resolve_stage2_phase_two_schema(OPTIONAL_INTERVENTION_SCHEMA_ID)
    payload = json.loads(schema.model_dump_json(warnings=False))
    assert type(payload) is dict

    unknown = dict(payload)
    unknown["schema_id"] = "unknown-schema/v1"
    unknown.pop("schema_digest")
    with pytest.raises(ValidationError):
        Stage2PhaseTwoSchema.model_validate_json(json.dumps(unknown))

    modified = dict(payload)
    document = dict(cast(dict[str, object], modified["json_schema"]))
    document["title"] = "ModifiedSelection"
    modified["json_schema"] = document
    modified.pop("schema_digest")
    with pytest.raises(ValidationError):
        Stage2PhaseTwoSchema.model_validate_json(json.dumps(modified))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("reference_prompt_bundle", "bundle_digest"), "0" * 64),
        (("reference_prompt_bundle", "templates", 0, "template_digest"), "0" * 64),
        (("grounding", "configuration", "duplicate_window_events"), 1),
        (("grounding", "configuration", "rendering", "max_output_bytes"), 2_048),
        (("requested_delivery_target",), "pre_action_replan"),
    ],
)
def test_shared_controls_reject_a_modified_non_retrieval_control(
    path: tuple[str | int, ...], replacement: object
) -> None:
    condition = resolve_stage2_condition(Stage2ConditionId.FIXED_STEP)
    payload = json.loads(condition.shared_controls.model_dump_json(warnings=False))
    assert type(payload) is dict
    target: object = payload
    for segment in path[:-1]:
        if type(segment) is int:
            assert type(target) is list
            target = target[segment]
        else:
            assert type(target) is dict
            target = target[segment]
    final = path[-1]
    if type(final) is int:
        assert type(target) is list
        target[final] = replacement
    else:
        assert type(target) is dict
        target[final] = replacement
    with pytest.raises(ValidationError):
        Stage2SharedControls.model_validate_json(json.dumps(payload))


def test_shared_controls_reject_valid_but_unreviewed_prompt_or_grounding_controls() -> None:
    controls = resolve_stage2_condition(Stage2ConditionId.FIXED_STEP).shared_controls
    payload = json.loads(controls.model_dump_json(warnings=False))
    assert type(payload) is dict

    reversed_bundle = PromptBundleIdentity.from_templates(
        PAPER_TWO_PHASE_V1.identity.bundle_id,
        (
            PAPER_TWO_PHASE_V1.intervention_template,
            PAPER_TWO_PHASE_V1.memory_edit_template,
        ),
    )
    changed_prompt = dict(payload)
    changed_prompt["reference_prompt_bundle"] = reversed_bundle.model_dump(
        mode="json", warnings=False
    )
    with pytest.raises(ValidationError):
        Stage2SharedControls.model_validate_json(json.dumps(changed_prompt))

    grounding = GroundingConfig.model_validate_json(
        canonical_json(controls.grounding.configuration)
    )
    changed_grounding = resolve_grounding_configuration(
        grounding.model_copy(update={"duplicate_window_events": 1})
    )
    changed_configuration = dict(payload)
    changed_configuration["grounding"] = changed_grounding.model_dump(mode="json", warnings=False)
    with pytest.raises(ValidationError):
        Stage2SharedControls.model_validate_json(json.dumps(changed_configuration))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), "stage2-condition/v9"),
        (("shared_controls", "retrieval", "top_k"), 1),
        (("shared_controls", "retrieval", "ranker_version"), "semantic-v9"),
        (("expected", "selection_mode"), "lexical_top_k"),
        (("expected", "call_phases"), ["memory_edit"]),
        (("expected", "max_memory_mutations"), 0),
        (("expected", "safe_silence_is_condition_violation"), True),
    ],
)
def test_registry_rejects_arbitrary_flag_combinations_even_with_a_fresh_digest(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = _json_payload(resolve_stage2_condition(Stage2ConditionId.FIXED_STEP))
    payload.pop("condition_digest")
    target = payload
    for segment in path[:-1]:
        nested = target[segment]
        assert type(nested) is dict
        target = cast(dict[str, object], nested)
    target[path[-1]] = replacement

    with pytest.raises(ValidationError):
        ResolvedStage2Condition.model_validate_json(json.dumps(payload))


def test_registry_rejects_a_stale_or_forged_condition_digest() -> None:
    payload = _json_payload(resolve_stage2_condition(Stage2ConditionId.FIXED_STEP))
    payload["condition_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        ResolvedStage2Condition.model_validate_json(json.dumps(payload))


def test_expected_and_observed_projection_accepts_all_declared_success_shapes() -> None:
    cases = (
        (Stage2ConditionId.NO_MEMORY, None, 0, 0, False),
        (Stage2ConditionId.FIXED_STEP, InterventionAction.SILENCE, 2, 0, False),
        (Stage2ConditionId.FIXED_STEP, InterventionAction.REMIND, 2, 1, False),
        (Stage2ConditionId.RETRIEVAL_ALWAYS, InterventionAction.SILENCE, 1, 0, False),
        (Stage2ConditionId.RETRIEVAL_ALWAYS, InterventionAction.REMIND, 1, 1, False),
        (Stage2ConditionId.ALWAYS_INJECT, InterventionAction.REMIND, 3, 1, False),
        (Stage2ConditionId.ALWAYS_INJECT, InterventionAction.SILENCE, 3, 0, True),
    )
    for condition_id, action, mutations, deliveries, violation in cases:
        condition = resolve_stage2_condition(condition_id)
        receipt = _condition_observation(
            condition_id,
            _observed(
                condition_id,
                action=action,
                mutations=mutations,
                deliveries=deliveries,
            ),
        )
        assert receipt.expected == condition.expected
        assert receipt.observed.call_phases == condition.expected.call_phases
        assert receipt.condition_violation is violation


@pytest.mark.parametrize(
    "observed",
    [
        _observed(Stage2ConditionId.FIXED_STEP).model_copy(
            update={
                "call_phases": (StructuredCallPhase.MEMORY_EDIT,),
                "call_receipt_digests": ("a" * 64,),
            }
        ),
        _observed(Stage2ConditionId.FIXED_STEP).model_copy(
            update={"selection_mode": SelectionMode.LEXICAL_TOP_K}
        ),
        _observed(Stage2ConditionId.FIXED_STEP).model_copy(update={"memory_mutation_count": 65}),
        _observed(Stage2ConditionId.FIXED_STEP).model_copy(
            update={
                "delivery_record_count": 1,
                "delivery_record_digests": ("9" * 64,),
            }
        ),
        _observed(
            Stage2ConditionId.FIXED_STEP,
            action=InterventionAction.REMIND,
        ),
    ],
)
def test_projection_rejects_an_execution_that_silently_ran_another_ablation(
    observed: Stage2ObservedBehavior,
) -> None:
    with pytest.raises(ValidationError):
        _condition_observation(Stage2ConditionId.FIXED_STEP, observed)


def test_observation_digest_and_forced_silence_violation_are_not_rewritable() -> None:
    receipt = _condition_observation(
        Stage2ConditionId.ALWAYS_INJECT,
        _observed(
            Stage2ConditionId.ALWAYS_INJECT,
            action=InterventionAction.SILENCE,
        ),
    )
    payload = _json_payload(receipt)

    for field, value in (
        ("condition_violation", False),
        ("condition_digest", "0" * 64),
        ("observation_digest", "0" * 64),
    ):
        tampered = dict(payload)
        tampered[field] = value
        with pytest.raises(ValidationError):
            Stage2ConditionObservation.model_validate_json(json.dumps(tampered))


def test_observed_calls_require_one_unique_receipt_digest_per_visible_call() -> None:
    fixed = _observed(Stage2ConditionId.FIXED_STEP)
    missing = fixed.model_copy(update={"call_receipt_digests": ()})
    duplicated = fixed.model_copy(update={"call_receipt_digests": ("a" * 64, "a" * 64)})

    for observed in (missing, duplicated):
        with pytest.raises(ValidationError):
            _condition_observation(Stage2ConditionId.FIXED_STEP, observed)


def test_no_memory_records_absence_of_intervention_and_cycle_evidence() -> None:
    observed = _observed(Stage2ConditionId.NO_MEMORY)
    receipt = _condition_observation(Stage2ConditionId.NO_MEMORY, observed)

    assert receipt.observed.intervention_action is None
    assert receipt.observed.intervention_digest is None
    assert receipt.observed.cycle_id is None
    assert receipt.observed.call_receipt_digests == ()

    fake_silence = observed.model_copy(
        update={
            "intervention_action": InterventionAction.SILENCE,
            "intervention_digest": "f" * 64,
        }
    )
    with pytest.raises(ValidationError):
        _condition_observation(Stage2ConditionId.NO_MEMORY, fake_silence)


def test_observation_binds_candidate_materialization_and_exact_phase_two_schema() -> None:
    condition = resolve_stage2_condition(Stage2ConditionId.ALWAYS_INJECT)
    observed = _observed(Stage2ConditionId.ALWAYS_INJECT)

    for change in (
        {"current_bank_view_digest": None},
        {"candidate_bank_view_digest": None},
        {"materialization_digest": None},
        {"phase_two_schema_digest": condition.shared_controls.optional_phase_two_schema_digest},
        {"retrieval_result_digest": "7" * 64},
    ):
        with pytest.raises(ValidationError):
            _condition_observation(
                Stage2ConditionId.ALWAYS_INJECT,
                observed.model_copy(update=change),
            )


def test_retrieval_projection_requires_both_selector_digests() -> None:
    observed = _observed(Stage2ConditionId.RETRIEVAL_ALWAYS)

    for field in ("retrieval_request_digest", "retrieval_result_digest"):
        with pytest.raises(ValidationError):
            _condition_observation(
                Stage2ConditionId.RETRIEVAL_ALWAYS,
                observed.model_copy(update={field: None}),
            )
