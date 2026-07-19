from __future__ import annotations

import json
from types import MappingProxyType
from uuid import UUID

import pytest

import saliencegate.shadow.atif as atif_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.shadow.atif import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowEnvironmentBinding,
)
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInvariantError,
    ShadowTraceInputError,
)

RUN_ID = UUID("77777777-7777-4777-8777-777777777777")
ENVIRONMENT_DIGEST = "7" * 64


def _environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory="/synthetic/coverage",
        environment_digest=ENVIRONMENT_DIGEST,
    )


def _adapter() -> ATIFShadowAdapter:
    return ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=_environment(),
    )


def _source(*, timestamp: str | None = None) -> bytes:
    step: dict[str, object] = {
        "step_id": 1,
        "source": "agent",
        "tool_calls": [
            {
                "tool_call_id": "coverage-call",
                "function_name": "exec_command",
                "arguments": {"cmd": "printf coverage"},
            }
        ],
    }
    if timestamp is not None:
        step["timestamp"] = timestamp
    return canonical_json(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "coverage-session",
            "agent": {"name": "codex"},
            "steps": [step],
        }
    )


def _trace():
    return _adapter().adapt_bytes(_source(), run_id=RUN_ID)


@pytest.mark.parametrize(
    "source",
    (
        b'{"a":1} trailing',
        b"{a:1}",
        b'{"a" 1}',
        b'{"a":1 "b":2}',
        b"[1 2]",
        b'{"a":"unterminated}',
        b'{"a":"\x01"}',
        b'{"a":"\\x"}',
        b'{"a":"\\u12"}',
        b'{"a":"\\u12xz"}',
        b'{"a":"\\ud800"}',
        b'{"a":"\\ud800\\u0041"}',
        b'{"a":"\\udc00"}',
        b'{"a":-}',
        b'{"a":01}',
        b'{"a":1.}',
        b'{"a":1e}',
        b'{"a":1e+}',
        b"trux",
        b'{"a":}',
    ),
)
def test_json_preflight_rejects_each_malformed_grammar_branch(source: bytes) -> None:
    with pytest.raises(ShadowTraceInputError) as captured:
        atif_module._parse_source(source)

    assert captured.value.reason_code == "invalid_json"
    assert captured.value.step_ordinal is None


@pytest.mark.parametrize(
    ("limit_name", "limit", "source"),
    (
        ("_MAX_JSON_DEPTH", 1, b'{"a":{}}'),
        ("_MAX_JSON_NODES", 1, b"[1]"),
        ("_MAX_OBJECT_MEMBERS", 1, b'{"a":1,"b":2}'),
        ("_MAX_ARRAY_ITEMS", 1, b"[1,2]"),
        ("_MAX_STRING_BYTES", 1, b'{"a":"xx"}'),
    ),
)
def test_json_preflight_maps_all_allocation_bounds_to_one_public_reason(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    source: bytes,
) -> None:
    monkeypatch.setattr(atif_module, limit_name, limit)

    with pytest.raises(ShadowTraceInputError) as captured:
        atif_module._parse_source(source)

    assert captured.value.reason_code == "input_limit_exceeded"


@pytest.mark.parametrize(
    "source",
    (
        b"\xef\xbb\xbf{}",
        b"\xff",
        b'{"duplicate":1,"duplicate":2}',
        b'{"value":NaN}',
        b"[]",
    ),
)
def test_source_decoder_rejects_non_exact_json_boundaries(source: bytes) -> None:
    with pytest.raises(ShadowTraceInputError) as captured:
        atif_module._parse_source(source)

    assert captured.value.reason_code == "invalid_json"


def test_source_decoder_sanitizes_unexpected_parser_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parser(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private parser detail")

    monkeypatch.setattr(atif_module.json, "loads", fail_parser)

    with pytest.raises(ShadowTraceInputError) as captured:
        atif_module._parse_source(b"{}")

    assert captured.value.reason_code == "invalid_json"
    assert captured.value.__cause__ is not None
    assert "private" not in str(captured.value)


def test_low_level_profile_seals_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ATIFProfile.HARBOR_CODEX_V1

    with monkeypatch.context() as scoped:
        scoped.setattr(
            atif_module,
            "_EXPECTED_PROFILE_DIGESTS",
            MappingProxyType({profile: "0" * 64}),
        )
        with pytest.raises(RuntimeError, match="descriptor seal"):
            atif_module._make_contract(
                profile,
                ("ATIF-v1.7",),
                "codex",
                "exec_command",
                "producer_claimed_structured",
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_freeze_json", lambda _value: {})
        with pytest.raises(RuntimeError, match="descriptor is invalid"):
            atif_module._make_contract(
                profile,
                ("ATIF-v1.7",),
                "codex",
                "exec_command",
                "producer_claimed_structured",
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_EXPECTED_MANIFEST_SHA256", "0" * 64)
        with pytest.raises(ShadowInvariantError):
            atif_module._load_manifest_bytes()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            atif_module.resources,
            "files",
            lambda _package: (_ for _ in ()).throw(RuntimeError("private manifest path")),
        )
        with pytest.raises(ShadowInvariantError):
            atif_module._load_manifest_bytes()


def test_sealed_report_claims_reject_aliases_and_impossible_counts() -> None:
    trace = _trace()
    diagnostics = trace.diagnostics
    contract = atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1]

    assert atif_module._sealed_report_contract(contract.profile.value) is not None
    assert atif_module._sealed_report_contract(object()) is None  # type: ignore[arg-type]
    assert atif_module._sealed_report_contract("harbor-codex/v1-alias") is None
    assert atif_module._matches_sealed_report_claims(
        profile_id=contract.profile.value,
        source_schema_version="ATIF-v1.7",
        timestamp_mode="logical_order",
        capture_scope="selected_events",
        diagnostics=diagnostics,
    )
    assert not atif_module._matches_sealed_report_claims(
        profile_id="harbor-codex/v1-alias",
        source_schema_version="ATIF-v1.7",
        timestamp_mode="logical_order",
        capture_scope="selected_events",
        diagnostics=diagnostics,
    )

    forged = diagnostics.model_copy(
        update={
            "tool_call_disposition_counts": (
                ("mapped_action", 0),
                *diagnostics.tool_call_disposition_counts[1:],
            ),
            "result_disposition_counts": (
                ("mapped_structured_outcome", 1),
                *diagnostics.result_disposition_counts[1:],
            ),
        }
    )
    assert not atif_module._matches_sealed_report_claims(
        profile_id=contract.profile.value,
        source_schema_version="ATIF-v1.7",
        timestamp_mode="logical_order",
        capture_scope="selected_events",
        diagnostics=forged,
    )


def test_trace_and_configuration_seals_reject_every_identity_layer() -> None:
    trace = _trace()
    binding = trace.binding
    diagnostics = trace.diagnostics
    configuration_bytes = trace._configuration_preimage()
    contract = atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1]

    assert atif_module._matches_sealed_trace_contract(
        profile_id=binding.adapter_profile_id,
        descriptor_bytes=trace._descriptor_preimage(),
        profile_digest=binding.adapter_profile_digest,
        manifest_digest=diagnostics.profile_audit_manifest_digest,
        outcome_authority=diagnostics.outcome_evidence_authority,
    )
    for changed in (
        {"profile_id": "harbor-codex/v1-alias"},
        {"descriptor_bytes": bytearray(trace._descriptor_preimage())},
        {"profile_digest": "0" * 64},
        {"manifest_digest": "0" * 64},
        {"outcome_authority": "none"},
    ):
        arguments: dict[str, object] = {
            "profile_id": binding.adapter_profile_id,
            "descriptor_bytes": trace._descriptor_preimage(),
            "profile_digest": binding.adapter_profile_digest,
            "manifest_digest": diagnostics.profile_audit_manifest_digest,
            "outcome_authority": diagnostics.outcome_evidence_authority,
        }
        arguments.update(changed)
        assert not atif_module._matches_sealed_trace_contract(**arguments)  # type: ignore[arg-type]

    assert atif_module._configuration_object(configuration_bytes) is not None
    assert atif_module._configuration_object(bytearray(configuration_bytes)) is None
    assert atif_module._configuration_object(b" " + configuration_bytes) is None
    assert atif_module._configuration_object(b"{") is None
    assert atif_module._matches_sealed_configuration_contract(
        profile_id=binding.adapter_profile_id,
        profile_digest=binding.adapter_profile_digest,
        configuration_bytes=configuration_bytes,
        configuration_digest=binding.adapter_configuration_digest,
        source_schema_version=binding.source_schema_version,
        timestamp_mode=binding.timestamp_mode,
        capture_scope=binding.capture_scope,
        records=trace.records,
        diagnostics=diagnostics,
    )

    parsed = json.loads(configuration_bytes)
    missing_field = dict(parsed)
    missing_field.pop("selection_scope")
    missing_bytes = canonical_json(missing_field)
    missing_digest = length_prefixed_sha256(
        missing_bytes,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )
    assert not atif_module._matches_sealed_configuration_contract(
        profile_id=binding.adapter_profile_id,
        profile_digest=binding.adapter_profile_digest,
        configuration_bytes=missing_bytes,
        configuration_digest=missing_digest,
        source_schema_version=binding.source_schema_version,
        timestamp_mode=binding.timestamp_mode,
        capture_scope=binding.capture_scope,
        records=trace.records,
        diagnostics=diagnostics,
    )

    invalid_environment = json.loads(configuration_bytes)
    invalid_environment["environment"]["environment_digest"] = "not-a-digest"
    invalid_bytes = canonical_json(invalid_environment)
    invalid_digest = length_prefixed_sha256(
        invalid_bytes,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )
    assert not atif_module._matches_sealed_configuration_contract(
        profile_id=contract.profile.value,
        profile_digest=contract.profile_digest,
        configuration_bytes=invalid_bytes,
        configuration_digest=invalid_digest,
        source_schema_version=binding.source_schema_version,
        timestamp_mode=binding.timestamp_mode,
        capture_scope=binding.capture_scope,
        records=trace.records,
        diagnostics=diagnostics,
    )


def test_topology_validator_rejects_forged_records_contexts_and_timestamps() -> None:
    trace = _trace()
    configuration = json.loads(trace._configuration_preimage())
    environment = ShadowEnvironmentBinding.model_validate(configuration["environment"])
    contexts = configuration["mapped_action_contexts"]
    contract = atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1]

    def matches(
        records: tuple[object, ...],
        *,
        selected_contexts: list[object] = contexts,
        timestamp_mode: str = "logical_order",
    ) -> bool:
        return atif_module._records_match_atif_topology(
            contract=contract,
            environment=environment,
            contexts=selected_contexts,
            records=records,
            timestamp_mode=timestamp_mode,
            diagnostics=trace.diagnostics,
        )

    assert matches(trace.records)
    assert not matches(trace.records[:2])
    assert not matches((dict(trace.records[0]), *trace.records[1:]))

    def changed_records(index: int, **changes: object) -> tuple[object, ...]:
        records = [dict(record) for record in trace.records]
        records[index].update(changes)
        return tuple(atif_module._freeze_json(record) for record in records)

    assert not matches(changed_records(0, unexpected=True))
    assert not matches(changed_records(0, source_event_id="wrong-start"))
    assert not matches((trace.records[0], object(), trace.records[-1]))
    assert not matches(changed_records(1, kind="controller_error"))
    assert not matches(changed_records(1, source_event_id="not-an-atif-coordinate"))
    assert not matches(changed_records(1, occurred_at=1))
    assert not matches(trace.records, selected_contexts=[])

    bad_context = dict(contexts[0])
    bad_context["unexpected"] = True
    assert not matches(trace.records, selected_contexts=[bad_context])
    assert not matches(trace.records, timestamp_mode="record_declared")

    source_timestamp_trace = _adapter().adapt_bytes(
        _source(timestamp="2026-07-17T10:00:00Z"),
        run_id=RUN_ID,
    )
    source_configuration = json.loads(source_timestamp_trace._configuration_preimage())
    bad_timestamp_records = [dict(record) for record in source_timestamp_trace.records]
    bad_timestamp_records[0]["occurred_at"] = "2026-07-17T10:00:00+00:00"
    frozen_bad_timestamps = tuple(
        atif_module._freeze_json(record) for record in bad_timestamp_records
    )
    assert not atif_module._records_match_atif_topology(
        contract=contract,
        environment=environment,
        contexts=source_configuration["mapped_action_contexts"],
        records=frozen_bad_timestamps,
        timestamp_mode="source_utc",
        diagnostics=source_timestamp_trace.diagnostics,
    )


def test_adapter_configuration_and_runtime_failures_keep_public_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()

    def invariant_failure() -> bytes:
        raise ShadowInvariantError()

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_load_manifest_bytes", invariant_failure)
        with pytest.raises(ShadowInvariantError):
            ATIFShadowAdapter(
                profile=ATIFProfile.HARBOR_CODEX_V1,
                environment=environment,
            )

    def configuration_failure() -> bytes:
        raise RuntimeError("private configuration")

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_load_manifest_bytes", configuration_failure)
        with pytest.raises(ShadowConfigurationError):
            ATIFShadowAdapter(
                profile=ATIFProfile.HARBOR_CODEX_V1,
                environment=environment,
            )

    configured = _adapter()

    def coordinated_failure(_source: bytes) -> dict[str, object]:
        raise ShadowTraceInputError(
            "invalid_tool_call",
            step_ordinal=2,
            call_ordinal=3,
            result_ordinal=4,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_parse_source", coordinated_failure)
        with pytest.raises(ShadowTraceInputError) as captured:
            configured.adapt_bytes(b"{}", run_id=RUN_ID)
        assert (
            captured.value.reason_code,
            captured.value.step_ordinal,
            captured.value.call_ordinal,
            captured.value.result_ordinal,
        ) == ("invalid_tool_call", 2, 3, 4)
        assert captured.value.__cause__ is None

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_load_manifest_bytes", invariant_failure)
        with pytest.raises(ShadowInvariantError):
            configured.adapt_bytes(b"{}", run_id=RUN_ID)

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_parse_source", configuration_failure)
        with pytest.raises(ShadowTraceInputError) as captured:
            configured.adapt_bytes(b"{}", run_id=RUN_ID)
        assert captured.value.reason_code == "invalid_step"

    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_parse_source", lambda _source: {})
        scoped.setattr(atif_module, "_plan_mapping", lambda *_args, **_kwargs: (object(), "x"))
        scoped.setattr(atif_module, "_build_trace", lambda **_kwargs: None)
        with pytest.raises(ShadowTraceInputError) as captured:
            configured.adapt_bytes(b"{}", run_id=RUN_ID)
        assert captured.value.reason_code == "invalid_step"


def test_environment_and_adapter_string_and_mutation_boundaries() -> None:
    environment = _environment()
    configured = _adapter()

    assert str(environment) == "ShadowEnvironmentBinding(<redacted>)"
    with pytest.raises(AttributeError, match="immutable"):
        configured._profile = ATIFProfile.HARBOR_TERMINUS_2_V1
    with pytest.raises(ShadowTraceInputError) as invalid_run:
        configured.adapt_bytes(b"{}", run_id=UUID(int=0))
    assert invalid_run.value.reason_code == "invalid_step"
    with pytest.raises(ShadowTraceInputError) as invalid_digest:
        configured.adapt_bytes(b"{}", run_id=RUN_ID, task_scope_digest="not-a-digest")
    assert invalid_digest.value.reason_code == "invalid_step"
