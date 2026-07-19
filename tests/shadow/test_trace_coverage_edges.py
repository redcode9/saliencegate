from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.shadow.conftest import TraceEventFactory
from tests.shadow.test_trace import RUN_ID, build_trace, complete_records
from tests.shadow.test_trace_report import (
    _build_report,
    _matching_shadow_report,
)
from tests.shadow.test_trace_report import (
    _trace as build_report_trace,
)

import saliencegate.shadow.atif as atif_module
import saliencegate.shadow.trace as trace_module
import saliencegate.shadow.trace_report as trace_report_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.shadow.errors import ShadowInvariantError, ShadowTraceInputError
from saliencegate.shadow.report import ShadowRunReport
from saliencegate.shadow.trace import (
    ShadowRecordDiagnostics,
    ShadowTrace,
    ShadowTraceBinding,
)
from saliencegate.shadow.trace_report import (
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
    verify_shadow_trace_source,
)


@dataclass(frozen=True)
class _ReportCase:
    source: bytes
    trace: ShadowTrace
    shadow_report: ShadowRunReport
    report: ShadowTraceReport


@pytest.fixture
def report_case(trace_event_factory: TraceEventFactory) -> _ReportCase:
    trace = build_report_trace()
    shadow_report = _matching_shadow_report(trace_event_factory, trace)
    return _ReportCase(
        source=b'{"native":"private-command /private/project"}',
        trace=trace,
        shadow_report=shadow_report,
        report=_build_report(trace, shadow_report),
    )


def _new_trace_arguments(trace: ShadowTrace) -> dict[str, object]:
    return {
        "run_id": trace.run_id,
        "binding": trace.binding,
        "diagnostics": trace.diagnostics,
        "records": trace.records,
        "record_bytes": trace._wire_record_bytes(),
        "adapter_descriptor_bytes": trace._descriptor_preimage(),
        "adapter_configuration_bytes": trace._configuration_preimage(),
    }


def test_exact_json_sizing_covers_utf8_escapes_and_scalar_types() -> None:
    text = "a\u00e9\u20ac\U0001f600"
    assert trace_module._utf8_size(text, limit=None) == 10
    assert trace_module._json_string_size('"\\\n\x00' + text, limit=100) == len(
        canonical_json('"\\\n\x00' + text)
    )

    value: dict[str, object] = {
        "none": None,
        "true": True,
        "false": False,
        "integer": -12,
        "float": 1.25,
        "text": text,
        "array": [1, 2],
    }
    copied, size = trace_module._copy_exact_json(
        value,
        max_bytes=1_024,
        max_depth=10,
        max_container_items=20,
        max_nodes=100,
        max_string_bytes=100,
    )

    assert copied == value
    assert copied is not value
    assert size == len(canonical_json(value))


def test_exact_json_sizing_rejects_surrogates_and_byte_overflow() -> None:
    with pytest.raises(trace_module._ExactJSONValueError):
        trace_module._utf8_size("\ud800", limit=None)
    with pytest.raises(trace_module._ExactJSONValueError):
        trace_module._json_string_size("\udfff", limit=10)
    with pytest.raises(trace_module._ExactJSONLimitError):
        trace_module._utf8_size("\u00e9", limit=1)
    with pytest.raises(trace_module._ExactJSONLimitError):
        trace_module._json_string_size("xx", limit=3)


def _copy_json(value: object, **overrides: object) -> tuple[object, int]:
    options: dict[str, object] = {
        "max_bytes": 100,
        "max_depth": 5,
        "max_container_items": 5,
        "max_nodes": 20,
        "max_string_bytes": 20,
    }
    options.update(overrides)
    return trace_module._copy_exact_json(value, **options)  # type: ignore[arg-type]


def test_exact_json_copy_rejects_cycles_types_and_each_resource_bound() -> None:
    cyclic_dict: dict[str, object] = {}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)

    for value, overrides, error_type in (
        ({}, {"max_bytes": 1}, trace_module._ExactJSONLimitError),
        ({"a": 1, "b": 2}, {"max_container_items": 1}, trace_module._ExactJSONLimitError),
        ({"a": 1}, {"max_depth": 0}, trace_module._ExactJSONLimitError),
        ({"a": 1}, {"max_nodes": 1}, trace_module._ExactJSONLimitError),
        ({"toolong": 1}, {"max_string_bytes": 2}, trace_module._ExactJSONLimitError),
        (cyclic_dict, {}, trace_module._ExactJSONLimitError),
        (cyclic_list, {}, trace_module._ExactJSONLimitError),
        ({1: "bad-key"}, {}, trace_module._ExactJSONValueError),
        (None, {"max_bytes": 3}, trace_module._ExactJSONLimitError),
        (False, {"max_bytes": 4}, trace_module._ExactJSONLimitError),
        (123, {"max_bytes": 2}, trace_module._ExactJSONLimitError),
        (1.25, {"max_bytes": 2}, trace_module._ExactJSONLimitError),
        (math.nan, {}, trace_module._ExactJSONValueError),
        (object(), {}, trace_module._ExactJSONValueError),
    ):
        with pytest.raises(error_type):
            _copy_json(value, **overrides)

    with pytest.raises(trace_module._ExactJSONLimitError):
        _copy_json(10**5_000)


def test_descriptor_copy_rejects_alias_types_and_serializer_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(trace_module._ExactJSONValueError):
        trace_module._copy_descriptor(MappingProxyType({}))

    monkeypatch.setattr(trace_module, "canonical_json", lambda _value: b"x")
    with pytest.raises(trace_module._ExactJSONValueError):
        trace_module._copy_descriptor({})


def test_frozen_json_checker_rejects_mutable_invalid_and_unbounded_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = trace_module._freeze_json({"nested": [1, 2.5, None]})
    copied = trace_module._copy_frozen_json(frozen)
    assert trace_module._frozen_json_is_exact(frozen)
    assert copied == frozen
    assert copied is not frozen

    for invalid in (
        MappingProxyType({1: "bad-key"}),
        MappingProxyType({"value": math.nan}),
        MappingProxyType({"value": object()}),
        (object(),),
    ):
        assert not trace_module._frozen_json_is_exact(invalid)

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "_MAX_RECORD_NODES", 0)
        assert not trace_module._frozen_json_is_exact(frozen)
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "_MAX_RECORD_DEPTH", 0)
        assert not trace_module._frozen_json_is_exact(frozen)
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "_MAX_RECORD_ITEMS", 0)
        assert not trace_module._frozen_json_is_exact(MappingProxyType({"a": 1}))
        assert not trace_module._frozen_json_is_exact((1,))


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        (trace_module._ExactJSONLimitError, "input_limit_exceeded"),
        (trace_module._ExactJSONValueError, "invalid_json"),
        (RuntimeError, "invalid_json"),
    ),
)
def test_wire_snapshot_sanitizes_internal_copy_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    reason: str,
) -> None:
    def fail_copy(*_args: object, **_kwargs: object) -> tuple[object, int]:
        raise failure("private record")

    monkeypatch.setattr(trace_module, "_copy_exact_json", fail_copy)
    with pytest.raises(ShadowTraceInputError) as captured:
        trace_module._snapshot_wire_record({}, ordinal=7, byte_budget=100)

    assert captured.value.reason_code == reason
    assert captured.value.step_ordinal == 7


def test_wire_snapshot_rejects_non_dict_encoding_failure_and_size_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ShadowTraceInputError) as non_dict:
        trace_module._snapshot_wire_record([], ordinal=2, byte_budget=100)
    assert non_dict.value.reason_code == "invalid_step"

    with monkeypatch.context() as scoped:
        scoped.setattr(
            trace_module,
            "canonical_json",
            lambda _value: (_ for _ in ()).throw(RuntimeError("private encoding")),
        )
        with pytest.raises(ShadowTraceInputError) as encoding:
            trace_module._snapshot_wire_record({}, ordinal=3, byte_budget=100)
        assert encoding.value.reason_code == "invalid_json"

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "_copy_exact_json", lambda *_args, **_kwargs: ({}, 99))
        with pytest.raises(ShadowTraceInputError) as mismatch:
            trace_module._snapshot_wire_record({}, ordinal=4, byte_budget=100)
        assert mismatch.value.reason_code == "invalid_json"


def test_direct_factory_rejects_exact_type_aliases_before_consuming_records() -> None:
    class TextAlias(str):
        pass

    class UUIDAlias(UUID):
        pass

    cases: tuple[dict[str, object], ...] = (
        {"run_id": UUIDAlias(str(RUN_ID))},
        {"adapter_profile_id": TextAlias("example-agent/v1")},
        {"source_format": TextAlias("shadow-records")},
        {"source_format": "atif", "source_bytes": b"{}"},
        {"source_schema_version": TextAlias("shadow-input/v1")},
        {"timestamp_mode": TextAlias("record_declared")},
        {"capture_scope": TextAlias("complete_run_declared")},
        {"task_scope_digest": TextAlias("a" * 64)},
        {"source_bytes": bytearray(b"source")},
        {"source_format": "custom"},
        {"source_schema_version": "custom/v1"},
    )

    for overrides in cases:
        consumed = False

        def records():
            nonlocal consumed
            consumed = True
            yield from complete_records()

        with pytest.raises(ShadowTraceInputError):
            build_trace(records(), **overrides)
        assert not consumed


def test_direct_factory_sanitizes_descriptor_and_class_alias_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(
            trace_module,
            "_copy_descriptor",
            lambda _value: (_ for _ in ()).throw(RuntimeError("private descriptor")),
        )
        with pytest.raises(ShadowTraceInputError) as captured:
            build_trace()
        assert captured.value.reason_code == "invalid_step"

    factory = ShadowTrace.from_records.__func__
    with pytest.raises(ShadowTraceInputError) as alias:
        factory(
            object,
            complete_records(),
            run_id=RUN_ID,
            adapter_profile_id="example-agent/v1",
            adapter_descriptor={"schema_version": "example/v1"},
            capture_scope="complete_run_declared",
        )
    assert alias.value.reason_code == "invalid_step"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda trace: object.__setattr__(trace, "schema_version", object()),
        lambda trace: object.__setattr__(trace, "run_id", object()),
        lambda trace: object.__setattr__(trace, "binding", object()),
        lambda trace: object.__setattr__(trace, "diagnostics", object()),
        lambda trace: object.__setattr__(trace, "_records", list(trace.records)),
        lambda trace: object.__setattr__(trace, "_record_bytes", list(trace._wire_record_bytes())),
        lambda trace: object.__setattr__(trace, "_adapter_descriptor_bytes", object()),
        lambda trace: object.__setattr__(trace, "_adapter_configuration_bytes", object()),
        lambda trace: object.__setattr__(trace, "_binding_bytes", object()),
        lambda trace: object.__setattr__(trace, "_diagnostics_bytes", object()),
        lambda trace: object.__setattr__(trace, "_run_id_bytes", object()),
        lambda trace: object.__setattr__(trace, "_run_id_bytes", b"wrong"),
        lambda trace: object.__setattr__(trace, "_binding_bytes", b"{}"),
        lambda trace: object.__setattr__(trace, "_diagnostics_bytes", b"{}"),
        lambda trace: object.__setattr__(
            trace, "_adapter_descriptor_bytes", trace._descriptor_preimage() + b"x"
        ),
        lambda trace: object.__setattr__(
            trace, "_adapter_configuration_bytes", trace._configuration_preimage() + b"x"
        ),
        lambda trace: object.__setattr__(trace, "_record_bytes", trace._wire_record_bytes()[:-1]),
        lambda trace: object.__setattr__(
            trace, "_records", (dict(trace.records[0]), *trace.records[1:])
        ),
        lambda trace: object.__setattr__(
            trace, "_record_bytes", ("not-bytes", *trace._wire_record_bytes()[1:])
        ),
        lambda trace: object.__setattr__(
            trace, "_record_bytes", (b"{}", *trace._wire_record_bytes()[1:])
        ),
    ),
)
def test_exact_trace_rejects_forged_public_private_and_mutable_state(mutation: Any) -> None:
    trace = build_trace()
    mutation(trace)

    assert not trace._is_exact()


def test_trusted_trace_factory_validates_every_preimage_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    baseline = _new_trace_arguments(trace)
    assert trace_module._new_shadow_trace(**baseline)._is_exact()  # type: ignore[arg-type]

    invalid_cases: tuple[dict[str, object], ...] = (
        {"run_id": UUID(int=0)},
        {"binding": object()},
        {"records": list(trace.records)},
        {"adapter_descriptor_bytes": bytearray(trace._descriptor_preimage())},
        {"records": (), "record_bytes": ()},
        {"record_bytes": (bytearray(b"x"), *trace._wire_record_bytes()[1:])},
        {"adapter_descriptor_bytes": b""},
        {"adapter_configuration_bytes": b""},
        {"records": (dict(trace.records[0]), *trace.records[1:])},
        {
            "diagnostics": trace.diagnostics.model_copy(
                update={"mapped_shadow_record_count": len(trace.records) + 1}
            )
        },
        {
            "diagnostics": trace.diagnostics.model_copy(
                update={"source_record_count": len(trace.records) + 1}
            )
        },
        {"binding": trace.binding.model_copy(update={"source_format": "atif"})},
        {"record_bytes": (b"{}", *trace._wire_record_bytes()[1:])},
    )
    for changed in invalid_cases:
        arguments = dict(baseline)
        arguments.update(changed)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            trace_module._new_shadow_trace(**arguments)  # type: ignore[arg-type]

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "MAX_SHADOW_TRACE_BYTES", 1)
        with pytest.raises(ValueError, match="record bytes are too large"):
            trace_module._new_shadow_trace(**baseline)  # type: ignore[arg-type]


def test_binding_and_diagnostics_before_validators_reject_bool_and_alias_values() -> None:
    trace = build_trace()
    binding_body = trace.binding.model_dump(mode="python", warnings=False)
    binding_body["source_byte_count"] = True
    with pytest.raises(ValidationError, match="byte count"):
        ShadowTraceBinding.model_validate(binding_body)

    binding_body = trace.binding.model_dump(mode="python", warnings=False)
    binding_body["source_format"] = 1
    with pytest.raises(ValidationError, match="binding text"):
        ShadowTraceBinding.model_validate(binding_body)

    diagnostics_body = trace.diagnostics.model_dump(mode="python", warnings=False)
    diagnostics_body["source_record_count"] = True
    with pytest.raises(ValidationError, match="diagnostics count"):
        ShadowRecordDiagnostics.model_validate(diagnostics_body)

    diagnostics_body = trace.diagnostics.model_dump(mode="python", warnings=False)
    diagnostics_body["input_kind_counts"] = object()
    with pytest.raises(ValidationError, match="kind counts"):
        ShadowRecordDiagnostics.model_validate(diagnostics_body)


def test_trace_report_primitive_guards_and_sealed_contract_shape(
    monkeypatch: pytest.MonkeyPatch,
    report_case: _ReportCase,
) -> None:
    assert trace_report_module._require_digest("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="digest is invalid"):
        trace_report_module._require_digest(str("A" * 64))
    with pytest.raises(ValueError, match="diagnostics are invalid"):
        trace_report_module._copy_trace_diagnostics(object())

    def fail_json(_value: object) -> bytes:
        raise RuntimeError("private comparison")

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "canonical_json", fail_json)
        assert not trace_report_module._models_match_exactly(
            report_case.trace.binding,
            report_case.trace.binding,
        )

    assert trace_report_module._sealed_atif_report_contract("unknown/v1") is None
    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_sealed_report_contract", lambda _profile: None)
        assert trace_report_module._sealed_atif_report_contract("harbor-codex/v1") is None
    with monkeypatch.context() as scoped:
        scoped.setattr(
            atif_module,
            "_sealed_report_contract",
            lambda _profile: ("bad", "bad", "bad"),
        )
        assert trace_report_module._sealed_atif_report_contract("harbor-codex/v1") is None


@pytest.mark.parametrize(
    ("constant", "limit", "data"),
    (
        ("_MAX_JSON_STRING_BYTES", 1, b'"xx"'),
        ("_MAX_JSON_DEPTH", 1, b"[[0]]"),
        ("_MAX_JSON_CONTAINER_ITEMS", 1, b"[0,1]"),
        ("_MAX_JSON_SCALAR_BYTES", 1, b"123"),
        ("_MAX_JSON_STRUCTURAL_TOKENS", 0, b"{}"),
    ),
)
def test_trace_report_preflight_enforces_each_structural_bound(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    data: bytes,
) -> None:
    monkeypatch.setattr(trace_report_module, constant, limit)
    with pytest.raises(ValueError):
        trace_report_module._preflight_canonical_json_structure(data)


@pytest.mark.parametrize("data", (b" ", b"}", b"[}", b"a" * 129))
def test_trace_report_preflight_rejects_noncanonical_or_unbalanced_tokens(data: bytes) -> None:
    with pytest.raises(ValueError):
        trace_report_module._preflight_canonical_json_structure(data)


def test_decoded_json_shape_enforces_nodes_depth_items_keys_strings_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_MAX_JSON_NODES", 0)
        with pytest.raises(ValueError, match="structure is too large"):
            trace_report_module._bounded_json_shape(None)
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_MAX_JSON_DEPTH", 1)
        with pytest.raises(ValueError, match="structure is too large"):
            trace_report_module._bounded_json_shape([[0]])
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_MAX_JSON_CONTAINER_ITEMS", 0)
        with pytest.raises(ValueError, match="object is too large"):
            trace_report_module._bounded_json_shape({"a": 1})
        with pytest.raises(ValueError, match="array is too large"):
            trace_report_module._bounded_json_shape([1])
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_MAX_JSON_STRING_BYTES", 1)
        with pytest.raises(ValueError, match="key is too large"):
            trace_report_module._bounded_json_shape({"xx": 1})
        with pytest.raises(ValueError, match="string is too large"):
            trace_report_module._bounded_json_shape("xx")
    with pytest.raises(ValueError, match="key is too large"):
        trace_report_module._bounded_json_shape({1: "value"})
    with pytest.raises(ValueError, match="value is invalid"):
        trace_report_module._bounded_json_shape(object())


def test_trace_report_json_hooks_reject_duplicates_and_nonfinite_numbers() -> None:
    assert trace_report_module._finite_json_float("1.25") == 1.25
    assert trace_report_module._unique_json_object([("a", 1)]) == {"a": 1}
    with pytest.raises(ValueError, match="non-finite"):
        trace_report_module._finite_json_float("1e999")
    with pytest.raises(ValueError, match="non-finite"):
        trace_report_module._reject_json_constant("NaN")
    with pytest.raises(ValueError, match="duplicate"):
        trace_report_module._unique_json_object([("a", 1), ("a", 2)])


def test_decoder_rejects_valid_but_noncanonical_key_order(report_case: _ReportCase) -> None:
    body = report_case.report.model_dump(mode="json", warnings=False)
    reordered = dict(reversed(tuple(body.items())))
    encoded = json.dumps(reordered, ensure_ascii=False, separators=(",", ":")).encode()
    assert encoded != canonical_json(body)

    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(encoded)


def test_decoder_and_encoder_check_exact_reconstructed_model_state(
    monkeypatch: pytest.MonkeyPatch,
    report_case: _ReportCase,
) -> None:
    encoded = encode_shadow_trace_report(report_case.report)
    calls = 0
    exact = trace_report_module._model_state_is_exact

    def reject_final_model(model_type: type[object], value: object) -> bool:
        nonlocal calls
        calls += 1
        if calls >= 4:
            return False
        return exact(model_type, value)  # type: ignore[arg-type]

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_model_state_is_exact", reject_final_model)
        with pytest.raises(ValueError, match="model is invalid"):
            trace_report_module._decode_shadow_trace_report(encoded)

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_report_module, "_decode_shadow_trace_report", lambda _data: object())
        with pytest.raises(ShadowInvariantError):
            encode_shadow_trace_report(report_case.report)


@pytest.mark.parametrize(
    "mutation",
    ("record_identity", "record_kind", "first_row", "observation", "retry", "extra_observation"),
)
def test_builder_evidence_links_reject_each_forged_topology(
    report_case: _ReportCase,
    mutation: str,
) -> None:
    trace = report_case.trace
    report = report_case.shadow_report
    if mutation in {"record_identity", "record_kind"}:
        records = list(trace.records)
        changed = dict(records[0])
        if mutation == "record_identity":
            changed["source_event_id"] = 1
        else:
            changed["kind"] = "unknown"
        records[0] = MappingProxyType(changed)
        object.__setattr__(trace, "_records", tuple(records))
    elif mutation == "first_row":
        report.rows[0].__dict__["first_occurrence_ordinal"] = 2
    elif mutation == "observation":
        report.observations[0].__dict__["cli_input_ordinal"] = 99
    elif mutation == "retry":
        report.rows[3].__dict__["retry_target_ordinal"] = 99
    else:
        report.__dict__["observations"] = (*report.observations, report.observations[0])

    with pytest.raises(ValueError):
        trace_report_module._require_builder_evidence_links(trace, report)


def test_copy_adapter_error_rejects_subclasses_and_corrupt_coordinates() -> None:
    class DerivedTraceError(ShadowTraceInputError):
        pass

    copied = trace_report_module._copy_adapter_error(
        DerivedTraceError("invalid_tool_call", step_ordinal=1)
    )
    assert copied.reason_code == "invalid_step"

    corrupt = ShadowTraceInputError("invalid_tool_call", step_ordinal=1)
    corrupt.step_ordinal = 0
    copied_corrupt = trace_report_module._copy_adapter_error(corrupt)
    assert copied_corrupt.reason_code == "invalid_step"


class _Adapter:
    def __init__(
        self,
        trace: object,
        *,
        profile_id: object,
        profile_digest: object,
        failure: BaseException | None = None,
    ) -> None:
        self.trace = trace
        self.profile_id = profile_id
        self.profile_digest = profile_digest
        self.failure = failure

    def adapt_bytes(self, _source: bytes, **_kwargs: object) -> object:
        if self.failure is not None:
            raise self.failure
        return self.trace


def test_source_verifier_preserves_coordinates_and_sanitizes_adapter_failures(
    report_case: _ReportCase,
) -> None:
    binding = report_case.report.binding

    with pytest.raises(ShadowTraceInputError) as empty:
        verify_shadow_trace_source(
            report_case.report,
            b"",
            adapter=_Adapter(
                report_case.trace,
                profile_id=binding.adapter_profile_id,
                profile_digest=binding.adapter_profile_digest,
            ),  # type: ignore[arg-type]
        )
    assert empty.value.reason_code == "input_limit_exceeded"

    coordinated = ShadowTraceInputError(
        "orphan_result",
        step_ordinal=3,
        result_ordinal=2,
    )
    with pytest.raises(ShadowTraceInputError) as copied:
        verify_shadow_trace_source(
            report_case.report,
            report_case.source,
            adapter=_Adapter(
                report_case.trace,
                profile_id=binding.adapter_profile_id,
                profile_digest=binding.adapter_profile_digest,
                failure=coordinated,
            ),  # type: ignore[arg-type]
        )
    assert (copied.value.reason_code, copied.value.step_ordinal, copied.value.result_ordinal) == (
        "orphan_result",
        3,
        2,
    )

    class DerivedTraceError(ShadowTraceInputError):
        pass

    for failure in (DerivedTraceError("invalid_tool_call"), RuntimeError("private adapter")):
        with pytest.raises(ShadowTraceInputError) as sanitized:
            verify_shadow_trace_source(
                report_case.report,
                report_case.source,
                adapter=_Adapter(
                    report_case.trace,
                    profile_id=binding.adapter_profile_id,
                    profile_digest=binding.adapter_profile_digest,
                    failure=failure,
                ),  # type: ignore[arg-type]
            )
        assert sanitized.value.reason_code == "invalid_step"

    with pytest.raises(ShadowTraceInputError) as wrong_type:
        verify_shadow_trace_source(
            report_case.report,
            report_case.source,
            adapter=_Adapter(
                object(),
                profile_id=binding.adapter_profile_id,
                profile_digest=binding.adapter_profile_digest,
            ),  # type: ignore[arg-type]
        )
    assert wrong_type.value.reason_code == "digest_mismatch"

    with pytest.raises(ShadowTraceInputError) as profile_alias:
        verify_shadow_trace_source(
            report_case.report,
            report_case.source,
            adapter=_Adapter(
                report_case.trace,
                profile_id=1,
                profile_digest=binding.adapter_profile_digest,
            ),  # type: ignore[arg-type]
        )
    assert profile_alias.value.reason_code == "profile_mismatch"


def test_trace_report_schema_and_nested_exactness_are_not_coercible(
    report_case: _ReportCase,
) -> None:
    body = report_case.report.model_dump(mode="python", warnings=False)
    body["schema_version"] = 1
    with pytest.raises(ValidationError, match="schema version is invalid"):
        ShadowTraceReport.model_validate(body)

    forged_binding = report_case.trace.binding.model_copy()
    forged_binding.__dict__["source_byte_count"] = 1
    body = report_case.report.model_dump(mode="python", warnings=False)
    body["binding"] = forged_binding
    body["binding_digest"] = forged_binding.binding_digest
    body.pop("report_digest")
    constructed = trace_report_module._ShadowTraceReportBody.model_construct(**body)
    with pytest.raises(ValueError, match="nested model is invalid"):
        constructed.validate_self_contained_commitments()


def test_profile_link_guard_rejects_wrong_diagnostic_branch_and_alias(
    report_case: _ReportCase,
) -> None:
    binding = report_case.trace.binding
    with pytest.raises(ValueError, match="diagnostic branch"):
        trace_report_module._require_profile_diagnostic_links(binding, object())  # type: ignore[arg-type]

    forged_binding = binding.model_copy(update={"adapter_profile_id": "harbor-codex/v1"})
    with pytest.raises(ValueError, match="built-in ATIF profile"):
        trace_report_module._require_profile_diagnostic_links(
            forged_binding,
            report_case.trace.diagnostics,
        )


def test_manual_body_digest_helper_matches_report_commitment(report_case: _ReportCase) -> None:
    body = trace_report_module._ShadowTraceReportBody.model_validate(
        report_case.report.model_dump(
            mode="python",
            exclude={"report_digest"},
            warnings=False,
        )
    )
    assert trace_report_module._trace_report_body_digest(body) == report_case.report.report_digest
    assert report_case.report.report_digest == length_prefixed_sha256(
        canonical_json(body.model_dump(mode="json", warnings=False)),
        domain="saliencegate:shadow:trace-report:v1",
    )
