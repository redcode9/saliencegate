from __future__ import annotations

from typing import Any

import pytest
from tests.shadow.test_atif_codex import adapt as adapt_codex
from tests.shadow.test_atif_codex import diagnostics as codex_diagnostics
from tests.shadow.test_atif_codex import single_call_source
from tests.shadow.test_trace import build_trace

import saliencegate.shadow.atif as atif_module
import saliencegate.shadow.trace as trace_module
import saliencegate.shadow.trace_report as trace_report_module
from saliencegate.shadow.errors import ShadowTraceInputError
from saliencegate.shadow.trace import (
    ATIFShadowDiagnostics,
    ShadowRecordDiagnostics,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_format", "bad format", "source format"),
        ("source_schema_version", "bad schema", "source schema"),
        ("adapter_profile_id", "bad profile", "profile identifier"),
        ("source_adapter", "bad adapter", "source adapter"),
        ("source_digest_kind", "uncommitted", "source digest kind"),
        ("identity_mode", "uncommitted", "identity mode"),
        ("identity_mode", "legacy_explicit", "legacy identity"),
        ("timestamp_mode", "local_time", "timestamp mode"),
        ("capture_scope", "implicit", "capture scope"),
        ("source_byte_digest", "g" * 64, "binding digest"),
        ("source_adapter", "example.valid-but-wrong", "source adapter identity"),
    ),
)
def test_binding_revalidation_rejects_corrupted_internal_state(
    field: str,
    value: object,
    message: str,
) -> None:
    binding = build_trace().binding.model_copy(update={field: value})

    with pytest.raises(ValueError, match=message):
        binding.validate_identity_and_digest()


def test_direct_diagnostics_revalidation_rejects_noncanonical_internal_state() -> None:
    diagnostics = build_trace().diagnostics
    assert type(diagnostics) is ShadowRecordDiagnostics

    invalid_states: tuple[tuple[dict[str, object], str], ...] = (
        ({"input_kind_counts": diagnostics.input_kind_counts[:-1]}, "incomplete"),
        (
            {
                "input_kind_counts": (
                    diagnostics.input_kind_counts[1],
                    diagnostics.input_kind_counts[0],
                    *diagnostics.input_kind_counts[2:],
                )
            },
            "not canonical",
        ),
        (
            {"repeated_source_identifier_count": diagnostics.source_record_count + 1},
            "retry count",
        ),
        ({"diagnostics_digest": "g" * 64}, "diagnostics digest"),
    )
    for update, message in invalid_states:
        forged = diagnostics.model_copy(update=update)
        with pytest.raises(ValueError, match=message):
            forged.validate_counts_and_digest()


def test_diagnostics_field_guards_reject_aliases_and_malformed_pairs() -> None:
    with pytest.raises(ValueError, match="diagnostics text"):
        ShadowRecordDiagnostics.require_exact_text(1)
    with pytest.raises(ValueError, match="kind count"):
        ShadowRecordDiagnostics.require_exact_nested_counts([object()])
    with pytest.raises(ValueError, match="kind count"):
        ShadowRecordDiagnostics.require_exact_nested_counts([("run_start", True)])

    with pytest.raises(ValueError, match="ATIF diagnostics text"):
        ATIFShadowDiagnostics.require_exact_text(1)
    with pytest.raises(ValueError, match="ATIF diagnostics count"):
        ATIFShadowDiagnostics.require_exact_count(True)
    with pytest.raises(ValueError, match="disposition counts"):
        ATIFShadowDiagnostics.require_exact_nested_counts(object())
    with pytest.raises(ValueError, match="disposition count"):
        ATIFShadowDiagnostics.require_exact_nested_counts([object()])


def test_sealed_diagnostics_models_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="ShadowRecordDiagnostics cannot be subclassed"):

        class DerivedDirectDiagnostics(ShadowRecordDiagnostics):
            pass

    with pytest.raises(TypeError, match="ATIFShadowDiagnostics cannot be subclassed"):

        class DerivedATIFDiagnostics(ATIFShadowDiagnostics):
            pass


def _replace_pair(
    values: tuple[tuple[str, int], ...],
    index: int,
    replacement: tuple[str, int],
) -> tuple[tuple[str, int], ...]:
    copied = list(values)
    copied[index] = replacement
    return tuple(copied)


def test_atif_diagnostics_revalidation_rejects_forged_equations_and_digests() -> None:
    diagnostics = codex_diagnostics(adapt_codex(single_call_source(0)))
    first_tool_count = diagnostics.tool_call_disposition_counts[0][1]
    first_result_count = diagnostics.result_disposition_counts[0][1]

    invalid_states: tuple[tuple[dict[str, object], str], ...] = (
        (
            {
                "tool_call_disposition_counts": _replace_pair(
                    diagnostics.tool_call_disposition_counts,
                    0,
                    ("forged_tool_kind", first_tool_count),
                )
            },
            "tool dispositions",
        ),
        (
            {
                "result_disposition_counts": _replace_pair(
                    diagnostics.result_disposition_counts,
                    0,
                    ("forged_result_kind", first_result_count),
                )
            },
            "result dispositions",
        ),
        (
            {
                "result_disposition_counts": _replace_pair(
                    diagnostics.result_disposition_counts,
                    0,
                    ("mapped_structured_outcome", first_result_count + 1),
                )
            },
            "result disposition equation",
        ),
        (
            {"ignored_message_step_count": diagnostics.total_step_count + 1},
            "ignored step count",
        ),
        (
            {"mapped_shadow_record_count": diagnostics.mapped_shadow_record_count + 1},
            "mapped record equation",
        ),
        ({"profile_audit_manifest_digest": "g" * 64}, "diagnostics digest"),
        ({"diagnostics_digest": "0" * 64}, "digest does not match"),
    )
    for update, message in invalid_states:
        forged = diagnostics.model_copy(update=update)
        with pytest.raises(ValueError, match=message):
            forged.validate_equations_and_digest()


def test_trace_discriminator_and_adapter_name_fail_closed_for_unknown_shapes() -> None:
    assert trace_module._diagnostics_discriminator(object()) is None
    with pytest.raises(ValueError, match="generated source adapter"):
        trace_module._source_adapter("atif", "x" * 200, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "text",
    (
        r'"\ud800\udc00"',
        r'"\u0061"',
    ),
)
def test_atif_json_preflight_accepts_well_formed_unicode_escapes(text: str) -> None:
    atif_module._JSONPreflight(text).run()


@pytest.mark.parametrize(
    "text",
    (
        '"\\',
        r'"\u12"',
        r'"\ud800\u0041"',
        "-",
    ),
)
def test_atif_json_preflight_rejects_truncated_or_invalid_tokens(text: str) -> None:
    with pytest.raises(atif_module._JSONSyntaxError):
        atif_module._JSONPreflight(text).run()


def test_atif_json_hooks_and_number_guards_reject_bounded_adversarial_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(atif_module, "_MAX_OBJECT_MEMBERS", 0)
        with pytest.raises(atif_module._JSONLimitError):
            atif_module._duplicate_safe_object([("key", "value")])

    with pytest.raises(atif_module._JSONSyntaxError):
        atif_module._reject_constant("NaN")
    assert atif_module._exact_integer(atif_module._JSONNumber("1" * 21, True)) is None
    assert atif_module._exact_integer(atif_module._JSONNumber("invalid", True)) is None

    with pytest.raises(ShadowTraceInputError) as too_long:
        atif_module._validate_duration(
            atif_module._JSONNumber("1" * 129, False),
            step=2,
            call=3,
        )
    assert (
        too_long.value.reason_code,
        too_long.value.step_ordinal,
        too_long.value.call_ordinal,
    ) == (
        "invalid_tool_call",
        2,
        3,
    )

    with pytest.raises(ShadowTraceInputError) as malformed:
        atif_module._validate_duration(
            atif_module._JSONNumber("invalid", False),
            step=4,
            call=5,
        )
    assert (
        malformed.value.reason_code,
        malformed.value.step_ordinal,
        malformed.value.call_ordinal,
    ) == (
        "invalid_tool_call",
        4,
        5,
    )

    with pytest.raises(ShadowTraceInputError) as timestamp:
        atif_module._normalize_timestamp(1, step=6)
    assert (timestamp.value.reason_code, timestamp.value.step_ordinal) == (
        "invalid_timestamp",
        6,
    )


def test_atif_source_boundary_rejects_mutable_bytes_and_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ShadowTraceInputError) as mutable:
        atif_module._parse_source(bytearray(b"{}"))  # type: ignore[arg-type]
    assert mutable.value.reason_code == "invalid_json"

    def interrupt(_self: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(atif_module._JSONPreflight, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        atif_module._parse_source(b"{}")


def test_trace_report_preflight_tracks_escaped_string_bytes() -> None:
    trace_report_module._preflight_canonical_json_structure(b'"a\\\\b"')


def test_trace_report_model_comparison_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = build_trace().binding

    def interrupt(_value: object) -> bytes:
        raise KeyboardInterrupt

    monkeypatch.setattr(trace_report_module, "canonical_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trace_report_module._models_match_exactly(binding, binding)


def test_atif_report_link_rejects_unsealed_profile_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = adapt_codex(single_call_source(0))
    monkeypatch.setattr(trace_report_module, "_sealed_atif_report_contract", lambda _profile: None)

    with pytest.raises(ValueError, match="profile is not sealed"):
        trace_report_module._require_profile_diagnostic_links(trace.binding, trace.diagnostics)
