from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_trace import build_trace

from saliencegate.shadow import ShadowAnalyzer, ShadowInvariantError
from saliencegate.shadow.io import (
    ShadowTraceReportBinding,
    _parse_canonical_timestamp,
    authorize_shadow_trace_report_publication,
    encode_shadow_trace_report,
    shadow_trace_report_binding,
    validate_shadow_trace_report_binding,
)
from saliencegate.shadow.trace import ShadowTraceBinding


def _binding_values() -> dict[str, object]:
    trace = build_trace()
    session = _memory_session(trace)
    return {
        "run_id": trace.run_id,
        "trace_binding": trace.binding,
        "diagnostics_digest": "1" * 64,
        "mapped_record_digest": trace.mapped_record_digest,
        "normalized_input_digest": "2" * 64,
        "redaction_policy_tag": session._redaction_policy_tag,
        "detector_profile_digest": session._config.detector_profile_digest,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", UUID("11111111-1111-1111-8111-111111111111")),
        ("trace_binding", object()),
        ("diagnostics_digest", "not-a-digest"),
    ),
)
def test_trace_report_binding_rejects_invalid_identity_fields(
    field: str,
    value: object,
) -> None:
    values = _binding_values()
    values[field] = value

    with pytest.raises(ShadowInvariantError):
        ShadowTraceReportBinding(**values)  # type: ignore[arg-type]


def test_trace_report_binding_rejects_validation_copy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _binding_values()
    replacement = build_trace(
        adapter_descriptor={
            "schema_version": "example-shadow-adapter/v1",
            "mapping": {"mode": "different"},
        }
    ).binding

    def drifted_copy(_cls: type[ShadowTraceBinding], _value: bytes) -> ShadowTraceBinding:
        return replacement

    monkeypatch.setattr(
        ShadowTraceBinding,
        "model_validate_json",
        classmethod(drifted_copy),
    )
    with pytest.raises(ShadowInvariantError):
        ShadowTraceReportBinding(**values)  # type: ignore[arg-type]


def test_timestamp_parser_rejects_noncanonical_fraction_width() -> None:
    with pytest.raises(ValueError, match="timestamp is not canonical"):
        _parse_canonical_timestamp("2026-07-17T09:00:00.0Z")


@pytest.mark.asyncio
async def test_trace_report_io_rejects_wrong_binding_and_replacement_types(
    tmp_path: Path,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)
    data = encode_shadow_trace_report(report)
    binding = shadow_trace_report_binding(report)

    assert validate_shadow_trace_report_binding(data, binding)
    assert not validate_shadow_trace_report_binding(data, object())  # type: ignore[arg-type]

    output = tmp_path / "trace-report.json"
    with pytest.raises(ShadowInvariantError):
        authorize_shadow_trace_report_publication(
            output,
            replacement_binding=binding,
            replacement_report=report,
        )
    with pytest.raises(ShadowInvariantError):
        authorize_shadow_trace_report_publication(
            output,
            replacement_binding=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ShadowInvariantError):
        authorize_shadow_trace_report_publication(
            output,
            replacement_report=object(),  # type: ignore[arg-type]
        )
