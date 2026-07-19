from __future__ import annotations

from dataclasses import replace

import pytest
from tests.shadow.conftest import TraceEventFactory
from tests.shadow.test_trace_report import _matching_shadow_report, _trace

import saliencegate.shadow.report as report_module
import saliencegate.shadow.trace_report as trace_report_module
from saliencegate.domain import canonical_json
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.report import ShadowRunReport, build_shadow_run_report
from saliencegate.shadow.trace_report import encode_shadow_trace_report

_PRIVATE_SENTINELS = ("private-command", "/private/project")


def _run_report_kwargs(report: ShadowRunReport) -> dict[str, object]:
    return {
        "run_id": report.run_id,
        "initial_ledger_entry_count": report.initial_ledger_entry_count,
        "initial_ledger_chain_tag": report.initial_ledger_chain_tag,
        "initial_ledger_projection_tag": report.initial_ledger_projection_tag,
        "initial_ledger_head_tag": report.initial_ledger_head_tag,
        "input_byte_digest": report.input_byte_digest,
        "normalized_input_digest": report.normalized_input_digest,
        "redaction_policy_tag": report.redaction_policy_tag,
        "detector_profile_digest": report.detector_profile_digest,
        "capture_scope": report.capture_scope,
        "task_scope_digest": report.task_scope_digest,
        "lineage_scope_digest": report.lineage_scope_digest,
        "capture_manifest_digest": report.capture_manifest_digest,
        "rows": report.rows,
        "observations": report.observations,
    }


def _trusted_run_report(
    report: ShadowRunReport,
) -> report_module._TrustedShadowRunReport:
    return report_module._build_shadow_run_report_trusted(  # type: ignore[arg-type]
        **_run_report_kwargs(report)
    )


def _assert_sanitized(error: ShadowInvariantError) -> None:
    rendered = f"{error!s} {error!r}"
    assert all(value not in rendered for value in _PRIVATE_SENTINELS)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_trusted_run_report_is_canonically_identical_to_public_builder(
    trace_event_factory: TraceEventFactory,
) -> None:
    trace = _trace()
    reference = _matching_shadow_report(trace_event_factory, trace)
    kwargs = _run_report_kwargs(reference)

    public = build_shadow_run_report(**kwargs)  # type: ignore[arg-type]
    trusted = report_module._build_shadow_run_report_trusted(  # type: ignore[arg-type]
        **kwargs
    )
    sealed = report_module._require_trusted_shadow_run_report(trusted)

    assert sealed == public
    assert sealed.report_digest == public.report_digest
    assert canonical_json(sealed) == canonical_json(public)
    assert sealed.model_dump_json(warnings=False) == public.model_dump_json(warnings=False)


def test_trusted_trace_report_matches_existing_builder_bytes_and_digests(
    trace_event_factory: TraceEventFactory,
) -> None:
    trace = _trace()
    public_run_report = _matching_shadow_report(trace_event_factory, trace)
    trusted_run_report = _trusted_run_report(public_run_report)
    arguments = {
        "trace": trace,
        "session_binding": trace.binding,
        "authenticated_start_source_adapter": trace.binding.source_adapter,
    }

    public = trace_report_module._build_shadow_trace_report(
        shadow_report=public_run_report,
        **arguments,
    )
    trusted = trace_report_module._build_shadow_trace_report_trusted(
        shadow_report=trusted_run_report,
        **arguments,
    )

    assert canonical_json(trusted) == canonical_json(public)
    assert encode_shadow_trace_report(trusted) == encode_shadow_trace_report(public)
    assert trusted.report_digest == public.report_digest
    assert trusted.shadow_report.report_digest == public.shadow_report.report_digest
    assert trusted.binding_digest == public.binding_digest
    assert trusted.diagnostics_digest == public.diagnostics_digest
    assert trusted.mapped_record_digest == public.mapped_record_digest
    assert trusted.normalized_input_digest == public.normalized_input_digest


@pytest.mark.parametrize(
    "tampering",
    ("wrong_token", "copied_report", "altered_report"),
)
def test_trusted_trace_report_rejects_tampered_wrapper_without_disclosure(
    trace_event_factory: TraceEventFactory,
    tampering: str,
) -> None:
    trace = _trace()
    public_run_report = _matching_shadow_report(trace_event_factory, trace)
    trusted = _trusted_run_report(public_run_report)
    if tampering == "wrong_token":
        damaged = replace(trusted, _token=object())
    elif tampering == "copied_report":
        copied_report = ShadowRunReport.model_validate_json(
            trusted.report.model_dump_json(warnings=False)
        )
        damaged = replace(trusted, report=copied_report)
    else:
        altered = trusted.report.model_copy(update={"report_digest": "0" * 64})
        damaged = replace(trusted, report=altered)

    with pytest.raises(ShadowInvariantError) as captured:
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=damaged,
            session_binding=trace.binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )

    _assert_sanitized(captured.value)
