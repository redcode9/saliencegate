from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import pytest
from tests.shadow.conftest import TraceEventFactory
from tests.shadow.test_report import _builder_kwargs, _make_case

import saliencegate.shadow.observation as observation_module
import saliencegate.shadow.report as report_module
from saliencegate.domain import PayloadDigest, PayloadDigestAlgorithm, SignalType

SYNTHETIC_TAG = PayloadDigest(
    algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
    value="e" * 64,
)


@pytest.mark.parametrize(
    "preflight",
    (
        observation_module._payload_digest_is_preflight_safe,
        observation_module._outcome_is_preflight_safe,
        observation_module._detector_evaluation_is_preflight_safe,
        observation_module._signal_is_preflight_safe,
        observation_module._extraction_report_is_preflight_safe,
        observation_module._shadow_config_is_preflight_safe,
        observation_module._heuristic_is_preflight_safe,
        observation_module._event_ref_is_preflight_safe,
        observation_module._shadow_observation_is_preflight_safe,
    ),
)
def test_observation_preflight_rejects_objects_without_an_exact_record_shape(
    preflight: Callable[[object], bool],
) -> None:
    """A duck-typed or forged object must never cross a trusted-copy boundary."""

    assert preflight(object()) is False


def test_observation_preflight_rejects_an_unrecognized_nested_model_type() -> None:
    assert observation_module._model_is_preflight_safe(PayloadDigest, object()) is False
    assert (
        observation_module._model_is_preflight_safe(report_module.ShadowReportRow, object())
        is False
    )


@pytest.mark.parametrize(
    ("copier", "bad_value", "message"),
    (
        (
            observation_module._copy_detector_evaluations,
            [],
            "every supported detector evaluation",
        ),
        (observation_module._copy_signals, [], "signals are invalid"),
        (
            observation_module._copy_heuristic_evaluations,
            (),
            "exactly one heuristic evaluation",
        ),
        (
            observation_module._copy_signal_types,
            (SignalType.TOOL_ERROR, "tool_error"),
            "type declarations are invalid",
        ),
        (observation_module._copy_event_prefix, (), "event prefix is invalid"),
        (
            observation_module._copy_detection_context,
            object(),
            "context failed preflight validation",
        ),
    ),
)
def test_observation_copiers_reject_noncanonical_container_shapes(
    copier: Callable[[object], object],
    bad_value: object,
    message: str,
) -> None:
    """Collection boundaries reject coercible lists, empty evidence, and mixed element types."""

    with pytest.raises(ValueError, match=message):
        copier(bad_value)


@pytest.mark.parametrize(
    ("operation", "message"),
    (
        (
            lambda: report_module._copy_exact_model(PayloadDigest, object()),
            "failed preflight validation",
        ),
        (lambda: report_module._copy_hmac_tag(SYNTHETIC_TAG), "must use HMAC"),
        (lambda: report_module._require_exact_digest("A" * 64), "digest is invalid"),
        (
            lambda: report_module._require_optional_exact_digest("g" * 64),
            "optional report digest is invalid",
        ),
        (
            lambda: report_module.ShadowReportRow.require_exact_fixed_strings(1),
            "row string is invalid",
        ),
        (lambda: report_module._copy_rows([]), "rows are invalid"),
        (lambda: report_module._copy_observations(()), "observations are invalid"),
        (
            lambda: report_module._copy_signal_types([SignalType.TOOL_ERROR]),
            "type declaration is invalid",
        ),
        (
            lambda: report_module._ShadowRunReportBody.require_exact_fixed_strings(object()),
            "fixed string is invalid",
        ),
    ),
)
def test_report_boundary_helpers_reject_coercible_or_untrusted_values(
    operation: Callable[[], object],
    message: str,
) -> None:
    """Report validation remains strict even for values Pydantic could otherwise coerce."""

    with pytest.raises(ValueError, match=message):
        operation()


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"run_id": UUID(int=0)}, "run identity is invalid"),
        ({"initial_ledger_entry_count": True}, "ledger count is invalid"),
        ({"capture_scope": "all_events"}, "capture scope is invalid"),
        ({"detector_profile_digest": "g" * 64}, "detector profile identity is invalid"),
        ({"task_scope_digest": "g" * 64}, "capture identity is invalid"),
        ({"redaction_policy_tag": SYNTHETIC_TAG}, "redaction identity is invalid"),
        (
            {
                "initial_ledger_entry_count": 1,
                "initial_ledger_chain_tag": SYNTHETIC_TAG,
                "initial_ledger_projection_tag": SYNTHETIC_TAG,
                "initial_ledger_head_tag": SYNTHETIC_TAG,
            },
            "ledger identity algorithm is invalid",
        ),
    ),
)
def test_report_aggregate_derivation_rejects_invalid_provenance_before_counting(
    trace_event_factory: TraceEventFactory,
    updates: dict[str, object],
    message: str,
) -> None:
    """Invalid capture provenance cannot be hidden behind otherwise valid evidence rows."""

    case = _make_case(trace_event_factory)
    kwargs = _aggregate_kwargs(case.rows, case.observations)
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        report_module._derive_aggregates(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scenario", "message"),
    (
        ("duplicate_source", "source identities are not unique"),
        ("missing_retry_target", "retry target is missing"),
        ("mismatched_retry", "retry row disagrees"),
        ("missing_start", "does not start"),
        ("multiple_starts", "multiple run starts"),
        ("nonfinal_finish", "run end is not unique and final"),
        ("duplicate_event", "duplicate events"),
        ("duplicate_observation_identity", "duplicate identities"),
    ),
)
def test_report_aggregate_derivation_rejects_ambiguous_row_topology(
    trace_event_factory: TraceEventFactory,
    scenario: str,
    message: str,
) -> None:
    """Rows and observations must prove one unambiguous ordered run before aggregation."""

    case = _make_case(trace_event_factory)
    rows = list(case.rows)
    observations = list(case.observations)

    if scenario == "duplicate_source":
        rows[1] = rows[1].model_copy(update={"source_event_digest": rows[0].source_event_digest})
    elif scenario == "missing_retry_target":
        rows[3] = rows[3].model_copy(update={"retry_target_ordinal": 999})
    elif scenario == "mismatched_retry":
        rows[3] = rows[3].model_copy(update={"source_event_digest": "f" * 64})
    elif scenario == "missing_start":
        rows[0] = rows[0].model_copy(update={"input_kind": rows[1].input_kind})
    elif scenario == "multiple_starts":
        rows[1] = rows[1].model_copy(update={"input_kind": rows[0].input_kind})
    elif scenario == "nonfinal_finish":
        rows[1] = rows[1].model_copy(update={"input_kind": rows[-1].input_kind})
    elif scenario == "duplicate_event":
        observations[1] = observations[1].model_copy(update={"event_id": observations[0].event_id})
    else:
        observations[1] = observations[1].model_copy(
            update={"observation_digest": observations[0].observation_digest}
        )

    kwargs = _aggregate_kwargs(tuple(rows), tuple(observations))
    with pytest.raises(ValueError, match=message):
        report_module._derive_aggregates(**kwargs)  # type: ignore[arg-type]


def _aggregate_kwargs(
    rows: tuple[object, ...],
    observations: tuple[object, ...],
) -> dict[str, object]:
    kwargs = _builder_kwargs(rows, observations)  # type: ignore[arg-type]
    del kwargs["input_byte_digest"]
    del kwargs["normalized_input_digest"]
    return kwargs
