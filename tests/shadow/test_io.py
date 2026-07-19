from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import saliencegate.shadow.io as io_module
from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
)
from saliencegate.security import InstallationKey, RedactionPolicy, SecureFileError
from saliencegate.shadow import (
    ShadowConfig,
    ShadowInputError,
    ShadowInvariantError,
    ShadowRunReport,
    ShadowSession,
)
from saliencegate.shadow.inputs import (
    ShadowActionInput,
    ShadowControllerErrorInput,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowObservationInput,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
)
from saliencegate.shadow.io import (
    MAX_SHADOW_INPUT_BYTES,
    MAX_SHADOW_INPUT_ROWS,
    MAX_SHADOW_LINE_BYTES,
    MAX_SHADOW_REPORT_BYTES,
    PreflightedShadowTrace,
    ShadowReportBinding,
    authorize_shadow_report_publication,
    decode_shadow_run_report,
    encode_shadow_run_report,
    read_shadow_trace,
    shadow_report_binding,
    validate_published_shadow_report,
    validate_shadow_report_replacement,
)
from saliencegate.shadow.report import ShadowReportRow, build_shadow_run_report

RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a")
KEY = InstallationKey(b"k" * 32)
TAG = PayloadDigest(
    algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
    value=KEY._hmac_sha256(
        canonical_json(
            {
                "literal_secrets": (),
                "structured_field_names": (),
            }
        ),
        domain=b"saliencegate:shadow:redaction-policy:v1",
    ),
)
ENVIRONMENT_DIGEST = "b" * 64


def _private_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def _row(**values: object) -> bytes:
    return canonical_json(values) + b"\n"


def _read(
    path: Path,
    *,
    redaction_policy: RedactionPolicy | None = None,
    capture_scope: str = "unknown",
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
    source_adapter: str = "test-shadow/v1",
    installation_key: InstallationKey = KEY,
) -> PreflightedShadowTrace:
    policy = redaction_policy or RedactionPolicy()
    tag = PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=installation_key._hmac_sha256(
            canonical_json(
                {
                    "literal_secrets": policy.literal_secrets,
                    "structured_field_names": policy.structured_field_names,
                }
            ),
            domain=b"saliencegate:shadow:redaction-policy:v1",
        ),
    )
    return read_shadow_trace(
        path,
        run_id=RUN_ID,
        config=ShadowConfig.reference(),
        installation_key=installation_key,
        redaction_policy=policy,
        redaction_policy_tag=tag,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
        source_adapter=source_adapter,
    )


def _complete_trace() -> bytes:
    return b"".join(
        (
            _row(
                schema_version="shadow-input/v1",
                kind="run_start",
                source_event_id="start",
                occurred_at="2026-07-16T10:00:00Z",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="action",
                source_event_id="action-1",
                occurred_at="2026-07-16T10:01:00Z",
                argv=["pytest", "-q"],
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="tool_result",
                source_event_id="tool-1",
                occurred_at="2026-07-16T10:02:00Z",
                action_source_event_id="action-1",
                status="failed",
                exit_status=1,
                exception_type="AssertionError",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="test_result",
                source_event_id="tests-1",
                occurred_at="2026-07-16T10:03:00Z",
                action_source_event_id="action-1",
                framework="pytest",
                status="failed",
                failures=[
                    {
                        "schema_version": "1.0",
                        "test_id": "tests/test_api.py::test_result",
                        "failure_type": "AssertionError",
                        "signature": "assert-result",
                    }
                ],
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="observation",
                source_event_id="observation-1",
                occurred_at="2026-07-16T10:04:00Z",
                source="task_input",
                payload={"state": "waiting"},
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="controller_error",
                source_event_id="controller-1",
                occurred_at="2026-07-16T10:05:00Z",
                error_code="queue_unavailable",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="run_end",
                source_event_id="finish",
                occurred_at="2026-07-16T10:06:00Z",
            ),
        )
    )


def test_reader_preflights_all_wire_kinds_and_preserves_exact_bytes(tmp_path: Path) -> None:
    data = _complete_trace()
    path = _private_file(tmp_path / "events.ndjson", data)

    trace = _read(path, capture_scope="complete_run_declared")

    assert trace.input_bytes == data
    assert trace.input_byte_digest == hashlib.sha256(data).hexdigest()
    assert len(trace.normalized_input_digest) == 64
    assert trace.authorization.path == str(path)
    assert trace.run_id == RUN_ID
    assert tuple(type(row.input_record) for row in trace.rows) == (
        ShadowStartInput,
        ShadowActionInput,
        ShadowToolResultInput,
        ShadowTestResultInput,
        ShadowObservationInput,
        ShadowControllerErrorInput,
        ShadowFinishInput,
    )
    assert tuple(row.input_kind for row in trace.rows) == tuple(ShadowInputKind)
    assert tuple(row.event_sequence for row in trace.rows) == tuple(range(1, 8))
    assert all(row.retry_target_ordinal is None for row in trace.rows)
    tool = trace.rows[2].input_record
    tests = trace.rows[3].input_record
    assert isinstance(tool, ShadowToolResultInput)
    assert isinstance(tests, ShadowTestResultInput)
    assert tool.action == tests.action == trace.rows[1].event_ref
    assert tool.action.sequence == 2
    assert trace.rows[-1].input_kind is ShadowInputKind.FINISH
    assert "action-1" not in repr(trace)


def test_decoded_record_preflight_matches_the_ndjson_wrapper_exactly() -> None:
    lines = tuple(_complete_trace().splitlines())
    records = tuple(io_module._parse_json_object(line) for line in lines)
    original_records = canonical_json(records)
    options = io_module._prepare_options(
        run_id=RUN_ID,
        config=ShadowConfig.reference(),
        installation_key=KEY,
        redaction_policy=RedactionPolicy(),
        redaction_policy_tag=TAG,
        capture_scope="complete_run_declared",
        task_scope_digest=None,
        lineage_scope_digest=None,
        capture_manifest_digest=None,
        source_adapter="test-shadow/v1",
    )

    from_lines = io_module._preflight_rows(lines, options)
    from_values = io_module._preflight_record_values(records, options)

    assert from_values == from_lines
    assert canonical_json(records) == original_records
    assert io_module._normalized_input_digest(
        options,
        tuple(row.input_record for row in from_values),
    ) == io_module._normalized_input_digest(
        options,
        tuple(row.input_record for row in from_lines),
    )


def test_normalized_input_digest_has_a_frozen_golden_and_binds_all_provenance(
    tmp_path: Path,
) -> None:
    data = _complete_trace()
    path = _private_file(tmp_path / "events.ndjson", data)
    baseline = _read(path, capture_scope="complete_run_declared")

    assert baseline.normalized_input_digest == (
        "2c0b3e29f3d5ec743349749a5b65472ea55c259067fd3696a8d0c469d92ea229"
    )
    mutations = (
        _read(path, capture_scope="unknown"),
        _read(
            path,
            capture_scope="complete_run_declared",
            task_scope_digest="1" * 64,
        ),
        _read(
            path,
            capture_scope="complete_run_declared",
            lineage_scope_digest="2" * 64,
        ),
        _read(
            path,
            capture_scope="complete_run_declared",
            capture_manifest_digest="3" * 64,
        ),
        _read(
            path,
            capture_scope="complete_run_declared",
            source_adapter="different-shadow/v1",
        ),
        _read(
            path,
            capture_scope="complete_run_declared",
            installation_key=InstallationKey(b"z" * 32),
        ),
    )
    assert all(
        candidate.normalized_input_digest != baseline.normalized_input_digest
        for candidate in mutations
    )


def test_normalized_input_digest_binds_order_and_strict_content(tmp_path: Path) -> None:
    start = _row(
        schema_version="shadow-input/v1",
        kind="run_start",
        source_event_id="start",
        occurred_at="2026-07-16T10:00:00Z",
    )

    def observation(source_event_id: str, value: int) -> bytes:
        return _row(
            schema_version="shadow-input/v1",
            kind="observation",
            source_event_id=source_event_id,
            occurred_at="2026-07-16T10:01:00Z",
            source="task_input",
            payload={"value": value},
        )

    first = _read(
        _private_file(
            tmp_path / "first.ndjson",
            start + observation("a", 1) + observation("b", 2),
        )
    )
    reordered = _read(
        _private_file(
            tmp_path / "reordered.ndjson",
            start + observation("b", 2) + observation("a", 1),
        )
    )
    changed = _read(
        _private_file(
            tmp_path / "changed.ndjson",
            start + observation("a", 1) + observation("b", 3),
        )
    )

    assert (
        len(
            {
                first.normalized_input_digest,
                reordered.normalized_input_digest,
                changed.normalized_input_digest,
            }
        )
        == 3
    )


def test_reader_marks_exact_and_redaction_equivalent_retries(tmp_path: Path) -> None:
    data = b"".join(
        (
            _row(
                schema_version="shadow-input/v1",
                kind="run_start",
                source_event_id="start",
                occurred_at="2026-07-16T10:00:00Z",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="action",
                source_event_id="action-1",
                occurred_at="2026-07-16T10:01:00Z",
                command="token-alpha",
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="action",
                source_event_id="action-1",
                occurred_at="2026-07-16T10:01:00Z",
                command="token-beta",
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
        )
    )
    path = _private_file(tmp_path / "events.ndjson", data)

    trace = _read(
        path,
        redaction_policy=RedactionPolicy(literal_secrets=("token-alpha", "token-beta")),
    )

    assert trace.unique_input_event_count == 2
    assert trace.retry_row_count == 1
    retry = trace.rows[2]
    assert retry.first_occurrence_ordinal == 2
    assert retry.retry_target_ordinal == 2
    assert retry.event_sequence == 2
    assert retry.source_event_digest == trace.rows[1].source_event_digest


def test_reader_rejects_a_source_collision_after_redaction(tmp_path: Path) -> None:
    data = _complete_trace().splitlines(keepends=True)[:2]
    conflicting = _row(
        schema_version="shadow-input/v1",
        kind="action",
        source_event_id="action-1",
        occurred_at="2026-07-16T10:01:00Z",
        command="different-command",
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )
    path = _private_file(tmp_path / "events.ndjson", b"".join((*data, conflicting)))

    with pytest.raises(ShadowInputError, match="shadow input is invalid") as captured:
        _read(path)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "data",
    (
        b"",
        b"\n",
        b'{"schema_version":"shadow-input/v1","kind":"run_start","kind":"action"}\n',
        b"\xff\n",
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ),
)
def test_reader_rejects_empty_blank_duplicate_malformed_and_nonfinite_json(
    tmp_path: Path,
    data: bytes,
) -> None:
    path = _private_file(tmp_path / "events.ndjson", data)

    with pytest.raises(ShadowInputError):
        _read(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "shadow-input/v2"),
        ("occurred_at", "2026-07-16T10:00:00+00:00"),
        ("occurred_at", "2026-07-16T12:00:00+02:00"),
        ("occurred_at", "2026-07-16 10:00:00Z"),
    ),
)
def test_reader_rejects_unsupported_versions_and_noncanonical_timestamps(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    row = {
        "schema_version": "shadow-input/v1",
        "kind": "run_start",
        "source_event_id": "start",
        "occurred_at": "2026-07-16T10:00:00Z",
    }
    row[field] = value
    path = _private_file(tmp_path / "events.ndjson", _row(**row))

    with pytest.raises(ShadowInputError):
        _read(path)


def test_reader_accepts_an_explicit_six_digit_zero_fraction(tmp_path: Path) -> None:
    data = _complete_trace().replace(
        b"2026-07-16T10:00:00Z",
        b"2026-07-16T10:00:00.000000Z",
        1,
    )
    path = _private_file(tmp_path / "events.ndjson", data)

    trace = _read(path)

    assert len(trace.rows) == 7


@pytest.mark.parametrize(
    "rows",
    (
        (
            {
                "schema_version": "shadow-input/v1",
                "kind": "action",
                "source_event_id": "action",
                "occurred_at": "2026-07-16T10:00:00Z",
                "command": "pytest",
                "working_directory": "/project",
                "environment_digest": ENVIRONMENT_DIGEST,
            },
        ),
        (
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_start",
                "source_event_id": "start",
                "occurred_at": "2026-07-16T10:00:00Z",
            },
            {
                "schema_version": "shadow-input/v1",
                "kind": "tool_result",
                "source_event_id": "result",
                "occurred_at": "2026-07-16T10:01:00Z",
                "action_source_event_id": "missing",
                "status": "failed",
                "exit_status": 1,
            },
        ),
        (
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_start",
                "source_event_id": "start",
                "occurred_at": "2026-07-16T10:00:00Z",
            },
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_end",
                "source_event_id": "finish",
                "occurred_at": "2026-07-16T10:01:00Z",
            },
            {
                "schema_version": "shadow-input/v1",
                "kind": "observation",
                "source_event_id": "late",
                "occurred_at": "2026-07-16T10:02:00Z",
                "source": "task_input",
                "payload": {"value": 1},
            },
        ),
        (
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_start",
                "source_event_id": "start",
                "occurred_at": "2026-07-16T10:00:00Z",
            },
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_start",
                "source_event_id": "start-2",
                "occurred_at": "2026-07-16T10:01:00Z",
            },
        ),
    ),
)
def test_reader_rejects_invalid_lifecycle_and_parent_graph(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
) -> None:
    path = _private_file(tmp_path / "events.ndjson", b"".join(_row(**row) for row in rows))

    with pytest.raises(ShadowInputError):
        _read(path)


def test_reader_rejects_decreasing_unique_timestamps(tmp_path: Path) -> None:
    data = b"".join(
        (
            _row(
                schema_version="shadow-input/v1",
                kind="run_start",
                source_event_id="start",
                occurred_at="2026-07-16T10:01:00Z",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="observation",
                source_event_id="earlier",
                occurred_at="2026-07-16T10:00:00Z",
                source="task_input",
                payload={"value": 1},
            ),
        )
    )

    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "events.ndjson", data))


@pytest.mark.parametrize("parent_source", ("future-action", "start"))
def test_reader_rejects_forward_and_non_action_parents(
    tmp_path: Path,
    parent_source: str,
) -> None:
    rows = [
        _row(
            schema_version="shadow-input/v1",
            kind="run_start",
            source_event_id="start",
            occurred_at="2026-07-16T10:00:00Z",
        ),
        _row(
            schema_version="shadow-input/v1",
            kind="tool_result",
            source_event_id="result",
            occurred_at="2026-07-16T10:01:00Z",
            action_source_event_id=parent_source,
            status="failed",
            exit_status=1,
        ),
    ]
    if parent_source == "future-action":
        rows.append(
            _row(
                schema_version="shadow-input/v1",
                kind="action",
                source_event_id="future-action",
                occurred_at="2026-07-16T10:02:00Z",
                command="pytest",
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            )
        )

    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "events.ndjson", b"".join(rows)))


@pytest.mark.parametrize(
    "mutation",
    (
        {"extra": "field"},
        {"schema_version": 1},
        {"kind": 1},
        {"source_event_id": 1},
        {"occurred_at": 1},
    ),
)
def test_wire_schema_is_strict_and_forbids_extra_fields(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "schema_version": "shadow-input/v1",
        "kind": "run_start",
        "source_event_id": "start",
        "occurred_at": "2026-07-16T10:00:00Z",
    }
    values.update(mutation)

    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "events.ndjson", _row(**values)))


def test_complete_capture_requires_a_unique_final_run_end(tmp_path: Path) -> None:
    open_trace = b"".join(_complete_trace().splitlines(keepends=True)[:-1])
    path = _private_file(tmp_path / "events.ndjson", open_trace)

    with pytest.raises(ShadowInputError):
        _read(path, capture_scope="complete_run_declared")

    assert _read(path, capture_scope="bounded_window").rows[-1].input_kind is not (
        ShadowInputKind.FINISH
    )


def test_reader_enforces_exact_row_line_and_input_bounds_before_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_SHADOW_INPUT_ROWS == 10_000
    assert MAX_SHADOW_LINE_BYTES == 2 * 1024 * 1024
    assert MAX_SHADOW_INPUT_BYTES == 64 * 1024 * 1024
    assert MAX_SHADOW_REPORT_BYTES == 128 * 1024 * 1024

    too_many = b"\n" * (MAX_SHADOW_INPUT_ROWS + 1)
    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "too-many.ndjson", too_many))

    oversized_line = b"{" + b" " * MAX_SHADOW_LINE_BYTES + b"}\n"
    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "long.ndjson", oversized_line))

    monkeypatch.setattr("saliencegate.shadow.io.MAX_SHADOW_INPUT_BYTES", 16)
    with pytest.raises(ShadowInputError):
        _read(
            _private_file(
                tmp_path / "aggregate.ndjson",
                b"{" + b" " * 15 + b"}\n",
            )
        )


def test_reader_uses_private_owner_file_policy(tmp_path: Path) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())
    path.chmod(0o666)

    with pytest.raises(ShadowInputError):
        _read(path)


def test_reader_rejects_a_divergent_redaction_tag_before_parsing(tmp_path: Path) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())

    with pytest.raises(ShadowInputError):
        read_shadow_trace(
            path,
            run_id=RUN_ID,
            config=ShadowConfig.reference(),
            installation_key=KEY,
            redaction_policy=RedactionPolicy(),
            redaction_policy_tag=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="f" * 64,
            ),
            capture_scope="unknown",
            source_adapter="test-shadow/v1",
        )


def test_reader_rejects_secret_like_source_metadata_before_any_repository_use(
    tmp_path: Path,
) -> None:
    data = _row(
        schema_version="shadow-input/v1",
        kind="run_start",
        source_event_id="sk-proj-abcdefghijklmnop",
        occurred_at="2026-07-16T10:00:00Z",
    )

    with pytest.raises(ShadowInputError):
        _read(_private_file(tmp_path / "events.ndjson", data))


@pytest.mark.parametrize(
    "policy",
    (
        RedactionPolicy(structured_field_names=("schema_version",)),
        RedactionPolicy(structured_field_names=("capture_scope",)),
        RedactionPolicy(structured_field_names=("detector_version",)),
        RedactionPolicy(structured_field_names=("source_event_id",)),
        RedactionPolicy(structured_field_names=("source_adapter",)),
        RedactionPolicy(literal_secrets=(str(RUN_ID),)),
    ),
)
def test_reader_rejects_static_marker_identity_metadata_and_signal_conflicts(
    tmp_path: Path,
    policy: RedactionPolicy,
) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())

    with pytest.raises(ShadowInputError):
        _read(path, redaction_policy=policy, capture_scope="complete_run_declared")


def test_reader_rejects_payload_that_is_not_canonical_after_redaction(
    tmp_path: Path,
) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())

    with pytest.raises(ShadowInputError):
        _read(
            path,
            redaction_policy=RedactionPolicy(
                structured_field_names=("environment_digest",),
            ),
            capture_scope="complete_run_declared",
        )


def test_reader_rejects_dynamic_signal_timestamp_redaction_conflicts(
    tmp_path: Path,
) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())

    with pytest.raises(ShadowInputError):
        _read(
            path,
            redaction_policy=RedactionPolicy(
                literal_secrets=("2026-07-16T10",),
            ),
            capture_scope="complete_run_declared",
        )


@pytest.mark.parametrize(
    "source_adapter",
    (
        "saliencegate.repository",
        "SalienceGate.Repository",
        "contains whitespace",
        "",
        "x" * 257,
    ),
)
def test_reader_matches_the_session_source_adapter_contract(
    tmp_path: Path,
    source_adapter: str,
) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())

    with pytest.raises(ShadowInputError):
        _read(path, source_adapter=source_adapter)


async def _report() -> ShadowRunReport:
    config = ShadowConfig.reference()
    session = ShadowSession.in_memory(
        run_id=RUN_ID,
        config=config,
        installation_key=KEY,
        capture_scope="unknown",
        source_adapter="test-shadow/v1",
    )
    result = await session._submit(
        ShadowStartInput(
            source_event_id="start",
            occurred_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        ),
        cli_input_ordinal=1,
    )
    observation = result.observation
    row = ShadowReportRow(
        input_ordinal=1,
        source_event_digest=observation.source_event_digest,
        first_occurrence_ordinal=1,
        retry_target_ordinal=None,
        event_type=EventType.RUN_START,
        phase=EventPhase.INITIALIZATION,
        input_kind=ShadowInputKind.START,
        persistence_disposition="appended",
        observation_digest=observation.observation_digest,
    )
    report = build_shadow_run_report(
        run_id=RUN_ID,
        initial_ledger_entry_count=0,
        initial_ledger_chain_tag=None,
        initial_ledger_projection_tag=None,
        initial_ledger_head_tag=None,
        input_byte_digest="1" * 64,
        normalized_input_digest="2" * 64,
        redaction_policy_tag=observation.redaction_policy_tag,
        detector_profile_digest=config.detector_profile_digest,
        capture_scope="unknown",
        rows=(row,),
        observations=(observation,),
    )
    await session.aclose()
    return report


@pytest.mark.asyncio
async def test_report_codec_is_bounded_canonical_and_self_verifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = await _report()

    encoded = encode_shadow_run_report(report)

    assert encoded == canonical_json(report)
    assert not encoded.endswith(b"\n")
    assert decode_shadow_run_report(encoded) == report
    with pytest.raises(ShadowInvariantError):
        decode_shadow_run_report(encoded + b"\n")
    duplicate = encoded.replace(b'{"', b'{"schema_version":"shadow-run-report/v1","', 1)
    with pytest.raises(ShadowInvariantError):
        decode_shadow_run_report(duplicate)
    monkeypatch.setattr("saliencegate.shadow.io._MAX_REPORT_BYTES", len(encoded) - 1)
    with pytest.raises(ShadowInvariantError):
        encode_shadow_run_report(report)
    with pytest.raises(ShadowInvariantError):
        decode_shadow_run_report(encoded)


@pytest.mark.asyncio
async def test_replacement_and_postpublication_validators_bind_exact_report() -> None:
    report = await _report()
    encoded = encode_shadow_run_report(report)
    binding = shadow_report_binding(report)

    assert validate_shadow_report_replacement(encoded, binding) is True
    assert validate_published_shadow_report(encoded, report) is True
    changed = json.loads(encoded)
    changed["input_byte_digest"] = "3" * 64
    changed_bytes = canonical_json(changed)
    assert validate_shadow_report_replacement(changed_bytes, binding) is False
    assert validate_published_shadow_report(changed_bytes, report) is False
    assert validate_shadow_report_replacement(b"not-json", binding) is False
    assert "1" * 64 not in repr(binding)


@pytest.mark.asyncio
async def test_report_binding_is_constructible_prepublication_and_checks_every_field() -> None:
    report = await _report()
    binding = shadow_report_binding(report)
    prepublication = ShadowReportBinding(
        run_id=report.run_id,
        input_byte_digest=report.input_byte_digest,
        normalized_input_digest=report.normalized_input_digest,
        redaction_policy_tag=report.redaction_policy_tag,
        detector_profile_digest=report.detector_profile_digest,
        capture_scope=report.capture_scope,
        task_scope_digest=report.task_scope_digest,
        lineage_scope_digest=report.lineage_scope_digest,
        capture_manifest_digest=report.capture_manifest_digest,
    )
    encoded = encode_shadow_run_report(report)

    assert prepublication == binding
    mutations = (
        replace(binding, run_id=UUID("f35f05f3-555b-4f09-8996-a7b3693bb54a")),
        replace(binding, input_byte_digest="3" * 64),
        replace(binding, normalized_input_digest="4" * 64),
        replace(
            binding,
            redaction_policy_tag=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="5" * 64,
            ),
        ),
        replace(binding, detector_profile_digest="6" * 64),
        replace(binding, capture_scope="bounded_window"),
        replace(binding, task_scope_digest="7" * 64),
        replace(binding, lineage_scope_digest="8" * 64),
        replace(binding, capture_manifest_digest="9" * 64),
    )
    assert all(
        validate_shadow_report_replacement(encoded, candidate) is False for candidate in mutations
    )


@pytest.mark.asyncio
async def test_publication_adapter_preserves_unrelated_output_and_revalidates_exact_bytes(
    tmp_path: Path,
) -> None:
    report = await _report()
    encoded = encode_shadow_run_report(report)
    output = tmp_path / "report.json"

    publication = authorize_shadow_report_publication(output)
    reopened = publication.publish(
        encoded,
        validate_published=lambda data: validate_published_shadow_report(data, report),
    )

    assert reopened.data == encoded
    assert output.read_bytes() == encoded
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(SecureFileError):
        authorize_shadow_report_publication(output)
    replacement = authorize_shadow_report_publication(
        output,
        replacement_binding=shadow_report_binding(report),
    )
    replaced = replacement.publish(
        encoded,
        validate_published=lambda data: validate_published_shadow_report(data, report),
    )
    assert replaced.data == encoded

    output.write_bytes(b"unrelated")
    output.chmod(0o600)
    with pytest.raises(SecureFileError):
        authorize_shadow_report_publication(
            output,
            replacement_binding=shadow_report_binding(report),
        )
    assert output.read_bytes() == b"unrelated"


@pytest.mark.parametrize(
    "override",
    (
        {"run_id": UUID("b35f05f3-555b-1f09-8996-a7b3693bb54a")},
        {"input_byte_digest": "invalid"},
        {"redaction_policy_tag": object()},
        {
            "redaction_policy_tag": PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value="a" * 64,
            )
        },
        {"detector_profile_digest": "invalid"},
        {"capture_scope": "unsupported"},
        {"task_scope_digest": "invalid"},
    ),
)
def test_report_binding_rejects_invalid_invariants(override: dict[str, object]) -> None:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "input_byte_digest": "1" * 64,
        "normalized_input_digest": "2" * 64,
        "redaction_policy_tag": TAG,
        "detector_profile_digest": "3" * 64,
        "capture_scope": "unknown",
    }
    values.update(override)

    with pytest.raises(ShadowInvariantError):
        ShadowReportBinding(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    (
        {"run_id": UUID("b35f05f3-555b-1f09-8996-a7b3693bb54a")},
        {"source_adapter": 1},
        {"installation_key": object()},
        {"redaction_policy": object()},
    ),
)
def test_reader_rejects_invalid_preflight_options_before_open(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "config": ShadowConfig.reference(),
        "installation_key": KEY,
        "redaction_policy": RedactionPolicy(),
        "redaction_policy_tag": TAG,
        "capture_scope": "unknown",
        "source_adapter": "test-shadow/v1",
    }
    values.update(override)

    def forbid_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid preflight reached the filesystem")

    monkeypatch.setattr(io_module, "read_stable_file", forbid_open)

    with pytest.raises(ShadowInputError):
        read_shadow_trace("never-opened", **values)  # type: ignore[arg-type]


def test_redaction_identity_probe_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingRedactor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def redact_payload(self, _payload: object) -> object:
            raise ValueError("injected redaction failure")

    monkeypatch.setattr(io_module, "Redactor", FailingRedactor)

    assert io_module._payload_is_redaction_identity(RedactionPolicy(), {}) is False


def test_low_level_wire_parsers_reject_leaf_type_mismatches() -> None:
    with pytest.raises(ValueError):
        io_module._finite_float("1e999")
    with pytest.raises(ValueError):
        io_module._parse_json_object(b"[]")
    with pytest.raises(ValueError):
        io_module._action_parent(1, {})

    common = {
        "schema_version": "shadow-input/v1",
        "source_event_id": "event-1",
        "occurred_at": "2026-07-16T10:00:00Z",
    }
    invalid_records = (
        {**common, "kind": "unsupported"},
        {
            **common,
            "kind": "action",
            "argv": "pytest",
            "working_directory": "/project",
            "environment_digest": ENVIRONMENT_DIGEST,
        },
        {
            **common,
            "kind": "test_result",
            "action_source_event_id": "action-1",
            "framework": "pytest",
            "status": "failed",
            "failures": {},
        },
        {
            **common,
            "kind": "observation",
            "source": 1,
            "payload": {},
        },
    )
    for record in invalid_records:
        with pytest.raises(ValueError):
            io_module._parse_input_record(record, known_sources={})


@pytest.mark.parametrize("interruption_type", (KeyboardInterrupt, SystemExit))
def test_reader_reraises_process_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
) -> None:
    path = _private_file(tmp_path / "events.ndjson", _complete_trace())
    interruption = interruption_type()

    def interrupt(*_args: object, **_kwargs: object) -> object:
        raise interruption

    monkeypatch.setattr(io_module, "read_stable_file", interrupt)

    with pytest.raises(interruption_type) as raised:
        _read(path)

    assert raised.value is interruption


@pytest.mark.asyncio
async def test_report_codec_and_publication_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = await _report()
    encoded = encode_shadow_run_report(report)

    with pytest.raises(ShadowInvariantError):
        decode_shadow_run_report(b"[]")
    with pytest.raises(ShadowInvariantError):
        encode_shadow_run_report(object())  # type: ignore[arg-type]
    assert validate_shadow_report_replacement(encoded, object()) is False  # type: ignore[arg-type]
    assert validate_published_shadow_report(encoded, object()) is False  # type: ignore[arg-type]
    with pytest.raises(ShadowInvariantError):
        authorize_shadow_report_publication(
            tmp_path / "report.json",
            replacement_binding=object(),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(io_module, "_decode_report", lambda _data: object())
    with pytest.raises(ShadowInvariantError):
        encode_shadow_run_report(report)


def test_preflight_row_repr_is_payload_free(tmp_path: Path) -> None:
    trace = _read(_private_file(tmp_path / "events.ndjson", _complete_trace()))

    rendered = repr(trace.rows[1])

    assert rendered == (
        "PreflightedShadowRow(input_ordinal=2, input_kind='action', "
        "event_sequence=2, is_retry=False)"
    )
    assert "pytest" not in rendered
    assert "/project" not in rendered
