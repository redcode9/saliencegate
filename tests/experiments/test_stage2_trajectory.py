from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.experiments.trajectory as trajectory_module
from saliencegate.domain import (
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    TextSpan,
    TrustLabel,
    canonical_json,
)
from saliencegate.experiments.trajectory import (
    PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID,
    STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION,
    STAGE2_TRAJECTORY_VERSION,
    Stage2Trajectory,
    Stage2TrajectoryFixtureError,
    Stage2TrajectoryRecord,
    build_stage2_trajectory,
    load_stage2_trajectory,
)
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
)
from saliencegate.runtime.fixed_step import FixedStepEventInput

RUN_ID = UUID("00000000-0000-4000-8000-000000008001")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000008002")
EVENT_IDS = (
    UUID("00000000-0000-4000-8000-000000008011"),
    UUID("00000000-0000-4000-8000-000000008012"),
    UUID("00000000-0000-4000-8000-000000008013"),
)
NOW = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)


def _draft(
    ordinal: int,
    *,
    run_id: UUID = RUN_ID,
    timestamp: datetime | None = None,
    source_event_id: str | None = None,
    event_type: EventType | None = None,
    payload: dict[str, object] | None = None,
    parent_ids: tuple[UUID, ...] | None = None,
    trust_label: TrustLabel = TrustLabel.SYNTHETIC_FIXTURE,
) -> NormalizedTraceEventDraft:
    return NormalizedTraceEventDraft(
        run_id=run_id,
        source_event_id=source_event_id or f"stage2-trajectory-event-{ordinal}",
        timestamp=timestamp or NOW + timedelta(seconds=ordinal - 1),
        event_type=event_type or (EventType.RUN_START if ordinal == 1 else EventType.MODEL_OUTPUT),
        phase=(EventPhase.INITIALIZATION if ordinal == 1 else EventPhase.POST_ACTION),
        payload=payload
        or (
            {"task": "Preserve verified release constraints.", "step": 1}
            if ordinal == 1
            else {"message": f"Logical message {ordinal}.", "step": ordinal - 1}
        ),
        parent_ids=(
            parent_ids
            if parent_ids is not None
            else (() if ordinal == 1 else (EVENT_IDS[ordinal - 2],))
        ),
        source_adapter="stage2-trajectory-fixture/v1",
        trust_label=trust_label,
    )


def _inputs(*, second_message: str = "Logical message 2.") -> tuple[FixedStepEventInput, ...]:
    return (
        FixedStepEventInput(
            draft=_draft(1),
            expected_event_id=EVENT_IDS[0],
            task_description=EventTextSelector(field_path="/payload/task"),
            action_step=ActionStepBinding(field_path="/payload/step"),
        ),
        FixedStepEventInput(
            draft=_draft(2, payload={"message": second_message, "step": 1}),
            expected_event_id=EVENT_IDS[1],
            logical_messages=(
                LogicalMessageBinding(
                    role=LogicalMessageRole.USER,
                    selector=EventTextSelector(field_path="/payload/message"),
                ),
            ),
            action_step=ActionStepBinding(field_path="/payload/step"),
            target_request_id="stage2-target-request-1",
        ),
        FixedStepEventInput(
            draft=_draft(3),
            expected_event_id=EVENT_IDS[2],
            logical_messages=(
                LogicalMessageBinding(
                    role=LogicalMessageRole.ASSISTANT,
                    selector=EventTextSelector(field_path="/payload/message"),
                ),
            ),
            action_step=ActionStepBinding(field_path="/payload/step"),
            target_request_id="stage2-target-request-2",
        ),
    )


def _replace(
    item: FixedStepEventInput,
    **changes: object,
) -> FixedStepEventInput:
    values = item.model_dump(mode="json", warnings=False)
    values.update(changes)
    return FixedStepEventInput.model_validate_json(canonical_json(values))


def _replace_draft(
    item: FixedStepEventInput,
    **changes: object,
) -> FixedStepEventInput:
    values = item.draft.model_dump(mode="json", warnings=False)
    values.update(changes)
    draft = NormalizedTraceEventDraft.model_validate_json(canonical_json(values))
    return _replace(item, draft=draft.model_dump(mode="json", warnings=False))


def _write(path: Path, trajectory: Stage2Trajectory) -> None:
    path.write_bytes(trajectory.canonical_bytes)


def test_builder_seals_complete_inputs_into_stable_canonical_records() -> None:
    inputs = _inputs()
    first = build_stage2_trajectory(inputs)
    second = build_stage2_trajectory(inputs)

    assert first == second
    assert first.run_id == RUN_ID
    assert first.fixture_id == PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID
    assert first.schema_version == STAGE2_TRAJECTORY_VERSION
    assert first.inputs == inputs
    assert first.inputs[0] is not inputs[0]
    assert tuple(record.ordinal for record in first.records) == (1, 2, 3)
    assert len({record.input_digest for record in first.records}) == 3
    assert len({record.record_digest for record in first.records}) == 3
    assert all(record.fixture_digest == first.fixture_digest for record in first.records)
    assert first.canonical_bytes.endswith(b"\n")
    assert b"\r" not in first.canonical_bytes
    assert all(
        canonical_json(json.loads(line)) == line for line in first.canonical_bytes.splitlines()
    )


def test_fixture_digest_is_domain_bound_to_identity_order_and_full_inputs() -> None:
    baseline = build_stage2_trajectory(_inputs())
    changed_input = build_stage2_trajectory(_inputs(second_message="Changed evidence."))
    changed_identity = build_stage2_trajectory(_inputs(), fixture_id="another-fixture/v1")

    assert baseline.fixture_digest != changed_input.fixture_digest
    assert baseline.fixture_digest != changed_identity.fixture_digest
    assert baseline.records[1].input_digest != changed_input.records[1].input_digest
    assert baseline.records[0].input_digest == changed_input.records[0].input_digest
    with pytest.raises(Stage2TrajectoryFixtureError):
        build_stage2_trajectory(tuple(reversed(_inputs())))


def test_record_and_container_are_strict_frozen_bounded_and_value_safe() -> None:
    trajectory = build_stage2_trajectory(_inputs())
    record = trajectory.records[0]

    assert STAGE2_TRAJECTORY_VERSION == "stage2-trajectory/v1"
    assert STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION == "stage2-trajectory-record/v1"
    assert "event_input=" not in repr(record)
    assert "records=" not in repr(trajectory)
    with pytest.raises(ValidationError):
        record.ordinal = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        trajectory.fixture_id = "changed/v1"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Stage2TrajectoryRecord.model_validate(
            record.model_dump(mode="python") | {"unexpected": True}
        )
    with pytest.raises(Stage2TrajectoryFixtureError) as raised:
        build_stage2_trajectory(_inputs(), fixture_id="contains spaces and secret")
    assert "secret" not in str(raised.value)


def test_builder_requires_one_coherent_closed_synthetic_trajectory() -> None:
    valid = _inputs()
    cases: tuple[object, ...] = (
        (),
        list(valid),
        (_replace(valid[0], task_description=None), *valid[1:]),
        (
            _replace_draft(valid[0], event_type=EventType.MODEL_OUTPUT.value),
            *valid[1:],
        ),
        (
            valid[0],
            _replace_draft(valid[1], run_id=str(OTHER_RUN_ID)),
            valid[2],
        ),
        (
            valid[0],
            _replace_draft(valid[1], trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT.value),
            valid[2],
        ),
        (
            valid[0],
            _replace(valid[1], expected_event_id=str(EVENT_IDS[0])),
            valid[2],
        ),
        (
            valid[0],
            _replace_draft(valid[1], source_event_id=valid[0].draft.source_event_id),
            valid[2],
        ),
        (
            valid[0],
            _replace_draft(valid[1], timestamp=(NOW - timedelta(seconds=1)).isoformat()),
            valid[2],
        ),
        (
            valid[0],
            _replace_draft(valid[1], parent_ids=[str(EVENT_IDS[2])]),
            valid[2],
        ),
        (
            valid[0],
            _replace(valid[1], target_request_id="stage2-target-request-2"),
            valid[2],
        ),
        tuple(_replace(item, target_request_id=None) for item in valid),
        (
            valid[0],
            _replace_draft(valid[1], payload={"message": "Logical message 2.", "step": 3}),
            valid[2],
        ),
        (
            valid[0],
            _replace(
                valid[1],
                logical_messages=[
                    {
                        "role": "user",
                        "selector": {"field_path": "/payload/missing", "span": None},
                    }
                ],
            ),
            valid[2],
        ),
    )

    for case in cases:
        with pytest.raises(Stage2TrajectoryFixtureError):
            build_stage2_trajectory(cast(tuple[FixedStepEventInput, ...], case))


def test_builder_requires_message_step_and_routing_coverage() -> None:
    valid = _inputs()
    no_messages = tuple(_replace(item, logical_messages=[]) for item in valid)
    no_steps = tuple(_replace(item, action_step=None) for item in valid)

    for inputs in (no_messages, no_steps):
        with pytest.raises(Stage2TrajectoryFixtureError):
            build_stage2_trajectory(inputs)


def test_builder_revalidates_exact_input_instances_and_record_digests() -> None:
    valid = _inputs()
    with pytest.raises(Stage2TrajectoryFixtureError):
        build_stage2_trajectory((cast(FixedStepEventInput, object()), *valid[1:]))

    unequal = valid[0].model_copy()
    unequal.__dict__["expected_event_id"] = str(EVENT_IDS[0])
    with pytest.raises(Stage2TrajectoryFixtureError):
        build_stage2_trajectory((unequal, *valid[1:]))

    unserializable = valid[0].model_copy()
    unserializable.__dict__["draft"] = object()
    with pytest.raises(Stage2TrajectoryFixtureError):
        build_stage2_trajectory((unserializable, *valid[1:]))

    record = build_stage2_trajectory(valid).records[0]
    for field in ("input_digest", "record_digest"):
        payload = record.model_dump(mode="json", warnings=False)
        payload[field] = "0" * 64
        with pytest.raises(ValidationError):
            Stage2TrajectoryRecord.model_validate_json(canonical_json(payload))


def test_defensive_copy_and_structural_helpers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = build_stage2_trajectory(_inputs())
    record = trajectory.records[0]

    with pytest.raises(Stage2TrajectoryFixtureError):
        trajectory_module._copy_record(object())
    unequal_input = record.event_input.model_copy()
    unequal_input.__dict__["expected_event_id"] = str(EVENT_IDS[0])
    unequal = record.model_copy(update={"event_input": unequal_input})
    with pytest.raises(Stage2TrajectoryFixtureError):
        trajectory_module._copy_record(unequal)
    malformed = record.model_copy()
    malformed.__dict__["record_digest"] = 0
    with pytest.raises(Stage2TrajectoryFixtureError):
        trajectory_module._copy_record(malformed)

    assert not trajectory_module._json_is_bounded({1: "non-string key"})
    assert trajectory_module._json_is_bounded(0.5)
    assert not trajectory_module._json_is_bounded(object())
    monkeypatch.setattr(trajectory_module, "_MAX_JSON_NODES", 2)
    assert not trajectory_module._json_is_bounded([1, 2])

    def fail_isfinite(_value: float) -> bool:
        raise RuntimeError

    monkeypatch.setattr("saliencegate.experiments.trajectory.math.isfinite", fail_isfinite)
    assert not trajectory_module._json_is_bounded(0.5)


def test_builder_sanitizes_unexpected_internal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_digest(_inputs: tuple[FixedStepEventInput, ...], *, fixture_id: str) -> str:
        del fixture_id
        raise RuntimeError("sensitive internal failure")

    monkeypatch.setattr(trajectory_module, "_fixture_digest", fail_digest)
    with pytest.raises(Stage2TrajectoryFixtureError) as raised:
        build_stage2_trajectory(_inputs())
    assert "sensitive" not in str(raised.value)


def test_builder_resolves_array_selectors_spans_and_rejects_ambiguous_paths() -> None:
    valid = _inputs()
    nested = _replace_draft(
        valid[1],
        payload={"messages": ["café"], "empty": "", "step": 1, "score": 0.5},
    )
    nested = _replace(
        nested,
        logical_messages=[
            {
                "role": "user",
                "selector": {
                    "field_path": "/payload/messages/0",
                    "span": TextSpan(start_byte=0, end_byte=5).model_dump(mode="json"),
                },
            }
        ],
    )
    assert build_stage2_trajectory((valid[0], nested, valid[2])).inputs[1] == nested

    invalid_selectors = (
        EventTextSelector(field_path="/payload/messages/00"),
        EventTextSelector(field_path="/payload/step/value"),
        EventTextSelector(field_path="/payload/empty"),
        EventTextSelector(
            field_path="/payload/messages/0",
            span=TextSpan(start_byte=0, end_byte=6),
        ),
        EventTextSelector(
            field_path="/payload/messages/0",
            span=TextSpan(start_byte=0, end_byte=4),
        ),
    )
    for selector in invalid_selectors:
        changed = _replace(
            nested,
            logical_messages=[
                {
                    "role": "user",
                    "selector": selector.model_dump(mode="json"),
                }
            ],
        )
        with pytest.raises(Stage2TrajectoryFixtureError):
            build_stage2_trajectory((valid[0], changed, valid[2]))

    duplicate_path = _replace(
        valid[1],
        logical_messages=[
            {
                "role": "user",
                "selector": {"field_path": "/payload/step", "span": None},
            }
        ],
    )
    zero_step = _replace_draft(valid[1], payload={"message": "Logical message 2.", "step": 0})
    for changed in (duplicate_path, zero_step):
        with pytest.raises(Stage2TrajectoryFixtureError):
            build_stage2_trajectory((valid[0], changed, valid[2]))


def test_loader_returns_a_closed_exact_fixture_with_external_anchors(tmp_path: Path) -> None:
    expected = build_stage2_trajectory(_inputs())
    path = tmp_path / "trajectory.jsonl"
    _write(path, expected)

    loaded = load_stage2_trajectory(
        path,
        expected_fixture_id=PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID,
        expected_fixture_digest=expected.fixture_digest,
    )
    loaded_from_text_path = load_stage2_trajectory(
        str(path), expected_fixture_digest=expected.fixture_digest
    )
    path.unlink()

    assert loaded == loaded_from_text_path == expected
    assert loaded.canonical_bytes == expected.canonical_bytes
    assert loaded.run_id == RUN_ID
    assert loaded.inputs == expected.inputs


def test_loader_defaults_to_paper_fixture_but_accepts_an_explicit_identity(
    tmp_path: Path,
) -> None:
    custom = build_stage2_trajectory(_inputs(), fixture_id="reviewed-custom-trajectory/v1")
    path = tmp_path / "custom.jsonl"
    _write(path, custom)

    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)
    assert (
        load_stage2_trajectory(
            path,
            expected_fixture_id="reviewed-custom-trajectory/v1",
            expected_fixture_digest=custom.fixture_digest,
        )
        == custom
    )
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(
            path,
            expected_fixture_id="reviewed-custom-trajectory/v1",
            expected_fixture_digest="0" * 64,
        )
    for digest in ("not-a-digest", "A" * 64):
        with pytest.raises(Stage2TrajectoryFixtureError):
            load_stage2_trajectory(path, expected_fixture_digest=digest)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path, expected_fixture_id="invalid fixture id")


@pytest.mark.parametrize(
    "contents",
    (
        b"",
        b"\n",
        b"[]\n",
        b'{"ordinal":1,"ordinal":1}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":1e999}\n',
        b"\xff\n",
        b" {}\n",
        b"{}\r\n",
        b"{}",
    ),
)
def test_loader_rejects_empty_noncanonical_or_invalid_jsonl(
    tmp_path: Path,
    contents: bytes,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(contents)

    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)


def test_loader_rejects_duplicate_keys_at_any_depth(tmp_path: Path) -> None:
    trajectory = build_stage2_trajectory(_inputs())
    first, *rest = trajectory.canonical_bytes.splitlines(keepends=True)
    duplicated = first.replace(
        b'"schema_version":"fixed-step-event-input/v1"',
        b'"schema_version":"fixed-step-event-input/v1",'
        b'"schema_version":"fixed-step-event-input/v1"',
        1,
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_bytes(duplicated + b"".join(rest))

    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)


def test_loader_rejects_tamper_reordering_missing_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    trajectory = build_stage2_trajectory(_inputs())
    path = tmp_path / "tampered.jsonl"
    first = trajectory.records[0].model_dump(mode="json", warnings=False)
    missing = dict(first)
    del missing["input_digest"]
    stale_input = json.loads(canonical_json(first))
    stale_input["event_input"]["draft"]["payload"]["task"] = "tampered"
    non_finite = canonical_json(first).replace(b'"step":1', b'"step":1e999', 1) + b"\n"
    variants = (
        canonical_json(missing) + b"\n",
        canonical_json(first | {"record_digest": "0" * 64}) + b"\n",
        canonical_json(stale_input) + b"\n",
        b"".join(reversed(trajectory.canonical_bytes.splitlines(keepends=True))),
        json.dumps(first, sort_keys=True).encode("utf-8") + b"\n",
        trajectory.canonical_bytes + b"\n",
        non_finite,
    )

    for contents in variants:
        path.write_bytes(contents)
        with pytest.raises(Stage2TrajectoryFixtureError):
            load_stage2_trajectory(path)


def test_container_rejects_forged_record_and_derived_run_id() -> None:
    trajectory = build_stage2_trajectory(_inputs())
    first = trajectory.records[0]
    forged_record = first.model_copy(update={"record_digest": "0" * 64})

    with pytest.raises(ValidationError):
        Stage2Trajectory(
            fixture_id=trajectory.fixture_id,
            fixture_digest=trajectory.fixture_digest,
            run_id=trajectory.run_id,
            records=(forged_record, *trajectory.records[1:]),
        )
    with pytest.raises(ValidationError):
        Stage2Trajectory(
            fixture_id=trajectory.fixture_id,
            fixture_digest=trajectory.fixture_digest,
            run_id=OTHER_RUN_ID,
            records=trajectory.records,
        )


def test_loader_applies_record_line_total_depth_and_node_bounds_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = build_stage2_trajectory(_inputs())
    path = tmp_path / "bounded.jsonl"
    _write(path, trajectory)

    monkeypatch.setattr(trajectory_module, "_MAX_RECORDS", 2)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)
    monkeypatch.setattr(trajectory_module, "_MAX_RECORDS", 10_000)
    monkeypatch.setattr(trajectory_module, "_MAX_LINE_BYTES", 8)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)
    monkeypatch.setattr(trajectory_module, "_MAX_LINE_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(trajectory_module, "_MAX_FIXTURE_BYTES", path.stat().st_size - 1)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)
    monkeypatch.setattr(trajectory_module, "_MAX_FIXTURE_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(trajectory_module, "_MAX_JSON_DEPTH", 1)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)
    monkeypatch.setattr(trajectory_module, "_MAX_JSON_DEPTH", 72)
    monkeypatch.setattr(trajectory_module, "_MAX_JSON_NODES", 1)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(path)


def test_loader_requires_one_regular_single_link_stable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = build_stage2_trajectory(_inputs())
    source = tmp_path / "trajectory.jsonl"
    _write(source, trajectory)
    symbolic = tmp_path / "symbolic.jsonl"
    symbolic.symlink_to(source)
    hard = tmp_path / "hard.jsonl"
    hard.hardlink_to(source)

    for path in (tmp_path, symbolic, hard):
        with pytest.raises(Stage2TrajectoryFixtureError):
            load_stage2_trajectory(path)
    hard.unlink()

    original_stat = os.stat

    def changed_stat(path: object, *, follow_symlinks: bool = True) -> SimpleNamespace:
        value = original_stat(cast(str | Path, path), follow_symlinks=follow_symlinks)
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_nlink=value.st_nlink,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr("saliencegate.experiments.trajectory.os.stat", changed_stat)
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(source)


def test_loader_rejects_wrong_missing_and_non_text_path_values(tmp_path: Path) -> None:
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(cast(Path, object()))
    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(tmp_path / "missing.jsonl")

    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"secret-trajectory-path"

    with pytest.raises(Stage2TrajectoryFixtureError) as raised:
        load_stage2_trajectory(cast(Path, BytesPath()))
    assert "secret" not in str(raised.value)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_loader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "trajectory.fifo"
    os.mkfifo(fifo)

    with pytest.raises(Stage2TrajectoryFixtureError):
        load_stage2_trajectory(fifo)
