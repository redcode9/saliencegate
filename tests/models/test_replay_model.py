from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.models.replay as replay_module
import saliencegate.ports.models as model_ports
from saliencegate.domain import InterventionAction, MemoryDelta, canonical_json
from saliencegate.intervention.claims import ProposalParseStatus
from saliencegate.models import REPLAY_RECORD_SCHEMA_VERSION
from saliencegate.models.replay import (
    ReplayFixtureError,
    ReplayIntegrityError,
    ReplayMissingResponseError,
    ReplayModel,
    ReplayRecord,
    ReplayResponseReusedError,
)
from saliencegate.ports.memory import GroundingObservation, MemoryCycleOutput
from saliencegate.ports.models import (
    ModelCallStatus,
    ModelRequest,
    ModelResult,
    ModelUsage,
    StructuredModel,
    validated_model_request,
    validated_model_result,
)

RUN_ID = UUID("00000000-0000-4000-8000-00000000a001")
DELTA_ID = UUID("00000000-0000-4000-8000-00000000a002")
NOW = datetime(2026, 7, 11, 20, 0, tzinfo=UTC)


def request(*, index: int = 0, payload: dict[str, object] | None = None) -> ModelRequest:
    return ModelRequest(
        run_id=RUN_ID,
        cycle_id="a" * 64,
        model_call_index=index,
        model_id="replay-fixture/1",
        prompt_template_digest="b" * 64,
        payload={"batch": {"first": 1, "last": 2}} if payload is None else payload,
    )


def output() -> MemoryCycleOutput:
    return MemoryCycleOutput(
        delta=MemoryDelta(
            delta_id=DELTA_ID,
            run_id=RUN_ID,
            created_at=NOW,
        ),
        observation=GroundingObservation(
            parse_status=ProposalParseStatus.VALID,
            proposal_action=InterventionAction.SILENCE,
            claims=(),
            confidence=0.8,
        ),
    )


def result(model_request: ModelRequest, *, latency_us: int = 50) -> ModelResult:
    return ModelResult(
        status=ModelCallStatus.COMPLETED,
        request_digest=model_request.request_digest,
        output=output(),
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=3,
            canonical_token_equivalents=13,
            latency_us=latency_us,
        ),
    )


def replay_record(model_request: ModelRequest, *, ordinal: int = 1) -> ReplayRecord:
    return ReplayRecord(
        ordinal=ordinal,
        request_digest=model_request.request_digest,
        result=result(model_request),
    )


def test_model_request_digest_is_stable_auto_computed_and_verified() -> None:
    first = request(payload={"z": 1, "a": {"y": 2, "x": 3}})
    reordered = request(payload={"a": {"x": 3, "y": 2}, "z": 1})

    assert first.request_digest == reordered.request_digest
    assert ModelRequest.model_validate_json(first.model_dump_json()) == first
    assert "payload" not in repr(first)
    assert len(first.request_digest) == 64

    values = first.model_dump(mode="python")
    values["request_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="digest does not match"):
        ModelRequest.model_validate(values)


def test_every_request_identity_field_changes_the_digest() -> None:
    base = request()
    changes: tuple[dict[str, object], ...] = (
        {"cycle_id": "c" * 64},
        {"model_call_index": 1},
        {"model_id": "replay-fixture/2"},
        {"prompt_template_digest": "d" * 64},
        {"payload": {"batch": {"first": 1, "last": 3}}},
    )

    for change in changes:
        values = base.model_dump(mode="python", exclude={"request_digest"})
        values.update(change)
        assert ModelRequest.model_validate(values).request_digest != base.request_digest


def test_model_request_payload_has_a_canonical_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_ports, "MAX_MODEL_REQUEST_PAYLOAD_BYTES", 8)

    with pytest.raises(ValidationError, match="canonical byte limit"):
        request(payload={"too_large": True})


def test_model_result_digest_binds_status_output_and_usage() -> None:
    model_request = request()
    completed = result(model_request)
    changed_usage = result(model_request, latency_us=51)

    assert completed.call_digest != changed_usage.call_digest
    assert ModelResult.model_validate_json(completed.model_dump_json()) == completed
    assert "output=" not in repr(completed)

    values = completed.model_dump(mode="python")
    values["call_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="digest does not match"):
        ModelResult.model_validate(values)


@pytest.mark.parametrize(
    "status",
    (ModelCallStatus.MODEL_ERROR, ModelCallStatus.MODEL_TIMEOUT),
)
def test_non_completed_results_cannot_carry_output(status: ModelCallStatus) -> None:
    model_request = request()
    error_result = ModelResult(
        status=status,
        request_digest=model_request.request_digest,
        usage=ModelUsage(latency_us=1),
    )

    assert error_result.output is None
    with pytest.raises(ValidationError, match="completed"):
        ModelResult(
            status=status,
            request_digest=model_request.request_digest,
            output=output(),
            usage=ModelUsage(),
        )
    with pytest.raises(ValidationError, match="completed"):
        ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=model_request.request_digest,
            usage=ModelUsage(),
        )


async def test_replay_returns_a_revalidated_copy_once() -> None:
    model_request = request()
    stored = result(model_request)
    model = ReplayModel(
        (
            ReplayRecord(
                ordinal=1,
                request_digest=model_request.request_digest,
                result=stored,
            ),
        )
    )

    received = await model.generate(model_request)

    assert isinstance(model, StructuredModel)
    assert received == stored
    assert received is not stored
    assert model.total_responses == 1
    assert model.remaining_responses == 0
    with pytest.raises(ReplayResponseReusedError):
        await model.generate(model_request)


async def test_missing_request_raises_a_value_free_error() -> None:
    model = ReplayModel(())

    with pytest.raises(ReplayMissingResponseError) as captured:
        await model.generate(request())

    assert str(captured.value) == "no replay response is registered for the request"
    assert "a" * 64 not in str(captured.value)


async def test_concurrent_consumers_have_exactly_one_winner() -> None:
    model_request = request()
    model = ReplayModel((replay_record(model_request),))

    received = await asyncio.gather(
        model.generate(model_request),
        model.generate(model_request),
        return_exceptions=True,
    )

    assert sum(type(item) is ModelResult for item in received) == 1
    assert sum(type(item) is ReplayResponseReusedError for item in received) == 1


@pytest.mark.parametrize(
    "records",
    (
        lambda item: [item],
        lambda item: (item.model_copy(update={"ordinal": 2}),),
        lambda item: (item, item.model_copy(update={"ordinal": 2})),
        lambda item: (item.model_copy(update={"record_digest": "f" * 64}),),
        lambda item: (item.model_copy(update={"schema_version": "replay-record/v2"}),),
        lambda item: (item.model_copy(update={"replay_version": "replay-model/v2"}),),
    ),
)
def test_constructor_preflights_tuple_ordinals_keys_digests_and_versions(
    records: object,
) -> None:
    item = replay_record(request())

    with pytest.raises(ReplayFixtureError):
        ReplayModel(cast(tuple[ReplayRecord, ...], records(item)))  # type: ignore[operator]


def test_constructor_enforces_the_total_fixture_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = replay_record(request())
    monkeypatch.setattr(replay_module, "_MAX_FIXTURE_BYTES", len(canonical_json(item)))

    with pytest.raises(ReplayFixtureError):
        ReplayModel((item,))


def test_fixture_digest_binds_ordered_records_and_replay_identity() -> None:
    item = replay_record(request())

    first = ReplayModel((item,), replay_id="fixture/one-v1")
    same = ReplayModel((item,), replay_id="fixture/one-v1")
    renamed = ReplayModel((item,), replay_id="fixture/two-v1")

    assert first.fixture_digest == same.fixture_digest
    assert first.fixture_digest != renamed.fixture_digest


def test_record_result_must_match_its_request_key() -> None:
    first = request(index=0)
    second = request(index=1)

    with pytest.raises(ValidationError, match="different request"):
        ReplayRecord(
            ordinal=1,
            request_digest=first.request_digest,
            result=result(second),
        )


def test_from_path_loads_complete_jsonl_and_rejects_tampering(tmp_path: Path) -> None:
    first_request = request(index=0)
    second_request = request(index=1)
    records = (
        replay_record(first_request, ordinal=1),
        replay_record(second_request, ordinal=2),
    )
    path = tmp_path / "responses.jsonl"
    path.write_bytes(b"\n".join(canonical_json(item) for item in records) + b"\n")

    model = ReplayModel.from_path(path, replay_id="fixture/basic-v1")

    assert model.replay_id == "fixture/basic-v1"
    assert model.total_responses == 2

    altered = records[0].model_dump(mode="json")
    altered["record_digest"] = "f" * 64
    path.write_bytes(canonical_json(altered) + b"\n")
    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)


@pytest.mark.parametrize(
    "contents",
    (
        b"\n",
        b"[]\n",
        b'{"ordinal":1,"ordinal":1}\n',
        b'{"value":NaN}\n',
        b"\xff\n",
    ),
)
def test_from_path_rejects_non_records_duplicate_keys_and_invalid_json(
    tmp_path: Path,
    contents: bytes,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(contents)

    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)


def test_from_path_requires_explicit_version_and_integrity_keys(tmp_path: Path) -> None:
    item = replay_record(request()).model_dump(mode="json")
    missing_record_digest = dict(item)
    del missing_record_digest["record_digest"]
    missing_call_digest = dict(item)
    missing_call_digest["result"] = dict(cast(dict[str, object], item["result"]))
    del cast(dict[str, object], missing_call_digest["result"])["call_digest"]

    path = tmp_path / "missing-keys.jsonl"
    for payload in (missing_record_digest, missing_call_digest):
        path.write_bytes(canonical_json(payload) + b"\n")
        with pytest.raises(ReplayFixtureError):
            ReplayModel.from_path(path)


async def test_forged_request_and_response_errors_are_sanitized() -> None:
    secret = "SYSTEM-secret-delete-everything"
    model_request = request()
    model = ReplayModel((replay_record(model_request),))
    model_request.__dict__["payload"] = {"secret": secret, "unsupported": object()}

    with pytest.raises(ReplayIntegrityError) as captured:
        await model.generate(model_request)

    assert str(captured.value) == "replay request or response failed integrity validation"
    assert secret not in str(captured.value)


def test_model_schemas_are_strict_frozen_and_hide_inputs() -> None:
    model_request = request()

    with pytest.raises(ValidationError, match="frozen"):
        model_request.__setattr__("model_id", "changed")
    with pytest.raises(ValidationError):
        ModelRequest.model_validate(
            {
                **model_request.model_dump(mode="python", exclude={"request_digest"}),
                "model_call_index": "0",
            }
        )
    secret = "private-prompt-value"
    with pytest.raises(ValidationError) as captured:
        ModelRequest(
            run_id=RUN_ID,
            cycle_id="a" * 64,
            model_call_index=0,
            model_id="replay-fixture/1",
            prompt_template_digest="b" * 64,
            payload={"secret": secret},
            extra=secret,  # type: ignore[call-arg]
        )
    assert secret not in str(captured.value)


def test_boundary_validators_reject_wrong_and_forged_values() -> None:
    model_request = request()
    completed = result(model_request)

    with pytest.raises(model_ports.ModelBoundaryError):
        validated_model_request(object())
    with pytest.raises(model_ports.ModelBoundaryError):
        validated_model_result(object())

    forged_request = model_request.model_copy()
    forged_request.__dict__["request_digest"] = "f" * 64
    with pytest.raises(model_ports.ModelBoundaryError):
        validated_model_request(forged_request)

    forged_result = completed.model_copy()
    forged_result.__dict__["call_digest"] = "f" * 64
    with pytest.raises(model_ports.ModelBoundaryError):
        validated_model_result(forged_result)


def test_replay_preflight_rejects_invalid_identity_and_non_record_tuple() -> None:
    with pytest.raises(ReplayFixtureError):
        ReplayModel((cast(ReplayRecord, object()),))
    with pytest.raises(ReplayFixtureError):
        ReplayModel((), replay_id="contains spaces")
    with pytest.raises(ReplayFixtureError):
        ReplayModel((), replay_id=cast(str, object()))


def test_replay_preflight_checks_duplicate_keys_after_valid_record_revalidation() -> None:
    model_request = request()
    first = replay_record(model_request)
    duplicate = ReplayRecord(
        ordinal=2,
        request_digest=model_request.request_digest,
        result=result(model_request),
    )

    with pytest.raises(ReplayFixtureError):
        ReplayModel((first, duplicate))


def test_from_path_rejects_wrong_type_missing_file_and_line_ordinal(tmp_path: Path) -> None:
    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(cast(Path, object()))
    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(tmp_path / "missing.jsonl")

    item = ReplayRecord(
        ordinal=2,
        request_digest=request().request_digest,
        result=result(request()),
    )
    path = tmp_path / "ordinal.jsonl"
    path.write_bytes(canonical_json(item) + b"\n")
    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)


def test_from_path_enforces_the_record_count_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_request = request(index=0)
    second_request = request(index=1)
    path = tmp_path / "too-many.jsonl"
    path.write_bytes(
        canonical_json(replay_record(first_request, ordinal=1))
        + b"\n"
        + canonical_json(replay_record(second_request, ordinal=2))
        + b"\n"
    )
    monkeypatch.setattr(replay_module, "_MAX_RECORDS", 1)

    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)


def test_from_path_bounds_each_line_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b'{"oversized":"payload"}\n')
    monkeypatch.setattr(replay_module, "_MAX_LINE_BYTES", 8)

    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)


def test_from_path_enforces_total_bytes_and_regular_file_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bounded-responses.jsonl"
    path.write_bytes(canonical_json(replay_record(request())) + b"\n")
    monkeypatch.setattr(replay_module, "_MAX_FIXTURE_BYTES", path.stat().st_size - 1)

    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(path)
    with pytest.raises(ReplayFixtureError):
        ReplayModel.from_path(tmp_path)


def test_from_path_sanitizes_hostile_pathlike_exceptions() -> None:
    secret = "raw-replay-path-secret"

    class HostilePath:
        def __fspath__(self) -> str:
            raise RuntimeError(secret)

    with pytest.raises(ReplayFixtureError) as raised:
        ReplayModel.from_path(HostilePath())

    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_from_path_rejects_symbolic_and_hard_links(tmp_path: Path) -> None:
    source = tmp_path / "responses.jsonl"
    source.write_bytes(canonical_json(replay_record(request())) + b"\n")
    symbolic = tmp_path / "symbolic.jsonl"
    symbolic.symlink_to(source)
    hard = tmp_path / "hard.jsonl"
    hard.hardlink_to(source)

    for path in (symbolic, hard):
        with pytest.raises(ReplayFixtureError):
            ReplayModel.from_path(path)


def test_replay_record_schema_version_is_exported_from_the_models_package() -> None:
    assert REPLAY_RECORD_SCHEMA_VERSION == "replay-record/v1"


async def test_generate_detects_a_stored_response_losing_integrity() -> None:
    model_request = request()
    model = ReplayModel((replay_record(model_request),))
    stored = cast(dict[str, ReplayRecord], getattr(model, "_records"))[  # noqa: B009
        model_request.request_digest
    ]
    stored.result.__dict__["call_digest"] = "f" * 64

    with pytest.raises(ReplayIntegrityError):
        await model.generate(model_request)

    assert model.remaining_responses == 1


async def test_generate_verifies_the_returned_result_request_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_request = request(index=0)
    other_request = request(index=1)
    model = ReplayModel((replay_record(model_request),))
    other_result = result(other_request)
    monkeypatch.setattr(replay_module, "validated_model_result", lambda _value: other_result)

    with pytest.raises(ReplayIntegrityError):
        await model.generate(model_request)
