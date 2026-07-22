from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.capture.store_support import (
    INSTALLATION_KEY,
    authenticated_intake,
    initialized_store,
    register_connection,
)

from saliencegate.capture.sessions import verify_capture_session_snapshot
from saliencegate.capture.store import CaptureAppendDisposition
from saliencegate.domain import canonical_json
from saliencegate.integrations.hook import CaptureHookError, read_capture_hook_document

_SAFE_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        blacklist_characters=("\\", '"'),
    ),
    max_size=32,
)
_JSON_VALUE = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(1 << 31), max_value=(1 << 31) - 1)
    | _SAFE_TEXT,
    lambda children: (
        st.lists(children, max_size=6) | st.dictionaries(_SAFE_TEXT, children, max_size=6)
    ),
    max_leaves=32,
)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(document=st.dictionaries(_SAFE_TEXT, _JSON_VALUE, max_size=12))
def test_bounded_hook_parser_round_trips_canonical_json_objects(
    document: dict[str, object],
) -> None:
    encoded = canonical_json(document)

    parsed = read_capture_hook_document(BytesIO(encoded))

    assert parsed == encoded
    assert json.loads(parsed) == document


@settings(max_examples=100, deadline=None, derandomize=True)
@given(source=st.binary(min_size=0, max_size=2_048))
def test_adversarial_hook_parser_is_total_and_content_free(source: bytes) -> None:
    try:
        parsed = read_capture_hook_document(BytesIO(source))
    except CaptureHookError as error:
        assert str(error) == "capture hook failed"
        assert repr(error) == "CaptureHookError('capture hook failed')"
    else:
        assert parsed == source
        assert type(json.loads(parsed)) is dict


@settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(
    session_native=st.binary(min_size=1, max_size=32),
    producer_indices=st.lists(
        st.integers(min_value=2, max_value=10_000),
        min_size=1,
        max_size=8,
        unique=True,
    ).map(sorted),
)
def test_authenticated_store_replays_are_idempotent_and_keep_a_contiguous_chain(
    tmp_path: Path,
    session_native: bytes,
    producer_indices: list[int],
) -> None:
    with tempfile.TemporaryDirectory(dir=tmp_path) as raw:
        database = Path(raw) / "capture.sqlite3"
        with initialized_store(database) as store:
            register_connection(store)
            first = authenticated_intake(
                "session_started",
                session_native=session_native,
                producer_index=1,
            )
            assert store.append(first).disposition is CaptureAppendDisposition.ADMITTED
            intakes = tuple(
                authenticated_intake(
                    "turn_finished",
                    session_native=session_native,
                    producer_index=producer_index,
                )
                for producer_index in producer_indices
            )
            for intake in intakes:
                assert store.append(intake).disposition is CaptureAppendDisposition.ADMITTED
            for intake in reversed(intakes):
                assert store.append(intake).disposition is CaptureAppendDisposition.REPLAYED
            assert store.append(first).disposition is CaptureAppendDisposition.REPLAYED
            snapshot = verify_capture_session_snapshot(
                store.snapshot_session(first.connection_id, first.session_id),
                installation_key=INSTALLATION_KEY,
            )

    expected_count = len(intakes) + 1
    assert snapshot.event_count == expected_count
    assert tuple(item.receipt_ordinal for item in snapshot.events) == tuple(
        range(1, expected_count + 1)
    )
    assert tuple(item.event.intake.producer_event_digest for item in snapshot.events) == (
        first.producer_event_digest,
        *(intake.producer_event_digest for intake in intakes),
    )
