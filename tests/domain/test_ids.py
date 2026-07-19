from __future__ import annotations

from uuid import RFC_4122, UUID

import pytest

from saliencegate.domain import (
    DeliveryTarget,
    content_digest,
    cycle_id,
    delivery_id,
    length_prefixed_sha256,
    new_repository_id,
)

GROUNDING_VERSION = "grounding-pipeline/v1"
GROUNDING_DIGEST = "b" * 64
TARGET = DeliveryTarget.NEXT_MODEL_CALL


def test_length_prefix_prevents_concatenation_aliases() -> None:
    assert length_prefixed_sha256("ab", "c") != length_prefixed_sha256("a", "bc")
    assert length_prefixed_sha256("", "abc") != length_prefixed_sha256("abc")
    assert length_prefixed_sha256("a", "b") != length_prefixed_sha256("b", "a")


def test_length_prefix_uses_utf8_bytes_and_has_a_golden_vector() -> None:
    assert (
        length_prefixed_sha256("é", "", domain="saliencegate:test:v1")
        == "7543279a0e938f35b17dbcc76def62ef760819912e732933539c818e86577247"
    )


def test_deterministic_ids_are_domain_separated() -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    configuration_digest = "a" * 64
    assert cycle_id(
        run_id,
        1,
        2,
        "policy/1",
        configuration_digest,
        GROUNDING_VERSION,
        GROUNDING_DIGEST,
        TARGET,
    ) != content_digest(str(run_id), "1", "2", "policy/1")


def test_repository_ids_are_uuid4() -> None:
    first = new_repository_id()
    second = new_repository_id()

    assert first.version == 4
    assert first.variant == RFC_4122
    assert first != second


def test_delivery_id_is_stable_uuid4_and_changes_with_its_binding() -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    intervention_id = UUID("00000000-0000-4000-8000-000000000002")
    values = (
        run_id,
        "a" * 64,
        intervention_id,
        "request-1",
        DeliveryTarget.NEXT_MODEL_CALL,
        "adapter/1",
        "b" * 64,
        "c" * 64,
    )

    first = delivery_id(*values)
    second = delivery_id(*values)
    changed = delivery_id(*values[:-1], "d" * 64)

    assert first == second
    assert first != changed
    assert first.version == 4
    assert first.variant == RFC_4122


@pytest.mark.parametrize(
    "changes",
    (
        {"run_id": UUID(int=1)},
        {"cycle_identifier": "short"},
        {"intervention_id": UUID(int=2)},
        {"target_request_id": ""},
        {"target": "next_model_call"},
        {"adapter_id": ""},
        {"adapter_capabilities_digest": "B" * 64},
        {"rendered_text_digest": "C" * 64},
    ),
)
def test_delivery_id_rejects_invalid_identity_inputs(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "run_id": UUID("00000000-0000-4000-8000-000000000001"),
        "cycle_identifier": "a" * 64,
        "intervention_id": UUID("00000000-0000-4000-8000-000000000002"),
        "target_request_id": "request-1",
        "target": DeliveryTarget.NEXT_MODEL_CALL,
        "adapter_id": "adapter/1",
        "adapter_capabilities_digest": "b" * 64,
        "rendered_text_digest": "c" * 64,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        delivery_id(**values)  # type: ignore[arg-type]


def test_cycle_id_rejects_invalid_event_ranges() -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    with pytest.raises(ValueError, match="positive"):
        cycle_id(
            run_id,
            0,
            1,
            "policy/1",
            "a" * 64,
            GROUNDING_VERSION,
            GROUNDING_DIGEST,
            TARGET,
        )
    with pytest.raises(ValueError, match="precede"):
        cycle_id(
            run_id,
            2,
            1,
            "policy/1",
            "a" * 64,
            GROUNDING_VERSION,
            GROUNDING_DIGEST,
            TARGET,
        )


def test_cycle_id_changes_with_policy_configuration() -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    first = cycle_id(
        run_id,
        1,
        2,
        "policy/1",
        "a" * 64,
        GROUNDING_VERSION,
        GROUNDING_DIGEST,
        TARGET,
    )
    second = cycle_id(
        run_id,
        1,
        2,
        "policy/1",
        "b" * 64,
        GROUNDING_VERSION,
        GROUNDING_DIGEST,
        TARGET,
    )
    assert first != second


@pytest.mark.parametrize(
    ("grounding_version", "grounding_digest", "target"),
    [
        ("grounding-pipeline/v2", GROUNDING_DIGEST, TARGET),
        (GROUNDING_VERSION, "c" * 64, TARGET),
        (GROUNDING_VERSION, GROUNDING_DIGEST, DeliveryTarget.PRE_ACTION_REPLAN),
        (GROUNDING_VERSION, GROUNDING_DIGEST, None),
    ],
)
def test_cycle_id_changes_with_the_pre_model_grounding_pin(
    grounding_version: str,
    grounding_digest: str,
    target: DeliveryTarget | None,
) -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    baseline = cycle_id(
        run_id,
        1,
        2,
        "policy/1",
        "a" * 64,
        GROUNDING_VERSION,
        GROUNDING_DIGEST,
        TARGET,
    )

    assert baseline != cycle_id(
        run_id,
        1,
        2,
        "policy/1",
        "a" * 64,
        grounding_version,
        grounding_digest,
        target,
    )


@pytest.mark.parametrize(
    ("policy_version", "configuration_digest"),
    [("", "a" * 64), ("policy/1", "A" * 64), ("policy/1", "short")],
)
def test_cycle_id_rejects_invalid_identity_fields(
    policy_version: str,
    configuration_digest: str,
) -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    with pytest.raises(ValueError):
        cycle_id(
            run_id,
            1,
            2,
            policy_version,
            configuration_digest,
            GROUNDING_VERSION,
            GROUNDING_DIGEST,
            TARGET,
        )


@pytest.mark.parametrize(
    ("grounding_version", "grounding_digest", "target"),
    [
        ("", GROUNDING_DIGEST, TARGET),
        (GROUNDING_VERSION, "A" * 64, TARGET),
        (GROUNDING_VERSION, "short", TARGET),
        (GROUNDING_VERSION, GROUNDING_DIGEST, "next_model_call"),
    ],
)
def test_cycle_id_rejects_invalid_grounding_identity_fields(
    grounding_version: str,
    grounding_digest: str,
    target: object,
) -> None:
    run_id = UUID("00000000-0000-4000-8000-000000000001")
    with pytest.raises(ValueError):
        cycle_id(
            run_id,
            1,
            2,
            "policy/1",
            "a" * 64,
            grounding_version,
            grounding_digest,
            target,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("first", "last", "policy", "policy_digest", "grounding_digest"),
    (
        (True, 2, "policy/1", "a" * 64, GROUNDING_DIGEST),
        (1, True, "policy/1", "a" * 64, GROUNDING_DIGEST),
        (1, 2, 1, "a" * 64, GROUNDING_DIGEST),
        (1, 2, "policy/1", b"a" * 64, GROUNDING_DIGEST),
        (1, 2, "policy/1", "a" * 64, b"b" * 64),
    ),
)
def test_cycle_id_rejects_inexact_public_helper_types(
    first: object,
    last: object,
    policy: object,
    policy_digest: object,
    grounding_digest: object,
) -> None:
    with pytest.raises(ValueError):
        cycle_id(
            UUID("00000000-0000-4000-8000-000000000001"),
            first,  # type: ignore[arg-type]
            last,  # type: ignore[arg-type]
            policy,  # type: ignore[arg-type]
            policy_digest,  # type: ignore[arg-type]
            GROUNDING_VERSION,
            grounding_digest,  # type: ignore[arg-type]
            TARGET,
        )
