from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from saliencegate.domain.enums import DeliveryTarget

DigestPart = str | bytes


def new_repository_id() -> UUID:
    """Return a random RFC 4122 version-4 identifier for repository-owned records."""

    return uuid4()


def length_prefixed_sha256(
    *parts: DigestPart,
    domain: DigestPart = "saliencegate:digest:v1",
) -> str:
    """Hash framed byte strings so adjacent components cannot alias one another."""

    digest = hashlib.sha256()
    for part in (domain, *parts):
        encoded = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def content_digest(*parts: DigestPart) -> str:
    return length_prefixed_sha256(*parts, domain="saliencegate:content:v1")


def delivery_id(
    run_id: UUID,
    cycle_identifier: str,
    intervention_id: UUID,
    target_request_id: str,
    target: DeliveryTarget,
    adapter_id: str,
    adapter_capabilities_digest: str,
    rendered_text_digest: str,
) -> UUID:
    """Derive one stable UUID4 delivery identity without consuming event IDs."""

    if type(run_id) is not UUID or run_id.version != 4:
        raise ValueError("run_id must be an exact UUID4 value")
    if type(intervention_id) is not UUID or intervention_id.version != 4:
        raise ValueError("intervention_id must be an exact UUID4 value")
    for label, value in (
        ("cycle_identifier", cycle_identifier),
        ("target_request_id", target_request_id),
        ("adapter_id", adapter_id),
    ):
        if type(value) is not str or not value:
            raise ValueError(f"{label} must be a non-empty exact string")
    for label, value in (
        ("cycle_identifier", cycle_identifier),
        ("adapter_capabilities_digest", adapter_capabilities_digest),
        ("rendered_text_digest", rendered_text_digest),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if type(target) is not DeliveryTarget:
        raise ValueError("target must be an exact delivery target")
    digest = length_prefixed_sha256(
        str(run_id),
        cycle_identifier,
        str(intervention_id),
        target_request_id,
        target.value,
        adapter_id,
        adapter_capabilities_digest,
        rendered_text_digest,
        domain="saliencegate:delivery:v1",
    )
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def cycle_id(
    run_id: UUID,
    first_event_sequence: int,
    last_event_sequence: int,
    policy_version: str,
    configuration_digest: str,
    grounding_version: str,
    grounding_configuration_digest: str,
    requested_delivery_target: DeliveryTarget | None,
) -> str:
    if type(run_id) is not UUID or run_id.version != 4:
        raise ValueError("run_id must be an exact UUID4 value")
    run_id = UUID(int=run_id.int)
    if type(first_event_sequence) is not int or first_event_sequence < 1:
        raise ValueError("first_event_sequence must be positive")
    if type(last_event_sequence) is not int or last_event_sequence < first_event_sequence:
        raise ValueError("last_event_sequence must not precede first_event_sequence")
    if type(policy_version) is not str or not policy_version:
        raise ValueError("policy_version must be a non-empty exact string")
    if (
        type(configuration_digest) is not str
        or len(configuration_digest) != 64
        or any(character not in "0123456789abcdef" for character in configuration_digest)
    ):
        raise ValueError("configuration_digest must be a lowercase SHA-256 digest")
    if type(grounding_version) is not str or not grounding_version:
        raise ValueError("grounding_version must be a non-empty exact string")
    if (
        type(grounding_configuration_digest) is not str
        or len(grounding_configuration_digest) != 64
        or any(character not in "0123456789abcdef" for character in grounding_configuration_digest)
    ):
        raise ValueError("grounding_configuration_digest must be a lowercase SHA-256 digest")
    if (
        requested_delivery_target is not None
        and type(requested_delivery_target) is not DeliveryTarget
    ):
        raise ValueError("requested_delivery_target must be an exact delivery target or None")
    target_identity = (
        "none" if requested_delivery_target is None else f"target:{requested_delivery_target.value}"
    )
    return length_prefixed_sha256(
        str(run_id),
        str(first_event_sequence),
        str(last_event_sequence),
        policy_version,
        configuration_digest,
        grounding_version,
        grounding_configuration_digest,
        target_identity,
        domain="saliencegate:cycle:v2",
    )
