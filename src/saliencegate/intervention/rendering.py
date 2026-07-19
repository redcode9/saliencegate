from __future__ import annotations

from contextlib import suppress
from typing import Annotated, ClassVar, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import EvidenceReference, InterventionClaim, TrustLabel
from saliencegate.runtime.token_counting import (
    DeterministicTokenCounter,
)

from .claims import ClaimInputError, grounded_claim_text
from .versions import (
    FIXED_ASCII_RENDERER_VERSION,
    GROUNDING_PIPELINE_VERSION,
    TOKEN_COUNTER_VERSION,
)

_MAX_CLAIMS = 2
_MAX_EVIDENCE_BYTES = 1_024
_MAX_OUTPUT_BYTES = 4_096
_MAX_TOKEN_EQUIVALENTS = 1_024
_SAFE_EVIDENCE_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,_-/"
)


class RenderingInputError(ValueError):
    """A sanitized failure at the deterministic rendering boundary."""

    _MESSAGES: ClassVar[dict[str, str]] = {
        "claim": "reminder claim input failed validation",
        "configuration": "rendering configuration failed validation",
        "evidence_bytes": "reminder evidence byte limit exceeded",
        "output_bytes": "reminder output byte limit exceeded",
        "output_tokens": "reminder output token limit exceeded",
        "text": "rendering text input failed validation",
    }

    def __init__(self, reason: str = "text") -> None:
        super().__init__(self._MESSAGES.get(reason, self._MESSAGES["text"]))


def _raise_rendering_error(reason: str) -> Never:
    raise RenderingInputError(reason) from None


def _exact_utf8_text(value: object) -> tuple[str, bytes] | None:
    if type(value) is not str:
        return None
    assert isinstance(value, str)
    encoded: bytes | None = None
    with suppress(UnicodeEncodeError):
        encoded = value.encode("utf-8", errors="strict")
    if encoded is None:
        return None
    return value, encoded


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class RenderingConfig(_FrozenModel):
    """A complete renderer configuration with immutable deterministic hard caps."""

    schema_version: Literal["1.0"]
    renderer_version: Literal["fixed-ascii/v1"]
    token_counter_version: Literal["utf8-bytes-ceil-div-4-v1"]
    max_claims: Annotated[int, Field(ge=1, le=_MAX_CLAIMS)]
    max_evidence_bytes: Annotated[int, Field(ge=1, le=_MAX_EVIDENCE_BYTES)]
    max_output_bytes: Annotated[int, Field(ge=1, le=_MAX_OUTPUT_BYTES)]
    max_token_equivalents: Annotated[int, Field(ge=1, le=_MAX_TOKEN_EQUIVALENTS)]
    include_provenance: bool


class GroundedClaim(_FrozenModel):
    """A domain claim paired with the exact resolved source text and its origin."""

    claim: InterventionClaim
    source_text: str = Field(repr=False)
    origin_trust_label: TrustLabel

    @field_validator("source_text", mode="before")
    @classmethod
    def source_is_exact_utf8(cls, value: object) -> object:
        validated = _exact_utf8_text(value)
        if validated is None:
            raise ValueError("source text must be exact UTF-8 text") from None
        text, _encoded = validated
        return text

    @model_validator(mode="after")
    def fields_are_source_derived(self) -> Self:
        try:
            selected = grounded_claim_text(self.claim)
        except ClaimInputError:
            raise ValueError("claim source-derived field validation failed") from None
        if selected != self.source_text:
            raise ValueError("claim source-derived field does not match source text")
        return self

    @property
    def pipeline_version(self) -> Literal["grounding-pipeline/v1"]:
        return GROUNDING_PIPELINE_VERSION


class _RenderedReminder(_FrozenModel):
    """Internal immutable measurements for a successfully rendered reminder."""

    renderer_version: Literal["fixed-ascii/v1"]
    text: str = Field(repr=False)
    utf8_bytes: Annotated[int, Field(ge=1, le=_MAX_OUTPUT_BYTES)]
    token_equivalents: Annotated[int, Field(ge=1, le=_MAX_TOKEN_EQUIVALENTS)]


def _escaped_ascii(encoded: bytes) -> str:
    return "".join(
        chr(value) if value in _SAFE_EVIDENCE_BYTES else f"\\x{value:02x}" for value in encoded
    )


def quote_untrusted_evidence(text: str) -> str:
    """Quote UTF-8 evidence in a canonical printable-ASCII byte escape language."""

    validated = _exact_utf8_text(text)
    if validated is None:
        _raise_rendering_error("text")
    _text, encoded = validated
    return f'"{_escaped_ascii(encoded)}"'


def _validated_configuration(value: object) -> RenderingConfig | None:
    if type(value) is not RenderingConfig:
        return None
    assert isinstance(value, RenderingConfig)
    validated: RenderingConfig | None = None
    with suppress(Exception):
        validated = RenderingConfig.model_validate(
            {
                "schema_version": value.schema_version,
                "renderer_version": value.renderer_version,
                "token_counter_version": value.token_counter_version,
                "max_claims": value.max_claims,
                "max_evidence_bytes": value.max_evidence_bytes,
                "max_output_bytes": value.max_output_bytes,
                "max_token_equivalents": value.max_token_equivalents,
                "include_provenance": value.include_provenance,
            }
        )
    return validated


def _validated_grounded_claim(value: object) -> GroundedClaim | None:
    if type(value) is not GroundedClaim:
        return None
    assert isinstance(value, GroundedClaim)
    validated: GroundedClaim | None = None
    with suppress(Exception):
        validated = GroundedClaim(
            claim=value.claim,
            source_text=value.source_text,
            origin_trust_label=value.origin_trust_label,
        )
    return validated


def _provenance_lines(index: int, evidence: EvidenceReference) -> list[str] | None:
    source_id: str | None = None
    field_path: str | None = None
    with suppress(Exception):
        source_id = _escaped_ascii(str(evidence.source_id).encode("ascii"))
        field_path = _escaped_ascii(evidence.field_path.encode("utf-8", errors="strict"))
    if source_id is None or field_path is None:
        return None

    prefix = f"claim.{index}.provenance"
    lines = [
        f"{prefix}.source={evidence.source.value}",
        f"{prefix}.source_id={source_id}",
    ]
    if evidence.revision is not None:
        lines.append(f"{prefix}.revision={evidence.revision}")
    lines.append(f"{prefix}.field_path={field_path}")
    if evidence.span is not None:
        lines.extend(
            (
                f"{prefix}.span.start_byte={evidence.span.start_byte}",
                f"{prefix}.span.end_byte={evidence.span.end_byte}",
            )
        )
    return lines


class DeterministicReminderRenderer:
    """Render grounded source data with fixed non-authoritative ASCII framing."""

    __slots__ = ("_configuration", "_counter")

    def __init__(self, configuration: RenderingConfig) -> None:
        validated = _validated_configuration(configuration)
        if validated is None:
            _raise_rendering_error("configuration")
        self._configuration = validated
        self._counter = DeterministicTokenCounter()

    @property
    def configuration(self) -> RenderingConfig:
        return self._configuration

    @property
    def renderer_version(self) -> Literal["fixed-ascii/v1"]:
        return FIXED_ASCII_RENDERER_VERSION

    @property
    def token_counter_version(self) -> Literal["utf8-bytes-ceil-div-4-v1"]:
        return TOKEN_COUNTER_VERSION

    def render(self, claims: tuple[GroundedClaim, ...]) -> str:
        if type(claims) is not tuple or not 1 <= len(claims) <= self._configuration.max_claims:
            _raise_rendering_error("claim")
        if len(claims) > _MAX_CLAIMS:
            _raise_rendering_error("claim")

        candidates = tuple(_validated_grounded_claim(claim) for claim in claims)
        if any(candidate is None for candidate in candidates):
            _raise_rendering_error("claim")
        validated = tuple(candidate for candidate in candidates if candidate is not None)
        lines = [
            f"[SALIENCEGATE_REMINDER {FIXED_ASCII_RENDERER_VERSION}]",
            "authority=none",
            "reason=grounded_reminder",
            "ttl_steps=1",
        ]
        for index, item in enumerate(validated, start=1):
            evidence_size = None
            with suppress(Exception):
                evidence_size = self._counter.measure(item.source_text)
            if evidence_size is None:
                _raise_rendering_error("claim")
            if evidence_size.utf8_bytes > self._configuration.max_evidence_bytes:
                _raise_rendering_error("evidence_bytes")

            evidence = item.claim.evidence[0]
            lines.extend(
                (
                    f"claim.{index}.kind={item.claim.kind.value}",
                    f"claim.{index}.origin={item.origin_trust_label.value}",
                    f"claim.{index}.evidence={quote_untrusted_evidence(item.source_text)}",
                )
            )
            if self._configuration.include_provenance:
                provenance = _provenance_lines(index, evidence)
                if provenance is None:
                    _raise_rendering_error("claim")
                lines.extend(provenance)

        lines.append("[/SALIENCEGATE_REMINDER]")
        text = "\n".join(lines)
        output_size = None
        with suppress(Exception):
            output_size = self._counter.measure(text)
        if output_size is None:
            _raise_rendering_error("text")
        if output_size.utf8_bytes > self._configuration.max_output_bytes:
            _raise_rendering_error("output_bytes")
        if output_size.approximate_tokens > self._configuration.max_token_equivalents:
            _raise_rendering_error("output_tokens")

        rendered = _RenderedReminder(
            renderer_version=FIXED_ASCII_RENDERER_VERSION,
            text=text,
            utf8_bytes=output_size.utf8_bytes,
            token_equivalents=output_size.approximate_tokens,
        )
        return rendered.text


__all__ = [
    "FIXED_ASCII_RENDERER_VERSION",
    "GROUNDING_PIPELINE_VERSION",
    "DeterministicReminderRenderer",
    "GroundedClaim",
    "RenderingConfig",
    "RenderingInputError",
    "quote_untrusted_evidence",
]
