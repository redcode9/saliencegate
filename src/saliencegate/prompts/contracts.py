from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from saliencegate.domain import (
    MAX_MEMORY_PROVENANCE_ITEMS,
    EvidenceReference,
    JsonObject,
    MemoryKind,
    MemoryRecord,
    PayloadDigest,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    NonNegativeInt,
    PositiveInt,
    Sha256Digest,
    UnitInterval,
    UtcDatetime,
)
from saliencegate.intervention import quote_untrusted_evidence
from saliencegate.ports.model_calls import (
    MAX_STRUCTURED_CALL_PAYLOAD_BYTES,
    StructuredCallPhase,
)
from saliencegate.runtime.message_window import MessageWindow

MAX_PROMPT_PAYLOAD_BYTES = MAX_STRUCTURED_CALL_PAYLOAD_BYTES
MAX_ACTIVE_BANK_RECORDS = 4_096
PROMPT_TEXT_ENCODING: Literal["quoted-utf8-bytes/v1"] = "quoted-utf8-bytes/v1"
UNTRUSTED_PROMPT_DATA_BEGIN: Literal["<<<SALIENCEGATE_UNTRUSTED_PROMPT_DATA_V1>>>"] = (
    "<<<SALIENCEGATE_UNTRUSTED_PROMPT_DATA_V1>>>"
)
UNTRUSTED_PROMPT_DATA_END: Literal["<<<END_SALIENCEGATE_UNTRUSTED_PROMPT_DATA_V1>>>"] = (
    "<<<END_SALIENCEGATE_UNTRUSTED_PROMPT_DATA_V1>>>"
)

_QUOTED_TEXT_DIGEST_DOMAIN = "saliencegate:prompt:quoted-data-text:v1"
_MEMORY_RECORD_DIGEST_DOMAIN = "saliencegate:prompt:quoted-memory-record:v1"
_BANK_VIEW_DIGEST_DOMAIN = "saliencegate:prompt:active-bank-view:v1"
_PROMPT_TEMPLATE_DIGEST_DOMAIN = "saliencegate:prompt:template:v1"
_PROMPT_INPUT_DIGEST_DOMAIN = "saliencegate:prompt:input:v1"
_PROMPT_PAYLOAD_DIGEST_DOMAIN = "saliencegate:prompt:provider-payload:v1"
_BUILT_PROMPT_DIGEST_DOMAIN = "saliencegate:prompt:built-prompt:v1"
_PROMPT_BUNDLE_DIGEST_DOMAIN = "saliencegate:prompt:bundle:v1"


class PromptErrorCode(StrEnum):
    INVALID_BANK = "invalid_bank"
    CROSS_RUN = "cross_run"
    WRONG_BANK_VIEW = "wrong_bank_view"
    INVALID_ENVELOPE = "invalid_envelope"
    LIMIT_EXCEEDED = "limit_exceeded"


class PromptContractError(ValueError):
    """A typed, value-free failure at the provider prompt boundary."""

    def __init__(self, code: PromptErrorCode) -> None:
        self.code = code
        super().__init__(f"prompt contract failed: {code.value}")


class _PromptModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _digest_json(value: object, *, domain: str) -> str:
    return length_prefixed_sha256(canonical_json(value), domain=domain)


def _quoted_text_digest(encoded: bytes) -> str:
    return length_prefixed_sha256(encoded, domain=_QUOTED_TEXT_DIGEST_DOMAIN)


def _decode_quoted_bytes(value: str) -> bytes:
    if type(value) is not str or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ValueError("quoted prompt text has invalid framing")
    body = value[1:-1]
    decoded = bytearray()
    index = 0
    while index < len(body):
        character = body[index]
        ordinal = ord(character)
        if character == "\\":
            if index + 3 >= len(body) or body[index + 1] != "x":
                raise ValueError("quoted prompt text has invalid escape")
            pair = body[index + 2 : index + 4]
            if any(item not in "0123456789abcdef" for item in pair):
                raise ValueError("quoted prompt text has invalid escape")
            decoded.append(int(pair, 16))
            index += 4
            continue
        if ordinal > 0x7F or character in {'"', "\\"}:
            raise ValueError("quoted prompt text contains an unsafe byte")
        decoded.append(ordinal)
        index += 1
    return bytes(decoded)


class QuotedDataText(_PromptModel):
    schema_version: Literal["quoted-data-text/v1"] = "quoted-data-text/v1"
    quoted: Annotated[str, Field(min_length=2, repr=False)]
    source_utf8_bytes: Annotated[int, Field(ge=0, le=MAX_PROMPT_PAYLOAD_BYTES)]
    source_digest: Sha256Digest

    @classmethod
    def from_text(cls, value: str) -> QuotedDataText:
        try:
            if type(value) is not str or len(value) > MAX_PROMPT_PAYLOAD_BYTES:
                raise TypeError
            encoded = value.encode("utf-8", errors="strict")
            if len(encoded) > MAX_PROMPT_PAYLOAD_BYTES:
                raise ValueError
            quoted = quote_untrusted_evidence(value)
        except Exception:
            # Route construction failures through Pydantic so callers get the same
            # sanitized boundary exception as they do for a deserialized value.
            return cls(quoted='""', source_utf8_bytes=1, source_digest="0" * 64)
        return cls(
            quoted=quoted,
            source_utf8_bytes=len(encoded),
            source_digest=_quoted_text_digest(encoded),
        )

    def decode(self) -> str:
        try:
            return _decode_quoted_bytes(self.quoted).decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("quoted prompt text failed decoding") from error

    @model_validator(mode="after")
    def content_matches_measurements(self) -> Self:
        try:
            encoded = _decode_quoted_bytes(self.quoted)
            decoded = encoded.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            raise ValueError("quoted prompt text failed validation") from None
        if (
            len(encoded) != self.source_utf8_bytes
            or _quoted_text_digest(encoded) != self.source_digest
            or quote_untrusted_evidence(decoded) != self.quoted
        ):
            raise ValueError("quoted prompt text measurements do not match")
        return self


def _memory_record_digest(record: MemoryRecord) -> str:
    return _digest_json(record, domain=_MEMORY_RECORD_DIGEST_DOMAIN)


class QuotedMemoryRecord(_PromptModel):
    schema_version: Literal["quoted-memory-record/v1"] = "quoted-memory-record/v1"
    record_type: Literal["memory_record"] = "memory_record"
    memory_id: UUID4
    run_id: UUID4
    kind: MemoryKind
    content: QuotedDataText = Field(repr=False)
    provenance: Annotated[
        tuple[EvidenceReference, ...],
        Field(min_length=1, max_length=MAX_MEMORY_PROVENANCE_ITEMS, repr=False),
    ]
    confidence: UnitInterval
    validity: ValidityState
    revision: PositiveInt
    created_at: UtcDatetime
    updated_at: UtcDatetime
    access_count: NonNegativeInt
    last_accessed_at: UtcDatetime | None
    expires_at: UtcDatetime | None
    invalidated_at: UtcDatetime | None
    trust_label: TrustLabel
    record_digest: Sha256Digest

    @classmethod
    def from_memory_record(cls, record: MemoryRecord) -> QuotedMemoryRecord:
        try:
            exact = MemoryRecord.model_validate_json(record.model_dump_json(warnings=False))
        except Exception as error:
            raise PromptContractError(PromptErrorCode.INVALID_BANK) from error
        return cls(
            schema_version="quoted-memory-record/v1",
            record_type=exact.record_type,
            memory_id=exact.memory_id,
            run_id=exact.run_id,
            kind=exact.kind,
            content=QuotedDataText.from_text(exact.content),
            provenance=exact.provenance,
            confidence=exact.confidence,
            validity=exact.validity,
            revision=exact.revision,
            created_at=exact.created_at,
            updated_at=exact.updated_at,
            access_count=exact.access_count,
            last_accessed_at=exact.last_accessed_at,
            expires_at=exact.expires_at,
            invalidated_at=exact.invalidated_at,
            trust_label=exact.trust_label,
            record_digest=_memory_record_digest(exact),
        )

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            schema_version="1.0",
            record_type=self.record_type,
            memory_id=self.memory_id,
            run_id=self.run_id,
            kind=self.kind,
            content=self.content.decode(),
            provenance=self.provenance,
            confidence=self.confidence,
            validity=self.validity,
            revision=self.revision,
            created_at=self.created_at,
            updated_at=self.updated_at,
            access_count=self.access_count,
            last_accessed_at=self.last_accessed_at,
            expires_at=self.expires_at,
            invalidated_at=self.invalidated_at,
            trust_label=self.trust_label,
        )

    @model_validator(mode="after")
    def record_is_exact_and_content_bound(self) -> Self:
        try:
            candidate = self.to_memory_record()
            record = MemoryRecord.model_validate_json(candidate.model_dump_json(warnings=False))
        except Exception:
            raise ValueError("quoted memory record failed validation") from None
        if _memory_record_digest(record) != self.record_digest:
            raise ValueError("quoted memory record digest does not match")
        return self


class BankViewKind(StrEnum):
    CURRENT = "current"
    CANDIDATE_POST_DELTA = "candidate_post_delta"


def _bank_view_digest(values: Mapping[str, object]) -> str:
    as_of = values["as_of"]
    if isinstance(as_of, datetime):
        as_of = as_of.isoformat().replace("+00:00", "Z")
    return _digest_json(
        {
            "schema_version": values["schema_version"],
            "kind": values["kind"],
            "run_id": str(values["run_id"]),
            "as_of": as_of,
            "source_projection_digest": values["source_projection_digest"],
            "records": values["records"],
        },
        domain=_BANK_VIEW_DIGEST_DOMAIN,
    )


class ActiveBankPromptView(_PromptModel):
    schema_version: Literal["active-bank-prompt-view/v1"] = "active-bank-prompt-view/v1"
    kind: BankViewKind
    run_id: UUID4
    as_of: UtcDatetime
    source_projection_digest: PayloadDigest
    records: tuple[QuotedMemoryRecord, ...] = Field(repr=False)
    view_digest: Sha256Digest = Field(default_factory=_bank_view_digest)

    @model_validator(mode="after")
    def records_are_exact_active_and_ordered(self) -> Self:
        if len(self.records) > MAX_ACTIVE_BANK_RECORDS:
            raise ValueError("active bank exceeds its record limit")
        try:
            records = tuple(item.to_memory_record() for item in self.records)
            source_projection_digest = PayloadDigest.model_validate_json(
                self.source_projection_digest.model_dump_json(warnings=False)
            )
        except Exception:
            raise ValueError("active bank records failed validation") from None
        if type(self.source_projection_digest) is not PayloadDigest or (
            source_projection_digest != self.source_projection_digest
        ):
            raise ValueError("active bank projection digest failed exact validation")
        keys = tuple((item.kind.value, str(item.memory_id)) for item in records)
        if keys != tuple(sorted(keys)) or len({item.memory_id for item in records}) != len(records):
            raise ValueError("active bank records are not unique and canonically ordered")
        if sum(item.kind is MemoryKind.PRIVATE_STATUS for item in records) > 1:
            raise ValueError("active bank has more than one private status")
        for item in records:
            if item.run_id != self.run_id:
                raise ValueError("active bank contains a cross-run record")
            if (
                item.validity is not ValidityState.ACTIVE
                or item.created_at > self.as_of
                or (item.expires_at is not None and item.expires_at <= self.as_of)
            ):
                raise ValueError("active bank contains a non-active record")
        values = self.model_dump(mode="json", exclude={"view_digest"})
        if self.view_digest != _bank_view_digest(values):
            raise ValueError("active bank view digest does not match")
        return self


def build_active_bank_prompt_view(
    *,
    kind: BankViewKind,
    run_id: UUID,
    as_of: datetime,
    source_projection_digest: PayloadDigest,
    records: tuple[MemoryRecord, ...],
) -> ActiveBankPromptView:
    try:
        if type(records) is not tuple:
            raise PromptContractError(PromptErrorCode.INVALID_BANK)
        if len(records) > MAX_ACTIVE_BANK_RECORDS:
            raise PromptContractError(PromptErrorCode.LIMIT_EXCEEDED)
        exact_records: list[MemoryRecord] = []
        source_input_bytes = 0
        for record in records:
            if type(record) is not MemoryRecord:
                raise PromptContractError(PromptErrorCode.INVALID_BANK)
            if (
                type(record.provenance) is not tuple
                or not 1 <= len(record.provenance) <= MAX_MEMORY_PROVENANCE_ITEMS
                or any(type(item) is not EvidenceReference for item in record.provenance)
            ):
                raise PromptContractError(PromptErrorCode.INVALID_BANK)
            source_input_bytes += len(record.content.encode("utf-8", errors="strict"))
            source_input_bytes += sum(
                len(item.field_path.encode("utf-8", errors="strict")) + 256
                for item in record.provenance
            )
            if source_input_bytes > MAX_PROMPT_PAYLOAD_BYTES:
                raise PromptContractError(PromptErrorCode.LIMIT_EXCEEDED)
            exact = MemoryRecord.model_validate_json(record.model_dump_json(warnings=False))
            if exact.run_id != run_id:
                raise PromptContractError(PromptErrorCode.CROSS_RUN)
            exact_records.append(exact)
        keys = tuple((item.kind.value, str(item.memory_id)) for item in exact_records)
        if keys != tuple(sorted(keys)):
            raise PromptContractError(PromptErrorCode.INVALID_BANK)
        if len({item.memory_id for item in exact_records}) != len(exact_records):
            raise PromptContractError(PromptErrorCode.INVALID_BANK)
        if sum(item.kind is MemoryKind.PRIVATE_STATUS for item in exact_records) > 1:
            raise PromptContractError(PromptErrorCode.INVALID_BANK)
        if any(
            item.validity is not ValidityState.ACTIVE
            or item.created_at > as_of
            or (item.expires_at is not None and item.expires_at <= as_of)
            for item in exact_records
        ):
            raise PromptContractError(PromptErrorCode.INVALID_BANK)
        exact_projection_digest = PayloadDigest.model_validate_json(
            source_projection_digest.model_dump_json(warnings=False)
        )
        quoted_records: list[QuotedMemoryRecord] = []
        quoted_canonical_bytes = 2
        for item in exact_records:
            quoted = QuotedMemoryRecord.from_memory_record(item)
            quoted_canonical_bytes += len(canonical_json(quoted)) + 1
            if quoted_canonical_bytes > MAX_PROMPT_PAYLOAD_BYTES:
                raise PromptContractError(PromptErrorCode.LIMIT_EXCEEDED)
            quoted_records.append(quoted)
        view = ActiveBankPromptView(
            kind=kind,
            run_id=run_id,
            as_of=as_of,
            source_projection_digest=exact_projection_digest,
            records=tuple(quoted_records),
        )
        if len(canonical_json(view)) > MAX_PROMPT_PAYLOAD_BYTES:
            raise PromptContractError(PromptErrorCode.LIMIT_EXCEEDED)
        return view
    except PromptContractError:
        raise
    except Exception as error:
        raise PromptContractError(PromptErrorCode.INVALID_BANK) from error


class StrictJsonSchema(_PromptModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    name: Annotated[str, Field(min_length=1, max_length=256)]
    strict: Literal[True]
    schema_value: JsonObject = Field(alias="schema", repr=False)

    @model_validator(mode="after")
    def schema_is_closed_root_object(self) -> Self:
        if (
            self.schema_value.get("type") != "object"
            or self.schema_value.get("additionalProperties") is not False
        ):
            raise ValueError("strict response schema must be a closed root object")
        return self


class JsonSchemaResponseFormat(_PromptModel):
    type: Literal["json_schema"]
    json_schema: StrictJsonSchema


class SystemPromptMessage(_PromptModel):
    role: Literal["system"]
    content: Annotated[str, Field(min_length=1, repr=False)]

    @field_validator("content")
    @classmethod
    def instruction_is_lf_only_utf8(cls, value: str) -> str:
        if "\r" in value:
            raise ValueError("system instruction must use LF line endings")
        value.encode("utf-8", errors="strict")
        return value


class UntrustedPromptDataMessage(_PromptModel):
    role: Literal["user"]
    content: Annotated[str, Field(min_length=1, repr=False)]

    @field_validator("content")
    @classmethod
    def data_is_lf_only_utf8(cls, value: str) -> str:
        if "\r" in value:
            raise ValueError("prompt data must use LF line endings")
        value.encode("utf-8", errors="strict")
        return value


class StructuredPromptPayload(_PromptModel):
    messages: tuple[SystemPromptMessage, UntrustedPromptDataMessage] = Field(repr=False)
    response_format: JsonSchemaResponseFormat

    @model_validator(mode="after")
    def provider_payload_is_bounded(self) -> Self:
        if len(canonical_json(self)) > MAX_PROMPT_PAYLOAD_BYTES:
            raise ValueError("provider prompt payload exceeds its byte ceiling")
        return self

    def as_json_object(self) -> dict[str, object]:
        value = json.loads(canonical_json(self))
        if type(value) is not dict:  # pragma: no cover - this model always serializes as an object
            raise TypeError("prompt payload did not serialize as an object")
        return cast(dict[str, object], value)


class PromptTemplateIdentity(_PromptModel):
    bundle_id: str
    template_id: str
    phase: StructuredCallPhase
    response_schema_version: str
    template_digest: Sha256Digest


def _template_digest(values: Mapping[str, object]) -> str:
    return _digest_json(
        {
            "bundle_id": values["bundle_id"],
            "template_id": values["template_id"],
            "phase": values["phase"],
            "response_schema_version": values["response_schema_version"],
            "data_schema_version": values["data_schema_version"],
            "system_instruction": values["system_instruction"],
            "static_sections": values["static_sections"],
            "response_format": values["response_format"],
        },
        domain=_PROMPT_TEMPLATE_DIGEST_DOMAIN,
    )


class PromptDataSectionName(StrEnum):
    TASK = "task"
    MESSAGE_WINDOW = "message_window"
    MEMORY_BANK = "memory_bank"
    OPERATION_SEMANTICS = "operation_semantics"
    TRUST_POLICY = "trust_policy"
    CLAIM_POLICY = "claim_policy"
    RESPONSE_SCHEMA = "response_schema"


class PromptTemplate(_PromptModel):
    bundle_id: str
    template_id: str
    phase: StructuredCallPhase
    response_schema_version: str
    data_schema_version: str
    system_instruction: Annotated[str, Field(min_length=1, repr=False)]
    static_sections: tuple[JsonObject, ...]
    response_format: JsonSchemaResponseFormat
    template_digest: Sha256Digest = Field(default_factory=_template_digest)

    @field_validator("system_instruction")
    @classmethod
    def instruction_is_canonical_text(cls, value: str) -> str:
        if "\r" in value or any(line.endswith((" ", "\t")) for line in value.split("\n")):
            raise ValueError("prompt template instruction is not canonical LF text")
        value.encode("utf-8", errors="strict")
        return value

    @model_validator(mode="after")
    def digest_matches_static_contract(self) -> Self:
        values = self.model_dump(mode="json", exclude={"template_digest"})
        if self.template_digest != _template_digest(values):
            raise ValueError("prompt template digest does not match")
        if len({canonical_json(item) for item in self.static_sections}) != len(
            self.static_sections
        ):
            raise ValueError("prompt template sections must be unique")
        return self

    @property
    def identity(self) -> PromptTemplateIdentity:
        return PromptTemplateIdentity(
            bundle_id=self.bundle_id,
            template_id=self.template_id,
            phase=self.phase,
            response_schema_version=self.response_schema_version,
            template_digest=self.template_digest,
        )


class PromptDataSection(_PromptModel):
    name: PromptDataSectionName
    payload: JsonObject = Field(repr=False)


def _task_prompt_payload(window: MessageWindow) -> dict[str, object]:
    task = window.task_description
    return {
        "version": task.version,
        "content": QuotedDataText.from_text(task.content).model_dump(mode="json"),
        "evidence": task.source.evidence.model_dump(mode="json"),
        "trust_label": task.source.trust_label.value,
    }


def _message_window_prompt_payload(window: MessageWindow) -> dict[str, object]:
    return {
        "version": window.payload.version,
        "messages": [
            {
                "trajectory_role_label": message.role.value,
                "content": QuotedDataText.from_text(message.content).model_dump(mode="json"),
                "evidence": message.evidence.model_dump(mode="json"),
                "trust_label": message.trust_label.value,
            }
            for message in window.payload.messages
        ],
    }


def _memory_bank_prompt_payload(bank: ActiveBankPromptView) -> dict[str, object]:
    records = [
        cast(
            dict[str, object],
            item.model_dump(mode="json", exclude={"record_digest"}),
        )
        for item in bank.records
    ]
    return {
        "schema_version": "active-bank-provider-view/v1",
        "kind": bank.kind.value,
        "run_id": str(bank.run_id),
        "as_of": bank.as_of.isoformat().replace("+00:00", "Z"),
        "records": records,
    }


def build_prompt_dynamic_sections(
    *, window: MessageWindow, bank: ActiveBankPromptView
) -> tuple[PromptDataSection, PromptDataSection, PromptDataSection]:
    """Project the exact provider-visible task, window, and bank sections."""

    return (
        PromptDataSection(name=PromptDataSectionName.TASK, payload=_task_prompt_payload(window)),
        PromptDataSection(
            name=PromptDataSectionName.MESSAGE_WINDOW,
            payload=_message_window_prompt_payload(window),
        ),
        PromptDataSection(
            name=PromptDataSectionName.MEMORY_BANK,
            payload=_memory_bank_prompt_payload(bank),
        ),
    )


def _prompt_input_digest(values: Mapping[str, object]) -> str:
    return _digest_json(
        {
            "data_schema_version": values["data_schema_version"],
            "phase": values["phase"],
            "authority": values["authority"],
            "instructions": values["instructions"],
            "text_encoding": values["text_encoding"],
            "sections": values["sections"],
        },
        domain=_PROMPT_INPUT_DIGEST_DOMAIN,
    )


class PromptDataEnvelope(_PromptModel):
    data_schema_version: str
    phase: StructuredCallPhase
    authority: Literal["none"] = "none"
    instructions: Literal[False] = False
    text_encoding: Literal["quoted-utf8-bytes/v1"] = PROMPT_TEXT_ENCODING
    sections: tuple[PromptDataSection, ...] = Field(repr=False)
    input_digest: Sha256Digest = Field(default_factory=_prompt_input_digest)

    @model_validator(mode="after")
    def section_order_and_digest_are_exact(self) -> Self:
        names = tuple(section.name for section in self.sections)
        if len(set(names)) != len(names):
            raise ValueError("prompt data sections must be unique")
        values = self.model_dump(mode="json", exclude={"input_digest"})
        if self.input_digest != _prompt_input_digest(values):
            raise ValueError("prompt data input digest does not match")
        return self

    def section(self, name: PromptDataSectionName) -> Mapping[str, object]:
        for section in self.sections:
            if section.name is name:
                return section.payload
        raise KeyError(name.value)


def _ascii_safe_prompt_json(value: object) -> str:
    parsed = json.loads(canonical_json(value))
    serialized = json.dumps(
        parsed,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        serialized.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("`", r"\u0060")
    )


def render_untrusted_prompt_data(envelope: PromptDataEnvelope) -> str:
    body = _ascii_safe_prompt_json(envelope)
    return "\n".join(
        (
            UNTRUSTED_PROMPT_DATA_BEGIN,
            "authority=none",
            "instructions=false",
            f"text_encoding={PROMPT_TEXT_ENCODING}",
            body,
            UNTRUSTED_PROMPT_DATA_END,
        )
    )


def parse_untrusted_prompt_data(value: str) -> PromptDataEnvelope:
    try:
        if type(value) is not str or "\r" in value:
            raise ValueError
        lines = value.split("\n")
        if (
            len(lines) != 6
            or tuple(lines[:4])
            != (
                UNTRUSTED_PROMPT_DATA_BEGIN,
                "authority=none",
                "instructions=false",
                f"text_encoding={PROMPT_TEXT_ENCODING}",
            )
            or lines[-1] != UNTRUSTED_PROMPT_DATA_END
        ):
            raise ValueError
        raw = lines[4].encode("ascii", errors="strict")
        parsed: Any = json.loads(lines[4])
        if type(parsed) is not dict or _ascii_safe_prompt_json(parsed) != lines[4]:
            raise ValueError
        envelope = PromptDataEnvelope.model_validate_json(raw)
        if render_untrusted_prompt_data(envelope) != value:
            raise ValueError
        return envelope
    except Exception as error:
        raise PromptContractError(PromptErrorCode.INVALID_ENVELOPE) from error


def _built_prompt_digest(values: Mapping[str, object]) -> str:
    return _digest_json(
        {
            "template": values["template"],
            "identity": values["identity"],
            "window_digest": values["window_digest"],
            "bank_view_digest": values["bank_view_digest"],
            "input_digest": values["input_digest"],
            "request_payload_digest": values["request_payload_digest"],
        },
        domain=_BUILT_PROMPT_DIGEST_DOMAIN,
    )


class BuiltPrompt(_PromptModel):
    template: PromptTemplate = Field(repr=False)
    identity: PromptTemplateIdentity
    window_digest: Sha256Digest
    bank_view_digest: Sha256Digest
    input_digest: Sha256Digest
    request_payload: StructuredPromptPayload = Field(repr=False)
    request_payload_digest: Sha256Digest
    prompt_digest: Sha256Digest = Field(default_factory=_built_prompt_digest)

    @classmethod
    def create(
        cls,
        *,
        template: PromptTemplate,
        window: MessageWindow,
        bank: ActiveBankPromptView,
        envelope: PromptDataEnvelope,
    ) -> BuiltPrompt:
        expected_dynamic = build_prompt_dynamic_sections(window=window, bank=bank)
        if tuple(canonical_json(item) for item in envelope.sections[:3]) != tuple(
            canonical_json(item) for item in expected_dynamic
        ):
            raise PromptContractError(PromptErrorCode.INVALID_ENVELOPE)
        payload = StructuredPromptPayload(
            messages=(
                SystemPromptMessage(role="system", content=template.system_instruction),
                UntrustedPromptDataMessage(
                    role="user", content=render_untrusted_prompt_data(envelope)
                ),
            ),
            response_format=template.response_format,
        )
        return cls(
            template=template,
            identity=template.identity,
            window_digest=window.window_digest,
            bank_view_digest=bank.view_digest,
            input_digest=envelope.input_digest,
            request_payload=payload,
            request_payload_digest=_digest_json(
                payload.as_json_object(), domain=_PROMPT_PAYLOAD_DIGEST_DOMAIN
            ),
        )

    @model_validator(mode="after")
    def payload_and_input_digests_match(self) -> Self:
        if self.request_payload_digest != _digest_json(
            self.request_payload.as_json_object(), domain=_PROMPT_PAYLOAD_DIGEST_DOMAIN
        ):
            raise ValueError("provider prompt payload digest does not match")
        try:
            envelope = parse_untrusted_prompt_data(self.request_payload.messages[1].content)
        except PromptContractError:
            raise ValueError("provider prompt input envelope failed validation") from None
        if envelope.input_digest != self.input_digest:
            raise ValueError("provider prompt input digest does not match")
        actual_static_sections = tuple(
            section.model_dump(mode="json") for section in envelope.sections[3:]
        )
        if (
            self.identity != self.template.identity
            or envelope.phase is not self.template.phase
            or envelope.data_schema_version != self.template.data_schema_version
            or tuple(section.name for section in envelope.sections[:3])
            != (
                PromptDataSectionName.TASK,
                PromptDataSectionName.MESSAGE_WINDOW,
                PromptDataSectionName.MEMORY_BANK,
            )
            or len(envelope.sections) != 3 + len(self.template.static_sections)
            or tuple(canonical_json(item) for item in actual_static_sections)
            != tuple(canonical_json(item) for item in self.template.static_sections)
            or self.request_payload.messages[0].content != self.template.system_instruction
            or self.request_payload.response_format != self.template.response_format
        ):
            raise ValueError("built prompt template contract does not match")
        values = self.model_dump(mode="json", exclude={"prompt_digest", "request_payload"})
        if self.prompt_digest != _built_prompt_digest(values):
            raise ValueError("built prompt digest does not match")
        return self


class PromptBundleIdentity(_PromptModel):
    bundle_id: str
    templates: tuple[PromptTemplateIdentity, ...]
    bundle_digest: Sha256Digest

    @classmethod
    def from_templates(
        cls, bundle_id: str, templates: tuple[PromptTemplate, ...]
    ) -> PromptBundleIdentity:
        identities = tuple(template.identity for template in templates)
        digest = _digest_json(
            {"bundle_id": bundle_id, "templates": identities},
            domain=_PROMPT_BUNDLE_DIGEST_DOMAIN,
        )
        return cls(bundle_id=bundle_id, templates=identities, bundle_digest=digest)

    @model_validator(mode="after")
    def bundle_digest_matches_templates(self) -> Self:
        expected = _digest_json(
            {"bundle_id": self.bundle_id, "templates": self.templates},
            domain=_PROMPT_BUNDLE_DIGEST_DOMAIN,
        )
        if self.bundle_digest != expected:
            raise ValueError("prompt bundle digest does not match")
        template_ids = tuple(item.template_id for item in self.templates)
        phases = tuple(item.phase for item in self.templates)
        if (
            not self.templates
            or any(item.bundle_id != self.bundle_id for item in self.templates)
            or len(set(template_ids)) != len(template_ids)
            or len(set(phases)) != len(phases)
        ):
            raise ValueError("prompt bundle templates do not match the bundle contract")
        return self


_UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "patternProperties",
    }
)


def strict_provider_schema(value: Mapping[str, object]) -> dict[str, object]:
    """Normalize a reviewed Pydantic schema to the strict cross-provider subset."""

    definitions = value.get("$defs")

    def resolve_union_branch(branch: object) -> Mapping[str, object] | None:
        if not isinstance(branch, Mapping):
            return None
        reference = branch.get("$ref")
        if reference is None:
            return cast(Mapping[str, object], branch)
        if (
            len(branch) != 1
            or type(reference) is not str
            or not reference.startswith("#/$defs/")
            or not isinstance(definitions, Mapping)
        ):
            return None
        target = definitions.get(reference.removeprefix("#/$defs/"))
        return cast(Mapping[str, object], target) if isinstance(target, Mapping) else None

    def union_is_disjoint(branches: object) -> bool:
        if not isinstance(branches, (list, tuple)) or len(branches) < 2:
            return False
        branch_tags: list[dict[str, bytes]] = []
        for branch in branches:
            target = resolve_union_branch(branch)
            properties = None if target is None else target.get("properties")
            if not isinstance(properties, Mapping):
                return False
            tags: dict[str, bytes] = {}
            for field_name, field_schema in properties.items():
                if (
                    type(field_name) is str
                    and isinstance(field_schema, Mapping)
                    and "const" in field_schema
                ):
                    tags[field_name] = canonical_json(field_schema["const"])
            if not tags:
                return False
            branch_tags.append(tags)
        common_fields = set(branch_tags[0]).intersection(*(set(item) for item in branch_tags[1:]))
        return any(
            len({item[field_name] for item in branch_tags}) == len(branch_tags)
            for field_name in common_fields
        )

    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            if any(key in item for key in _UNSUPPORTED_SCHEMA_KEYWORDS):
                raise ValueError("response schema contains an unsupported composition keyword")
            if "oneOf" in item and not union_is_disjoint(item["oneOf"]):
                raise ValueError("response schema oneOf branches are not provably disjoint")
            if "$ref" in item:
                reference = item["$ref"]
                if (
                    len(item) != 1
                    or type(reference) is not str
                    or not reference.startswith("#/$defs/")
                    or not isinstance(definitions, Mapping)
                    or reference.removeprefix("#/$defs/") not in definitions
                ):
                    raise ValueError(
                        "response schema reference is not a resolvable local definition"
                    )
            result: dict[str, object] = {}
            for key, nested in item.items():
                if key == "default" or key in {"minLength", "maxLength", "discriminator"}:
                    continue
                normalized_key = "anyOf" if key == "oneOf" else key
                if normalized_key in result:
                    raise ValueError("response schema normalization produced a duplicate keyword")
                result[normalized_key] = normalize(nested)
            if result.get("format") == "uuid4":
                result["format"] = "uuid"
            properties = result.get("properties")
            if properties is not None and result.get("type") != "object":
                raise ValueError("strict response properties require an explicit object type")
            if result.get("type") == "object":
                if not isinstance(properties, Mapping):
                    raise ValueError("strict response object must declare properties")
                additional = item.get("additionalProperties")
                if additional not in (None, False):
                    raise ValueError("strict response object cannot allow additional properties")
                result["additionalProperties"] = False
                result["required"] = list(properties.keys())
            return result
        if isinstance(item, (list, tuple)):
            return [normalize(nested) for nested in cast(Sequence[object], item)]
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise ValueError("response schema contains a non-JSON value")

    normalized = normalize(value)
    if type(normalized) is not dict or normalized.get("type") != "object":
        raise ValueError("strict response schema root must be an object")
    return cast(dict[str, object], normalized)


__all__ = [
    "MAX_ACTIVE_BANK_RECORDS",
    "MAX_PROMPT_PAYLOAD_BYTES",
    "PROMPT_TEXT_ENCODING",
    "UNTRUSTED_PROMPT_DATA_BEGIN",
    "UNTRUSTED_PROMPT_DATA_END",
    "ActiveBankPromptView",
    "BankViewKind",
    "BuiltPrompt",
    "JsonSchemaResponseFormat",
    "PromptBundleIdentity",
    "PromptContractError",
    "PromptDataEnvelope",
    "PromptDataSection",
    "PromptDataSectionName",
    "PromptErrorCode",
    "PromptTemplate",
    "PromptTemplateIdentity",
    "QuotedDataText",
    "QuotedMemoryRecord",
    "StrictJsonSchema",
    "StructuredPromptPayload",
    "SystemPromptMessage",
    "UntrustedPromptDataMessage",
    "build_active_bank_prompt_view",
    "build_prompt_dynamic_sections",
    "parse_untrusted_prompt_data",
    "render_untrusted_prompt_data",
    "strict_provider_schema",
]
