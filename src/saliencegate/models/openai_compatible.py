from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    InterventionAction,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
)
from saliencegate.domain.records import ComponentIdentifier
from saliencegate.memory.proposals import (
    BankOperationsProposal,
    InterventionSelectionOutput,
)
from saliencegate.ports.model_calls import (
    COMPLETION_DIGEST_SCOPE,
    MAX_STRUCTURED_CALL_OUTPUT_BYTES,
    MAX_STRUCTURED_CALL_PAYLOAD_DEPTH,
    MAX_STRUCTURED_CALL_PAYLOAD_NODES,
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    StructuredPhaseOutput,
    validated_result_for_request,
    validated_structured_call_request,
)
from saliencegate.prompts.contracts import StructuredPromptPayload
from saliencegate.prompts.paper_two_phase_v1 import (
    FORCED_REMINDER_RESPONSE_SCHEMA,
    INTERVENTION_RESPONSE_SCHEMA,
    MEMORY_EDIT_RESPONSE_SCHEMA,
)
from saliencegate.runtime.model_token_counting import (
    ModelTokenCount,
    ModelTokenCounter,
    ModelTokenCounterIdentity,
    ModelTokenDirection,
    validated_live_model_token_usage,
)
from saliencegate.security.keys import InstallationKey

OPENAI_COMPATIBLE_CLIENT_VERSION: Literal["openai-compatible-client/v1"] = (
    "openai-compatible-client/v1"
)

_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
_MAX_SIGNED_64 = (1 << 63) - 1
_MIN_SIGNED_64 = -(1 << 63)
_COMPLETION_HMAC_DOMAIN = f"saliencegate:model:{COMPLETION_DIGEST_SCOPE}".encode("ascii")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

NonNegativeSigned64 = Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
Seed = Annotated[int, Field(ge=_MIN_SIGNED_64, le=_MAX_SIGNED_64)]
TimeoutSeconds = Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)]
ResponseByteLimit = Annotated[
    int,
    Field(ge=1, le=MAX_STRUCTURED_CALL_OUTPUT_BYTES),
]


class OpenAICompatibleErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    CLOSED = "closed"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    STATUS = "status"
    MIME = "mime"
    RESPONSE_TOO_LARGE = "response_too_large"
    MALFORMED_JSON = "malformed_json"
    INVALID_RESPONSE = "invalid_response"
    USAGE_UNAVAILABLE = "usage_unavailable"
    TOKEN_ACCOUNTING = "token_accounting"


class OpenAICompatibleError(RuntimeError):
    """A stable, value-free failure before a provider attempt can be attested."""

    def __init__(self, code: OpenAICompatibleErrorCode) -> None:
        self.code = code
        super().__init__(f"openai-compatible client failed: {code.value}")


def _canonical_dns_name(host: str) -> bool:
    if not host or len(host) > 253 or host.endswith("."):
        return False
    labels = host.split(".")
    return len(labels) >= 2 and all(_DNS_LABEL.fullmatch(label) for label in labels)


def _canonical_base_url(value: str, *, allow_remote: bool) -> str:
    try:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or any(ord(character) <= 0x20 or ord(character) >= 0x7F for character in value)
            or "\\" in value
            or "%" in value
        ):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("http", "https")
            or not value.startswith(f"{parsed.scheme}://")
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        host = parsed.hostname
        port = parsed.port
        if host is None or parsed.netloc.endswith(":"):
            raise ValueError
        host = host.lower()

        address: ipaddress.IPv4Address | ipaddress.IPv6Address | None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None

        destination_is_loopback = address is not None and address.is_loopback
        if address is not None:
            if isinstance(address, ipaddress.IPv6Address):
                if address.ipv4_mapped is not None or not parsed.netloc.startswith("["):
                    raise ValueError
                canonical_host = address.compressed
                if host != canonical_host:
                    raise ValueError
                authority = f"[{canonical_host}]"
            else:
                authority = str(address)
            if not allow_remote and not address.is_loopback:
                raise ValueError
        else:
            if (
                not allow_remote
                or not _canonical_dns_name(host)
                or all(character in "0123456789." for character in host)
            ):
                raise ValueError
            authority = host

        if not destination_is_loopback and parsed.scheme != "https":
            raise ValueError

        if port is not None:
            authority = f"{authority}:{port}"

        path = parsed.path
        if path:
            if not path.startswith("/") or "//" in path:
                raise ValueError
            segments = path.split("/")
            if any(segment in (".", "..") for segment in segments):
                raise ValueError
            path = path.rstrip("/")
        normalized = urlunsplit((parsed.scheme, authority, path, "", ""))
        return normalized
    except Exception:
        raise ValueError("OpenAI-compatible base URL failed validation") from None


class OpenAICompatibleConfig(BaseModel):
    """Secret-free configuration for exactly one OpenAI-compatible endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    base_url: str = _DEFAULT_BASE_URL
    model: ComponentIdentifier = "gpt-oss:20b"
    timeout_seconds: TimeoutSeconds = 120.0
    max_response_bytes: ResponseByteLimit = MAX_STRUCTURED_CALL_OUTPUT_BYTES
    seed: Seed = 0
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    credential_env: Annotated[str, Field(pattern=_ENVIRONMENT_NAME.pattern)] | None = None
    allow_remote: bool = False

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def timeout_is_an_exact_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("OpenAI-compatible timeout must be an exact float")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate_endpoint(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        values = cast(dict[str, object], value)
        base_url = values.get("base_url", _DEFAULT_BASE_URL)
        allow_remote = values.get("allow_remote", False)
        if type(base_url) is not str or type(allow_remote) is not bool:
            return value
        copied = dict(values)
        copied["base_url"] = _canonical_base_url(base_url, allow_remote=allow_remote)
        return copied

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"


class _AttemptError(Exception):
    __slots__ = ("code",)

    def __init__(self, code: OpenAICompatibleErrorCode) -> None:
        self.code = code
        super().__init__()


class _StructuredResponseContract(StrEnum):
    MEMORY_EDIT = "memory_edit"
    INTERVENTION_OPTIONAL = "intervention_optional"
    INTERVENTION_FORCED_REMINDER = "intervention_forced_reminder"


_RESPONSE_FORMAT_ALLOWLIST: Mapping[
    StructuredCallPhase,
    tuple[tuple[_StructuredResponseContract, str, Mapping[str, object]], ...],
] = MappingProxyType(
    {
        StructuredCallPhase.MEMORY_EDIT: (
            (
                _StructuredResponseContract.MEMORY_EDIT,
                "saliencegate_memory_edit_output_v1",
                MEMORY_EDIT_RESPONSE_SCHEMA,
            ),
        ),
        StructuredCallPhase.INTERVENTION: (
            (
                _StructuredResponseContract.INTERVENTION_OPTIONAL,
                "saliencegate_intervention_output_v1",
                INTERVENTION_RESPONSE_SCHEMA,
            ),
            (
                _StructuredResponseContract.INTERVENTION_FORCED_REMINDER,
                "saliencegate_forced_reminder_output_v1",
                FORCED_REMINDER_RESPONSE_SCHEMA,
            ),
        ),
    }
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _AttemptError(OpenAICompatibleErrorCode.MALFORMED_JSON)
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise _AttemptError(OpenAICompatibleErrorCode.MALFORMED_JSON)


def _bounded_json(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_STRUCTURED_CALL_PAYLOAD_NODES or depth > MAX_STRUCTURED_CALL_PAYLOAD_DEPTH:
            return False
        if type(item) in (dict, MappingProxyType):
            assert isinstance(item, (dict, MappingProxyType))
            if len(item) > MAX_STRUCTURED_CALL_PAYLOAD_NODES - nodes - len(stack):
                return False
            stack.extend((nested, depth + 1) for nested in item.values())
        elif type(item) in (list, tuple):
            assert isinstance(item, (list, tuple))
            if len(item) > MAX_STRUCTURED_CALL_PAYLOAD_NODES - nodes - len(stack):
                return False
            stack.extend((nested, depth + 1) for nested in item)
    return True


def _parse_json_object(body: bytes) -> dict[str, object]:
    try:
        decoded = body.decode("utf-8", errors="strict")
        parsed: Any = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(parsed) is not dict or not _bounded_json(parsed):
            raise ValueError
        return cast(dict[str, object], parsed)
    except _AttemptError:
        raise
    except Exception:
        raise _AttemptError(OpenAICompatibleErrorCode.MALFORMED_JSON) from None


class _WireModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class _WireUsage(_WireModel):
    prompt_tokens: NonNegativeSigned64
    completion_tokens: NonNegativeSigned64
    total_tokens: NonNegativeSigned64 | None = None

    @model_validator(mode="after")
    def total_is_exact(self) -> Self:
        total = self.prompt_tokens + self.completion_tokens
        if total > _MAX_SIGNED_64 or (self.total_tokens is not None and self.total_tokens != total):
            raise ValueError("OpenAI-compatible token usage failed validation")
        return self


class _WireAssistantMessage(_WireModel):
    role: Literal["assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def visible_content_is_utf8(cls, value: str) -> str:
        value.encode("utf-8", errors="strict")
        return value


class _WireChoice(_WireModel):
    index: Literal[0]
    message: _WireAssistantMessage
    finish_reason: Literal["stop"]


class _WireResponse(_WireModel):
    model: str
    choices: Annotated[tuple[_WireChoice, ...], Field(min_length=1, max_length=1)]
    usage: _WireUsage | None = None


def _wire_response(envelope: Mapping[str, object], *, model: str) -> _WireResponse:
    try:
        response = _WireResponse.model_validate_json(canonical_json(envelope))
        if response.model != model:
            raise ValueError
        return response
    except Exception:
        raise _AttemptError(OpenAICompatibleErrorCode.INVALID_RESPONSE) from None


def _content_type_is_json(value: str) -> bool:
    if type(value) is not str or not value or "," in value:
        return False
    parts = value.split(";")
    if parts[0].strip().lower() != "application/json":
        return False
    if len(parts) == 1:
        return True
    if len(parts) != 2:
        return False
    parameter = parts[1].strip().lower().replace('"', "")
    return parameter == "charset=utf-8"


def _phase_response_contract(
    payload: StructuredPromptPayload,
    phase: StructuredCallPhase,
) -> _StructuredResponseContract | None:
    response_schema = payload.response_format.json_schema
    for contract, expected_name, expected_schema in _RESPONSE_FORMAT_ALLOWLIST[phase]:
        if response_schema.name == expected_name and canonical_json(
            response_schema.schema_value
        ) == canonical_json(expected_schema):
            return contract
    return None


def _counter_count(
    value: object,
    *,
    expected_identity: ModelTokenCounterIdentity,
    direction: ModelTokenDirection,
) -> ModelTokenCount:
    try:
        if type(value) is not ModelTokenCount:
            raise ValueError
        checked = ModelTokenCount.model_validate_json(value.model_dump_json(warnings=False))
        if checked.counter_identity != expected_identity or checked.direction is not direction:
            raise ValueError
        return checked
    except Exception:
        raise OpenAICompatibleError(OpenAICompatibleErrorCode.TOKEN_ACCOUNTING) from None


def _completion_output(
    content: str,
    *,
    contract: _StructuredResponseContract,
) -> tuple[StructuredCallParseStatus, StructuredPhaseOutput | None]:
    try:
        raw: Any = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(raw) is not dict or not _bounded_json(raw):
            raise ValueError
        if contract is not _StructuredResponseContract.MEMORY_EDIT:
            claims = raw.get("claims")
            if raw.get("action") == "remind" and type(claims) is list and not claims:
                return StructuredCallParseStatus.EMPTY_REMINDER, None
            if (
                contract is _StructuredResponseContract.INTERVENTION_FORCED_REMINDER
                and raw.get("action") != "remind"
            ):
                return StructuredCallParseStatus.SCHEMA_INVALID, None
            if type(claims) is list and len(claims) > 2:
                return StructuredCallParseStatus.CLAIM_OVER_LIMIT, None
            output = InterventionSelectionOutput.model_validate_json(content)
            if (
                contract is _StructuredResponseContract.INTERVENTION_FORCED_REMINDER
                and output.action is not InterventionAction.REMIND
            ):
                return StructuredCallParseStatus.SCHEMA_INVALID, None
            return StructuredCallParseStatus.VALID, output
        return (
            StructuredCallParseStatus.VALID,
            BankOperationsProposal.model_validate_json(content),
        )
    except Exception:
        return StructuredCallParseStatus.SCHEMA_INVALID, None


def _completion_digest(content: bytes, key: InstallationKey) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=key._hmac_sha256(content, domain=_COMPLETION_HMAC_DOMAIN),
    )


class OpenAICompatibleClient:
    """One endpoint, one model, strict JSON, and no implicit fallback path."""

    __slots__ = (
        "_client",
        "_close_complete",
        "_closed",
        "_config",
        "_counter",
        "_counter_identity",
        "_credential_lookup",
        "_integrity_key",
        "_lifecycle_lock",
        "_monotonic_ns",
        "_transport",
    )

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        installation_key: InstallationKey,
        model_token_counter: ModelTokenCounter | None = None,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
        credential_lookup: Callable[[str], str | None] = os.environ.get,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        invalid = False
        try:
            if type(config) is not OpenAICompatibleConfig:
                raise TypeError
            checked_config = OpenAICompatibleConfig.model_validate_json(
                config.model_dump_json(warnings=False)
            )
            if type(installation_key) is not InstallationKey:
                raise TypeError
            copied_key = installation_key._copy()
            if not callable(credential_lookup) or not callable(monotonic_ns):
                raise TypeError
            counter_identity: ModelTokenCounterIdentity | None = None
            if model_token_counter is not None:
                identity = model_token_counter.identity
                if type(identity) is not ModelTokenCounterIdentity:
                    raise TypeError
                counter_identity = ModelTokenCounterIdentity.model_validate_json(
                    identity.model_dump_json(warnings=False)
                )
                if counter_identity.model_id != checked_config.model:
                    raise ValueError
            transport = (
                httpx.AsyncHTTPTransport(retries=0, trust_env=False)
                if transport_factory is None
                else transport_factory()
            )
            if not isinstance(transport, httpx.AsyncBaseTransport):
                raise TypeError
            client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(checked_config.timeout_seconds),
                transport=transport,
                trust_env=False,
            )
        except Exception:
            invalid = True
        if invalid:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.INVALID_REQUEST)

        self._config = checked_config
        self._integrity_key = copied_key
        self._counter = model_token_counter
        self._counter_identity = counter_identity
        self._credential_lookup = credential_lookup
        self._monotonic_ns = monotonic_ns
        self._client = client
        self._transport = transport
        self._closed = False
        self._close_complete = False
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(model={self._config.model!r}, closed={self.is_closed!r})"

    @property
    def is_closed(self) -> bool:
        return self._closed or self._client.is_closed

    async def __aenter__(self) -> Self:
        if self.is_closed:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.CLOSED)
        return self

    async def __aexit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        await self.aclose()

    def _credential_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
        }
        environment_name = self._config.credential_env
        if environment_name is None:
            return headers
        invalid = False
        credential: str | None = None
        try:
            credential = self._credential_lookup(environment_name)
            if credential is not None and type(credential) is not str:
                raise TypeError
            if credential and any(not 0x21 <= ord(character) <= 0x7E for character in credential):
                raise ValueError
        except Exception:
            invalid = True
        if invalid:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.INVALID_REQUEST)
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        return headers

    def _preflight(
        self,
        request: object,
    ) -> tuple[
        StructuredCallRequest,
        StructuredPromptPayload,
        ModelTokenCount | None,
        _StructuredResponseContract,
    ]:
        invalid = False
        checked: StructuredCallRequest | None = None
        payload: StructuredPromptPayload | None = None
        input_count: ModelTokenCount | None = None
        response_contract: _StructuredResponseContract | None = None
        try:
            checked = validated_structured_call_request(request)
            if checked.model_id != self._config.model:
                raise ValueError
            payload = StructuredPromptPayload.model_validate_json(canonical_json(checked.payload))
            response_contract = _phase_response_contract(payload, checked.phase)
            if response_contract is None:
                raise ValueError
        except Exception:
            invalid = True
        if invalid or checked is None or payload is None or response_contract is None:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.INVALID_REQUEST)
        counter_failed = False
        if self._counter is not None and self._counter_identity is not None:
            try:
                input_count = _counter_count(
                    self._counter.count_input(checked),
                    expected_identity=self._counter_identity,
                    direction=ModelTokenDirection.INPUT,
                )
            except Exception:
                counter_failed = True
        if counter_failed:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.TOKEN_ACCOUNTING)
        return checked, payload, input_count, response_contract

    def _latency_us(self, started_ns: int) -> int:
        try:
            finished_ns = self._monotonic_ns()
            if type(finished_ns) is not int or type(started_ns) is not int:
                raise ValueError
            elapsed = max(0, finished_ns - started_ns) // 1_000
            return min(elapsed, _MAX_SIGNED_64)
        except Exception:
            return 0

    def _usage(
        self,
        *,
        latency_us: int,
        provider: tuple[int, int] | None,
        input_count: ModelTokenCount | None,
        output_count: ModelTokenCount | None,
    ) -> StructuredCallUsage:
        identity = self._counter_identity
        has_local = identity is not None and (input_count is not None or output_count is not None)
        return StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=None if provider is None else provider[0],
            provider_output_tokens=None if provider is None else provider[1],
            provider_usage_provenance=(
                ProviderUsageProvenance.UNAVAILABLE
                if provider is None
                else ProviderUsageProvenance.PROVIDER_REPORTED
            ),
            latency_us=latency_us,
            canonical_input_tokens=None if input_count is None else input_count.token_count,
            canonical_output_tokens=None if output_count is None else output_count.token_count,
            canonical_usage_provenance=(
                CanonicalUsageProvenance.LOCAL_COUNTER
                if has_local
                else CanonicalUsageProvenance.UNAVAILABLE
            ),
            local_counter_id=identity.counter_id if has_local and identity is not None else None,
            local_counter_version=(
                identity.counter_version if has_local and identity is not None else None
            ),
            local_counter_configuration_digest=(
                identity.configuration_digest if has_local and identity is not None else None
            ),
            local_counter_model_id=(
                identity.model_id if has_local and identity is not None else None
            ),
        )

    def _failed_result(
        self,
        request: StructuredCallRequest,
        *,
        started_ns: int,
        input_count: ModelTokenCount | None,
        timeout: bool,
    ) -> StructuredCallResult:
        return StructuredCallResult(
            schema_version="structured-call-result/v1",
            request_digest=request.request_digest,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            response_schema_version=request.response_schema_version,
            status=(
                StructuredCallStatus.MODEL_TIMEOUT if timeout else StructuredCallStatus.MODEL_ERROR
            ),
            parse_status=StructuredCallParseStatus.NOT_ATTEMPTED,
            output=None,
            completion_digest=None,
            completion_byte_count=None,
            usage=self._usage(
                latency_us=self._latency_us(started_ns),
                provider=None,
                input_count=input_count,
                output_count=None,
            ),
        )

    async def _response_body(self, response: httpx.Response) -> bytes:
        if not 200 <= response.status_code < 300:
            raise _AttemptError(OpenAICompatibleErrorCode.STATUS)
        if not _content_type_is_json(response.headers.get("content-type", "")):
            raise _AttemptError(OpenAICompatibleErrorCode.MIME)
        content_encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if content_encoding != "identity":
            raise _AttemptError(OpenAICompatibleErrorCode.INVALID_RESPONSE)
        declared = response.headers.get("content-length")
        if (
            declared is not None
            and declared.isascii()
            and declared.isdigit()
            and int(declared) > self._config.max_response_bytes
        ):
            raise _AttemptError(OpenAICompatibleErrorCode.RESPONSE_TOO_LARGE)
        if response.is_stream_consumed:
            body = response.content
            if len(body) > self._config.max_response_bytes:
                raise _AttemptError(OpenAICompatibleErrorCode.RESPONSE_TOO_LARGE)
            return body
        chunks: list[bytes] = []
        observed = 0
        async for chunk in response.aiter_raw():
            observed += len(chunk)
            if observed > self._config.max_response_bytes:
                raise _AttemptError(OpenAICompatibleErrorCode.RESPONSE_TOO_LARGE)
            chunks.append(chunk)
        return b"".join(chunks)

    async def _attempt(
        self,
        request: StructuredCallRequest,
        payload: StructuredPromptPayload,
        input_count: ModelTokenCount | None,
        response_contract: _StructuredResponseContract,
        *,
        headers: Mapping[str, str],
        started_ns: int,
    ) -> StructuredCallResult:
        payload_json = payload.as_json_object()
        request_body = canonical_json(
            {
                "messages": payload_json["messages"],
                "model": self._config.model,
                "reasoning_effort": self._config.reasoning_effort,
                "response_format": payload_json["response_format"],
                "seed": self._config.seed,
                "stream": False,
                "temperature": 0,
            }
        )
        async with asyncio.timeout(self._config.timeout_seconds):
            async with self._client.stream(
                "POST",
                self._config.endpoint_url,
                content=request_body,
                headers=headers,
            ) as response:
                body = await self._response_body(response)

        envelope = _parse_json_object(body)
        wire_response = _wire_response(envelope, model=self._config.model)
        provider = (
            None
            if wire_response.usage is None
            else (wire_response.usage.prompt_tokens, wire_response.usage.completion_tokens)
        )
        content = wire_response.choices[0].message.content
        encoded_content = content.encode("utf-8", errors="strict")

        output_count: ModelTokenCount | None = None
        if self._counter is not None and self._counter_identity is not None:
            output_count = _counter_count(
                self._counter.count_output(model_id=request.model_id, completion=content),
                expected_identity=self._counter_identity,
                direction=ModelTokenDirection.OUTPUT,
            )

        usage = self._usage(
            latency_us=self._latency_us(started_ns),
            provider=provider,
            input_count=input_count,
            output_count=output_count,
        )
        try:
            usage = validated_live_model_token_usage(
                usage,
                configured_counter=self._counter_identity,
            )
        except Exception:
            raise _AttemptError(OpenAICompatibleErrorCode.USAGE_UNAVAILABLE) from None

        parse_status, output = _completion_output(content, contract=response_contract)
        result = StructuredCallResult(
            schema_version="structured-call-result/v1",
            request_digest=request.request_digest,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            response_schema_version=request.response_schema_version,
            status=StructuredCallStatus.COMPLETED,
            parse_status=parse_status,
            output=output,
            completion_digest=_completion_digest(encoded_content, self._integrity_key),
            completion_byte_count=len(encoded_content),
            usage=usage,
        )
        return validated_result_for_request(request, result)

    async def _generate_serialized(
        self,
        request: StructuredCallRequest,
    ) -> StructuredCallResult:
        async with self._lifecycle_lock:
            if self.is_closed:
                raise OpenAICompatibleError(OpenAICompatibleErrorCode.CLOSED)
            checked, payload, input_count, response_contract = self._preflight(request)
            headers = self._credential_headers()
            invalid_clock = False
            try:
                started_ns = self._monotonic_ns()
                if type(started_ns) is not int or started_ns < 0:
                    raise ValueError
            except Exception:
                invalid_clock = True
            if invalid_clock:
                raise OpenAICompatibleError(OpenAICompatibleErrorCode.INVALID_REQUEST)
            try:
                return await self._attempt(
                    checked,
                    payload,
                    input_count,
                    response_contract,
                    headers=headers,
                    started_ns=started_ns,
                )
            except (TimeoutError, httpx.TimeoutException):
                return self._failed_result(
                    checked,
                    started_ns=started_ns,
                    input_count=input_count,
                    timeout=True,
                )
            except Exception:
                return self._failed_result(
                    checked,
                    started_ns=started_ns,
                    input_count=input_count,
                    timeout=False,
                )

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        cancelled = False
        result: StructuredCallResult | None = None
        try:
            result = await self._generate_serialized(request)
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            raise asyncio.CancelledError()
        if result is None:  # pragma: no cover - exhaustive branches above
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.TRANSPORT)
        return result

    async def _close_serialized(self) -> None:
        async with self._lifecycle_lock:
            if self._close_complete:
                return
            self._closed = True
            if self._client.is_closed:
                await self._transport.aclose()
            else:
                await self._client.aclose()
            self._close_complete = True

    async def aclose(self) -> None:
        cancelled = False
        failed = False
        try:
            await self._close_serialized()
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            failed = True
        if cancelled:
            raise asyncio.CancelledError()
        if failed:
            raise OpenAICompatibleError(OpenAICompatibleErrorCode.TRANSPORT)


__all__ = [
    "OPENAI_COMPATIBLE_CLIENT_VERSION",
    "OpenAICompatibleClient",
    "OpenAICompatibleConfig",
    "OpenAICompatibleError",
    "OpenAICompatibleErrorCode",
]
