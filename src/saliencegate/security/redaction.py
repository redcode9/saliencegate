from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import unquote

from pydantic import ValidationError

from saliencegate.domain import (
    NormalizedTraceEventDraft,
    RedactedTraceEventDraft,
    canonical_json,
)
from saliencegate.domain.validation import validation_error_without_input
from saliencegate.security.digests import (
    RedactedPayload,
    create_payload_digest,
    verify_payload_digest,
)
from saliencegate.security.keys import InstallationKey

REDACTED = "[REDACTED]"
REDACTED_PRIVATE_KEY = "[REDACTED PRIVATE KEY]"


class SecretInFieldNameError(ValueError):
    pass


class AmbiguousFieldNameError(ValueError):
    pass


_DEFAULT_SECRET_FIELDS = frozenset(
    {
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "pwd",
        "refreshtoken",
        "secret",
        "token",
    }
)

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----.*?"
    r"-----END (?P=label)-----",
    flags=re.DOTALL,
)
_TRUNCATED_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*$",
    flags=re.DOTALL,
)
_URI_AUTHORITY = re.compile(
    r"(?P<prefix>(?:\b[a-zA-Z][a-zA-Z0-9+.-]*:)?//)(?P<authority>[^\s/?#]+)"
)
_BEARER = re.compile(
    r"\bBearer(?:[ \t\u00a0\u200b]+|\r?\n[ \t\u00a0\u200b]+)"
    r"(?P<token>[A-Za-z0-9._~+/=\-\u200b\u200c\u200d\u2060]{8,})",
    flags=re.IGNORECASE,
)
_COMMON_API_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{12,}"
    r")(?![A-Za-z0-9_-])"
)
_ASSIGNED_SECRET = re.compile(
    r"(?P<name>[\"']?\b(?:(?:x[ \t_-]*)?api[ \t_-]*key|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password|passwd|pwd)\b[\"']?)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>"
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r'|"(?:\\.|[^"\\])*$'
    r"|'(?:\\.|[^'\\])*$"
    r"|[^\s,;]+)",
    flags=re.IGNORECASE,
)

_FIELD_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u0456": "i",
        "\u0458": "j",
        "\u043a": "k",
        "\u043c": "m",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0455": "s",
        "\u0442": "t",
        "\u0443": "y",
        "\u0445": "x",
        "\u03b1": "a",
        "\u03b5": "e",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u03bf": "o",
        "\u03c1": "p",
        "\u03c4": "t",
        "\u03c5": "y",
        "\u03c7": "x",
    }
)


@dataclass(frozen=True, slots=True)
class RedactionFinding:
    path: str
    detector: str
    replacements: int


@dataclass(frozen=True, slots=True)
class RedactionResult:
    payload: RedactedPayload
    findings: tuple[RedactionFinding, ...]


@dataclass(frozen=True, slots=True)
class EventRedactionResult:
    event: RedactedTraceEventDraft
    payload: RedactedPayload
    findings: tuple[RedactionFinding, ...]


def _field_name(
    value: str,
    *,
    sensitive_fields: frozenset[str] = frozenset(),
) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    scripts = {
        script
        for character in normalized
        if character.isalpha()
        for script in ("LATIN", "CYRILLIC", "GREEK")
        if script in unicodedata.name(character, "")
    }
    if len(scripts) > 1:
        raise AmbiguousFieldNameError("mixed-script JSON field names cannot be classified safely")
    normalized = unicodedata.normalize("NFKD", normalized).translate(_FIELD_CONFUSABLES)
    parts = tuple(character for character in normalized if character.isalnum())
    wildcard_parts: tuple[str | None, ...] = tuple(
        None
        if character.isalpha()
        and not character.isascii()
        and any(
            script in unicodedata.name(character, "") for script in ("LATIN", "CYRILLIC", "GREEK")
        )
        else character
        for character in parts
    )
    if any(part is None for part in wildcard_parts):
        for sensitive_field in sensitive_fields:
            if len(sensitive_field) == len(wildcard_parts) and all(
                part is None or part == expected
                for part, expected in zip(wildcard_parts, sensitive_field, strict=True)
            ):
                raise AmbiguousFieldNameError(
                    "a Unicode JSON field name is confusable with a sensitive field"
                )
    return "".join(parts)


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _path(parent: str, part: str) -> str:
    return f"{parent}/{_pointer_part(part)}"


def _literal_variants(literals: Sequence[str]) -> tuple[str, ...]:
    variants: set[str] = set()
    for literal in literals:
        if type(literal) is not str:
            raise ValueError("configured literal secrets must be exact strings")
        if not literal:
            raise ValueError("configured literal secrets cannot be empty")
        if literal in REDACTED or literal in REDACTED_PRIVATE_KEY:
            raise ValueError("configured literal cannot be a substring of a redaction marker")
        variants.update(
            {
                literal,
                unicodedata.normalize("NFC", literal),
                unicodedata.normalize("NFD", literal),
                unicodedata.normalize("NFKC", literal),
                unicodedata.normalize("NFKD", literal),
            }
        )
    return tuple(sorted(variants, key=lambda item: (-len(item), item)))


def _literal_match_form(value: str) -> str:
    without_format_controls = "".join(
        character for character in value if unicodedata.category(character) != "Cf"
    )
    normalized = unicodedata.normalize("NFKC", without_format_controls)
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


class Redactor:
    __slots__ = (
        "_literal_match_forms",
        "_literal_secrets",
        "_matched_secret_fields",
        "_secret_fields",
    )

    def __init__(
        self,
        *,
        literal_secrets: Sequence[str] = (),
        structured_field_names: Sequence[str] = (),
    ) -> None:
        self._literal_secrets = _literal_variants(literal_secrets)
        self._literal_match_forms = frozenset(
            _literal_match_form(secret) for secret in self._literal_secrets
        )
        if any(type(name) is not str for name in structured_field_names):
            raise ValueError("structured secret field names must be exact strings")
        custom_fields = {_field_name(name) for name in structured_field_names}
        if "" in custom_fields:
            raise ValueError("structured secret field names cannot be empty")
        self._secret_fields = _DEFAULT_SECRET_FIELDS | custom_fields
        self._matched_secret_fields = self._secret_fields | frozenset(
            f"x{field_name}" for field_name in self._secret_fields
        )

    def __repr__(self) -> str:
        return (
            f"Redactor(literal_secrets={len(self._literal_secrets)}, "
            f"structured_fields={len(self._secret_fields)})"
        )

    def _redact_text(
        self,
        value: str,
        path: str,
        findings: list[RedactionFinding],
    ) -> str:
        if value in (REDACTED, REDACTED_PRIVATE_KEY):
            return value

        updated, count = _PRIVATE_KEY.subn(REDACTED_PRIVATE_KEY, value)
        if count:
            findings.append(RedactionFinding(path, "private_key", count))

        updated, count = _TRUNCATED_PRIVATE_KEY.subn(REDACTED_PRIVATE_KEY, updated)
        if count:
            findings.append(RedactionFinding(path, "truncated_private_key", count))

        uri_count = 0

        def redact_uri(match: re.Match[str]) -> str:
            nonlocal uri_count
            authority = match.group("authority")
            if "@" not in authority:
                return match.group(0)
            userinfo, host = authority.rsplit("@", 1)
            if ":" in userinfo:
                username, _password = userinfo.split(":", 1)
                replacement = f"{username}:{REDACTED}"
            elif ":" in unicodedata.normalize("NFKC", unquote(userinfo)):
                replacement = REDACTED
            else:
                return match.group(0)
            uri_count += 1
            return f"{match.group('prefix')}{replacement}@{host}"

        updated = _URI_AUTHORITY.sub(redact_uri, updated)
        if uri_count:
            findings.append(RedactionFinding(path, "uri_password", uri_count))

        updated, count = _BEARER.subn(f"Bearer {REDACTED}", updated)
        if count:
            findings.append(RedactionFinding(path, "bearer_token", count))

        updated, count = _COMMON_API_KEY.subn(REDACTED, updated)
        if count:
            findings.append(RedactionFinding(path, "api_key", count))

        updated, count = _ASSIGNED_SECRET.subn(
            lambda match: f"{match.group('name')}{match.group('separator')}{REDACTED}",
            updated,
        )
        if count:
            findings.append(RedactionFinding(path, "assigned_secret", count))

        literal_count = 0
        for secret in self._literal_secrets:
            updated, count = re.subn(re.escape(secret), REDACTED, updated)
            literal_count += count
        normalized_updated = _literal_match_form(updated)
        if any(secret in normalized_updated for secret in self._literal_match_forms):
            updated = REDACTED
            literal_count += 1
        if literal_count:
            findings.append(RedactionFinding(path, "configured_literal", literal_count))
        return updated

    def _redact_value(
        self,
        value: object,
        path: str,
        findings: list[RedactionFinding],
    ) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key in value:
                if self._redact_text(key, "", []) != key:
                    raise SecretInFieldNameError(
                        "secret material in a JSON field name cannot be persisted safely"
                    )
            for key in sorted(value):
                child_path = _path(path, key)
                child = value[key]
                if (
                    _field_name(key, sensitive_fields=self._matched_secret_fields)
                    in self._matched_secret_fields
                ):
                    if child not in (REDACTED, REDACTED_PRIVATE_KEY):
                        findings.append(RedactionFinding(child_path, "structured_field", 1))
                    result[key] = REDACTED
                else:
                    result[key] = self._redact_value(child, child_path, findings)
            return result
        if isinstance(value, (list, tuple)):
            return tuple(
                self._redact_value(item, _path(path, str(index)), findings)
                for index, item in enumerate(value)
            )
        if isinstance(value, str):
            return self._redact_text(value, path, findings)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise ValueError(f"unsupported JSON value type: {type(value).__name__}")

    def redact_payload(self, payload: Mapping[str, object]) -> RedactionResult:
        findings: list[RedactionFinding] = []
        redacted = self._redact_value(payload, "", findings)
        if not isinstance(redacted, Mapping):  # pragma: no cover - input type guarantees this
            raise TypeError("redacted payload must remain a JSON object")
        validation_error: ValidationError | None = None
        try:
            redacted_payload = RedactedPayload(root=redacted)
        except ValidationError as error:
            validation_error = validation_error_without_input(error)
        if validation_error is not None:
            raise validation_error
        return RedactionResult(payload=redacted_payload, findings=tuple(findings))

    def redact_event(
        self,
        draft: NormalizedTraceEventDraft,
        *,
        key: InstallationKey | None = None,
        synthetic_benchmark: bool = False,
    ) -> EventRedactionResult:
        result = self.redact_payload(draft.payload)
        digest = create_payload_digest(
            result.payload,
            key=key,
            synthetic_benchmark=synthetic_benchmark,
        )
        values = draft.model_dump(mode="python")
        values.update(
            record_type="redacted_trace_event_draft",
            payload=result.payload.root,
            payload_digest=digest,
        )
        validation_error = None
        try:
            event = RedactedTraceEventDraft.model_validate(values)
        except ValidationError as error:
            validation_error = validation_error_without_input(error)
        if validation_error is not None:
            raise validation_error
        return EventRedactionResult(
            event=event,
            payload=result.payload,
            findings=result.findings,
        )


@dataclass(frozen=True, slots=True, repr=False, init=False)
class RedactionPolicy:
    """Immutable configuration used to build the repository-owned redactor."""

    literal_secrets: tuple[str, ...]
    structured_field_names: tuple[str, ...]

    def __init__(
        self,
        *,
        literal_secrets: Sequence[str] = (),
        structured_field_names: Sequence[str] = (),
    ) -> None:
        literals = tuple(literal_secrets)
        fields = tuple(structured_field_names)
        Redactor(
            literal_secrets=literals,
            structured_field_names=fields,
        )
        object.__setattr__(self, "literal_secrets", literals)
        object.__setattr__(self, "structured_field_names", fields)

    def __repr__(self) -> str:
        return (
            f"RedactionPolicy(literal_secrets={len(self.literal_secrets)}, "
            f"structured_fields={len(self.structured_field_names)})"
        )


def verify_redacted_event(
    event: RedactedTraceEventDraft,
    *,
    redactor: Redactor,
    key: InstallationKey | None = None,
    synthetic_benchmark: bool = False,
) -> bool:
    try:
        redacted_again = redactor.redact_payload(event.payload)
    except ValueError:
        return False
    if canonical_json(redacted_again.payload.root) != canonical_json(event.payload):
        return False
    return verify_payload_digest(
        redacted_again.payload,
        event.payload_digest,
        key=key,
        synthetic_benchmark=synthetic_benchmark,
    )
