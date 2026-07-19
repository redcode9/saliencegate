from __future__ import annotations


class DomainError(ValueError):
    """Base class for stable domain-boundary failures."""


class CanonicalJSONError(DomainError):
    """Raised when a value cannot be represented as canonical JSON."""


class InvalidSchemaVersionError(DomainError):
    def __init__(self, version: object) -> None:
        self.version = version
        super().__init__(f"schema_version must use '<major>.<minor>', got {version!r}")


class UnsupportedSchemaVersionError(DomainError):
    def __init__(self, version: str, supported: tuple[str, ...]) -> None:
        self.version = version
        self.supported = supported
        supported_text = ", ".join(supported)
        super().__init__(f"unsupported schema version {version!r}; supported: {supported_text}")


class UnknownRecordTypeError(DomainError):
    def __init__(self, record_type: object) -> None:
        self.record_type = record_type
        super().__init__(f"unknown record_type {record_type!r}")
