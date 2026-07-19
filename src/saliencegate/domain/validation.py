from __future__ import annotations

from typing import cast

from pydantic import ValidationError
from pydantic_core import InitErrorDetails

from saliencegate.domain.records import NormalizedTraceEventDraft


def validation_error_without_input(error: ValidationError) -> ValidationError:
    """Copy a Pydantic error without retaining its original input values."""

    return ValidationError.from_exception_data(
        error.title,
        cast(list[InitErrorDetails], error.errors(include_input=False)),
    )


def validate_normalized_trace_event_draft(value: object) -> NormalizedTraceEventDraft:
    """Validate untrusted adapter data without exposing it through structured errors."""

    sanitized_error: ValidationError | None = None
    try:
        draft = NormalizedTraceEventDraft.model_validate(value)
    except ValidationError as error:
        sanitized_error = validation_error_without_input(error)
    if sanitized_error is not None:
        raise sanitized_error
    return draft
