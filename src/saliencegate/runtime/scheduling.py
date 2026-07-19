from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import UUID4, PositiveSigned64Offset, Sha256Digest
from saliencegate.ports.repository import RunRepository
from saliencegate.ports.trajectory import (
    MAX_TRAJECTORY_EVENTS,
    AttestedTrajectoryPrefix,
    TrajectoryError,
    TrajectoryErrorCode,
    _resolve_attested_payload_value,
    verify_attested_trajectory_prefix,
)

FIXED_STEP_SCHEDULE_VERSION: Literal["first-and-every-action-step/v1"] = (
    "first-and-every-action-step/v1"
)
_SCHEDULE_DIGEST_DOMAIN = "saliencegate:trajectory:fixed-step-schedule:v1"
_MAX_SIGNED_64 = (1 << 63) - 1


class FixedStepReason(StrEnum):
    BOOTSTRAP = "bootstrap"
    ACTION_STEP = "action_step"
    CURRENT_ACTION_STEP = "current_action_step"
    NO_ACTION_STEP = "no_action_step"


class _ScheduleModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class FixedStepDecision(_ScheduleModel):
    event_id: UUID4
    event_sequence: PositiveSigned64Offset
    invoke: bool
    invocation_ordinal: PositiveSigned64Offset | None = None
    reason: FixedStepReason
    action_step_ordinal: PositiveSigned64Offset | None = None

    @model_validator(mode="after")
    def fields_describe_one_unambiguous_decision(self) -> Self:
        if self.invoke != (self.invocation_ordinal is not None):
            raise ValueError("only an invocation carries an invocation ordinal")
        if self.reason is FixedStepReason.BOOTSTRAP:
            if not self.invoke or self.event_sequence != 1:
                raise ValueError("bootstrap must invoke on the first event")
        elif self.reason is FixedStepReason.ACTION_STEP:
            if not self.invoke or self.action_step_ordinal is None:
                raise ValueError("a new action step must invoke")
        elif self.reason is FixedStepReason.CURRENT_ACTION_STEP:
            if self.invoke or self.action_step_ordinal is None:
                raise ValueError("the current action step must stay silent")
        elif self.invoke or self.action_step_ordinal is not None:
            raise ValueError("an event without a step must stay silent")
        return self


def _schedule_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schedule_version": values["schedule_version"],
                "run_id": str(values["run_id"]),
                "boundary_event_sequence": values["boundary_event_sequence"],
                "trajectory_prefix_digest": values["trajectory_prefix_digest"],
                "decisions": values["decisions"],
                "invocation_count": values["invocation_count"],
            }
        ),
        domain=_SCHEDULE_DIGEST_DOMAIN,
    )


class FixedStepSchedule(_ScheduleModel):
    """One deterministic decision per event in an attested trajectory prefix."""

    schedule_version: Literal["first-and-every-action-step/v1"]
    run_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_TRAJECTORY_EVENTS)]
    trajectory_prefix_digest: Sha256Digest
    decisions: Annotated[
        tuple[FixedStepDecision, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ]
    invocation_count: Annotated[int, Field(ge=1, le=MAX_TRAJECTORY_EVENTS)]
    schedule_digest: Sha256Digest = Field(default_factory=_schedule_digest)

    @model_validator(mode="after")
    def decisions_cover_the_prefix_exactly(self) -> Self:
        if len(self.decisions) != self.boundary_event_sequence:
            raise ValueError("schedule must contain one decision per event")
        if any(
            decision.event_sequence != expected
            for expected, decision in enumerate(self.decisions, start=1)
        ):
            raise ValueError("schedule decisions must be in contiguous source order")
        event_ids = tuple(decision.event_id for decision in self.decisions)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("schedule decisions must identify unique events")
        invocations = tuple(
            decision.invocation_ordinal
            for decision in self.decisions
            if decision.invocation_ordinal is not None
        )
        if len(invocations) != self.invocation_count or any(
            ordinal != expected for expected, ordinal in enumerate(invocations, start=1)
        ):
            raise ValueError("schedule invocation ordinals must be contiguous")
        if self.decisions[0].reason is not FixedStepReason.BOOTSTRAP:
            raise ValueError("the schedule must begin with bootstrap")
        last_step: int | None = None
        for decision in self.decisions:
            step = decision.action_step_ordinal
            if step is None:
                continue
            if last_step is not None and step < last_step:
                raise ValueError("schedule action steps cannot move backwards")
            if decision.reason is FixedStepReason.ACTION_STEP and step == last_step:
                raise ValueError("only a distinct action step can invoke")
            if decision.reason is FixedStepReason.CURRENT_ACTION_STEP and step != last_step:
                raise ValueError("silent action-step events must retain the current step")
            last_step = step
        values = self.model_dump(mode="json", exclude={"schedule_digest"})
        if self.schedule_digest != _schedule_digest(values):
            raise ValueError("fixed-step schedule digest does not match")
        return self


def _action_step(prefix: AttestedTrajectoryPrefix, index: int) -> int | None:
    item = prefix.items[index]
    selector = item.binding.action_step
    if selector is None:
        return None
    value = _resolve_attested_payload_value(item, selector.field_path)
    if type(value) is not int or not 1 <= value <= _MAX_SIGNED_64:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_POINTER)
    return value


def _project_verified_fixed_step_schedule(
    validated: AttestedTrajectoryPrefix,
) -> FixedStepSchedule:
    """Project one repository-verified prefix in linear time."""

    decisions: list[FixedStepDecision] = []
    last_step: int | None = None
    invocation_ordinal = 0
    for index, item in enumerate(validated.items):
        step = _action_step(validated, index)
        if index == 0:
            invocation_ordinal = 1
            decisions.append(
                FixedStepDecision(
                    event_id=item.event.event_id,
                    event_sequence=item.event.sequence,
                    invoke=True,
                    invocation_ordinal=invocation_ordinal,
                    reason=FixedStepReason.BOOTSTRAP,
                    action_step_ordinal=step,
                )
            )
            last_step = step
            continue
        if step is None:
            decisions.append(
                FixedStepDecision(
                    event_id=item.event.event_id,
                    event_sequence=item.event.sequence,
                    invoke=False,
                    reason=FixedStepReason.NO_ACTION_STEP,
                )
            )
            continue
        if last_step is not None and step < last_step:
            raise TrajectoryError(TrajectoryErrorCode.RETROGRADE_BINDING)
        if last_step is not None and step == last_step:
            decisions.append(
                FixedStepDecision(
                    event_id=item.event.event_id,
                    event_sequence=item.event.sequence,
                    invoke=False,
                    reason=FixedStepReason.CURRENT_ACTION_STEP,
                    action_step_ordinal=step,
                )
            )
            continue
        invocation_ordinal += 1
        decisions.append(
            FixedStepDecision(
                event_id=item.event.event_id,
                event_sequence=item.event.sequence,
                invoke=True,
                invocation_ordinal=invocation_ordinal,
                reason=FixedStepReason.ACTION_STEP,
                action_step_ordinal=step,
            )
        )
        last_step = step
    try:
        return FixedStepSchedule(
            schedule_version=FIXED_STEP_SCHEDULE_VERSION,
            run_id=validated.run_id,
            boundary_event_sequence=validated.boundary_event_sequence,
            trajectory_prefix_digest=validated.prefix_digest,
            decisions=tuple(decisions),
            invocation_count=invocation_ordinal,
        )
    except Exception:  # pragma: no cover - constructed entirely from validated values
        raise TrajectoryError(TrajectoryErrorCode.INVALID_INPUT) from None


async def project_fixed_step_schedule(
    repository: RunRepository,
    prefix: AttestedTrajectoryPrefix,
) -> FixedStepSchedule:
    """Verify against the ledger, then project fixed-step cycle boundaries."""

    validated = await verify_attested_trajectory_prefix(repository, prefix)
    return _project_verified_fixed_step_schedule(validated)


async def validated_fixed_step_schedule_for_prefix(
    repository: RunRepository,
    prefix: AttestedTrajectoryPrefix,
    value: object,
) -> FixedStepSchedule:
    """Reproject and require byte-equivalent schedule output."""

    try:
        if type(value) is not FixedStepSchedule:
            raise TypeError
        validated = FixedStepSchedule.model_validate_json(value.model_dump_json(warnings=False))
        expected = await project_fixed_step_schedule(repository, prefix)
        if validated != value or canonical_json(validated) != canonical_json(expected):
            raise ValueError
        return validated
    except TrajectoryError:
        raise
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE) from None


__all__ = [
    "FIXED_STEP_SCHEDULE_VERSION",
    "FixedStepDecision",
    "FixedStepReason",
    "FixedStepSchedule",
    "project_fixed_step_schedule",
    "validated_fixed_step_schedule_for_prefix",
]
