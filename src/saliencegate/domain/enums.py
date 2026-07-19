from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    MODEL_OUTPUT = "model_output"
    ACTION_PROPOSAL = "action_proposal"
    TOOL_START = "tool_start"
    TOOL_COMPLETION = "tool_completion"
    OBSERVATION = "observation"
    CONTROLLER_ERROR = "controller_error"


class EventPhase(StrEnum):
    INITIALIZATION = "initialization"
    PRE_ACTION = "pre_action"
    ACTION_EXECUTION = "action_execution"
    POST_ACTION = "post_action"
    TERMINAL = "terminal"
    INTERNAL = "internal"


class SignalType(StrEnum):
    TOOL_ERROR = "tool_error"
    TEST_FAILURE = "test_failure"
    REPEATED_ACTION = "repeated_action"
    REPEATED_FAILURE = "repeated_failure"
    CONTEXT_SHIFT = "context_shift"
    STALE_CONSTRAINT = "stale_constraint"
    STAGNATION = "stagnation"
    IRREVERSIBLE_ACTION = "irreversible_action"
    CONFLICT = "conflict"


class MemoryKind(StrEnum):
    PRIVATE_STATUS = "private_status"
    KNOWLEDGE = "knowledge"
    PROCEDURAL = "procedural"


class ValidityState(StrEnum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class CycleState(StrEnum):
    PENDING = "pending"
    RESERVED = "reserved"
    RUNNING = "running"
    COMMITTED = "committed"
    FAILED = "failed"


class InterventionAction(StrEnum):
    SILENCE = "silence"
    REMIND = "remind"


class DeliveryTarget(StrEnum):
    NEXT_MODEL_CALL = "next_model_call"
    PRE_ACTION_REPLAN = "pre_action_replan"


class DeliveryState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    ATTEMPTING = "attempting"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class DeliveryOutcome(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"
    REFUSED = "refused"


class DeduplicationGuarantee(StrEnum):
    AT_MOST_ONCE_ATTEMPT = "at_most_once_attempt"
    DURABLE_DELIVERY_ID = "durable_delivery_id"


class PayloadDigestAlgorithm(StrEnum):
    HMAC_SHA256 = "hmac_sha256"
    SYNTHETIC_SHA256 = "synthetic_sha256"


class ExpirationAction(StrEnum):
    KEEP = "keep"
    SET = "set"
    CLEAR = "clear"


class TrustLabel(StrEnum):
    TRUSTED_RUNTIME = "trusted_runtime"
    TRUSTED_CONTROLLER = "trusted_controller"
    UNTRUSTED_TASK_INPUT = "untrusted_task_input"
    UNTRUSTED_TOOL_OUTPUT = "untrusted_tool_output"
    UNTRUSTED_MODEL_OUTPUT = "untrusted_model_output"
    UNTRUSTED_EXTERNAL_MEMORY = "untrusted_external_memory"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class EvidenceSource(StrEnum):
    EVENT = "event"
    MEMORY = "memory"


class ClaimKind(StrEnum):
    REQUIREMENT = "requirement"
    ENVIRONMENT_FACT = "environment_fact"
    FAILED_ATTEMPT = "failed_attempt"
    DIAGNOSIS = "diagnosis"
    OPEN_SUBGOAL = "open_subgoal"


class RepeatedErrorStatus(StrEnum):
    AVOIDED = "avoided"
    REPEATED = "repeated"
    NOT_OBSERVED = "not_observed"
    UNKNOWN = "unknown"


class ConstraintStatus(StrEnum):
    RESPECTED = "respected"
    VIOLATED = "violated"
    NOT_OBSERVED = "not_observed"
    UNKNOWN = "unknown"


class UtilityLabel(StrEnum):
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    REDUNDANT = "redundant"
    UNRESOLVED = "unresolved"


class OutcomeEvidenceMode(StrEnum):
    LIVE_OBSERVATION = "live_observation"
    DETERMINISTIC_ORACLE = "deterministic_oracle"
    PAIRED_ROLLOUT = "paired_rollout"
    POLICY_REPLAY = "policy_replay"


class ReasonCode(StrEnum):
    BOOTSTRAP = "bootstrap"
    POLICY_ALWAYS = "always_invoke"
    POLICY_NEVER = "never_invoke"
    SCRIPTED_INVOKE = "scripted_invoke"
    SCRIPTED_SILENCE = "scripted_silence"
    SCRIPT_EXHAUSTED = "script_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    COOLDOWN_ACTIVE = "cooldown_active"
    WATCHDOG = "watchdog"
    HARD_SIGNAL = "hard_signal"
    RISK_THRESHOLD_MET = "risk_threshold_met"
    RISK_BELOW_THRESHOLD = "risk_below_threshold"
    TOOL_ERROR = "tool_error"
    TEST_FAILURE = "test_failure"
    REPEATED_ACTION = "repeated_action"
    REPEATED_FAILURE = "repeated_failure"
    CONTEXT_SHIFT = "context_shift"
    STALE_CONSTRAINT = "stale_constraint"
    STAGNATION = "stagnation"
    IRREVERSIBLE_ACTION = "irreversible_action"
    CONFLICT = "conflict"
    MANDATORY_INPUT_OVERFLOW = "mandatory_input_overflow"
    REMINDER_ACCEPTED = "reminder_accepted"
    SILENCE_SELECTED = "silence_selected"
    GROUNDED_REMINDER = "grounded_reminder"
    NO_GROUNDED_CLAIMS = "no_grounded_claims"
    UNGROUNDED = "ungrounded"
    INVALID_PROVENANCE = "invalid_provenance"
    CITATION_MISSING = "citation_missing"
    CITATION_CROSS_RUN = "citation_cross_run"
    CITATION_EXPIRED = "citation_expired"
    CITATION_INVALIDATED = "citation_invalidated"
    CLAIM_OVER_LIMIT = "claim_over_limit"
    DUPLICATE_REMINDER = "duplicate_reminder"
    COOLDOWN_BLOCKED = "cooldown_blocked"
    UNSUPPORTED_DELIVERY_TARGET = "unsupported_delivery_target"
    UNSUPPORTED_DELIVERY_CHANNEL = "unsupported_delivery_channel"
    UNSAFE_ROLE_MAPPING = "unsafe_role_mapping"
    TARGET_UNAVAILABLE = "target_unavailable"
    DELIVERY_SUCCEEDED = "delivery_succeeded"
    DELIVERY_FAILED = "delivery_failed"
    DELIVERY_UNKNOWN = "delivery_unknown"
    SCHEMA_INVALID = "schema_invalid"
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"
    MEMORY_CONFLICT = "memory_conflict"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    SOURCE_EVENT_COLLISION = "source_event_collision"
    FAILED_UNKNOWN_COST = "failed_unknown_cost"
    NO_INTERVENTION = "no_intervention"
