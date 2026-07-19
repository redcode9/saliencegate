from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from saliencegate.runtime.batching import (
    BatchBuildResult,
    BatchConfig,
    BatchInputError,
    BatchIntegrityError,
    BatchManifest,
    BatchMemory,
    BatchMemoryRole,
    BatchPayload,
    BatchPriorityKind,
    BatchRequest,
    BatchStatus,
    DeterministicBatcher,
    EventAggregate,
    SequenceRange,
    VerbatimEvent,
)
from saliencegate.runtime.budget import (
    BudgetError,
    BudgetGovernor,
    BudgetInputError,
    BudgetReservationDeniedError,
    BudgetSettlementError,
)
from saliencegate.runtime.cycles import (
    CycleCommandFactory,
    CycleCoordinator,
    CycleCoordinatorError,
    CycleCoordinatorIdentityError,
    CycleCoordinatorInputError,
    CycleCoordinatorStateError,
)
from saliencegate.runtime.delivery import (
    DeliveryOutbox,
    DeliveryRuntimeError,
    DeliveryWorker,
    DeliveryWorkerResult,
)
from saliencegate.runtime.message_window import (
    MAX_MESSAGE_WINDOW_CANONICAL_BYTES,
    MAX_MESSAGE_WINDOW_ITEMS,
    MAX_TASK_DESCRIPTION_UTF8_BYTES,
    MESSAGE_WINDOW_VERSION,
    TASK_DESCRIPTION_VERSION,
    AttestedTaskDescription,
    MessageWindow,
    MessageWindowError,
    MessageWindowMessage,
    MessageWindowPayload,
    TrajectoryTextSource,
    project_message_window,
    validated_message_window_for_prefix,
)
from saliencegate.runtime.scheduling import (
    FIXED_STEP_SCHEDULE_VERSION,
    FixedStepDecision,
    FixedStepReason,
    FixedStepSchedule,
    project_fixed_step_schedule,
    validated_fixed_step_schedule_for_prefix,
)
from saliencegate.runtime.token_counting import (
    APPROXIMATE_TOKEN_ALGORITHM_VERSION,
    DeterministicTokenCounter,
    TextSize,
    TokenCountingInputError,
)

if TYPE_CHECKING:
    from saliencegate.runtime.algorithm_result import (
        AlgorithmConfigurationAttestation,
        AlgorithmRunResult,
        FixedStepRecoveryResult,
        ModelTokenUsageAttestation,
        ModelTokenUsageSource,
        algorithm_result_digest,
        algorithm_runtime_uuid,
        algorithm_trace_digest,
        derive_cycle_reservation,
        fixed_step_recovery_digest,
        model_token_usage_attestation,
        semantic_projection_digests,
    )
    from saliencegate.runtime.engine import (
        ReplayEngine,
        ReplayEngineConfig,
        ReplayEngineError,
        ReplayEngineInputError,
        ReplayEngineInvariantError,
        ReplayEngineModelError,
        ReplayEventResult,
        ReplayModelPayload,
        ReplayRoutingBinding,
        ReplayRunResult,
        ReplaySignalExtractor,
        ReplayTraceAdapter,
        ReplayTriggerPolicy,
        normalized_trace_digest,
    )
    from saliencegate.runtime.fixed_step import (
        FixedStepEventInput,
        FixedStepExecutionError,
        FixedStepInputError,
        FixedStepInvariantError,
        FixedStepRunner,
        FixedStepRuntimeError,
    )
    from saliencegate.runtime.fixed_step_core import (
        FixedStepTraceBoundary,
        FixedStepTraceDriver,
        FixedStepTraceInput,
        FixedStepTraceInputError,
        FixedStepTraceInvariantError,
        FixedStepTraceResult,
        FixedStepTraceSpine,
    )
    from saliencegate.runtime.model_token_counting import (
        MODEL_TOKEN_COUNT_SCHEMA_VERSION,
        MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION,
        DeterministicFakeModelTokenCounter,
        DeterministicModelTokenCounter,
        HarmonyTokenCounter,
        ModelTokenAccountingUnavailableError,
        ModelTokenCount,
        ModelTokenCounter,
        ModelTokenCounterIdentity,
        ModelTokenCounterInputError,
        ModelTokenCounterPairingError,
        ModelTokenCounterUnavailableError,
        ModelTokenDirection,
        validated_live_model_token_usage,
    )

_LAZY_ENGINE_EXPORTS = frozenset(
    {
        "ReplayEngine",
        "ReplayEngineConfig",
        "ReplayEngineError",
        "ReplayEngineInputError",
        "ReplayEngineInvariantError",
        "ReplayEngineModelError",
        "ReplayEventResult",
        "ReplayModelPayload",
        "ReplayRoutingBinding",
        "ReplayRunResult",
        "ReplaySignalExtractor",
        "ReplayTraceAdapter",
        "ReplayTriggerPolicy",
        "normalized_trace_digest",
    }
)

_LAZY_ALGORITHM_RESULT_EXPORTS = frozenset(
    {
        "AlgorithmConfigurationAttestation",
        "AlgorithmRunResult",
        "FixedStepRecoveryResult",
        "ModelTokenUsageAttestation",
        "ModelTokenUsageSource",
        "algorithm_result_digest",
        "algorithm_runtime_uuid",
        "algorithm_trace_digest",
        "derive_cycle_reservation",
        "fixed_step_recovery_digest",
        "model_token_usage_attestation",
        "semantic_projection_digests",
    }
)

_LAZY_MODEL_TOKEN_COUNTING_EXPORTS = frozenset(
    {
        "MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION",
        "MODEL_TOKEN_COUNT_SCHEMA_VERSION",
        "DeterministicFakeModelTokenCounter",
        "DeterministicModelTokenCounter",
        "HarmonyTokenCounter",
        "ModelTokenAccountingUnavailableError",
        "ModelTokenCount",
        "ModelTokenCounter",
        "ModelTokenCounterIdentity",
        "ModelTokenCounterInputError",
        "ModelTokenCounterPairingError",
        "ModelTokenCounterUnavailableError",
        "ModelTokenDirection",
        "validated_live_model_token_usage",
    }
)

_LAZY_FIXED_STEP_EXPORTS = frozenset(
    {
        "FixedStepEventInput",
        "FixedStepExecutionError",
        "FixedStepInputError",
        "FixedStepInvariantError",
        "FixedStepRunner",
        "FixedStepRuntimeError",
    }
)

_LAZY_FIXED_STEP_CORE_EXPORTS = frozenset(
    {
        "FixedStepTraceBoundary",
        "FixedStepTraceDriver",
        "FixedStepTraceInput",
        "FixedStepTraceInputError",
        "FixedStepTraceInvariantError",
        "FixedStepTraceResult",
        "FixedStepTraceSpine",
    }
)


def __getattr__(name: str) -> object:
    if name in _LAZY_ENGINE_EXPORTS:
        module_name = "saliencegate.runtime.engine"
    elif name in _LAZY_ALGORITHM_RESULT_EXPORTS:
        module_name = "saliencegate.runtime.algorithm_result"
    elif name in _LAZY_FIXED_STEP_EXPORTS:
        module_name = "saliencegate.runtime.fixed_step"
    elif name in _LAZY_FIXED_STEP_CORE_EXPORTS:
        module_name = "saliencegate.runtime.fixed_step_core"
    elif name in _LAZY_MODEL_TOKEN_COUNTING_EXPORTS:
        module_name = "saliencegate.runtime.model_token_counting"
    else:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "APPROXIMATE_TOKEN_ALGORITHM_VERSION",
    "FIXED_STEP_SCHEDULE_VERSION",
    "MAX_MESSAGE_WINDOW_CANONICAL_BYTES",
    "MAX_MESSAGE_WINDOW_ITEMS",
    "MAX_TASK_DESCRIPTION_UTF8_BYTES",
    "MESSAGE_WINDOW_VERSION",
    "MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION",
    "MODEL_TOKEN_COUNT_SCHEMA_VERSION",
    "TASK_DESCRIPTION_VERSION",
    "AlgorithmConfigurationAttestation",
    "AlgorithmRunResult",
    "AttestedTaskDescription",
    "BatchBuildResult",
    "BatchConfig",
    "BatchInputError",
    "BatchIntegrityError",
    "BatchManifest",
    "BatchMemory",
    "BatchMemoryRole",
    "BatchPayload",
    "BatchPriorityKind",
    "BatchRequest",
    "BatchStatus",
    "BudgetError",
    "BudgetGovernor",
    "BudgetInputError",
    "BudgetReservationDeniedError",
    "BudgetSettlementError",
    "CycleCommandFactory",
    "CycleCoordinator",
    "CycleCoordinatorError",
    "CycleCoordinatorIdentityError",
    "CycleCoordinatorInputError",
    "CycleCoordinatorStateError",
    "DeliveryOutbox",
    "DeliveryRuntimeError",
    "DeliveryWorker",
    "DeliveryWorkerResult",
    "DeterministicBatcher",
    "DeterministicFakeModelTokenCounter",
    "DeterministicModelTokenCounter",
    "DeterministicTokenCounter",
    "EventAggregate",
    "FixedStepDecision",
    "FixedStepEventInput",
    "FixedStepExecutionError",
    "FixedStepInputError",
    "FixedStepInvariantError",
    "FixedStepReason",
    "FixedStepRecoveryResult",
    "FixedStepRunner",
    "FixedStepRuntimeError",
    "FixedStepSchedule",
    "FixedStepTraceBoundary",
    "FixedStepTraceDriver",
    "FixedStepTraceInput",
    "FixedStepTraceInputError",
    "FixedStepTraceInvariantError",
    "FixedStepTraceResult",
    "FixedStepTraceSpine",
    "HarmonyTokenCounter",
    "MessageWindow",
    "MessageWindowError",
    "MessageWindowMessage",
    "MessageWindowPayload",
    "ModelTokenAccountingUnavailableError",
    "ModelTokenCount",
    "ModelTokenCounter",
    "ModelTokenCounterIdentity",
    "ModelTokenCounterInputError",
    "ModelTokenCounterPairingError",
    "ModelTokenCounterUnavailableError",
    "ModelTokenDirection",
    "ModelTokenUsageAttestation",
    "ModelTokenUsageSource",
    "ReplayEngine",
    "ReplayEngineConfig",
    "ReplayEngineError",
    "ReplayEngineInputError",
    "ReplayEngineInvariantError",
    "ReplayEngineModelError",
    "ReplayEventResult",
    "ReplayModelPayload",
    "ReplayRoutingBinding",
    "ReplayRunResult",
    "ReplaySignalExtractor",
    "ReplayTraceAdapter",
    "ReplayTriggerPolicy",
    "SequenceRange",
    "TextSize",
    "TokenCountingInputError",
    "TrajectoryTextSource",
    "VerbatimEvent",
    "algorithm_result_digest",
    "algorithm_runtime_uuid",
    "algorithm_trace_digest",
    "derive_cycle_reservation",
    "fixed_step_recovery_digest",
    "model_token_usage_attestation",
    "normalized_trace_digest",
    "project_fixed_step_schedule",
    "project_message_window",
    "semantic_projection_digests",
    "validated_fixed_step_schedule_for_prefix",
    "validated_live_model_token_usage",
    "validated_message_window_for_prefix",
]
