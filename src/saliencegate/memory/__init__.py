from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saliencegate.memory.materialize import (
        MATERIALIZATION_REQUEST_SCHEMA_VERSION,
        MATERIALIZATION_RESULT_SCHEMA_VERSION,
        OPERATION_HANDLE_PREFIX,
        MaterializationFailureReason,
        MaterializedBankOperations,
        MemoryOperationMaterializationError,
        OperationMaterializationRequest,
        materialize_bank_operations,
        operation_handle,
        source_operations_digest,
        validated_materialized_bank_operations,
        validated_materialized_bank_operations_for_request,
        validated_operation_materialization_request,
        verified_materialized_bank_operations_for_request,
    )
    from saliencegate.memory.proposals import (
        INTERVENTION_OUTPUT_SCHEMA_VERSION,
        MAX_PROPOSAL_POINTER_SEGMENTS,
        MAX_PROPOSAL_POINTER_UTF8_BYTES,
        MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        BankOperation,
        BankOperationsProposal,
        DeleteMemory,
        InterventionSelectionOutput,
        SaveKnowledge,
        SaveProcedural,
        UpdatePrivateStatus,
    )
    from saliencegate.memory.two_phase import (
        PaperTwoPhaseCycleExecutor,
        RepositoryOperationMaterializer,
        TwoPhaseExecutionCancelled,
        TwoPhaseExecutionError,
    )

_PROPOSAL_EXPORTS = frozenset(
    {
        "INTERVENTION_OUTPUT_SCHEMA_VERSION",
        "MAX_PROPOSAL_POINTER_SEGMENTS",
        "MAX_PROPOSAL_POINTER_UTF8_BYTES",
        "MEMORY_EDIT_OUTPUT_SCHEMA_VERSION",
        "BankOperation",
        "BankOperationsProposal",
        "DeleteMemory",
        "InterventionSelectionOutput",
        "SaveKnowledge",
        "SaveProcedural",
        "UpdatePrivateStatus",
    }
)

_MATERIALIZATION_EXPORTS = frozenset(
    {
        "MATERIALIZATION_REQUEST_SCHEMA_VERSION",
        "MATERIALIZATION_RESULT_SCHEMA_VERSION",
        "OPERATION_HANDLE_PREFIX",
        "MaterializationFailureReason",
        "MaterializedBankOperations",
        "MemoryOperationMaterializationError",
        "OperationMaterializationRequest",
        "materialize_bank_operations",
        "operation_handle",
        "source_operations_digest",
        "validated_materialized_bank_operations",
        "validated_materialized_bank_operations_for_request",
        "validated_operation_materialization_request",
        "verified_materialized_bank_operations_for_request",
    }
)

_TWO_PHASE_EXPORTS = frozenset(
    {
        "PaperTwoPhaseCycleExecutor",
        "RepositoryOperationMaterializer",
        "TwoPhaseExecutionCancelled",
        "TwoPhaseExecutionError",
    }
)


def __getattr__(name: str) -> object:
    if name in _MATERIALIZATION_EXPORTS:
        value = getattr(import_module("saliencegate.memory.materialize"), name)
        globals()[name] = value
        return value
    if name in _TWO_PHASE_EXPORTS:
        value = getattr(import_module("saliencegate.memory.two_phase"), name)
        globals()[name] = value
        return value
    if name not in _PROPOSAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("saliencegate.memory.proposals"), name)
    globals()[name] = value
    return value


__all__ = [
    "INTERVENTION_OUTPUT_SCHEMA_VERSION",
    "MATERIALIZATION_REQUEST_SCHEMA_VERSION",
    "MATERIALIZATION_RESULT_SCHEMA_VERSION",
    "MAX_PROPOSAL_POINTER_SEGMENTS",
    "MAX_PROPOSAL_POINTER_UTF8_BYTES",
    "MEMORY_EDIT_OUTPUT_SCHEMA_VERSION",
    "OPERATION_HANDLE_PREFIX",
    "BankOperation",
    "BankOperationsProposal",
    "DeleteMemory",
    "InterventionSelectionOutput",
    "MaterializationFailureReason",
    "MaterializedBankOperations",
    "MemoryOperationMaterializationError",
    "OperationMaterializationRequest",
    "PaperTwoPhaseCycleExecutor",
    "RepositoryOperationMaterializer",
    "SaveKnowledge",
    "SaveProcedural",
    "TwoPhaseExecutionCancelled",
    "TwoPhaseExecutionError",
    "UpdatePrivateStatus",
    "materialize_bank_operations",
    "operation_handle",
    "source_operations_digest",
    "validated_materialized_bank_operations",
    "validated_materialized_bank_operations_for_request",
    "validated_operation_materialization_request",
    "verified_materialized_bank_operations_for_request",
]
