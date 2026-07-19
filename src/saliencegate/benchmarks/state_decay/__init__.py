"""Deterministic synthetic paired-pivot diagnostics for behavioral state decay."""

from saliencegate.benchmarks.state_decay.schema import (
    GENERATOR_VERSION,
    STATE_DECAY_SCENARIO_SCHEMA_VERSION,
    AllowedAction,
    CandidateMemory,
    ContinuationBranch,
    ContinuationOutcome,
    EvidenceCriteria,
    InterventionLabel,
    MemorySourceRef,
    OracleCriteria,
    PairedContinuation,
    Pivot,
    ScenarioFamily,
    StateDecayScenario,
    TrajectoryEvent,
)

__all__ = [
    "GENERATOR_VERSION",
    "STATE_DECAY_SCENARIO_SCHEMA_VERSION",
    "AllowedAction",
    "CandidateMemory",
    "ContinuationBranch",
    "ContinuationOutcome",
    "EvidenceCriteria",
    "InterventionLabel",
    "MemorySourceRef",
    "OracleCriteria",
    "PairedContinuation",
    "Pivot",
    "ScenarioFamily",
    "StateDecayScenario",
    "TrajectoryEvent",
]
