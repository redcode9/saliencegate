"""Provider-free public contracts for SalienceGate Shadow Mode."""

from saliencegate.shadow.adapters import ShadowTraceAdapter
from saliencegate.shadow.analyzer import ShadowAnalyzer, analyze_atif_bytes
from saliencegate.shadow.atif import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowEnvironmentBinding,
)
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
    ShadowTraceInputError,
)
from saliencegate.shadow.inputs import ShadowEventRef
from saliencegate.shadow.observation import ShadowEventResult, ShadowObservation
from saliencegate.shadow.report import ShadowRunReport, build_shadow_run_report
from saliencegate.shadow.session import ShadowSession
from saliencegate.shadow.trace import (
    ShadowTrace,
    ShadowTraceBinding,
    ShadowTraceDiagnostics,
)
from saliencegate.shadow.trace_report import (
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
    verify_shadow_trace_source,
)

__all__ = [
    "ATIFProfile",
    "ATIFShadowAdapter",
    "ShadowAnalyzer",
    "ShadowConfig",
    "ShadowConfigurationError",
    "ShadowEnvironmentBinding",
    "ShadowEventRef",
    "ShadowEventResult",
    "ShadowInputError",
    "ShadowInvariantError",
    "ShadowObservation",
    "ShadowRunReport",
    "ShadowSession",
    "ShadowStateError",
    "ShadowTrace",
    "ShadowTraceAdapter",
    "ShadowTraceBinding",
    "ShadowTraceDiagnostics",
    "ShadowTraceInputError",
    "ShadowTraceReport",
    "analyze_atif_bytes",
    "build_shadow_run_report",
    "decode_shadow_trace_report",
    "encode_shadow_trace_report",
    "verify_shadow_trace_source",
]
