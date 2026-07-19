from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import saliencegate.shadow as public_shadow
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


def test_shadow_root_exports_the_public_contract() -> None:
    assert public_shadow.__all__ == [
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
    assert public_shadow.ATIFProfile is ATIFProfile
    assert public_shadow.ATIFShadowAdapter is ATIFShadowAdapter
    assert public_shadow.ShadowAnalyzer is ShadowAnalyzer
    assert public_shadow.ShadowConfig is ShadowConfig
    assert public_shadow.ShadowConfigurationError is ShadowConfigurationError
    assert public_shadow.ShadowEventRef is ShadowEventRef
    assert public_shadow.ShadowEventResult is ShadowEventResult
    assert public_shadow.ShadowEnvironmentBinding is ShadowEnvironmentBinding
    assert public_shadow.ShadowInputError is ShadowInputError
    assert public_shadow.ShadowInvariantError is ShadowInvariantError
    assert public_shadow.ShadowObservation is ShadowObservation
    assert public_shadow.ShadowRunReport is ShadowRunReport
    assert public_shadow.ShadowSession is ShadowSession
    assert public_shadow.ShadowStateError is ShadowStateError
    assert public_shadow.ShadowTrace is ShadowTrace
    assert public_shadow.ShadowTraceAdapter is ShadowTraceAdapter
    assert public_shadow.ShadowTraceBinding is ShadowTraceBinding
    assert public_shadow.ShadowTraceDiagnostics is ShadowTraceDiagnostics
    assert public_shadow.ShadowTraceInputError is ShadowTraceInputError
    assert public_shadow.ShadowTraceReport is ShadowTraceReport
    assert public_shadow.analyze_atif_bytes is analyze_atif_bytes
    assert public_shadow.build_shadow_run_report is build_shadow_run_report
    assert public_shadow.decode_shadow_trace_report is decode_shadow_trace_report
    assert public_shadow.encode_shadow_trace_report is encode_shadow_trace_report
    assert public_shadow.verify_shadow_trace_source is verify_shadow_trace_source
    assert not hasattr(public_shadow, "ShadowReportRow")


def test_root_package_does_not_import_or_reexport_shadow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import saliencegate, sys; "
                "assert 'saliencegate.shadow' not in sys.modules; "
                "assert not hasattr(saliencegate, 'ShadowConfig')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_shadow_contract_has_no_forbidden_direct_imports() -> None:
    forbidden = (
        "anthropic",
        "harbor",
        "httpx",
        "openai",
        "openai_harmony",
        "saliencegate.commands",
        "saliencegate.intervention",
        "saliencegate.memory",
        "saliencegate.models",
        "saliencegate.ports.model_calls",
        "saliencegate.ports.models",
        "saliencegate.ports.two_phase",
        "saliencegate.repository",
        "saliencegate.runtime",
    )
    imported_by_path: dict[Path, set[str]] = {}
    for path in sorted(Path("src/saliencegate/shadow").glob("*.py")):
        imported: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        imported_by_path[path] = imported

    violations: list[tuple[Path, str]] = []
    for path, imported in imported_by_path.items():
        for name in imported:
            for prefix in forbidden:
                if name == prefix or name.startswith(f"{prefix}."):
                    if path.name == "session.py" and prefix == "saliencegate.repository":
                        continue
                    violations.append((path, name))

    assert violations == []


def test_shadow_import_remains_provider_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import saliencegate.shadow, sys; "
                "assert 'anthropic' not in sys.modules; "
                "assert 'harbor' not in sys.modules; "
                "assert 'httpx' not in sys.modules; "
                "assert 'openai' not in sys.modules; "
                "assert 'openai_harmony' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_example_uses_only_the_core_observational_surface() -> None:
    source = Path("examples/shadow_asyncio.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert "saliencegate.shadow" in imported
    assert all(
        name not in source
        for name in (
            "saliencegate.models",
            "saliencegate.memory",
            "saliencegate.runtime",
            "saliencegate.commands",
            "httpx",
            "openai_harmony",
            "os.environ",
            "getenv(",
            "socket",
        )
    )
