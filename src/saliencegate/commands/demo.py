from __future__ import annotations

from typing import Literal, Never

from pydantic import BaseModel, ConfigDict

from saliencegate.benchmarks.state_decay.diagnostic import (
    StateDecayDiagnosticError,
    run_state_decay_diagnostic,
)
from saliencegate.benchmarks.state_decay.schema import InterventionLabel
from saliencegate.domain import canonical_json, length_prefixed_sha256

CLI_DEMO_SCHEMA_VERSION: Literal["cli-demo-report/v1"] = "cli-demo-report/v1"
DEMO_RESULT_DIGEST: Literal["13704b753086925db1abfd7467f3e202edb202d60f620d150b1cdb6099c57d0f"] = (
    "13704b753086925db1abfd7467f3e202edb202d60f620d150b1cdb6099c57d0f"
)
_RESULT_DIGEST_DOMAIN = "saliencegate:demo:state-decay-smoke:v1"


class DemoCommandReport(BaseModel):
    """Stable aggregate report for the in-memory StateDecayBench diagnostic."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["cli-demo-report/v1"] = CLI_DEMO_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    suite_id: Literal["state-decay-smoke"] = "state-decay-smoke"
    evidence_level: Literal["synthetic_diagnostic"] = "synthetic_diagnostic"
    diagnostic: Literal[True] = True
    synthetic: Literal[True] = True
    confirmatory: Literal[False] = False
    external_claims_supported: Literal[False] = False
    external_claims_assessment: Literal["insufficient"] = "insufficient"
    scenario_count: Literal[32] = 32
    family_count: Literal[8] = 8
    intervene_count: Literal[16] = 16
    silence_count: Literal[16] = 16
    oracle_passed: Literal[32] = 32
    oracle_failed: Literal[0] = 0
    result_digest: Literal["13704b753086925db1abfd7467f3e202edb202d60f620d150b1cdb6099c57d0f"]


def _fail() -> Never:
    raise StateDecayDiagnosticError() from None


def _validated_report(value: object) -> DemoCommandReport:
    checked: DemoCommandReport | None = None
    try:
        if type(value) is DemoCommandReport:
            checked = DemoCommandReport.model_validate_json(canonical_json(value))
            if checked != value:
                checked = None
    except Exception:
        pass
    if checked is None:
        _fail()
    return checked


def run_demo() -> DemoCommandReport:
    """Run the frozen synthetic diagnostic without filesystem or external access."""

    report: DemoCommandReport | None = None
    try:
        diagnostic = run_state_decay_diagnostic()
        scenarios = diagnostic.scenarios
        oracle_results = diagnostic.oracle_results
        report = DemoCommandReport.model_validate(
            {
                "scenario_count": len(scenarios),
                "family_count": len({scenario.family for scenario in scenarios}),
                "intervene_count": sum(
                    scenario.label is InterventionLabel.INTERVENE for scenario in scenarios
                ),
                "silence_count": sum(
                    scenario.label is InterventionLabel.SILENCE for scenario in scenarios
                ),
                "oracle_passed": sum(result.matched is True for result in oracle_results),
                "oracle_failed": sum(result.matched is not True for result in oracle_results),
                "result_digest": length_prefixed_sha256(
                    canonical_json(scenarios),
                    canonical_json(oracle_results),
                    domain=_RESULT_DIGEST_DOMAIN,
                ),
            }
        )
    except Exception:
        pass
    if report is None:
        _fail()
    return report


def render_demo_json(report: DemoCommandReport) -> str:
    """Render one canonical JSON report line."""

    checked = _validated_report(report)
    return canonical_json(checked).decode("utf-8") + "\n"


def render_demo_human(report: DemoCommandReport) -> str:
    """Render the stable human report and its explicit evidence limitation."""

    checked = _validated_report(report)
    return (
        "SalienceGate offline demo\n"
        f"suite: {checked.suite_id}\n"
        "evidence: synthetic diagnostic\n"
        f"scenarios: {checked.scenario_count} across {checked.family_count} families\n"
        f"decisions: {checked.intervene_count} intervene, {checked.silence_count} silence\n"
        f"oracle: {checked.oracle_passed} passed, {checked.oracle_failed} failed\n"
        "confirmatory: no\n"
        f"external claims: {checked.external_claims_assessment}\n"
        f"result digest: {checked.result_digest}\n"
        "This verifies deterministic mechanics, not agent task efficacy.\n"
    )


__all__ = [
    "CLI_DEMO_SCHEMA_VERSION",
    "DEMO_RESULT_DIGEST",
    "DemoCommandReport",
    "render_demo_human",
    "render_demo_json",
    "run_demo",
]
