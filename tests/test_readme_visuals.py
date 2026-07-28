from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import get_args

import pytest
from scripts.check_public_docs import OBSERVATIONAL_FORBIDDEN_PATTERNS
from scripts.check_readme_visuals import (
    _OBSERVATIONAL_FORBIDDEN_PATTERNS,
    contrast_ratio,
    project_runtime_metadata_digest,
    runtime_source_digest,
    validate_capture_headline_render,
    validate_evidence_manifest,
    validate_markdown_images,
    validate_readme_assets,
    validate_svg_text,
)
from scripts.render_capture_headlines import (
    _SUMMARY_MAX_WIDTH,
    CaptureHeadlineFixtureError,
    _estimated_svg_text_width,
    load_capture_headline_fixture,
    render_capture_headlines_svg,
    validate_capture_headline_fixture,
)

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.normalization import normalize_capture_session_snapshot
from saliencegate.capture.report import CaptureReportHeadline, CaptureSessionReport
from saliencegate.capture.report import build_capture_session_report as build_capture_report
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureStore,
    CaptureStoreMode,
)
from saliencegate.domain import SignalType, canonical_json
from saliencegate.integrations.bootstrap import IntegrationBootstrap
from saliencegate.integrations.opencode import (
    OPENCODE_HOST_VERSION,
    OpenCodeCaptureAdapter,
)
from saliencegate.security import InstallationKey

ROOT = Path(__file__).parents[1]
MANIFEST = Path("benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.manifest.json")
REPORT = Path("benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json")
ASSETS = {
    "pipeline": Path("docs/assets/readme/pipeline.svg"),
    "capture": Path("docs/assets/readme/capture-headlines.svg"),
    "reference": Path("docs/assets/readme/reference-run.svg"),
}
CAPTURE_FIXTURE = Path("examples/capture/headline-results.json")


def _svg(body: str, *, labelledby: str = "sample-title sample-desc") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 600"
  role="img" aria-labelledby="{labelledby}">
  <title id="sample-title">Reference benchmark</title>
  <desc id="sample-desc">Measured memory and SQLite results</desc>
  <rect width="1600" height="600" fill="#f6f1e8"/>
  <text x="40" y="60" fill="#22211f" font-size="32">{body}</text>
</svg>"""


def test_reference_evidence_manifest_binds_every_measured_input() -> None:
    assert validate_evidence_manifest(ROOT, MANIFEST) == ()
    count, digest = runtime_source_digest(ROOT)
    assert count == 202
    assert digest == "83e6dcbdd50bbb8a05eb38c1494bc86a82e88a81ffc7bd52c306b48dcc014236"


def test_evidence_metadata_binding_ignores_editorial_packaging_fields(tmp_path: Path) -> None:
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(source, encoding="utf-8")
    baseline = project_runtime_metadata_digest(tmp_path)

    pyproject.write_text(
        source.replace(
            'readme = "docs/package-description.md"',
            'readme = "README-for-package-index.md"',
        ),
        encoding="utf-8",
    )
    assert project_runtime_metadata_digest(tmp_path) == baseline

    pyproject.write_text(
        source.replace('"pydantic>=2.11,<3"', '"pydantic>=2.12,<3"'),
        encoding="utf-8",
    )
    assert project_runtime_metadata_digest(tmp_path) != baseline


def test_live_readme_asset_set_passes_the_standalone_checker_contract() -> None:
    assert validate_readme_assets(ROOT) == ()


def test_svg_validator_accepts_a_minimal_accessible_editorial_asset() -> None:
    assert (
        validate_svg_text(
            Path("docs/assets/readme/reference-run.svg"),
            _svg("Five measured runs"),
            required_text=("reference", "memory", "sqlite"),
        )
        == ()
    )


def test_svg_validator_rejects_active_external_and_comment_content() -> None:
    unsafe = _svg("Five measured runs").replace(
        "</svg>",
        '<script>alert(1)</script><image href="data:image/png;base64,AAAA"/></svg>',
    )
    unsafe = "<!-- Editorial note -->\n" + unsafe

    findings = validate_svg_text(
        Path("docs/assets/readme/reference-run.svg"),
        unsafe,
        required_text=("reference", "memory", "sqlite"),
    )

    assert any("comment" in finding for finding in findings)
    assert any("forbidden element" in finding for finding in findings)
    assert any("embedded or external resource" in finding for finding in findings)


def test_svg_validator_rejects_empty_or_disconnected_accessibility_text() -> None:
    empty = _svg("Five measured runs", labelledby="sample-title missing-desc").replace(
        "Measured memory and SQLite results", "   "
    )

    findings = validate_svg_text(
        Path("docs/assets/readme/reference-run.svg"),
        empty,
        required_text=("reference", "memory", "sqlite"),
    )

    assert any("aria-labelledby" in finding for finding in findings)
    assert any("non-empty" in finding for finding in findings)
    assert any("asset-specific" in finding for finding in findings)


def test_svg_validator_rejects_small_text_and_event_handlers() -> None:
    unsafe = _svg("Five measured runs").replace('font-size="32"', 'font-size="18" onclick="x"')

    findings = validate_svg_text(
        Path("docs/assets/readme/reference-run.svg"),
        unsafe,
        required_text=("reference", "memory", "sqlite"),
    )

    assert any("28 SVG units" in finding for finding in findings)
    assert any("event handler" in finding for finding in findings)


def test_svg_validator_rejects_observational_efficacy_claims() -> None:
    findings = validate_svg_text(
        Path("docs/assets/readme/pipeline.svg"),
        _svg("Improves task success."),
    )

    assert any("task-success claim" in finding for finding in findings)


def test_svg_and_public_prose_checkers_share_the_observational_claim_boundary() -> None:
    def contract(patterns: object) -> frozenset[tuple[str, str]]:
        assert isinstance(patterns, tuple)
        return frozenset((pattern.pattern, description) for pattern, description in patterns)

    assert contract(_OBSERVATIONAL_FORBIDDEN_PATTERNS) == contract(OBSERVATIONAL_FORBIDDEN_PATTERNS)


def test_contrast_calculation_freezes_the_editorial_palette_boundary() -> None:
    assert contrast_ratio("#22211f", "#f6f1e8") >= 4.5
    assert contrast_ratio("#777777", "#f6f1e8") < 4.5


def test_markdown_images_must_be_local_and_have_meaningful_alt_text() -> None:
    valid = "![Five-run memory and SQLite reference](docs/assets/readme/reference-run.svg)"
    assert validate_markdown_images(Path("README.md"), valid) == ()

    external = "![Benchmark](https://example.test/chart.svg)"
    generic = "![Chart](docs/assets/readme/reference-run.svg)"
    external_findings = validate_markdown_images(Path("README.md"), external)
    generic_findings = validate_markdown_images(Path("README.md"), generic)
    assert any("external image" in finding for finding in external_findings)
    assert any("meaningful alt text" in finding for finding in generic_findings)


def _asset(name: str, *required_text: str) -> tuple[str, ET.Element]:
    path = ROOT / ASSETS[name]
    assert path.is_file(), f"missing {ASSETS[name]}"
    text = path.read_text(encoding="utf-8")
    assert validate_svg_text(ASSETS[name], text, required_text=required_text) == ()
    root = ET.fromstring(text)
    assert root.get("viewBox", "").split()[2] == "1600"
    assert root.get("width") == "100%"
    return text, root


def _metrics(root: ET.Element) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for element in root.iter():
        metric = element.get("data-metric")
        if metric is None:
            continue
        assert metric not in metrics
        metrics[metric] = " ".join("".join(element.itertext()).split())
    return metrics


def test_pipeline_visual_is_bound_to_the_current_capture_contract() -> None:
    text, root = _asset("pipeline", "capture", "hmac", "bounded report")
    assert len(CaptureProfile) == 4
    assert len(SignalType) == 9
    assert len(CaptureReportHeadline) == 3
    assert get_args(CaptureSessionReport.model_fields["raw_content_persisted"].annotation) == (
        False,
    )
    assert get_args(CaptureSessionReport.model_fields["model_calls"].annotation) == (0,)
    assert get_args(CaptureSessionReport.model_fields["decision_authority"].annotation) == (False,)
    assert _metrics(root) == {
        "pipeline.profile_count": str(len(CaptureProfile)),
        "pipeline.raw_content_persisted": "false",
        "pipeline.signal_type_count": str(len(SignalType)),
        "pipeline.headline_count": str(len(CaptureReportHeadline)),
        "pipeline.model_calls": "0",
        "pipeline.decision_authority": "false",
    }
    folded = " ".join(text.casefold().split())
    assert "capture first" in folded
    assert "hmac integrity is not encryption" in folded
    assert "no action control, memory edit, reminder, or provider call" in folded
    assert "provider causal order" in folded


def test_capture_headline_visual_is_an_exact_rendering_of_the_public_fixture() -> None:
    text, root = _asset(
        "capture",
        "synthetic",
        "memory review suggested",
        "no current evidence",
        "insufficient evidence",
    )
    fixture = load_capture_headline_fixture(ROOT / CAPTURE_FIXTURE)
    expected_metrics: dict[str, str] = {}
    cases = fixture["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        case_id = case["id"]
        expected = case["expected"]
        assert isinstance(case_id, str)
        assert isinstance(expected, dict)
        counts = expected["counts"]
        coverage = expected["coverage"]
        detectors = expected["detectors"]
        assert isinstance(counts, dict)
        assert isinstance(coverage, dict)
        assert isinstance(detectors, dict)
        repeated_action = detectors["repeated_action"]
        tool_error = detectors["tool_error"]
        repeated_failure = detectors["repeated_failure"]
        assert isinstance(repeated_action, dict)
        assert isinstance(tool_error, dict)
        assert isinstance(repeated_failure, dict)
        expected_metrics.update(
            {
                f"{case_id}.headline": str(expected["headline"]),
                f"{case_id}.action_identities": str(counts["action_identities"]),
                f"{case_id}.structured_results": str(counts["structured_results"]),
                f"{case_id}.detected_signals": str(counts["detected_signals"]),
                f"{case_id}.repeated_action.detected": str(repeated_action["detected_count"]),
                f"{case_id}.tool_error.detected": str(tool_error["detected_count"]),
                f"{case_id}.repeated_failure.support": str(repeated_failure["support"]),
                f"{case_id}.coverage_degraded": str(coverage["coverage_degraded"]).lower(),
                f"{case_id}.limit_count": str(len(coverage["limits"])),
            }
        )
    invariants = fixture["report_invariants"]
    assert isinstance(invariants, dict)
    expected_metrics.update(
        {
            "examples.model_calls": str(invariants["model_calls"]),
            "examples.decision_authority": str(invariants["decision_authority"]).lower(),
            "examples.confirmatory": str(invariants["confirmatory"]).lower(),
            "examples.evidence_level": str(invariants["evidence_level"]),
        }
    )
    assert _metrics(root) == expected_metrics
    assert render_capture_headlines_svg(fixture) == text
    assert validate_capture_headline_render(ROOT) == ()
    folded = " ".join(text.casefold().split())
    assert "not a real-world efficacy result" in folded


def test_capture_headline_summary_wrapping_stays_inside_each_card() -> None:
    _text, root = _asset("capture", "synthetic")
    card_left = {
        "memory-review-suggested": 60,
        "no-current-evidence": 570,
        "insufficient-evidence": 1080,
    }
    summaries: dict[str, list[ET.Element]] = {case_id: [] for case_id in card_left}
    for element in root.iter():
        case_id = element.get("data-summary")
        if case_id is not None:
            summaries[case_id].append(element)

    assert all(len(elements) == 3 for elements in summaries.values())
    for case_id, elements in summaries.items():
        for element in elements:
            text = " ".join("".join(element.itertext()).split())
            estimated_width = _estimated_svg_text_width(text)
            assert int(element.get("data-estimated-width", "-1")) == estimated_width
            assert estimated_width <= _SUMMARY_MAX_WIDTH
            assert int(element.get("x", "-1")) + estimated_width <= card_left[case_id] + 430


def _replace_fixture_value(
    value: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    current = value
    for part in path[:-1]:
        if isinstance(part, int):
            assert isinstance(current, list)
            current = current[part]
        else:
            assert isinstance(current, dict)
            current = current[part]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(current, list)
        current[final] = replacement
    else:
        assert isinstance(current, dict)
        current[final] = replacement


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("cases", 0, "expected", "counts", "action_identities"), 999, "counts contract"),
        (
            ("cases", 0, "expected", "detectors", "tool_error", "disposition"),
            "not_flagged",
            "detector contract",
        ),
        (
            ("cases", 2, "expected", "coverage", "limits"),
            ["capture_degraded"],
            "coverage contract",
        ),
        (("cases", 0, "events", 0, "event_id"), "repeated-start-x", "event contract"),
    ),
)
def test_capture_headline_fixture_rejects_semantically_incoherent_mutations(
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    fixture = deepcopy(load_capture_headline_fixture(ROOT / CAPTURE_FIXTURE))
    _replace_fixture_value(fixture, path, replacement)

    with pytest.raises(CaptureHeadlineFixtureError, match=message):
        validate_capture_headline_fixture(fixture)


def _replay_capture_case(
    tmp_path: Path,
    case: dict[str, object],
    *,
    ordinal: int,
) -> CaptureSessionReport:
    state_directory = tmp_path / str(case["id"])
    state_directory.mkdir(mode=0o700)
    locations = CaptureStoreLocations(
        platform="windows" if os.name == "nt" else "posix",
        state_directory=state_directory,
        database_path=state_directory / "capture.sqlite3",
        spool_directory=state_directory / "capture-spool",
    )
    installation_key = InstallationKey(b"synthetic-fixture-replay-key!!!!")
    context = CaptureDigestContext(installation_key)
    spool = CaptureSpool.open(locations, installation_key)
    initialize_capture_store(locations.database_path)

    profile_id = CaptureProfile.OPENCODE_PLUGIN_V1
    capability_digest = capture_capability_digest(capture_profile(profile_id))
    connection_id = "sg-" + str(ordinal) * 48
    bootstrap = IntegrationBootstrap(
        profile=profile_id,
        connection_id=connection_id,
        launcher_path=state_directory / ("capture-hook.cmd" if os.name == "nt" else "capture-hook"),
        capability_digest=capability_digest,
        bundle_digest="3" * 64,
        receipt_mac="4" * 64,
    )
    project_root = tmp_path / "synthetic-project"
    project_root.mkdir(exist_ok=True)
    adapter = OpenCodeCaptureAdapter(
        connection_id=connection_id,
        bootstrap=bootstrap,
        project_root=project_root,
    )
    native_session_id = case["native_session_id"]
    events = case["events"]
    assert isinstance(native_session_id, str)
    assert isinstance(events, list)
    payload = canonical_json(
        {
            "schema_version": "capture-batch/v1",
            "bootstrap": bootstrap.model_dump(mode="json", warnings="error"),
            "batch_id": str(ordinal + 3) * 64,
            "session_id": native_session_id,
            "chunk_index": 0,
            "chunk_count": 1,
            "events": events,
        }
    )

    with CaptureStore.open(
        locations.database_path,
        installation_key=installation_key,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=connection_id,
            project_digest=str(ordinal + 6) * 64,
            profile_id=profile_id,
            capability_manifest_digest=capability_digest,
            host_version=OPENCODE_HOST_VERSION,
        )
        store.transition_connection(
            connection_id=connection_id,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        receipt = store.append_transport_chunk(
            adapter.transport_chunk(payload, context=context),
            adapter.adapt_bytes(payload, context=context),
        )
        assert receipt.disposition.value == "admitted"
        snapshot = store.snapshot_session(
            connection_id,
            context.session_id(native_session_id.encode("utf-8")),
        )

    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=installation_key,
    )
    return build_capture_report(
        snapshot,
        normalization,
        installation_key=installation_key,
        spool=spool,
    )


def test_capture_headline_fixture_replays_through_the_real_opencode_report_path(
    tmp_path: Path,
) -> None:
    fixture = load_capture_headline_fixture(ROOT / CAPTURE_FIXTURE)
    invariants = fixture["report_invariants"]
    cases = fixture["cases"]
    assert isinstance(invariants, dict)
    assert isinstance(cases, list)
    observed_headlines: list[str] = []
    for ordinal, case in enumerate(cases, start=1):
        assert isinstance(case, dict)
        report = _replay_capture_case(tmp_path, case, ordinal=ordinal)
        expected = case["expected"]
        assert isinstance(expected, dict)
        counts = expected["counts"]
        coverage = expected["coverage"]
        detectors = expected["detectors"]
        assert isinstance(counts, dict)
        assert isinstance(coverage, dict)
        assert isinstance(detectors, dict)

        observed_headlines.append(report.headline.value)
        assert report.profile_id is CaptureProfile.OPENCODE_PLUGIN_V1
        assert report.host_version == OPENCODE_HOST_VERSION
        assert report.headline.value == expected["headline"]
        assert report.shadow_disposition.value == expected["shadow_disposition"]
        assert report.session_state.value == expected["session_state"]
        assert report.counts.model_dump(mode="json", warnings="error") == counts
        assert report.coverage.coverage_degraded is coverage["coverage_degraded"]
        assert [item.value for item in report.coverage.limits] == coverage["limits"]
        actual_detectors = {item.signal_type.value: item for item in report.detectors}
        for signal_type, expected_detector in detectors.items():
            assert isinstance(signal_type, str)
            assert isinstance(expected_detector, dict)
            actual = actual_detectors[signal_type]
            assert actual.support.value == expected_detector["support"]
            assert actual.disposition.value == expected_detector["disposition"]
            assert actual.detected_count == expected_detector["detected_count"]
        for field in ("evidence_level", "model_calls", "decision_authority", "confirmatory"):
            assert getattr(report, field) == invariants[field]

    assert observed_headlines == [
        "memory_review_suggested",
        "no_current_evidence",
        "insufficient_evidence",
    ]


def test_capture_headline_renderer_has_strict_write_check_and_fixture_modes(
    tmp_path: Path,
) -> None:
    fixture_path = ROOT / CAPTURE_FIXTURE
    output = tmp_path / "capture-headlines.svg"
    base = [
        sys.executable,
        "scripts/render_capture_headlines.py",
        "--fixture",
        str(fixture_path),
        "--output",
        str(output),
    ]
    written = subprocess.run(
        [*base, "--write"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stdout + written.stderr
    assert output.read_text(encoding="utf-8") == (ROOT / ASSETS["capture"]).read_text(
        encoding="utf-8"
    )
    checked = subprocess.run(
        [*base, "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    stale = subprocess.run(
        [*base, "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 1
    assert "visual is stale" in stale.stderr

    fixture = load_capture_headline_fixture(fixture_path)
    cases = fixture["cases"]
    assert isinstance(cases, list)
    repeated = cases[0]
    assert isinstance(repeated, dict)
    expected = repeated["expected"]
    assert isinstance(expected, dict)
    detectors = expected["detectors"]
    assert isinstance(detectors, dict)
    repeated_failure = detectors["repeated_failure"]
    assert isinstance(repeated_failure, dict)
    repeated_failure["support"] = "conditional"
    with pytest.raises(CaptureHeadlineFixtureError, match="repeated-failure boundary"):
        render_capture_headlines_svg(fixture)


def test_reference_visual_matches_every_tracked_measurement_and_budget() -> None:
    _text, root = _asset("reference", "memory", "sqlite", "local reference run")
    report = json.loads((ROOT / REPORT).read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for backend in ("memory", "sqlite"):
        result = report["backends"][backend]
        for ordinal, measurement in enumerate(result["measurements"], start=1):
            expected[f"{backend}.duration.{ordinal}"] = str(measurement["duration_seconds"])
        expected[f"{backend}.median"] = str(result["median_seconds"])
        expected[f"{backend}.time_budget"] = str(result["median_budget_seconds"])
        expected[f"{backend}.peak_rss"] = str(result["maximum_peak_rss_mib"])
        expected[f"{backend}.rss_budget"] = str(result["peak_rss_budget_mib"])
    expected.update(
        {
            "environment.platform": report["metadata"]["platform"],
            "environment.python": (
                f"{report['metadata']['python_implementation']} "
                f"{report['metadata']['python_version']}"
            ),
            "environment.cores": str(report["metadata"]["logical_core_count"]),
            "environment.memory_mib": str(report["metadata"]["memory_capacity_mib"]),
            "environment.runner_image": report["metadata"]["runner_image"],
        }
    )
    assert _metrics(root) == expected
    visual_text = " ".join("".join(root.itertext()).split())
    for backend in ("memory", "sqlite"):
        ratio = str(report["backends"][backend]["scaling_ratio_250_to_1000"])
        assert ratio not in visual_text


def test_visual_palette_uses_text_and_shape_contrast_without_color_only_labels() -> None:
    for name, required in (
        ("pipeline", ("capture",)),
        ("capture", ("synthetic",)),
        ("reference", ("local reference run",)),
    ):
        _text, root = _asset(name, *required)
        for element in root.iter():
            if element.get("data-surface") == "true":
                assert contrast_ratio("#22211f", element.get("fill", "")) >= 4.5
            if element.get("data-series") is not None:
                assert element.get("aria-label")
                assert contrast_ratio(element.get("fill", ""), "#f6f1e8") >= 3
