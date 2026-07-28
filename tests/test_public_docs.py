from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tomllib import loads

from scripts.check_public_docs import (
    OBSERVATIONAL_CLAIM_DOCUMENTS,
    PUBLIC_DOCUMENT_EXCLUSIONS,
    README_CLI_COMMANDS,
    README_INSTALL_COMMANDS,
    _public_paths,
    count_readme_command,
    scan_text,
    validate_markdown_links,
)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "docs/benchmarks/foundation-evidence.md",
    "docs/research-claims.md",
    "docs/reference/artifacts.md",
    "docs/reference/cli.md",
    "docs/reference/evaluation.md",
    "docs/reference/shadow-mode.md",
    "docs/reference/state-decay-v2-review.md",
    "docs/security.md",
    "examples/atif-shadow/README.md",
    "examples/capture/README.md",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/benchmark.yml",
)
README_HEADINGS = (
    "# SalienceGate",
    "## Install",
    "## Connect",
    "## Inspect and disconnect",
    "## How it works",
    "## Privacy and limits",
    "## Development",
    "## License",
)


def test_required_public_documents_exist() -> None:
    for relative_path in REQUIRED_DOCUMENTS:
        path = ROOT / relative_path
        assert path.is_file(), f"missing {relative_path}"
        assert path.stat().st_size > 0, f"empty {relative_path}"

    project = loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist_include = project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/docs/benchmarks" in sdist_include
    assert "/docs/research-claims.md" in sdist_include
    assert "/docs/reference" in sdist_include
    assert "/docs/security.md" in sdist_include
    assert "/examples" in sdist_include


def test_readme_uses_the_required_concrete_structure() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()
    assert all(lines.count(heading) == 1 for heading in README_HEADINGS)
    positions = [lines.index(heading) for heading in README_HEADINGS]
    assert positions == sorted(positions)
    assert 550 <= len(readme.split()) <= 850

    introduction = readme.split("\n## Install\n", maxsplit=1)[0]
    assert all(provider in introduction for provider in ("Codex", "Claude Code", "OpenCode", "Pi"))

    assert readme.count("![") == 3
    for asset in (
        "docs/assets/readme/saliencegate-logo.svg",
        "docs/assets/readme/install-flow.svg",
        "docs/assets/readme/pipeline.svg",
    ):
        assert readme.count(asset) == 1

    assert all(
        count_readme_command(readme, command) == 1
        for command in (*README_INSTALL_COMMANDS, *README_CLI_COMMANDS)
    )
    limits = readme.split("\n## Privacy and limits\n", maxsplit=1)[1].split(
        "\n## Development\n", maxsplit=1
    )[0]
    assert sum(line.startswith("- ") for line in limits.splitlines()) <= 5


def test_public_prose_inventory_includes_examples_and_repository_templates() -> None:
    inventory = {path.relative_to(ROOT).as_posix() for path in _public_paths(ROOT)}
    assert set(REQUIRED_DOCUMENTS) <= inventory
    assert (
        Path(".artifacts"),
        Path("reports/generated"),
    ) == PUBLIC_DOCUMENT_EXCLUSIONS


def test_capture_example_states_the_three_headlines_and_observational_boundary() -> None:
    example = (ROOT / "examples/capture/README.md").read_text(encoding="utf-8")
    flattened = " ".join(example.split())
    for fragment in (
        "opencode-plugin/v1",
        "memory_review_suggested",
        "no_current_evidence",
        "insufficient_evidence",
        "two `tool_error` signals",
        "one `repeated_action` signal",
        "`repeated_failure` is explicitly `unsupported`",
        "`model_calls=0`",
        "`decision_authority=false`",
        "`confirmatory=false`",
        "scripts/render_capture_headlines.py --check",
        "tests/test_readme_visuals.py -k capture_headline",
    ):
        assert fragment in flattened
    assert "do not establish task efficacy" in flattened


def test_relative_markdown_links_require_exact_tracked_regular_targets(tmp_path: Path) -> None:
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "README.md").write_text("# Readme\n", encoding="utf-8")
    (docs / "Case.md").write_text("# Case\n", encoding="utf-8")
    (docs / "result.svg").write_text("<svg/>\n", encoding="utf-8")
    tracked = ("README.md", "docs/guide.md", "docs/Case.md", "docs/result.svg")
    valid = """
[Root](../README.md?plain=1#top)
[Local heading](#details)
[Paper](https://arxiv.org/abs/2607.08716)
![Measured result](result.svg)
```markdown
[Example only](missing.md)
```
"""

    assert (
        validate_markdown_links(Path("docs/guide.md"), valid, tracked_paths=tracked, root=root)
        == ()
    )

    invalid = """
[Wrong case](case.md)
[Untracked](missing.md)
[Escape](../../private.md)
![Remote](https://example.test/result.svg)
"""
    findings = validate_markdown_links(
        Path("docs/guide.md"), invalid, tracked_paths=tracked, root=root
    )
    assert any("case-mismatched target docs/case.md" in finding for finding in findings)
    assert any("untracked target docs/missing.md" in finding for finding in findings)
    assert any("unsafe relative target" in finding for finding in findings)
    assert any("external image target" in finding for finding in findings)


def test_relative_markdown_links_reject_a_tracked_symlink_target(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs / "real.md").write_text("# Real\n", encoding="utf-8")
    (docs / "linked.md").symlink_to("real.md")

    findings = validate_markdown_links(
        Path("docs/guide.md"),
        "[Linked](linked.md)",
        tracked_paths=("docs/guide.md", "docs/linked.md"),
        root=tmp_path,
    )

    assert any("non-regular target docs/linked.md" in finding for finding in findings)


def test_review_guide_exposes_the_review_gate_and_evidence_boundary() -> None:
    guide = (ROOT / "docs/reference/state-decay-v2-review.md").read_text(encoding="utf-8")
    for command in ("build-pack", "review", "status", "build-envelope"):
        assert f"saliencegate-review {command}" in guide
    for boundary in (
        "I ACCEPT PUBLICATION",
        "180 current accepted envelopes",
        "append-only",
        "owner-controlled",
        "Generation boundary",
    ):
        assert boundary in guide


def test_feedback_docs_freeze_the_local_evaluation_and_activation_boundary() -> None:
    evaluation = (ROOT / "docs/reference/evaluation.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    evaluation_flat = " ".join(evaluation.split())
    cli_flat = " ".join(cli.split())

    for fragment in (
        "saliencegate feedback SESSION_ID",
        "memory-needed|not-memory-needed|uncertain",
        "insufficient_real_world_evidence",
        "`confirmatory=false`",
        "`decision_authority=false`",
        "even for a `DECLARED_E01` dataset",
        "external review remains required",
        "at least 200 human-adjudicated sessions in the locked final-test partition",
        "at least three projects and two providers in that partition",
        "at least 30 `memory-needed` and 30 `not-memory-needed` final-test labels",
        "temporal separation",
        "public seed `9f4c8dc1d7f87c2bf08bfc24f9cb6bb4de27c57fa3466a7a63d7f01e13961e7e`",
        "fixed-size resampling with replacement",
        "nearest ranks 50 and 1950",
        "upper endpoints rounded up",
        "observed raw denominator is at least 30",
        "`finite_sample_safety_bound=false`",
        "Provider strata below 30 final-test sessions are not emitted",
        "including system abstentions",
        "confusion cells and system-abstention cells are disjoint",
        "`evaluate_capture_feedback_dataset`",
        "there is no CLI evaluation command",
        "separate explicit Python API call",
        "`build_capture_feedback_dataset`",
        "`build_capture_feedback_export_record`",
        "`build_synthetic_capture_feedback_export_record`",
        "refuses to declare that origin as E01 evidence",
        "exact boolean `True`",
        "export nonce is exactly 32 bytes",
        "domain-separated pseudonyms",
        "`CaptureStore.list_feedback(label_freeze=...)`",
        "`labeled_at < label_freeze`",
        "rebuilds the report from those inputs",
        "exact authenticated spool",
        "report-selection policy",
        "explicit JSON `null` values",
        "originating installation key",
        "not probabilistic calibration",
        "external study declarations",
        "a non-empty development or tuning cohort",
        "label-revision cutoff",
        "consent and preregistration evidence",
        "whole-database rollback resistance",
        "cannot detect deletion of every feedback row",
        "public-key publication protocol",
        "no such path exists",
    ):
        assert fragment in evaluation_flat

    for fragment in (
        "saliencegate feedback SESSION_ID",
        "`feedback` records one bounded human label",
        "current project",
        "idempotent success",
        "`capture-feedback-receipt/v1`",
        "store contention exits 3",
        "never exports a dataset or runs classification evaluation",
        "evaluation.md",
    ):
        assert fragment in cli_flat

    assert "population prevalence" not in evaluation.casefold()
    assert "automatically enables" not in evaluation.casefold()


def test_shadow_docs_freeze_the_observational_sdk_and_cli_boundary() -> None:
    guide = (ROOT / "docs/reference/shadow-mode.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    for fragment in (
        "examples/shadow_asyncio.py",
        "from saliencegate.shadow import ShadowSession",
        "schema_version",
        '"kind":"run_start"',
        '"kind":"action"',
        '"kind":"tool_result"',
        '"kind":"run_end"',
        "any-detected-signal-baseline/v1",
        "repeated_action",
        "repeated_failure",
        "test_failure",
        "tool_error",
        "conflict",
        "context_shift",
        "irreversible_action",
        "stagnation",
        "stale_constraint",
        "10,000",
        "2 MiB",
        "64 MiB",
        "128 MiB",
        "0600",
        "HMAC integrity is not encryption",
        "descriptive_observational",
        "representativeness_supported=false",
        "task_efficacy_supported=false",
        "counterfactual_effect_supported=false",
    ):
        assert fragment in guide

    for fragment in (
        "saliencegate shadow analyze TRACE",
        "--run-id UUID4",
        "--output PATH",
        "--repository :memory:|PATH",
        "--capture-scope",
        "--task-scope-digest SHA256",
        "--lineage-scope-digest SHA256",
        "--capture-manifest-digest SHA256",
        "--source-adapter ID",
        "--replace",
        "--json",
        "shadow-run-report/v1",
        "shadow-command-report/v1",
        "10,000 rows",
        "Exit",
    ):
        assert fragment in cli

    assert "Shadow Mode SDK" in changelog
    assert "four supported deterministic detectors" in " ".join(changelog.split())

    folded_shadow_docs = guide.casefold()
    for forbidden_claim in (
        "improves task success",
        "task-success improvement",
        "saves tokens",
        "token savings",
        "causal effect",
        "calibrated trigger",
        "population prevalence",
        "false positive",
        "memory opportunity",
        "useful intervention",
        "avoided call",
    ):
        assert forbidden_claim not in folded_shadow_docs


def test_atif_docs_freeze_profiles_omissions_security_and_claims() -> None:
    guide = (ROOT / "docs/reference/shadow-mode.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    security = (ROOT / "docs/security.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs/research-claims.md").read_text(encoding="utf-8")
    examples = (ROOT / "examples/atif-shadow/README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    cli_flat = " ".join(cli.split())
    security_flat = " ".join(security.split())
    claims_flat = " ".join(claims.split())
    examples_flat = " ".join(examples.split())

    for fragment in (
        "analyze_atif_bytes",
        "harbor-codex/v1",
        "harbor-terminus-2/v1",
        "ATIF-v1.6",
        "ATIF-v1.7",
        "producer_authentication=none",
        "producer_claimed_structured",
        "write_stdin",
        "root trajectory segment",
        "Complete execution-session coverage is always false",
        "1,000",
        "64 MiB",
        "10,000",
        "shadow-trace-report/v1",
        "verify_shadow_trace_source",
        "Codex-version compatibility guarantee",
    ):
        assert fragment in guide

    for fragment in (
        "saliencegate shadow analyze-atif TRACE",
        "--profile {harbor-terminus-2-v1,harbor-codex-v1}",
        "--working-directory PATH",
        "--environment-digest SHA256",
        "capture_scope",
        "selected_events",
        "shadow-atif-command-report/v1",
        "local ledger-authentication state",
        "not a provider",
        "credential",
        "owner-private",
        "--replace",
    ):
        assert fragment in cli_flat

    for fragment in (
        "## Parsing and adaptation",
        "Adaptation is pure",
        "never executes",
        "provider credential variables",
        "socket",
        "HMAC integrity is not encryption",
        "producer_authentication=none",
        "caller-attested",
    ):
        assert fragment in security_flat

    for fragment in (
        "## Claims not made",
        "## Performance protocol",
        "improved or preserved task success",
        "token, monetary-cost, or latency reduction",
        "comparative superiority",
        "Codex CLI runtime version",
        "Remember When It Matters",
        "250-to-1,000",
    ):
        assert fragment in claims_flat

    for fragment in (
        "--profile harbor-codex-v1",
        "--profile harbor-terminus-2-v1",
        "python examples/atif-shadow/one_call.py",
        "install -m 600",
    ):
        assert fragment in examples_flat

    assert "analyze_atif_bytes" in changelog
    assert "saliencegate shadow analyze-atif" in changelog


def test_checker_rejects_credentials_claims_and_icons() -> None:
    credential = "OPENAI_API_KEY=sk-" + "x" * 32
    assert scan_text(Path("README.md"), credential)
    assert scan_text(Path("README.md"), "Launch ready 🚀")
    for forbidden in (
        "This is state of the art.",
        "This is best-in-class.",
        "This is production-ready.",
        "enterprise plan",
        "enterprise subscription",
        "workspace subscription",
        "organization plan",
        "AKIA" + "A" * 16,
        "ghp_" + "a" * 36,
        "github_pat_" + "a" * 82,
        "xoxb-" + "1" * 16,
        "AIza" + "A" * 35,
        "-----BEGIN PRIVATE KEY-----",
        "Stage " + "3 is next.",
        "Task " + "6B remains closed.",
    ):
        assert scan_text(Path("README.md"), forbidden)


def test_checker_rejects_unqualified_observational_claims_in_shadow_and_capture_docs() -> None:
    assert {
        "README.md",
        "docs/reference/shadow-mode.md",
        "docs/reference/evaluation.md",
        "examples/capture/README.md",
    } <= OBSERVATIONAL_CLAIM_DOCUMENTS
    for forbidden in (
        "Shadow Mode improves task success.",
        "Shadow Mode preserves task success.",
        "Shadow Mode saves tokens.",
        "Shadow Mode provides token savings.",
        "Shadow Mode proves a causal effect.",
        "Shadow Mode is a calibrated trigger.",
        "The report estimates population prevalence.",
        "The agent needs memory.",
        "A reminder would help.",
    ):
        assert scan_text(Path("docs/reference/shadow-mode.md"), forbidden)
        assert scan_text(Path("examples/capture/README.md"), forbidden)


def test_public_documents_pass_the_repository_checker() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_public_docs.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
