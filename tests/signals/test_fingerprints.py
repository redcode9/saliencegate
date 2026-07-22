from __future__ import annotations

import pickle
from dataclasses import asdict
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

import saliencegate.signals as public_signals
import saliencegate.signals.fingerprints as fingerprints_module
from saliencegate.domain import EventPhase, EventType, TraceEvent
from saliencegate.signals.base import AbstentionReason
from saliencegate.signals.fingerprints import (
    ActionFingerprint,
    FailureFingerprint,
    FingerprintUnavailableError,
    NormalizedTestFailure,
    ShellActionEvidence,
    ToolOutcome,
    ToolOutcomeStatus,
    action_fingerprint,
    classify_test_report,
    classify_tool_outcome,
    failure_fingerprint,
    normalize_test_id,
    parse_test_report,
    parse_tool_outcome,
)
from saliencegate.signals.fingerprints import (
    TestFailureEvidence as FailureEvidence,
)
from saliencegate.signals.fingerprints import (
    TestReport as ParsedTestReport,
)
from saliencegate.signals.fingerprints import (
    TestReportEvidence as ReportEvidence,
)
from saliencegate.signals.fingerprints import (
    TestReportStatus as ReportStatus,
)

SCHEMA_VERSION = "1.0"
ENVIRONMENT_DIGEST = "a" * 64
WORKING_DIRECTORY = "/workspace"
OPAQUE_ACTION_DIGEST = "b" * 64
OPAQUE_WORKSPACE_DIGEST = "c" * 64
OPAQUE_ENVIRONMENT_DIGEST = "d" * 64


def action_payload(**source: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "shell",
        "working_directory": WORKING_DIRECTORY,
        "environment_digest": ENVIRONMENT_DIGEST,
        **source,
    }


def action(event_factory: Any, sequence: int, value: dict[str, object]) -> object:
    return event_factory(
        sequence,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={"action": value},
    )


def opaque_action_payload(
    *,
    identity_authority: str = "exact",
    **overrides: object,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "opaque",
        "action_digest": OPAQUE_ACTION_DIGEST,
        "workspace_digest": OPAQUE_WORKSPACE_DIGEST,
        "environment_digest": OPAQUE_ENVIRONMENT_DIGEST,
        "identity_authority": identity_authority,
        **overrides,
    }


def opaque_action(
    event_factory: Any,
    sequence: int,
    value: dict[str, object] | None = None,
) -> object:
    return event_factory(
        sequence,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={"action_identity": opaque_action_payload() if value is None else value},
    )


def tool_event(event_factory: Any, sequence: int, **evidence: object) -> object:
    return event_factory(
        sequence,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": {"schema_version": SCHEMA_VERSION, **evidence}},
    )


def failure(test_id: str, **details: object) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "test_id": test_id, **details}


def report_event(
    event_factory: Any,
    sequence: int,
    *,
    status: str = "failed",
    failures: list[dict[str, object]] | None = None,
    framework: str = "pytest",
    event_type: EventType = EventType.TOOL_COMPLETION,
) -> object:
    selected_failures = (
        [failure("tests/test_api.py::test_timeout")] if failures is None else failures
    )
    return event_factory(
        sequence,
        event_type=event_type,
        payload={
            "test_report": {
                "schema_version": SCHEMA_VERSION,
                "framework": framework,
                "status": status,
                "failures": selected_failures,
            }
        },
    )


def unavailable_reason(call: Any) -> AbstentionReason:
    with pytest.raises(FingerprintUnavailableError) as error:
        call()
    return error.value.reason


def test_public_opaque_action_contract_is_strict_frozen_and_value_safe() -> None:
    model_type = fingerprints_module.OpaqueActionEvidence
    assert public_signals.OpaqueActionEvidence is model_type
    assert "OpaqueActionEvidence" in public_signals.__all__
    evidence = model_type(
        schema_version="1.0",
        kind="opaque",
        action_digest=OPAQUE_ACTION_DIGEST,
        workspace_digest=OPAQUE_WORKSPACE_DIGEST,
        environment_digest=OPAQUE_ENVIRONMENT_DIGEST,
        identity_authority="exact",
    )

    assert evidence.model_dump(mode="json") == opaque_action_payload()
    assert all(
        digest not in repr(evidence)
        for digest in (
            OPAQUE_ACTION_DIGEST,
            OPAQUE_WORKSPACE_DIGEST,
            OPAQUE_ENVIRONMENT_DIGEST,
        )
    )
    with pytest.raises(ValidationError):
        evidence.identity_authority = "coarse"

    for authority in ("exact", "coarse", "unavailable"):
        assert model_type(
            **opaque_action_payload(identity_authority=authority)
        ).identity_authority == (authority)

    class DigestSubclass(str):
        pass

    invalid_values = (
        {"action_digest": "A" * 64},
        {"workspace_digest": "c" * 63},
        {"environment_digest": DigestSubclass(OPAQUE_ENVIRONMENT_DIGEST)},
        {"identity_authority": "unknown"},
        {"kind": "shell"},
        {"unknown": True},
    )
    for invalid in invalid_values:
        with pytest.raises(ValidationError):
            model_type(**opaque_action_payload(**invalid))

    sensitive = "fixture-sensitive-opaque-digest"
    with pytest.raises(ValidationError) as caught:
        model_type(**opaque_action_payload(action_digest=sensitive))
    assert sensitive not in str(caught.value)
    assert sensitive not in repr(caught.value)


def test_exact_opaque_action_fingerprints_compare_all_three_digests(
    event_factory: Any,
) -> None:
    baseline = opaque_action(event_factory, 1)
    equivalent = opaque_action(event_factory, 2)
    changed_action = opaque_action(
        event_factory,
        3,
        opaque_action_payload(action_digest="e" * 64),
    )
    changed_workspace = opaque_action(
        event_factory,
        4,
        opaque_action_payload(workspace_digest="e" * 64),
    )
    changed_environment = opaque_action(
        event_factory,
        5,
        opaque_action_payload(environment_digest="e" * 64),
    )

    baseline_fingerprint = action_fingerprint(baseline)
    assert baseline_fingerprint.execution_mode == "opaque"
    assert baseline_fingerprint == action_fingerprint(equivalent)
    assert baseline_fingerprint != action_fingerprint(changed_action)
    assert baseline_fingerprint != action_fingerprint(changed_workspace)
    assert baseline_fingerprint != action_fingerprint(changed_environment)
    assert all(
        digest not in repr(baseline_fingerprint)
        for digest in (
            OPAQUE_ACTION_DIGEST,
            OPAQUE_WORKSPACE_DIGEST,
            OPAQUE_ENVIRONMENT_DIGEST,
        )
    )

    coarse = fingerprints_module.OpaqueActionEvidence(
        **opaque_action_payload(identity_authority="coarse")
    )
    with pytest.raises(ValueError, match="opaque action fingerprint is invalid"):
        ActionFingerprint._from_opaque(coarse)


def test_opaque_and_shell_action_namespaces_never_compare_equal(event_factory: Any) -> None:
    opaque = opaque_action(event_factory, 1)
    shell = action(
        event_factory,
        2,
        action_payload(
            argv=(OPAQUE_ACTION_DIGEST,),
            working_directory=OPAQUE_WORKSPACE_DIGEST,
            environment_digest=OPAQUE_ENVIRONMENT_DIGEST,
        ),
    )

    assert action_fingerprint(opaque) != action_fingerprint(shell)


@pytest.mark.parametrize("identity_authority", ("coarse", "unavailable"))
def test_non_exact_opaque_action_identity_abstains_without_exposing_digests(
    event_factory: Any,
    identity_authority: str,
) -> None:
    exact_fingerprint = action_fingerprint(opaque_action(event_factory, 1))
    event = opaque_action(
        event_factory,
        2,
        opaque_action_payload(identity_authority=identity_authority),
    )

    assert exact_fingerprint == action_fingerprint(opaque_action(event_factory, 3))
    with pytest.raises(FingerprintUnavailableError) as caught:
        action_fingerprint(event)

    assert caught.value.reason is AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert all(
        digest not in repr(caught.value)
        for digest in (
            OPAQUE_ACTION_DIGEST,
            OPAQUE_WORKSPACE_DIGEST,
            OPAQUE_ENVIRONMENT_DIGEST,
        )
    )


def test_action_fingerprint_rejects_ambiguous_shell_and_opaque_namespaces(
    event_factory: Any,
) -> None:
    event = event_factory(
        1,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={
            "action": action_payload(argv=("echo", "ok")),
            "action_identity": opaque_action_payload(),
        },
    )

    assert unavailable_reason(lambda: action_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_simple_ascii_shell_whitespace_converges_but_argv_remains_distinct(
    event_factory: Any,
) -> None:
    first = action(
        event_factory,
        1,
        action_payload(command="pytest\t-q   tests/a.py"),
    )
    equivalent_shell = action(
        event_factory,
        2,
        action_payload(command="pytest -q tests/a.py"),
    )
    structured_argv = action(
        event_factory,
        3,
        action_payload(argv=("pytest", "-q", "tests/a.py")),
    )

    assert action_fingerprint(first) == action_fingerprint(equivalent_shell)
    assert action_fingerprint(first) != action_fingerprint(structured_argv)


def test_unicode_whitespace_in_structured_argv_remains_semantic(
    event_factory: Any,
) -> None:
    em_space = action(event_factory, 1, action_payload(argv=("printf", "a\u2003b")))
    ascii_space = action(event_factory, 2, action_payload(argv=("printf", "a b")))

    assert action_fingerprint(em_space) != action_fingerprint(ascii_space)


def test_non_ascii_raw_shell_whitespace_fails_closed(event_factory: Any) -> None:
    event = action(event_factory, 1, action_payload(command="pytest\u2003-q tests/a.py"))

    assert unavailable_reason(lambda: action_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_unicode_normalization_is_never_implicit(event_factory: Any) -> None:
    composed = action(event_factory, 1, action_payload(argv=("echo", "caf\u00e9")))
    decomposed = action(event_factory, 2, action_payload(argv=("echo", "cafe\u0301")))

    assert action_fingerprint(composed) != action_fingerprint(decomposed)


def test_only_allowlisted_pytest_flags_before_terminator_are_commutative(
    event_factory: Any,
) -> None:
    safe_left = action(
        event_factory,
        1,
        action_payload(argv=("pytest", "--verbose", "-q", "tests")),
    )
    safe_right = action(
        event_factory,
        2,
        action_payload(argv=("pytest", "-q", "--verbose", "tests")),
    )
    after_terminator = action(
        event_factory,
        3,
        action_payload(argv=("pytest", "--", "-q")),
    )
    before_terminator = action(
        event_factory,
        4,
        action_payload(argv=("pytest", "-q", "--")),
    )

    assert action_fingerprint(safe_left) == action_fingerprint(safe_right)
    assert action_fingerprint(after_terminator) != action_fingerprint(before_terminator)


def test_safe_flags_move_across_positionals_but_not_unknown_options(
    event_factory: Any,
) -> None:
    option_value = action(
        event_factory,
        1,
        action_payload(argv=("pytest", "--override-ini", "-q")),
    )
    moved_before_option = action(
        event_factory,
        2,
        action_payload(argv=("pytest", "-q", "--override-ini")),
    )
    unknown_value_left = action(
        event_factory,
        5,
        action_payload(argv=("pytest", "--custom", "-v", "-q")),
    )
    unknown_value_right = action(
        event_factory,
        6,
        action_payload(argv=("pytest", "--custom", "-q", "-v")),
    )
    after_positional = action(
        event_factory,
        3,
        action_payload(argv=("pytest", "tests", "-q")),
    )
    before_positional = action(
        event_factory,
        4,
        action_payload(argv=("pytest", "-q", "tests")),
    )

    assert action_fingerprint(option_value) != action_fingerprint(moved_before_option)
    assert action_fingerprint(unknown_value_left) != action_fingerprint(unknown_value_right)
    assert action_fingerprint(after_positional) == action_fingerprint(before_positional)


def test_python_module_prefix_is_preserved_and_ambiguous_order_does_not_collapse(
    event_factory: Any,
) -> None:
    left = action(
        event_factory,
        1,
        action_payload(argv=("python", "-m", "pytest", "--verbose", "-q", "tests")),
    )
    right = action(
        event_factory,
        2,
        action_payload(argv=("python", "-m", "pytest", "-q", "--verbose", "tests")),
    )
    ambiguous = action(
        event_factory,
        3,
        action_payload(argv=("python", "-q", "-m", "pytest", "tests")),
    )

    assert action_fingerprint(left) == action_fingerprint(right)
    assert action_fingerprint(left) != action_fingerprint(ambiguous)


def test_unknown_and_destructive_flag_orders_remain_distinct(event_factory: Any) -> None:
    pairs = (
        (("custom-cli", "-q", "-v"), ("custom-cli", "-v", "-q")),
        ((r"/tmp/custom\pytest", "-q", "-v"), (r"/tmp/custom\pytest", "-v", "-q")),
        (("/tmp/pytest", "-q", "-v"), ("/tmp/pytest", "-v", "-q")),
        (("rm", "-r", "-f", "target"), ("rm", "-f", "-r", "target")),
        (("rm", "-r", "-f", "target"), ("rm", "-rf", "target")),
    )

    for sequence, (left, right) in enumerate(pairs, start=1):
        first = action(event_factory, sequence * 2, action_payload(argv=left))
        second = action(event_factory, sequence * 2 + 1, action_payload(argv=right))
        assert action_fingerprint(first) != action_fingerprint(second)


def test_working_directory_and_environment_are_required_and_semantic(
    event_factory: Any,
) -> None:
    base = action_payload(argv=("pytest", "-q"))
    first = action(event_factory, 1, base)
    changed_directory = action(event_factory, 2, {**base, "working_directory": "/other"})
    changed_environment = action(event_factory, 3, {**base, "environment_digest": "b" * 64})
    missing_directory = action(
        event_factory,
        4,
        {key: value for key, value in base.items() if key != "working_directory"},
    )

    assert action_fingerprint(first) != action_fingerprint(changed_directory)
    assert action_fingerprint(first) != action_fingerprint(changed_environment)
    assert unavailable_reason(lambda: action_fingerprint(missing_directory)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


@pytest.mark.parametrize(
    "command",
    (
        "echo $HOME",
        "echo '$HOME'",
        "echo *.py",
        "echo ok; true",
        "echo 'a|b'",
        "echo $(whoami)",
        "echo foo\nbar",
        "echo #comment",
        "echo !42",
        "echo %PATH%",
        "echo ^value",
        "FOO=bar true",
        "'FOO=bar' true",
        r"FOO\=bar true",
        "echo \x00value",
    ),
)
def test_raw_shell_expansion_and_metacharacters_fail_closed(
    event_factory: Any,
    command: str,
) -> None:
    event = action(event_factory, 1, action_payload(command=command))

    assert unavailable_reason(lambda: action_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_redaction_is_reported_before_shell_blacklist(event_factory: Any) -> None:
    event = action(event_factory, 1, action_payload(command="echo [REDACTED] | true"))

    assert unavailable_reason(lambda: action_fingerprint(event)) is (
        AbstentionReason.REDACTED_EQUIVALENCE_INPUT
    )


@pytest.mark.parametrize(
    "value",
    (
        action_payload(command="unterminated '"),
        action_payload(command="   "),
        action_payload(command="''"),
        action_payload(command='echo ""'),
        action_payload(command="echo ok", argv=("echo", "ok")),
        action_payload(argv=("",)),
        action_payload(command=" ".join(("x",) * 257)),
        action_payload(command="x" * (16 * 1_024 + 1)),
        action_payload(command="\U0001f642" * 5_000),
        action_payload(command="x" * (128 * 1_024 + 1)),
        {"kind": "shell", "command": "echo ok"},
        {**action_payload(command="echo ok"), "schema_version": "2.0"},
        {**action_payload(command="echo ok"), "unknown": True},
        {**action_payload(command="echo ok"), "unknown": 1.5},
        {
            **action_payload(command="echo ok"),
            "unknown": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}},
        },
    ),
)
def test_malformed_unversioned_or_oversized_actions_abstain_safely(
    event_factory: Any,
    value: dict[str, object],
) -> None:
    event = action(event_factory, 1, value)

    assert unavailable_reason(lambda: action_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_action_fingerprint_is_opaque_and_not_a_pydantic_record(event_factory: Any) -> None:
    secret = "fixture-sensitive-argument"
    event = action(event_factory, 1, action_payload(argv=("echo", secret)))
    fingerprint = action_fingerprint(event)

    assert secret not in repr(fingerprint)
    assert secret not in str(fingerprint)
    assert secret not in repr(asdict(fingerprint))
    assert not hasattr(fingerprint, "tokens")
    assert not hasattr(fingerprint, "model_dump")
    assert not hasattr(fingerprint, "model_dump_json")
    with pytest.raises(TypeError):
        pickle.dumps(fingerprint)


def test_invalid_action_payload_does_not_survive_as_exception_context(
    event_factory: Any,
) -> None:
    secret = "context-sensitive-command"
    event = action(
        event_factory,
        1,
        action_payload(command=secret, argv=(secret,)),
    )

    with pytest.raises(FingerprintUnavailableError) as caught:
        action_fingerprint(event)

    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert secret not in repr(caught.value)


def test_forged_mapping_subclasses_are_rejected_without_iteration(event_factory: Any) -> None:
    class ExplosiveMapping(dict[str, object]):
        def __contains__(self, key: object) -> bool:
            raise AssertionError("mapping subclass was consulted")

        def items(self) -> object:
            raise AssertionError("mapping subclass was iterated")

    valid = action(event_factory, 1, action_payload(command="echo ok"))
    forged_root = valid.model_copy(update={"payload": ExplosiveMapping()})
    forged_nested = valid.model_copy(
        update={"payload": MappingProxyType({"action": ExplosiveMapping()})}
    )

    for event in (forged_root, forged_nested):
        assert unavailable_reason(lambda event=event: action_fingerprint(event)) is (
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID
        )


def test_public_action_contract_is_strict_frozen_and_bounded() -> None:
    evidence = ShellActionEvidence(
        schema_version="1.0",
        kind="shell",
        argv=("echo", "ok"),
        working_directory=WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
    )
    assert "echo" not in repr(evidence)
    with pytest.raises(ValidationError):
        ShellActionEvidence(
            schema_version="1.0",
            kind="shell",
            argv=("x" * 16_000,) * 17,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        )
    with pytest.raises(ValidationError):
        ShellActionEvidence(
            schema_version="1.0",
            kind="shell",
            argv=("echo",),
            working_directory=WORKING_DIRECTORY,
            environment_digest="bad",
        )
    with pytest.raises(ValidationError):
        ShellActionEvidence(
            schema_version="1.0",
            kind="shell",
            argv=("\ud800",),
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        )
    with pytest.raises(ValidationError):
        evidence.kind = "shell"


def test_action_missing_or_wrong_event_type_has_sanitized_reason(event_factory: Any) -> None:
    missing = event_factory(1, event_type=EventType.ACTION_PROPOSAL)
    wrong_type = event_factory(
        2,
        event_type=EventType.OBSERVATION,
        payload={"action": action_payload(command="echo ok")},
    )

    assert unavailable_reason(lambda: action_fingerprint(missing)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )
    assert unavailable_reason(lambda: action_fingerprint(wrong_type)) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )


def test_present_null_envelopes_are_invalid_not_missing(event_factory: Any) -> None:
    null_action = event_factory(
        1,
        event_type=EventType.ACTION_PROPOSAL,
        payload={"action": None},
    )
    null_tool = event_factory(
        2,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": None},
    )
    null_report = event_factory(
        3,
        event_type=EventType.OBSERVATION,
        payload={"test_report": None},
    )

    for call in (
        lambda: action_fingerprint(null_action),
        lambda: classify_tool_outcome(null_tool),
        lambda: classify_test_report(null_report),
    ):
        assert unavailable_reason(call) is AbstentionReason.STRUCTURED_EVIDENCE_INVALID


def test_every_on_wire_envelope_requires_explicit_schema_v1(event_factory: Any) -> None:
    legacy_action = action(
        event_factory,
        1,
        {
            "kind": "shell",
            "command": "echo ok",
            "working_directory": WORKING_DIRECTORY,
            "environment_digest": ENVIRONMENT_DIGEST,
        },
    )
    legacy_tool = event_factory(
        2,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": {"exit_status": 2}},
    )
    legacy_report = event_factory(
        3,
        event_type=EventType.OBSERVATION,
        payload={
            "test_report": {
                "framework": "pytest",
                "status": "failed",
                "failures": [failure("tests/a.py::test_a")],
            }
        },
    )
    legacy_failure = report_event(
        event_factory,
        4,
        failures=[{"test_id": "tests/a.py::test_a"}],
    )

    for call in (
        lambda: action_fingerprint(legacy_action),
        lambda: classify_tool_outcome(legacy_tool),
        lambda: classify_test_report(legacy_report),
        lambda: classify_test_report(legacy_failure),
    ):
        assert unavailable_reason(call) is AbstentionReason.STRUCTURED_EVIDENCE_INVALID


def test_test_identifier_normalization_is_minimal_and_byte_conservative() -> None:
    assert normalize_test_id("./tests/api.py::test_timeout") == "tests/api.py::test_timeout"
    preserved = (
        "/tests/api.py::test_timeout",
        r"\\server\tests\api.py::test_timeout",
        r"tests\api.py::test_timeout",
        "tests/api.py::test_x[a:: b]",
        " tests/api.py :: test_timeout ",
        ".//tmp/test_api.py::test_timeout",
        "././tests/api.py::test_timeout",
        r"./C:\tests\api.py::test_timeout",
    )
    assert tuple(normalize_test_id(value) for value in preserved) == preserved
    assert normalize_test_id(".//tmp/test_api.py::test_timeout") != normalize_test_id(
        "/tmp/test_api.py::test_timeout"
    )
    assert normalize_test_id(r"./C:\tests\api.py::test_timeout") != normalize_test_id(
        r"C:\tests\api.py::test_timeout"
    )
    assert all(
        normalize_test_id(normalize_test_id(value)) == normalize_test_id(value)
        for value in preserved
    )
    assert normalize_test_id("tests/caf\u00e9.py::test") != normalize_test_id(
        "tests/cafe\u0301.py::test"
    )


@pytest.mark.parametrize(
    "value",
    (
        "../outside.py::test_x",
        "tests/../outside.py::test_x",
        r"tests\..\outside.py::test_x",
        "::test_x",
        "tests/api.py::",
        "./",
        "tests/api.py::[REDACTED]",
        "tests/api.py::test\x00x",
        "\ud800",
    ),
)
def test_invalid_ambiguous_or_redacted_test_identifiers_abstain(value: str) -> None:
    reason = unavailable_reason(lambda: normalize_test_id(value))
    expected = (
        AbstentionReason.REDACTED_EQUIVALENCE_INPUT
        if "[REDACTED]" in value
        else AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )
    assert reason is expected


def test_report_order_is_stable_but_framework_and_details_remain_exact(
    event_factory: Any,
) -> None:
    failures = [
        failure(
            "./tests/b.py::test_b",
            failure_type="AssertionError",
            signature="expected  1\n got 2",
        ),
        failure(
            "tests/a.py::test_a",
            failure_type="ValueError",
            signature="bad value",
        ),
    ]
    first = report_event(event_factory, 1, failures=failures)
    reordered = report_event(event_factory, 2, failures=list(reversed(failures)))
    changed_whitespace = report_event(
        event_factory,
        3,
        failures=[{**failures[0], "signature": "expected 1 got 2"}, failures[1]],
    )
    changed_case = report_event(event_factory, 4, failures=failures, framework="PyTest")

    assert parse_test_report(first) == parse_test_report(reordered)
    assert failure_fingerprint(first) == failure_fingerprint(reordered)
    assert failure_fingerprint(first) != failure_fingerprint(changed_whitespace)
    assert failure_fingerprint(first) != failure_fingerprint(changed_case)


def test_minimal_test_classification_ignores_redacted_details_and_duplicates(
    event_factory: Any,
) -> None:
    item = failure(
        "tests/a.py::test_a",
        failure_type="[REDACTED]",
        signature="[REDACTED]",
    )
    event = report_event(event_factory, 1, failures=[item, item])

    assert classify_test_report(event) is ReportStatus.FAILED
    assert unavailable_reason(lambda: parse_test_report(event)) is (
        AbstentionReason.REDACTED_EQUIVALENCE_INPUT
    )


def test_duplicate_complete_test_failures_are_only_invalid_for_equivalence(
    event_factory: Any,
) -> None:
    item = failure(
        "tests/a.py::test_a",
        failure_type="AssertionError",
        signature="nope",
    )
    event = report_event(event_factory, 1, failures=[item, item])

    assert classify_test_report(event) is ReportStatus.FAILED
    assert unavailable_reason(lambda: parse_test_report(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_test_report_contract_rejects_status_version_and_size_errors(
    event_factory: Any,
) -> None:
    passed_with_failure = report_event(
        event_factory,
        1,
        status="passed",
        failures=[failure("tests/a.py::test_a")],
    )
    failed_without_failure = report_event(event_factory, 2, failures=[])
    unsupported_nested = report_event(
        event_factory,
        3,
        failures=[{"schema_version": "2.0", "test_id": "tests/a.py::test_a"}],
    )
    oversized = report_event(
        event_factory,
        4,
        failures=[
            failure(f"tests/{index}.py::test", signature="x" * 120_000) for index in range(4)
        ],
    )

    for event in (passed_with_failure, failed_without_failure, unsupported_nested, oversized):
        assert unavailable_reason(lambda event=event: classify_test_report(event)) is (
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID
        )


def test_public_test_report_contract_enforces_aggregate_bound_before_nesting() -> None:
    oversized_failures = tuple(
        FailureEvidence(
            schema_version="1.0",
            test_id=f"tests/{index}.py::test",
            signature="x" * 120_000,
        )
        for index in range(4)
    )

    with pytest.raises(ValidationError):
        ReportEvidence(
            schema_version="1.0",
            framework="pytest",
            status="failed",
            failures=oversized_failures,
        )


def test_passing_report_is_classified_but_not_a_failure_fingerprint(
    event_factory: Any,
) -> None:
    event = report_event(
        event_factory,
        1,
        status="passed",
        failures=[],
        event_type=EventType.OBSERVATION,
    )

    assert classify_test_report(event) is ReportStatus.PASSED
    assert parse_test_report(event).status is ReportStatus.PASSED
    assert unavailable_reason(lambda: failure_fingerprint(event)) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )


def test_incomplete_failure_has_no_repetition_fingerprint(event_factory: Any) -> None:
    event = report_event(
        event_factory,
        1,
        failures=[failure("tests/a.py::test_a", failure_type="AssertionError")],
    )

    assert classify_test_report(event) is ReportStatus.FAILED
    assert unavailable_reason(lambda: failure_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )


def test_test_report_requires_applicable_event_and_structured_namespace(
    event_factory: Any,
) -> None:
    free_text = event_factory(
        1,
        event_type=EventType.OBSERVATION,
        payload={"message": "FAILED tests/api.py::test_timeout"},
    )
    wrong_type = report_event(event_factory, 2, event_type=EventType.ACTION_PROPOSAL)

    assert unavailable_reason(lambda: parse_test_report(free_text)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )
    assert unavailable_reason(lambda: parse_test_report(wrong_type)) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )


def test_report_view_repr_is_safe_and_non_pydantic(event_factory: Any) -> None:
    secret = "sensitive-signature"
    event = report_event(
        event_factory,
        1,
        failures=[
            failure(
                "tests/a.py::test_a",
                failure_type="AssertionError",
                signature=secret,
            )
        ],
    )
    report = parse_test_report(event)
    fingerprint = failure_fingerprint(event)

    assert secret not in repr(report)
    assert secret not in repr(report.failures[0])
    assert secret not in repr(fingerprint)
    assert secret not in repr(asdict(fingerprint))
    assert not hasattr(report, "model_dump_json")
    assert not hasattr(fingerprint, "model_dump_json")
    with pytest.raises(TypeError):
        pickle.dumps(fingerprint)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    (
        ({"exit_status": 2}, ToolOutcomeStatus.FAILED),
        ({"exception_type": "TimeoutError"}, ToolOutcomeStatus.FAILED),
        ({"error_code": "E_TIMEOUT"}, ToolOutcomeStatus.FAILED),
        ({"failure_signature": "connection closed"}, ToolOutcomeStatus.FAILED),
        ({"exit_status": 0}, ToolOutcomeStatus.SUCCEEDED),
        ({"status": "failed"}, ToolOutcomeStatus.FAILED),
        ({"status": "failed", "exit_status": 0}, ToolOutcomeStatus.FAILED),
        ({"status": "succeeded"}, ToolOutcomeStatus.SUCCEEDED),
    ),
)
def test_tool_status_is_optional_and_inferred_from_structured_evidence(
    event_factory: Any,
    evidence: dict[str, object],
    expected: ToolOutcomeStatus,
) -> None:
    event = tool_event(event_factory, 1, **evidence)

    assert classify_tool_outcome(event) is expected
    assert parse_tool_outcome(event).status is expected


@pytest.mark.parametrize(
    "evidence",
    (
        {"status": "succeeded", "exit_status": 2},
        {"status": "succeeded", "exception_type": "TimeoutError"},
        {"status": "succeeded", "error_code": "E_FAIL"},
        {"status": "succeeded", "failure_signature": "failed"},
        {"exit_status": 0, "failure_signature": "failed"},
        {},
        {"status": True},
        {"schema_version": "2.0", "exit_status": 2},
    ),
)
def test_tool_contract_rejects_contradictory_untyped_or_unversioned_evidence(
    event_factory: Any,
    evidence: dict[str, object],
) -> None:
    payload = evidence if "schema_version" in evidence else {"schema_version": "1.0", **evidence}
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": payload},
    )

    assert unavailable_reason(lambda: classify_tool_outcome(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_minimal_tool_classification_ignores_redacted_optional_detail(
    event_factory: Any,
) -> None:
    event = tool_event(
        event_factory,
        1,
        status="failed",
        error_code="[REDACTED]",
    )

    assert classify_tool_outcome(event) is ToolOutcomeStatus.FAILED
    assert unavailable_reason(lambda: parse_tool_outcome(event)) is (
        AbstentionReason.REDACTED_EQUIVALENCE_INPUT
    )


def test_tool_view_repr_is_safe_and_non_pydantic(event_factory: Any) -> None:
    secret = "sensitive-tool-signature"
    event = tool_event(event_factory, 1, status="failed", failure_signature=secret)
    outcome = parse_tool_outcome(event)

    assert secret not in repr(outcome)
    assert not hasattr(outcome, "model_dump_json")


def test_tool_failure_details_are_byte_exact(event_factory: Any) -> None:
    first = tool_event(
        event_factory,
        1,
        status="failed",
        exit_status=2,
        failure_signature=" connection\n closed ",
    )
    whitespace_changed = tool_event(
        event_factory,
        2,
        status="failed",
        exit_status=2,
        failure_signature="connection closed",
    )
    composed = tool_event(
        event_factory,
        3,
        status="failed",
        error_code="caf\u00e9",
    )
    decomposed = tool_event(
        event_factory,
        4,
        status="failed",
        error_code="cafe\u0301",
    )

    assert failure_fingerprint(first) != failure_fingerprint(whitespace_changed)
    assert failure_fingerprint(composed) != failure_fingerprint(decomposed)


def test_logical_zero_exit_remains_distinct_from_missing_exit(event_factory: Any) -> None:
    zero = tool_event(
        event_factory,
        1,
        status="failed",
        exit_status=0,
        error_code="E_LOGICAL",
    )
    missing = tool_event(
        event_factory,
        2,
        status="failed",
        error_code="E_LOGICAL",
    )

    assert failure_fingerprint(zero) != failure_fingerprint(missing)


def test_success_and_detail_free_failure_have_no_failure_fingerprint(
    event_factory: Any,
) -> None:
    success = tool_event(event_factory, 1, exit_status=0)
    generic_failure = tool_event(event_factory, 2, status="failed")
    logical_failure_without_identity = tool_event(
        event_factory,
        3,
        status="failed",
        exit_status=0,
    )

    assert unavailable_reason(lambda: failure_fingerprint(success)) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )
    assert unavailable_reason(lambda: failure_fingerprint(generic_failure)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )
    assert unavailable_reason(lambda: failure_fingerprint(logical_failure_without_identity)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )


def test_tool_outcome_requires_applicable_event_and_namespace(event_factory: Any) -> None:
    missing = event_factory(1, event_type=EventType.TOOL_COMPLETION)
    wrong_type = event_factory(
        2,
        event_type=EventType.OBSERVATION,
        payload={"tool_outcome": {"schema_version": "1.0", "exit_status": 2}},
    )

    assert unavailable_reason(lambda: parse_tool_outcome(missing)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    )
    assert unavailable_reason(lambda: parse_tool_outcome(wrong_type)) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )


def test_ambiguous_failure_namespaces_are_rejected(event_factory: Any) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {"schema_version": "1.0", "exit_status": 2},
            "test_report": {
                "schema_version": "1.0",
                "framework": "pytest",
                "status": "failed",
                "failures": [failure("tests/a.py::test_a")],
            },
        },
    )

    assert unavailable_reason(lambda: failure_fingerprint(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )
    assert unavailable_reason(lambda: classify_tool_outcome(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )
    assert unavailable_reason(lambda: classify_test_report(event)) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_transient_public_views_enforce_their_invariants() -> None:
    with pytest.raises(ValueError):
        ActionFingerprint(
            execution_mode="shell",
            tokens=(),
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        )
    with pytest.raises(ValueError):
        ActionFingerprint(
            execution_mode="argv",
            tokens=["echo"],  # type: ignore[arg-type]
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        )
    for invalid_token in ("", "\ud800", "[REDACTED]", "x\x00y"):
        with pytest.raises(ValueError):
            ActionFingerprint(
                execution_mode="argv",
                tokens=(invalid_token,),
                working_directory=WORKING_DIRECTORY,
                environment_digest=ENVIRONMENT_DIGEST,
            )
    with pytest.raises(ValueError):
        ActionFingerprint(
            execution_mode="argv",
            tokens=("echo",),
            working_directory=WORKING_DIRECTORY,
            environment_digest="bad",
        )
    with pytest.raises(ValueError):
        ToolOutcome(
            status=ToolOutcomeStatus.SUCCEEDED,
            exception_type="TimeoutError",
        )
    with pytest.raises(ValueError):
        ToolOutcome(status="failed")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ToolOutcome(status=ToolOutcomeStatus.FAILED, exit_status=True)
    with pytest.raises(ValueError):
        NormalizedTestFailure(test_id="./tests/a.py::test_a")
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="pytest",
            status=ReportStatus.PASSED,
            failures=(NormalizedTestFailure(test_id="tests/a.py::test_a"),),
        )
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="pytest",
            status="passed",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="pytest",
            status=ReportStatus.FAILED,
            failures=(object(),),  # type: ignore[arg-type]
        )
    duplicate = NormalizedTestFailure(test_id="tests/a.py::test_a")
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="pytest",
            status=ReportStatus.FAILED,
            failures=(duplicate, duplicate),
        )
    with pytest.raises(ValueError):
        FailureFingerprint(category="tool", components=())
    with pytest.raises(ValueError):
        FailureFingerprint(category="tool", components=["failure"])  # type: ignore[arg-type]


def test_invalid_event_object_and_text_subclass_fail_closed() -> None:
    class TextSubclass(str):
        pass

    assert unavailable_reason(lambda: failure_fingerprint(object())) is (
        AbstentionReason.EVENT_NOT_APPLICABLE
    )
    assert unavailable_reason(lambda: normalize_test_id(TextSubclass("tests/a.py::test"))) is (
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )


def test_payload_bounder_exercises_node_key_tuple_and_byte_limits() -> None:
    assert not fingerprints_module._payload_is_bounded(
        {"a": 1, "b": 2},
        max_bytes=1_000,
        max_nodes=1,
    )
    assert not fingerprints_module._payload_is_bounded(
        {1: "invalid-key"},
        max_bytes=1_000,
        max_nodes=100,
    )
    assert not fingerprints_module._payload_is_bounded(
        (1, 2),
        max_bytes=1_000,
        max_nodes=1,
    )
    assert not fingerprints_module._payload_is_bounded(
        "too-large",
        max_bytes=1,
        max_nodes=100,
    )


def test_evidence_models_reject_prevalidation_and_postvalidation_aggregate_overflow() -> None:
    failure_item = FailureEvidence(
        schema_version="1.0",
        test_id="t" * fingerprints_module._MAX_TEST_ID_BYTES,
    )
    with pytest.raises(ValidationError):
        ReportEvidence(
            schema_version="1.0",
            framework="pytest",
            status="failed",
            failures=(failure_item,) * (fingerprints_module._MAX_TEST_FAILURES + 1),
        )
    with pytest.raises(ValidationError):
        ReportEvidence(
            schema_version="1.0",
            framework="pytest",
            status="failed",
            failures=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        detailed_failure = FailureEvidence(
            schema_version="1.0",
            test_id="t" * fingerprints_module._MAX_TEST_ID_BYTES,
            failure_type="f" * 5_238,
        )
        ReportEvidence(
            schema_version="1.0",
            framework="f" * fingerprints_module._MAX_SHORT_TEXT_BYTES,
            status="failed",
            failures=(detailed_failure, *(failure_item,) * 9),
        )


def test_transient_fingerprints_reject_aggregate_overflow_after_item_validation() -> None:
    token = "x" * fingerprints_module._MAX_ARG_BYTES
    with pytest.raises(ValueError):
        ActionFingerprint(
            execution_mode="argv",
            tokens=(token,) * fingerprints_module._MAX_ARGV_ITEMS,
            working_directory=WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        )

    normalized = NormalizedTestFailure(test_id="t" * fingerprints_module._MAX_TEST_ID_BYTES)
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="pytest",
            status=ReportStatus.FAILED,
            failures=(normalized,) * (fingerprints_module._MAX_TEST_FAILURES + 1),
        )
    with pytest.raises(ValueError):
        ParsedTestReport(
            framework="f" * fingerprints_module._MAX_SHORT_TEXT_BYTES,
            status=ReportStatus.FAILED,
            failures=(normalized,) * 10,
        )
    with pytest.raises(ValueError):
        FailureFingerprint(
            category="tool",
            components=("s" * fingerprints_module._MAX_SIGNATURE_BYTES,) * 3,
        )


def test_internal_loaders_reject_mutable_payload_state_even_on_exact_event_types() -> None:
    mutable_tool_event = TraceEvent.model_construct(
        event_type=EventType.TOOL_COMPLETION,
        payload={},
    )
    mutable_test_event = TraceEvent.model_construct(
        event_type=EventType.OBSERVATION,
        payload={},
    )
    with pytest.raises(FingerprintUnavailableError):
        fingerprints_module._load_tool_outcome(mutable_tool_event)
    with pytest.raises(FingerprintUnavailableError):
        fingerprints_module._load_test_report(mutable_test_event)
