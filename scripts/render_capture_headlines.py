from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "examples/capture/headline-results.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/readme/capture-headlines.svg"

_CASE_IDS = (
    "memory-review-suggested",
    "no-current-evidence",
    "insufficient-evidence",
)
_CASE_SUMMARIES = {
    "memory-review-suggested": "Repeated exact action plus two structured tool failures",
    "no-current-evidence": "Closed clean window with two distinct successful actions",
    "insufficient-evidence": "Open window with one unresolved action and a transport gap",
}
_CASE_HEADLINES = {
    "memory-review-suggested": "memory_review_suggested",
    "no-current-evidence": "no_current_evidence",
    "insufficient-evidence": "insufficient_evidence",
}
_CASE_DISPOSITIONS = {
    "memory-review-suggested": "flagged",
    "no-current-evidence": "not_flagged",
    "insufficient-evidence": "indeterminate",
}
_CASE_STATES = {
    "memory-review-suggested": "closed",
    "no-current-evidence": "closed",
    "insufficient-evidence": "open",
}
_CASE_SESSION_IDS = {
    "memory-review-suggested": "synthetic-repeated-session",
    "no-current-evidence": "synthetic-clean-session",
    "insufficient-evidence": "synthetic-incomplete-session",
}
_COUNT_KEYS = (
    "captured_events",
    "projected_events",
    "action_identities",
    "structured_results",
    "detected_signals",
    "ignored_records",
)
_DETECTOR_KEYS = ("tool_error", "repeated_action", "repeated_failure")
_LIMITS = frozenset(
    {
        "capture_degraded",
        "detector_minimum_not_met",
        "gap_detected",
        "session_open",
    }
)
_CASE_EXPECTED_COUNTS = {
    "memory-review-suggested": {
        "captured_events": 6,
        "projected_events": 6,
        "action_identities": 2,
        "structured_results": 2,
        "detected_signals": 3,
        "ignored_records": 0,
    },
    "no-current-evidence": {
        "captured_events": 6,
        "projected_events": 6,
        "action_identities": 2,
        "structured_results": 2,
        "detected_signals": 0,
        "ignored_records": 0,
    },
    "insufficient-evidence": {
        "captured_events": 3,
        "projected_events": 2,
        "action_identities": 1,
        "structured_results": 0,
        "detected_signals": 0,
        "ignored_records": 1,
    },
}
_CASE_EXPECTED_COVERAGE = {
    "memory-review-suggested": {"coverage_degraded": False, "limits": []},
    "no-current-evidence": {"coverage_degraded": False, "limits": []},
    "insufficient-evidence": {
        "coverage_degraded": True,
        "limits": [
            "capture_degraded",
            "detector_minimum_not_met",
            "gap_detected",
            "session_open",
        ],
    },
}
_CASE_EXPECTED_DETECTORS = {
    "memory-review-suggested": {
        "tool_error": {
            "support": "supported",
            "disposition": "flagged",
            "detected_count": 2,
        },
        "repeated_action": {
            "support": "conditional",
            "disposition": "flagged",
            "detected_count": 1,
        },
        "repeated_failure": {
            "support": "unsupported",
            "disposition": "not_applicable",
            "detected_count": 0,
        },
    },
    "no-current-evidence": {
        "tool_error": {
            "support": "supported",
            "disposition": "not_flagged",
            "detected_count": 0,
        },
        "repeated_action": {
            "support": "conditional",
            "disposition": "not_flagged",
            "detected_count": 0,
        },
        "repeated_failure": {
            "support": "unsupported",
            "disposition": "not_applicable",
            "detected_count": 0,
        },
    },
    "insufficient-evidence": {
        "tool_error": {
            "support": "supported",
            "disposition": "indeterminate",
            "detected_count": 0,
        },
        "repeated_action": {
            "support": "conditional",
            "disposition": "indeterminate",
            "detected_count": 0,
        },
        "repeated_failure": {
            "support": "unsupported",
            "disposition": "not_applicable",
            "detected_count": 0,
        },
    },
}
_EVENT_SIGNATURE_KEYS = (
    "kind",
    "event_id",
    "call_id",
    "tool",
    "input",
    "identity_authority",
    "outcome",
    "reason",
)
_CASE_EVENT_SIGNATURES = {
    "memory-review-suggested": (
        (
            "tool_started",
            "repeated-start-1",
            "repeated-call-1",
            "read",
            {"path": "synthetic-public-example.txt"},
            "exact",
            None,
            None,
        ),
        (
            "tool_finished",
            "repeated-finish-1",
            "repeated-call-1",
            None,
            None,
            None,
            "failed",
            None,
        ),
        (
            "tool_started",
            "repeated-start-2",
            "repeated-call-2",
            "read",
            {"path": "synthetic-public-example.txt"},
            "exact",
            None,
            None,
        ),
        (
            "tool_finished",
            "repeated-finish-2",
            "repeated-call-2",
            None,
            None,
            None,
            "failed",
            None,
        ),
        ("session_finished", "repeated-close", None, None, None, None, None, None),
    ),
    "no-current-evidence": (
        (
            "tool_started",
            "clean-start-1",
            "clean-call-1",
            "read",
            {"path": "synthetic-public-alpha.txt"},
            "exact",
            None,
            None,
        ),
        (
            "tool_finished",
            "clean-finish-1",
            "clean-call-1",
            None,
            None,
            None,
            "succeeded",
            None,
        ),
        (
            "tool_started",
            "clean-start-2",
            "clean-call-2",
            "read",
            {"path": "synthetic-public-beta.txt"},
            "exact",
            None,
            None,
        ),
        (
            "tool_finished",
            "clean-finish-2",
            "clean-call-2",
            None,
            None,
            None,
            "succeeded",
            None,
        ),
        ("session_finished", "clean-close", None, None, None, None, None, None),
    ),
    "insufficient-evidence": (
        (
            "tool_started",
            "incomplete-start",
            "incomplete-call",
            "read",
            {"path": "synthetic-public-open.txt"},
            "exact",
            None,
            None,
        ),
        ("coverage_degraded", None, None, None, None, None, None, "transport_gap"),
    ),
}
_SUMMARY_MAX_WIDTH = 360
_SUMMARY_BASELINES = (320, 356, 392)
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_SYNTHETIC_TEXT = re.compile(r"^synthetic-[a-z0-9-]{3,80}$")
_SYNTHETIC_PATH = re.compile(r"^synthetic-public-[a-z0-9-]+\.txt$")


class CaptureHeadlineFixtureError(ValueError):
    """The public capture-headline fixture is outside its frozen contract."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureHeadlineFixtureError("duplicate JSON key")
        result[key] = value
    return result


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise CaptureHeadlineFixtureError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise CaptureHeadlineFixtureError(f"{label} must be an array")
    return value


def _keys(value: Mapping[str, object], expected: Sequence[str], *, label: str) -> None:
    if frozenset(value) != frozenset(expected):
        raise CaptureHeadlineFixtureError(f"{label} has unexpected fields")


def _text(value: object, *, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise CaptureHeadlineFixtureError(f"{label} must be bounded text")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise CaptureHeadlineFixtureError(f"{label} has invalid text")
    return value


def _count(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000:
        raise CaptureHeadlineFixtureError(f"{label} must be a bounded count")
    return value


def _validate_event(value: object, *, case_id: str, session_id: str) -> dict[str, object]:
    event = _mapping(value, label=f"{case_id} event")
    kind = _text(event.get("kind"), label=f"{case_id} event kind", pattern=_IDENTIFIER)
    expected_keys: tuple[str, ...]
    if kind == "tool_started":
        expected_keys = (
            "kind",
            "session_id",
            "event_id",
            "call_id",
            "tool",
            "input",
            "identity_authority",
        )
    elif kind == "tool_finished":
        expected_keys = ("kind", "session_id", "event_id", "call_id", "outcome")
    elif kind == "session_finished":
        expected_keys = ("kind", "session_id", "event_id")
    elif kind == "coverage_degraded":
        expected_keys = ("kind", "session_id", "reason")
    else:
        raise CaptureHeadlineFixtureError(f"{case_id} has an unsupported event kind")
    _keys(event, expected_keys, label=f"{case_id} {kind} event")
    if event["session_id"] != session_id:
        raise CaptureHeadlineFixtureError(f"{case_id} event session does not match")

    if "event_id" in event:
        _text(event["event_id"], label=f"{case_id} event id", pattern=_IDENTIFIER)
    if kind in {"tool_started", "tool_finished"}:
        _text(event["call_id"], label=f"{case_id} call id", pattern=_IDENTIFIER)
    if kind == "tool_started":
        if event["tool"] != "read" or event["identity_authority"] != "exact":
            raise CaptureHeadlineFixtureError(f"{case_id} action authority is not frozen")
        action_input = _mapping(event["input"], label=f"{case_id} action input")
        _keys(action_input, ("path",), label=f"{case_id} action input")
        _text(
            action_input["path"],
            label=f"{case_id} synthetic path",
            pattern=_SYNTHETIC_PATH,
        )
    elif kind == "tool_finished" and event["outcome"] not in {"succeeded", "failed"}:
        raise CaptureHeadlineFixtureError(f"{case_id} outcome is outside the closed vocabulary")
    elif kind == "coverage_degraded" and event["reason"] != "transport_gap":
        raise CaptureHeadlineFixtureError(f"{case_id} gap reason is not frozen")
    return event


def _validate_case_semantics(case_id: str, events: Sequence[dict[str, object]]) -> None:
    starts = [event for event in events if event["kind"] == "tool_started"]
    finishes = [event for event in events if event["kind"] == "tool_finished"]
    closes = [event for event in events if event["kind"] == "session_finished"]
    gaps = [event for event in events if event["kind"] == "coverage_degraded"]
    start_calls = [event["call_id"] for event in starts]
    finish_calls = [event["call_id"] for event in finishes]
    if len(start_calls) != len(set(start_calls)) or len(finish_calls) != len(set(finish_calls)):
        raise CaptureHeadlineFixtureError(f"{case_id} repeats a call identifier")

    if case_id == "memory-review-suggested":
        inputs = [event["input"] for event in starts]
        valid = (
            len(starts) == len(finishes) == 2
            and len(closes) == 1
            and not gaps
            and start_calls == finish_calls
            and inputs[0] == inputs[1]
            and all(event["outcome"] == "failed" for event in finishes)
        )
    elif case_id == "no-current-evidence":
        inputs = [event["input"] for event in starts]
        valid = (
            len(starts) == len(finishes) == 2
            and len(closes) == 1
            and not gaps
            and start_calls == finish_calls
            and len({json.dumps(item, sort_keys=True) for item in inputs}) == 2
            and all(event["outcome"] == "succeeded" for event in finishes)
        )
    else:
        valid = len(starts) == len(gaps) == 1 and not finishes and not closes
    if not valid:
        raise CaptureHeadlineFixtureError(f"{case_id} event semantics changed")


def _event_signature(event: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(event.get(key) for key in _EVENT_SIGNATURE_KEYS)


def _validate_expected(case_id: str, value: object) -> dict[str, object]:
    expected = _mapping(value, label=f"{case_id} expected result")
    _keys(
        expected,
        ("headline", "shadow_disposition", "session_state", "counts", "coverage", "detectors"),
        label=f"{case_id} expected result",
    )
    if (
        expected["headline"] != _CASE_HEADLINES[case_id]
        or expected["shadow_disposition"] != _CASE_DISPOSITIONS[case_id]
        or expected["session_state"] != _CASE_STATES[case_id]
    ):
        raise CaptureHeadlineFixtureError(f"{case_id} conclusion changed")

    counts = _mapping(expected["counts"], label=f"{case_id} counts")
    _keys(counts, _COUNT_KEYS, label=f"{case_id} counts")
    for key in _COUNT_KEYS:
        _count(counts[key], label=f"{case_id} {key}")
    if counts["captured_events"] != counts["projected_events"] + counts["ignored_records"]:
        raise CaptureHeadlineFixtureError(f"{case_id} counts disagree")
    if counts != _CASE_EXPECTED_COUNTS[case_id]:
        raise CaptureHeadlineFixtureError(f"{case_id} counts contract changed")

    coverage = _mapping(expected["coverage"], label=f"{case_id} coverage")
    _keys(coverage, ("coverage_degraded", "limits"), label=f"{case_id} coverage")
    if type(coverage["coverage_degraded"]) is not bool:
        raise CaptureHeadlineFixtureError(f"{case_id} coverage flag must be boolean")
    limits = _sequence(coverage["limits"], label=f"{case_id} limits")
    if (
        any(type(item) is not str or item not in _LIMITS for item in limits)
        or limits != sorted(set(limits))
        or coverage["coverage_degraded"] is not bool(limits)
    ):
        raise CaptureHeadlineFixtureError(f"{case_id} report limits disagree")
    if coverage != _CASE_EXPECTED_COVERAGE[case_id]:
        raise CaptureHeadlineFixtureError(f"{case_id} coverage contract changed")

    detectors = _mapping(expected["detectors"], label=f"{case_id} detectors")
    _keys(detectors, _DETECTOR_KEYS, label=f"{case_id} detectors")
    for signal_type in _DETECTOR_KEYS:
        detector = _mapping(detectors[signal_type], label=f"{case_id} {signal_type}")
        _keys(
            detector,
            ("support", "disposition", "detected_count"),
            label=f"{case_id} {signal_type}",
        )
        if detector["support"] not in {"supported", "conditional", "unsupported"}:
            raise CaptureHeadlineFixtureError(f"{case_id} detector support is invalid")
        if detector["disposition"] not in {
            "flagged",
            "not_flagged",
            "indeterminate",
            "not_applicable",
        }:
            raise CaptureHeadlineFixtureError(f"{case_id} detector disposition is invalid")
        _count(detector["detected_count"], label=f"{case_id} detector count")
    if detectors["tool_error"]["support"] != "supported":
        raise CaptureHeadlineFixtureError(f"{case_id} tool-error support changed")
    if detectors["repeated_action"]["support"] != "conditional":
        raise CaptureHeadlineFixtureError(f"{case_id} repeated-action support changed")
    if detectors["repeated_failure"] != {
        "support": "unsupported",
        "disposition": "not_applicable",
        "detected_count": 0,
    }:
        raise CaptureHeadlineFixtureError(f"{case_id} repeated-failure boundary changed")
    if counts["detected_signals"] != sum(
        detectors[signal_type]["detected_count"] for signal_type in _DETECTOR_KEYS
    ):
        raise CaptureHeadlineFixtureError(f"{case_id} detected-signal total disagrees")
    if detectors != _CASE_EXPECTED_DETECTORS[case_id]:
        raise CaptureHeadlineFixtureError(f"{case_id} detector contract changed")
    return expected


def validate_capture_headline_fixture(value: object) -> dict[str, object]:
    document = _mapping(value, label="fixture")
    _keys(
        document,
        (
            "schema_version",
            "profile_id",
            "host_version",
            "provenance",
            "report_invariants",
            "cases",
        ),
        label="fixture",
    )
    if (
        document["schema_version"] != "capture-headline-examples/v1"
        or document["profile_id"] != "opencode-plugin/v1"
        or document["host_version"] != "1.18.3"
        or document["provenance"] != "fully_synthetic_no_provider_or_model_call"
    ):
        raise CaptureHeadlineFixtureError("fixture identity changed")
    invariants = _mapping(document["report_invariants"], label="report invariants")
    if invariants != {
        "evidence_level": "descriptive_observational",
        "model_calls": 0,
        "decision_authority": False,
        "confirmatory": False,
    }:
        raise CaptureHeadlineFixtureError("report invariants changed")

    cases = _sequence(document["cases"], label="cases")
    validated_cases: list[dict[str, object]] = []
    for index, value_case in enumerate(cases):
        case = _mapping(value_case, label="case")
        _keys(
            case,
            ("id", "native_session_id", "summary", "events", "expected"),
            label="case",
        )
        case_id = _text(case["id"], label="case id", pattern=_IDENTIFIER)
        if index >= len(_CASE_IDS) or case_id != _CASE_IDS[index]:
            raise CaptureHeadlineFixtureError("case order or identity changed")
        session_id = _text(
            case["native_session_id"],
            label=f"{case_id} native session",
            pattern=_SYNTHETIC_TEXT,
        )
        if session_id != _CASE_SESSION_IDS[case_id]:
            raise CaptureHeadlineFixtureError(f"{case_id} native session changed")
        if case["summary"] != _CASE_SUMMARIES[case_id]:
            raise CaptureHeadlineFixtureError(f"{case_id} summary changed")
        raw_events = _sequence(case["events"], label=f"{case_id} events")
        events = tuple(
            _validate_event(event, case_id=case_id, session_id=session_id) for event in raw_events
        )
        event_ids = [event["event_id"] for event in events if "event_id" in event]
        if len(event_ids) != len(set(event_ids)):
            raise CaptureHeadlineFixtureError(f"{case_id} repeats an event identifier")
        _validate_case_semantics(case_id, events)
        if tuple(_event_signature(event) for event in events) != _CASE_EVENT_SIGNATURES[case_id]:
            raise CaptureHeadlineFixtureError(f"{case_id} event contract changed")
        _validate_expected(case_id, case["expected"])
        validated_cases.append(case)
    if len(validated_cases) != len(_CASE_IDS):
        raise CaptureHeadlineFixtureError("fixture must contain exactly three cases")
    return document


def load_capture_headline_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, object]:
    try:
        data = path.read_bytes()
        if not data or len(data) > 256 * 1_024:
            raise CaptureHeadlineFixtureError("fixture byte size is invalid")
        decoded = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except CaptureHeadlineFixtureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CaptureHeadlineFixtureError("fixture is unreadable") from None
    return validate_capture_headline_fixture(decoded)


def _metric(name: str, value: object, *, x: int, y: int, size: int = 30) -> str:
    escaped_name = html.escape(name, quote=True)
    escaped_value = html.escape(str(value).lower(), quote=False)
    return (
        f'  <text data-metric="{escaped_name}" x="{x}" y="{y}" '
        f'fill="#22211f" font-size="{size}" font-weight="700">'
        f"{escaped_value}</text>"
    )


def _estimated_svg_text_width(text: str) -> int:
    """Return a deliberately conservative width estimate for 28-unit system sans text."""
    width = 0
    for character in text:
        if character == " ":
            width += 10
        elif character in "ijlrtfI.,:;!'":
            width += 12
        elif character in "mwMW@%&":
            width += 24
        elif character.isupper():
            width += 21
        else:
            width += 17
    return width


def _wrapped_summary(summary: str) -> tuple[str, ...]:
    words = summary.split()
    if not words or any(_estimated_svg_text_width(word) > _SUMMARY_MAX_WIDTH for word in words):
        raise CaptureHeadlineFixtureError("case summary does not fit the visual")

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _estimated_svg_text_width(candidate) <= _SUMMARY_MAX_WIDTH:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if not 2 <= len(lines) <= len(_SUMMARY_BASELINES):
        raise CaptureHeadlineFixtureError("case summary does not fit the visual")
    return tuple(lines)


def render_capture_headlines_svg(value: object) -> str:
    document = validate_capture_headline_fixture(value)
    raw_cases = _sequence(document["cases"], label="cases")
    cards: list[str] = []
    card_styles = (
        (60, "#dce9e9", "#315f6f"),
        (570, "#ffffff", "#315f6f"),
        (1080, "#f1e0c6", "#9b5c13"),
    )
    for raw_case, (x, surface, accent) in zip(raw_cases, card_styles, strict=True):
        case = _mapping(raw_case, label="case")
        case_id = str(case["id"])
        expected = _mapping(case["expected"], label=f"{case_id} expected result")
        counts = _mapping(expected["counts"], label=f"{case_id} counts")
        coverage = _mapping(expected["coverage"], label=f"{case_id} coverage")
        detectors = _mapping(expected["detectors"], label=f"{case_id} detectors")
        repeated_action = _mapping(detectors["repeated_action"], label="repeated action")
        tool_error = _mapping(detectors["tool_error"], label="tool error")
        repeated_failure = _mapping(detectors["repeated_failure"], label="repeated failure")
        summary_lines = _wrapped_summary(str(case["summary"]))
        human_headline = str(expected["headline"]).replace("_", " ").capitalize()
        prefix = case_id
        summary_elements = tuple(
            f'  <text data-summary="{prefix}" '
            f'data-estimated-width="{_estimated_svg_text_width(line)}" '
            f'x="{x + 30}" y="{baseline}" fill="#22211f" font-size="28">'
            f"{html.escape(line)}</text>"
            for line, baseline in zip(
                summary_lines,
                _SUMMARY_BASELINES[: len(summary_lines)],
                strict=True,
            )
        )
        cards.extend(
            (
                f'  <rect data-surface="true" x="{x}" y="165" width="460" height="610" '
                f'rx="14" fill="{surface}" stroke="{accent}" stroke-width="4"/>',
                f'  <rect data-series="{prefix}" aria-label="{html.escape(human_headline)} '
                f'headline lane" x="{x}" y="165" width="460" height="14" rx="7" '
                f'fill="{accent}"/>',
                f'  <text x="{x + 30}" y="226" fill="#22211f" font-size="28" '
                f'font-weight="700">{html.escape(human_headline)}</text>',
                _metric(f"{prefix}.headline", expected["headline"], x=x + 30, y=270, size=28),
                *summary_elements,
                f'  <text x="{x + 30}" y="432" fill="#22211f" font-size="28">actions</text>',
                _metric(
                    f"{prefix}.action_identities",
                    counts["action_identities"],
                    x=x + 30,
                    y=470,
                    size=40,
                ),
                f'  <text x="{x + 174}" y="432" fill="#22211f" font-size="28">results</text>',
                _metric(
                    f"{prefix}.structured_results",
                    counts["structured_results"],
                    x=x + 174,
                    y=470,
                    size=40,
                ),
                f'  <text x="{x + 318}" y="432" fill="#22211f" font-size="28">signals</text>',
                _metric(
                    f"{prefix}.detected_signals",
                    counts["detected_signals"],
                    x=x + 318,
                    y=470,
                    size=40,
                ),
                f'  <line x1="{x + 30}" y1="492" x2="{x + 430}" y2="492" '
                'stroke="#c7c0b4" stroke-width="3"/>',
                f'  <text x="{x + 30}" y="536" fill="#22211f" '
                'font-size="28">repeated_action</text>',
                _metric(
                    f"{prefix}.repeated_action.detected",
                    repeated_action["detected_count"],
                    x=x + 390,
                    y=536,
                    size=28,
                ),
                f'  <text x="{x + 30}" y="578" fill="#22211f" font-size="28">tool_error</text>',
                _metric(
                    f"{prefix}.tool_error.detected",
                    tool_error["detected_count"],
                    x=x + 390,
                    y=578,
                    size=28,
                ),
                f'  <text x="{x + 30}" y="620" fill="#22211f" '
                'font-size="28">repeated_failure</text>',
                _metric(
                    f"{prefix}.repeated_failure.support",
                    repeated_failure["support"],
                    x=x + 30,
                    y=654,
                    size=28,
                ),
                f'  <text x="{x + 30}" y="700" fill="#22211f" '
                'font-size="28">coverage degraded</text>',
                _metric(
                    f"{prefix}.coverage_degraded",
                    coverage["coverage_degraded"],
                    x=x + 335,
                    y=700,
                    size=28,
                ),
                f'  <text x="{x + 30}" y="744" fill="#22211f" font-size="28">report limits</text>',
                _metric(
                    f"{prefix}.limit_count",
                    len(_sequence(coverage["limits"], label=f"{case_id} limits")),
                    x=x + 390,
                    y=744,
                    size=28,
                ),
            )
        )

    invariants = _mapping(document["report_invariants"], label="report invariants")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1000" width="100%" '
        'role="img" aria-labelledby="capture-headlines-title capture-headlines-desc" '
        'font-family="ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, '
        'sans-serif">',
        '  <title id="capture-headlines-title">Three synthetic OpenCode capture report '
        "headlines</title>",
        '  <desc id="capture-headlines-desc">Synthetic replay produces Memory review suggested, '
        "No current evidence, or Insufficient evidence while preserving zero model calls and no "
        "decision authority.</desc>",
        '  <rect data-surface="true" x="0" y="0" width="1600" height="1000" fill="#f6f1e8"/>',
        '  <text x="60" y="72" fill="#22211f" font-size="52" font-weight="700">Three '
        "bounded capture headlines</text>",
        '  <text x="60" y="118" fill="#22211f" font-size="30">Fully synthetic OpenCode '
        "contract replay; not a real-world efficacy result</text>",
        *cards,
        '  <rect data-surface="true" x="60" y="810" width="1480" height="140" rx="12" '
        'fill="#ffffff"/>',
        '  <text x="100" y="858" fill="#22211f" font-size="28">model calls</text>',
        _metric("examples.model_calls", invariants["model_calls"], x=100, y=910, size=40),
        '  <text x="520" y="858" fill="#22211f" font-size="28">decision authority</text>',
        _metric(
            "examples.decision_authority",
            invariants["decision_authority"],
            x=520,
            y=910,
            size=40,
        ),
        '  <text x="960" y="858" fill="#22211f" font-size="28">confirmatory</text>',
        _metric("examples.confirmatory", invariants["confirmatory"], x=960, y=910, size=40),
        '  <text x="1160" y="858" fill="#22211f" font-size="28">evidence</text>',
        _metric(
            "examples.evidence_level",
            invariants["evidence_level"],
            x=1160,
            y=910,
            size=28,
        ),
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the synthetic capture-headline visual")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the exact fixture rendering")
    mode.add_argument("--check", action="store_true", help="verify the tracked rendering")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rendered = render_capture_headlines_svg(load_capture_headline_fixture(args.fixture))
        if args.write:
            if args.output.is_symlink() or not args.output.parent.is_dir():
                raise CaptureHeadlineFixtureError("output boundary is invalid")
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
            print(f"Wrote {args.output}")
            return 0
        if args.output.is_symlink() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"{args.output}: capture-headline visual is stale", file=sys.stderr)
            return 1
    except (CaptureHeadlineFixtureError, OSError, UnicodeError) as error:
        print(f"Capture-headline render failed: {error}", file=sys.stderr)
        return 1
    print("Capture-headline visual is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
