from __future__ import annotations

import json
from argparse import ArgumentTypeError
from decimal import Decimal
from pathlib import Path

import pytest
from scripts.check_coverage_thresholds import (
    CoverageThresholdError,
    _minimum,
    enforce_thresholds,
    main,
)


def _totals(*, lines: int = 95, branches: int = 95) -> dict[str, int]:
    return {
        "covered_lines": lines,
        "num_statements": 100,
        "covered_branches": branches,
        "num_branches": 100,
    }


def test_thresholds_are_enforced_independently() -> None:
    assert enforce_thresholds(_totals(), minimum=Decimal(95)) == (
        Decimal(95),
        Decimal(95),
    )

    with pytest.raises(CoverageThresholdError, match=r"statements=99.00%.*branches=94.00%"):
        enforce_thresholds(_totals(lines=99, branches=94), minimum=Decimal(95))
    with pytest.raises(CoverageThresholdError, match=r"statements=94.00%.*branches=99.00%"):
        enforce_thresholds(_totals(lines=94, branches=99), minimum=Decimal(95))


@pytest.mark.parametrize(
    "totals",
    (
        None,
        {},
        {**_totals(), "covered_lines": True},
        {**_totals(), "covered_branches": 101},
        {**_totals(), "num_branches": 0},
    ),
)
def test_invalid_totals_fail_closed(totals: object) -> None:
    with pytest.raises(CoverageThresholdError):
        enforce_thresholds(totals, minimum=Decimal(95))


@pytest.mark.parametrize("value", ("not-a-number", "NaN", "Infinity", "-1", "101"))
def test_invalid_minimum_is_rejected(value: str) -> None:
    with pytest.raises(ArgumentTypeError):
        _minimum(value)


def test_cli_reads_coverage_json_and_reports_exact_percentages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"totals": _totals(lines=97, branches=96)}), encoding="utf-8")

    assert main([str(report), "--minimum", "95"]) == 0
    assert capsys.readouterr().out == (
        "independent coverage thresholds satisfied: "
        "statements=97.00% branches=96.00% minimum=95.00%\n"
    )


@pytest.mark.parametrize(
    "payload",
    ("not-json", "[]", json.dumps({"totals": _totals(branches=94)})),
)
def test_cli_fails_closed_for_invalid_or_below_threshold_reports(
    tmp_path: Path,
    payload: str,
) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(payload, encoding="utf-8")

    with pytest.raises(SystemExit):
        main([str(report)])
