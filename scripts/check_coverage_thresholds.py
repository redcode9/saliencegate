"""Enforce statement and branch coverage thresholds independently."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class CoverageThresholdError(ValueError):
    """Raised when a coverage report is invalid or below its required threshold."""


def _percentage(*, covered: object, total: object, label: str) -> Decimal:
    if (
        not isinstance(covered, int)
        or isinstance(covered, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or not 0 <= covered <= total
    ):
        raise CoverageThresholdError(f"invalid {label} coverage totals")
    return Decimal(covered) * Decimal(100) / Decimal(total)


def enforce_thresholds(
    totals: object,
    *,
    minimum: Decimal,
) -> tuple[Decimal, Decimal]:
    """Return exact percentages after enforcing both independent thresholds."""

    if not isinstance(totals, dict):
        raise CoverageThresholdError("coverage report totals must be an object")
    statements = _percentage(
        covered=totals.get("covered_lines"),
        total=totals.get("num_statements"),
        label="statement",
    )
    branches = _percentage(
        covered=totals.get("covered_branches"),
        total=totals.get("num_branches"),
        label="branch",
    )
    failures = [
        f"statements={statements:.2f}%",
        f"branches={branches:.2f}%",
    ]
    if statements < minimum or branches < minimum:
        raise CoverageThresholdError(
            f"independent coverage threshold failed: {', '.join(failures)}; minimum={minimum:.2f}%"
        )
    return statements, branches


def _minimum(value: str) -> Decimal:
    try:
        minimum = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("minimum must be a decimal percentage") from error
    if not minimum.is_finite() or not Decimal(0) <= minimum <= Decimal(100):
        raise argparse.ArgumentTypeError("minimum must be between 0 and 100")
    return minimum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    parser.add_argument("--minimum", type=_minimum, default=Decimal(95))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document: Any = json.loads(args.report.read_text(encoding="utf-8"))
        totals = document.get("totals") if isinstance(document, dict) else None
        statements, branches = enforce_thresholds(totals, minimum=args.minimum)
    except (CoverageThresholdError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        "independent coverage thresholds satisfied: "
        f"statements={statements:.2f}% branches={branches:.2f}% "
        f"minimum={args.minimum:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
