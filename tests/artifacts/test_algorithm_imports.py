from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
_EXPORT = "saliencegate.artifacts.algorithm_export"
_PROJECTION = "saliencegate.artifacts.algorithm_projection"
_VALIDATE = "saliencegate.artifacts.algorithm_validate"
_IMPORT_ORDER_PROBE = r"""
import importlib
import sys

package = importlib.import_module("saliencegate.artifacts")
targets = (
    "saliencegate.artifacts.algorithm_export",
    "saliencegate.artifacts.algorithm_projection",
    "saliencegate.artifacts.algorithm_validate",
)
for target in targets:
    sys.modules.pop(target, None)
    package.__dict__.pop(target.rsplit(".", 1)[1], None)

first, second = sys.argv[1:]
importlib.import_module(first)
assert second not in sys.modules
importlib.import_module(second)

exporter = importlib.import_module("saliencegate.artifacts.algorithm_export")
projection = importlib.import_module("saliencegate.artifacts.algorithm_projection")
validator = importlib.import_module("saliencegate.artifacts.algorithm_validate")
assert exporter._projection is projection
assert validator._projection is projection
assert callable(exporter.export_algorithm_artifact)
assert callable(projection._project_algorithm_components)
assert callable(projection._validate_source_execution_binding)
assert callable(validator.validate_algorithm_artifact)
"""


@pytest.mark.parametrize(
    ("first", "second"),
    ((_EXPORT, _VALIDATE), (_VALIDATE, _EXPORT)),
    ids=("export-then-validate", "validate-then-export"),
)
def test_algorithm_modules_import_in_either_order_without_a_cycle(
    first: str,
    second: str,
) -> None:
    completed = subprocess.run(
        (sys.executable, "-I", "-c", _IMPORT_ORDER_PROBE, first, second),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""
