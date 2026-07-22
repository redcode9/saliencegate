from __future__ import annotations

import json
import subprocess
import sys


def test_security_root_defers_submodules_until_a_public_name_is_accessed() -> None:
    program = """
import json
import sys

import saliencegate.security as security

loaded_before = sorted(
    name for name in sys.modules if name.startswith("saliencegate.security.")
)
first = security.InstallationKey
second = security.InstallationKey
loaded_after = sorted(
    name for name in sys.modules if name.startswith("saliencegate.security.")
)
print(json.dumps({
    "cached": first is second and security.__dict__["InstallationKey"] is first,
    "declared": "InstallationKey" in security.__all__,
    "discoverable": "InstallationKey" in dir(security),
    "domain_records_loaded": "saliencegate.domain.records" in sys.modules,
    "loaded_before": loaded_before,
    "loaded_after": loaded_after,
}))
"""

    completed = subprocess.run(
        (sys.executable, "-c", program),
        capture_output=True,
        check=False,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    observation = json.loads(completed.stdout)
    assert observation["loaded_before"] == []
    assert observation["cached"] is True
    assert observation["declared"] is True
    assert observation["discoverable"] is True
    assert observation["domain_records_loaded"] is False
    assert "saliencegate.security.keys" in observation["loaded_after"]
    assert "saliencegate.security.digests" not in observation["loaded_after"]
    assert "saliencegate.security.redaction" not in observation["loaded_after"]


def test_security_root_star_import_resolves_every_declared_export() -> None:
    program = """
import json
import saliencegate.security as security
from saliencegate.security import *

scope = locals()
print(json.dumps({
    "missing": sorted(name for name in security.__all__ if name not in scope),
    "mismatched": sorted(
        name for name in security.__all__
        if name in scope and scope[name] is not getattr(security, name)
    ),
}))
"""

    completed = subprocess.run(
        (sys.executable, "-c", program),
        capture_output=True,
        check=False,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {"mismatched": [], "missing": []}
