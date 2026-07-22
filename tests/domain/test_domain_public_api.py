from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from saliencegate.domain import primitives, records


def test_domain_root_defers_record_import_until_public_access() -> None:
    program = """
import json
import sys

import saliencegate.domain as domain

loaded_before = sorted(
    name for name in sys.modules if name.startswith("saliencegate.domain.")
)
first = domain.canonical_json
second = domain.canonical_json
loaded_after = sorted(
    name for name in sys.modules if name.startswith("saliencegate.domain.")
)
print(json.dumps({
    "cached": first is second and domain.__dict__["canonical_json"] is first,
    "declared": "canonical_json" in domain.__all__,
    "discoverable": "canonical_json" in dir(domain),
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
    assert "saliencegate.domain.serde" in observation["loaded_after"]
    assert "saliencegate.domain.records" not in observation["loaded_after"]
    assert "saliencegate.domain.validation" not in observation["loaded_after"]


def test_domain_root_star_import_resolves_every_declared_export() -> None:
    program = """
import json
import saliencegate.domain as domain
from saliencegate.domain import *

scope = locals()
print(json.dumps({
    "missing": sorted(name for name in domain.__all__ if name not in scope),
    "mismatched": sorted(
        name for name in domain.__all__
        if name in scope and scope[name] is not getattr(domain, name)
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


def test_record_scalar_aliases_are_the_exact_primitive_objects() -> None:
    assert records.ComponentIdentifier is primitives.ComponentIdentifier
    assert records.Sha256Digest is primitives.Sha256Digest
    assert records.UtcDatetime is primitives.UtcDatetime
    assert records.EventMetadataIdentifier is primitives.ComponentIdentifier


def test_extracted_scalar_aliases_preserve_validation() -> None:
    component = TypeAdapter(primitives.ComponentIdentifier)
    digest = TypeAdapter(primitives.Sha256Digest)
    timestamp = TypeAdapter(primitives.UtcDatetime)

    assert component.validate_python("capture/profile", strict=True) == "capture/profile"
    assert digest.validate_python("a" * 64, strict=True) == "a" * 64
    assert timestamp.validate_python(
        datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        strict=True,
    ) == datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    with pytest.raises(ValidationError):
        component.validate_python("-invalid", strict=True)
    with pytest.raises(ValidationError):
        digest.validate_python("A" * 64, strict=True)
    with pytest.raises(ValidationError, match="UTC"):
        timestamp.validate_python(datetime(2026, 7, 21, 12, 0), strict=True)
    with pytest.raises(ValidationError, match="UTC"):
        timestamp.validate_python(
            datetime(
                2026,
                7,
                21,
                13,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            ),
            strict=True,
        )
