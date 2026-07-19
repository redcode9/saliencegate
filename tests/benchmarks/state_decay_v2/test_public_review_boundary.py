from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_public_review_boundary import (
    BOUNDARY_MODULES,
    scan_source,
    validate_public_review_boundary,
)

_PACKAGE = Path("src/saliencegate/benchmarks/state_decay_v2")


def _write_boundary_module(root: Path, source: str, *, name: str = "templates.py") -> Path:
    path = root / _PACKAGE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("source", "category"),
    (
        (
            "from saliencegate.benchmarks.state_decay_v2.config import "
            "allocate_balanced_outcomes\n",
            "allocation",
        ),
        (
            "def build(seed, family, lineages):\n"
            "    return allocate_balanced_outcomes(seed, family, lineages)\n",
            "allocation",
        ),
        (
            "from saliencegate.benchmarks.state_decay_v2.config import "
            "SeedPurpose as Purpose\n"
            "purpose = Purpose.ALLOCATION\n",
            "allocation-seed",
        ),
        ("import socket as transport\n", "network"),
        ("from urllib.request import urlopen\n", "network"),
        ("from urllib import request\n", "network"),
        ("import urllib3\n", "network"),
        ("from openai import OpenAI\n", "provider"),
        ("import langchain_openai\n", "provider"),
        ("from subprocess import run\nrun(('curl', 'https://example.test'))\n", "process"),
        ("from os import system\nsystem('env')\n", "process"),
        ("import os\nvalue = os.environ['REVIEW_TOKEN']\n", "ambient-environment"),
        ("from os import getenv as read_environment\n", "ambient-environment"),
    ),
)
def test_boundary_rejects_each_forbidden_authority(
    tmp_path: Path,
    source: str,
    category: str,
) -> None:
    _write_boundary_module(tmp_path, source)

    findings = validate_public_review_boundary(tmp_path)

    assert findings
    assert any(f": {category}:" in finding for finding in findings)


@pytest.mark.parametrize("name", BOUNDARY_MODULES)
def test_every_declared_boundary_module_is_scanned(tmp_path: Path, name: str) -> None:
    _write_boundary_module(tmp_path, "import _socket\n", name=name)

    findings = validate_public_review_boundary(tmp_path)

    assert any(f"/{name}:1:" in finding for finding in findings)


def test_review_prefix_modules_are_discovered_without_an_inventory_change(tmp_path: Path) -> None:
    _write_boundary_module(tmp_path, "import anthropic\n", name="review_projection.py")

    findings = validate_public_review_boundary(tmp_path)

    assert any("/review_projection.py:1: provider:" in finding for finding in findings)


def test_aliases_and_dynamic_access_cannot_bypass_the_boundary(tmp_path: Path) -> None:
    path = _write_boundary_module(
        tmp_path,
        "import importlib as loader\n"
        "import os as operating_system\n"
        "from saliencegate.benchmarks.state_decay_v2 import config as generation\n"
        "network = loader.import_module('httpx')\n"
        "token = getattr(operating_system, 'environ')\n"
        "purpose = getattr(generation.SeedPurpose, 'ALLOCATION')\n",
    )

    findings = scan_source(path.relative_to(tmp_path), path.read_text(encoding="utf-8"))

    assert {finding.split(": ", maxsplit=2)[1] for finding in findings} >= {
        "allocation-seed",
        "ambient-environment",
        "network",
    }


@pytest.mark.parametrize(
    ("source", "category"),
    (
        (
            "from saliencegate.benchmarks.state_decay_v2.config import SeedPurpose\n"
            "purpose = SeedPurpose('allocation')\n",
            "allocation-seed",
        ),
        (
            "from saliencegate.benchmarks.state_decay_v2.config import SeedPurpose\n"
            "name = 'ALLOCATION'\n"
            "purpose = SeedPurpose[name]\n",
            "allocation-seed",
        ),
        (
            "from saliencegate.benchmarks.state_decay_v2.config import SeedPurpose\n"
            "purpose = next(item for item in SeedPurpose if item.value == 'allocation')\n",
            "allocation-seed",
        ),
        (
            "from saliencegate.benchmarks.state_decay_v2 import config\n"
            "name = 'allocate_' + 'balanced_outcomes'\n"
            "allocator = getattr(config, name)\n",
            "allocation",
        ),
        (
            "import importlib\nload = importlib.import_module\nnetwork = load('socket')\n",
            "network",
        ),
        (
            "import os\nname = 'environ'\nenvironment = getattr(os, name)\n",
            "ambient-environment",
        ),
    ),
)
def test_indirect_dynamic_authority_access_fails_closed(
    tmp_path: Path,
    source: str,
    category: str,
) -> None:
    _write_boundary_module(tmp_path, source)

    findings = validate_public_review_boundary(tmp_path)

    assert any(f": {category}:" in finding for finding in findings)


def test_public_contract_is_inside_the_checked_boundary(tmp_path: Path) -> None:
    _write_boundary_module(tmp_path, "import socket\n", name="public_contract.py")

    findings = validate_public_review_boundary(tmp_path)

    assert any("/public_contract.py:1: network:" in finding for finding in findings)


def test_only_nonallocation_seed_purpose_attributes_are_allowed(tmp_path: Path) -> None:
    _write_boundary_module(
        tmp_path,
        "from saliencegate.benchmarks.state_decay_v2.config import SeedPurpose\n"
        "id_purpose = SeedPurpose.ID\n"
        "public_purpose = SeedPurpose.PUBLIC\n",
    )

    assert validate_public_review_boundary(tmp_path) == []


def test_comments_prose_declarative_names_and_local_os_paths_are_allowed(tmp_path: Path) -> None:
    _write_boundary_module(
        tmp_path,
        '"""Reviewers attest that they did not consult an allocation or provider."""\n'
        "import os\n"
        "from enum import StrEnum\n"
        "class ChecklistItem(StrEnum):\n"
        "    REVIEWER_ALLOCATION_NONCONSULTATION = "
        "'reviewer_allocation_nonconsultation'\n"
        "def review_path(root: str, request: str) -> str:\n"
        "    return os.path.join(root, request)\n",
        name="review_contract.py",
    )

    assert validate_public_review_boundary(tmp_path) == []


def test_non_boundary_modules_are_not_scanned(tmp_path: Path) -> None:
    _write_boundary_module(tmp_path, "import socket\n", name="generator.py")

    assert validate_public_review_boundary(tmp_path) == []


def test_syntax_or_read_failure_is_fail_closed(tmp_path: Path) -> None:
    path = _write_boundary_module(tmp_path, "def incomplete(:\n")

    findings = validate_public_review_boundary(tmp_path)

    assert findings == [f"{path.relative_to(tmp_path)}:1: parse: invalid Python source"]


def test_current_repository_satisfies_the_boundary() -> None:
    root = Path(__file__).resolve().parents[3]

    assert validate_public_review_boundary(root) == []
