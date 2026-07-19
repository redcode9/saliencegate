from __future__ import annotations

import pytest

import saliencegate.runtime as runtime
from saliencegate.runtime.engine import ReplayEngine


def test_lazy_engine_export_is_resolved_and_cached() -> None:
    runtime.__dict__.pop("ReplayEngine", None)

    resolved = runtime.ReplayEngine

    assert resolved is ReplayEngine
    assert runtime.__dict__["ReplayEngine"] is ReplayEngine


def test_unknown_runtime_export_raises_attribute_error() -> None:
    missing_name = "missing_runtime_export"

    with pytest.raises(AttributeError, match=missing_name):
        getattr(runtime, missing_name)
