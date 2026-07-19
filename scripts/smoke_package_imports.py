from __future__ import annotations

import importlib.util
import sys

import saliencegate
import saliencegate.capture
import saliencegate.models
import saliencegate.shadow
from saliencegate.capture import CaptureProfile, load_capture_capability_registry

registry = load_capture_capability_registry()
if tuple(profile.profile_id for profile in registry.profiles) != tuple(CaptureProfile):
    raise RuntimeError("capture capability registry is incomplete")

for optional_module in ("anthropic", "harbor", "httpx", "openai", "openai_harmony"):
    if importlib.util.find_spec(optional_module) is not None:
        raise RuntimeError("optional model runtime is present in the core environment")
    if optional_module in sys.modules:
        raise RuntimeError("optional model runtime was imported by the core package")

print(saliencegate.__version__)
