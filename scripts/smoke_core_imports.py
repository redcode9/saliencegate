from __future__ import annotations

import importlib.util
import sys

import saliencegate
import saliencegate.cli
import saliencegate.commands.demo
import saliencegate.models
import saliencegate.shadow

for optional_module in ("anthropic", "harbor", "httpx", "openai", "openai_harmony"):
    if importlib.util.find_spec(optional_module) is not None:
        raise RuntimeError("optional model runtime is present in the core environment")
    if optional_module in sys.modules:
        raise RuntimeError("optional model runtime was imported by the core package")

print(saliencegate.__version__)
