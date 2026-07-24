from __future__ import annotations

import importlib.util
import sys

import saliencegate
import saliencegate.capture
import saliencegate.models
import saliencegate.shadow
from saliencegate.capture import CaptureProfile, load_capture_capability_registry
from saliencegate.capture.migrations import discover_capture_migrations

registry = load_capture_capability_registry()
if tuple(profile.profile_id for profile in registry.profiles) != tuple(CaptureProfile):
    raise RuntimeError("capture capability registry is incomplete")

migrations = discover_capture_migrations()
if tuple((migration.version, migration.name, migration.checksum) for migration in migrations) != (
    (
        1,
        "capture_store",
        "b829f4b21bc4859ab352a1ed8513672686622edda2f5bc248a7dc195b4677a77",
    ),
    (
        2,
        "transport_receipts",
        "aecc36dde4533ee6b86bccde783a2c73dd2881e4bd0beccec9ceb136a1ee2c42",
    ),
    (
        3,
        "global_scopes",
        "ed1b4fedd2481eeda33704f91fc4ea41a05a52a870a5d340bfe12c0d4519ed7f",
    ),
):
    raise RuntimeError("capture migration resources are incomplete")

for optional_module in ("anthropic", "harbor", "httpx", "openai", "openai_harmony"):
    if importlib.util.find_spec(optional_module) is not None:
        raise RuntimeError("optional model runtime is present in the core environment")
    if optional_module in sys.modules:
        raise RuntimeError("optional model runtime was imported by the core package")

print(saliencegate.__version__)
