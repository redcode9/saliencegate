from __future__ import annotations

import saliencegate.commands.pilot as pilot_command
from saliencegate.models.openai_compatible import OpenAICompatibleConfig

configuration = OpenAICompatibleConfig(
    base_url="http://127.0.0.1:11434/v1",
    model="gpt-oss:20b",
)
if pilot_command.__name__ != "saliencegate.commands.pilot":
    raise RuntimeError("guarded pilot module did not import from the installed package")
if configuration.model != "gpt-oss:20b":
    raise RuntimeError("optional model runtime configuration changed unexpectedly")

print(configuration.model)
