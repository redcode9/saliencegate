"""Single-source version identifiers for deterministic grounding artifacts."""

from typing import Literal

from saliencegate.runtime.token_counting import APPROXIMATE_TOKEN_ALGORITHM_VERSION

GROUNDING_PIPELINE_VERSION: Literal["grounding-pipeline/v1"] = "grounding-pipeline/v1"
FIXED_ASCII_RENDERER_VERSION: Literal["fixed-ascii/v1"] = "fixed-ascii/v1"
TOKEN_COUNTER_VERSION: Literal["utf8-bytes-ceil-div-4-v1"] = APPROXIMATE_TOKEN_ALGORITHM_VERSION

__all__ = [
    "FIXED_ASCII_RENDERER_VERSION",
    "GROUNDING_PIPELINE_VERSION",
    "TOKEN_COUNTER_VERSION",
]
