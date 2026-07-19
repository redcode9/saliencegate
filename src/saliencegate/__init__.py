from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saliencegate")
except PackageNotFoundError:  # pragma: no cover - only used outside an installed package
    __version__ = "0.1.0"

__all__ = ["__version__"]
