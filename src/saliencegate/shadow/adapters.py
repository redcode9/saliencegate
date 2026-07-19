"""Structural whole-trace adapter contract for provider-neutral integrations."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from saliencegate.shadow.trace import ShadowTrace


class ShadowTraceAdapter(Protocol):
    """Adapt complete native bytes into one validated immutable Shadow trace."""

    @property
    def profile_id(self) -> str: ...

    @property
    def profile_digest(self) -> str: ...

    def adapt_bytes(
        self,
        source: bytes,
        *,
        run_id: UUID,
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
    ) -> ShadowTrace: ...


__all__ = ["ShadowTraceAdapter"]
