from __future__ import annotations

from uuid import UUID

import pytest

from saliencegate.shadow.adapters import ShadowTraceAdapter
from saliencegate.shadow.trace import ShadowTrace


class _ExampleAdapter:
    @staticmethod
    def _descriptor() -> dict[str, object]:
        return {
            "schema_version": "example-shadow-adapter/v1",
            "mapping": "documented",
        }

    @property
    def profile_id(self) -> str:
        return "example/v1"

    @property
    def profile_digest(self) -> str:
        return ShadowTrace.adapter_profile_digest(self._descriptor())

    def adapt_bytes(
        self,
        source: bytes,
        *,
        run_id: UUID,
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
    ) -> ShadowTrace:
        return ShadowTrace.from_records(
            [
                {
                    "schema_version": "shadow-input/v1",
                    "kind": "run_start",
                    "source_event_id": "start-1",
                    "occurred_at": "2026-07-17T09:00:00Z",
                },
                {
                    "schema_version": "shadow-input/v1",
                    "kind": "run_end",
                    "source_event_id": "finish-1",
                    "occurred_at": "2026-07-17T09:00:01Z",
                },
            ],
            run_id=run_id,
            adapter_profile_id=self.profile_id,
            adapter_descriptor=self._descriptor(),
            source_bytes=source,
            source_format="example",
            source_schema_version="example/v1",
            capture_scope="complete_run_declared",
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
        )


def _accept_adapter(adapter: ShadowTraceAdapter) -> str:
    return adapter.profile_id


def test_trace_adapter_is_structural_static_guidance_only() -> None:
    adapter = _ExampleAdapter()

    assert _accept_adapter(adapter) == "example/v1"
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(adapter, ShadowTraceAdapter)  # type: ignore[misc]


def test_custom_adapter_can_keep_profile_identity_coherent_using_public_api() -> None:
    adapter: ShadowTraceAdapter = _ExampleAdapter()

    trace = adapter.adapt_bytes(
        b'{"native":"source"}',
        run_id=UUID("11111111-1111-4111-8111-111111111111"),
    )

    assert trace.binding.adapter_profile_id == adapter.profile_id
    assert trace.binding.adapter_profile_digest == adapter.profile_digest
    assert trace.binding.source_digest_kind == "original_bytes"


def test_adapter_module_exports_only_the_protocol() -> None:
    from saliencegate.shadow import adapters

    assert adapters.__all__ == ["ShadowTraceAdapter"]
