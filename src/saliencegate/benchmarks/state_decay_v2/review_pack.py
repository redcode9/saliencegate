from __future__ import annotations

import os
import unicodedata
from collections.abc import Mapping, Sequence
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, Never, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from saliencegate.artifacts.tree import (
    ClosedTreeDescriptor,
    ClosedTreeFileSpec,
    ClosedTreeRead,
    ClosedTreeReadError,
    publish_closed_tree_exclusive,
    read_closed_tree,
)
from saliencegate.benchmarks.state_decay_v2.protocol import LINEAGE_REVIEW_PROTOCOL
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicLineageCandidate,
    PublicLineageRegistry,
)
from saliencegate.benchmarks.state_decay_v2.review import (
    build_public_family_comparisons,
    build_public_review_drafts,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    MAX_REVIEW_PACK_LARGE_CHILD_BYTES,
    MAX_REVIEW_PACK_MANIFEST_FILE_BYTES,
    MAX_REVIEW_PACK_SMALL_CHILD_BYTES,
    PUBLIC_REVIEW_CHECKLIST,
    PublicFamilyComparison,
    PublicReviewChecklist,
    PublicReviewDraft,
    PublicReviewPackBasename,
    PublicReviewPackChild,
    PublicReviewPackManifest,
    pack_child_digest,
    pack_manifest_digest,
)
from saliencegate.benchmarks.state_decay_v2.templates import (
    PUBLIC_GENERATOR_ALGORITHM,
    PUBLIC_GENERATOR_CONFIGURATION,
    PUBLIC_PROFILE_CATALOG,
)
from saliencegate.domain import canonical_json

PUBLIC_REVIEW_PACK_MANIFEST_NAME: Literal["review-pack.json"] = "review-pack.json"


class PublicReviewPackErrorCode(StrEnum):
    MALFORMED_INPUT = "malformed-input"
    NONCANONICAL_CONTENT = "noncanonical-content"
    DIGEST_MISMATCH = "digest-mismatch"
    BINDING_MISMATCH = "binding-mismatch"
    INCOMPLETE_PACK = "incomplete-pack"
    UNSAFE_STORAGE = "unsafe-storage"


_ERROR_MESSAGES = {
    PublicReviewPackErrorCode.MALFORMED_INPUT: "public review pack input is malformed",
    PublicReviewPackErrorCode.NONCANONICAL_CONTENT: ("public review pack content is not canonical"),
    PublicReviewPackErrorCode.DIGEST_MISMATCH: "public review pack digest does not match",
    PublicReviewPackErrorCode.BINDING_MISMATCH: "public review pack bindings do not agree",
    PublicReviewPackErrorCode.INCOMPLETE_PACK: "public review pack is incomplete",
    PublicReviewPackErrorCode.UNSAFE_STORAGE: "public review pack storage is unsafe",
}


class PublicReviewPackError(ValueError):
    """A stable, value-free failure at the Review Pack schema boundary."""

    def __init__(self, code: PublicReviewPackErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def _build_public_review_guide_bytes() -> bytes:
    lines = [
        "# StateDecayBench v2 Public Review Guide",
        "",
        "## Publication warning",
        "",
        (
            "Your reviewer ID, review rationale, checklist answers, decision, and any "
            "superseded submissions included in a final accepted chain may become public "
            "repository data. Submit only after you consent to that publication."
        ),
        "",
        "## Review rules",
        "",
        (
            "Review one candidate at a time without consulting or computing allocation, an "
            "allocation rank, or an assigned outcome."
        ),
        (
            "Every checklist answer and decision must be entered explicitly. The tool supplies "
            "no acceptance, rationale, answer, or correction defaults."
        ),
        (
            "A correction is a new append-only submission that names the current predecessor; "
            "historical submissions remain audit data."
        ),
        "",
        "## Frozen checklist",
        "",
    ]
    for index, item in enumerate(PUBLIC_REVIEW_CHECKLIST.items, start=1):
        lines.extend((f"{index}. `{item.item_id.value}`", f"   {item.text}", ""))
    text = "\n".join(lines).rstrip("\n") + "\n"
    if unicodedata.normalize("NFC", text) != text or "\r" in text:
        raise RuntimeError("public review guide literal failed validation")
    return text.encode("utf-8")


PUBLIC_REVIEW_GUIDE_BYTES = _build_public_review_guide_bytes()


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ValidatedPublicReviewPack(_StrictModel):
    manifest: PublicReviewPackManifest
    registry: PublicLineageRegistry
    family_comparisons: Annotated[
        tuple[PublicFamilyComparison, ...],
        Field(min_length=6, max_length=6),
    ]
    drafts: Annotated[
        tuple[PublicReviewDraft, ...],
        Field(min_length=180, max_length=180),
    ]
    checklist: PublicReviewChecklist

    @model_validator(mode="after")
    def contents_are_exactly_bound(self) -> Self:
        if (
            self.manifest.candidate_registry_digest != self.registry.registry_digest
            or self.manifest.checklist_digest != self.checklist.checklist_digest
            or self.checklist != PUBLIC_REVIEW_CHECKLIST
        ):
            raise ValueError("validated review pack bindings do not agree")
        expected_comparisons = build_public_family_comparisons(registry=self.registry)
        expected_drafts = build_public_review_drafts(
            registry=self.registry,
            comparisons=expected_comparisons,
        )
        if canonical_json(self.family_comparisons) != canonical_json(
            expected_comparisons
        ) or canonical_json(self.drafts) != canonical_json(expected_drafts):
            raise ValueError("validated review pack projections do not agree")
        return self


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _fail(code: PublicReviewPackErrorCode) -> Never:
    raise PublicReviewPackError(code)


def _validation_code(error: ValidationError) -> PublicReviewPackErrorCode:
    messages = tuple(
        str(item.get("msg", "")).casefold()
        for item in error.errors(include_url=False, include_input=False)
    )
    if any("digest" in message for message in messages):
        return PublicReviewPackErrorCode.DIGEST_MISMATCH
    return PublicReviewPackErrorCode.MALFORMED_INPUT


def _revalidate(model_type: type[_ModelT], value: object) -> _ModelT:
    try:
        serializable = (
            value.model_dump(mode="json", warnings="error")
            if isinstance(value, BaseModel)
            else value
        )
        return model_type.model_validate_json(canonical_json(serializable))
    except ValidationError as error:
        raise PublicReviewPackError(_validation_code(error)) from None
    except Exception:
        raise PublicReviewPackError(PublicReviewPackErrorCode.MALFORMED_INPUT) from None


def _bounded_sequence(
    value: object,
    *,
    exact_length: int,
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(PublicReviewPackErrorCode.MALFORMED_INPUT)
    try:
        if len(value) != exact_length:
            _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
        copied: tuple[object, ...] = tuple(islice(iter(value), exact_length + 1))
    except PublicReviewPackError:
        raise
    except Exception:
        raise PublicReviewPackError(PublicReviewPackErrorCode.MALFORMED_INPUT) from None
    if len(copied) != exact_length:
        _fail(PublicReviewPackErrorCode.MALFORMED_INPUT)
    return copied


def _validate_global_bindings(registry: PublicLineageRegistry) -> None:
    if (
        registry.generator_configuration_digest
        != PUBLIC_GENERATOR_CONFIGURATION.configuration_digest
        or registry.generator_algorithm_digest != PUBLIC_GENERATOR_ALGORITHM.algorithm_digest
        or canonical_json(registry.profile_catalog) != canonical_json(PUBLIC_PROFILE_CATALOG)
    ):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)


def _checked_registry(value: object) -> PublicLineageRegistry:
    registry = _revalidate(PublicLineageRegistry, value)
    _validate_global_bindings(registry)
    return registry


def _ordered_review_materials(
    *,
    registry: PublicLineageRegistry,
    comparisons: object,
    drafts: object,
) -> tuple[tuple[PublicFamilyComparison, ...], tuple[PublicReviewDraft, ...]]:
    raw_comparisons = _bounded_sequence(comparisons, exact_length=6)
    checked_comparisons = tuple(
        _revalidate(PublicFamilyComparison, item) for item in raw_comparisons
    )
    comparison_by_family = {item.family: item for item in checked_comparisons}
    family_order = tuple(dict.fromkeys(candidate.family for candidate in registry.candidates))
    if len(comparison_by_family) != 6 or set(comparison_by_family) != set(family_order):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)
    ordered_comparisons = tuple(comparison_by_family[family] for family in family_order)
    expected_comparisons = build_public_family_comparisons(registry=registry)
    if canonical_json(ordered_comparisons) != canonical_json(expected_comparisons):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)

    raw_drafts = _bounded_sequence(drafts, exact_length=180)
    checked_drafts = tuple(_revalidate(PublicReviewDraft, item) for item in raw_drafts)
    draft_by_key = {item.lineage_registry_key: item for item in checked_drafts}
    candidate_keys = tuple(candidate.lineage_registry_key for candidate in registry.candidates)
    if len(draft_by_key) != 180 or set(draft_by_key) != set(candidate_keys):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)
    ordered_drafts = tuple(draft_by_key[key] for key in candidate_keys)
    expected_drafts = build_public_review_drafts(
        registry=registry,
        comparisons=expected_comparisons,
    )
    if canonical_json(ordered_drafts) != canonical_json(expected_drafts):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)
    return ordered_comparisons, ordered_drafts


def _jsonl(models: Sequence[BaseModel]) -> bytes:
    return b"".join(canonical_json(model) + b"\n" for model in models)


def _single_json(model: BaseModel) -> bytes:
    return canonical_json(model) + b"\n"


def _build_pack_files(
    *,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> tuple[PublicReviewPackManifest, dict[str, bytes]]:
    child_bytes = {
        PublicReviewPackBasename.CANDIDATES: _jsonl(registry.candidates),
        PublicReviewPackBasename.DRAFTS: _jsonl(drafts),
        PublicReviewPackBasename.FAMILY_COMPARISONS: _jsonl(comparisons),
        PublicReviewPackBasename.CHECKLIST: _single_json(PUBLIC_REVIEW_CHECKLIST),
        PublicReviewPackBasename.REVIEW_GUIDE: PUBLIC_REVIEW_GUIDE_BYTES,
    }
    children = tuple(
        PublicReviewPackChild(
            basename=basename,
            canonical_byte_count=len(child_bytes[basename]),
            content_digest=pack_child_digest(basename, child_bytes[basename]),
        )
        for basename in PublicReviewPackBasename
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-pack/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": registry.generation_contract_digest,
        "lineage_review_protocol_digest": registry.lineage_review_protocol_digest,
        "generator_configuration_digest": registry.generator_configuration_digest,
        "generator_algorithm_digest": registry.generator_algorithm_digest,
        "profile_catalog_digest": registry.profile_catalog.catalog_digest,
        "candidate_registry_digest": registry.registry_digest,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "children": children,
    }
    payload["manifest_digest"] = pack_manifest_digest(payload)
    try:
        manifest = PublicReviewPackManifest.model_validate(payload)
    except ValidationError as error:
        raise PublicReviewPackError(_validation_code(error)) from None
    files = {basename.value: child_bytes[basename] for basename in PublicReviewPackBasename}
    files[PUBLIC_REVIEW_PACK_MANIFEST_NAME] = _single_json(manifest)
    return manifest, files


def _validate_manifest_bindings(manifest: PublicReviewPackManifest) -> None:
    if (
        manifest.generation_contract_digest
        != PUBLIC_GENERATOR_CONFIGURATION.generation_contract_digest
        or manifest.lineage_review_protocol_digest != LINEAGE_REVIEW_PROTOCOL.protocol_digest
        or manifest.generator_configuration_digest
        != PUBLIC_GENERATOR_CONFIGURATION.configuration_digest
        or manifest.generator_algorithm_digest != PUBLIC_GENERATOR_ALGORITHM.algorithm_digest
        or manifest.profile_catalog_digest != PUBLIC_PROFILE_CATALOG.catalog_digest
        or manifest.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest
    ):
        _fail(PublicReviewPackErrorCode.BINDING_MISMATCH)


def _canonical_single_model(
    raw: bytes,
    model_type: type[_ModelT],
) -> _ModelT:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b"\r" in raw
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        _fail(PublicReviewPackErrorCode.NONCANONICAL_CONTENT)
    payload = raw[:-1]
    try:
        model = model_type.model_validate_json(payload)
    except ValidationError as error:
        raise PublicReviewPackError(_validation_code(error)) from None
    except Exception:
        raise PublicReviewPackError(PublicReviewPackErrorCode.MALFORMED_INPUT) from None
    if canonical_json(model) != payload:
        _fail(PublicReviewPackErrorCode.NONCANONICAL_CONTENT)
    return model


def _canonical_jsonl(
    raw: bytes,
    model_type: type[_ModelT],
    *,
    exact_lines: int,
) -> tuple[_ModelT, ...]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b"\r" in raw
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        _fail(PublicReviewPackErrorCode.NONCANONICAL_CONTENT)
    if raw.count(b"\n") != exact_lines:
        _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
    lines = raw[:-1].split(b"\n", maxsplit=exact_lines - 1)
    if len(lines) != exact_lines or any(not line for line in lines):
        _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
    models: list[_ModelT] = []
    for line in lines:
        try:
            model = model_type.model_validate_json(line)
        except ValidationError as error:
            raise PublicReviewPackError(_validation_code(error)) from None
        except Exception:
            raise PublicReviewPackError(PublicReviewPackErrorCode.MALFORMED_INPUT) from None
        if canonical_json(model) != line:
            _fail(PublicReviewPackErrorCode.NONCANONICAL_CONTENT)
        models.append(model)
    return tuple(models)


def _parse_manifest_descriptor(
    raw: bytes,
) -> ClosedTreeDescriptor[PublicReviewPackManifest, PublicReviewPackBasename]:
    manifest = _canonical_single_model(raw, PublicReviewPackManifest)
    _validate_manifest_bindings(manifest)
    children = {child.basename: child for child in manifest.children}
    files = tuple(
        ClosedTreeFileSpec(
            key=basename,
            name=basename.value,
            maximum_bytes=(
                MAX_REVIEW_PACK_SMALL_CHILD_BYTES
                if basename
                in (
                    PublicReviewPackBasename.CHECKLIST,
                    PublicReviewPackBasename.REVIEW_GUIDE,
                )
                else MAX_REVIEW_PACK_LARGE_CHILD_BYTES
            ),
            expected_bytes=children[basename].canonical_byte_count,
        )
        for basename in PublicReviewPackBasename
    )
    return ClosedTreeDescriptor(
        manifest=manifest,
        manifest_name=PUBLIC_REVIEW_PACK_MANIFEST_NAME,
        manifest_digest=manifest.manifest_digest,
        replacement_key=f"public-review-pack:{manifest.candidate_registry_digest}",
        files=files,
    )


_ParsedPart = (
    tuple[PublicLineageCandidate, ...]
    | tuple[PublicReviewDraft, ...]
    | tuple[PublicFamilyComparison, ...]
    | PublicReviewChecklist
    | None
)


def _parse_child(
    key: PublicReviewPackBasename,
    raw: bytes,
    *,
    manifest: PublicReviewPackManifest,
) -> _ParsedPart:
    child = next((item for item in manifest.children if item.basename is key), None)
    if child is None:
        _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
    if (
        len(raw) != child.canonical_byte_count
        or pack_child_digest(key, raw) != child.content_digest
    ):
        _fail(PublicReviewPackErrorCode.DIGEST_MISMATCH)
    if key is PublicReviewPackBasename.CANDIDATES:
        return _canonical_jsonl(raw, PublicLineageCandidate, exact_lines=180)
    if key is PublicReviewPackBasename.DRAFTS:
        return _canonical_jsonl(raw, PublicReviewDraft, exact_lines=180)
    if key is PublicReviewPackBasename.FAMILY_COMPARISONS:
        return _canonical_jsonl(raw, PublicFamilyComparison, exact_lines=6)
    if key is PublicReviewPackBasename.CHECKLIST:
        return _canonical_single_model(raw, PublicReviewChecklist)
    if key is PublicReviewPackBasename.REVIEW_GUIDE:
        if raw != PUBLIC_REVIEW_GUIDE_BYTES:
            _fail(PublicReviewPackErrorCode.NONCANONICAL_CONTENT)
        return None
    _fail(PublicReviewPackErrorCode.MALFORMED_INPUT)


def _finish_pack(
    manifest: PublicReviewPackManifest,
    parts: Mapping[PublicReviewPackBasename, _ParsedPart],
) -> ValidatedPublicReviewPack:
    if set(parts) != set(PublicReviewPackBasename):
        _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
    candidates = parts[PublicReviewPackBasename.CANDIDATES]
    comparisons = parts[PublicReviewPackBasename.FAMILY_COMPARISONS]
    drafts = parts[PublicReviewPackBasename.DRAFTS]
    checklist = parts[PublicReviewPackBasename.CHECKLIST]
    if (
        not isinstance(candidates, tuple)
        or not isinstance(comparisons, tuple)
        or not isinstance(drafts, tuple)
        or type(checklist) is not PublicReviewChecklist
    ):
        _fail(PublicReviewPackErrorCode.MALFORMED_INPUT)
    registry_payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-lineage-registry/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": manifest.generation_contract_digest,
        "lineage_review_protocol_digest": manifest.lineage_review_protocol_digest,
        "generator_configuration_digest": manifest.generator_configuration_digest,
        "generator_algorithm_digest": manifest.generator_algorithm_digest,
        "profile_catalog": PUBLIC_PROFILE_CATALOG,
        "candidates": candidates,
        "registry_digest": manifest.candidate_registry_digest,
    }
    try:
        registry = PublicLineageRegistry.model_validate(registry_payload)
        value = ValidatedPublicReviewPack(
            manifest=manifest,
            registry=registry,
            family_comparisons=cast(tuple[PublicFamilyComparison, ...], comparisons),
            drafts=cast(tuple[PublicReviewDraft, ...], drafts),
            checklist=checklist,
        )
    except ValidationError as error:
        raise PublicReviewPackError(_validation_code(error)) from None
    except PublicReviewPackError:
        raise
    except Exception:
        raise PublicReviewPackError(PublicReviewPackErrorCode.BINDING_MISMATCH) from None
    return value


def load_public_review_pack(
    *,
    pack: os.PathLike[str] | str,
) -> ClosedTreeRead[
    ValidatedPublicReviewPack,
    PublicReviewPackManifest,
    PublicReviewPackBasename,
]:
    if isinstance(pack, bytes):
        _fail(PublicReviewPackErrorCode.MALFORMED_INPUT)
    try:
        root = Path(os.fspath(pack))
    except Exception:
        raise PublicReviewPackError(PublicReviewPackErrorCode.MALFORMED_INPUT) from None
    holder: dict[str, PublicReviewPackManifest] = {}

    def parse_manifest(
        raw: bytes,
    ) -> ClosedTreeDescriptor[PublicReviewPackManifest, PublicReviewPackBasename]:
        descriptor = _parse_manifest_descriptor(raw)
        holder["manifest"] = descriptor.manifest
        return descriptor

    def parse_file(key: PublicReviewPackBasename, raw: bytes) -> _ParsedPart:
        manifest = holder.get("manifest")
        if manifest is None:
            _fail(PublicReviewPackErrorCode.INCOMPLETE_PACK)
        return _parse_child(key, raw, manifest=manifest)

    try:
        return read_closed_tree(
            root / PUBLIC_REVIEW_PACK_MANIFEST_NAME,
            maximum_manifest_bytes=MAX_REVIEW_PACK_MANIFEST_FILE_BYTES,
            parse_manifest=parse_manifest,
            parse_file=parse_file,
            finish=_finish_pack,
        )
    except ClosedTreeReadError:
        raise PublicReviewPackError(PublicReviewPackErrorCode.UNSAFE_STORAGE) from None


def publish_public_review_pack(
    *,
    output: os.PathLike[str] | str,
    registry: PublicLineageRegistry,
    comparisons: Sequence[PublicFamilyComparison],
    drafts: Sequence[PublicReviewDraft],
) -> PublicReviewPackManifest:
    checked_registry = _checked_registry(registry)
    ordered_comparisons, ordered_drafts = _ordered_review_materials(
        registry=checked_registry,
        comparisons=comparisons,
        drafts=drafts,
    )
    manifest, files = _build_pack_files(
        registry=checked_registry,
        comparisons=ordered_comparisons,
        drafts=ordered_drafts,
    )

    def validate_tree(
        path: Path,
        expected_digest: str | None,
    ) -> ClosedTreeDescriptor[PublicReviewPackManifest, PublicReviewPackBasename]:
        loaded = load_public_review_pack(pack=path)
        if expected_digest is not None and loaded.manifest_digest != expected_digest:
            _fail(PublicReviewPackErrorCode.DIGEST_MISMATCH)
        return loaded.descriptor

    publish_closed_tree_exclusive(
        output,
        files,
        manifest_name=PUBLIC_REVIEW_PACK_MANIFEST_NAME,
        maximum_manifest_bytes=MAX_REVIEW_PACK_MANIFEST_FILE_BYTES,
        parse_manifest=_parse_manifest_descriptor,
        validate_tree=validate_tree,
    )
    return manifest


__all__ = [
    "PUBLIC_REVIEW_GUIDE_BYTES",
    "PUBLIC_REVIEW_PACK_MANIFEST_NAME",
    "PublicReviewPackError",
    "PublicReviewPackErrorCode",
    "ValidatedPublicReviewPack",
    "load_public_review_pack",
    "publish_public_review_pack",
]
