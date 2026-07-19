from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import stat
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import pytest

import saliencegate.benchmarks.state_decay_v2.review_pack as review_pack_module
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ClosedTreeRead,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicLineageCandidate,
    PublicLineageRegistry,
    candidate_packet_digest,
    candidate_registry_digest,
)
from saliencegate.benchmarks.state_decay_v2.review import (
    build_public_family_comparisons,
    build_public_review_drafts,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    MAX_REVIEW_PACK_LARGE_CHILD_BYTES,
    MAX_REVIEW_PACK_MANIFEST_FILE_BYTES,
    MAX_REVIEW_PACK_SMALL_CHILD_BYTES,
    MAX_REVIEW_PACK_TOTAL_BYTES,
    PUBLIC_REVIEW_CHECKLIST,
    PublicFamilyComparison,
    PublicReviewChecklist,
    PublicReviewDraft,
    PublicReviewPackBasename,
    PublicReviewPackManifest,
    family_comparison_digest,
    pack_child_digest,
    pack_manifest_digest,
    review_draft_digest,
)
from saliencegate.benchmarks.state_decay_v2.review_pack import (
    PUBLIC_REVIEW_GUIDE_BYTES,
    PublicReviewPackError,
    PublicReviewPackErrorCode,
    ValidatedPublicReviewPack,
    load_public_review_pack,
    publish_public_review_pack,
)
from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY
from saliencegate.domain import canonical_json

_MANIFEST_NAME: Final = "review-pack.json"
_FILE_NAMES: Final = (
    "candidates.jsonl",
    "drafts.jsonl",
    "family-comparisons.jsonl",
    "checklist.json",
    "review-guide.md",
    _MANIFEST_NAME,
)
_CHILD_GOLDENS: Final = (
    (
        "candidates.jsonl",
        11_768_976,
        "ec684fc094c1ce52fd410375365d3950b4ee465f0bfef83aa714af5a4d5e1e08",
    ),
    (
        "drafts.jsonl",
        214_110,
        "08cce3fb955ba23a08c4e9c913d3938ed30687f422939317361476135822b9c5",
    ),
    (
        "family-comparisons.jsonl",
        1_985_511,
        "9b9a154ed64ed4e6b0026dccd419a53c75de5103e0dd07ae0ad588aab06df43d",
    ),
    (
        "checklist.json",
        1_455,
        "37d99e20ef00d18381e2da6a357391e416960bd05280a5c020f5d4c6c9e59869",
    ),
    (
        "review-guide.md",
        1_853,
        "27f71fb0cb04210bdd5cae88eff4b7fb33b8cbc6e7f689cadc754bcc4dc34191",
    ),
)
_GUIDE_SHA256_GOLDEN: Final = "fddc84fa7a2a7da4ff17a448ec5fb29e02afe779f424fb1ff9090e3d1f133423"
_MANIFEST_DIGEST_GOLDEN: Final = "8fa9e264270ea58730deb9247dcf9b2183e0414b510ab0bb4c6d9b0a1466f44c"
_MANIFEST_FILE_BYTES_GOLDEN: Final = 1_910


@pytest.fixture(scope="module")
def registry() -> PublicLineageRegistry:
    return PUBLIC_LINEAGE_REGISTRY


@pytest.fixture(scope="module")
def comparisons(
    registry: PublicLineageRegistry,
) -> tuple[PublicFamilyComparison, ...]:
    return build_public_family_comparisons(registry=registry)


@pytest.fixture(scope="module")
def drafts(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
) -> tuple[PublicReviewDraft, ...]:
    return build_public_review_drafts(registry=registry, comparisons=comparisons)


@pytest.fixture(scope="module")
def canonical_pack_root(
    tmp_path_factory: pytest.TempPathFactory,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> Path:
    root = tmp_path_factory.mktemp("canonical-review-pack") / "pack"
    _publish(root, registry, comparisons, drafts)
    return root


def _publish(
    output: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> PublicReviewPackManifest:
    return publish_public_review_pack(
        output=output,
        registry=registry,
        comparisons=comparisons,
        drafts=drafts,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, int, int, bytes | None], ...]:
    snapshot: list[tuple[str, int, int, int, int, bytes | None]] = []
    for path in sorted(root.iterdir()):
        metadata = path.lstat()
        data = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        snapshot.append(
            (
                path.name,
                metadata.st_mode,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_mtime_ns,
                data,
            )
        )
    return tuple(snapshot)


def _copy_pack(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    destination.chmod(0o700)
    for path in destination.iterdir():
        path.chmod(0o600)


def _jsonl(models: tuple[object, ...]) -> bytes:
    return b"".join(canonical_json(model) + b"\n" for model in models)


def _line_payloads(data: bytes) -> tuple[bytes, ...]:
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert b"\r" not in data
    lines = data.splitlines(keepends=True)
    assert all(line.endswith(b"\n") and line != b"\n" for line in lines)
    return tuple(line[:-1] for line in lines)


def _revised_registry(registry: PublicLineageRegistry) -> PublicLineageRegistry:
    candidate_payload = registry.candidates[0].model_dump(mode="python")
    candidate_payload["semantic_rationale"] = (
        "Revised pack-specific semantic rationale for immutable destination testing."
    )
    candidate_payload["candidate_packet_digest"] = candidate_packet_digest(candidate_payload)
    revised_candidate = PublicLineageCandidate.model_validate(candidate_payload)

    registry_payload = registry.model_dump(mode="python")
    registry_payload["candidates"] = (revised_candidate, *registry.candidates[1:])
    registry_payload["registry_digest"] = candidate_registry_digest(registry_payload)
    return PublicLineageRegistry.model_validate(registry_payload)


def _rewrite_child_and_manifest(
    root: Path,
    basename: PublicReviewPackBasename,
    data: bytes,
) -> PublicReviewPackManifest:
    manifest_path = root / _MANIFEST_NAME
    manifest = PublicReviewPackManifest.model_validate_json(manifest_path.read_bytes())
    payload = manifest.model_dump(mode="python")
    matching = tuple(child for child in payload["children"] if child["basename"] is basename)
    assert len(matching) == 1
    matching[0]["canonical_byte_count"] = len(data)
    matching[0]["content_digest"] = pack_child_digest(basename, data)
    payload["manifest_digest"] = pack_manifest_digest(payload)
    rewritten = PublicReviewPackManifest.model_validate(payload)
    (root / basename.value).write_bytes(data)
    manifest_path.write_bytes(canonical_json(rewritten) + b"\n")
    return rewritten


def _assert_pack_rejected(root: Path, *, secret: str | None = None) -> None:
    with pytest.raises((ValueError, RuntimeError)) as error:
        load_public_review_pack(pack=root)
    if secret is not None:
        assert secret not in str(error.value)
        assert secret not in repr(error.value)


def test_public_review_pack_exact_bytes_inventory_and_bounds(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    root = tmp_path / "review-pack"
    manifest = _publish(root, registry, comparisons, drafts)
    files = _tree_bytes(root)

    assert tuple(sorted(files)) == tuple(sorted(_FILE_NAMES))
    assert files[PublicReviewPackBasename.CANDIDATES.value] == _jsonl(registry.candidates)
    assert files[PublicReviewPackBasename.DRAFTS.value] == _jsonl(drafts)
    assert files[PublicReviewPackBasename.FAMILY_COMPARISONS.value] == _jsonl(comparisons)
    assert files[PublicReviewPackBasename.CHECKLIST.value] == (
        canonical_json(PUBLIC_REVIEW_CHECKLIST) + b"\n"
    )
    assert files[PublicReviewPackBasename.REVIEW_GUIDE.value] == (PUBLIC_REVIEW_GUIDE_BYTES)
    assert files[_MANIFEST_NAME] == canonical_json(manifest) + b"\n"

    assert len(_line_payloads(files[PublicReviewPackBasename.CANDIDATES.value])) == 180
    assert len(_line_payloads(files[PublicReviewPackBasename.DRAFTS.value])) == 180
    assert len(_line_payloads(files[PublicReviewPackBasename.FAMILY_COMPARISONS.value])) == 6
    assert tuple(child.basename for child in manifest.children) == tuple(PublicReviewPackBasename)
    assert (
        tuple(
            (child.basename.value, child.canonical_byte_count, child.content_digest)
            for child in manifest.children
        )
        == _CHILD_GOLDENS
    )
    assert manifest.manifest_digest == _MANIFEST_DIGEST_GOLDEN
    assert len(files[_MANIFEST_NAME]) == _MANIFEST_FILE_BYTES_GOLDEN
    for child in manifest.children:
        data = files[child.basename.value]
        assert child.canonical_byte_count == len(data)
        assert child.content_digest == pack_child_digest(child.basename, data)
        maximum = (
            MAX_REVIEW_PACK_SMALL_CHILD_BYTES
            if child.basename
            in (PublicReviewPackBasename.CHECKLIST, PublicReviewPackBasename.REVIEW_GUIDE)
            else MAX_REVIEW_PACK_LARGE_CHILD_BYTES
        )
        assert len(data) <= maximum
    assert len(files[_MANIFEST_NAME]) <= MAX_REVIEW_PACK_MANIFEST_FILE_BYTES
    assert sum(map(len, files.values())) <= MAX_REVIEW_PACK_TOTAL_BYTES

    root_metadata = root.stat()
    assert stat.S_IMODE(root_metadata.st_mode) == 0o700
    assert root_metadata.st_uid == os.getuid()
    for path in root.iterdir():
        metadata = path.stat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1


def test_public_review_guide_is_static_nfc_and_contains_the_full_warning(
    registry: PublicLineageRegistry,
) -> None:
    assert type(PUBLIC_REVIEW_GUIDE_BYTES) is bytes
    assert hashlib.sha256(PUBLIC_REVIEW_GUIDE_BYTES).hexdigest() == _GUIDE_SHA256_GOLDEN
    assert PUBLIC_REVIEW_GUIDE_BYTES.endswith(b"\n")
    assert not PUBLIC_REVIEW_GUIDE_BYTES.endswith(b"\n\n")
    assert b"\r" not in PUBLIC_REVIEW_GUIDE_BYTES
    text = PUBLIC_REVIEW_GUIDE_BYTES.decode("utf-8")
    assert unicodedata.normalize("NFC", text) == text

    warning = text.casefold()
    for phrase in (
        "reviewer id",
        "review rationale",
        "checklist answers",
        "decision",
        "superseded",
        "public repository data",
    ):
        assert phrase in warning
    for item in PUBLIC_REVIEW_CHECKLIST.items:
        assert item.item_id.value in text
        assert item.text in text

    assert all(
        candidate.lineage_registry_key not in text and candidate.semantic_rationale not in text
        for candidate in registry.candidates
    )
    assert "synthetic-public-reviewer" not in text


def test_review_pack_publication_signature_exposes_no_replace_or_text_inputs() -> None:
    parameters = inspect.signature(publish_public_review_pack).parameters
    assert tuple(parameters) == ("output", "registry", "comparisons", "drafts")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert not {
        "replace",
        "repair",
        "resume",
        "guide",
        "checklist",
        "reviewer_id",
        "review_rationale",
    }.intersection(parameters)


def test_public_review_pack_is_deterministic_from_reversed_material_iteration(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    first = tmp_path / "first-pack"
    second = tmp_path / "second-pack"
    first_manifest = _publish(first, registry, comparisons, drafts)
    second_manifest = _publish(
        second,
        registry,
        tuple(reversed(comparisons)),
        tuple(reversed(drafts)),
    )

    assert second_manifest == first_manifest
    assert _tree_bytes(second) == _tree_bytes(first)


def test_public_review_pack_bounds_a_dishonest_sequence_before_publication(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    class DishonestComparisons(list[PublicFamilyComparison]):
        yielded: int = 0

        def __len__(self) -> int:
            return 6

        def __iter__(self) -> Iterator[PublicFamilyComparison]:
            while True:
                self.yielded += 1
                yield comparisons[0]

    output = tmp_path / "must-not-exist"
    hostile = DishonestComparisons(comparisons)

    with pytest.raises(PublicReviewPackError) as error:
        publish_public_review_pack(
            output=output,
            registry=registry,
            comparisons=hostile,
            drafts=drafts,
        )

    assert error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT
    assert hostile.yielded == 7
    assert not output.exists()


def test_public_review_pack_load_returns_only_strict_models_and_identity(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    root = tmp_path / "loaded-pack"
    manifest = _publish(root, registry, comparisons, drafts)
    loaded = load_public_review_pack(pack=root)

    assert type(loaded) is ClosedTreeRead
    assert type(loaded.value) is ValidatedPublicReviewPack
    assert loaded.value.manifest == manifest
    assert loaded.value.registry == registry
    assert loaded.value.family_comparisons == comparisons
    assert loaded.value.drafts == drafts
    assert loaded.value.checklist == PUBLIC_REVIEW_CHECKLIST
    assert loaded.manifest == manifest
    assert loaded.manifest_digest == manifest.manifest_digest
    assert loaded.replacement_key == f"public-review-pack:{registry.registry_digest}"
    assert loaded.directory_identity.matches(root.stat())

    descriptor = loaded.descriptor
    assert descriptor.manifest_name == _MANIFEST_NAME
    assert tuple(item.key for item in descriptor.files) == tuple(PublicReviewPackBasename)
    assert tuple(item.name for item in descriptor.files) == tuple(
        basename.value for basename in PublicReviewPackBasename
    )
    assert tuple(item.expected_bytes for item in descriptor.files) == tuple(
        child.canonical_byte_count for child in manifest.children
    )
    assert not any(
        name in dir(loaded.value)
        for name in ("files", "raw_files", "review_guide_bytes", "authority", "readiness")
    )

    parameters = inspect.signature(load_public_review_pack).parameters
    assert tuple(parameters) == ("pack",)
    assert parameters["pack"].kind is inspect.Parameter.KEYWORD_ONLY


def test_identical_pack_is_a_noop_but_different_complete_pack_is_immutable(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    root = tmp_path / "immutable-pack"
    manifest = _publish(root, registry, comparisons, drafts)
    before_root = root.stat()
    before = _tree_snapshot(root)

    assert (
        _publish(
            root,
            registry,
            tuple(reversed(comparisons)),
            tuple(reversed(drafts)),
        )
        == manifest
    )
    after_root = root.stat()
    assert (after_root.st_dev, after_root.st_ino, after_root.st_mtime_ns) == (
        before_root.st_dev,
        before_root.st_ino,
        before_root.st_mtime_ns,
    )
    assert _tree_snapshot(root) == before

    revised_registry = _revised_registry(registry)
    revised_comparisons = build_public_family_comparisons(registry=revised_registry)
    revised_drafts = build_public_review_drafts(
        registry=revised_registry,
        comparisons=revised_comparisons,
    )
    with pytest.raises(ArtifactExistsError):
        _publish(root, revised_registry, revised_comparisons, revised_drafts)
    assert _tree_snapshot(root) == before


@pytest.mark.parametrize("shape", ("empty", "children-only", "file"))
def test_incomplete_destination_is_preserved_and_never_repaired(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    shape: str,
) -> None:
    root = tmp_path / f"incomplete-{shape}"
    if shape == "file":
        root.write_bytes(b"incomplete")
        before_file = root.read_bytes()
    else:
        root.mkdir(mode=0o700)
        if shape == "children-only":
            child = root / PublicReviewPackBasename.CHECKLIST.value
            child.write_bytes(canonical_json(PUBLIC_REVIEW_CHECKLIST) + b"\n")
            child.chmod(0o600)
        before = _tree_snapshot(root)

    with pytest.raises(ArtifactExistsError):
        _publish(root, registry, comparisons, drafts)

    if shape == "file":
        assert root.is_file()
        assert root.read_bytes() == before_file
    else:
        assert _tree_snapshot(root) == before


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-child",
        "extra-child",
        "missing-terminal-lf",
        "crlf",
        "noncanonical-json",
        "digest-tamper",
    ),
)
def test_loader_rejects_incomplete_noncanonical_and_digest_tampered_children(
    tmp_path: Path,
    canonical_pack_root: Path,
    mutation: str,
) -> None:
    root = tmp_path / f"tampered-{mutation}"
    _copy_pack(canonical_pack_root, root)
    candidates = root / PublicReviewPackBasename.CANDIDATES.value
    checklist = root / PublicReviewPackBasename.CHECKLIST.value

    if mutation == "missing-child":
        candidates.unlink()
    elif mutation == "extra-child":
        extra = root / "unexpected.json"
        extra.write_bytes(b"{}\n")
        extra.chmod(0o600)
    elif mutation == "missing-terminal-lf":
        candidates.write_bytes(candidates.read_bytes()[:-1])
    elif mutation == "crlf":
        data = candidates.read_bytes()
        candidates.write_bytes(data.replace(b"\n", b"\r\n", 1))
    elif mutation == "noncanonical-json":
        checklist.write_bytes(b" " + checklist.read_bytes())
    else:
        tampered = bytearray(candidates.read_bytes())
        tampered[len(tampered) // 2] ^= 1
        candidates.write_bytes(bytes(tampered))

    _assert_pack_rejected(root)


@pytest.mark.parametrize(
    "mutation",
    ("leading-space", "missing-terminal-lf", "crlf", "invalid-self-digest"),
)
def test_loader_requires_the_exact_canonical_manifest_file(
    tmp_path: Path,
    canonical_pack_root: Path,
    mutation: str,
) -> None:
    root = tmp_path / f"manifest-{mutation}"
    _copy_pack(canonical_pack_root, root)
    path = root / _MANIFEST_NAME
    data = path.read_bytes()
    if mutation == "leading-space":
        data = b" " + data
    elif mutation == "missing-terminal-lf":
        data = data[:-1]
    elif mutation == "crlf":
        data = data[:-1] + b"\r\n"
    else:
        manifest = PublicReviewPackManifest.model_validate_json(data)
        payload = manifest.model_dump(mode="python")
        payload["manifest_digest"] = "0" * 64
        data = canonical_json(payload) + b"\n"
    path.write_bytes(data)

    _assert_pack_rejected(root)


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate-order",
        "candidate-extra-line",
        "draft-order",
        "comparison-order",
        "noncanonical-checklist",
        "changed-guide",
    ),
)
def test_loader_rejects_self_consistent_semantic_and_canonical_tampering(
    tmp_path: Path,
    canonical_pack_root: Path,
    mutation: str,
) -> None:
    root = tmp_path / f"self-consistent-{mutation}"
    _copy_pack(canonical_pack_root, root)

    if mutation == "candidate-order":
        basename = PublicReviewPackBasename.CANDIDATES
        lines = list((root / basename.value).read_bytes().splitlines(keepends=True))
        lines[0], lines[1] = lines[1], lines[0]
        data = b"".join(lines)
    elif mutation == "candidate-extra-line":
        basename = PublicReviewPackBasename.CANDIDATES
        original = (root / basename.value).read_bytes()
        data = original + original.splitlines(keepends=True)[0]
    elif mutation == "draft-order":
        basename = PublicReviewPackBasename.DRAFTS
        lines = list((root / basename.value).read_bytes().splitlines(keepends=True))
        lines[0], lines[1] = lines[1], lines[0]
        data = b"".join(lines)
    elif mutation == "comparison-order":
        basename = PublicReviewPackBasename.FAMILY_COMPARISONS
        lines = list((root / basename.value).read_bytes().splitlines(keepends=True))
        lines[0], lines[1] = lines[1], lines[0]
        data = b"".join(lines)
    elif mutation == "noncanonical-checklist":
        basename = PublicReviewPackBasename.CHECKLIST
        data = b" " + (root / basename.value).read_bytes()
    else:
        basename = PublicReviewPackBasename.REVIEW_GUIDE
        data = (root / basename.value).read_bytes() + b"Untrusted guide change.\n"

    _rewrite_child_and_manifest(root, basename, data)
    _assert_pack_rejected(root, secret="Untrusted guide change")


@pytest.mark.parametrize(
    ("target", "mode"),
    (
        ("root", 0o500),
        ("root", 0o750),
        ("manifest", 0o400),
        ("manifest", 0o640),
        ("child", 0o400),
        ("child", 0o640),
    ),
)
def test_loader_requires_exact_posix_pack_modes(
    tmp_path: Path,
    canonical_pack_root: Path,
    target: str,
    mode: int,
) -> None:
    root = tmp_path / f"mode-{target}-{mode:o}"
    _copy_pack(canonical_pack_root, root)
    path = (
        root
        if target == "root"
        else root
        / (_MANIFEST_NAME if target == "manifest" else PublicReviewPackBasename.CANDIDATES.value)
    )
    path.chmod(mode)
    try:
        _assert_pack_rejected(root)
    finally:
        path.chmod(0o700 if target == "root" else 0o600)


@pytest.mark.skipif(os.name != "posix", reason="POSIX link semantics")
@pytest.mark.parametrize("mutation", ("child-symlink", "child-hardlink", "root-symlink"))
def test_loader_rejects_pack_links_and_aliases(
    tmp_path: Path,
    canonical_pack_root: Path,
    mutation: str,
) -> None:
    root = tmp_path / f"links-{mutation}"
    _copy_pack(canonical_pack_root, root)
    candidates = root / PublicReviewPackBasename.CANDIDATES.value
    if mutation == "child-symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes(candidates.read_bytes())
        outside.chmod(0o600)
        candidates.unlink()
        candidates.symlink_to(outside)
        selected = root
    elif mutation == "child-hardlink":
        os.link(candidates, tmp_path / "candidate-alias.jsonl")
        selected = root
    else:
        selected = tmp_path / "pack-alias"
        selected.symlink_to(root, target_is_directory=True)
    _assert_pack_rejected(selected)


@pytest.mark.parametrize("case", ("non-sequence", "short", "broken-length"))
def test_publication_bounds_malformed_material_sequences_before_writing(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    case: str,
) -> None:
    class BrokenLength(list[PublicFamilyComparison]):
        def __len__(self) -> int:
            raise RuntimeError("synthetic length failure")

    if case == "non-sequence":
        selected = cast(tuple[PublicFamilyComparison, ...], object())
        expected = PublicReviewPackErrorCode.MALFORMED_INPUT
    elif case == "short":
        selected = comparisons[:-1]
        expected = PublicReviewPackErrorCode.INCOMPLETE_PACK
    else:
        selected = cast(
            tuple[PublicFamilyComparison, ...],
            BrokenLength(comparisons),
        )
        expected = PublicReviewPackErrorCode.MALFORMED_INPUT
    output = tmp_path / case

    with pytest.raises(PublicReviewPackError) as captured:
        publish_public_review_pack(
            output=output,
            registry=registry,
            comparisons=selected,
            drafts=drafts,
        )

    assert captured.value.code is expected
    assert not output.exists()


@pytest.mark.parametrize("case", ("invalid-registry-digest", "unserializable-registry"))
def test_publication_revalidates_registry_inputs_value_free_before_writing(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    case: str,
) -> None:
    secret = "synthetic-untrusted-registry-value"
    selected = (
        registry.model_copy(update={"registry_digest": "0" * 64})
        if case == "invalid-registry-digest"
        else cast(PublicLineageRegistry, {secret})
    )
    output = tmp_path / case

    with pytest.raises(PublicReviewPackError) as captured:
        publish_public_review_pack(
            output=output,
            registry=selected,
            comparisons=comparisons,
            drafts=drafts,
        )

    assert captured.value.code is (
        PublicReviewPackErrorCode.DIGEST_MISMATCH
        if case == "invalid-registry-digest"
        else PublicReviewPackErrorCode.MALFORMED_INPUT
    )
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert not output.exists()


@pytest.mark.parametrize(
    "case",
    (
        "duplicate-comparison",
        "changed-comparison",
        "duplicate-draft",
        "changed-draft",
    ),
)
def test_publication_rejects_complete_but_incorrect_material_projections(
    tmp_path: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    case: str,
) -> None:
    selected_comparisons = comparisons
    selected_drafts = drafts
    if case == "duplicate-comparison":
        selected_comparisons = (*comparisons[:-1], comparisons[0])
    elif case == "changed-comparison":
        payload = comparisons[0].model_dump(mode="python")
        entries = list(payload["entries"])
        entry = dict(entries[0])
        entry["semantic_rationale"] = "Synthetic valid but incorrect family-comparison rationale."
        entries[0] = entry
        payload["entries"] = tuple(entries)
        payload["family_comparison_digest"] = family_comparison_digest(payload)
        changed_comparison = PublicFamilyComparison.model_validate(payload)
        selected_comparisons = (changed_comparison, *comparisons[1:])
    elif case == "duplicate-draft":
        selected_drafts = (*drafts[:-1], drafts[0])
    else:
        payload = drafts[0].model_dump(mode="python")
        payload["candidate_packet_digest"] = registry.candidates[1].candidate_packet_digest
        payload["draft_digest"] = review_draft_digest(payload)
        changed_draft = PublicReviewDraft.model_validate(payload)
        selected_drafts = (changed_draft, *drafts[1:])
    output = tmp_path / case

    with pytest.raises(PublicReviewPackError) as captured:
        publish_public_review_pack(
            output=output,
            registry=registry,
            comparisons=selected_comparisons,
            drafts=selected_drafts,
        )

    assert captured.value.code is PublicReviewPackErrorCode.BINDING_MISMATCH
    assert not output.exists()


def test_validated_pack_and_global_registry_bindings_fail_closed(
    canonical_pack_root: Path,
    registry: PublicLineageRegistry,
) -> None:
    loaded = load_public_review_pack(pack=canonical_pack_root).value
    pack_payload = loaded.model_dump(mode="python")
    manifest_payload = loaded.manifest.model_dump(mode="python")
    manifest_payload["candidate_registry_digest"] = "0" * 64
    manifest_payload["manifest_digest"] = pack_manifest_digest(manifest_payload)
    pack_payload["manifest"] = PublicReviewPackManifest.model_validate(manifest_payload)

    with pytest.raises(ValueError, match="bindings do not agree"):
        ValidatedPublicReviewPack.model_validate(pack_payload)

    unbound_registry = registry.model_copy(update={"generator_algorithm_digest": "0" * 64})
    with pytest.raises(PublicReviewPackError) as captured:
        review_pack_module._validate_global_bindings(unbound_registry)
    assert captured.value.code is PublicReviewPackErrorCode.BINDING_MISMATCH


def test_loader_rejects_a_self_attesting_manifest_with_foreign_global_bindings(
    tmp_path: Path,
    canonical_pack_root: Path,
) -> None:
    root = tmp_path / "foreign-manifest-binding"
    _copy_pack(canonical_pack_root, root)
    manifest_path = root / _MANIFEST_NAME
    payload = PublicReviewPackManifest.model_validate_json(manifest_path.read_bytes()).model_dump(
        mode="python"
    )
    payload["generator_algorithm_digest"] = "0" * 64
    payload["manifest_digest"] = pack_manifest_digest(payload)
    manifest_path.write_bytes(
        canonical_json(PublicReviewPackManifest.model_validate(payload)) + b"\n"
    )

    with pytest.raises(PublicReviewPackError) as captured:
        load_public_review_pack(pack=root)

    assert captured.value.code is PublicReviewPackErrorCode.BINDING_MISMATCH


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("bom", PublicReviewPackErrorCode.NONCANONICAL_CONTENT),
        ("double-terminal-lf", PublicReviewPackErrorCode.NONCANONICAL_CONTENT),
        ("blank-line", PublicReviewPackErrorCode.INCOMPLETE_PACK),
        ("invalid-json", PublicReviewPackErrorCode.MALFORMED_INPUT),
        ("noncanonical-line", PublicReviewPackErrorCode.NONCANONICAL_CONTENT),
    ),
)
def test_loader_distinguishes_self_consistent_jsonl_shape_failures(
    tmp_path: Path,
    canonical_pack_root: Path,
    mutation: str,
    expected_code: PublicReviewPackErrorCode,
) -> None:
    root = tmp_path / f"jsonl-shape-{mutation}"
    _copy_pack(canonical_pack_root, root)
    basename = PublicReviewPackBasename.CANDIDATES
    original = (root / basename.value).read_bytes()
    lines = original.splitlines(keepends=True)
    if mutation == "bom":
        data = b"\xef\xbb\xbf" + original
    elif mutation == "double-terminal-lf":
        data = original + b"\n"
    elif mutation == "blank-line":
        lines[0] = b"\n"
        data = b"".join(lines)
    elif mutation == "invalid-json":
        lines[0] = b"{\n"
        data = b"".join(lines)
    else:
        lines[0] = b" " + lines[0]
        data = b"".join(lines)
    _rewrite_child_and_manifest(root, basename, data)

    with pytest.raises(PublicReviewPackError) as captured:
        load_public_review_pack(pack=root)

    assert captured.value.code is expected_code


def test_review_pack_path_inputs_and_internal_parse_order_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPath:
        def __fspath__(self) -> str:
            raise RuntimeError("synthetic path failure")

    with pytest.raises(PublicReviewPackError) as bytes_error:
        load_public_review_pack(pack=cast(str, b"pack"))
    assert bytes_error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT

    with pytest.raises(PublicReviewPackError) as path_error:
        load_public_review_pack(pack=cast(str, BrokenPath()))
    assert path_error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT

    def parse_child_before_manifest(*args: object, **kwargs: object) -> object:
        del args
        parse_file = cast(
            Callable[[PublicReviewPackBasename, bytes], object],
            kwargs["parse_file"],
        )
        return parse_file(PublicReviewPackBasename.CANDIDATES, b"")

    monkeypatch.setattr(review_pack_module, "read_closed_tree", parse_child_before_manifest)
    with pytest.raises(PublicReviewPackError) as ordering_error:
        load_public_review_pack(pack="unused")
    assert ordering_error.value.code is PublicReviewPackErrorCode.INCOMPLETE_PACK


def test_review_pack_defensive_parsers_return_value_free_errors(
    canonical_pack_root: Path,
) -> None:
    class ExplodingModel:
        @classmethod
        def model_validate_json(cls, value: object) -> object:
            del cls, value
            raise RuntimeError("synthetic parser secret")

    exploding = cast(type[PublicReviewChecklist], ExplodingModel)
    with pytest.raises(PublicReviewPackError) as single_error:
        review_pack_module._canonical_single_model(b"{}\n", exploding)
    assert single_error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT
    assert "secret" not in str(single_error.value)

    with pytest.raises(PublicReviewPackError) as jsonl_error:
        review_pack_module._canonical_jsonl(b"{}\n", exploding, exact_lines=1)
    assert jsonl_error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT

    manifest = load_public_review_pack(pack=canonical_pack_root).manifest
    incomplete_manifest = manifest.model_copy()
    object.__setattr__(incomplete_manifest, "children", ())
    with pytest.raises(PublicReviewPackError) as child_error:
        review_pack_module._parse_child(
            PublicReviewPackBasename.CANDIDATES,
            b"",
            manifest=incomplete_manifest,
        )
    assert child_error.value.code is PublicReviewPackErrorCode.INCOMPLETE_PACK

    with pytest.raises(PublicReviewPackError) as parts_error:
        review_pack_module._finish_pack(manifest, {})
    assert parts_error.value.code is PublicReviewPackErrorCode.INCOMPLETE_PACK

    malformed_parts = {key: None for key in PublicReviewPackBasename}
    with pytest.raises(PublicReviewPackError) as type_error:
        review_pack_module._finish_pack(manifest, malformed_parts)
    assert type_error.value.code is PublicReviewPackErrorCode.MALFORMED_INPUT


def test_publication_rejects_an_internal_manifest_digest_disagreement(
    tmp_path: Path,
    canonical_pack_root: Path,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_public_review_pack(pack=canonical_pack_root)
    fake = SimpleNamespace(
        manifest_digest="0" * 64,
        descriptor=loaded.descriptor,
    )
    monkeypatch.setattr(review_pack_module, "load_public_review_pack", lambda **kwargs: fake)

    with pytest.raises(ArtifactDestinationError):
        _publish(tmp_path / "digest-disagreement", registry, comparisons, drafts)


def test_public_review_guide_literal_guard_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unicodedata, "normalize", lambda *args: "changed")
    with pytest.raises(RuntimeError, match="literal failed validation"):
        review_pack_module._build_public_review_guide_bytes()
