from __future__ import annotations

import gc
import inspect
import os
import stat
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, cast

import pytest

import saliencegate.benchmarks.state_decay_v2.review_io as review_io_module
from saliencegate.benchmarks.state_decay_v2.protocol import ReviewDecision
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicLineageCandidate,
    PublicLineageRegistry,
    candidate_packet_digest,
    candidate_registry_digest,
)
from saliencegate.benchmarks.state_decay_v2.review import (
    PublicReviewError,
    PublicReviewErrorCode,
    PublicReviewProgressState,
    build_public_family_comparisons,
    build_public_review_drafts,
    build_public_review_submission,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    PublicReviewAnswer,
    PublicReviewChecklistAnswer,
    PublicReviewChecklistItemId,
    PublicReviewEnvelope,
    PublicReviewSubmission,
    review_envelope_digest,
    review_submission_digest,
)
from saliencegate.benchmarks.state_decay_v2.review_io import (
    PublicReviewDirectorySnapshot,
    PublicReviewIOError,
    PublicReviewIOErrorCode,
    PublicReviewWriteResult,
    append_public_review_submission,
    load_public_review_progress,
    persist_public_review_envelope,
    replay_public_review_submissions,
)
from saliencegate.benchmarks.state_decay_v2.review_pack import (
    ValidatedPublicReviewPack,
    load_public_review_pack,
    publish_public_review_pack,
)
from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY
from saliencegate.domain import canonical_json

_REVIEWER_ID: Final = "synthetic-io-reviewer"
_REVIEW_RATIONALE: Final = "Synthetic explicit rationale for public review I/O tests."
_SECOND_REVIEWER_ID: Final = "synthetic-io-reviewer-two"
_SECOND_REVIEW_RATIONALE: Final = "Synthetic second explicit rationale for public review I/O tests."


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> ValidatedPublicReviewPack:
    comparisons = build_public_family_comparisons(registry=PUBLIC_LINEAGE_REGISTRY)
    drafts = build_public_review_drafts(
        registry=PUBLIC_LINEAGE_REGISTRY,
        comparisons=comparisons,
    )
    root = tmp_path_factory.mktemp("public-review-io-pack") / "pack"
    publish_public_review_pack(
        output=root,
        registry=PUBLIC_LINEAGE_REGISTRY,
        comparisons=comparisons,
        drafts=drafts,
    )
    return load_public_review_pack(pack=root).value


def _answers(*failed_indexes: int) -> tuple[PublicReviewChecklistAnswer, ...]:
    failed = frozenset(failed_indexes)
    return tuple(
        PublicReviewChecklistAnswer(
            item_id=item_id,
            answer=(PublicReviewAnswer.FAILED if index in failed else PublicReviewAnswer.PASSED),
        )
        for index, item_id in enumerate(PublicReviewChecklistItemId)
    )


def _append(
    root: Path,
    pack: ValidatedPublicReviewPack,
    *,
    predecessor: str | None,
    lineage_key: str | None = None,
    reviewer_id: str = _REVIEWER_ID,
    rationale: str = _REVIEW_RATIONALE,
    failed_indexes: tuple[int, ...] = (),
    warning: bool = True,
) -> PublicReviewWriteResult:
    return append_public_review_submission(
        review_directory=root,
        pack=pack,
        lineage_registry_key=(
            pack.drafts[0].lineage_registry_key if lineage_key is None else lineage_key
        ),
        reviewer_id=reviewer_id,
        review_rationale=rationale,
        checklist_answers=_answers(*failed_indexes),
        decision=(ReviewDecision.REJECTED if failed_indexes else ReviewDecision.ACCEPTED),
        supersedes_submission_digest=predecessor,
        publication_warning_accepted=warning,
    )


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int, int, bytes | None], ...]:
    values: list[tuple[str, int, int, int, int, bytes | None]] = []
    for path in sorted(root.iterdir()):
        metadata = path.lstat()
        values.append(
            (
                path.name,
                metadata.st_mode,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_mtime_ns,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
            )
        )
    return tuple(values)


def _immutable_file_snapshot(path: Path) -> tuple[int, int, int, int, int, int, int, bytes]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        path.read_bytes(),
    )


def _review_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("review--*.json")))


def _envelope_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("envelope--*.json")))


def _read_submission(path: Path) -> PublicReviewSubmission:
    raw = path.read_bytes()
    submission = PublicReviewSubmission.model_validate_json(raw)
    assert raw == canonical_json(submission) + b"\n"
    assert path.name == (
        f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
    )
    return submission


def _read_envelope(path: Path) -> PublicReviewEnvelope:
    raw = path.read_bytes()
    envelope = PublicReviewEnvelope.model_validate_json(raw)
    assert raw == canonical_json(envelope) + b"\n"
    assert path.name == (
        f"envelope--{envelope.lineage_registry_key}--{envelope.envelope_digest}.json"
    )
    return envelope


def _write_owner_only(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def _revised_registry(registry: PublicLineageRegistry) -> PublicLineageRegistry:
    original = registry.candidates[0]
    candidate_payload = original.model_dump(mode="python")
    candidate_payload["semantic_rationale"] = (
        "Revised explicit semantic rationale for the review I/O stale-pack test."
    )
    candidate_payload["candidate_packet_digest"] = candidate_packet_digest(candidate_payload)
    revised_candidate = PublicLineageCandidate.model_validate(candidate_payload)

    registry_payload = registry.model_dump(mode="python")
    registry_payload["candidates"] = (revised_candidate, *registry.candidates[1:])
    registry_payload["registry_digest"] = candidate_registry_digest(registry_payload)
    return PublicLineageRegistry.model_validate(registry_payload)


def _publish_revised_pack(
    root: Path,
    pack: ValidatedPublicReviewPack,
) -> ValidatedPublicReviewPack:
    registry = _revised_registry(pack.registry)
    comparisons = build_public_family_comparisons(registry=registry)
    drafts = build_public_review_drafts(registry=registry, comparisons=comparisons)
    publish_public_review_pack(
        output=root,
        registry=registry,
        comparisons=comparisons,
        drafts=drafts,
    )
    return load_public_review_pack(pack=root).value


@pytest.fixture(scope="module")
def revised_pack(
    tmp_path_factory: pytest.TempPathFactory,
    pack: ValidatedPublicReviewPack,
) -> ValidatedPublicReviewPack:
    return _publish_revised_pack(
        tmp_path_factory.mktemp("revised-public-review-io-pack") / "pack",
        pack,
    )


def _two_reviewer_chain(
    pack: ValidatedPublicReviewPack,
) -> tuple[PublicReviewSubmission, PublicReviewSubmission]:
    draft = pack.drafts[0]
    rejected = build_public_review_submission(
        draft=draft,
        reviewer_id=_REVIEWER_ID,
        review_rationale="Synthetic selected rejection for forensic replay.",
        checklist_answers=_answers(0),
        decision=ReviewDecision.REJECTED,
        supersedes_submission_digest=None,
    )
    accepted = build_public_review_submission(
        draft=draft,
        reviewer_id=_SECOND_REVIEWER_ID,
        review_rationale="Synthetic selected correction for forensic replay.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=rejected.submission_digest,
    )
    return rejected, accepted


def test_review_io_api_requires_explicit_reviewer_and_observed_head_inputs() -> None:
    expected: dict[Callable[..., object], tuple[str, ...]] = {
        load_public_review_progress: (
            "review_directory",
            "pack",
            "create",
        ),
        append_public_review_submission: (
            "review_directory",
            "pack",
            "lineage_registry_key",
            "reviewer_id",
            "review_rationale",
            "checklist_answers",
            "decision",
            "supersedes_submission_digest",
            "publication_warning_accepted",
        ),
        persist_public_review_envelope: (
            "review_directory",
            "pack",
            "lineage_registry_key",
        ),
        replay_public_review_submissions: (
            "review_directory",
            "pack",
            "submissions",
            "publication_warning_accepted_reviewer_ids",
        ),
    }
    for function, names in expected.items():
        parameters = inspect.signature(function).parameters
        assert tuple(parameters) == names
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
        )
        assert all(
            parameter.default is inspect.Parameter.empty for parameter in parameters.values()
        )

    assert not issubclass(PublicReviewDirectorySnapshot, tuple)
    assert not hasattr(PublicReviewDirectorySnapshot, "model_fields")
    assert not hasattr(PublicReviewDirectorySnapshot, "schema_version")
    assert not hasattr(PublicReviewWriteResult, "model_fields")
    assert not hasattr(PublicReviewWriteResult, "schema_version")


def test_first_append_is_canonical_owner_only_and_materializes_its_envelope(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "reviews"
    empty = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=True,
    )
    assert empty.submissions == ()
    assert empty.envelopes == ()
    assert empty.progress.missing_count == 180
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert (root / "review.lock").read_bytes() == b""
    assert stat.S_IMODE((root / "review.lock").stat().st_mode) == 0o600

    written = _append(root, pack, predecessor=None)

    review_paths = _review_paths(root)
    envelope_paths = _envelope_paths(root)
    assert len(review_paths) == len(envelope_paths) == 1
    submission = _read_submission(review_paths[0])
    envelope = _read_envelope(envelope_paths[0])
    assert submission == written.submission
    assert envelope == written.envelope
    assert envelope.submissions == (submission,)
    assert written.progress.accepted_count == 1
    assert {path.name for path in root.iterdir()} == {
        "review.lock",
        review_paths[0].name,
        envelope_paths[0].name,
    }
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.iterdir())

    reloaded = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    assert reloaded.submissions == (submission,)
    assert reloaded.envelopes == (envelope,)
    assert reloaded.progress.accepted_count == 1
    first_progress = reloaded.progress.candidates[0]
    assert first_progress.state is PublicReviewProgressState.ACCEPTED
    assert first_progress.head_submission_digest == submission.submission_digest


def test_warning_retry_stale_head_and_correction_are_fail_closed_and_append_only(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "reviews"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    before_warning = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as warning_error:
        _append(root, pack, predecessor=None, failed_indexes=(0,), warning=False)
    assert warning_error.value.code is PublicReviewIOErrorCode.WARNING_REQUIRED
    assert _snapshot(root) == before_warning

    first = _append(root, pack, predecessor=None, failed_indexes=(0,), warning=True)
    first_submission_path = _review_paths(root)[0]
    first_envelope_path = _envelope_paths(root)[0]
    first_submission_snapshot = _immutable_file_snapshot(first_submission_path)
    first_envelope_snapshot = _immutable_file_snapshot(first_envelope_path)
    before_retry = _snapshot(root)

    retried = _append(root, pack, predecessor=None, failed_indexes=(0,), warning=False)
    assert retried == first
    assert _snapshot(root) == before_retry

    with pytest.raises(PublicReviewIOError) as stale_error:
        _append(
            root,
            pack,
            predecessor=None,
            reviewer_id=_SECOND_REVIEWER_ID,
            rationale=_SECOND_REVIEW_RATIONALE,
            failed_indexes=(1,),
            warning=True,
        )
    assert stale_error.value.code is PublicReviewIOErrorCode.STALE_HEAD
    assert _snapshot(root) == before_retry

    correction = _append(
        root,
        pack,
        predecessor=first.submission.submission_digest,
        warning=False,
    )
    assert correction.submission.supersedes_submission_digest == (
        first.submission.submission_digest
    )
    assert correction.envelope.submissions == (first.submission, correction.submission)
    assert correction.progress.accepted_count == 1
    assert len(_review_paths(root)) == len(_envelope_paths(root)) == 2
    assert _immutable_file_snapshot(first_submission_path) == first_submission_snapshot
    assert _immutable_file_snapshot(first_envelope_path) == first_envelope_snapshot

    before_new_reviewer = _snapshot(root)
    with pytest.raises(PublicReviewIOError) as new_warning_error:
        _append(
            root,
            pack,
            predecessor=correction.submission.submission_digest,
            reviewer_id=_SECOND_REVIEWER_ID,
            rationale=_SECOND_REVIEW_RATIONALE,
            warning=False,
        )
    assert new_warning_error.value.code is PublicReviewIOErrorCode.WARNING_REQUIRED
    assert _snapshot(root) == before_new_reviewer


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("unexpected-name", PublicReviewIOErrorCode.UNSAFE_STORAGE),
        ("wrong-lineage-name", PublicReviewIOErrorCode.BINDING_MISMATCH),
        ("wrong-digest-name", PublicReviewIOErrorCode.DIGEST_MISMATCH),
        ("file-mode", PublicReviewIOErrorCode.UNSAFE_STORAGE),
        ("symlink", PublicReviewIOErrorCode.UNSAFE_STORAGE),
        ("hardlink", PublicReviewIOErrorCode.UNSAFE_STORAGE),
        ("oversized", PublicReviewIOErrorCode.UNSAFE_STORAGE),
    ),
)
def test_unsafe_inventory_rejects_the_entire_directory_unchanged(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
    expected_code: PublicReviewIOErrorCode,
) -> None:
    root = tmp_path / case
    written = _append(root, pack, predecessor=None)
    submission_path = _review_paths(root)[0]
    if case == "unexpected-name":
        _write_owner_only(root / "notes.json", b"{}\n")
    elif case == "wrong-lineage-name":
        submission_path.rename(
            root / (f"review--pub-fr-01--{written.submission.submission_digest}.json")
        )
    elif case == "wrong-digest-name":
        submission_path.rename(root / f"review--pub-fr-00--{'0' * 64}.json")
    elif case == "file-mode":
        submission_path.chmod(0o640)
    elif case == "symlink":
        outside = tmp_path / "outside.json"
        _write_owner_only(outside, submission_path.read_bytes())
        submission_path.unlink()
        submission_path.symlink_to(outside)
    elif case == "hardlink":
        os.link(submission_path, tmp_path / "review-alias.json")
    else:
        oversized = root / ("review--pub-fr-00--" + "f" * 64 + ".json")
        _write_owner_only(oversized, b"x" * (320 * 1024 + 2))
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(
            review_directory=root,
            pack=pack,
            create=False,
        )

    assert captured.value.code is expected_code
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("noncanonical", PublicReviewIOErrorCode.NONCANONICAL_CONTENT),
        ("digest", PublicReviewIOErrorCode.DIGEST_MISMATCH),
        ("binding", PublicReviewIOErrorCode.BINDING_MISMATCH),
    ),
)
def test_canonical_digest_and_binding_tamper_are_distinguished_value_free(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
    expected_code: PublicReviewIOErrorCode,
) -> None:
    root = tmp_path / case
    written = _append(root, pack, predecessor=None)
    secret = "fixture-secret-review-rationale"
    if case in {"noncanonical", "digest"}:
        path = _review_paths(root)[0]
        if case == "noncanonical":
            path.write_bytes(b" " + path.read_bytes())
        else:
            payload = written.submission.model_dump(mode="python")
            payload["review_rationale"] = secret
            path.write_bytes(canonical_json(payload) + b"\n")
        path.chmod(0o600)
    else:
        old_path = _envelope_paths(root)[0]
        payload = written.envelope.model_dump(mode="python")
        payload["draft_digest"] = "a" * 64
        payload["envelope_digest"] = review_envelope_digest(payload)
        new_path = root / (
            f"envelope--{written.envelope.lineage_registry_key}--{payload['envelope_digest']}.json"
        )
        old_path.unlink()
        _write_owner_only(new_path, canonical_json(payload) + b"\n")
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(
            review_directory=root,
            pack=pack,
            create=False,
        )

    assert captured.value.code is expected_code
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert _snapshot(root) == before


def test_external_fork_fails_the_whole_directory_without_mutation(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "fork"
    first = _append(root, pack, predecessor=None, failed_indexes=(0,))
    draft = pack.drafts[0]
    left = build_public_review_submission(
        draft=draft,
        reviewer_id="synthetic-left-fork-reviewer",
        review_rationale="Synthetic explicit left fork rationale.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=first.submission.submission_digest,
    )
    right = build_public_review_submission(
        draft=draft,
        reviewer_id="synthetic-right-fork-reviewer",
        review_rationale="Synthetic explicit right fork rationale.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=first.submission.submission_digest,
    )
    for submission in (left, right):
        path = root / (
            f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
        )
        _write_owner_only(path, canonical_json(submission) + b"\n")
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(
            review_directory=root,
            pack=pack,
            create=False,
        )

    assert captured.value.code is PublicReviewIOErrorCode.FORK
    assert _snapshot(root) == before


def test_concurrent_corrections_from_one_observed_head_cannot_fork(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "concurrent"
    first = _append(root, pack, predecessor=None, failed_indexes=(0,))
    barrier = threading.Barrier(2)

    def correct(index: int) -> PublicReviewWriteResult | PublicReviewIOError:
        barrier.wait(timeout=5)
        try:
            return _append(
                root,
                pack,
                predecessor=first.submission.submission_digest,
                reviewer_id=f"synthetic-concurrent-reviewer-{index}",
                rationale=f"Synthetic explicit concurrent correction rationale {index}.",
                warning=True,
            )
        except PublicReviewIOError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(correct, (0, 1)))

    successes = tuple(
        outcome for outcome in outcomes if isinstance(outcome, PublicReviewWriteResult)
    )
    failures = tuple(outcome for outcome in outcomes if isinstance(outcome, PublicReviewIOError))
    assert len(successes) == len(failures) == 1
    assert failures[0].code is PublicReviewIOErrorCode.STALE_HEAD

    loaded = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    assert len(loaded.submissions) == len(loaded.envelopes) == 2
    head = successes[0].submission
    assert head.supersedes_submission_digest == first.submission.submission_digest
    assert loaded.submissions == (first.submission, head)
    assert loaded.envelopes[-1] == successes[0].envelope
    assert loaded.progress.ambiguous_count == 0
    assert loaded.progress.accepted_count == 1


def test_orphan_submission_is_projected_and_persisted_without_rewriting_history(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "orphan"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    draft = pack.drafts[0]
    submission = build_public_review_submission(
        draft=draft,
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    submission_path = root / (
        f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
    )
    _write_owner_only(submission_path, canonical_json(submission) + b"\n")
    submission_before = _immutable_file_snapshot(submission_path)

    loaded = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    assert loaded.submissions == (submission,)
    assert len(loaded.envelopes) == 1
    assert _envelope_paths(root) == ()
    assert loaded.envelopes[0].submissions == (submission,)

    persisted = persist_public_review_envelope(
        review_directory=root,
        pack=pack,
        lineage_registry_key=submission.lineage_registry_key,
    )
    assert persisted.submission == submission
    assert persisted.envelope == loaded.envelopes[0]
    assert len(_envelope_paths(root)) == 1
    assert _read_envelope(_envelope_paths(root)[0]) == persisted.envelope
    assert _immutable_file_snapshot(submission_path) == submission_before


def test_identical_append_retry_materializes_orphan_envelope_without_rewriting_submission(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "orphan-append-retry"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    submission = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    submission_path = root / (
        f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
    )
    _write_owner_only(submission_path, canonical_json(submission) + b"\n")
    submission_before = _immutable_file_snapshot(submission_path)
    assert _envelope_paths(root) == ()

    retried = _append(
        root,
        pack,
        predecessor=None,
        warning=False,
    )

    assert retried.submission == submission
    assert retried.envelope.submissions == (submission,)
    assert len(_envelope_paths(root)) == 1
    assert _read_envelope(_envelope_paths(root)[0]) == retried.envelope
    assert _immutable_file_snapshot(submission_path) == submission_before


def test_persist_counts_only_physical_envelopes_at_the_lineage_bound(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_io_module, "MAX_REVIEW_ENVELOPES_PER_LINEAGE", 2)
    root = tmp_path / "physical-envelope-bound"
    first = _append(root, pack, predecessor=None, failed_indexes=(0,))
    correction = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale="Synthetic orphan correction at the physical envelope bound.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=first.submission.submission_digest,
    )
    correction_path = root / (
        f"review--{correction.lineage_registry_key}--{correction.submission_digest}.json"
    )
    _write_owner_only(correction_path, canonical_json(correction) + b"\n")
    correction_before = _immutable_file_snapshot(correction_path)

    projected = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    assert len(_envelope_paths(root)) == 1
    assert len(projected.envelopes) == 2
    assert projected.envelopes[-1].submissions == (first.submission, correction)

    persisted = persist_public_review_envelope(
        review_directory=root,
        pack=pack,
        lineage_registry_key=correction.lineage_registry_key,
    )

    assert persisted.submission == correction
    assert persisted.envelope.submissions == (first.submission, correction)
    assert len(_envelope_paths(root)) == 2
    assert _immutable_file_snapshot(correction_path) == correction_before


def test_candidate_history_stops_at_32_submissions_without_partial_write(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "candidate-history-bound"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    draft = pack.drafts[0]
    submissions: list[PublicReviewSubmission] = []
    predecessor: str | None = None
    for index in range(32):
        submission = build_public_review_submission(
            draft=draft,
            reviewer_id=_REVIEWER_ID,
            review_rationale=f"Synthetic bounded history entry {index}.",
            checklist_answers=_answers(),
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=predecessor,
        )
        _write_owner_only(
            root
            / (f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"),
            canonical_json(submission) + b"\n",
        )
        submissions.append(submission)
        predecessor = submission.submission_digest

    loaded = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    assert loaded.submissions == tuple(submissions)
    assert len(loaded.envelopes) == 1
    assert loaded.envelopes[0].submissions == tuple(submissions)
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        _append(
            root,
            pack,
            predecessor=submissions[-1].submission_digest,
            warning=False,
        )

    assert captured.value.code is PublicReviewIOErrorCode.HISTORY_LIMIT
    assert _snapshot(root) == before


def test_orphan_submission_becomes_stale_comparison_under_a_compatible_revised_pack(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    revised_pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "stale-orphan"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    sibling_draft = pack.drafts[1]
    submission = build_public_review_submission(
        draft=sibling_draft,
        reviewer_id=_REVIEWER_ID,
        review_rationale="Synthetic accepted review before a sibling revision.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    _write_owner_only(
        root / (f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"),
        canonical_json(submission) + b"\n",
    )
    assert _envelope_paths(root) == ()

    assert (
        revised_pack.registry.candidates[1].candidate_packet_digest
        == pack.registry.candidates[1].candidate_packet_digest
    )
    assert (
        revised_pack.registry.profile_catalog.catalog_digest
        == pack.registry.profile_catalog.catalog_digest
    )
    assert (
        revised_pack.registry.generator_configuration_digest
        == pack.registry.generator_configuration_digest
    )
    assert (
        revised_pack.registry.generator_algorithm_digest == pack.registry.generator_algorithm_digest
    )

    loaded = load_public_review_progress(
        review_directory=root,
        pack=revised_pack,
        create=False,
    )
    progress = {item.lineage_registry_key: item for item in loaded.progress.candidates}
    sibling_progress = progress[submission.lineage_registry_key]
    assert sibling_progress.state is PublicReviewProgressState.STALE_COMPARISON
    assert sibling_progress.head_submission_digest == submission.submission_digest
    assert loaded.progress.stale_comparison_count == 1
    assert loaded.progress.missing_count == 179
    assert len(loaded.envelopes) == 1
    assert loaded.envelopes[0].submissions == (submission,)
    assert loaded.envelopes[0].family_comparison_digest == (submission.family_comparison_digest)
    assert _envelope_paths(root) == ()


def test_progress_priority_is_rejected_then_missing_then_accepted(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "progress-priority"
    accepted = _append(
        root,
        pack,
        lineage_key=pack.drafts[0].lineage_registry_key,
        predecessor=None,
    )
    rejected = _append(
        root,
        pack,
        lineage_key=pack.drafts[1].lineage_registry_key,
        predecessor=None,
        reviewer_id=_SECOND_REVIEWER_ID,
        rationale=_SECOND_REVIEW_RATIONALE,
        failed_indexes=(0,),
    )
    loaded = load_public_review_progress(
        review_directory=root,
        pack=pack,
        create=False,
    )
    progress = {item.lineage_registry_key: item for item in loaded.progress.candidates}
    assert (
        progress[accepted.submission.lineage_registry_key].state
        is PublicReviewProgressState.ACCEPTED
    )
    assert (
        progress[rejected.submission.lineage_registry_key].state
        is PublicReviewProgressState.REJECTED
    )
    assert progress[pack.drafts[2].lineage_registry_key].state is PublicReviewProgressState.MISSING
    assert loaded.progress.rejected_count == 1
    assert loaded.progress.missing_count == 178
    assert loaded.progress.accepted_count == 1

    expected_priority = (
        PublicReviewProgressState.AMBIGUOUS,
        PublicReviewProgressState.STALE_COMPARISON,
        PublicReviewProgressState.REJECTED,
        PublicReviewProgressState.MISSING,
        PublicReviewProgressState.ACCEPTED,
    )
    assert tuple(PublicReviewProgressState) == expected_priority
    rank = {state: index for index, state in enumerate(expected_priority)}
    prioritized = tuple(
        sorted(
            loaded.progress.candidates,
            key=lambda item: rank[item.state],
        )
    )
    assert prioritized[0].lineage_registry_key == rejected.submission.lineage_registry_key
    assert prioritized[1].lineage_registry_key == pack.drafts[2].lineage_registry_key
    assert prioritized[-1].lineage_registry_key == accepted.submission.lineage_registry_key


def test_deep_pack_mutation_after_cache_entry_fails_closed_without_directory_write(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    isolated = ValidatedPublicReviewPack.model_validate_json(canonical_json(pack))
    root = tmp_path / "deep-pack-mutation"
    load_public_review_progress(
        review_directory=root,
        pack=isolated,
        create=True,
    )
    before = _snapshot(root)
    object.__setattr__(
        isolated.registry.candidates[0],
        "semantic_rationale",
        "Deep mutation after the validated pack entered the I/O cache.",
    )

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(
            review_directory=root,
            pack=isolated,
            create=False,
        )

    assert captured.value.code is PublicReviewIOErrorCode.DIGEST_MISMATCH
    assert _snapshot(root) == before


def test_replay_preflight_rejects_warning_and_malformed_inputs_before_creation(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    chain = _two_reviewer_chain(pack)
    warning_root = tmp_path / "warning-replay"
    with pytest.raises(PublicReviewIOError) as warning_error:
        replay_public_review_submissions(
            review_directory=warning_root,
            pack=pack,
            submissions=chain,
            publication_warning_accepted_reviewer_ids=(_REVIEWER_ID,),
        )
    assert warning_error.value.code is PublicReviewIOErrorCode.WARNING_REQUIRED
    assert not warning_root.exists()

    malformed_root = tmp_path / "malformed-replay"
    with pytest.raises(PublicReviewIOError) as malformed_error:
        replay_public_review_submissions(
            review_directory=malformed_root,
            pack=pack,
            submissions=cast(tuple[PublicReviewSubmission, ...], object()),
            publication_warning_accepted_reviewer_ids=(
                _REVIEWER_ID,
                _SECOND_REVIEWER_ID,
            ),
        )
    assert malformed_error.value.code is PublicReviewIOErrorCode.MALFORMED_INPUT
    assert not malformed_root.exists()


def test_replay_reversed_multi_history_is_canonical_idempotent_and_source_independent(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    revised_pack: ValidatedPublicReviewPack,
) -> None:
    historical_chain = _two_reviewer_chain(pack)
    current_first = build_public_review_submission(
        draft=revised_pack.drafts[0],
        reviewer_id="synthetic-current-revision-reviewer",
        review_rationale="Synthetic current revised-candidate recovery review.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    current_second = build_public_review_submission(
        draft=revised_pack.drafts[1],
        reviewer_id="synthetic-second-lineage-reviewer",
        review_rationale="Synthetic second-lineage recovery review.",
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    selected = (*historical_chain, current_first, current_second)
    confirmed_reviewers = tuple(item.reviewer_id for item in selected)
    source = tmp_path / "forensic-source"
    source.mkdir(mode=0o700)
    _write_owner_only(source / "selected-history.json", canonical_json(selected) + b"\n")
    source_before = _snapshot(source)
    destination = tmp_path / "replayed"

    replayed = replay_public_review_submissions(
        review_directory=destination,
        pack=revised_pack,
        submissions=tuple(reversed(selected)),
        publication_warning_accepted_reviewer_ids=tuple(reversed(confirmed_reviewers)),
    )

    assert {item.submission_digest for item in replayed.submissions} == {
        item.submission_digest for item in selected
    }
    assert (
        tuple(
            item
            for item in replayed.submissions
            if item.candidate_packet_digest == historical_chain[0].candidate_packet_digest
        )
        == historical_chain
    )
    envelope_heads = {envelope.submissions[-1].submission_digest for envelope in replayed.envelopes}
    assert envelope_heads == {
        current_first.submission_digest,
        current_second.submission_digest,
    }
    assert all(
        envelope.candidate_packet_digest != historical_chain[0].candidate_packet_digest
        for envelope in replayed.envelopes
    )
    assert replayed.progress.accepted_count == 2
    assert replayed.progress.missing_count == 178
    assert len(_review_paths(destination)) == len(selected)
    assert len(_envelope_paths(destination)) == 2
    assert {
        item.submission_digest for item in map(_read_submission, _review_paths(destination))
    } == {item.submission_digest for item in selected}
    assert {item.envelope_digest for item in map(_read_envelope, _envelope_paths(destination))} == {
        item.envelope_digest for item in replayed.envelopes
    }
    assert _snapshot(source) == source_before

    before_retry = _snapshot(destination)
    retried = replay_public_review_submissions(
        review_directory=destination,
        pack=revised_pack,
        submissions=tuple(reversed(selected)),
        publication_warning_accepted_reviewer_ids=confirmed_reviewers,
    )
    assert retried == replayed
    assert _snapshot(destination) == before_retry
    assert _snapshot(source) == source_before


@pytest.mark.parametrize("case", ("partial", "different"))
def test_replay_refuses_partial_or_different_destination_unchanged(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
) -> None:
    chain = _two_reviewer_chain(pack)
    root = tmp_path / case
    if case == "partial":
        load_public_review_progress(review_directory=root, pack=pack, create=True)
        first = chain[0]
        _write_owner_only(
            root / (f"review--{first.lineage_registry_key}--{first.submission_digest}.json"),
            canonical_json(first) + b"\n",
        )
    else:
        different = build_public_review_submission(
            draft=pack.drafts[0],
            reviewer_id="synthetic-different-replay-reviewer",
            review_rationale="Synthetic different recovery chain.",
            checklist_answers=_answers(),
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=None,
        )
        replay_public_review_submissions(
            review_directory=root,
            pack=pack,
            submissions=(different,),
            publication_warning_accepted_reviewer_ids=(different.reviewer_id,),
        )
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        replay_public_review_submissions(
            review_directory=root,
            pack=pack,
            submissions=chain,
            publication_warning_accepted_reviewer_ids=(
                _REVIEWER_ID,
                _SECOND_REVIEWER_ID,
            ),
        )

    assert captured.value.code is PublicReviewIOErrorCode.IMMUTABLE_CONFLICT
    assert _snapshot(root) == before


def test_persist_without_a_current_submission_is_value_free_and_nonmutating(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "empty"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        persist_public_review_envelope(
            review_directory=root,
            pack=pack,
            lineage_registry_key=pack.drafts[0].lineage_registry_key,
        )

    assert captured.value.code is PublicReviewIOErrorCode.NO_CURRENT_SUBMISSION
    assert pack.drafts[0].lineage_registry_key not in str(captured.value)
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "case",
    (
        "non-boolean-create",
        "malformed-pack",
        "non-string-lineage",
        "malformed-lineage",
        "non-sequence-answers",
        "non-boolean-warning",
        "invalid-reviewer",
    ),
)
def test_review_io_public_preflight_rejects_malformed_inputs_before_creation(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
) -> None:
    root = tmp_path / case

    with pytest.raises(PublicReviewIOError) as captured:
        if case == "non-boolean-create":
            load_public_review_progress(
                review_directory=root,
                pack=pack,
                create=cast(bool, 1),
            )
        elif case == "malformed-pack":
            load_public_review_progress(
                review_directory=root,
                pack=cast(ValidatedPublicReviewPack, object()),
                create=True,
            )
        else:
            append_public_review_submission(
                review_directory=root,
                pack=pack,
                lineage_registry_key=(
                    cast(str, 1)
                    if case == "non-string-lineage"
                    else (
                        "pub-fr-30"
                        if case == "malformed-lineage"
                        else pack.drafts[0].lineage_registry_key
                    )
                ),
                reviewer_id="" if case == "invalid-reviewer" else _REVIEWER_ID,
                review_rationale=_REVIEW_RATIONALE,
                checklist_answers=(
                    cast(tuple[PublicReviewChecklistAnswer, ...], object())
                    if case == "non-sequence-answers"
                    else _answers()
                ),
                decision=ReviewDecision.ACCEPTED,
                supersedes_submission_digest=None,
                publication_warning_accepted=(
                    cast(bool, 1) if case == "non-boolean-warning" else True
                ),
            )

    assert captured.value.code is PublicReviewIOErrorCode.MALFORMED_INPUT
    assert not root.exists()


@pytest.mark.parametrize("case", ("wrong-length", "broken-length", "short-iterator"))
def test_append_bounds_hostile_checklist_sequences_before_creation(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
) -> None:
    class BrokenLength(list[PublicReviewChecklistAnswer]):
        def __len__(self) -> int:
            raise RuntimeError("synthetic checklist length failure")

    class ShortIterator(list[PublicReviewChecklistAnswer]):
        def __iter__(self) -> Iterator[PublicReviewChecklistAnswer]:
            return iter(list.__getitem__(self, slice(0, 6)))

    if case == "wrong-length":
        selected = _answers()[:-1]
    elif case == "broken-length":
        selected = cast(
            tuple[PublicReviewChecklistAnswer, ...],
            BrokenLength(_answers()),
        )
    else:
        selected = cast(
            tuple[PublicReviewChecklistAnswer, ...],
            ShortIterator(_answers()),
        )
    root = tmp_path / case

    with pytest.raises(PublicReviewIOError) as captured:
        append_public_review_submission(
            review_directory=root,
            pack=pack,
            lineage_registry_key=pack.drafts[0].lineage_registry_key,
            reviewer_id=_REVIEWER_ID,
            review_rationale=_REVIEW_RATIONALE,
            checklist_answers=selected,
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=None,
            publication_warning_accepted=True,
        )

    assert captured.value.code is PublicReviewIOErrorCode.MALFORMED_INPUT
    assert not root.exists()


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("empty", PublicReviewIOErrorCode.HISTORY_LIMIT),
        ("overdeclared", PublicReviewIOErrorCode.HISTORY_LIMIT),
        ("broken-length", PublicReviewIOErrorCode.MALFORMED_INPUT),
        ("short-iterator", PublicReviewIOErrorCode.MALFORMED_INPUT),
        ("malformed-item", PublicReviewIOErrorCode.MALFORMED_INPUT),
        ("invalid-item-digest", PublicReviewIOErrorCode.DIGEST_MISMATCH),
    ),
)
def test_replay_bounds_and_revalidates_hostile_submission_sequences(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
    expected_code: PublicReviewIOErrorCode,
) -> None:
    submission = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )

    class Overdeclared(list[PublicReviewSubmission]):
        def __len__(self) -> int:
            return review_io_module.MAX_REVIEW_SUBMISSIONS_TOTAL + 1

    class BrokenLength(list[PublicReviewSubmission]):
        def __len__(self) -> int:
            raise RuntimeError("synthetic replay length failure")

    class ShortIterator(list[PublicReviewSubmission]):
        def __iter__(self) -> Iterator[PublicReviewSubmission]:
            return iter(())

    selected: tuple[PublicReviewSubmission, ...]
    if case == "empty":
        selected = ()
    elif case == "overdeclared":
        selected = cast(tuple[PublicReviewSubmission, ...], Overdeclared((submission,)))
    elif case == "broken-length":
        selected = cast(tuple[PublicReviewSubmission, ...], BrokenLength((submission,)))
    elif case == "short-iterator":
        selected = cast(tuple[PublicReviewSubmission, ...], ShortIterator((submission,)))
    elif case == "malformed-item":
        selected = cast(tuple[PublicReviewSubmission, ...], (object(),))
    else:
        selected = (submission.model_copy(update={"submission_digest": "0" * 64}),)
    root = tmp_path / case

    with pytest.raises(PublicReviewIOError) as captured:
        replay_public_review_submissions(
            review_directory=root,
            pack=pack,
            submissions=selected,
            publication_warning_accepted_reviewer_ids=(_REVIEWER_ID,),
        )

    assert captured.value.code is expected_code
    assert not root.exists()


@pytest.mark.parametrize(
    "case",
    (
        "non-sequence",
        "overdeclared",
        "broken-length",
        "short-iterator",
        "non-string",
        "duplicate",
    ),
)
def test_replay_bounds_confirmation_ids_before_directory_creation(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    case: str,
) -> None:
    submission = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )

    class Overdeclared(list[str]):
        def __len__(self) -> int:
            return review_io_module.MAX_REVIEW_SUBMISSIONS_TOTAL + 1

    class BrokenLength(list[str]):
        def __len__(self) -> int:
            raise RuntimeError("synthetic confirmation length failure")

    class ShortIterator(list[str]):
        def __iter__(self) -> Iterator[str]:
            return iter(())

    if case == "non-sequence":
        confirmed = cast(tuple[str, ...], object())
    elif case == "overdeclared":
        confirmed = cast(tuple[str, ...], Overdeclared((_REVIEWER_ID,)))
    elif case == "broken-length":
        confirmed = cast(tuple[str, ...], BrokenLength((_REVIEWER_ID,)))
    elif case == "short-iterator":
        confirmed = cast(tuple[str, ...], ShortIterator((_REVIEWER_ID,)))
    elif case == "non-string":
        confirmed = cast(tuple[str, ...], (1,))
    else:
        confirmed = (_REVIEWER_ID, _REVIEWER_ID)
    root = tmp_path / case

    with pytest.raises(PublicReviewIOError) as captured:
        replay_public_review_submissions(
            review_directory=root,
            pack=pack,
            submissions=(submission,),
            publication_warning_accepted_reviewer_ids=confirmed,
        )

    assert captured.value.code is PublicReviewIOErrorCode.MALFORMED_INPUT
    assert not root.exists()


@pytest.mark.parametrize("mutation", ("missing-lf", "double-lf", "crlf", "bom"))
def test_review_loader_rejects_noncanonical_file_framing(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    mutation: str,
) -> None:
    root = tmp_path / mutation
    _append(root, pack, predecessor=None)
    path = _review_paths(root)[0]
    original = path.read_bytes()
    if mutation == "missing-lf":
        data = original[:-1]
    elif mutation == "double-lf":
        data = original + b"\n"
    elif mutation == "crlf":
        data = original[:-1] + b"\r\n"
    else:
        data = b"\xef\xbb\xbf" + original
    path.write_bytes(data)
    path.chmod(0o600)
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(review_directory=root, pack=pack, create=False)

    assert captured.value.code is PublicReviewIOErrorCode.NONCANONICAL_CONTENT
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("unsafe-text", PublicReviewIOErrorCode.UNSAFE_TEXT),
        ("inconsistent-decision", PublicReviewIOErrorCode.INCONSISTENT_REVIEW),
    ),
)
def test_review_loader_maps_canonical_model_validation_failures_value_free(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    mutation: str,
    expected_code: PublicReviewIOErrorCode,
) -> None:
    root = tmp_path / mutation
    written = _append(root, pack, predecessor=None)
    old_path = _review_paths(root)[0]
    payload = written.submission.model_dump(mode="python")
    if mutation == "unsafe-text":
        payload["review_rationale"] = "synthetic\nsecret"
    else:
        answers = list(payload["checklist_answers"])
        answers[0] = {**answers[0], "answer": PublicReviewAnswer.FAILED}
        payload["checklist_answers"] = tuple(answers)
    payload["submission_digest"] = review_submission_digest(payload)
    new_path = root / (
        f"review--{written.submission.lineage_registry_key}--{payload['submission_digest']}.json"
    )
    old_path.unlink()
    _write_owner_only(new_path, canonical_json(payload) + b"\n")

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(review_directory=root, pack=pack, create=False)

    assert captured.value.code is expected_code
    assert "secret" not in str(captured.value)
    assert "secret" not in repr(captured.value)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("envelope-lineage-name", PublicReviewIOErrorCode.BINDING_MISMATCH),
        ("envelope-digest-name", PublicReviewIOErrorCode.DIGEST_MISMATCH),
    ),
)
def test_review_loader_validates_envelope_filename_bindings(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    mutation: str,
    expected_code: PublicReviewIOErrorCode,
) -> None:
    root = tmp_path / mutation
    written = _append(root, pack, predecessor=None)
    path = _envelope_paths(root)[0]
    renamed = root / (
        "envelope--"
        + (
            "pub-fr-01"
            if mutation == "envelope-lineage-name"
            else written.envelope.lineage_registry_key
        )
        + "--"
        + (written.envelope.envelope_digest if mutation == "envelope-lineage-name" else "0" * 64)
        + ".json"
    )
    path.rename(renamed)
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(review_directory=root, pack=pack, create=False)

    assert captured.value.code is expected_code
    assert _snapshot(root) == before


@pytest.mark.parametrize("limit", ("submissions", "chain", "envelopes"))
def test_review_loader_enforces_each_physical_history_bound(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    root = tmp_path / limit
    _append(root, pack, predecessor=None)
    attribute = {
        "submissions": "MAX_REVIEW_SUBMISSIONS_PER_LINEAGE",
        "chain": "MAX_REVIEW_CHAIN_LENGTH",
        "envelopes": "MAX_REVIEW_ENVELOPES_PER_LINEAGE",
    }[limit]
    monkeypatch.setattr(review_io_module, attribute, 0)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(review_directory=root, pack=pack, create=False)

    assert captured.value.code is PublicReviewIOErrorCode.HISTORY_LIMIT


def test_review_loader_rejects_an_envelope_without_its_submission_group(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "envelope-without-submission"
    _append(root, pack, predecessor=None)
    _review_paths(root)[0].unlink()
    before = _snapshot(root)

    with pytest.raises(PublicReviewIOError) as captured:
        load_public_review_progress(review_directory=root, pack=pack, create=False)

    assert captured.value.code is PublicReviewIOErrorCode.BINDING_MISMATCH
    assert _snapshot(root) == before


def test_persist_rejects_the_physical_envelope_limit_before_writing(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "zero-envelope-limit"
    load_public_review_progress(review_directory=root, pack=pack, create=True)
    submission = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )
    _write_owner_only(
        root / f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json",
        canonical_json(submission) + b"\n",
    )
    before = _snapshot(root)
    monkeypatch.setattr(review_io_module, "MAX_REVIEW_ENVELOPES_PER_LINEAGE", 0)

    with pytest.raises(PublicReviewIOError) as captured:
        persist_public_review_envelope(
            review_directory=root,
            pack=pack,
            lineage_registry_key=submission.lineage_registry_key,
        )

    assert captured.value.code is PublicReviewIOErrorCode.HISTORY_LIMIT
    assert _snapshot(root) == before


def test_review_pack_cache_is_bounded_and_drops_dead_inputs(
    pack: ValidatedPublicReviewPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: dict[object, object] = {}
    monkeypatch.setattr(review_io_module, "_PACK_CACHE", cache)
    monkeypatch.setattr(review_io_module, "_MAX_PACK_CACHE_ENTRIES", 1)
    first = ValidatedPublicReviewPack.model_validate_json(canonical_json(pack))
    second = ValidatedPublicReviewPack.model_validate_json(canonical_json(pack))

    review_io_module._checked_pack(first)
    assert len(cache) == 1
    review_io_module._checked_pack(second)
    assert len(cache) == 1
    identity = id(second)
    del second
    gc.collect()
    assert identity not in cache

    from_mapping = review_io_module._checked_pack(pack.model_dump(mode="json"))
    assert from_mapping == pack


def test_review_entry_name_validation_fails_closed_on_parser_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_key(value: object) -> object:
        del value
        raise ValueError("synthetic key parser disagreement")

    monkeypatch.setattr(review_io_module, "parse_public_lineage_key", reject_key)
    assert not review_io_module._entry_name_is_valid(f"review--pub-fr-00--{'0' * 64}.json")


def test_envelope_history_defenses_reject_prefix_and_duplicate_head_ambiguity(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
) -> None:
    root = tmp_path / "envelope-history-defenses"
    first = _append(root, pack, predecessor=None, failed_indexes=(0,))
    second = _append(
        root,
        pack,
        predecessor=first.submission.submission_digest,
    )
    snapshot = load_public_review_progress(review_directory=root, pack=pack, create=False)
    key = (
        first.submission.lineage_registry_key,
        first.submission.candidate_packet_digest,
    )
    reversed_group = {key: tuple(reversed(snapshot.submissions))}

    with pytest.raises(PublicReviewIOError) as prefix_error:
        review_io_module._ordered_envelope_history(
            (first.envelope,),
            reversed_group,
            pack,
        )
    assert prefix_error.value.code is PublicReviewIOErrorCode.BINDING_MISMATCH

    ordered_group = {key: snapshot.submissions}
    with pytest.raises(PublicReviewIOError) as duplicate_error:
        review_io_module._ordered_envelope_history(
            (second.envelope, second.envelope),
            ordered_group,
            pack,
        )
    assert duplicate_error.value.code is PublicReviewIOErrorCode.IMMUTABLE_CONFLICT


@pytest.mark.parametrize("site", ("effective", "progress", "materialize", "replay"))
def test_review_io_maps_injected_review_layer_failures_value_free(
    tmp_path: Path,
    pack: ValidatedPublicReviewPack,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    root = tmp_path / site
    submission = build_public_review_submission(
        draft=pack.drafts[0],
        reviewer_id=_REVIEWER_ID,
        review_rationale=_REVIEW_RATIONALE,
        checklist_answers=_answers(),
        decision=ReviewDecision.ACCEPTED,
        supersedes_submission_digest=None,
    )

    def fail_review(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)

    if site == "effective":
        load_public_review_progress(review_directory=root, pack=pack, create=True)
        _write_owner_only(
            root
            / f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json",
            canonical_json(submission) + b"\n",
        )
        monkeypatch.setattr(review_io_module, "build_public_review_head_envelope", fail_review)
    elif site == "progress":
        monkeypatch.setattr(review_io_module, "evaluate_public_review_gate", fail_review)
    elif site == "materialize":
        monkeypatch.setattr(review_io_module, "build_public_review_head_envelope", fail_review)
    else:
        monkeypatch.setattr(review_io_module, "evaluate_public_review_gate", fail_review)

    def operation() -> object:
        if site == "effective":
            return load_public_review_progress(
                review_directory=root,
                pack=pack,
                create=False,
            )
        if site == "progress":
            return load_public_review_progress(
                review_directory=root,
                pack=pack,
                create=True,
            )
        if site == "materialize":
            return _append(root, pack, predecessor=None)
        return replay_public_review_submissions(
            review_directory=root,
            pack=pack,
            submissions=(submission,),
            publication_warning_accepted_reviewer_ids=(_REVIEWER_ID,),
        )

    with pytest.raises(PublicReviewIOError) as captured:
        operation()

    assert captured.value.code is PublicReviewIOErrorCode.BINDING_MISMATCH
