from __future__ import annotations

import os
import re
import weakref
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from typing import Never, TypeVar

from pydantic import BaseModel, ValidationError

from saliencegate.artifacts.exclusive import (
    ExclusiveStorageError,
    LockedFlatDirectory,
    open_locked_flat_directory,
)
from saliencegate.benchmarks.state_decay_v2.protocol import ReviewDecision
from saliencegate.benchmarks.state_decay_v2.public_contract import parse_public_lineage_key
from saliencegate.benchmarks.state_decay_v2.review import (
    PublicReviewError,
    PublicReviewErrorCode,
    PublicReviewGateReport,
    build_public_review_head_envelope,
    build_public_review_submission,
    evaluate_public_review_gate,
    validate_public_review_submission_chain,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    MAX_REVIEW_CHAIN_LENGTH,
    MAX_REVIEW_ENVELOPE_CANONICAL_BYTES,
    MAX_REVIEW_SUBMISSION_FILE_BYTES,
    PUBLIC_REVIEW_CHECKLIST,
    PUBLIC_REVIEW_RECORD_COUNT,
    PublicReviewChecklistAnswer,
    PublicReviewDraft,
    PublicReviewEnvelope,
    PublicReviewSubmission,
)
from saliencegate.benchmarks.state_decay_v2.review_pack import ValidatedPublicReviewPack
from saliencegate.domain import canonical_json

REVIEW_LOCK_NAME = "review.lock"
MAX_REVIEW_SUBMISSIONS_PER_LINEAGE = 64
MAX_REVIEW_ENVELOPES_PER_LINEAGE = 64
MAX_REVIEW_SUBMISSIONS_TOTAL = PUBLIC_REVIEW_RECORD_COUNT * MAX_REVIEW_SUBMISSIONS_PER_LINEAGE
MAX_REVIEW_DIRECTORY_ENTRIES = PUBLIC_REVIEW_RECORD_COUNT * (
    MAX_REVIEW_SUBMISSIONS_PER_LINEAGE + MAX_REVIEW_ENVELOPES_PER_LINEAGE
)
MAX_REVIEW_DIRECTORY_FILE_BYTES = MAX_REVIEW_ENVELOPE_CANONICAL_BYTES + 1

_ENTRY_NAME = re.compile(
    r"^(review|envelope)--(pub-(?:fr|fp|ns|sm|sf|rd)-(?:[0-2][0-9]))--"
    r"([0-9a-f]{64})\.json$"
)


class PublicReviewIOErrorCode(StrEnum):
    MALFORMED_INPUT = "malformed-input"
    MALFORMED_CONTENT = "malformed-content"
    NONCANONICAL_CONTENT = "noncanonical-content"
    DIGEST_MISMATCH = "digest-mismatch"
    BINDING_MISMATCH = "binding-mismatch"
    UNSAFE_TEXT = "unsafe-text"
    INCONSISTENT_REVIEW = "inconsistent-review"
    MISSING_PREDECESSOR = "missing-predecessor"
    FORK = "fork"
    CYCLE = "cycle"
    DUPLICATE = "duplicate"
    MULTIPLE_HEAD = "multiple-head"
    HISTORY_LIMIT = "history-limit"
    STALE_HEAD = "stale-head"
    WARNING_REQUIRED = "publication-warning-required"
    NO_CURRENT_SUBMISSION = "missing-head"
    IMMUTABLE_CONFLICT = "immutable-conflict"
    UNSAFE_STORAGE = "unsafe-storage"


_ERROR_MESSAGES = {
    PublicReviewIOErrorCode.MALFORMED_INPUT: "public review I/O input is malformed",
    PublicReviewIOErrorCode.MALFORMED_CONTENT: "public review I/O content is malformed",
    PublicReviewIOErrorCode.NONCANONICAL_CONTENT: "public review I/O content is not canonical",
    PublicReviewIOErrorCode.DIGEST_MISMATCH: "public review I/O digest does not match",
    PublicReviewIOErrorCode.BINDING_MISMATCH: "public review I/O bindings do not agree",
    PublicReviewIOErrorCode.UNSAFE_TEXT: "public review I/O text is unsafe",
    PublicReviewIOErrorCode.INCONSISTENT_REVIEW: "public review I/O declaration is inconsistent",
    PublicReviewIOErrorCode.MISSING_PREDECESSOR: "public review I/O predecessor is missing",
    PublicReviewIOErrorCode.FORK: "public review I/O history contains a fork",
    PublicReviewIOErrorCode.CYCLE: "public review I/O history contains a cycle",
    PublicReviewIOErrorCode.DUPLICATE: "public review I/O history contains a duplicate",
    PublicReviewIOErrorCode.MULTIPLE_HEAD: "public review I/O history has multiple heads",
    PublicReviewIOErrorCode.HISTORY_LIMIT: "public review I/O history limit was reached",
    PublicReviewIOErrorCode.STALE_HEAD: "public review I/O observed head is stale",
    PublicReviewIOErrorCode.WARNING_REQUIRED: "public review publication warning is required",
    PublicReviewIOErrorCode.NO_CURRENT_SUBMISSION: "public review I/O current head is missing",
    PublicReviewIOErrorCode.IMMUTABLE_CONFLICT: "public review I/O immutable content conflicts",
    PublicReviewIOErrorCode.UNSAFE_STORAGE: "public review I/O storage is unsafe",
}


class PublicReviewIOError(ValueError):
    """A stable, value-free failure at the local reviewer-directory boundary."""

    def __init__(self, code: PublicReviewIOErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class PublicReviewDirectorySnapshot:
    """Revalidated in-memory reviewer history and its progress-only report."""

    submissions: tuple[PublicReviewSubmission, ...]
    envelopes: tuple[PublicReviewEnvelope, ...]
    progress: PublicReviewGateReport


@dataclass(frozen=True, slots=True)
class PublicReviewWriteResult:
    submission: PublicReviewSubmission
    envelope: PublicReviewEnvelope
    progress: PublicReviewGateReport


_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ResultT = TypeVar("_ResultT")
_MAX_PACK_CACHE_ENTRIES = 8
_PACK_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[ValidatedPublicReviewPack],
        bytes,
        ValidatedPublicReviewPack,
    ],
] = {}


def _fail(code: PublicReviewIOErrorCode) -> Never:
    raise PublicReviewIOError(code)


def _validation_code(error: ValidationError) -> PublicReviewIOErrorCode:
    messages = tuple(
        str(item.get("msg", "")).casefold()
        for item in error.errors(include_url=False, include_input=False)
    )
    if any("digest" in message for message in messages):
        return PublicReviewIOErrorCode.DIGEST_MISMATCH
    if any("review-safe" in message for message in messages):
        return PublicReviewIOErrorCode.UNSAFE_TEXT
    if any("binding" in message or "does not match" in message for message in messages):
        return PublicReviewIOErrorCode.BINDING_MISMATCH
    if any(
        "checklist answers" in message or "decision is inconsistent" in message
        for message in messages
    ):
        return PublicReviewIOErrorCode.INCONSISTENT_REVIEW
    return PublicReviewIOErrorCode.MALFORMED_CONTENT


_REVIEW_ERROR_CODES = {
    PublicReviewErrorCode.MALFORMED_INPUT: PublicReviewIOErrorCode.MALFORMED_INPUT,
    PublicReviewErrorCode.INCOMPLETE_MATERIALS: PublicReviewIOErrorCode.MALFORMED_CONTENT,
    PublicReviewErrorCode.BINDING_MISMATCH: PublicReviewIOErrorCode.BINDING_MISMATCH,
    PublicReviewErrorCode.DIGEST_MISMATCH: PublicReviewIOErrorCode.DIGEST_MISMATCH,
    PublicReviewErrorCode.UNSAFE_TEXT: PublicReviewIOErrorCode.UNSAFE_TEXT,
    PublicReviewErrorCode.INCONSISTENT_REVIEW: PublicReviewIOErrorCode.INCONSISTENT_REVIEW,
    PublicReviewErrorCode.MISSING_PREDECESSOR: PublicReviewIOErrorCode.MISSING_PREDECESSOR,
    PublicReviewErrorCode.FORK: PublicReviewIOErrorCode.FORK,
    PublicReviewErrorCode.CYCLE: PublicReviewIOErrorCode.CYCLE,
    PublicReviewErrorCode.DUPLICATE: PublicReviewIOErrorCode.DUPLICATE,
    PublicReviewErrorCode.MULTIPLE_HEAD: PublicReviewIOErrorCode.MULTIPLE_HEAD,
}


def _map_review_error(error: PublicReviewError) -> Never:
    raise PublicReviewIOError(_REVIEW_ERROR_CODES[error.code]) from None


def _checked_pack(value: object) -> ValidatedPublicReviewPack:
    try:
        if type(value) is ValidatedPublicReviewPack:
            snapshot = canonical_json(value.model_dump(mode="json", warnings="error"))
            cached = _PACK_CACHE.get(id(value))
            if cached is not None and cached[0]() is value and cached[1] == snapshot:
                return cached[2]
            checked = ValidatedPublicReviewPack.model_validate_json(snapshot)
            identity = id(value)

            def discard(
                reference: weakref.ReferenceType[ValidatedPublicReviewPack],
            ) -> None:
                current = _PACK_CACHE.get(identity)
                if current is not None and current[0] is reference:
                    _PACK_CACHE.pop(identity, None)

            reference = weakref.ref(value, discard)
            if identity not in _PACK_CACHE and len(_PACK_CACHE) >= _MAX_PACK_CACHE_ENTRIES:
                _PACK_CACHE.pop(next(iter(_PACK_CACHE)))
            _PACK_CACHE[identity] = (reference, snapshot, checked)
            return checked
        return ValidatedPublicReviewPack.model_validate_json(canonical_json(value))
    except ValidationError as error:
        raise PublicReviewIOError(_validation_code(error)) from None
    except PublicReviewIOError:
        raise
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None


def _entry_name_is_valid(name: str) -> bool:
    match = _ENTRY_NAME.fullmatch(name)
    if match is None:
        return False
    try:
        parse_public_lineage_key(match.group(2))
    except ValueError:
        return False
    return True


def _parse_canonical_file(raw: bytes, model_type: type[_ModelT]) -> _ModelT:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b"\r" in raw
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        _fail(PublicReviewIOErrorCode.NONCANONICAL_CONTENT)
    payload = raw[:-1]
    try:
        model = model_type.model_validate_json(payload)
    except ValidationError as error:
        raise PublicReviewIOError(_validation_code(error)) from None
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_CONTENT) from None
    if canonical_json(model) != payload:
        _fail(PublicReviewIOErrorCode.NONCANONICAL_CONTENT)
    return model


def _file_bytes(model: BaseModel) -> bytes:
    try:
        return canonical_json(model) + b"\n"
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None


def _bounded_answers(value: object) -> tuple[PublicReviewChecklistAnswer, ...]:
    expected = len(PUBLIC_REVIEW_CHECKLIST.items)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    try:
        if len(value) != expected:
            _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
        copied = tuple(islice(iter(value), expected + 1))
    except PublicReviewIOError:
        raise
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None
    if len(copied) != expected:
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    return tuple(copied)


def _revalidate_submission(value: object) -> PublicReviewSubmission:
    try:
        serializable = (
            value.model_dump(mode="json", warnings="error")
            if isinstance(value, BaseModel)
            else value
        )
        return PublicReviewSubmission.model_validate_json(canonical_json(serializable))
    except ValidationError as error:
        raise PublicReviewIOError(_validation_code(error)) from None
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None


def _bounded_replay_submissions(value: object) -> tuple[PublicReviewSubmission, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    try:
        declared = len(value)
        if not 1 <= declared <= MAX_REVIEW_SUBMISSIONS_TOTAL:
            _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)
        copied = tuple(islice(iter(value), MAX_REVIEW_SUBMISSIONS_TOTAL + 1))
    except PublicReviewIOError:
        raise
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None
    if len(copied) != declared:
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    return tuple(_revalidate_submission(item) for item in copied)


def _bounded_confirmation_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    try:
        declared = len(value)
        if not 0 <= declared <= MAX_REVIEW_SUBMISSIONS_TOTAL:
            _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
        copied = tuple(islice(iter(value), MAX_REVIEW_SUBMISSIONS_TOTAL + 1))
    except PublicReviewIOError:
        raise
    except Exception:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None
    if (
        len(copied) != declared
        or any(type(item) is not str for item in copied)
        or len(set(copied)) != len(copied)
    ):
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    return copied


def _global_bindings_match(
    item: PublicReviewSubmission | PublicReviewEnvelope,
    pack: ValidatedPublicReviewPack,
) -> bool:
    return (
        item.checklist_digest == pack.checklist.checklist_digest
        and item.profile_catalog_digest == pack.registry.profile_catalog.catalog_digest
        and item.generator_configuration_digest == pack.registry.generator_configuration_digest
        and item.generator_algorithm_digest == pack.registry.generator_algorithm_digest
    )


def _validate_coordinates(
    item: PublicReviewSubmission | PublicReviewEnvelope,
    pack: ValidatedPublicReviewPack,
) -> None:
    candidate = next(
        (
            candidate
            for candidate in pack.registry.candidates
            if candidate.lineage_registry_key == item.lineage_registry_key
        ),
        None,
    )
    if (
        candidate is None
        or candidate.split is not item.split
        or candidate.family is not item.family
        or not _global_bindings_match(item, pack)
    ):
        _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)


def _read_entries(
    directory: LockedFlatDirectory,
    pack: ValidatedPublicReviewPack,
) -> tuple[list[PublicReviewSubmission], list[PublicReviewEnvelope]]:
    submissions: list[PublicReviewSubmission] = []
    envelopes: list[PublicReviewEnvelope] = []
    for name in directory.names:
        match = _ENTRY_NAME.fullmatch(name)
        if match is None:
            _fail(PublicReviewIOErrorCode.UNSAFE_STORAGE)
        role, lineage_key, named_digest = match.groups()
        if role == "review":
            raw = directory.read_regular(
                name,
                maximum_bytes=MAX_REVIEW_SUBMISSION_FILE_BYTES,
            )
            submission = _parse_canonical_file(
                raw,
                PublicReviewSubmission,
            )
            if submission.lineage_registry_key != lineage_key:
                _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
            if submission.submission_digest != named_digest:
                _fail(PublicReviewIOErrorCode.DIGEST_MISMATCH)
            submissions.append(submission)
            item: PublicReviewSubmission | PublicReviewEnvelope = submission
        else:
            raw = directory.read_regular(
                name,
                maximum_bytes=MAX_REVIEW_ENVELOPE_CANONICAL_BYTES + 1,
            )
            envelope = _parse_canonical_file(raw, PublicReviewEnvelope)
            if envelope.lineage_registry_key != lineage_key:
                _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
            if envelope.envelope_digest != named_digest:
                _fail(PublicReviewIOErrorCode.DIGEST_MISMATCH)
            envelopes.append(envelope)
            item = envelope
        _validate_coordinates(item, pack)
    return submissions, envelopes


def _ordered_submission_history(
    submissions: Sequence[PublicReviewSubmission],
    pack: ValidatedPublicReviewPack,
) -> tuple[
    tuple[PublicReviewSubmission, ...],
    dict[tuple[str, str], tuple[PublicReviewSubmission, ...]],
]:
    lineage_counts: defaultdict[str, int] = defaultdict(int)
    grouped: defaultdict[tuple[str, str], list[PublicReviewSubmission]] = defaultdict(list)
    for submission in submissions:
        lineage_counts[submission.lineage_registry_key] += 1
        if lineage_counts[submission.lineage_registry_key] > MAX_REVIEW_SUBMISSIONS_PER_LINEAGE:
            _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)
        grouped[(submission.lineage_registry_key, submission.candidate_packet_digest)].append(
            submission
        )

    ordered_groups: dict[tuple[str, str], tuple[PublicReviewSubmission, ...]] = {}
    for key, values in grouped.items():
        if len(values) > MAX_REVIEW_CHAIN_LENGTH:
            _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)
        try:
            ordered_groups[key] = validate_public_review_submission_chain(submissions=tuple(values))
        except PublicReviewError as error:
            _map_review_error(error)

    lineage_order = {
        candidate.lineage_registry_key: index
        for index, candidate in enumerate(pack.registry.candidates)
    }
    logical: list[PublicReviewSubmission] = []
    for key in sorted(
        ordered_groups,
        key=lambda item: (lineage_order[item[0]], item[1]),
    ):
        logical.extend(ordered_groups[key])
    return tuple(logical), ordered_groups


def _ordered_envelope_history(
    envelopes: Sequence[PublicReviewEnvelope],
    ordered_groups: dict[tuple[str, str], tuple[PublicReviewSubmission, ...]],
    pack: ValidatedPublicReviewPack,
) -> tuple[PublicReviewEnvelope, ...]:
    lineage_counts: defaultdict[str, int] = defaultdict(int)
    seen_heads: set[tuple[str, str, str]] = set()
    for envelope in envelopes:
        lineage_counts[envelope.lineage_registry_key] += 1
        if lineage_counts[envelope.lineage_registry_key] > MAX_REVIEW_ENVELOPES_PER_LINEAGE:
            _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)
        key = (envelope.lineage_registry_key, envelope.candidate_packet_digest)
        group = ordered_groups.get(key)
        if group is None or len(envelope.submissions) > len(group):
            _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
        expected_prefix = group[: len(envelope.submissions)]
        if canonical_json(envelope.submissions) != canonical_json(expected_prefix):
            _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
        head_key = (*key, envelope.submissions[-1].submission_digest)
        if head_key in seen_heads:
            _fail(PublicReviewIOErrorCode.IMMUTABLE_CONFLICT)
        seen_heads.add(head_key)

    lineage_order = {
        candidate.lineage_registry_key: index
        for index, candidate in enumerate(pack.registry.candidates)
    }
    return tuple(
        sorted(
            envelopes,
            key=lambda item: (
                lineage_order[item.lineage_registry_key],
                item.candidate_packet_digest,
                len(item.submissions),
                item.envelope_digest,
            ),
        )
    )


def _effective_envelopes(
    stored: tuple[PublicReviewEnvelope, ...],
    ordered_groups: dict[tuple[str, str], tuple[PublicReviewSubmission, ...]],
    pack: ValidatedPublicReviewPack,
) -> tuple[PublicReviewEnvelope, ...]:
    effective = {envelope.envelope_digest: envelope for envelope in stored}
    current_digests = {
        candidate.lineage_registry_key: candidate.candidate_packet_digest
        for candidate in pack.registry.candidates
    }
    for (lineage_key, candidate_digest), chain in ordered_groups.items():
        if current_digests[lineage_key] != candidate_digest:
            continue
        try:
            envelope = build_public_review_head_envelope(
                registry=pack.registry,
                submissions=chain,
            )
        except PublicReviewError as error:
            _map_review_error(error)
        effective.setdefault(envelope.envelope_digest, envelope)

    lineage_order = {
        candidate.lineage_registry_key: index
        for index, candidate in enumerate(pack.registry.candidates)
    }
    return tuple(
        sorted(
            effective.values(),
            key=lambda item: (
                lineage_order[item.lineage_registry_key],
                item.candidate_packet_digest,
                len(item.submissions),
                item.envelope_digest,
            ),
        )
    )


def _scan_locked(
    directory: LockedFlatDirectory,
    pack: ValidatedPublicReviewPack,
) -> PublicReviewDirectorySnapshot:
    raw_submissions, raw_envelopes = _read_entries(directory, pack)
    submissions, groups = _ordered_submission_history(raw_submissions, pack)
    stored_envelopes = _ordered_envelope_history(raw_envelopes, groups, pack)
    envelopes = _effective_envelopes(stored_envelopes, groups, pack)
    try:
        progress = evaluate_public_review_gate(
            registry=pack.registry,
            comparisons=pack.family_comparisons,
            drafts=pack.drafts,
            envelopes=envelopes,
        )
    except PublicReviewError as error:
        _map_review_error(error)
    return PublicReviewDirectorySnapshot(
        submissions=submissions,
        envelopes=envelopes,
        progress=progress,
    )


def _open_and_run(
    *,
    review_directory: os.PathLike[str] | str,
    create: bool,
    operation: Callable[[LockedFlatDirectory], _ResultT],
) -> _ResultT:
    if type(create) is not bool:
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    try:
        with open_locked_flat_directory(
            review_directory,
            create=create,
            lock_name=REVIEW_LOCK_NAME,
            maximum_entries=MAX_REVIEW_DIRECTORY_ENTRIES,
            maximum_file_bytes=MAX_REVIEW_DIRECTORY_FILE_BYTES,
            entry_name_validator=_entry_name_is_valid,
        ) as directory:
            return operation(directory)
    except PublicReviewIOError:
        raise
    except ExclusiveStorageError:
        raise PublicReviewIOError(PublicReviewIOErrorCode.UNSAFE_STORAGE) from None


def _checked_lineage_key(value: object, pack: ValidatedPublicReviewPack) -> str:
    if type(value) is not str:
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    try:
        parse_public_lineage_key(value)
    except ValueError:
        raise PublicReviewIOError(PublicReviewIOErrorCode.MALFORMED_INPUT) from None
    if all(candidate.lineage_registry_key != value for candidate in pack.registry.candidates):
        _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
    return value


def _draft_for_key(pack: ValidatedPublicReviewPack, lineage_key: str) -> PublicReviewDraft:
    draft = next(
        (item for item in pack.drafts if item.lineage_registry_key == lineage_key),
        None,
    )
    if draft is None:
        _fail(PublicReviewIOErrorCode.BINDING_MISMATCH)
    return draft


def _current_chain(
    snapshot: PublicReviewDirectorySnapshot,
    pack: ValidatedPublicReviewPack,
    lineage_key: str,
) -> tuple[PublicReviewSubmission, ...]:
    candidate_digest = next(
        candidate.candidate_packet_digest
        for candidate in pack.registry.candidates
        if candidate.lineage_registry_key == lineage_key
    )
    return tuple(
        submission
        for submission in snapshot.submissions
        if submission.lineage_registry_key == lineage_key
        and submission.candidate_packet_digest == candidate_digest
    )


def _result_for_head(
    snapshot: PublicReviewDirectorySnapshot,
    submission: PublicReviewSubmission,
) -> PublicReviewWriteResult:
    envelope = next(
        (
            item
            for item in snapshot.envelopes
            if item.submissions[-1].submission_digest == submission.submission_digest
        ),
        None,
    )
    if envelope is None:
        _fail(PublicReviewIOErrorCode.NO_CURRENT_SUBMISSION)
    return PublicReviewWriteResult(
        submission=submission,
        envelope=envelope,
        progress=snapshot.progress,
    )


def _materialize_head_envelope_locked(
    *,
    directory: LockedFlatDirectory,
    pack: ValidatedPublicReviewPack,
    lineage_key: str,
    chain: tuple[PublicReviewSubmission, ...],
) -> PublicReviewEnvelope:
    try:
        envelope = build_public_review_head_envelope(
            registry=pack.registry,
            submissions=chain,
        )
    except PublicReviewError as error:
        _map_review_error(error)
    name = f"envelope--{lineage_key}--{envelope.envelope_digest}.json"
    if name in directory.names:
        return envelope
    physical_count = sum(item.startswith(f"envelope--{lineage_key}--") for item in directory.names)
    if physical_count >= MAX_REVIEW_ENVELOPES_PER_LINEAGE:
        _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)
    directory.create_regular_exclusive(
        name,
        _file_bytes(envelope),
        maximum_bytes=MAX_REVIEW_ENVELOPE_CANONICAL_BYTES + 1,
    )
    return envelope


def load_public_review_progress(
    *,
    review_directory: os.PathLike[str] | str,
    pack: ValidatedPublicReviewPack,
    create: bool,
) -> PublicReviewDirectorySnapshot:
    checked_pack = _checked_pack(pack)
    return _open_and_run(
        review_directory=review_directory,
        create=create,
        operation=lambda directory: _scan_locked(directory, checked_pack),
    )


def append_public_review_submission(
    *,
    review_directory: os.PathLike[str] | str,
    pack: ValidatedPublicReviewPack,
    lineage_registry_key: str,
    reviewer_id: str,
    review_rationale: str,
    checklist_answers: Sequence[PublicReviewChecklistAnswer],
    decision: ReviewDecision,
    supersedes_submission_digest: str | None,
    publication_warning_accepted: bool,
) -> PublicReviewWriteResult:
    checked_pack = _checked_pack(pack)
    lineage_key = _checked_lineage_key(lineage_registry_key, checked_pack)
    answers = _bounded_answers(checklist_answers)
    if type(publication_warning_accepted) is not bool:
        _fail(PublicReviewIOErrorCode.MALFORMED_INPUT)
    draft = _draft_for_key(checked_pack, lineage_key)
    try:
        proposed = build_public_review_submission(
            draft=draft,
            reviewer_id=reviewer_id,
            review_rationale=review_rationale,
            checklist_answers=answers,
            decision=decision,
            supersedes_submission_digest=supersedes_submission_digest,
        )
    except PublicReviewError as error:
        _map_review_error(error)

    def append(directory: LockedFlatDirectory) -> PublicReviewWriteResult:
        before = _scan_locked(directory, checked_pack)
        known_reviewers = {item.reviewer_id for item in before.submissions}
        if proposed.reviewer_id not in known_reviewers and not publication_warning_accepted:
            _fail(PublicReviewIOErrorCode.WARNING_REQUIRED)

        chain = _current_chain(before, checked_pack, lineage_key)
        observed_head = chain[-1].submission_digest if chain else None
        if chain and proposed.submission_digest == observed_head:
            _materialize_head_envelope_locked(
                directory=directory,
                pack=checked_pack,
                lineage_key=lineage_key,
                chain=chain,
            )
            after_retry = _scan_locked(directory, checked_pack)
            return _result_for_head(after_retry, chain[-1])
        if supersedes_submission_digest != observed_head:
            _fail(PublicReviewIOErrorCode.STALE_HEAD)
        if any(item.submission_digest == proposed.submission_digest for item in before.submissions):
            _fail(PublicReviewIOErrorCode.IMMUTABLE_CONFLICT)
        if (
            len(chain) >= MAX_REVIEW_CHAIN_LENGTH
            or sum(item.lineage_registry_key == lineage_key for item in before.submissions)
            >= MAX_REVIEW_SUBMISSIONS_PER_LINEAGE
        ):
            _fail(PublicReviewIOErrorCode.HISTORY_LIMIT)

        submission_name = f"review--{lineage_key}--{proposed.submission_digest}.json"
        directory.create_regular_exclusive(
            submission_name,
            _file_bytes(proposed),
            maximum_bytes=MAX_REVIEW_SUBMISSION_FILE_BYTES,
        )
        _materialize_head_envelope_locked(
            directory=directory,
            pack=checked_pack,
            lineage_key=lineage_key,
            chain=(*chain, proposed),
        )
        after = _scan_locked(directory, checked_pack)
        return _result_for_head(after, proposed)

    return _open_and_run(
        review_directory=review_directory,
        create=True,
        operation=append,
    )


def persist_public_review_envelope(
    *,
    review_directory: os.PathLike[str] | str,
    pack: ValidatedPublicReviewPack,
    lineage_registry_key: str,
) -> PublicReviewWriteResult:
    checked_pack = _checked_pack(pack)
    lineage_key = _checked_lineage_key(lineage_registry_key, checked_pack)

    def persist(directory: LockedFlatDirectory) -> PublicReviewWriteResult:
        before = _scan_locked(directory, checked_pack)
        chain = _current_chain(before, checked_pack, lineage_key)
        if not chain:
            _fail(PublicReviewIOErrorCode.NO_CURRENT_SUBMISSION)
        head = chain[-1]
        _materialize_head_envelope_locked(
            directory=directory,
            pack=checked_pack,
            lineage_key=lineage_key,
            chain=chain,
        )
        after = _scan_locked(directory, checked_pack)
        return _result_for_head(after, head)

    return _open_and_run(
        review_directory=review_directory,
        create=False,
        operation=persist,
    )


def replay_public_review_submissions(
    *,
    review_directory: os.PathLike[str] | str,
    pack: ValidatedPublicReviewPack,
    submissions: Sequence[PublicReviewSubmission],
    publication_warning_accepted_reviewer_ids: Sequence[str],
) -> PublicReviewDirectorySnapshot:
    """Replay explicitly selected valid histories into one empty recovery directory."""

    checked_pack = _checked_pack(pack)
    selected = _bounded_replay_submissions(submissions)
    for submission in selected:
        _validate_coordinates(submission, checked_pack)
    ordered, groups = _ordered_submission_history(selected, checked_pack)
    confirmed_ids = _bounded_confirmation_ids(publication_warning_accepted_reviewer_ids)
    reviewer_ids = {submission.reviewer_id for submission in ordered}
    if set(confirmed_ids) != reviewer_ids:
        _fail(PublicReviewIOErrorCode.WARNING_REQUIRED)
    envelopes = _effective_envelopes((), groups, checked_pack)
    try:
        evaluate_public_review_gate(
            registry=checked_pack.registry,
            comparisons=checked_pack.family_comparisons,
            drafts=checked_pack.drafts,
            envelopes=envelopes,
        )
    except PublicReviewError as error:
        _map_review_error(error)
    expected_names = {
        *(
            f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
            for submission in ordered
        ),
        *(
            f"envelope--{envelope.lineage_registry_key}--{envelope.envelope_digest}.json"
            for envelope in envelopes
        ),
    }

    def replay(directory: LockedFlatDirectory) -> PublicReviewDirectorySnapshot:
        before = _scan_locked(directory, checked_pack)
        if (
            before.submissions == ordered
            and before.envelopes == envelopes
            and set(directory.names) == expected_names
        ):
            return before
        if before.submissions or before.envelopes or directory.names:
            _fail(PublicReviewIOErrorCode.IMMUTABLE_CONFLICT)
        for submission in ordered:
            name = f"review--{submission.lineage_registry_key}--{submission.submission_digest}.json"
            directory.create_regular_exclusive(
                name,
                _file_bytes(submission),
                maximum_bytes=MAX_REVIEW_SUBMISSION_FILE_BYTES,
            )
        for envelope in envelopes:
            envelope_name = (
                f"envelope--{envelope.lineage_registry_key}--{envelope.envelope_digest}.json"
            )
            directory.create_regular_exclusive(
                envelope_name,
                _file_bytes(envelope),
                maximum_bytes=MAX_REVIEW_ENVELOPE_CANONICAL_BYTES + 1,
            )
        return _scan_locked(directory, checked_pack)

    return _open_and_run(
        review_directory=review_directory,
        create=True,
        operation=replay,
    )


__all__ = [
    "MAX_REVIEW_DIRECTORY_ENTRIES",
    "MAX_REVIEW_DIRECTORY_FILE_BYTES",
    "MAX_REVIEW_ENVELOPES_PER_LINEAGE",
    "MAX_REVIEW_SUBMISSIONS_PER_LINEAGE",
    "MAX_REVIEW_SUBMISSIONS_TOTAL",
    "REVIEW_LOCK_NAME",
    "PublicReviewDirectorySnapshot",
    "PublicReviewIOError",
    "PublicReviewIOErrorCode",
    "PublicReviewWriteResult",
    "append_public_review_submission",
    "load_public_review_progress",
    "persist_public_review_envelope",
    "replay_public_review_submissions",
]
