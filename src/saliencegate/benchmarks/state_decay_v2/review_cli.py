"""Offline, human-only command adapter for the StateDecayBench v2 public review."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Never, TypeVar

if TYPE_CHECKING:
    from saliencegate.benchmarks.state_decay_v2.protocol import ReviewDecision
    from saliencegate.benchmarks.state_decay_v2.public_contract import PublicLineageCandidate
    from saliencegate.benchmarks.state_decay_v2.review import (
        PublicReviewCandidateProgress,
        PublicReviewGateReport,
    )
    from saliencegate.benchmarks.state_decay_v2.review_contract import (
        PublicReviewChecklistAnswer,
    )
    from saliencegate.benchmarks.state_decay_v2.review_pack import ValidatedPublicReviewPack

_SUCCESS = 0
_INVALID_INPUT = 2
_CORRUPTED_STATE = 5
_INTERNAL_ERROR = 70
_INTERRUPTED = 130

_BUILD_PACK_REPORT_SCHEMA = "state-decay-v2-public-review-cli-build-pack-report/v1"
_STATUS_REPORT_SCHEMA = "state-decay-v2-public-review-cli-status-report/v1"
_ENVELOPE_REPORT_SCHEMA = "state-decay-v2-public-review-cli-envelope-report/v1"
_MODULE_PROGRAM = "python -m saliencegate.benchmarks.state_decay_v2.review_cli"
_CONSOLE_PROGRAM = "saliencegate-review"
_PUBLICATION_CONFIRMATION = "I ACCEPT PUBLICATION"
_PUBLICATION_WARNING = (
    "Publication warning: your reviewer ID, review rationale, checklist answers, decision, "
    "and superseded submissions may become public repository data.\n"
)
_REVIEWER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_MAX_PROMPT_UTF8_BYTES = 4_096

_ValueT = TypeVar("_ValueT")


class _UsageError(ValueError):
    pass


class _ReviewInputError(ValueError):
    pass


class _ReviewStateError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> Never:
        del message
        raise _UsageError


def _parser(program: str = _MODULE_PROGRAM) -> _SafeArgumentParser:
    parser = _SafeArgumentParser(prog=program)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build_pack = subcommands.add_parser("build-pack")
    build_pack.add_argument("--output", required=True)
    build_pack.add_argument("--json", action="store_true")

    review = subcommands.add_parser("review")
    review.add_argument("--pack", required=True)
    review.add_argument("--reviews", required=True)

    status = subcommands.add_parser("status")
    status.add_argument("--pack", required=True)
    status.add_argument("--reviews", required=True)
    status.add_argument("--json", action="store_true")

    build_envelope = subcommands.add_parser("build-envelope")
    build_envelope.add_argument("--pack", required=True)
    build_envelope.add_argument("--reviews", required=True)
    build_envelope.add_argument("--lineage-key", required=True)
    build_envelope.add_argument("--json", action="store_true")
    return parser


def _invocation_program(argv: Sequence[str] | None) -> str:
    if argv is not None or not sys.argv or type(sys.argv[0]) is not str:
        return _MODULE_PROGRAM
    basename = sys.argv[0].replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if basename.removesuffix(".exe") == _CONSOLE_PROGRAM:
        return _CONSOLE_PROGRAM
    return _MODULE_PROGRAM


def _validate_path_arguments(arguments: argparse.Namespace) -> None:
    for name in ("output", "pack", "reviews"):
        if not hasattr(arguments, name):
            continue
        value = getattr(arguments, name)
        try:
            encoded = value.encode("utf-8", errors="strict") if type(value) is str else b""
        except UnicodeError:
            raise _ReviewInputError from None
        if not encoded or len(encoded) > 4_096 or b"\0" in encoded:
            raise _ReviewInputError


def _write_stdout(value: str) -> None:
    sys.stdout.write(value)
    sys.stdout.flush()


def _write_stderr(value: str) -> None:
    sys.stderr.write(value)
    sys.stderr.flush()


def _write_json(value: object) -> None:
    _write_stdout(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _escaped_model_json(
    value: object,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("safe review rendering requires a model")
    payload = model_dump(mode="json", warnings="error", exclude=exclude)
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_pack(path: str) -> ValidatedPublicReviewPack:
    from saliencegate.benchmarks.state_decay_v2.review_pack import load_public_review_pack

    return load_public_review_pack(pack=path).value


def _build_pack_report(
    *,
    candidate_count: int,
    candidate_registry_digest: str,
    checklist_digest: str,
    manifest_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": _BUILD_PACK_REPORT_SCHEMA,
        "status": "ok",
        "candidate_count": candidate_count,
        "candidate_registry_digest": candidate_registry_digest,
        "checklist_digest": checklist_digest,
        "pack_manifest_digest": manifest_digest,
    }


def _status_report(progress: PublicReviewGateReport) -> dict[str, object]:
    return {
        "schema_version": _STATUS_REPORT_SCHEMA,
        "status": "ok",
        "ambiguous_count": progress.ambiguous_count,
        "stale_comparison_count": progress.stale_comparison_count,
        "rejected_count": progress.rejected_count,
        "missing_count": progress.missing_count,
        "accepted_count": progress.accepted_count,
        "progress_complete": progress.progress_complete,
        "candidate_registry_digest": progress.candidate_registry_digest,
        "checklist_digest": progress.checklist_digest,
    }


def _render_build_pack_human(report: dict[str, object]) -> str:
    return (
        "Public review pack built.\n"
        f"candidates: {report['candidate_count']}\n"
        f"candidate registry digest: {report['candidate_registry_digest']}\n"
        f"checklist digest: {report['checklist_digest']}\n"
        f"pack manifest digest: {report['pack_manifest_digest']}\n"
    )


def _render_status_human(progress: PublicReviewGateReport) -> str:
    return (
        "Public review progress\n"
        f"ambiguous: {progress.ambiguous_count}\n"
        f"stale comparison: {progress.stale_comparison_count}\n"
        f"rejected: {progress.rejected_count}\n"
        f"missing: {progress.missing_count}\n"
        f"accepted: {progress.accepted_count}\n"
        f"complete: {'yes' if progress.progress_complete else 'no'}\n"
        f"candidate registry digest: {progress.candidate_registry_digest}\n"
        f"checklist digest: {progress.checklist_digest}\n"
    )


def _dispatch_build_pack(arguments: argparse.Namespace) -> int:
    from saliencegate.benchmarks.state_decay_v2.review import (
        build_public_family_comparisons,
        build_public_review_drafts,
    )
    from saliencegate.benchmarks.state_decay_v2.review_contract import (
        PUBLIC_REVIEW_CHECKLIST,
    )
    from saliencegate.benchmarks.state_decay_v2.review_pack import (
        publish_public_review_pack,
    )
    from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY

    comparisons = build_public_family_comparisons(registry=PUBLIC_LINEAGE_REGISTRY)
    drafts = build_public_review_drafts(
        registry=PUBLIC_LINEAGE_REGISTRY,
        comparisons=comparisons,
    )
    manifest = publish_public_review_pack(
        output=arguments.output,
        registry=PUBLIC_LINEAGE_REGISTRY,
        comparisons=comparisons,
        drafts=drafts,
    )
    report = _build_pack_report(
        candidate_count=len(PUBLIC_LINEAGE_REGISTRY.candidates),
        candidate_registry_digest=PUBLIC_LINEAGE_REGISTRY.registry_digest,
        checklist_digest=PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        manifest_digest=manifest.manifest_digest,
    )
    if arguments.json:
        _write_json(report)
    else:
        _write_stdout(_render_build_pack_human(report))
    return _SUCCESS


def _dispatch_status(arguments: argparse.Namespace) -> int:
    from saliencegate.benchmarks.state_decay_v2.review_io import (
        load_public_review_progress,
    )

    pack = _load_pack(arguments.pack)
    snapshot = load_public_review_progress(
        review_directory=arguments.reviews,
        pack=pack,
        create=False,
    )
    if arguments.json:
        _write_json(_status_report(snapshot.progress))
    else:
        _write_stdout(_render_status_human(snapshot.progress))
    return _SUCCESS


def _target_progress(
    progress: PublicReviewGateReport,
) -> PublicReviewCandidateProgress | None:
    from saliencegate.benchmarks.state_decay_v2.review import PublicReviewProgressState

    if progress.ambiguous_count:
        raise _ReviewStateError
    for state in (
        PublicReviewProgressState.STALE_COMPARISON,
        PublicReviewProgressState.REJECTED,
        PublicReviewProgressState.MISSING,
    ):
        target = next((item for item in progress.candidates if item.state is state), None)
        if target is not None:
            return target
    return None


def _candidate_for_key(pack: ValidatedPublicReviewPack, lineage_key: str) -> PublicLineageCandidate:
    candidate = next(
        (item for item in pack.registry.candidates if item.lineage_registry_key == lineage_key),
        None,
    )
    if candidate is None:
        raise _ReviewStateError
    return candidate


def _render_review_materials(
    *,
    pack: ValidatedPublicReviewPack,
    candidate: PublicLineageCandidate,
    target_state: str,
    observed_head: str | None,
) -> None:
    comparison = next(item for item in pack.family_comparisons if item.family is candidate.family)
    _write_stdout("Family comparison (JSON):\n")
    _write_stdout(_escaped_model_json(comparison) + "\n")
    _write_stdout("Candidate (JSON):\n")
    _write_stdout(_escaped_model_json(candidate, exclude=frozenset({"previews"})) + "\n")
    for index, preview in enumerate(candidate.previews, start=1):
        _write_stdout(f"Preview {index} (JSON):\n")
        _write_stdout(_escaped_model_json(preview) + "\n")
    _write_stdout("Checklist (JSON):\n")
    _write_stdout(_escaped_model_json(pack.checklist) + "\n")
    _write_stdout(f"Target state: {target_state}\n")
    _write_stdout(f"Observed head: {observed_head if observed_head is not None else 'none'}\n")


def _read_prompt_line(prompt: str) -> str:
    _write_stdout(prompt)
    try:
        value = sys.stdin.readline(_MAX_PROMPT_UTF8_BYTES + 2)
    except UnicodeError:
        raise _ReviewInputError from None
    if value == "":
        raise EOFError
    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise _ReviewInputError from None
    if len(encoded) > _MAX_PROMPT_UTF8_BYTES:
        raise _ReviewInputError
    return value


def _prompt_until_valid(prompt: str, validator: Callable[[str], _ValueT]) -> _ValueT:
    while True:
        value = _read_prompt_line(prompt)
        try:
            return validator(value)
        except (TypeError, ValueError):
            _write_stdout("Invalid input.\n")


def _reviewer_id(value: str) -> str:
    if type(value) is not str or _REVIEWER_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid reviewer ID")
    return value


def _rationale(value: str) -> str:
    from saliencegate.benchmarks.state_decay_v2.public_contract import (
        validate_review_safe_text,
    )

    return validate_review_safe_text(value)


def _choice(*choices: str) -> Callable[[str], str]:
    allowed = frozenset(choices)

    def validate(value: str) -> str:
        if value not in allowed:
            raise ValueError("invalid choice")
        return value

    return validate


def _prompt_review(
    *,
    pack: ValidatedPublicReviewPack,
    known_reviewer_ids: frozenset[str],
) -> tuple[str, bool, tuple[PublicReviewChecklistAnswer, ...], str, ReviewDecision]:
    from saliencegate.benchmarks.state_decay_v2.protocol import ReviewDecision
    from saliencegate.benchmarks.state_decay_v2.review_contract import (
        PublicReviewAnswer,
        PublicReviewChecklistAnswer,
    )

    reviewer_id = _prompt_until_valid("Reviewer ID (required): ", _reviewer_id)
    warning_accepted = reviewer_id in known_reviewer_ids
    if not warning_accepted:
        _write_stdout(_PUBLICATION_WARNING)
        _prompt_until_valid(
            f"Type exactly {_PUBLICATION_CONFIRMATION}: ",
            _choice(_PUBLICATION_CONFIRMATION),
        )
        warning_accepted = True

    answers = tuple(
        PublicReviewChecklistAnswer(
            item_id=item.item_id,
            answer=PublicReviewAnswer(
                _prompt_until_valid(
                    f"{item.item_id.value} [passed/failed]: ",
                    _choice(PublicReviewAnswer.PASSED.value, PublicReviewAnswer.FAILED.value),
                )
            ),
        )
        for item in pack.checklist.items
    )
    rationale = _prompt_until_valid("Review rationale (required): ", _rationale)
    all_passed = all(item.answer is PublicReviewAnswer.PASSED for item in answers)
    while True:
        decision = ReviewDecision(
            _prompt_until_valid(
                "Decision [accepted/rejected]: ",
                _choice(ReviewDecision.ACCEPTED.value, ReviewDecision.REJECTED.value),
            )
        )
        if (decision is ReviewDecision.ACCEPTED) == all_passed:
            break
        _write_stdout("Invalid input.\n")
    return reviewer_id, warning_accepted, answers, rationale, decision


def _dispatch_review(arguments: argparse.Namespace) -> int:
    from saliencegate.benchmarks.state_decay_v2.review_io import (
        append_public_review_submission,
        load_public_review_progress,
    )

    pack = _load_pack(arguments.pack)
    snapshot = load_public_review_progress(
        review_directory=arguments.reviews,
        pack=pack,
        create=True,
    )
    target = _target_progress(snapshot.progress)
    if target is None:
        _write_stdout("Review complete.\n")
        return _SUCCESS

    lineage_key = target.lineage_registry_key
    candidate = _candidate_for_key(pack, lineage_key)
    _render_review_materials(
        pack=pack,
        candidate=candidate,
        target_state=target.state.value,
        observed_head=target.head_submission_digest,
    )
    reviewer_id, warning_accepted, answers, rationale, decision = _prompt_review(
        pack=pack,
        known_reviewer_ids=frozenset(item.reviewer_id for item in snapshot.submissions),
    )
    result = append_public_review_submission(
        review_directory=arguments.reviews,
        pack=pack,
        lineage_registry_key=lineage_key,
        reviewer_id=reviewer_id,
        review_rationale=rationale,
        checklist_answers=answers,
        decision=decision,
        supersedes_submission_digest=target.head_submission_digest,
        publication_warning_accepted=warning_accepted,
    )
    _write_stdout("Review submission recorded.\n")
    _write_stdout(f"submission digest: {result.submission.submission_digest}\n")
    _write_stdout(f"envelope digest: {result.envelope.envelope_digest}\n")
    return _SUCCESS


def _dispatch_build_envelope(arguments: argparse.Namespace) -> int:
    from saliencegate.benchmarks.state_decay_v2.review_io import (
        persist_public_review_envelope,
    )

    pack = _load_pack(arguments.pack)
    if arguments.lineage_key not in {draft.lineage_registry_key for draft in pack.drafts}:
        raise _ReviewInputError
    result = persist_public_review_envelope(
        review_directory=arguments.reviews,
        pack=pack,
        lineage_registry_key=arguments.lineage_key,
    )
    state = next(
        item.state
        for item in result.progress.candidates
        if item.lineage_registry_key == result.submission.lineage_registry_key
    )
    report = {
        "schema_version": _ENVELOPE_REPORT_SCHEMA,
        "status": "ok",
        "lineage_registry_key": result.submission.lineage_registry_key,
        "head_submission_digest": result.submission.submission_digest,
        "envelope_digest": result.envelope.envelope_digest,
        "state": state.value,
    }
    if arguments.json:
        _write_json(report)
    else:
        _write_stdout(
            "Public review envelope built.\n"
            f"lineage key: {report['lineage_registry_key']}\n"
            f"head submission digest: {report['head_submission_digest']}\n"
            f"envelope digest: {report['envelope_digest']}\n"
            f"state: {report['state']}\n"
        )
    return _SUCCESS


def _dispatch(arguments: argparse.Namespace) -> int:
    from saliencegate.artifacts.tree import ArtifactExportError
    from saliencegate.benchmarks.state_decay_v2.review import PublicReviewError
    from saliencegate.benchmarks.state_decay_v2.review_io import PublicReviewIOError
    from saliencegate.benchmarks.state_decay_v2.review_pack import PublicReviewPackError

    try:
        if arguments.command == "build-pack":
            return _dispatch_build_pack(arguments)
        if arguments.command == "review":
            return _dispatch_review(arguments)
        if arguments.command == "status":
            return _dispatch_status(arguments)
        if arguments.command == "build-envelope":
            return _dispatch_build_envelope(arguments)
        raise _UsageError
    except PublicReviewIOError:
        raise _ReviewStateError from None
    except (PublicReviewPackError, PublicReviewError):
        raise _ReviewStateError from None
    except ArtifactExportError:
        raise _ReviewInputError from None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        try:
            arguments = _parser(_invocation_program(argv)).parse_args(argv)
        except _UsageError:
            _write_stderr("error: invalid command line\n")
            return _INVALID_INPUT
        except SystemExit as error:
            return int(error.code or 0)
        _validate_path_arguments(arguments)
        return _dispatch(arguments)
    except (EOFError, _ReviewInputError):
        _write_stderr("error: public review input is invalid\n")
        return _INVALID_INPUT
    except _ReviewStateError:
        _write_stderr("error: public review state is corrupted\n")
        return _CORRUPTED_STATE
    except BrokenPipeError:
        return _SUCCESS
    except KeyboardInterrupt:
        return _INTERRUPTED
    except Exception:
        _write_stderr("error: internal error\n")
        return _INTERNAL_ERROR


def entrypoint() -> Never:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()


__all__ = ["entrypoint", "main"]
