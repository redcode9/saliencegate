from __future__ import annotations

import argparse
import builtins
import contextlib
import inspect
import io
import json
import os
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Never, cast

import pytest
from pydantic import BaseModel

import saliencegate.benchmarks.state_decay_v2.review_cli as review_cli
import saliencegate.benchmarks.state_decay_v2.review_io as review_io
from saliencegate.benchmarks.state_decay_v2 import config as generation_config
from saliencegate.benchmarks.state_decay_v2.review import (
    PublicReviewGateReport,
    PublicReviewProgressState,
)
from saliencegate.benchmarks.state_decay_v2.review_io import (
    PublicReviewIOError,
    PublicReviewIOErrorCode,
    load_public_review_progress,
)
from saliencegate.benchmarks.state_decay_v2.review_pack import (
    ValidatedPublicReviewPack,
    load_public_review_pack,
)

ROOT = Path(__file__).resolve().parents[3]

_REGISTRY_DIGEST: Final = "1b396e5fbee6e7a95ffb2739a47ddd97807cb76c0398ffd20363a26b9f076372"
_CHECKLIST_DIGEST: Final = "ed2c09d956d3a7241ed590a6dc825c912597e274554096e3a2f61bf23113a051"
_PACK_MANIFEST_DIGEST: Final = "8fa9e264270ea58730deb9247dcf9b2183e0414b510ab0bb4c6d9b0a1466f44c"


@dataclass(frozen=True, slots=True)
class _Invocation:
    code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class _CLIPack:
    path: Path
    report: dict[str, object]
    value: ValidatedPublicReviewPack


class _AmbientEnvironmentTrap:
    _LOCALE_KEYS = frozenset({"COLUMNS", "LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LINES"})

    def __init__(self, source: dict[str, str]) -> None:
        self._source = source

    def _fail(self) -> Never:
        raise AssertionError("ambient environment was accessed")

    def __getitem__(self, key: str) -> str:
        if key not in self._LOCALE_KEYS:
            self._fail()
        return self._source[key]

    def __iter__(self) -> Never:
        self._fail()

    def __len__(self) -> Never:
        self._fail()

    def __contains__(self, key: object) -> bool:
        if key not in self._LOCALE_KEYS:
            self._fail()
        return key in self._source

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in self._LOCALE_KEYS:
            self._fail()
        return self._source.get(key, default)

    def items(self) -> Never:
        self._fail()

    def keys(self) -> Never:
        self._fail()

    def values(self) -> Never:
        self._fail()

    def copy(self) -> Never:
        self._fail()


def _invoke(
    arguments: list[str],
    *,
    stdin: str = "",
    input_stream: io.TextIOBase | None = None,
) -> _Invocation:
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(stdin) if input_stream is None else input_stream
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = review_cli.main(arguments)
    finally:
        sys.stdin = previous_stdin
    return _Invocation(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def _lines(*values: str) -> str:
    return "\n".join(values) + "\n"


def _accepted_input(
    *,
    reviewer_id: str = "cli-reviewer",
    rationale: str = "Explicit accepted review rationale.",
    include_warning: bool = True,
) -> str:
    values = [reviewer_id]
    if include_warning:
        values.append("I ACCEPT PUBLICATION")
    values.extend((*("passed" for _ in range(7)), rationale, "accepted"))
    return _lines(*values)


@pytest.fixture(scope="module")
def cli_pack(tmp_path_factory: pytest.TempPathFactory) -> _CLIPack:
    path = tmp_path_factory.mktemp("public-review-cli-pack") / "pack"
    result = _invoke(["build-pack", "--output", str(path), "--json"])

    assert result.code == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    return _CLIPack(path=path, report=report, value=load_public_review_pack(pack=path).value)


def test_review_cli_surface_is_narrow_explicit_and_additive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert review_cli.__all__ == ["entrypoint", "main"]
    signature = inspect.signature(review_cli.main)
    assert tuple(signature.parameters) == ("argv",)

    missing = _invoke([])
    assert missing == _Invocation(
        code=2,
        stdout="",
        stderr="error: invalid command line\n",
    )

    help_result = _invoke(["--help"])
    assert help_result.code == 0
    assert help_result.stderr == ""
    assert "{build-pack,review,status,build-envelope}" in help_result.stdout
    for forbidden in (
        "accept-all",
        "non-interactive",
        "reviewer-id",
        "rationale",
        "provider",
        "model",
        "endpoint",
        "replace",
        "finalize",
        "generate",
        "export",
    ):
        assert forbidden not in help_result.stdout

    rejected_flag = _invoke(
        [
            "review",
            "--pack",
            "pack",
            "--reviews",
            "reviews",
            "--non-interactive",
        ]
    )
    assert rejected_flag == missing

    malformed_path = _invoke(["status", "--pack", "", "--reviews", "reviews", "--json"])
    assert malformed_path == _Invocation(
        code=2,
        stdout="",
        stderr="error: public review input is invalid\n",
    )

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "main", lambda arguments=None: 130)
        with pytest.raises(SystemExit) as raised:
            review_cli.entrypoint()
    assert raised.value.code == 130

    with monkeypatch.context() as patch:
        patch.setattr(
            review_cli,
            "_dispatch",
            lambda arguments: (_ for _ in ()).throw(RuntimeError("private-internal-value")),
        )
        internal = _invoke(["status", "--pack", "pack", "--reviews", "reviews", "--json"])
    assert internal == _Invocation(
        code=70,
        stdout="",
        stderr="error: internal error\n",
    )


def test_installed_entrypoint_reports_the_public_program_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["/isolated/bin/saliencegate-review", "--help"])

    with pytest.raises(SystemExit) as raised:
        review_cli.entrypoint()

    captured = capsys.readouterr()
    assert raised.value.code == 0
    assert captured.err == ""
    assert captured.out.startswith("usage: saliencegate-review ")
    assert "python -m saliencegate" not in captured.out


def test_review_parser_actions_are_exact_and_have_no_hidden_inputs() -> None:
    parser = review_cli._parser("saliencegate-review")
    assert len(parser._actions) == 2
    help_action, command_action = parser._actions
    assert (
        tuple(help_action.option_strings),
        help_action.dest,
        help_action.required,
        help_action.nargs,
    ) == (("-h", "--help"), "help", False, 0)
    assert isinstance(command_action, argparse._SubParsersAction)
    assert command_action.dest == "command"
    assert command_action.required is True
    assert command_action.option_strings == []

    expected = {
        "build-pack": (
            (("-h", "--help"), "help", False, 0),
            (("--output",), "output", True, None),
            (("--json",), "json", False, 0),
        ),
        "review": (
            (("-h", "--help"), "help", False, 0),
            (("--pack",), "pack", True, None),
            (("--reviews",), "reviews", True, None),
        ),
        "status": (
            (("-h", "--help"), "help", False, 0),
            (("--pack",), "pack", True, None),
            (("--reviews",), "reviews", True, None),
            (("--json",), "json", False, 0),
        ),
        "build-envelope": (
            (("-h", "--help"), "help", False, 0),
            (("--pack",), "pack", True, None),
            (("--reviews",), "reviews", True, None),
            (("--lineage-key",), "lineage_key", True, None),
            (("--json",), "json", False, 0),
        ),
    }
    assert tuple(command_action.choices) == tuple(expected)
    for command, child in command_action.choices.items():
        observed = tuple(
            (tuple(action.option_strings), action.dest, action.required, action.nargs)
            for action in child._actions
        )
        assert observed == expected[command]


def test_build_pack_json_is_stable_value_free_and_self_bound(cli_pack: _CLIPack) -> None:
    assert cli_pack.report == {
        "schema_version": "state-decay-v2-public-review-cli-build-pack-report/v1",
        "status": "ok",
        "candidate_count": 180,
        "candidate_registry_digest": _REGISTRY_DIGEST,
        "checklist_digest": _CHECKLIST_DIGEST,
        "pack_manifest_digest": _PACK_MANIFEST_DIGEST,
    }
    assert cli_pack.value.registry.registry_digest == _REGISTRY_DIGEST
    assert cli_pack.value.checklist.checklist_digest == _CHECKLIST_DIGEST
    assert cli_pack.value.manifest.manifest_digest == _PACK_MANIFEST_DIGEST
    assert set(cli_pack.report).isdisjoint({"output", "pack", "path", "reviewer_id"})


def test_status_is_read_only_and_corrupt_or_missing_state_is_exit_five(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviews = tmp_path / "missing-reviews"

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        result = _invoke(
            ["status", "--pack", str(cli_pack.path), "--reviews", str(reviews), "--json"]
        )

    assert result == _Invocation(
        code=5,
        stdout="",
        stderr="error: public review state is corrupted\n",
    )
    assert not reviews.exists()

    missing_pack = _invoke(
        [
            "status",
            "--pack",
            str(tmp_path / "missing-pack"),
            "--reviews",
            str(reviews),
            "--json",
        ]
    )
    assert missing_pack == result
    assert str(tmp_path) not in result.stderr + missing_pack.stderr


def test_eof_leaves_no_submission_and_resume_status_is_exact(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
    reviews = tmp_path / "reviews"

    interrupted = _invoke(["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)])

    assert interrupted.code == 2
    assert interrupted.stderr == "error: public review input is invalid\n"
    assert "Reviewer ID (required): " in interrupted.stdout
    assert tuple(path.name for path in reviews.iterdir()) == ("review.lock",)

    status = _invoke(["status", "--pack", str(cli_pack.path), "--reviews", str(reviews), "--json"])
    assert status.code == 0
    assert status.stderr == ""
    assert json.loads(status.stdout) == {
        "schema_version": "state-decay-v2-public-review-cli-status-report/v1",
        "status": "ok",
        "ambiguous_count": 0,
        "stale_comparison_count": 0,
        "rejected_count": 0,
        "missing_count": 180,
        "accepted_count": 0,
        "progress_complete": False,
        "candidate_registry_digest": _REGISTRY_DIGEST,
        "checklist_digest": _CHECKLIST_DIGEST,
    }


def test_interactive_acceptance_requires_every_field_and_builds_bound_envelope(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
    reviews = tmp_path / "reviews"
    private_rationale = "Private caf\u00e9 acceptance rationale."
    stdin = _lines(
        "",
        "cli-reviewer",
        "NO",
        "I ACCEPT PUBLICATION",
        *("passed" for _ in range(7)),
        private_rationale,
        "rejected",
        "accepted",
    )

    result = _invoke(
        ["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)],
        stdin=stdin,
    )

    assert result.code == 0
    assert result.stderr == ""
    assert result.stdout.count("Reviewer ID (required): ") == 2
    assert result.stdout.count("Type exactly I ACCEPT PUBLICATION: ") == 2
    assert result.stdout.count("Decision [accepted/rejected]: ") == 2
    assert result.stdout.count("Invalid input.\n") == 3
    assert "Target state: missing\n" in result.stdout
    assert "Observed head: none\n" in result.stdout
    assert "Review submission recorded.\n" in result.stdout
    assert private_rationale not in result.stdout + result.stderr
    assert len(tuple(reviews.glob("review--*.json"))) == 1
    assert len(tuple(reviews.glob("envelope--*.json"))) == 1

    snapshot = load_public_review_progress(
        review_directory=reviews,
        pack=cli_pack.value,
        create=False,
    )
    assert snapshot.progress.accepted_count == 1
    assert snapshot.progress.missing_count == 179
    head = snapshot.submissions[-1]

    envelope = _invoke(
        [
            "build-envelope",
            "--pack",
            str(cli_pack.path),
            "--reviews",
            str(reviews),
            "--lineage-key",
            head.lineage_registry_key,
            "--json",
        ]
    )
    assert envelope.code == 0
    assert envelope.stderr == ""
    envelope_report = json.loads(envelope.stdout)
    assert envelope_report == {
        "schema_version": "state-decay-v2-public-review-cli-envelope-report/v1",
        "status": "ok",
        "lineage_registry_key": head.lineage_registry_key,
        "head_submission_digest": head.submission_digest,
        "envelope_digest": snapshot.envelopes[-1].envelope_digest,
        "state": "accepted",
    }
    assert private_rationale not in envelope.stdout

    unknown = _invoke(
        [
            "build-envelope",
            "--pack",
            str(cli_pack.path),
            "--reviews",
            str(reviews),
            "--lineage-key",
            "pub-fr-99",
            "--json",
        ]
    )
    assert unknown == _Invocation(
        code=2,
        stdout="",
        stderr="error: public review input is invalid\n",
    )


def test_rejected_submission_is_success_and_correction_never_reuses_history(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
    reviews = tmp_path / "reviews"
    historical_rationale = "HISTORICAL-REJECTION-RATIONALE-DO-NOT-RENDER"
    rejected_input = _lines(
        "cli-reviewer",
        "I ACCEPT PUBLICATION",
        "failed",
        *("passed" for _ in range(6)),
        historical_rationale,
        "rejected",
    )

    rejected = _invoke(
        ["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)],
        stdin=rejected_input,
    )
    assert rejected.code == 0
    assert rejected.stderr == ""
    rejected_snapshot = load_public_review_progress(
        review_directory=reviews,
        pack=cli_pack.value,
        create=False,
    )
    assert rejected_snapshot.progress.rejected_count == 1
    rejected_head = rejected_snapshot.submissions[-1].submission_digest

    corrected = _invoke(
        ["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)],
        stdin=_accepted_input(
            include_warning=False,
            rationale="Explicit correction rationale.",
        ),
    )
    assert corrected.code == 0
    assert corrected.stderr == ""
    assert "Target state: rejected\n" in corrected.stdout
    assert f"Observed head: {rejected_head}\n" in corrected.stdout
    assert "Publication warning:" not in corrected.stdout
    assert historical_rationale not in corrected.stdout + corrected.stderr
    assert "Explicit correction rationale." not in corrected.stdout + corrected.stderr

    corrected_snapshot = load_public_review_progress(
        review_directory=reviews,
        pack=cli_pack.value,
        create=False,
    )
    assert corrected_snapshot.progress.rejected_count == 0
    assert corrected_snapshot.progress.accepted_count == 1
    assert len(corrected_snapshot.submissions) == 2
    assert corrected_snapshot.submissions[-1].supersedes_submission_digest == rejected_head


def test_stale_comparison_has_priority_and_reenters_with_the_observed_head(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_candidate, rejected_candidate, stale_candidate = cli_pack.value.registry.candidates[:3]
    stale_head = "a" * 64
    candidates = (
        SimpleNamespace(
            state=PublicReviewProgressState.MISSING,
            lineage_registry_key=missing_candidate.lineage_registry_key,
            head_submission_digest=None,
        ),
        SimpleNamespace(
            state=PublicReviewProgressState.REJECTED,
            lineage_registry_key=rejected_candidate.lineage_registry_key,
            head_submission_digest="b" * 64,
        ),
        SimpleNamespace(
            state=PublicReviewProgressState.STALE_COMPARISON,
            lineage_registry_key=stale_candidate.lineage_registry_key,
            head_submission_digest=stale_head,
        ),
    )
    progress = SimpleNamespace(ambiguous_count=0, candidates=candidates)
    snapshot = SimpleNamespace(progress=progress, submissions=())
    captured: dict[str, object] = {}

    def render(**values: object) -> None:
        captured["render"] = values

    def append(**values: object) -> SimpleNamespace:
        captured["append"] = values
        return SimpleNamespace(
            submission=SimpleNamespace(submission_digest="c" * 64),
            envelope=SimpleNamespace(envelope_digest="d" * 64),
        )

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(review_cli, "_render_review_materials", render)
        patch.setattr(review_io, "load_public_review_progress", lambda **values: snapshot)
        patch.setattr(review_io, "append_public_review_submission", append)
        result = _invoke(
            [
                "review",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(tmp_path / "stale-comparison-reviews"),
            ],
            stdin=_accepted_input(),
        )

    assert result.code == 0
    assert result.stderr == ""
    rendered = captured["render"]
    appended = captured["append"]
    assert isinstance(rendered, dict)
    assert isinstance(appended, dict)
    assert rendered["candidate"] == stale_candidate
    assert rendered["target_state"] == "stale-comparison"
    assert rendered["observed_head"] == stale_head
    assert appended["lineage_registry_key"] == stale_candidate.lineage_registry_key
    assert appended["supersedes_submission_digest"] == stale_head


def test_stale_head_interrupt_and_invalid_utf8_are_value_free(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(
            review_io,
            "load_public_review_progress",
            lambda **values: SimpleNamespace(
                progress=SimpleNamespace(ambiguous_count=1),
                submissions=(),
            ),
        )
        patch.setattr(
            review_cli,
            "_read_prompt_line",
            lambda prompt: (_ for _ in ()).throw(AssertionError("prompted ambiguous state")),
        )
        ambiguous = _invoke(
            [
                "review",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(tmp_path / "ambiguous-reviews"),
            ]
        )
    assert ambiguous == _Invocation(
        code=5,
        stdout="",
        stderr="error: public review state is corrupted\n",
    )

    reviews = tmp_path / "stale-reviews"

    def stale_append(**values: object) -> None:
        del values
        raise PublicReviewIOError(PublicReviewIOErrorCode.STALE_HEAD)

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(review_cli, "_render_review_materials", lambda **values: None)
        patch.setattr(review_io, "append_public_review_submission", stale_append)
        stale = _invoke(
            ["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)],
            stdin=_accepted_input(rationale="STALE-RACE-PRIVATE-RATIONALE"),
        )
    assert stale.code == 5
    assert stale.stderr == "error: public review state is corrupted\n"
    assert "STALE-RACE-PRIVATE-RATIONALE" not in stale.stdout + stale.stderr
    assert not tuple(reviews.glob("review--*.json"))

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(review_cli, "_render_review_materials", lambda **values: None)
        patch.setattr(
            review_cli,
            "_read_prompt_line",
            lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt),
        )
        interrupted = _invoke(
            [
                "review",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(tmp_path / "interrupt-reviews"),
            ]
        )
    assert interrupted == _Invocation(code=130, stdout="", stderr="")

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(review_cli, "_render_review_materials", lambda **values: None)
        invalid_utf8 = _invoke(
            [
                "review",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(tmp_path / "utf8-reviews"),
            ],
            input_stream=io.TextIOWrapper(
                io.BytesIO(b"\xff\n"),
                encoding="utf-8",
                errors="strict",
            ),
        )
    assert invalid_utf8 == _Invocation(
        code=2,
        stdout="Reviewer ID (required): ",
        stderr="error: public review input is invalid\n",
    )

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: cli_pack.value)
        patch.setattr(review_cli, "_render_review_materials", lambda **values: None)
        oversized = _invoke(
            [
                "review",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(tmp_path / "oversized-reviews"),
            ],
            stdin="x" * 4_097 + "\n",
        )
    assert oversized.code == 2
    assert oversized.stdout == "Reviewer ID (required): "
    assert oversized.stderr == "error: public review input is invalid\n"


def test_every_command_avoids_network_provider_environment_and_direct_authority(
    cli_pack: _CLIPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_import_prefixes = (
        "anthropic",
        "dotenv",
        "httpcore",
        "httpx",
        "keyring",
        "openai",
        "openai_harmony",
        "pydantic_settings",
        "requests",
        "saliencegate.benchmarks.state_decay_v2.generation_authority",
        "saliencegate.benchmarks.state_decay_v2.generator",
        "saliencegate.model_runtime",
    )
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden_import_prefixes
        ):
            raise AssertionError("forbidden command import")
        return original_import(name, globals, locals, fromlist, level)

    def forbidden_call(*values: object, **named_values: object) -> Never:
        del values, named_values
        raise AssertionError("forbidden command authority")

    generated_pack = tmp_path / "offline-pack"
    reviews = tmp_path / "offline-reviews"
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded_import)
        patch.setattr(socket, "socket", forbidden_call)
        patch.setattr(socket, "create_connection", forbidden_call)
        patch.setattr(urllib.request, "urlopen", forbidden_call)
        patch.setattr(os, "environ", _AmbientEnvironmentTrap(dict(os.environ)))
        patch.setattr(os, "getenv", forbidden_call)
        patch.setattr(os, "putenv", forbidden_call)
        patch.setattr(os, "unsetenv", forbidden_call)
        patch.setattr(generation_config, "allocate_balanced_outcomes", forbidden_call)
        patch.setattr(generation_config, "validate_balanced_allocations", forbidden_call)

        built = _invoke(["build-pack", "--output", str(generated_pack), "--json"])
        reviewed = _invoke(
            ["review", "--pack", str(cli_pack.path), "--reviews", str(reviews)],
            stdin=_lines(
                "offline-reviewer",
                "I ACCEPT PUBLICATION",
                "failed",
                *("passed" for _ in range(6)),
                "Explicit offline rejection rationale.",
                "rejected",
            ),
        )
        status = _invoke(
            ["status", "--pack", str(cli_pack.path), "--reviews", str(reviews), "--json"]
        )
        envelope = _invoke(
            [
                "build-envelope",
                "--pack",
                str(cli_pack.path),
                "--reviews",
                str(reviews),
                "--lineage-key",
                cli_pack.value.drafts[0].lineage_registry_key,
                "--json",
            ]
        )

    assert (built.code, reviewed.code, status.code, envelope.code) == (0, 0, 0, 0)
    assert built.stderr == reviewed.stderr == status.stderr == envelope.stderr == ""
    assert json.loads(built.stdout)["pack_manifest_digest"] == _PACK_MANIFEST_DIGEST
    assert json.loads(status.stdout)["rejected_count"] == 1
    assert json.loads(envelope.stdout)["state"] == "rejected"


def test_terminal_rendering_is_ascii_escaped_and_module_invocation_is_isolated() -> None:
    class Probe(BaseModel):
        text: str

    raw = "prefix\x1b[31m caf\u00e9 \u202esuffix"
    rendered = review_cli._escaped_model_json(Probe(text=raw))
    assert rendered == '{"text":"prefix\\u001b[31m caf\\u00e9 \\u202esuffix"}'
    assert raw not in rendered
    assert json.loads(rendered) == {"text": raw}

    probe = r"""
import contextlib
import io
import sys

import saliencegate.benchmarks.state_decay_v2.review_cli as review_cli

forbidden = (
    "httpx",
    "openai",
    "saliencegate.model_runtime",
    "saliencegate.benchmarks.state_decay_v2.generator",
    "saliencegate.benchmarks.state_decay_v2.generation_authority",
)
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    code = review_cli.main([])
assert code == 2
assert stdout.getvalue() == ""
assert stderr.getvalue() == "error: invalid command line\n"
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""
    imported = subprocess.run(
        (sys.executable, "-I", "-c", probe),
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout == ""
    assert imported.stderr == ""

    invoked = subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "saliencegate.benchmarks.state_decay_v2.review_cli",
            "--help",
        ),
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert invoked.returncode == 0
    assert "{build-pack,review,status,build-envelope}" in invoked.stdout
    assert invoked.stderr == ""
    assert "RuntimeWarning" not in invoked.stdout + invoked.stderr


def test_cli_defensive_rendering_and_input_edges_are_value_free(
    cli_pack: _CLIPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(review_cli._ReviewInputError):
        review_cli._validate_path_arguments(argparse.Namespace(pack="\ud800"))
    with pytest.raises(TypeError, match="requires a model"):
        review_cli._escaped_model_json(object())
    with pytest.raises(review_cli._ReviewStateError):
        review_cli._candidate_for_key(cli_pack.value, "missing-lineage")

    build_report = {
        "candidate_count": 180,
        "candidate_registry_digest": "1" * 64,
        "checklist_digest": "2" * 64,
        "pack_manifest_digest": "3" * 64,
    }
    assert review_cli._render_build_pack_human(build_report).startswith(
        "Public review pack built.\n"
    )
    progress = SimpleNamespace(
        ambiguous_count=0,
        stale_comparison_count=0,
        rejected_count=0,
        missing_count=180,
        accepted_count=0,
        progress_complete=False,
        candidate_registry_digest="1" * 64,
        checklist_digest="2" * 64,
    )
    typed_progress = cast(PublicReviewGateReport, progress)
    assert review_cli._render_status_human(typed_progress).startswith("Public review progress\n")
    assert (
        review_cli._target_progress(
            cast(
                PublicReviewGateReport,
                SimpleNamespace(ambiguous_count=0, candidates=()),
            )
        )
        is None
    )

    previous_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("value\r\n")
        with contextlib.redirect_stdout(io.StringIO()):
            assert review_cli._read_prompt_line("prompt") == "value"
        sys.stdin = io.StringIO("\ud800\n")
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(review_cli._ReviewInputError):
            review_cli._read_prompt_line("prompt")
    finally:
        sys.stdin = previous_stdin

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_load_pack", lambda path: object())
        patch.setattr(
            review_io,
            "load_public_review_progress",
            lambda **kwargs: SimpleNamespace(
                progress=SimpleNamespace(ambiguous_count=0, candidates=()),
                submissions=(),
            ),
        )
        complete = _invoke(["review", "--pack", "pack", "--reviews", "reviews"])
    assert complete == _Invocation(code=0, stdout="Review complete.\n", stderr="")


def test_cli_human_envelope_and_dispatch_error_edges(
    cli_pack: _CLIPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saliencegate.artifacts.tree import ArtifactExportError

    lineage_key = cli_pack.value.drafts[0].lineage_registry_key
    fake_result = SimpleNamespace(
        submission=SimpleNamespace(
            lineage_registry_key=lineage_key,
            submission_digest="4" * 64,
        ),
        envelope=SimpleNamespace(envelope_digest="5" * 64),
        progress=SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    lineage_registry_key=lineage_key,
                    state=PublicReviewProgressState.ACCEPTED,
                ),
            )
        ),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            review_cli,
            "_load_pack",
            lambda path: SimpleNamespace(
                drafts=(SimpleNamespace(lineage_registry_key=lineage_key),)
            ),
        )
        patch.setattr(
            review_io,
            "persist_public_review_envelope",
            lambda **kwargs: fake_result,
        )
        result = _invoke(
            [
                "build-envelope",
                "--pack",
                "pack",
                "--reviews",
                "reviews",
                "--lineage-key",
                lineage_key,
            ]
        )
    assert result.code == 0
    assert result.stderr == ""
    assert result.stdout.startswith("Public review envelope built.\n")

    with pytest.raises(review_cli._UsageError):
        review_cli._dispatch(argparse.Namespace(command="unknown"))

    for state_error_code in (
        PublicReviewIOErrorCode.MALFORMED_INPUT,
        PublicReviewIOErrorCode.UNSAFE_TEXT,
        PublicReviewIOErrorCode.INCONSISTENT_REVIEW,
        PublicReviewIOErrorCode.WARNING_REQUIRED,
    ):

        def raise_state_error(
            arguments: object,
            error_code: PublicReviewIOErrorCode = state_error_code,
        ) -> Never:
            del arguments
            raise PublicReviewIOError(error_code)

        with monkeypatch.context() as patch:
            patch.setattr(review_cli, "_dispatch_status", raise_state_error)
            with pytest.raises(review_cli._ReviewStateError):
                review_cli._dispatch(argparse.Namespace(command="status"))

    def raise_artifact_error(arguments: object) -> Never:
        raise ArtifactExportError("failure")

    with monkeypatch.context() as patch:
        patch.setattr(review_cli, "_dispatch_build_pack", raise_artifact_error)
        with pytest.raises(review_cli._ReviewInputError):
            review_cli._dispatch(argparse.Namespace(command="build-pack"))

    with monkeypatch.context() as patch:
        patch.setattr(
            review_cli,
            "_dispatch",
            lambda arguments: (_ for _ in ()).throw(BrokenPipeError),
        )
        broken_pipe = _invoke(["status", "--pack", "pack", "--reviews", "reviews", "--json"])
    assert broken_pipe == _Invocation(code=0, stdout="", stderr="")
