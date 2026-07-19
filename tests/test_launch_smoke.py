from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from scripts import smoke_launch_contracts

ROOT = Path(__file__).resolve().parents[1]
_SHADOW_EXAMPLE_OUTPUT = (
    "SalienceGate Shadow Mode example\n"
    "evaluated events: 3\n"
    "supported detectors: 4 of 9\n"
    "failed-result disposition: flagged\n"
    "detected signals: tool_error\n"
    "evidence: descriptive observational; no decision authority\n"
    "model calls: 0\n"
)

_SUCCESS = "launch-contracts-ok\n"
_FAILURE = "launch-contracts-failed\n"
_COMMANDS = ("build-pack", "review", "status", "build-envelope")
_COMMAND_OPTIONS = {
    "build-pack": ("output", "json"),
    "review": ("pack", "reviews"),
    "status": ("pack", "reviews", "json"),
    "build-envelope": ("pack", "reviews", "lineage-key", "json"),
}
_FORBIDDEN = (
    "finalize",
    "generate",
    "export",
    "accept-all",
    "non-interactive",
    "provider",
    "model",
    "endpoint",
    "replace",
)


def _snapshot_tree(path: Path) -> tuple[tuple[str, bytes | None], ...]:
    return tuple(
        (item.relative_to(path).as_posix(), item.read_bytes() if item.is_file() else None)
        for item in sorted(path.rglob("*"))
    )


def test_shadow_example_is_deterministic_socket_free_and_filesystem_free(tmp_path: Path) -> None:
    command = (
        sys.executable,
        str(ROOT / "scripts/run_without_sockets.py"),
        str(ROOT / "examples/shadow_asyncio.py"),
    )
    completed: list[subprocess.CompletedProcess[str]] = []
    for ordinal in (1, 2):
        sandbox = tmp_path / f"sandbox-{ordinal}"
        home = sandbox / "home"
        work = sandbox / "work"
        config = sandbox / "config"
        for directory in (sandbox, home, work, config):
            directory.mkdir(mode=0o700, exist_ok=True)
        before = _snapshot_tree(sandbox)
        result = subprocess.run(
            command,
            cwd=work,
            env={
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(config),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": str(ordinal),
                "PYTHONUTF8": "1",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert _snapshot_tree(sandbox) == before
        completed.append(result)

    first, second = completed
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == _SHADOW_EXAMPLE_OUTPUT
    for sensitive in (
        "example-tool",
        "/example",
        "run-start",
        "action-1",
        "tool-result-1",
        "InstallationKey",
    ):
        assert sensitive not in first.stdout
    assert re.search(r"[0-9a-f]{32,}", first.stdout) is None


def _valid_demo() -> dict[str, object]:
    return {
        "schema_version": "cli-demo-report/v1",
        "status": "ok",
        "suite_id": "state-decay-smoke",
        "evidence_level": "synthetic_diagnostic",
        "diagnostic": True,
        "synthetic": True,
        "confirmatory": False,
        "external_claims_supported": False,
        "external_claims_assessment": "insufficient",
        "scenario_count": 32,
        "family_count": 8,
        "intervene_count": 16,
        "silence_count": 16,
        "oracle_passed": 32,
        "oracle_failed": 0,
        "result_digest": "13704b753086925db1abfd7467f3e202edb202d60f620d150b1cdb6099c57d0f",
    }


def _valid_help() -> str:
    choices = ",".join(_COMMANDS)
    root_help = (
        "usage: saliencegate-review [-h] "
        f"{{{choices}}} ...\n\n"
        f"positional arguments:\n  {{{choices}}}\n\n"
        "options:\n  -h, --help  show this help message and exit\n"
    )
    command_help = ""
    for command in _COMMANDS:
        option_lines = "".join(f"  --{option} VALUE\n" for option in _COMMAND_OPTIONS[command])
        command_help += (
            f"usage: saliencegate-review {command} [-h]\n\n"
            "options:\n  -h, --help  show this help message and exit\n"
            f"{option_lines}"
        )
    return root_help + command_help


def _replace_in_command_help(
    help_text: str,
    *,
    command: str,
    old: str,
    new: str,
) -> str:
    usage = f"usage: saliencegate-review {command} "
    start = help_text.index(usage)
    later_starts = tuple(
        position
        for candidate in _COMMANDS
        if candidate != command
        and (position := help_text.find(f"usage: saliencegate-review {candidate} ", start + 1))
        != -1
    )
    end = min(later_starts, default=len(help_text))
    block = help_text[start:end]
    if block.count(old) != 1:
        raise ValueError
    return help_text[:start] + block.replace(old, new, 1) + help_text[end:]


def _write_inputs(
    tmp_path: Path,
    *,
    demo: dict[str, object] | str | bytes | None = None,
    help_text: str | bytes | None = None,
) -> tuple[Path, Path]:
    demo_path = tmp_path / "demo.json"
    help_path = tmp_path / "review-help.txt"
    if isinstance(demo, (str, bytes)):
        demo_payload: str | bytes = demo
    else:
        demo_payload = (
            json.dumps(
                _valid_demo() if demo is None else demo,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
    help_payload = _valid_help() if help_text is None else help_text
    if isinstance(demo_payload, str):
        demo_path.write_text(demo_payload, encoding="utf-8")
    else:
        demo_path.write_bytes(demo_payload)
    if isinstance(help_payload, str):
        help_path.write_text(help_payload, encoding="utf-8")
    else:
        help_path.write_bytes(help_payload)
    return demo_path, help_path


def _invoke(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = smoke_launch_contracts.main(arguments)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_verifier_accepts_only_the_exact_launch_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo, review_help = _write_inputs(tmp_path)

    assert _invoke([str(demo), str(review_help)], capsys) == (0, _SUCCESS, "")


def test_verifier_rejects_internal_module_usage_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    internal_help = _valid_help().replace(
        "usage: saliencegate-review",
        "usage: python -m saliencegate.benchmarks.state_decay_v2.review_cli",
    )
    demo, review_help = _write_inputs(tmp_path, help_text=internal_help)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("arguments", ([], ["one"], ["one", "two", "three"]))
def test_verifier_rejects_unexpected_arguments_without_values(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _invoke(arguments, capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize(
    "payload",
    (
        "not-json",
        "[]",
        '{"schema_version":"cli-demo-report/v1","schema_version":"forged"}',
        b"\xff",
    ),
)
def test_verifier_rejects_malformed_or_ambiguous_demo_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str | bytes,
) -> None:
    demo, review_help = _write_inputs(tmp_path, demo=payload)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("field", tuple(_valid_demo()))
def test_verifier_rejects_each_missing_demo_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    payload = _valid_demo()
    del payload[field]
    demo, review_help = _write_inputs(tmp_path, demo=payload)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


def test_verifier_rejects_extra_demo_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _valid_demo()
    payload["unexpected"] = "value"
    demo, review_help = _write_inputs(tmp_path, demo=payload)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize(
    "payload",
    (
        json.dumps(_valid_demo(), sort_keys=True, separators=(",", ":")),
        " " + json.dumps(_valid_demo(), sort_keys=True, separators=(",", ":")) + "\n",
        json.dumps(_valid_demo(), sort_keys=True, indent=2) + "\n",
        json.dumps(_valid_demo(), sort_keys=False, separators=(",", ":")) + "\n",
    ),
    ids=("missing-newline", "leading-space", "pretty", "unsorted"),
)
def test_verifier_rejects_noncanonical_demo_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str,
) -> None:
    demo, review_help = _write_inputs(tmp_path, demo=payload)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("schema_version", "cli-demo-report/v2"),
        ("status", "ready"),
        ("suite_id", "other"),
        ("evidence_level", "confirmatory"),
        ("diagnostic", 1),
        ("synthetic", False),
        ("confirmatory", True),
        ("external_claims_supported", True),
        ("external_claims_assessment", "sufficient"),
        ("scenario_count", 31),
        ("family_count", 7),
        ("intervene_count", 15),
        ("silence_count", 17),
        ("oracle_passed", 31),
        ("oracle_failed", 1),
        ("result_digest", "0" * 64),
        ("result_digest", "A" * 64),
        ("result_digest", "0" * 63),
    ),
)
def test_verifier_rejects_wrong_demo_contract_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    wrong: object,
) -> None:
    payload = _valid_demo()
    payload[field] = wrong
    demo, review_help = _write_inputs(tmp_path, demo=payload)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("missing", _COMMANDS)
def test_verifier_rejects_each_missing_review_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    demo, review_help = _write_inputs(tmp_path, help_text=_valid_help().replace(missing, "omitted"))

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("missing", _COMMANDS)
def test_verifier_rejects_each_missing_subcommand_help(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    missing_usage = f"usage: saliencegate-review {missing} [-h]\n"
    help_text = _valid_help().replace(missing_usage, "", 1)
    demo, review_help = _write_inputs(tmp_path, help_text=help_text)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("option", ("reviewer-id", "rationale", "decision", "checklist"))
def test_verifier_rejects_automated_review_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
) -> None:
    review_usage = "usage: saliencegate-review review [-h]\n"
    help_text = _replace_in_command_help(
        _valid_help(),
        command="review",
        old=review_usage,
        new=f"{review_usage}  --{option} VALUE\n",
    )
    demo, review_help = _write_inputs(tmp_path, help_text=help_text)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("option", ("r", "d", "c"))
def test_verifier_rejects_automated_review_short_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
) -> None:
    review_usage = "usage: saliencegate-review review [-h]\n"
    help_text = _replace_in_command_help(
        _valid_help(),
        command="review",
        old=review_usage,
        new=f"{review_usage}  -{option} VALUE\n",
    )
    demo, review_help = _write_inputs(tmp_path, help_text=help_text)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize(
    ("command", "option"),
    tuple((command, option) for command, options in _COMMAND_OPTIONS.items() for option in options),
)
def test_verifier_rejects_each_missing_public_option(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    option: str,
) -> None:
    help_text = _replace_in_command_help(
        _valid_help(),
        command=command,
        old=f"  --{option} VALUE\n",
        new="",
    )
    demo, review_help = _write_inputs(tmp_path, help_text=help_text)

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("forbidden", _FORBIDDEN)
def test_verifier_rejects_unsupported_and_unsafe_review_operations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    forbidden: str,
) -> None:
    demo, review_help = _write_inputs(
        tmp_path,
        help_text=f"{_valid_help()}  --{forbidden} FORBIDDEN\n",
    )

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda demo, review: (demo.parent, review),
        lambda demo, review: (demo, review.parent),
        lambda demo, review: (demo.with_name("demo-link.json"), review),
        lambda demo, review: (demo, review.with_name("help-link.txt")),
    ),
    ids=("demo-directory", "help-directory", "demo-symlink", "help-symlink"),
)
def test_verifier_rejects_non_regular_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutate: Callable[[Path, Path], tuple[Path, Path]],
) -> None:
    demo, review_help = _write_inputs(tmp_path)
    demo.with_name("demo-link.json").symlink_to(demo.name, target_is_directory=False)
    review_help.with_name("help-link.txt").symlink_to(review_help.name, target_is_directory=False)
    supplied = mutate(demo, review_help)

    assert _invoke([str(supplied[0]), str(supplied[1])], capsys) == (1, "", _FAILURE)


def test_verifier_rejects_fifo_inputs_without_opening_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _demo, review_help = _write_inputs(tmp_path)
    fifo = tmp_path / "demo.fifo"
    os.mkfifo(fifo)

    assert _invoke([str(fifo), str(review_help)], capsys) == (1, "", _FAILURE)


@pytest.mark.parametrize("target", ("demo", "help"))
def test_verifier_rejects_oversized_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    payload = "x" * 1_000_000
    demo, review_help = _write_inputs(
        tmp_path,
        demo=payload if target == "demo" else None,
        help_text=payload if target == "help" else None,
    )

    assert _invoke([str(demo), str(review_help)], capsys) == (1, "", _FAILURE)
