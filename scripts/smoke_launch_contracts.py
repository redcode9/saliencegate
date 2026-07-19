from __future__ import annotations

import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

_SUCCESS: Final = "launch-contracts-ok"
_FAILURE: Final = "launch-contracts-failed"
_MAX_DEMO_BYTES: Final = 16 * 1024
_MAX_HELP_BYTES: Final = 64 * 1024
_COMMAND_ORDER: Final = ("build-pack", "review", "status", "build-envelope")
_REQUIRED_COMMANDS: Final = frozenset(_COMMAND_ORDER)
_REQUIRED_LONG_OPTIONS: Final = {
    "": frozenset({"--help"}),
    "build-pack": frozenset({"--help", "--output", "--json"}),
    "review": frozenset({"--help", "--pack", "--reviews"}),
    "status": frozenset({"--help", "--pack", "--reviews", "--json"}),
    "build-envelope": frozenset({"--help", "--pack", "--reviews", "--lineage-key", "--json"}),
}
_LONG_OPTION_PATTERN: Final = re.compile(r"(?<![a-z0-9_-])--[^\s,\[\](){}=]+")
_SHORT_OPTION_PATTERN: Final = re.compile(r"(?<![a-z0-9_-])-[a-z0-9](?![a-z0-9_-])")
_REQUIRED_SHORT_OPTIONS: Final = frozenset({"-h"})
_FORBIDDEN_OPERATIONS: Final = (
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
_DEMO_FIELDS: Final = {
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


class _LaunchContractError(Exception):
    pass


def _read_regular_file(path_value: str, *, maximum_bytes: int) -> bytes:
    path = Path(path_value)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
        raise _LaunchContractError

    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise _LaunchContractError
        with os.fdopen(descriptor, mode="rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) != opened.st_size
            or len(payload) > maximum_bytes
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise _LaunchContractError
        return payload
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _LaunchContractError
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> object:
    raise _LaunchContractError


def _verify_demo(payload: bytes) -> None:
    parsed = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(parsed, dict):
        raise _LaunchContractError
    if set(parsed) != set(_DEMO_FIELDS):
        raise _LaunchContractError
    for field, expected in _DEMO_FIELDS.items():
        observed = parsed[field]
        if type(observed) is not type(expected) or observed != expected:
            raise _LaunchContractError
    canonical = (
        json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if payload != canonical:
        raise _LaunchContractError


def _bounded_word_pattern(value: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])")


def _verify_review_help(payload: bytes) -> None:
    help_text = payload.decode("utf-8")
    if "\x00" in help_text:
        raise _LaunchContractError
    folded = help_text.casefold()
    if re.match(r"\Ausage: saliencegate-review(?:[ \r\n])", folded) is None:
        raise _LaunchContractError
    if any(_bounded_word_pattern(value).search(folded) for value in _FORBIDDEN_OPERATIONS):
        raise _LaunchContractError

    usage_matches = tuple(
        re.finditer(
            r"(?m)^usage: saliencegate-review(?: ([a-z][a-z-]*))?(?=[ \r\n])",
            folded,
        )
    )
    usage_commands = tuple(match.group(1) or "" for match in usage_matches)
    if usage_commands != ("", *_COMMAND_ORDER):
        raise _LaunchContractError
    for index, (command, match) in enumerate(zip(usage_commands, usage_matches, strict=True)):
        end = usage_matches[index + 1].start() if index + 1 < len(usage_matches) else len(folded)
        observed_options = frozenset(_LONG_OPTION_PATTERN.findall(folded[match.start() : end]))
        if observed_options != _REQUIRED_LONG_OPTIONS[command]:
            raise _LaunchContractError
        observed_short_options = frozenset(
            _SHORT_OPTION_PATTERN.findall(folded[match.start() : end])
        )
        if observed_short_options != _REQUIRED_SHORT_OPTIONS:
            raise _LaunchContractError

    choice_groups = re.findall(r"\{([^{}\r\n]+)\}", folded)
    if not choice_groups:
        raise _LaunchContractError
    for group in choice_groups:
        choices = group.split(",")
        if len(choices) != len(set(choices)) or frozenset(choices) != _REQUIRED_COMMANDS:
            raise _LaunchContractError
    if any(_bounded_word_pattern(command).search(folded) is None for command in _REQUIRED_COMMANDS):
        raise _LaunchContractError


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) != 2:
            raise _LaunchContractError
        _verify_demo(_read_regular_file(arguments[0], maximum_bytes=_MAX_DEMO_BYTES))
        _verify_review_help(_read_regular_file(arguments[1], maximum_bytes=_MAX_HELP_BYTES))
    except Exception:
        print(_FAILURE, file=sys.stderr)
        return 1
    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
