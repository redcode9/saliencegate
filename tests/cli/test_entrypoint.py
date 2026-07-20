from __future__ import annotations

import argparse

from tests.cli.conftest import RunCli

from saliencegate import __version__
from saliencegate import cli as cli_module

_SUPPRESSED = "<suppressed>"
_BASELINE_TOP_LEVEL_COMMANDS = (
    "demo",
    "doctor",
    "replay",
    "shadow",
    "algorithm",
    "pilot",
    "benchmark",
    "inspect",
    "validate",
)


def _argument(
    dest: str,
    *option_strings: str,
    action: str = "store",
    nargs: object = None,
    required: bool = False,
    default: object = None,
    const: object = None,
    choices: tuple[str, ...] | None = None,
    value_type: str | None = None,
    metavar: object = None,
) -> tuple[object, ...]:
    return (
        action,
        option_strings,
        dest,
        nargs,
        required,
        default,
        const,
        choices,
        value_type,
        metavar,
    )


_HELP = _argument("help", "-h", "--help", action="help", nargs=0, default=_SUPPRESSED)
_JSON = _argument("json", "--json", action="store_true", nargs=0, default=False, const=True)
_REPLACE = _argument(
    "replace",
    "--replace",
    action="store_true",
    nargs=0,
    default=False,
    const=True,
)
_BASELINE_CLI_PARSER_TREE = {
    (): (
        "saliencegate",
        False,
        (
            _HELP,
            _argument(
                "version",
                "--version",
                action="version",
                nargs=0,
                default=_SUPPRESSED,
            ),
            _argument(
                "command",
                action="subparsers",
                nargs="subcommands",
                required=True,
                choices=_BASELINE_TOP_LEVEL_COMMANDS,
            ),
        ),
    ),
    ("demo",): ("saliencegate demo", False, (_HELP, _JSON)),
    ("doctor",): (
        "saliencegate doctor",
        False,
        (
            _HELP,
            _argument("repository", "--repository", default="saliencegate.sqlite3"),
            _argument("key", "--key"),
            _argument("endpoint", "--endpoint"),
            _argument(
                "capture",
                "--capture",
                action="store_true",
                nargs=0,
                default=False,
                const=True,
            ),
            _JSON,
        ),
    ),
    ("replay",): (
        "saliencegate replay",
        False,
        (
            _HELP,
            _argument("trace", required=True),
            _argument("output", "--output", required=True),
            _argument("responses", "--responses"),
            _REPLACE,
            _JSON,
        ),
    ),
    ("shadow",): (
        "saliencegate shadow",
        False,
        (
            _HELP,
            _argument(
                "shadow_command",
                action="subparsers",
                nargs="subcommands",
                required=True,
                choices=("analyze", "analyze-atif"),
            ),
        ),
    ),
    ("shadow", "analyze"): (
        "saliencegate shadow analyze",
        False,
        (
            _HELP,
            _argument("trace", required=True),
            _argument("run_id", "--run-id", required=True),
            _argument("output", "--output", required=True),
            _argument("repository", "--repository", default=":memory:"),
            _argument(
                "capture_scope",
                "--capture-scope",
                default="unknown",
                choices=(
                    "unknown",
                    "selected_events",
                    "bounded_window",
                    "complete_run_declared",
                ),
            ),
            _argument("task_scope_digest", "--task-scope-digest"),
            _argument("lineage_scope_digest", "--lineage-scope-digest"),
            _argument("capture_manifest_digest", "--capture-manifest-digest"),
            _argument("source_adapter", "--source-adapter", default="saliencegate-shadow/v1"),
            _REPLACE,
            _JSON,
        ),
    ),
    ("shadow", "analyze-atif"): (
        "saliencegate shadow analyze-atif",
        False,
        (
            _HELP,
            _argument("trace", required=True),
            _argument(
                "profile",
                "--profile",
                required=True,
                choices=("harbor-terminus-2-v1", "harbor-codex-v1"),
            ),
            _argument("run_id", "--run-id", required=True),
            _argument("working_directory", "--working-directory", required=True),
            _argument("environment_digest", "--environment-digest", required=True),
            _argument("output", "--output", required=True),
            _argument("repository", "--repository", default=":memory:"),
            _argument("task_scope_digest", "--task-scope-digest"),
            _argument("lineage_scope_digest", "--lineage-scope-digest"),
            _argument("capture_manifest_digest", "--capture-manifest-digest"),
            _REPLACE,
            _JSON,
        ),
    ),
    ("algorithm",): (
        "saliencegate algorithm",
        False,
        (
            _HELP,
            _argument(
                "algorithm_command",
                action="subparsers",
                nargs="subcommands",
                required=True,
                choices=("replay",),
            ),
        ),
    ),
    ("algorithm", "replay"): (
        "saliencegate algorithm replay",
        False,
        (
            _HELP,
            _argument("trace", required=True),
            _argument("responses", "--responses"),
            _argument(
                "condition",
                "--condition",
                required=True,
                choices=("no_memory", "fixed_step", "retrieval_always", "always_inject"),
            ),
            _argument("output", "--output", required=True),
            _REPLACE,
            _JSON,
        ),
    ),
    ("pilot",): (
        "saliencegate pilot",
        False,
        (
            _HELP,
            _argument(
                "pilot_command",
                action="subparsers",
                nargs="subcommands",
                required=True,
                choices=("paper-two-phase",),
            ),
        ),
    ),
    ("pilot", "paper-two-phase"): (
        "saliencegate pilot paper-two-phase",
        False,
        (
            _HELP,
            _argument("endpoint", "--endpoint", required=True),
            _argument("model", "--model", required=True),
            _argument("output", "--output", required=True),
            _argument("warmup", "--warmup", default="warm", choices=("warm", "cold")),
            _JSON,
        ),
    ),
    ("benchmark",): (
        "saliencegate benchmark",
        False,
        (
            _HELP,
            _argument("suite", required=True),
            _argument("output", "--output", required=True),
            _REPLACE,
            _JSON,
        ),
    ),
    ("inspect",): (
        "saliencegate inspect",
        False,
        (
            _HELP,
            _argument("run_id", required=True),
            _argument("artifact", "--artifact", required=True),
            _JSON,
        ),
    ),
    ("validate",): (
        "saliencegate validate",
        False,
        (
            _HELP,
            _argument("artifact", required=True),
            _argument("expected_digest", "--expected-digest"),
            _argument(
                "require_confirmatory",
                "--require-confirmatory",
                action="store_true",
                nargs=0,
                default=False,
                const=True,
            ),
            _JSON,
        ),
    ),
}


def _action_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._SubParsersAction):
        return "subparsers"
    if isinstance(action, argparse._HelpAction):
        return "help"
    if isinstance(action, argparse._VersionAction):
        return "version"
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreAction):
        return "store"
    return type(action).__name__


def _value_type_name(action: argparse.Action) -> str | None:
    if action.type is None:
        return None
    module = getattr(action.type, "__module__", type(action.type).__module__)
    name = getattr(action.type, "__qualname__", type(action.type).__qualname__)
    return f"{module}.{name}"


def _normalized_action(
    action: argparse.Action,
    *,
    subcommands: tuple[str, ...] | None = None,
) -> tuple[object, ...]:
    default = _SUPPRESSED if action.default == argparse.SUPPRESS else action.default
    choices = subcommands if subcommands is not None else action.choices
    nargs = "subcommands" if isinstance(action, argparse._SubParsersAction) else action.nargs
    return (
        _action_kind(action),
        tuple(action.option_strings),
        action.dest,
        nargs,
        action.required,
        default,
        action.const,
        None if choices is None else tuple(choices),
        _value_type_name(action),
        action.metavar,
    )


def _normalized_baseline_parser_tree(
    parser: argparse.ArgumentParser,
) -> dict[tuple[str, ...], tuple[object, ...]]:
    tree: dict[tuple[str, ...], tuple[object, ...]] = {}

    def visit(current: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        normalized_actions: list[tuple[object, ...]] = []
        child_parsers: list[tuple[str, argparse.ArgumentParser]] = []
        for action in current._actions:
            if not isinstance(action, argparse._SubParsersAction):
                normalized_actions.append(_normalized_action(action))
                continue

            names = tuple(action.choices)
            if not path:
                names = tuple(name for name in names if name in _BASELINE_TOP_LEVEL_COMMANDS)
            normalized_actions.append(_normalized_action(action, subcommands=names))
            child_parsers.extend((name, action.choices[name]) for name in names)

        tree[path] = (current.prog, current.allow_abbrev, tuple(normalized_actions))
        for name, child in child_parsers:
            visit(child, (*path, name))

    visit(parser, ())
    return tree


def test_existing_cli_parser_tree_is_semantically_frozen() -> None:
    parser = cli_module._parser()
    root_subparsers = tuple(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    assert len(root_subparsers) == 1
    assert set(_BASELINE_TOP_LEVEL_COMMANDS) <= root_subparsers[0].choices.keys()
    assert _normalized_baseline_parser_tree(parser) == _BASELINE_CLI_PARSER_TREE


def test_module_entrypoint_reports_version_without_diagnostics(run_cli: RunCli) -> None:
    completed = run_cli("--version")

    assert completed.returncode == 0
    assert completed.stdout == f"{__version__}\n"
    assert completed.stderr == ""


def test_module_entrypoint_missing_command_has_stable_value_free_error(run_cli: RunCli) -> None:
    completed = run_cli()

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: invalid command line\n"


def test_shadow_entrypoint_requires_the_exact_subcommand_and_flags(run_cli: RunCli) -> None:
    invalid_commands = (
        ("shadow",),
        ("shadow", "analyze"),
        ("shadow", "analyze", "trace", "--run-id", "not-a-uuid"),
        (
            "shadow",
            "analyze",
            "trace",
            "--run-id",
            "b35f05f3-555b-4f09-8996-a7b3693bb54a",
            "--output",
            "report",
            "--repo",
            ":memory:",
        ),
        (
            "shadow",
            "analyze",
            "trace",
            "--run-id",
            "b35f05f3-555b-4f09-8996-a7b3693bb54a",
            "--output",
            "report",
            "--model",
            "fixture-secret-model",
        ),
    )

    for arguments in invalid_commands:
        completed = run_cli(*arguments)
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: invalid command line\n"


def test_shadow_entrypoint_rejects_noncanonical_or_non_v4_run_ids(run_cli: RunCli) -> None:
    invalid_run_ids = (
        "b35f05f3555b4f098996a7b3693bb54a",
        "B35F05F3-555B-4F09-8996-A7B3693BB54A",
        "{b35f05f3-555b-4f09-8996-a7b3693bb54a}",
        "b35f05f3-555b-1f09-8996-a7b3693bb54a",
        " b35f05f3-555b-4f09-8996-a7b3693bb54a",
    )

    for run_id in invalid_run_ids:
        completed = run_cli(
            "shadow",
            "analyze",
            "fixture-secret-trace",
            "--run-id",
            run_id,
            "--output",
            "fixture-secret-output",
        )
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: shadow input or output is invalid\n"
        assert "fixture-secret" not in completed.stderr
