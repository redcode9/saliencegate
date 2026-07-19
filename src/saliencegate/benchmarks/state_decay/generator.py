from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from saliencegate.benchmarks.state_decay.schema import (
    GENERATOR_VERSION,
    AllowedAction,
    CandidateMemory,
    ContinuationBranch,
    ContinuationOutcome,
    EvidenceCriteria,
    InterventionLabel,
    MemorySourceRef,
    OracleCriteria,
    PairedContinuation,
    Pivot,
    ScenarioFamily,
    StateDecayScenario,
    TrajectoryEvent,
)
from saliencegate.domain import ValidityState, canonical_json, length_prefixed_sha256

SMOKE_SEED = 20_260_711
SMOKE_SCENARIO_COUNT = 32
_SCENARIO_DIGEST_DOMAIN = "saliencegate:state-decay:scenario:v1"


class SmokeCoverageError(ValueError):
    """A value-free invalid deterministic smoke suite error."""

    def __init__(self) -> None:
        super().__init__("state decay smoke coverage validation failed")


class SmokeFixtureWriteError(ValueError):
    """A value-free committed-fixture generation error."""

    def __init__(self) -> None:
        super().__init__("state decay smoke fixture generation failed")


def _fail() -> NoReturn:
    raise SmokeCoverageError() from None


@dataclass(frozen=True, slots=True)
class _FamilyTemplate:
    family: ScenarioFamily
    retained_fact: str
    revision_fact: str
    retained_action: str
    revised_action: str


_FAMILY_TEMPLATES = (
    _FamilyTemplate(
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "The release must preserve the version-one response field names.",
        "The migration owner approved version-two field names for this isolated fixture.",
        "Preserve the version-one response field names.",
        "Use the newly approved version-two response field names.",
    ),
    _FamilyTemplate(
        ScenarioFamily.STABLE_ENVIRONMENT_FACT,
        "The test database listens on port 5432 in this environment.",
        "The test database was deliberately moved to port 6432 before the pivot.",
        "Connect to the test database on port 5432.",
        "Connect to the test database on its revised port 6432.",
    ),
    _FamilyTemplate(
        ScenarioFamily.FAILED_PRIOR_ATTEMPT,
        "The direct retry failed while the required dependency was absent.",
        "The missing dependency is now installed, so the earlier retry constraint is obsolete.",
        "Avoid repeating the direct retry and choose the prepared fallback.",
        "Retry directly now that the blocking dependency is present.",
    ),
    _FamilyTemplate(
        ScenarioFamily.RETAINED_DIAGNOSIS,
        "The deterministic evidence identifies a malformed configuration as the root cause.",
        "A later checksum disproves that diagnosis and identifies a truncated fixture instead.",
        "Repair the malformed configuration identified by the evidence.",
        "Repair the newly identified truncated fixture.",
    ),
    _FamilyTemplate(
        ScenarioFamily.NEGLECTED_SUBGOAL,
        "The schema migration remains an open prerequisite for the release.",
        "The schema migration completed and the verification subgoal is now the prerequisite.",
        "Complete the open schema-migration subgoal first.",
        "Run the verification subgoal after the completed migration.",
    ),
    _FamilyTemplate(
        ScenarioFamily.STALE_MEMORY,
        "The generated manifest belongs under the legacy artifacts directory.",
        "The repository layout now places generated manifests under the evidence directory.",
        "Write the manifest to the still-current artifacts directory.",
        "Write the manifest to the revised evidence directory.",
    ),
    _FamilyTemplate(
        ScenarioFamily.CONFLICTING_EVIDENCE,
        "The signed configuration is authoritative over the older console message.",
        "A newer signed configuration supersedes the earlier one and reverses the selected mode.",
        "Follow the retained signed configuration rather than the console message.",
        "Follow the newer signed configuration that resolved the conflict.",
    ),
    _FamilyTemplate(
        ScenarioFamily.IRREVERSIBLE_ACTION,
        "Destructive cleanup is forbidden until the synthetic approval marker exists.",
        "The synthetic approval marker now exists and explicitly authorizes the bounded cleanup.",
        "Refuse destructive cleanup because approval is absent.",
        "Perform only the newly approved bounded cleanup.",
    ),
)


def scenario_digest(scenario: StateDecayScenario) -> str:
    payload = scenario.model_dump(
        mode="json",
        exclude={"scenario_id"},
        warnings=False,
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=_SCENARIO_DIGEST_DOMAIN,
    )


def _continuations(
    *,
    reminder_required: bool,
    required_action_id: str,
    alternate_action_id: str,
    decisive_source_ids: tuple[str, ...],
    decisive_memory_ids: tuple[str, ...],
) -> tuple[PairedContinuation, PairedContinuation]:
    successful_branch = (
        ContinuationBranch.REMINDER if reminder_required else ContinuationBranch.SILENCE
    )

    def continuation(branch: ContinuationBranch) -> PairedContinuation:
        succeeds = branch is successful_branch
        return PairedContinuation(
            branch=branch,
            selected_action_id=required_action_id if succeeds else alternate_action_id,
            outcome=ContinuationOutcome.SUCCESS if succeeds else ContinuationOutcome.FAILURE,
            explanation=(
                "The branch selects the required action using all decisive evidence."
                if succeeds
                else "The branch selects the alternate action without decisive evidence."
            ),
            evidence_source_ids=decisive_source_ids if succeeds else (),
            evidence_memory_ids=decisive_memory_ids if succeeds else (),
        )

    return (
        continuation(ContinuationBranch.REMINDER),
        continuation(ContinuationBranch.SILENCE),
    )


def _scenario(
    template: _FamilyTemplate,
    *,
    variant: int,
) -> StateDecayScenario:
    family_slug = template.family.value.replace("_", "-")
    suffix = f"{family_slug}-{variant + 1}"
    retained_source_id = f"source-{suffix}-retained"
    routine_source_id = f"source-{suffix}-routine"
    current_source_id = f"source-{suffix}-current"
    pivot_source_id = f"pivot-{suffix}"
    memory_id = f"memory-{suffix}"
    required_action_id = f"action-{suffix}-required"
    alternate_action_id = f"action-{suffix}-alternate"
    state_was_revised = variant >= 2
    memory_is_required = not state_was_revised
    current_statement = (
        "Routine progress adds no evidence that changes the retained fact."
        if memory_is_required
        else template.revision_fact
    )
    required_statement = template.retained_action if memory_is_required else template.revised_action
    alternate_statement = (
        template.revised_action if memory_is_required else template.retained_action
    )
    decisive_source_ids = (retained_source_id,) if memory_is_required else (current_source_id,)
    decisive_memory_ids = (memory_id,) if memory_is_required else ()
    memory_validity = ValidityState.ACTIVE if memory_is_required else ValidityState.SUPERSEDED

    provisional = StateDecayScenario(
        generator_version=GENERATOR_VERSION,
        seed=SMOKE_SEED,
        scenario_id="0" * 64,
        template_lineage_id=f"state-decay-{family_slug}-template",
        family=template.family,
        trajectory_prefix=(
            TrajectoryEvent(
                step=1,
                source_id=retained_source_id,
                statement=template.retained_fact,
            ),
            TrajectoryEvent(
                step=2,
                source_id=routine_source_id,
                statement=f"Routine deterministic work unit {variant + 1} completed.",
            ),
            TrajectoryEvent(
                step=3,
                source_id=current_source_id,
                statement=current_statement,
            ),
        ),
        candidate_memories=(
            CandidateMemory(
                memory_id=memory_id,
                statement=template.retained_fact,
                source_refs=(MemorySourceRef(source_id=retained_source_id, source_step=1),),
                revision=1,
                validity=memory_validity,
                validity_step=None if memory_is_required else 3,
                recorded_step=2,
            ),
        ),
        pivot=Pivot(
            step=4,
            source_id=pivot_source_id,
            statement="Choose exactly one allowed next action at this paired decision pivot.",
        ),
        allowed_actions=(
            AllowedAction(
                action_id=required_action_id,
                statement=required_statement,
            ),
            AllowedAction(
                action_id=alternate_action_id,
                statement=alternate_statement,
            ),
        ),
        label=InterventionLabel.INTERVENE,
        oracle=OracleCriteria(
            required_action_id=required_action_id,
            success_condition="Select the required action at the paired pivot.",
            failure_condition="Select the alternate action at the paired pivot.",
        ),
        evidence_criteria=EvidenceCriteria(
            decisive_source_ids=decisive_source_ids,
            decisive_memory_ids=decisive_memory_ids,
        ),
        paired_continuations=_continuations(
            reminder_required=memory_is_required,
            required_action_id=required_action_id,
            alternate_action_id=alternate_action_id,
            decisive_source_ids=decisive_source_ids,
            decisive_memory_ids=decisive_memory_ids,
        ),
    )
    # The observational label is derived from the executable paired transition;
    # it is never an input to action, evidence, or outcome generation.
    from saliencegate.benchmarks.state_decay.oracle import evaluate_scenario

    derived_label = evaluate_scenario(provisional).expected_label
    labelled = provisional.model_copy(update={"label": derived_label})
    identified = labelled.model_copy(update={"scenario_id": scenario_digest(labelled)})
    return StateDecayScenario.model_validate_json(identified.model_dump_json(warnings=False))


def generate_smoke_scenarios() -> tuple[StateDecayScenario, ...]:
    scenarios = tuple(
        _scenario(
            template,
            variant=variant,
        )
        for template in _FAMILY_TEMPLATES
        for variant in range(4)
    )
    validate_smoke_coverage(scenarios)
    return scenarios


def validate_smoke_coverage(scenarios: tuple[StateDecayScenario, ...]) -> None:
    valid = False
    try:
        if type(scenarios) is not tuple or len(scenarios) != SMOKE_SCENARIO_COUNT:
            _fail()
        validated = tuple(
            StateDecayScenario.model_validate_json(item.model_dump_json(warnings=False))
            for item in scenarios
            if type(item) is StateDecayScenario
        )
        if len(validated) != len(scenarios) or validated != scenarios:
            _fail()
        if any(
            scenario.generator_version != GENERATOR_VERSION
            or scenario.seed != SMOKE_SEED
            or scenario.scenario_id != scenario_digest(scenario)
            for scenario in validated
        ):
            _fail()
        scenario_ids = tuple(scenario.scenario_id for scenario in validated)
        lineage_ids = tuple(scenario.template_lineage_id for scenario in validated)
        if len(set(scenario_ids)) != len(scenario_ids):
            _fail()
        source_ids = tuple(
            source_id
            for scenario in validated
            for source_id in (
                *(event.source_id for event in scenario.trajectory_prefix),
                scenario.pivot.source_id,
            )
        )
        memory_ids = tuple(
            memory.memory_id for scenario in validated for memory in scenario.candidate_memories
        )
        action_ids = tuple(
            action.action_id for scenario in validated for action in scenario.allowed_actions
        )
        if any(len(values) != len(set(values)) for values in (source_ids, memory_ids, action_ids)):
            _fail()
        lineage_counts = {
            lineage: sum(row.template_lineage_id == lineage for row in validated)
            for lineage in set(lineage_ids)
        }
        if len(lineage_counts) != len(tuple(ScenarioFamily)) or set(lineage_counts.values()) != {4}:
            _fail()
        if {scenario.family for scenario in validated} != set(ScenarioFamily):
            _fail()
        for family in ScenarioFamily:
            family_rows = tuple(row for row in validated if row.family is family)
            if len(family_rows) != 4:
                _fail()
            if sum(row.label is InterventionLabel.INTERVENE for row in family_rows) != 2:
                _fail()
            if sum(row.label is InterventionLabel.SILENCE for row in family_rows) != 2:
                _fail()
            if len({row.template_lineage_id for row in family_rows}) != 1:
                _fail()
        valid = True
    except SmokeCoverageError:
        raise
    except Exception:
        pass
    if not valid:
        _fail()


def encode_scenarios_jsonl(scenarios: tuple[StateDecayScenario, ...]) -> bytes:
    validate_smoke_coverage(scenarios)
    return b"".join(
        canonical_json(scenario.model_dump(mode="json", warnings=False)) + b"\n"
        for scenario in scenarios
    )


def _fixture_directory(value: str | os.PathLike[str]) -> Path:
    if isinstance(value, bytes):
        raise SmokeFixtureWriteError() from None
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise TypeError
        path = Path(raw)
        if path.name in ("", ".", ".."):
            raise ValueError
        existing = path.lstat() if path.exists() or path.is_symlink() else None
        if existing is not None and (
            stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode)
        ):
            raise ValueError
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (
                os.name == "posix"
                and (
                    (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                )
            )
        ):
            raise ValueError
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SmokeFixtureWriteError() from None


def _replace_fixture_file(
    directory: Path,
    name: str,
    data: bytes,
    *,
    replace: bool,
) -> None:
    destination = directory / name
    try:
        current = destination.lstat()
    except FileNotFoundError:
        current = None
    except OSError:
        raise SmokeFixtureWriteError() from None
    if current is not None:
        if not replace:
            raise SmokeFixtureWriteError() from None
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (os.name == "posix" and hasattr(os, "getuid") and current.st_uid != os.getuid())
        ):
            raise SmokeFixtureWriteError() from None

    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.tmp-",
            dir=directory,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o644)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
        temporary = None
        if destination.read_bytes() != data:
            raise OSError
    except OSError:
        raise SmokeFixtureWriteError() from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _preflight_fixture_file(directory: Path, name: str, *, replace: bool) -> None:
    try:
        current = (directory / name).lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise SmokeFixtureWriteError() from None
    if not replace or (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (os.name == "posix" and hasattr(os, "getuid") and current.st_uid != os.getuid())
    ):
        raise SmokeFixtureWriteError() from None


def write_smoke_fixture(
    output: str | os.PathLike[str],
    *,
    replace: bool = False,
) -> tuple[Path, Path]:
    """Regenerate the two reviewable source fixtures from package code only."""

    if type(replace) is not bool:
        raise SmokeFixtureWriteError() from None
    directory = _fixture_directory(output)
    try:
        from saliencegate.benchmarks.state_decay.runner import (
            MANIFEST_NAME,
            state_decay_artifact_files,
        )

        files = state_decay_artifact_files()
        manifest = files[MANIFEST_NAME]
        fixture = files["smoke.jsonl"]
        # Authorize both targets before changing either one. Each completed
        # temporary file is then published with one atomic rename.
        _preflight_fixture_file(directory, "smoke.jsonl", replace=replace)
        _preflight_fixture_file(
            directory,
            "smoke_manifest.json",
            replace=replace,
        )
        _replace_fixture_file(
            directory,
            "smoke.jsonl",
            fixture,
            replace=replace,
        )
        _replace_fixture_file(
            directory,
            "smoke_manifest.json",
            manifest,
            replace=replace,
        )
        if (directory / "smoke.jsonl").read_bytes() != fixture or (
            directory / "smoke_manifest.json"
        ).read_bytes() != manifest:
            raise SmokeFixtureWriteError() from None
        return directory / "smoke_manifest.json", directory / "smoke.jsonl"
    except SmokeFixtureWriteError:
        raise
    except Exception:
        raise SmokeFixtureWriteError() from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m saliencegate.benchmarks.state_decay.generator",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--replace", action="store_true")
    try:
        arguments = parser.parse_args(argv)
        write_smoke_fixture(arguments.output, replace=arguments.replace)
    except SmokeFixtureWriteError:
        sys.stderr.write("error: state decay smoke fixture generation failed\n")
        return 2
    sys.stdout.write("generated StateDecayBench smoke fixture\n")
    return 0


__all__ = [
    "SMOKE_SCENARIO_COUNT",
    "SMOKE_SEED",
    "SmokeCoverageError",
    "SmokeFixtureWriteError",
    "encode_scenarios_jsonl",
    "generate_smoke_scenarios",
    "main",
    "scenario_digest",
    "validate_smoke_coverage",
    "write_smoke_fixture",
]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
