from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import saliencegate.benchmarks.state_decay.generator as generator_module
from saliencegate.benchmarks.state_decay.generator import (
    SMOKE_SCENARIO_COUNT,
    SMOKE_SEED,
    SmokeCoverageError,
    SmokeFixtureWriteError,
    encode_scenarios_jsonl,
    generate_smoke_scenarios,
    scenario_digest,
    validate_smoke_coverage,
    write_smoke_fixture,
)
from saliencegate.benchmarks.state_decay.schema import (
    GENERATOR_VERSION,
    InterventionLabel,
    ScenarioFamily,
    StateDecayScenario,
)
from saliencegate.domain import canonical_json

ROOT = Path(__file__).parents[3]
COMMITTED_FIXTURES = ROOT / "benchmarks" / "state_decay"


def _reseal(
    scenario: StateDecayScenario,
    **updates: object,
) -> StateDecayScenario:
    altered = scenario.model_copy(update=updates)
    identified = altered.model_copy(update={"scenario_id": scenario_digest(altered)})
    return StateDecayScenario.model_validate_json(identified.model_dump_json(warnings=False))


def test_smoke_generator_is_fixed_complete_and_balanced_inside_each_family() -> None:
    scenarios = generate_smoke_scenarios()

    assert len(scenarios) == SMOKE_SCENARIO_COUNT == 32
    assert len(tuple(ScenarioFamily)) == 8
    assert all(row.seed == SMOKE_SEED for row in scenarios)
    assert all(row.generator_version == GENERATOR_VERSION for row in scenarios)
    assert Counter(row.family for row in scenarios) == {family: 4 for family in ScenarioFamily}
    for family in ScenarioFamily:
        labels = Counter(row.label for row in scenarios if row.family is family)
        assert labels == {
            InterventionLabel.INTERVENE: 2,
            InterventionLabel.SILENCE: 2,
        }


def test_smoke_generator_has_unique_ids_and_linked_template_lineage() -> None:
    scenarios = generate_smoke_scenarios()

    assert len({row.scenario_id for row in scenarios}) == 32
    assert len({row.template_lineage_id for row in scenarios}) == 8
    assert Counter(row.template_lineage_id for row in scenarios) == {
        f"state-decay-{family.value.replace('_', '-')}-template": 4 for family in ScenarioFamily
    }
    assert all(row.scenario_id == scenario_digest(row) for row in scenarios)
    source_ids = [
        source_id
        for row in scenarios
        for source_id in (
            *(event.source_id for event in row.trajectory_prefix),
            row.pivot.source_id,
        )
    ]
    memory_ids = [memory.memory_id for row in scenarios for memory in row.candidate_memories]
    action_ids = [action.action_id for row in scenarios for action in row.allowed_actions]
    assert len(source_ids) == len(set(source_ids))
    assert len(memory_ids) == len(set(memory_ids))
    assert len(action_ids) == len(set(action_ids))
    validate_smoke_coverage(scenarios)


def test_every_reference_resolves_at_or_before_the_pivot() -> None:
    for scenario in generate_smoke_scenarios():
        source_steps = {event.source_id: event.step for event in scenario.trajectory_prefix}
        source_steps[scenario.pivot.source_id] = scenario.pivot.step
        memories = {memory.memory_id: memory for memory in scenario.candidate_memories}

        assert all(
            earlier.step < later.step for earlier, later in pairwise(scenario.trajectory_prefix)
        )
        assert scenario.trajectory_prefix[-1].step < scenario.pivot.step
        for memory in scenario.candidate_memories:
            assert memory.recorded_step <= scenario.pivot.step
            for reference in memory.source_refs:
                assert source_steps[reference.source_id] == reference.source_step
                assert reference.source_step <= memory.recorded_step
        for source_id in scenario.evidence_criteria.decisive_source_ids:
            assert source_steps[source_id] <= scenario.pivot.step
        for memory_id in scenario.evidence_criteria.decisive_memory_ids:
            assert memories[memory_id].recorded_step <= scenario.pivot.step
        for continuation in scenario.paired_continuations:
            assert all(
                source_steps[source_id] <= scenario.pivot.step
                for source_id in continuation.evidence_source_ids
            )
            assert all(
                memories[memory_id].recorded_step <= scenario.pivot.step
                for memory_id in continuation.evidence_memory_ids
            )


def test_jsonl_is_canonical_newline_terminated_and_byte_identical() -> None:
    first = generate_smoke_scenarios()
    second = generate_smoke_scenarios()
    encoded = encode_scenarios_jsonl(first)

    assert first == second
    assert encoded == encode_scenarios_jsonl(second)
    assert encoded.endswith(b"\n")
    lines = encoded.splitlines(keepends=True)
    assert len(lines) == 32
    for scenario, line in zip(first, lines, strict=True):
        assert line == canonical_json(scenario.model_dump(mode="json", warnings=False)) + b"\n"
        assert StateDecayScenario.model_validate_json(line) == scenario


def test_generation_ignores_hash_seed_timezone_locale_and_working_directory(
    tmp_path: Path,
) -> None:
    command = (
        sys.executable,
        "-c",
        "from saliencegate.benchmarks.state_decay.generator import "
        "generate_smoke_scenarios,encode_scenarios_jsonl;"
        "import sys;sys.stdout.buffer.write(encode_scenarios_jsonl(generate_smoke_scenarios()))",
    )
    outputs: list[bytes] = []
    for hash_seed, timezone, cwd in (
        ("1", "UTC", ROOT),
        ("987654", "Europe/Rome", tmp_path),
    ):
        environment = os.environ.copy()
        environment.update(
            LC_ALL="C",
            PYTHONHASHSEED=hash_seed,
            PYTHONPATH=str(ROOT / "src"),
            TZ=timezone,
        )
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == b""
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("mutation", ["id", "balance", "future", "duplicate"])
def test_coverage_validator_rejects_forged_or_incomplete_suites_value_free(
    mutation: str,
) -> None:
    secret = "fixture-secret-lineage"
    scenarios = generate_smoke_scenarios()
    first = scenarios[0]
    if mutation == "id":
        altered = first.model_copy(update={"template_lineage_id": secret})
        forged = (altered, *scenarios[1:])
    elif mutation == "balance":
        altered = first.model_copy(update={"label": InterventionLabel.SILENCE})
        forged = (altered, *scenarios[1:])
    elif mutation == "future":
        memory = first.candidate_memories[0].model_copy(update={"recorded_step": 5})
        altered = first.model_copy(update={"candidate_memories": (memory,)})
        forged = (altered, *scenarios[1:])
    else:
        forged = (first, first, *scenarios[2:])

    with pytest.raises(SmokeCoverageError) as captured:
        validate_smoke_coverage(forged)

    assert str(captured.value) == "state decay smoke coverage validation failed"
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_coverage_validator_rejects_the_wrong_container_type() -> None:
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(list(generate_smoke_scenarios()))  # type: ignore[arg-type]


def test_documented_fixture_writer_uses_only_generated_artifact_bytes(
    tmp_path: Path,
) -> None:
    from saliencegate.benchmarks.state_decay.runner import state_decay_artifact_files

    output = tmp_path / "state_decay"
    manifest_path, fixture_path = write_smoke_fixture(output)
    expected = state_decay_artifact_files()

    assert manifest_path == output / "smoke_manifest.json"
    assert fixture_path == output / "smoke.jsonl"
    assert manifest_path.read_bytes() == expected["manifest.json"]
    assert fixture_path.read_bytes() == expected["smoke.jsonl"]
    before = {
        manifest_path.name: manifest_path.read_bytes(),
        fixture_path.name: fixture_path.read_bytes(),
    }

    with pytest.raises(SmokeFixtureWriteError):
        write_smoke_fixture(output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    write_smoke_fixture(output, replace=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_fixture_module_command_has_stable_output(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "saliencegate.benchmarks.state_decay.generator",
            "--output",
            str(output),
        ),
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "generated StateDecayBench smoke fixture\n"
    assert completed.stderr == ""
    assert {path.name for path in output.iterdir()} == {
        "smoke.jsonl",
        "smoke_manifest.json",
    }


def test_fixture_writer_preflights_both_targets_before_mutating(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    output.mkdir()
    manifest = output / "smoke_manifest.json"
    manifest.write_bytes(b"fixture-secret-existing-manifest")

    with pytest.raises(SmokeFixtureWriteError):
        write_smoke_fixture(output)

    assert manifest.read_bytes() == b"fixture-secret-existing-manifest"
    assert not (output / "smoke.jsonl").exists()


def test_committed_fixture_is_exactly_the_documented_generator_output() -> None:
    from saliencegate.benchmarks.state_decay.runner import state_decay_artifact_files

    expected = state_decay_artifact_files()

    assert (COMMITTED_FIXTURES / "smoke.jsonl").read_bytes() == expected["smoke.jsonl"]
    assert (COMMITTED_FIXTURES / "smoke_manifest.json").read_bytes() == expected["manifest.json"]


def test_coverage_validator_checks_every_cross_scenario_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = generate_smoke_scenarios()

    wrong_type = cast(
        "tuple[StateDecayScenario, ...]",
        (object(), *scenarios[1:]),
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(wrong_type)

    duplicate_pivot = scenarios[1].pivot.model_copy(
        update={"source_id": scenarios[0].pivot.source_id}
    )
    duplicate_source = (
        scenarios[0],
        _reseal(scenarios[1], pivot=duplicate_pivot),
        *scenarios[2:],
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(duplicate_source)

    wrong_lineage = (
        _reseal(
            scenarios[0],
            template_lineage_id=scenarios[4].template_lineage_id,
        ),
        *scenarios[1:],
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(wrong_lineage)

    missing_family = tuple(
        _reseal(row, family=scenarios[4].family) if index < 4 else row
        for index, row in enumerate(scenarios)
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(missing_family)

    uneven_family = (
        _reseal(scenarios[0], family=scenarios[4].family),
        *scenarios[1:],
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(uneven_family)

    wrong_intervene_balance = (
        _reseal(scenarios[0], label=InterventionLabel.SILENCE),
        *scenarios[1:],
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(wrong_intervene_balance)

    monkeypatch.setattr(
        generator_module,
        "InterventionLabel",
        SimpleNamespace(
            INTERVENE=InterventionLabel.INTERVENE,
            SILENCE=object(),
        ),
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(scenarios)
    monkeypatch.setattr(generator_module, "InterventionLabel", InterventionLabel)

    first_lineage = scenarios[0].template_lineage_id
    second_lineage = scenarios[4].template_lineage_id
    swapped_lineages = (
        _reseal(scenarios[0], template_lineage_id=second_lineage),
        *scenarios[1:4],
        _reseal(scenarios[4], template_lineage_id=first_lineage),
        *scenarios[5:],
    )
    with pytest.raises(SmokeCoverageError):
        validate_smoke_coverage(swapped_lineages)


def test_fixture_directory_rejects_unsafe_paths_and_permissions(tmp_path: Path) -> None:
    class BytePath:
        def __fspath__(self) -> bytes:
            return b"fixture"

    for invalid in (b"fixture", "", ".", "..", BytePath()):
        with pytest.raises(SmokeFixtureWriteError):
            generator_module._fixture_directory(cast("str", invalid))

    regular = tmp_path / "regular"
    regular.write_text("preserve")
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._fixture_directory(regular)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._fixture_directory(link)

    if os.name == "posix":
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir(mode=0o700)
        unsafe.chmod(0o777)
        try:
            with pytest.raises(SmokeFixtureWriteError):
                generator_module._fixture_directory(unsafe)
        finally:
            unsafe.chmod(0o700)


def test_fixture_file_low_level_guards_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "fixture.json"
    destination.write_bytes(b"old")
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._replace_fixture_file(
            tmp_path,
            destination.name,
            b"new",
            replace=False,
        )

    external = tmp_path / "external"
    external.write_bytes(b"old")
    destination.unlink()
    os.link(external, destination)
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._replace_fixture_file(
            tmp_path,
            destination.name,
            b"new",
            replace=True,
        )
    destination.unlink()

    monkeypatch.setattr(generator_module.os, "write", lambda descriptor, data: 0)
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._replace_fixture_file(
            tmp_path,
            destination.name,
            b"new",
            replace=False,
        )
    assert not tuple(tmp_path.glob(f".{destination.name}.tmp-*"))


def test_fixture_file_lstat_readback_and_preflight_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "fixture.json"
    real_lstat = Path.lstat

    def fail_fixture_lstat(path: Path) -> os.stat_result:
        if path == destination:
            raise OSError
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_fixture_lstat)
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._replace_fixture_file(
            tmp_path,
            destination.name,
            b"new",
            replace=False,
        )
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._preflight_fixture_file(
            tmp_path,
            destination.name,
            replace=True,
        )
    monkeypatch.setattr(Path, "lstat", real_lstat)

    real_read_bytes = Path.read_bytes

    def wrong_readback(path: Path) -> bytes:
        if path == destination:
            return b"different"
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", wrong_readback)
    with pytest.raises(SmokeFixtureWriteError):
        generator_module._replace_fixture_file(
            tmp_path,
            destination.name,
            b"new",
            replace=False,
        )


def test_fixture_writer_maps_invalid_flags_postwrite_and_unexpected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SmokeFixtureWriteError):
        write_smoke_fixture(tmp_path / "invalid-flag", replace=cast("bool", 1))

    real_replace = generator_module._replace_fixture_file

    def write_wrong(
        directory: Path,
        name: str,
        data: bytes,
        *,
        replace: bool,
    ) -> None:
        del data, replace
        (directory / name).write_bytes(b"wrong")

    monkeypatch.setattr(generator_module, "_replace_fixture_file", write_wrong)
    with pytest.raises(SmokeFixtureWriteError):
        write_smoke_fixture(tmp_path / "wrong-postwrite")
    monkeypatch.setattr(generator_module, "_replace_fixture_file", real_replace)

    import saliencegate.benchmarks.state_decay.runner as runner_module

    monkeypatch.setattr(runner_module, "state_decay_artifact_files", lambda: {})
    with pytest.raises(SmokeFixtureWriteError):
        write_smoke_fixture(tmp_path / "unexpected")

    output_file = tmp_path / "output-file"
    output_file.write_text("preserve")
    assert generator_module.main(["--output", str(output_file)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: state decay smoke fixture generation failed\n"
