from __future__ import annotations

import json

import pytest

from saliencegate.integrations.config_files import (
    ConfigFileError,
    ConfigSyntax,
    OwnedConfigSpec,
    _owned_config_edit_matches_spec,
    plan_owned_config_install,
    remove_owned_config_edit,
)

_MARKER = "saliencegate-owned:test-json-object-merge-v1"
_OWNED_MEMBERS = (
    b'\n    "SessionStart": [{"command": "saliencegate-owned:'
    b'test-json-object-merge-v1"}],\n    "PreToolUse": '
    b'[{"command":"saliencegate-event:PreToolUse"}]\n  '
)


def _spec(
    *,
    owned_value: bytes | None = None,
    bind_json_paths: bool = False,
) -> OwnedConfigSpec:
    value = b"{" + _OWNED_MEMBERS + b"}" if owned_value is None else owned_value
    return OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=_MARKER,
        bind_json_paths=bind_json_paths,
        owned_fragment=b'"hooks": ' + value,
    )


def test_existing_json_object_merges_one_span_and_reverses_exact_bytes() -> None:
    foreign = (
        b'{\n  "model": "foreign",\n  "hooks" : {\n'
        b'    "Notification": [{"command":"foreign"}]\n  },\n'
        b'  "permissions": {"allow":["Read"]}\n}\n'
    )

    planned = plan_owned_config_install(foreign, _spec())
    span = b"," + _OWNED_MEMBERS
    start = foreign.index(b'},\n  "permissions"')
    expected = foreign[:start] + span + foreign[start:]

    assert planned.installed_bytes == expected
    assert planned.reverse_edit.start == start
    assert planned.reverse_edit.end == start + len(span)
    assert planned.reverse_edit.owned_span() == span
    assert json.loads(planned.installed_bytes)["hooks"] == {
        "Notification": [{"command": "foreign"}],
        "SessionStart": [{"command": _MARKER}],
        "PreToolUse": [{"command": "saliencegate-event:PreToolUse"}],
    }
    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign


def test_existing_empty_json_object_merges_without_leading_comma() -> None:
    foreign = b'{"hooks": { \n }, "foreign": "unchanged"}\n'

    planned = plan_owned_config_install(foreign, _spec())
    span = _OWNED_MEMBERS
    start = foreign.index(b'}, "foreign"')

    assert planned.installed_bytes == foreign[:start] + span + foreign[start:]
    assert planned.reverse_edit.start == start
    assert planned.reverse_edit.owned_span() == span
    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign


def test_nested_json_owned_span_can_be_removed_after_disjoint_foreign_drift() -> None:
    foreign = b'{"hooks":{"Notification":[]},"model":"foreign"}'
    planned = plan_owned_config_install(foreign, _spec())
    drifted = planned.installed_bytes.replace(
        b'},"model"',
        b',"Stop":[]},"model"',
    )

    assert remove_owned_config_edit(drifted, planned.reverse_edit) == (
        b'{"hooks":{"Notification":[],"Stop":[]},"model":"foreign"}'
    )
    assert _owned_config_edit_matches_spec(planned.reverse_edit, _spec())


@pytest.mark.parametrize(
    ("foreign", "spec"),
    (
        (b'{"hooks":7}', _spec()),
        (b'{"hooks":[]}', _spec()),
        (b'{"hooks":{}}', _spec(owned_value=b'"saliencegate-owned:test-json-object-merge-v1"')),
    ),
)
def test_existing_json_owned_key_collision_or_type_mismatch_fails_closed(
    foreign: bytes,
    spec: OwnedConfigSpec,
) -> None:
    with pytest.raises(ConfigFileError):
        plan_owned_config_install(foreign, spec)


def test_existing_json_arrays_receive_unique_owned_groups_and_reverse_all_spans() -> None:
    foreign = (
        b'{"hooks":{"SessionStart":[{"command":"foreign-start"}],'
        b'"PreToolUse":[{"command":"foreign-tool"}]},"model":"foreign"}'
    )

    planned = plan_owned_config_install(foreign, _spec())
    document = json.loads(planned.installed_bytes)

    assert len(document["hooks"]["SessionStart"]) == 2
    assert len(document["hooks"]["PreToolUse"]) == 2
    assert len(planned.reverse_edit.additional_spans) == 1
    assert planned.reverse_edit.json_path == ("hooks", "SessionStart")
    assert planned.reverse_edit.additional_spans[0].json_path == ("hooks", "PreToolUse")
    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign

    extra = planned.reverse_edit.additional_spans[0]
    drifted = (
        planned.installed_bytes[: extra.end]
        + b',{"command":"foreign-later"}'
        + planned.installed_bytes[extra.end :]
    ).replace(b']},"model"', b'],"Stop":[]},"model"')
    assert remove_owned_config_edit(drifted, planned.reverse_edit) == (
        b'{"hooks":{"SessionStart":[{"command":"foreign-start"}],'
        b'"PreToolUse":[{"command":"foreign-tool"},{"command":"foreign-later"}],'
        b'"Stop":[]},"model":"foreign"}'
    )

    missing = planned.installed_bytes.replace(
        extra.owned_span(),
        b"",
        1,
    )
    with pytest.raises(ConfigFileError):
        remove_owned_config_edit(missing, planned.reverse_edit)

    duplicated = (
        planned.installed_bytes[: extra.end]
        + extra.owned_span()
        + planned.installed_bytes[extra.end :]
    )
    with pytest.raises(ConfigFileError):
        remove_owned_config_edit(duplicated, planned.reverse_edit)


def test_composite_drift_removal_never_uses_an_owned_copy_outside_its_json_path() -> None:
    foreign = (
        b'{"hooks":{"SessionStart":[{"command":"foreign-start"}],'
        b'"PreToolUse":[{"command":"foreign-tool"}]},"model":"foreign"}'
    )
    planned = plan_owned_config_install(foreign, _spec())
    extra = planned.reverse_edit.additional_spans[0]
    span = extra.owned_span()
    reformatted = b', { "command" : "saliencegate-event:PreToolUse" }'
    drifted = planned.installed_bytes.replace(span, reformatted, 1).replace(
        b'},"model"',
        b'},"foreign_copy":[{"foreign":true}' + span + b'],"model"',
        1,
    )

    assert json.loads(drifted)["foreign_copy"][1] == {"command": "saliencegate-event:PreToolUse"}
    with pytest.raises(ConfigFileError):
        remove_owned_config_edit(drifted, planned.reverse_edit)

    same_path_copy = planned.installed_bytes.replace(span, reformatted, 1).replace(
        b']},"model"',
        span + b']},"model"',
        1,
    )
    with pytest.raises(ConfigFileError):
        remove_owned_config_edit(same_path_copy, planned.reverse_edit)

    main = planned.reverse_edit
    main_span = main.owned_span()
    altered_main = main_span.replace(b"merge-v1", b"merge-v2")
    assert altered_main != main_span and len(altered_main) == len(main_span)
    main_drifted = planned.installed_bytes.replace(main_span, altered_main, 1).replace(
        b'},"model"',
        b'},"foreign_copy":[{"foreign":true}' + main_span + b'],"model"',
        1,
    )
    with pytest.raises(ConfigFileError):
        remove_owned_config_edit(main_drifted, planned.reverse_edit)


def test_single_span_model_dump_omits_composite_default_for_legacy_mac_compatibility() -> None:
    planned = plan_owned_config_install(b"{}", _spec())

    assert planned.reverse_edit.additional_spans == ()
    assert "additional_spans" not in planned.reverse_edit.model_dump(mode="json")
    assert "json_path" not in planned.reverse_edit.model_dump(mode="json")


def test_opted_in_single_json_span_binds_the_root_container() -> None:
    planned = plan_owned_config_install(
        b'{"model":"foreign"}',
        _spec(bind_json_paths=True),
    )

    assert planned.reverse_edit.json_path == ()
    assert planned.reverse_edit.model_dump(mode="json")["json_path"] == []


def test_absent_json_owned_key_keeps_whole_fragment_insertion_behavior() -> None:
    foreign = b'{\n  "model": "foreign"\n}\n'
    spec = _spec()

    planned = plan_owned_config_install(foreign, spec)
    span = b"," + spec.owned_fragment
    start = foreign.rindex(b"}")

    assert planned.installed_bytes == foreign[:start] + span + foreign[start:]
    assert planned.reverse_edit.owned_span() == span
    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign
