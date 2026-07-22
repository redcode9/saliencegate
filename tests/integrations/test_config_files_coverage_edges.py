"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import saliencegate.integrations.config_files as config
from saliencegate.integrations.config_files import (
    ConfigFileError,
    ConfigSyntax,
    OwnedConfigReverseEdit,
    OwnedConfigSpan,
    OwnedConfigSpec,
    TomlBooleanConstraint,
)

MARKER = "saliencegate-owned:config-coverage-v1"


def _opaque_spec() -> OwnedConfigSpec:
    return OwnedConfigSpec(
        syntax=ConfigSyntax.OPAQUE_TEXT,
        marker=MARKER,
        owned_fragment=f"# {MARKER}\ncommand = saliencegate-capture-hook\n".encode(),
    )


def _toml_spec(*, expected: bool = True) -> OwnedConfigSpec:
    return OwnedConfigSpec(
        syntax=ConfigSyntax.TOML_DOCUMENT,
        marker=MARKER,
        owned_fragment=(f"\n# {MARKER}\n[saliencegate]\nenabled = true\n".encode()),
        toml_boolean_constraints=(
            TomlBooleanConstraint(path=("provider", "enabled"), expected=expected),
        ),
    )


def _json_spec(*, bind_json_paths: bool = False) -> OwnedConfigSpec:
    return OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        bind_json_paths=bind_json_paths,
        owned_fragment=(
            b'"hooks":{"SessionStart":[{"command":"'
            + MARKER.encode()
            + b'"}],"PreToolUse":[{"command":"saliencegate-capture-hook"}]}'
        ),
    )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"duplicate":1,"duplicate":2}',
        b'{"constant":NaN}',
        b"{not-json}",
    ),
)
def test_strict_json_rejects_duplicates_constants_and_syntax(payload: bytes) -> None:
    with pytest.raises(ConfigFileError):
        config._strict_json_loads(payload)


@pytest.mark.parametrize(
    ("payload", "start", "expected"),
    (
        (b'"escaped\\"quote"', 0, 16),
        (b"  scalar,", 0, 8),
        (b'[{"nested":[1,{"key":"value"}]}],tail', 0, 32),
        (b'{"key":[1,2]} trailing', 0, 13),
    ),
)
def test_json_value_scanner_handles_strings_scalars_and_nested_values(
    payload: bytes,
    start: int,
    expected: int,
) -> None:
    assert config._json_value_end(payload, start) == expected


@pytest.mark.parametrize(
    ("function", "args"),
    (
        (config._json_string_end, (b"not-a-string", 0)),
        (config._json_string_end, (b'"unterminated', 0)),
        (config._json_value_end, (b"   ", 0)),
        (config._json_value_end, (b"]", 0)),
        (config._json_value_end, (b"[{]", 0)),
        (config._json_value_end, (b"[[", 0)),
        (config._json_object_member_value_span, (b"[]", "key")),
        (config._json_object_member_value_span, (b'{"key" 1}', "key")),
        (config._json_object_member_value_span, (b'{"first":1 x}', "absent")),
        (config._json_path_value_span, (b'{"outer":{}}', ("outer", "missing"))),
    ),
)
def test_json_scanners_fail_closed_on_truncation_and_wrong_delimiters(
    function: object,
    args: tuple[object, ...],
) -> None:
    with pytest.raises(ConfigFileError):
        function(*args)  # type: ignore[operator]


def test_json_semantics_and_string_identity_walk_are_type_strict() -> None:
    assert config._json_insertion_semantics(b',"key":"value"', {}) == {"key": "value"}
    assert config._json_insertion_semantics(b',{"key":"value"}', []) == [{"key": "value"}]
    assert config._json_string_values({"nested": ["first", {"value": "second"}, 7, None]}) == (
        "first",
        "second",
    )
    for span, container in ((b"", []), (b"", {}), (b'"value"', "scalar")):
        with pytest.raises(ConfigFileError):
            config._json_insertion_semantics(span, container)


def test_direct_json_span_location_ignores_nested_and_quoted_copies() -> None:
    span = b',"owned":"value"'
    document = b'{"quoted":",\\"owned\\":\\"value\\"","nested":{' + span[1:] + b"}" + span + b"}"

    assert config._direct_json_span_offsets(document, span) == (document.rindex(span),)
    with pytest.raises(ConfigFileError):
        config._direct_json_span_offsets(b"[" + b"x" * 65 + b"]", b"x")
    with pytest.raises(ConfigFileError):
        config._direct_json_span_offsets(b'{"unterminated":"value}', b"value")


def test_config_models_reject_noncanonical_fragments_spans_and_reverse_edits() -> None:
    invalid_specs = (
        {
            "syntax": ConfigSyntax.OPAQUE_TEXT,
            "marker": MARKER,
            "owned_fragment": b"marker absent",
        },
        {
            "syntax": ConfigSyntax.OPAQUE_TEXT,
            "marker": MARKER,
            "owned_fragment": MARKER.encode() + b"\x00",
        },
        {
            "syntax": ConfigSyntax.OPAQUE_TEXT,
            "marker": MARKER,
            "owned_fragment": MARKER.encode() + b"\xff",
        },
        {
            "syntax": ConfigSyntax.JSON_OBJECT,
            "marker": MARKER,
            "owned_fragment": b'"one":"' + MARKER.encode() + b'","two":2',
        },
        {
            "syntax": ConfigSyntax.OPAQUE_TEXT,
            "marker": MARKER,
            "owned_fragment": MARKER.encode(),
            "bind_json_paths": True,
        },
        {
            "syntax": ConfigSyntax.OPAQUE_TEXT,
            "marker": MARKER,
            "owned_fragment": MARKER.encode(),
            "toml_boolean_constraints": (TomlBooleanConstraint(path=("enabled",), expected=True),),
        },
    )
    for payload in invalid_specs:
        with pytest.raises((ValidationError, ConfigFileError)):
            OwnedConfigSpec.model_validate(payload)

    with pytest.raises(ValidationError):
        OwnedConfigSpan(
            start=0,
            end=4,
            owned_span_digest=hashlib.sha256(b"data").hexdigest(),
            owned_span_base64="!!!!",
        )

    plan = config.plan_owned_config_install(b"{}", _json_spec())
    payload = plan.reverse_edit.model_dump(mode="python")
    payload["preimage_digest"] = "0" * 64
    forged = OwnedConfigReverseEdit.model_construct(**payload)
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(plan.installed_bytes, forged)
    assert repr(plan.reverse_edit) == "OwnedConfigReverseEdit(<redacted>)"


@pytest.mark.parametrize(
    ("current", "suffix"),
    (
        (None, b""),
        (b"foreign=true", b"\n"),
        (b"foreign=true\n", b""),
        (b"foreign=true\r", b""),
    ),
)
def test_opaque_install_and_removal_preserve_foreign_bytes(
    current: bytes | None,
    suffix: bytes,
) -> None:
    plan = config.plan_owned_config_install(current, _opaque_spec())
    original = b"" if current is None else current
    assert plan.installed_bytes == original + suffix + _opaque_spec().owned_fragment
    restored = config.remove_owned_config_edit(plan.installed_bytes, plan.reverse_edit)
    assert restored == current
    assert config._owned_config_edit_matches_spec(plan.reverse_edit, _opaque_spec())


def test_opaque_drift_removes_one_unique_owned_span_and_rejects_duplicates() -> None:
    plan = config.plan_owned_config_install(b"foreign=true\n", _opaque_spec())
    drifted = b"new-foreign=true\n" + plan.installed_bytes
    assert config.remove_owned_config_edit(drifted, plan.reverse_edit) == (
        b"new-foreign=true\nforeign=true\n"
    )
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(drifted + plan.reverse_edit.owned_span(), plan.reverse_edit)


def test_toml_constraints_and_drift_removal_are_semantic_and_byte_preserving() -> None:
    current = b"[provider]\nenabled = true\nforeign = 'kept'\n"
    plan = config.plan_owned_config_install(current, _toml_spec())
    assert config.remove_owned_config_edit(plan.installed_bytes, plan.reverse_edit) == current
    drifted = b"# later foreign comment\n" + plan.installed_bytes
    assert config.remove_owned_config_edit(drifted, plan.reverse_edit) == (
        b"# later foreign comment\n" + current
    )
    assert config._owned_config_edit_matches_spec(plan.reverse_edit, _toml_spec())

    for malformed in (
        b"[provider]\nenabled = false\n",
        b"[provider]\nenabled = 'true'\n",
    ):
        with pytest.raises(ConfigFileError):
            config.plan_owned_config_install(malformed, _toml_spec())
    missing = b"[unrelated]\nvalue = true\n"
    assert config.plan_owned_config_install(missing, _toml_spec()).installed_bytes.startswith(
        missing
    )


@pytest.mark.parametrize(
    "current",
    (
        "not-bytes",
        b"x" * (config.MAX_PROVIDER_CONFIG_BYTES + 1),
        MARKER.encode(),
    ),
)
def test_install_plan_rejects_wrong_types_oversize_and_existing_marker(current: object) -> None:
    with pytest.raises(ConfigFileError):
        config.plan_owned_config_install(current, _opaque_spec())  # type: ignore[arg-type]


def test_json_removal_handles_following_and_preceding_comma_candidates() -> None:
    spec = OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b'"owned":"' + MARKER.encode() + b'"',
    )
    for current, expected in (
        (b'{"foreign":1}', b'{"before":0,"foreign":1}'),
        (b"{}", b'{"before":0}'),
    ):
        plan = config.plan_owned_config_install(current, spec)
        drifted = plan.installed_bytes.replace(b"{", b'{"before":0,', 1)
        assert config.remove_owned_config_edit(drifted, plan.reverse_edit) == expected


def test_posix_config_publication_is_compare_and_swap_and_cleans_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "config"
    parent.mkdir(mode=0o700)
    path = (parent / "settings.json").resolve()

    assert config.read_config_bytes(path) is None
    config.publish_config_bytes(path, expected=None, data=b"{}")
    assert config.read_config_bytes(path) == b"{}"
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(path, expected=b"stale", data=b'{"next":1}')
    config.publish_config_bytes(path, expected=b"{}", data=b'{"next":1}')
    with pytest.raises(ConfigFileError):
        config.delete_config_bytes(path, expected=b"stale")
    config.delete_config_bytes(path, expected=b'{"next":1}')
    assert config.read_config_bytes(path) is None

    real_write = config.os.write
    monkeypatch.setattr(config.os, "write", lambda _descriptor, _data: 0)
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(path, expected=None, data=b"data")
    monkeypatch.setattr(config.os, "write", real_write)
    assert not tuple(parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("path_factory", "expected", "data"),
    (
        (lambda root: Path("relative.json"), None, b"{}"),
        (lambda root: root / ".." / "escape.json", None, b"{}"),
        (lambda root: root / "config.json", "not-bytes", b"{}"),
        (lambda root: root / "config.json", None, "not-bytes"),
        (lambda root: root / "config.json", None, b""),
    ),
)
def test_config_publication_rejects_invalid_boundary_arguments(
    tmp_path: Path,
    path_factory: object,
    expected: object,
    data: object,
) -> None:
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(
            path_factory(tmp_path),  # type: ignore[operator]
            expected=expected,  # type: ignore[arg-type]
            data=data,  # type: ignore[arg-type]
        )


class _Authorization:
    def __init__(self) -> None:
        self.revalidations = 0

    def revalidate(self) -> None:
        self.revalidations += 1


class _WindowsConfigOperations:
    def __init__(self, *, data: bytes | None, published: bytes | None = None) -> None:
        self.data = data
        self.published = published
        self.authorization = _Authorization()
        self.deleted = False

    def inspect_path(self, _path: PureWindowsPath) -> object | None:
        return None if self.deleted or self.data is None else object()

    def read_managed_file(self, _path: PureWindowsPath, *, maximum_bytes: int) -> object:
        assert maximum_bytes == config.MAX_PROVIDER_CONFIG_BYTES
        if self.data is None:
            raise FileNotFoundError
        return SimpleNamespace(data=self.data, authorization=self.authorization)

    def publish_managed_file(self, _path: PureWindowsPath, data: bytes, **kwargs: object) -> object:
        validator = kwargs["validate_replacement"]
        if validator is not None and not validator(self.data):  # type: ignore[operator]
            raise RuntimeError("replacement rejected")
        validate_published = kwargs["validate_published"]
        result = data if self.published is None else self.published
        assert validate_published(result) is (result == data)  # type: ignore[operator]
        self.data = result
        return SimpleNamespace(data=result)

    def delete_authorized_file(self, authorization: object) -> None:
        assert authorization is self.authorization
        self.deleted = True


def test_windows_config_boundary_uses_authorized_stable_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "settings.json").resolve()
    authorization = _Authorization()
    operations = _WindowsConfigOperations(data=b"old")
    monkeypatch.setattr(config.os, "name", "nt")
    monkeypatch.setattr(config, "NativeWindowsSecurityOperations", lambda: operations)
    monkeypatch.setattr(
        config, "authorize_windows_managed_path", lambda *args, **kwargs: authorization
    )

    assert config.read_config_bytes(path) == b"old"
    config.publish_config_bytes(path, expected=b"old", data=b"new")
    assert operations.data == b"new"
    config.delete_config_bytes(path, expected=b"new")
    assert operations.deleted is True
    assert authorization.revalidations >= 3


def test_windows_config_boundary_detects_absence_races_and_bad_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "settings.json").resolve()
    authorization = _Authorization()
    absent = _WindowsConfigOperations(data=None)
    monkeypatch.setattr(config.os, "name", "nt")
    monkeypatch.setattr(config, "NativeWindowsSecurityOperations", lambda: absent)
    monkeypatch.setattr(
        config, "authorize_windows_managed_path", lambda *args, **kwargs: authorization
    )
    assert config.read_config_bytes(path) is None

    bad = _WindowsConfigOperations(data=b"old", published=b"wrong")
    monkeypatch.setattr(config, "NativeWindowsSecurityOperations", lambda: bad)
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(path, expected=b"old", data=b"new")
    with pytest.raises(ConfigFileError):
        config.delete_config_bytes(path, expected=b"stale")


def test_generic_snapshot_path_reads_missing_and_stable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "generic"
    parent.mkdir(mode=0o700)
    path = (parent / "config.txt").resolve()
    monkeypatch.setattr(config.os, "name", "generic")
    assert config.read_config_bytes(path) is None
    path.write_bytes(b"stable")
    path.chmod(0o600)
    assert config.read_config_bytes(path) == b"stable"


def test_span_and_reverse_edit_validators_reject_digest_order_overlap_and_path_mismatches() -> None:
    span = b"owned-span"
    valid = config._owned_span(10, span, json_path=("root",))
    bad_payloads = (
        {**valid.model_dump(mode="python"), "end": 25},
        {**valid.model_dump(mode="python"), "owned_span_digest": "0" * 64},
    )
    for payload in bad_payloads:
        with pytest.raises(ValidationError):
            OwnedConfigSpan.model_validate(payload)

    broken = OwnedConfigSpan.model_construct(
        start=0,
        end=4,
        owned_span_digest="0" * 64,
        owned_span_base64="!!!!",
        json_path=None,
    )
    with pytest.raises(ConfigFileError):
        broken.owned_span()

    composite = config.plan_owned_config_install(
        b'{"hooks":{"SessionStart":[],"PreToolUse":[]}}',
        _json_spec(),
    ).reverse_edit
    payload = composite.model_dump(mode="python")
    payload["additional_spans"] = tuple(reversed(payload["additional_spans"]))
    if len(payload["additional_spans"]) > 1:
        with pytest.raises(ValidationError):
            OwnedConfigReverseEdit.model_validate(payload)

    extra = composite.additional_spans[0]
    overlap = extra.model_copy(update={"start": composite.start, "end": composite.end})
    overlap_payload = composite.model_dump(mode="python")
    overlap_payload["additional_spans"] = (overlap,)
    with pytest.raises(ValidationError):
        OwnedConfigReverseEdit.model_validate(overlap_payload)

    opaque = config.plan_owned_config_install(b"foreign\n", _opaque_spec()).reverse_edit
    opaque_payload = opaque.model_dump(mode="python")
    opaque_payload["json_path"] = ("invalid",)
    with pytest.raises(ValidationError):
        OwnedConfigReverseEdit.model_validate(opaque_payload)


def test_reverse_edit_and_spec_helpers_normalize_invalid_objects() -> None:
    with pytest.raises(ConfigFileError):
        config._reverse_edit(
            syntax=ConfigSyntax.OPAQUE_TEXT,
            target_existed=False,
            original=b"",
            installed=b"fragment",
            marker=MARKER,
            spans=((0, b"fragment", ()),),
            bind_json_paths=False,
        )
    with pytest.raises(ConfigFileError):
        config._validated_spec(object())


def test_json_location_and_semantic_removal_reject_scalar_overlap_and_owned_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar = config._owned_span(0, b"1")
    with pytest.raises(ConfigFileError):
        config._locate_json_owned_spans(b"1", (scalar,))

    left = config._owned_span(1, b'"a":1')
    right = config._owned_span(3, b'a":1', json_path=None)
    monkeypatch.setattr(config, "_direct_json_span_offsets", lambda *_args: (1,))
    with pytest.raises(ConfigFileError):
        config._locate_json_owned_spans(b'{"a":1}', (left, right))

    identity = "saliencegate-owned-identity-value"
    with pytest.raises(ConfigFileError):
        config._validate_json_owned_semantics_removed(
            b'{"items":[{"id":"' + identity.encode() + b'"}]}',
            ((0, b"", ("items",), [{"id": identity}]),),
        )
    with pytest.raises(ConfigFileError):
        config._validate_json_owned_semantics_removed(
            b'{"items":{"owned":"' + identity.encode() + b'"}}',
            ((0, b"", ("items",), {"owned": identity}),),
        )
    with pytest.raises(ConfigFileError):
        config._validate_json_owned_semantics_removed(
            b'{"items":7}',
            ((0, b"", ("items",), [identity]),),
        )


def test_owned_edit_matcher_rejects_wrong_types_syntax_marker_and_path_contracts() -> None:
    plan = config.plan_owned_config_install(b"{}", _json_spec(bind_json_paths=True))
    assert not config._owned_config_edit_matches_spec(object(), _json_spec())  # type: ignore[arg-type]
    assert not config._owned_config_edit_matches_spec(plan.reverse_edit, _opaque_spec())
    unbound = plan.reverse_edit.model_copy(update={"json_path": None})
    assert not config._owned_config_edit_matches_spec(
        unbound,
        _json_spec(bind_json_paths=True),
    )

    malformed = OwnedConfigSpec.model_construct(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b"not-json",
        toml_boolean_constraints=(),
        bind_json_paths=False,
    )
    assert not config._owned_config_edit_matches_spec(plan.reverse_edit, malformed)


def test_json_object_insertion_error_partition_is_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (b"[]", b'"owned":"' + MARKER.encode() + b'"'),
        (b"{}", b'"one":"' + MARKER.encode() + b'","two":2'),
        (b'{"hooks":7}', _json_spec().owned_fragment),
    )
    for current, fragment in cases:
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(current, fragment)

    original_span = config._json_object_member_value_span
    monkeypatch.setattr(config, "_json_object_member_value_span", lambda *_args: None)
    with pytest.raises(ConfigFileError):
        config._json_object_insertion(
            b'{"hooks":{}}',
            _json_spec().owned_fragment,
        )
    monkeypatch.setattr(config, "_json_object_member_value_span", original_span)

    with pytest.raises(ConfigFileError):
        config._json_object_insertion(
            b'{"hooks":{"SessionStart":7}}',
            _json_spec().owned_fragment,
        )

    original_canonical = config.canonical_json
    monkeypatch.setattr(config, "canonical_json", lambda _value: b"{}")
    with pytest.raises(ConfigFileError):
        config._json_object_insertion(
            b'{"hooks":{"SessionStart":[]}}',
            _json_spec().owned_fragment,
        )
    monkeypatch.setattr(config, "canonical_json", original_canonical)

    owned = {
        f"Event{index}": [{"command": MARKER if index == 0 else f"owned-{index}"}]
        for index in range(17)
    }
    fragment = b'"hooks":' + config.canonical_json(owned)
    current = b'{"hooks":' + config.canonical_json({key: [] for key in owned}) + b"}"
    with pytest.raises(ConfigFileError):
        config._json_object_insertion(current, fragment)


def test_plan_and_remove_fault_partition_rejects_utf8_toml_duplicates_and_forged_offsets() -> None:
    with pytest.raises(ConfigFileError):
        config.plan_owned_config_install(b"\xff", _opaque_spec())
    with pytest.raises(ConfigFileError):
        config._validate_toml_boolean_constraints(
            {"provider": 7},
            (TomlBooleanConstraint(path=("provider", "enabled"), expected=True),),
        )

    plan = config.plan_owned_config_install(b"{}", _json_spec())
    shifted = plan.reverse_edit.model_copy(
        update={"start": plan.reverse_edit.start + 1, "end": plan.reverse_edit.end + 1}
    )
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(plan.installed_bytes, shifted)
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(object(), plan.reverse_edit)  # type: ignore[arg-type]
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(b"[]" + plan.reverse_edit.owned_span(), plan.reverse_edit)

    toml = config.plan_owned_config_install(b"", _toml_spec())
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(
            toml.installed_bytes + toml.reverse_edit.owned_span(),
            toml.reverse_edit,
        )
    opaque = config.plan_owned_config_install(b"", _opaque_spec())
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(
            opaque.installed_bytes + opaque.reverse_edit.owned_span(),
            opaque.reverse_edit,
        )


def test_config_filesystem_faults_reject_unsafe_parents_races_and_backend_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    path = (unsafe / "config.json").resolve()
    with pytest.raises(ConfigFileError):
        config.read_config_bytes(path)

    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    target = (safe / "config.json").resolve()
    config.publish_config_bytes(target, expected=None, data=b"{}")
    real_snapshot = config._snapshot_at
    calls = 0

    def raced_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_snapshot(*args, **kwargs)
        if calls == 2:
            return config._ConfigSnapshot(
                data=result.data, identity=(0, 0, 0, 0, 0, 0), mode=result.mode
            )
        return result

    monkeypatch.setattr(config, "_snapshot_at", raced_snapshot)
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(target, expected=b"{}", data=b'{"next":1}')
    monkeypatch.setattr(config, "_snapshot_at", real_snapshot)

    monkeypatch.setattr(config.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(ConfigFileError):
        config.publish_config_bytes(target, expected=b"{}", data=b'{"next":1}')
    with pytest.raises(ConfigFileError):
        config.delete_config_bytes(target, expected=object())  # type: ignore[arg-type]


def test_remaining_model_json_and_boundary_edges_are_reached_without_weakening_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = {
        "First": [{"command": MARKER}],
        "Second": [{"command": "second-owned-command"}],
        "Third": [{"command": "third-owned-command"}],
    }
    composite_spec = OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b'"hooks":' + config.canonical_json(owned),
    )
    current = b'{"hooks":' + config.canonical_json({key: [] for key in owned}) + b"}"
    composite = config.plan_owned_config_install(current, composite_spec).reverse_edit
    payload = composite.model_dump(mode="python")
    payload["additional_spans"] = tuple(reversed(payload["additional_spans"]))
    with pytest.raises(ValidationError):
        OwnedConfigReverseEdit.model_validate(payload)

    overlapping = config._owned_span(
        composite.start,
        b'"overlap":1',
        json_path=("hooks",),
    )
    payload = composite.model_dump(mode="python")
    payload["additional_spans"] = (overlapping, *composite.additional_spans)
    payload["additional_spans"] = tuple(
        sorted(payload["additional_spans"], key=lambda item: item.start)
    )
    with pytest.raises(ValidationError):
        OwnedConfigReverseEdit.model_validate(payload)

    opaque = config.plan_owned_config_install(b"", _opaque_spec()).reverse_edit
    nonoverlap = config._owned_span(
        opaque.end + 1,
        b"foreign",
        json_path=("invalid",),
    )
    payload = opaque.model_dump(mode="python")
    payload["additional_spans"] = (nonoverlap,)
    with pytest.raises(ValidationError):
        OwnedConfigReverseEdit.model_validate(payload)

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_strict_json_loads", lambda _value: 7)
        with pytest.raises(ConfigFileError):
            config._json_object_member_value_span(b'{"key":1}', "key")

    scalar_spec = OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b'"owned":"' + MARKER.encode() + b'"',
    )
    scalar_edit = config.plan_owned_config_install(b"{}", scalar_spec).reverse_edit
    assert config._owned_config_edit_matches_spec(scalar_edit, scalar_spec)

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_direct_json_span_offsets", lambda *_args: (0,))
        with pytest.raises(ConfigFileError):
            config._locate_json_owned_spans(b"1", (config._owned_span(0, b"1"),))

    oversized_current = b"x" * config.MAX_PROVIDER_CONFIG_BYTES
    with pytest.raises(ConfigFileError):
        config.plan_owned_config_install(oversized_current, _opaque_spec())
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(b"{}", object())  # type: ignore[arg-type]

    unsafe = tmp_path / "generic-unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        with pytest.raises(ConfigFileError):
            config.read_config_bytes((unsafe / "config.json").resolve())
        with pytest.raises(ConfigFileError):
            config._open_config_parent((unsafe / "config.json").resolve())


def test_remaining_json_member_and_matcher_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert config._json_object_member_value_span(b'{"first":1,"second":2}', "second") == (
        20,
        21,
    )
    assert config._json_object_member_value_span(b'{"first":1}', "absent") is None

    edit = config.plan_owned_config_install(b"{}", _json_spec()).reverse_edit
    malformed = OwnedConfigSpec.model_construct(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b'"one":"' + MARKER.encode() + b'","two":2',
        toml_boolean_constraints=(),
        bind_json_paths=False,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_validated_spec", lambda _value: malformed)
        assert not config._owned_config_edit_matches_spec(edit, malformed)

    nested = config.plan_owned_config_install(
        b'{"hooks":{"SessionStart":[],"PreToolUse":[]}}',
        _json_spec(),
    ).reverse_edit
    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_json_object_member_value_span", lambda *_args: None)
        assert not config._owned_config_edit_matches_spec(nested, _json_spec())


def test_json_insertion_reaches_closed_envelope_member_and_root_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment = _json_spec().owned_fragment
    original_span = config._json_object_member_value_span

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_json_object_member_value_span", lambda *_args: (9, 10))
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(b'{"hooks":{}}', fragment)

    calls = 0

    def missing_owned_span(document: bytes, member: str):
        nonlocal calls
        calls += 1
        return original_span(document, member) if calls == 1 else None

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_json_object_member_value_span", missing_owned_span)
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(b'{"hooks":{"Foreign":[]}}', fragment)

    missing_fragment = (
        b'"hooks":{"Foreign":[],"SessionStart":[{"command":"' + MARKER.encode() + b'"}]}'
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(config, "canonical_json", lambda _value: b"{}")
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(b'{"hooks":{"SessionStart":[]}}', missing_fragment)

    calls = 0

    def missing_member_span(document: bytes, member: str):
        nonlocal calls
        calls += 1
        return original_span(document, member) if calls == 1 else None

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_json_object_member_value_span", missing_member_span)
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(
                b'{"hooks":{"SessionStart":[]}}',
                fragment,
            )

    calls = 0

    def scalar_member_span(document: bytes, member: str):
        nonlocal calls
        calls += 1
        return original_span(document, member) if calls == 1 else (0, 1)

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_json_object_member_value_span", scalar_member_span)
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(
                b'{"hooks":{"SessionStart":[]}}',
                fragment,
            )

    assert config._json_object_insertion(b"  {}  ", b'"owned":1')[0] == b'  {"owned":1}  '
    with monkeypatch.context() as scoped:
        values = iter(({}, {"owned": 1}))
        scoped.setattr(config, "_strict_json_loads", lambda _value: next(values))
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(b"x", b'"owned":1')
    with monkeypatch.context() as scoped:
        values = iter(({}, {"owned": 1}))
        scoped.setattr(config, "_strict_json_loads", lambda _value: next(values))
        with pytest.raises(ConfigFileError):
            config._json_object_insertion(b"x}", b'"owned":1')


def test_plan_and_removal_reach_wrapped_ambiguous_and_syntax_count_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(
            config,
            "_opaque_insertion",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("insertion")),
        )
        with pytest.raises(ConfigFileError):
            config.plan_owned_config_install(b"", _opaque_spec())

    scalar_spec = OwnedConfigSpec(
        syntax=ConfigSyntax.JSON_OBJECT,
        marker=MARKER,
        owned_fragment=b'"owned":"' + MARKER.encode() + b'"',
    )
    scalar = config.plan_owned_config_install(b"{}", scalar_spec).reverse_edit
    with pytest.raises(ConfigFileError):
        config.remove_owned_config_edit(b'["' + MARKER.encode() + b'"]', scalar)

    span = scalar.owned_span()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            config,
            "_locate_json_owned_spans",
            lambda *_args: ((0, span, None, None),),
        )
        scoped.setattr(config, "_strict_json_loads", lambda _value: {})
        with pytest.raises(ConfigFileError):
            config.remove_owned_config_edit(span + b",", scalar)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            config,
            "_locate_json_owned_spans",
            lambda *_args: ((0, span, None, None),),
        )
        values = iter(({}, {}, []))
        scoped.setattr(config, "_strict_json_loads", lambda _value: next(values))
        scoped.setattr(config, "_validate_json_owned_semantics_removed", lambda *_args: None)
        assert config.remove_owned_config_edit(span + b",", scalar) == b","

    for plan, syntax in (
        (config.plan_owned_config_install(b"", _toml_spec()), ConfigSyntax.TOML_DOCUMENT),
        (config.plan_owned_config_install(b"", _opaque_spec()), ConfigSyntax.OPAQUE_TEXT),
    ):
        assert plan.reverse_edit.syntax is syntax
        absent = config._owned_span(0, b"absent")
        forged = plan.reverse_edit.model_copy(
            update={
                "start": absent.start,
                "end": absent.end,
                "owned_span_digest": absent.owned_span_digest,
                "owned_span_base64": absent.owned_span_base64,
            }
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                OwnedConfigReverseEdit,
                "model_validate",
                classmethod(lambda _cls, _value, _forged=forged: _forged),
            )
            with pytest.raises(ConfigFileError):
                config.remove_owned_config_edit(MARKER.encode(), plan.reverse_edit)


def test_posix_snapshot_revalidation_and_file_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not config._safe_ancestor(
        SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.getuid()),
        leaf=True,
    )

    parent = tmp_path / "snapshot"
    parent.mkdir(mode=0o700)
    path = (parent / "config.json").resolve()
    path.write_bytes(b"{}")
    path.chmod(0o600)
    descriptor, identity = config._open_config_parent(path)
    try:
        with pytest.raises(ConfigFileError):
            config._revalidate_config_parent(path, descriptor, (0, 0, 0, 0))
        with monkeypatch.context() as scoped:
            fresh = os.dup(descriptor)
            scoped.setattr(config, "_open_config_parent", lambda _path: (fresh, (0, 0, 0, 0)))
            with pytest.raises(ConfigFileError):
                config._revalidate_config_parent(path, descriptor, identity)
    finally:
        os.close(descriptor)

    path.chmod(0o666)
    with pytest.raises(ConfigFileError):
        config._snapshot(path)
    path.chmod(0o600)

    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        with pytest.raises(ConfigFileError):
            config._snapshot(parent)


def test_snapshot_read_loops_and_identity_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "reads"
    parent.mkdir(mode=0o700)
    path = (parent / "config.json").resolve()
    path.write_bytes(b"x")
    path.chmod(0o600)

    descriptor, identity = config._open_config_parent(path)
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(config.os, "read", lambda *_args: b"x" * (64 * 1024))
            with pytest.raises(ConfigFileError):
                config._snapshot_at(path, descriptor, identity)
    finally:
        os.close(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        scoped.setattr(config.os, "read", lambda *_args: b"x" * (64 * 1024))
        with pytest.raises(ConfigFileError):
            config._snapshot(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        real_identity = config._identity
        calls = 0

        def changed_identity(metadata: os.stat_result):
            nonlocal calls
            calls += 1
            value = real_identity(metadata)
            return value if calls < 3 else (*value[:-1], value[-1] + 1)

        scoped.setattr(config, "_identity", changed_identity)
        with pytest.raises(ConfigFileError):
            config._snapshot(path)


def test_read_fsync_publish_and_delete_remaining_boundary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "boundaries"
    parent.mkdir(mode=0o700)
    path = (parent / "config.json").resolve()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            config,
            "_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot")),
        )
        with pytest.raises(ConfigFileError):
            config.read_config_bytes(path)

    config._fsync_directory(parent)

    calls = 0
    real_snapshot_at = config._snapshot_at

    def wrong_published(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        result = real_snapshot_at(*args, **kwargs)
        return (
            result
            if calls < 3
            else config._ConfigSnapshot(data=b"wrong", identity=result.identity, mode=result.mode)
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_snapshot_at", wrong_published)
        with pytest.raises(ConfigFileError):
            config.publish_config_bytes(path, expected=None, data=b"new")

    path.write_bytes(b"delete")
    path.chmod(0o600)
    calls = 0

    def retained_after_delete(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        result = real_snapshot_at(*args, **kwargs)
        if calls == 2:
            return config._ConfigSnapshot(data=b"retained", identity=None, mode=0o600)
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(config, "_snapshot_at", retained_after_delete)
        with pytest.raises(ConfigFileError):
            config.delete_config_bytes(path, expected=b"delete")

    path.write_bytes(b"generic")
    path.chmod(0o600)
    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        with pytest.raises(ConfigFileError):
            config.delete_config_bytes(path, expected=b"wrong")
        config.delete_config_bytes(path, expected=b"generic")

    path.write_bytes(b"race")
    path.chmod(0o600)
    real_exists = Path.exists
    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        scoped.setattr(
            Path,
            "exists",
            lambda current: True if current == path else real_exists(current),
        )
        with pytest.raises(ConfigFileError):
            config.delete_config_bytes(path, expected=b"race")

    with monkeypatch.context() as scoped:
        scoped.setattr(config.os, "name", "generic")
        scoped.setattr(
            config,
            "_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot")),
        )
        with pytest.raises(ConfigFileError):
            config.delete_config_bytes(path, expected=b"race")
