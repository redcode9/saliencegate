from __future__ import annotations

import pytest
from pydantic import ValidationError

import saliencegate.shadow.config as config_module
from saliencegate.domain import SignalType
from saliencegate.shadow import ShadowConfig, ShadowConfigurationError
from saliencegate.shadow.config import ShadowApplicability, ShadowDetectorSpec
from saliencegate.shadow.inputs import ShadowInputKind


@pytest.mark.parametrize(
    "values",
    (
        {
            "signal_type": SignalType.TEST_FAILURE,
            "detector_version": "invalid detector version",
            "repetition_window_events": None,
        },
        {
            "signal_type": SignalType.REPEATED_ACTION,
            "detector_version": "repeated-action/v1",
            "repetition_window_events": True,
        },
        {
            "signal_type": SignalType.REPEATED_ACTION,
            "detector_version": "repeated-action/v1",
            "repetition_window_events": None,
        },
        {
            "signal_type": SignalType.TEST_FAILURE,
            "detector_version": "test-failure/v1",
            "repetition_window_events": 8,
        },
    ),
)
def test_detector_specs_reject_noncanonical_versions_windows_and_pairings(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ShadowDetectorSpec.model_validate(values)


def test_applicability_rejects_duplicate_signal_types() -> None:
    with pytest.raises(ValidationError, match="not unique"):
        ShadowApplicability(
            input_kind=ShadowInputKind.ACTION,
            applicable_signal_types=(
                SignalType.REPEATED_ACTION,
                SignalType.REPEATED_ACTION,
            ),
        )


def test_reference_builder_sanitizes_internal_profile_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_profile() -> tuple[ShadowDetectorSpec, ...]:
        raise RuntimeError("fixture-secret-profile-detail")

    monkeypatch.setattr(config_module, "_reference_detector_specs", fail_profile)
    with pytest.raises(ShadowConfigurationError) as captured:
        ShadowConfig.reference()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)


def test_config_safety_probe_fails_closed_on_missing_internal_state() -> None:
    config = ShadowConfig.reference()
    del config.__dict__["schema_version"]

    assert not config_module._config_is_safe(config)
    with pytest.raises(ShadowConfigurationError):
        config_module.validate_shadow_config(config)


def test_config_validator_rejects_validation_copy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ShadowConfig.reference()
    drifted = config.model_copy(update={"detector_profile_digest": "0" * 64})

    def return_drifted(_cls: type[ShadowConfig], _value: object) -> ShadowConfig:
        return drifted

    monkeypatch.setattr(
        ShadowConfig,
        "model_validate",
        classmethod(return_drifted),
    )
    with pytest.raises(ShadowConfigurationError):
        config_module.validate_shadow_config(config)


def test_extractor_builder_sanitizes_installed_detector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ShadowConfig.reference()

    def fail_extractor(_detectors: object) -> object:
        raise RuntimeError("fixture-secret-detector-detail")

    monkeypatch.setattr(config_module, "DeterministicSignalExtractor", fail_extractor)
    with pytest.raises(ShadowConfigurationError) as captured:
        config_module.build_shadow_extractor(config)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)


def test_extractor_builder_rejects_installed_profile_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ShadowConfig.reference()

    class DriftedDetector:
        signal_type = SignalType.REPEATED_ACTION
        detector_version = "repeated-action/drifted-v1"

    monkeypatch.setattr(config_module, "validate_shadow_config", lambda _value: config)
    monkeypatch.setattr(
        config_module,
        "RepeatedActionDetector",
        lambda _configuration: DriftedDetector(),
    )
    with pytest.raises(ShadowConfigurationError):
        config_module.build_shadow_extractor(config)
