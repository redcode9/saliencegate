from __future__ import annotations

import importlib.metadata
import os
import stat
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate import __version__
from saliencegate.artifacts import tree as _artifact_tree
from saliencegate.artifacts.manifest import (
    MAX_ARTIFACT_COMPONENT_BYTES,
    ArtifactAttestationsComponent,
    ArtifactBudgetsComponent,
    ArtifactClassification,
    ArtifactComponent,
    ArtifactComponentName,
    ArtifactCounters,
    ArtifactDecisionsComponent,
    ArtifactDeliveriesComponent,
    ArtifactEvidenceLevel,
    ArtifactManifest,
    ArtifactOutcomesComponent,
    ArtifactRunComponent,
    ArtifactSyntheticComponent,
    CycleAttestation,
    DeliveryAttestation,
    GroundingConfigurationAttestation,
    InterventionAttestation,
    PolicyConfigurationAttestation,
    RevisionEvidence,
    RevisionSource,
    component_content_digest,
    delivery_binding_digest,
    expected_component_path,
)
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ArtifactExportError,
    ClosedTreeDescriptor,
    ClosedTreeFileSpec,
    publish_closed_tree,
)
from saliencegate.domain import (
    BudgetAmounts,
    JsonObject,
    TrustLabel,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.intervention import (
    GROUNDING_RECEIPT_VERSION,
    GroundingReceipt,
    claim_fingerprint,
)
from saliencegate.runtime.engine import ReplayRunResult
from saliencegate.security import RedactionPolicy, Redactor

_MAX_DISTRIBUTION_FILES = 10_000
_MAX_DISTRIBUTION_BYTES = 128 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 4 * 1024
_GIT_TIMEOUT_SECONDS = 3.0
_MAX_SYNTHETIC_RESPONSES = 10_000
_MAX_MANIFEST_BYTES = 1024 * 1024
_DISTRIBUTION_DIGEST_DOMAIN = "saliencegate:artifact:distribution:v1"
_DEFAULT_REDACTION_POLICY = RedactionPolicy()


class SyntheticArtifactContent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    prompt: JsonObject = Field(repr=False)
    responses: Annotated[
        tuple[JsonObject, ...],
        Field(max_length=_MAX_SYNTHETIC_RESPONSES, repr=False),
    ] = ()

    @model_validator(mode="after")
    def content_fits_one_artifact_component(self) -> Self:
        if len(canonical_json(self.model_dump(mode="json", warnings=False))) > (
            MAX_ARTIFACT_COMPONENT_BYTES // 2
        ):
            raise ValueError("synthetic artifact content exceeds its safe input bound")
        return self


def _validated_result(value: object) -> ReplayRunResult:
    result: ReplayRunResult | None = None
    try:
        if type(value) is ReplayRunResult:
            candidate = ReplayRunResult.model_validate_json(value.model_dump_json(warnings=False))
            if candidate == value:
                result = candidate
    except Exception:
        result = None
    if result is None:
        raise ArtifactExportError("replay result failed artifact-boundary validation")
    return result


def _validated_revision(value: object) -> RevisionEvidence:
    revision: RevisionEvidence | None = None
    try:
        if type(value) is RevisionEvidence:
            candidate = RevisionEvidence.model_validate_json(value.model_dump_json(warnings=False))
            if candidate == value:
                revision = candidate
    except Exception:
        revision = None
    if revision is None:
        raise ArtifactExportError("revision evidence failed artifact-boundary validation")
    return revision


def _git_environment() -> dict[str, str]:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": os.defpath,
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def _run_git(
    arguments: tuple[str, ...],
    cwd: Path,
    *,
    maximum_output: int = _MAX_GIT_OUTPUT_BYTES,
) -> tuple[int, bytes]:
    if maximum_output < 1 or maximum_output > _MAX_GIT_OUTPUT_BYTES:
        return -1, b""
    try:
        process = subprocess.Popen(
            (
                "git",
                "-c",
                "color.ui=false",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "maintenance.auto=false",
                *arguments,
            ),
            cwd=cwd,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return -1, b""
    stream = process.stdout
    if stream is None:
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=1)
        return -2, b""

    output = bytearray()
    read_failed = threading.Event()
    read_finished = threading.Event()
    output_exceeded = threading.Event()

    def read_stdout() -> None:
        try:
            while len(output) <= maximum_output:
                chunk = stream.read(min(1024, maximum_output + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > maximum_output:
                    output_exceeded.set()
                    break
        except (OSError, ValueError):
            read_failed.set()
        finally:
            with suppress(OSError):
                stream.close()
            read_finished.set()

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    timed_out = False
    while process.poll() is None:
        if output_exceeded.wait(timeout=0.01):
            with suppress(OSError):
                process.kill()
            break
        if time.monotonic() >= deadline:
            timed_out = True
            with suppress(OSError):
                process.kill()
            break
    try:
        return_code = process.wait(timeout=1)
    except (OSError, subprocess.SubprocessError):
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=1)
        return_code = -2
    if not read_finished.wait(timeout=1):
        with suppress(OSError):
            stream.close()
        return -2, b""
    captured = bytes(output[:maximum_output])
    if timed_out or read_failed.is_set():
        return -2, captured
    if output_exceeded.is_set():
        return -2, captured
    return return_code, captured


def _git_revision(source_dir: Path) -> tuple[RevisionEvidence | None, bool]:
    code, raw_commit = _run_git(
        ("rev-parse", "--verify", "--quiet", "HEAD^{commit}"),
        source_dir,
        maximum_output=128,
    )
    if code != 0:
        return None, code != -1
    try:
        commit = raw_commit.decode("ascii").strip()
    except UnicodeDecodeError:
        return None, True
    status_code, status = _run_git(
        ("status", "--porcelain=v1", "--untracked-files=normal"),
        source_dir,
        maximum_output=1,
    )
    if status_code != 0 and not status:
        return None, True
    try:
        revision = RevisionEvidence(
            source=RevisionSource.GIT,
            package_version=__version__,
            commit=commit,
            dirty_worktree=bool(status),
            distribution_digest=None,
        )
    except Exception:
        return None, True
    return revision, True


def _regular_file_bytes(path: Path, *, maximum: int) -> bytes | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            return None
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(data) > maximum
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            return None
        return data
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _distribution_digest() -> str | None:
    try:
        distribution = importlib.metadata.distribution("saliencegate")
        if distribution.version != __version__:
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    package_root = Path(__file__).resolve().parents[1]
    inventory: list[dict[str, object]] = []
    total = 0
    try:
        paths = sorted(
            (
                path
                for path in package_root.rglob("*")
                if "__pycache__" not in path.parts and path.suffix != ".pyc"
            ),
            key=lambda path: path.relative_to(package_root).as_posix(),
        )
    except OSError:
        return None
    if len(paths) > _MAX_DISTRIBUTION_FILES:
        return None
    for path in paths:
        try:
            metadata = path.lstat()
        except OSError:
            return None
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return None
        remaining = _MAX_DISTRIBUTION_BYTES - total
        data = _regular_file_bytes(path, maximum=remaining)
        if data is None:
            return None
        total += len(data)
        inventory.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "byte_count": len(data),
                "content_digest": canonical_digest(data.hex()),
            }
        )
    if not inventory:
        return None
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "installed-distribution/v1",
                "package": "saliencegate",
                "version": __version__,
                "files": inventory,
            }
        ),
        domain=_DISTRIBUTION_DIGEST_DOMAIN,
    )


def _loaded_checkout_root() -> Path | None:
    try:
        package_root = Path(__file__).resolve(strict=True).parents[1]
        source_root = package_root.parent
        root = source_root.parent
        expected_package_root = (root / "src" / "saliencegate").resolve(strict=True)
        pyproject = (root / "pyproject.toml").lstat()
        git_marker = (root / ".git").lstat()
    except (IndexError, OSError, RuntimeError):
        return None
    if (
        source_root.name != "src"
        or package_root != expected_package_root
        or not stat.S_ISREG(pyproject.st_mode)
        or not (stat.S_ISDIR(git_marker.st_mode) or stat.S_ISREG(git_marker.st_mode))
    ):
        return None
    return root


def _distribution_revision() -> RevisionEvidence:
    digest = _distribution_digest()
    if digest is not None:
        return RevisionEvidence(
            source=RevisionSource.DISTRIBUTION,
            package_version=__version__,
            commit=None,
            dirty_worktree=None,
            distribution_digest=digest,
        )
    return RevisionEvidence(
        source=RevisionSource.UNATTESTED,
        package_version=__version__,
        commit=None,
        dirty_worktree=None,
        distribution_digest=None,
    )


def discover_revision(source_dir: os.PathLike[str] | str | None = None) -> RevisionEvidence:
    if source_dir is None:
        root = _loaded_checkout_root()
        if root is None:
            return _distribution_revision()
    else:
        try:
            root = Path(os.fspath(source_dir))
        except (TypeError, ValueError, OSError):
            return _distribution_revision()
    git_revision, git_was_available = _git_revision(root)
    if git_revision is not None:
        return git_revision
    if not git_was_available:
        return _distribution_revision()
    return RevisionEvidence(
        source=RevisionSource.UNATTESTED,
        package_version=__version__,
        commit=None,
        dirty_worktree=None,
        distribution_digest=None,
    )


def _intervention_attestation(intervention: object) -> InterventionAttestation:
    from saliencegate.domain import InterventionDecision

    if type(intervention) is not InterventionDecision:
        raise ArtifactExportError("intervention failed artifact-boundary validation")
    try:
        receipt = GroundingReceipt.model_validate_json(
            canonical_json(intervention.grounding_receipt)
        )
        if (
            receipt.receipt_version != GROUNDING_RECEIPT_VERSION
            or receipt.model_call_index is None
            or receipt.model_call_digest is None
        ):
            raise ArtifactExportError("intervention failed artifact-boundary validation")
        fingerprints = tuple(claim_fingerprint(claim) for claim in intervention.claims)
        evidence_counts = tuple(len(claim.evidence) for claim in intervention.claims)
        rendered_digest = (
            None
            if intervention.rendered_text is None
            else canonical_digest(intervention.rendered_text)
        )
        return InterventionAttestation(
            intervention_id=intervention.intervention_id,
            intervention_digest=canonical_digest(intervention),
            action=intervention.action,
            delivery_target=intervention.delivery_target,
            confidence=intervention.confidence,
            reason_code=intervention.reason_code,
            ttl_steps=intervention.ttl_steps,
            grounding_version=intervention.grounding_version,
            grounding_configuration_digest=intervention.grounding_configuration_digest,
            grounding_receipt_digest=canonical_digest(intervention.grounding_receipt),
            receipt_parse_status=receipt.parse_status,
            receipt_proposal_action=receipt.proposal_action,
            receipt_requested_delivery_target=receipt.requested_delivery_target,
            receipt_model_call_index=receipt.model_call_index,
            receipt_model_call_digest=receipt.model_call_digest,
            claim_fingerprints=fingerprints,
            claim_evidence_counts=evidence_counts,
            claim_set_digest=canonical_digest(fingerprints),
            cited_memory_ids=intervention.cited_memory_ids,
            cited_event_ids=intervention.cited_event_ids,
            rendered_text_digest=rendered_digest,
            created_at=intervention.created_at,
        )
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("intervention failed artifact-boundary validation") from None


def _cycle_attestation(item: object) -> CycleAttestation:
    from saliencegate.runtime.engine import ReplayEventResult

    if type(item) is not ReplayEventResult or item.cycle is None:
        raise ArtifactExportError("cycle failed artifact-boundary validation")
    cycle = item.cycle
    delta = cycle.validated_delta
    try:
        return CycleAttestation(
            cycle_id=cycle.cycle_id,
            revision=cycle.revision,
            invocation_decision_id=cycle.invocation_decision_id,
            policy_version=cycle.policy_version,
            configuration_digest=cycle.configuration_digest,
            grounding_version=cycle.grounding_version,
            grounding_configuration_digest=cycle.grounding_configuration_digest,
            requested_delivery_target=cycle.requested_delivery_target,
            first_event_sequence=cycle.first_event_sequence,
            last_event_sequence=cycle.last_event_sequence,
            state=cycle.state,
            budget_reservation=cycle.budget_reservation,
            budget_settlement=cycle.budget_settlement,
            batch_digest=cycle.batch_digest,
            model_request_digest=item.model_request_digest,
            model_call_digests=cycle.model_call_digests,
            model_call_latencies_us=cycle.model_call_latencies_us,
            memory_creates=0 if delta is None else len(delta.creates),
            memory_updates=0 if delta is None else len(delta.updates),
            memory_invalidations=0 if delta is None else len(delta.invalidations),
            private_status_replaced=(
                delta is not None and delta.private_status_replacement is not None
            ),
            intervention=(
                None
                if cycle.intervention is None
                else _intervention_attestation(cycle.intervention)
            ),
            failure_reason=cycle.failure_reason,
        )
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("cycle failed artifact-boundary validation") from None


def _delivery_attestation(item: object) -> DeliveryAttestation:
    from saliencegate.runtime.engine import ReplayEventResult

    if type(item) is not ReplayEventResult or item.delivery is None:
        raise ArtifactExportError("delivery failed artifact-boundary validation")
    delivery = item.delivery
    values: dict[str, object] = {
        "delivery_id": delivery.delivery_id,
        "event_sequence": item.event.sequence,
        "cycle_id": delivery.cycle_id,
        "intervention_id": delivery.intervention_id,
        "rendered_text_digest": delivery.rendered_text_digest,
        "target_request_id_digest": canonical_digest(delivery.target_request_id),
        "target": delivery.target,
        "state": delivery.state,
        "attempt_count": delivery.attempt_count,
        "adapter_id_digest": canonical_digest(delivery.adapter_id),
        "adapter_deduplicates": delivery.adapter_deduplicates,
        "adapter_deduplication_guarantee": delivery.adapter_deduplication_guarantee,
        "adapter_supports_pre_action": delivery.adapter_supports_pre_action,
        "adapter_contract_version": delivery.adapter_contract_version,
        "adapter_capabilities_digest": delivery.adapter_capabilities_digest,
        "claim_id": delivery.claim_id,
        "attempt_id": delivery.attempt_id,
        "receipt_digest": (
            None if delivery.receipt is None else canonical_digest(delivery.receipt)
        ),
        "outcome": delivery.outcome,
        "reason_code": delivery.reason_code,
        "created_at": delivery.created_at,
        "updated_at": delivery.updated_at,
        "source_delivery_digest": canonical_digest(delivery),
    }
    values["binding_digest"] = delivery_binding_digest(values)
    try:
        return DeliveryAttestation.model_validate(values)
    except Exception:
        raise ArtifactExportError("delivery failed artifact-boundary validation") from None


def _sum_budgets(values: tuple[BudgetAmounts, ...]) -> BudgetAmounts:
    fields = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    )
    return BudgetAmounts(
        **{field_name: sum(getattr(value, field_name) for value in values) for field_name in fields}
    )


def _components(
    result: ReplayRunResult,
    *,
    classification: ArtifactClassification,
    synthetic_content: SyntheticArtifactContent | None,
) -> dict[ArtifactComponentName, BaseModel]:
    decisions = tuple(item.decision for item in result.events)
    cycles = tuple(_cycle_attestation(item) for item in result.events if item.cycle is not None)
    deliveries = tuple(
        _delivery_attestation(item) for item in result.events if item.delivery is not None
    )
    policy_bindings = tuple(
        PolicyConfigurationAttestation(
            policy_version=policy_version,
            configuration_digest=configuration_digest,
        )
        for policy_version, configuration_digest in sorted(
            {(item.policy_version, item.configuration_digest) for item in decisions}
        )
    )
    grounding_bindings = tuple(
        GroundingConfigurationAttestation(
            grounding_version=grounding_version,
            configuration_digest=configuration_digest,
        )
        for grounding_version, configuration_digest in sorted(
            {(cycle.grounding_version, cycle.grounding_configuration_digest) for cycle in cycles}
        )
    )
    run = ArtifactRunComponent(
        run_id=result.run_id,
        trace_digest=result.trace_digest,
        trace_attestation_mode=result.trace_attestation_mode,
        trace_event_count=result.trace_event_count,
        model_id=result.model_id,
        prompt_template_digest=result.prompt_template_digest,
        engine_configuration=result.engine_configuration,
        engine_configuration_digest=result.engine_configuration_digest,
        policy_configurations=policy_bindings,
        grounding_configurations=grounding_bindings,
        model_execution_mode=result.model_execution_mode,
        replay_id=result.replay_id,
        fixture_digest=result.fixture_digest,
        fixture_response_count=result.fixture_response_count,
        fixture_consumed_count=result.fixture_consumed_count,
        routing_digest=result.routing_digest,
        projection_digests=result.projection_digests,
        ledger_entry_count=result.ledger_entry_count,
        ledger_head=result.ledger_head,
        rebuild_equivalent=result.rebuild_equivalent,
        source_result_digest=result.result_digest,
    )
    decision_component = ArtifactDecisionsComponent(
        run_id=result.run_id,
        decisions=decisions,
        decisions_digest=result.decisions_digest,
    )
    settlements = tuple(
        cycle.budget_settlement for cycle in cycles if cycle.budget_settlement is not None
    )
    budget_component = ArtifactBudgetsComponent(
        run_id=result.run_id,
        limits=result.engine_configuration.budget_limits,
        configured_reservation=result.engine_configuration.reservation,
        cycles=cycles,
        consumed=_sum_budgets(settlements),
        budget_projection_digest=result.projection_digests.budgets.value,
    )
    delivery_component = ArtifactDeliveriesComponent(
        run_id=result.run_id,
        deliveries=deliveries,
    )
    outcome_component = ArtifactOutcomesComponent(
        run_id=result.run_id,
        outcomes=result.outcomes,
    )
    model_request_digests = tuple(
        item.model_request_digest for item in result.events if item.model_request_digest is not None
    )
    attestation_component = ArtifactAttestationsComponent(
        run_id=result.run_id,
        normalized_trace_digest=result.normalized_trace_digest,
        normalized_draft_digests=result.normalized_draft_digests,
        persisted_event_draft_digests=result.persisted_event_draft_digests,
        routing_bindings=result.routing_bindings,
        routing_digest=result.routing_digest,
        trace_record_digests=result.trace_record_digests,
        trace_expected_event_ids=result.trace_expected_event_ids,
        events_digest=result.events_digest,
        model_request_digests=model_request_digests,
        decisions_digest=result.decisions_digest,
        projection_digests=result.projection_digests,
        ledger_head=result.ledger_head,
        source_result_digest=result.result_digest,
    )
    components: dict[ArtifactComponentName, BaseModel] = {
        ArtifactComponentName.RUN: run,
        ArtifactComponentName.DECISIONS: decision_component,
        ArtifactComponentName.BUDGETS: budget_component,
        ArtifactComponentName.DELIVERIES: delivery_component,
        ArtifactComponentName.OUTCOMES: outcome_component,
        ArtifactComponentName.ATTESTATIONS: attestation_component,
    }
    if synthetic_content is not None:
        components[ArtifactComponentName.SYNTHETIC] = ArtifactSyntheticComponent(
            run_id=result.run_id,
            trace_digest=result.trace_digest,
            prompt_template_digest=result.prompt_template_digest,
            model_request_digests=model_request_digests,
            model_call_digests=tuple(
                digest for cycle in cycles for digest in cycle.model_call_digests
            ),
            prompt=synthetic_content.prompt,
            responses=synthetic_content.responses,
        )
    return components


def _assert_export_is_redacted(
    components: Mapping[ArtifactComponentName, BaseModel],
    policy: RedactionPolicy,
) -> None:
    redactor = Redactor(
        literal_secrets=policy.literal_secrets,
        structured_field_names=policy.structured_field_names,
    )
    try:
        for component in components.values():
            payload = component.model_dump(mode="json", warnings=False)
            redacted = redactor.redact_payload(payload)
            if canonical_json(redacted.payload.root) != canonical_json(payload):
                raise ArtifactExportError("user artifact contains non-redacted structural data")
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("user artifact redaction verification failed") from None


def _encode_components(
    components: Mapping[ArtifactComponentName, BaseModel],
) -> tuple[dict[str, bytes], tuple[ArtifactComponent, ...]]:
    encoded: dict[str, bytes] = {}
    descriptors: list[ArtifactComponent] = []
    for name in sorted(components, key=lambda item: item.value):
        model = components[name]
        try:
            data = canonical_json(model)
        except Exception:
            raise ArtifactExportError("artifact component serialization failed") from None
        if len(data) > MAX_ARTIFACT_COMPONENT_BYTES:
            raise ArtifactExportError("artifact component exceeds its byte limit")
        if name is ArtifactComponentName.RUN or name is ArtifactComponentName.ATTESTATIONS:
            record_count = 1
        elif name is ArtifactComponentName.DECISIONS:
            record_count = len(model.decisions)  # type: ignore[attr-defined]
        elif name is ArtifactComponentName.BUDGETS:
            record_count = len(model.cycles)  # type: ignore[attr-defined]
        elif name is ArtifactComponentName.DELIVERIES:
            record_count = len(model.deliveries)  # type: ignore[attr-defined]
        elif name is ArtifactComponentName.OUTCOMES:
            record_count = len(model.outcomes)  # type: ignore[attr-defined]
        else:
            record_count = 1
        path = expected_component_path(name)
        encoded[path] = data
        descriptors.append(
            ArtifactComponent(
                name=name,
                path=path,
                byte_count=len(data),
                record_count=record_count,
                content_digest=component_content_digest(data),
            )
        )
    return encoded, tuple(descriptors)


def _safe_destination(output: os.PathLike[str] | str) -> Path:
    if isinstance(output, bytes):
        raise ArtifactDestinationError("artifact destination must be a text path")
    try:
        destination = Path(os.fspath(output))
    except (TypeError, ValueError, OSError):
        raise ArtifactDestinationError("artifact destination failed validation") from None
    if destination.name in ("", ".", ".."):
        raise ArtifactDestinationError("artifact destination failed validation")
    return destination


def _replay_tree_descriptor(
    manifest: ArtifactManifest,
) -> ClosedTreeDescriptor[ArtifactManifest, ArtifactComponentName]:
    files = tuple(
        ClosedTreeFileSpec(
            key=component.name,
            name=component.path,
            maximum_bytes=MAX_ARTIFACT_COMPONENT_BYTES,
            expected_bytes=component.byte_count,
        )
        for component in sorted(manifest.components, key=lambda item: item.path)
    )
    return ClosedTreeDescriptor(
        manifest=manifest,
        manifest_name="manifest.json",
        manifest_digest=manifest.manifest_digest,
        replacement_key=str(manifest.run_id),
        files=files,
    )


def _parse_replay_manifest(
    data: bytes,
) -> ClosedTreeDescriptor[ArtifactManifest, ArtifactComponentName]:
    manifest = ArtifactManifest.model_validate_json(data)
    return _replay_tree_descriptor(manifest)


def _validate_replay_tree(
    path: Path,
    expected_digest: str | None,
) -> ClosedTreeDescriptor[ArtifactManifest, ArtifactComponentName]:
    from saliencegate.artifacts.validate import load_validated_artifact

    loaded = load_validated_artifact(
        path / "manifest.json",
        expected_manifest_digest=expected_digest,
    )
    return _replay_tree_descriptor(loaded.manifest)


def _publish(
    destination: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool,
) -> None:
    publish_closed_tree(
        destination,
        files,
        manifest_name="manifest.json",
        maximum_manifest_bytes=_MAX_MANIFEST_BYTES,
        parse_manifest=_parse_replay_manifest,
        validate_tree=_validate_replay_tree,
        replace=replace,
    )


# Compatibility seams for filesystem fault-injection tests that predate the
# schema-neutral closed-tree writer. Production uses publish_closed_tree above.
_PathIdentity = _artifact_tree._PathIdentity
_write_file = _artifact_tree._write_file
_remove_owned_directory = _artifact_tree._remove_owned_directory
_remove_owned_staging = _artifact_tree._remove_owned_staging
_unlink_owned_regular = _artifact_tree._unlink_owned_regular
_read_replacement_marker = _artifact_tree._read_replacement_marker


@contextmanager
def _destination_lock(destination: Path, parent: Path) -> Iterator[None]:
    with _artifact_tree._destination_lock(destination, parent):
        yield


def _replacement_marker_bytes(
    destination: Path,
    original_metadata: os.stat_result,
    replacement_metadata: os.stat_result,
    *,
    run_id: UUID,
    original_manifest_digest: str,
    replacement_manifest_digest: str,
) -> bytes:
    return _artifact_tree._replacement_marker_bytes(
        destination,
        original_metadata,
        replacement_metadata,
        replacement_key=str(run_id),
        original_manifest_digest=original_manifest_digest,
        replacement_manifest_digest=replacement_manifest_digest,
    )


def _authorized_replace_target(
    destination: Path,
    metadata: os.stat_result,
    expected_run_id: UUID,
) -> object:
    return _artifact_tree._authorized_replace_target(
        destination,
        metadata,
        str(expected_run_id),
        validate_tree=_validate_replay_tree,
        manifest_name="manifest.json",
        maximum_manifest_bytes=_MAX_MANIFEST_BYTES,
    )


def export_replay_artifact(
    result: ReplayRunResult,
    output: os.PathLike[str] | str,
    *,
    classification: ArtifactClassification = ArtifactClassification.USER_REDACTED,
    evidence_level: ArtifactEvidenceLevel = ArtifactEvidenceLevel.EXPLORATORY,
    revision: RevisionEvidence | None = None,
    synthetic_content: SyntheticArtifactContent | None = None,
    redaction_policy: RedactionPolicy = _DEFAULT_REDACTION_POLICY,
    replace: bool = False,
    source_dir: os.PathLike[str] | str | None = None,
) -> ArtifactManifest:
    """Export a deterministic, minimized replay artifact through an atomic sibling rename."""

    validated_result = _validated_result(result)
    if type(classification) is not ArtifactClassification:
        raise ArtifactExportError("artifact classification failed validation")
    if type(evidence_level) is not ArtifactEvidenceLevel:
        raise ArtifactExportError("artifact evidence level failed validation")
    if type(redaction_policy) is not RedactionPolicy:
        raise ArtifactExportError("artifact redaction policy failed validation")
    if type(replace) is not bool:
        raise ArtifactExportError("artifact replace flag failed validation")
    if synthetic_content is not None and type(synthetic_content) is not SyntheticArtifactContent:
        raise ArtifactExportError("synthetic content failed artifact-boundary validation")
    if classification is ArtifactClassification.SYNTHETIC_RAW:
        if synthetic_content is None or any(
            item.event.trust_label is not TrustLabel.SYNTHETIC_FIXTURE
            for item in validated_result.events
        ):
            raise ArtifactExportError(
                "raw synthetic content requires an explicitly synthetic replay"
            )
    elif synthetic_content is not None:
        raise ArtifactExportError("synthetic content requires raw synthetic classification")
    destination = _safe_destination(output)
    provenance = (
        discover_revision(source_dir) if revision is None else _validated_revision(revision)
    )
    try:
        component_models = _components(
            validated_result,
            classification=classification,
            synthetic_content=synthetic_content,
        )
        if classification is not ArtifactClassification.SYNTHETIC_RAW:
            _assert_export_is_redacted(component_models, redaction_policy)
        encoded, descriptors = _encode_components(component_models)
        counters = ArtifactCounters(
            events=validated_result.trace_event_count,
            decisions=len(validated_result.events),
            invoked=sum(item.decision.invoke for item in validated_result.events),
            cycles=sum(item.cycle is not None for item in validated_result.events),
            model_calls=sum(
                1 for item in validated_result.events if item.model_request_digest is not None
            ),
            deliveries=sum(item.delivery is not None for item in validated_result.events),
            delivered=sum(
                item.delivery is not None and item.delivery.state.value == "delivered"
                for item in validated_result.events
            ),
            outcomes=len(validated_result.outcomes),
        )
        manifest = ArtifactManifest.create(
            classification=classification,
            evidence_level=evidence_level,
            run_id=validated_result.run_id,
            revision=provenance,
            engine_configuration_digest=validated_result.engine_configuration_digest,
            trace_digest=validated_result.trace_digest,
            model_id=validated_result.model_id,
            replay_id=validated_result.replay_id,
            prompt_template_digest=validated_result.prompt_template_digest,
            result_digest=validated_result.result_digest,
            components=descriptors,
            counters=counters,
        )
        if classification is not ArtifactClassification.SYNTHETIC_RAW:
            _assert_export_is_redacted(
                {ArtifactComponentName.RUN: manifest},
                redaction_policy,
            )
        encoded["manifest.json"] = canonical_json(manifest)
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("artifact construction failed") from None
    _publish(destination, encoded, replace=replace)
    return manifest


__all__ = [
    "ArtifactDestinationError",
    "ArtifactExistsError",
    "ArtifactExportError",
    "SyntheticArtifactContent",
    "discover_revision",
    "export_replay_artifact",
]
