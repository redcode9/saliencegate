# Artifact reference

SalienceGate publishes three artifact families. A replay artifact records an inspectable execution
result, but it is not a replay-input bundle. An algorithm-run artifact records one closed comparison
run. A StateDecayBench artifact records a synthetic diagnostic and the commitments needed to
reconstruct its oracle.

| Family | Producer | Layout | Validation assurance |
|---|---|---|---|
| Replay run | `saliencegate replay` or Python export API | Manifest plus six typed components | Producer-attested, digest-only grounding |
| Algorithm run | `saliencegate algorithm replay` or Python export API | Manifest plus nine typed components | Self-consistent producer attestation, or raw synthetic reprojection |
| StateDecayBench smoke | `saliencegate benchmark state-decay-smoke` | Manifest plus canonical JSONL | Deterministic synthetic oracle |

CLI reports, inspection reports, and validation reports are derived output. They are not files
inside the published artifact.

## Canonical encoding and digests

Artifact JSON is canonical UTF-8 with ordered keys, compact separators, finite numbers, and no
duplicate keys. Replay manifest and component files have no trailing newline. `smoke.jsonl`
contains exactly 32 canonical records and ends each record with a newline. Pretty-printing or
editing a file changes its committed bytes and invalidates the artifact.

Artifact digests use SHA-256 with domain separation and length framing. Consequently they are
not generally equal to `sha256sum FILE`:

- each replay component `content_digest` binds its exact bytes;
- each algorithm component `content_digest` binds its name and exact bytes in an algorithm-specific
  domain;
- `overall_content_digest` binds the ordered component descriptors;
- `manifest_digest` binds the canonical manifest excluding its own digest field;
- benchmark fixture, reconstructed oracle, overall content, and manifest use distinct domains.

`saliencegate validate --expected-digest` expects the `manifest_digest`. Supplying that digest
through a trusted channel provides an external equality anchor. The embedded digest alone proves
internal consistency, not authorship. These digests are not signatures, MACs, or encryption.

## Replay artifact layout

The normal layout is:

```text
artifact/
|-- manifest.json
|-- attestations.json
|-- budgets.json
|-- decisions.json
|-- deliveries.json
|-- outcomes.json
`-- run.json
```

Only the explicit `synthetic_raw` classification adds:

```text
`-- synthetic.json
```

The CLI never emits `synthetic_raw`; its synthetic fixture path is `synthetic_digest_only`.

| File and schema | Purpose |
|---|---|
| `run.json`, `artifact-run/v1` | Engine configuration, execution mode, frozen-fixture binding, routing, projection, ledger, rebuild, and result attestations |
| `decisions.json`, `artifact-decisions/v1` | Exactly one ordered invocation decision for each event |
| `budgets.json`, `artifact-budgets/v1` | Limits, configured reservation, consumption, terminal cycles, and intervention attestations |
| `deliveries.json`, `artifact-deliveries/v1` | Terminal delivery state, attempts, capability and binding digests, and receipt digest; never the raw receipt |
| `outcomes.json`, `artifact-outcomes/v1` | Versioned outcome measurements and unresolved status |
| `attestations.json`, `artifact-attestations/v1` | Trace, routing, event, model-request, decision, projection, ledger, and result digests |
| `synthetic.json`, `artifact-synthetic/v1` | Explicit raw synthetic prompts and responses; Python API only |

`manifest.json` uses schema `1.0`, `record_type: artifact_manifest`, and
`artifact_kind: replay_run`. Its principal fields are:

```text
classification, evidence_level, run_id, revision,
confirmatory_eligible, engine_configuration_digest,
trace_digest, model_id, replay_id, prompt_template_digest,
result_digest, components, counters,
overall_content_digest, manifest_digest
```

Every component descriptor contains its fixed name and path, byte count, record count, and
content digest. Counters contain:

```text
events, decisions, invoked, cycles, model_calls,
deliveries, delivered, outcomes
```

The schema enforces, among other constraints:

- `decisions == events` and `invoked == cycles`;
- model calls, deliveries, and outcomes cannot exceed their cycles;
- successful deliveries cannot exceed delivery attempts;
- component names and fixed paths are complete, unique, and ordered;
- component record counts match the manifest counters;
- run identifiers and lifecycle references agree across every component;
- configuration, result, ledger-head, projection, cycle, intervention, delivery, and outcome
  commitments recompute exactly.

## Classification, evidence level, and revision

Replay classification describes the permitted content boundary:

- `user_redacted` contains redacted execution evidence and never raw prompts or responses;
- `synthetic_digest_only` binds a synthetic fixture without carrying its raw model exchange;
- `synthetic_raw` permits explicit raw synthetic content in `synthetic.json` and must be selected
  through the Python API with a typed synthetic payload.

Classification is not a confidentiality guarantee. Redacted artifacts still contain task and
lifecycle metadata. Pattern-based redaction cannot recognize every domain-specific secret.

Evidence level is either `exploratory` or `confirmatory`. Revision evidence is one of:

- `git`: full commit plus clean or dirty worktree state;
- `distribution`: an installed-distribution inventory digest;
- `unattested`: package version without a revision claim.

`confirmatory_eligible` means only that revision provenance is a clean Git checkout or an
identified installed distribution. It does not set `evidence_level` and does not prove causal
utility. The replay CLI always chooses `exploratory`.

The offline algorithm-replay CLI additionally normalizes a discovered Git revision to
`dirty_worktree: true`, even when discovery initially reports a clean checkout. This conservative
choice keeps repeated output-path-independent exports byte-identical and prevents the CLI from
claiming clean provenance. The commit is retained; callers that require an exact clean/dirty
attestation can pass an explicit revision through the Python algorithm-export API.

Revision provenance participates in the replay manifest. The same execution content can
therefore retain one `overall_content_digest` while receiving different manifest digests in a
source checkout and an installed wheel. User-mode records can additionally depend on the local
installation key. Do not assume manifest-digest equality across installations.

## Replay validation

Validation performs one bounded filesystem pass:

1. open the artifact directory without following a symbolic link;
2. read and parse a bounded canonical `manifest.json`;
3. require the supported `1.0` manifest and fixed component paths;
4. require the exact file set, with no missing or extra entry;
5. reject symbolic links, hardlinks, FIFOs, non-regular files, and oversized files;
6. verify byte counts and component content digests;
7. parse every strict component schema;
8. recompute counters and internal invariants;
9. verify all cross-component bindings and grounding attestations;
10. recheck file and directory identities after reading.

The replay manifest is limited to 1 MiB. Each component is limited to 16 MiB, and record sets are
bounded at 100,000 entries. Unknown fields, unsupported versions, changed files, non-canonical
content, and invalid lifecycle combinations fail closed.

Python callers receive a specific `ArtifactValidationCode`:

```text
invalid_manifest
unsupported_version
unsafe_path
missing_component
unsafe_component
content_mismatch
invalid_component
inconsistent_counters
ungrounded_delivery
cross_component_invariant
expected_digest_mismatch
confirmatory_ineligible
```

The CLI intentionally collapses these details to exit 5 and `error: artifact validation failed`
so untrusted input values are not exposed.

`load_validated_artifact()` returns the validated manifest, run, decisions, budgets, deliveries,
outcomes, attestations, and validation report from that same pass. It never exposes the raw
synthetic component. The `inspect` command derives its minimized view from this object without
reopening files.

## Algorithm-run artifact layout

The fixed `algorithm-artifact/v1` layout is:

```text
algorithm-run/
|-- manifest.json
|-- attestations.json
|-- calls.json
|-- cycles.json
|-- decisions.json
|-- deliveries.json
|-- metrics.json
|-- outcomes.json
|-- run.json
`-- trajectory.json
```

| File and schema | Purpose |
|---|---|
| `run.json`, `algorithm-run/v1` | Closed condition, policy, prompt bundle, model profile, call policy, budget limits, execution attestation, and source-result digest |
| `trajectory.json`, `algorithm-trajectory/v1` | Synthetic trajectory identity, exact ordered event attestations, schedule, message-window attestations, and window-set digest |
| `calls.json`, `algorithm-calls/v1` | Ordered request digests, per-cycle call groups, exact call receipts, and intervention grounding binding |
| `decisions.json`, `algorithm-decisions/v1` | Exactly one ordered invocation decision for every trajectory event |
| `cycles.json`, `algorithm-cycles/v1` | Terminal cycle, budget, memory-delta, intervention, grounding, delivery-source, and boundary-evidence attestations |
| `deliveries.json`, `algorithm-deliveries/v1` | Terminal delivery records, adapter capability commitments, receipt digests, and binding digests |
| `outcomes.json`, `algorithm-outcomes/v1` | Ordered intervention outcomes committed by the ledger |
| `metrics.json`, `algorithm-metrics/v1` | Call and token metrics, final budget, and minimized final-memory attestation |
| `attestations.json`, `algorithm-attestations/v1` | Boundary, semantic and repository projection, complete ledger-chain, rebuild, and source-result attestations |

`manifest.json` uses `record_type: algorithm_artifact_manifest`,
`schema_version: algorithm-artifact/v1`, and `artifact_kind: algorithm_run`. Its principal fields
are:

```text
classification, evidence_level, run_id, revision, confirmatory_eligible,
condition_id, condition_digest, cycle_mode, trace_digest, schedule_digest,
window_digests, window_set_digest, prompt_bundle_digest, model_id,
model_profile_digest, execution, execution_digest, configuration_digest,
result_digest, components, counters, overall_content_digest, manifest_digest
```

The nine component descriptors are complete, unique, ordered, and bound to fixed paths. Algorithm
counters contain:

```text
events, scheduled_invocations, decisions, cycles, requests,
model_calls, deliveries, outcomes, ledger_entries
```

The manifest and components jointly bind the resolved condition, cycle mode, schedule and window
versions, both prompt identities, model profile, endpoint classification, runtime version,
checkpoint or model tag, quantization, sampling, tokenizer, hardware record, ordered receipts,
budgets, interventions, projections, ledger, revision, and result digest.

### Algorithm classification and evidence

`algorithm-artifact/v1` is synthetic-only. Its two classifications are:

- `synthetic_digest_only`, the CLI default, contains attestations and digests without the raw
  algorithm result;
- `synthetic_raw`, available only through the Python export API, embeds the explicit raw synthetic
  result inside `attestations.json` so validation can project it again.

There is no `user_redacted` algorithm classification. A trajectory containing any trust label
other than `synthetic_fixture` is rejected rather than relabelled. Synthetic raw content never
becomes confirmatory.

Every algorithm manifest fixes `evidence_level: exploratory` and
`confirmatory_eligible: false`; its derived confirmatory status and validation report are also
always false. A clean source revision does not change those values. Digest-only validation proves
producer-attested structural consistency, not that the source result was independently
reproduced, that cited prose is semantically entailed, or that an intervention improved task
performance.

The hardware record has only model, architecture, logical core count, memory capacity, operating
system, and operating-system version fields. Obvious emails, paths, URLs, and control characters
are rejected. Its labels remain producer-supplied and must be de-identified by the caller; this is
data minimization, not proof of anonymity.

### Algorithm validation

Algorithm validation uses the same closed-tree filesystem protections as replay validation and
additionally checks:

- all nine component digests, model self-digests, overall digest, and manifest digest;
- exact run, condition, schedule, message-window, prompt, model, tokenizer, and execution bindings;
- decision, request, call, cycle, delivery, outcome, and ledger ordering and cardinality;
- execution-mode-native call evidence and frozen replay-fixture identity;
- intervention grounding and delivery-source bindings;
- per-call policy limits, aggregate metrics, temporal budget reconciliation, final projections,
  ledger tags and head, and rebuild equivalence;
- for `synthetic_raw`, exact reprojection of every component from the embedded algorithm result.

`algorithm-validation-report/v1` returns `source_result_assurance` as
`producer_attested` for digest-only input or `recomputed_from_raw` after successful raw
reprojection. Both values require the complete structural and cross-component validation pass;
neither is a confirmatory efficacy claim. `load_validated_algorithm_artifact()` exposes a safe
validated view and deliberately omits the embedded raw result.

The algorithm manifest is limited to 1 MiB, each component to 128 MiB, ordinary record sets to
100,000 entries, and ledger attestations to 160,000 entries. Hardware text fields are limited to
256 UTF-8 bytes. The validator bounds every file before parsing, rejects non-canonical JSON and
duplicate keys, and rechecks identities after reading.

## StateDecayBench artifact layout

The runtime layout is:

```text
state-decay-smoke/
|-- manifest.json
`-- smoke.jsonl
```

The committed source fixture instead uses `smoke_manifest.json` plus `smoke.jsonl`. The different
name prevents it from being mistaken for a published runtime artifact; use the `benchmark`
command to produce the runtime layout.

The `state-decay-smoke-manifest/v1` fixes:

```text
suite_id = state-decay-smoke
suite_version = v1
generator_version = v1
oracle_version = paired-continuation-oracle/v1
seed = 20260711
scenario_count = 32
family_count = 8
intervene_count = 16
silence_count = 16
oracle_passed = 32
oracle_failed = 0
diagnostic = true
synthetic = true
balanced = true
external_claims_supported = false
external_claims_assessment = insufficient
```

It also records fixture byte count, fixture digest, reconstructed-oracle digest, overall content
digest, and manifest digest. The oracle result is reconstructed during validation; there is no
separate, producer-supplied oracle result file.

Each `state-decay-scenario/v1` record contains a trajectory prefix, candidate memories, pivot,
allowed actions, intervention label, deterministic oracle, evidence criteria, and both paired
continuations. The validator reconstructs the versioned generator output and checks:

- exact eight-family coverage and within-family label balance;
- four linked instances for every template lineage;
- unique and resolvable event, memory, action, and evidence references;
- temporal ordering and absence of future leakage;
- memory validity and revision behavior;
- both paired continuations, label, and required oracle action;
- all fixture, oracle, content, and manifest digests.

The benchmark manifest is limited to 64 KiB and the fixture to 8 MiB. Exactly two regular,
single-link files and 32 canonical records are required. The validation report always states
`assurance: deterministic_synthetic_oracle`, `confirmatory: false`, and
`external_claims_supported: false`.

## Publication, replacement, and recovery

Replay and benchmark publishers use a sibling staging directory on the same filesystem, write
and synchronize every file, synchronize the directory, then publish by atomic rename. New
publisher-created directories are owner-only and files are owner-readable and owner-writable.

A persistent sibling lock serializes cooperating writers. The resolved parent must be controlled
by the current owner; group- or world-writable parents are rejected on POSIX. Existing targets
are preserved unless `--replace` is supplied.

Replacement is deliberately narrow:

- replay requires an entirely valid existing replay artifact with the same run identifier;
- algorithm replay requires an entirely valid existing algorithm artifact with the same run
  identifier and condition;
- StateDecayBench requires an entirely valid artifact with the same suite, version, and fixture
  digest.

Unrelated directories and corrupted artifacts are never authorized merely by `--replace`.
Replacement uses identity- and digest-bound marker and backup entries. If a process stops during
publication, rerunning the same producer can recover only an unambiguous old or completed state.
Ambiguous paths are preserved and rejected rather than deleted.

Sibling implementation entries can include:

```text
.<artifact>.lock
.<artifact>.tmp-*
.<artifact>.backup
.<artifact>.replace.json
```

Do not remove these automatically after an interrupted operation; first rerun validation or the
same producer and investigate any fail-closed result.

## Python validation example

Artifact-compatible after installation:

```python
from pathlib import Path

from saliencegate.artifacts import (
    ArtifactValidationError,
    load_validated_artifact,
    validate_artifact,
)

manifest = Path("artifact/manifest.json")

try:
    report = validate_artifact(
        manifest,
        expected_manifest_digest="<64 lowercase hexadecimal characters>",
    )
except ArtifactValidationError as error:
    print(error.code.value)
else:
    loaded = load_validated_artifact(manifest)
    print(report.manifest_digest)
    print(loaded.manifest.run_id)
```

Custom producers can use `export_replay_artifact()`. `synthetic_raw` requires an explicit
`SyntheticArtifactContent`; never relabel user or model data as synthetic to bypass the normal
content boundary.

Algorithm producers use `export_algorithm_artifact()` with an exact
`AlgorithmExecutionAttestation`. Validate with `validate_algorithm_artifact()` or load the safe
typed view with `load_validated_algorithm_artifact()`. The algorithm exporter accepts only
synthetic algorithm results; its `synthetic_raw` classification is explicit and never a route for
user or production data.

## Security and evidence limits

- Integrity is not identity, authenticity, a signature, or encryption.
- Producer-attested grounding does not prove semantic entailment.
- Confirmatory evidence level does not by itself prove an intervention caused an outcome.
- A replay artifact is execution evidence, not a bundle containing the trace and response inputs
  needed to rerun it.
- A digest-only algorithm artifact cannot reproduce its source result; raw reprojection proves
  equality to the embedded synthetic result, not external task efficacy.
- Minimized artifacts can still expose correlatable metadata.
- A hostile process running as the same operating-system account can bypass advisory locks and
  mutate owner-only files.
- Whole-database rollback detection still needs an external monotonic anchor.
- Atomic artifact publication uses POSIX filesystem semantics. Windows publication is not
  supported or verified.
- StateDecayBench is a synthetic diagnostic and cannot establish external task performance.
- Replay schema v1 and algorithm-artifact/v1 are strict; there is no automatic migration of a
  future artifact schema.
