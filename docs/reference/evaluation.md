# Feedback and evaluation reference

SalienceGate feedback is local human annotation for a captured session. It does not change the
agent, select memory, inject context, enable a reminder, or establish that an intervention would
have helped. Capture reports remain descriptive observational evidence with no decision authority.

## Current evidence state

The checked-in [capture example](../../examples/capture/README.md) is synthetic. It exercises the
authenticated normalization and report contracts for all three headlines:
`memory_review_suggested`, `no_current_evidence`, and `insufficient_evidence`. Its repeated-action
and structured-failure case is constructed to emit a signal; its clean closed case is constructed
to meet an absence minimum; and its incomplete case is constructed to preserve an explicit limit.
Every result remains `confirmatory=false`, `decision_authority=false`, and `model_calls=0`.

That fixture demonstrates deterministic mechanics only. It is not a sampled provider population,
has no independently adjudicated task outcome, and cannot estimate prevalence, precision, recall,
false-positive rate, usefulness, or an intervention-attributable outcome. No real-world feedback
dataset or accepted E01 assessment is committed here. The current evidence status is therefore
exactly
`insufficient_real_world_evidence`, regardless of whether the synthetic rows reproduce their
expected headlines.

## Record local feedback

Artifact-compatible after installation:

```text
saliencegate feedback SESSION_ID \
  --label memory-needed|not-memory-needed|uncertain \
  [--json]
```

The session identifier is the short SalienceGate identifier shown by `sessions` and `report`, not a
provider-native identifier. The command resolves the current project, verifies that the session is
closed and belongs to it, and writes only one bounded label record to the owner-private capture
store. Open and quarantined sessions are rejected. Repeating the current label is idempotent. A
changed label creates an authenticated revision without copying captured events or provider
content.

The labels mean:

| Label | Evaluation role |
|---|---|
| `memory-needed` | Human annotation that the session belongs in the positive reference class. |
| `not-memory-needed` | Human annotation that the session belongs in the negative reference class. |
| `uncertain` | Human abstention; it is retained and is never forced into either binary class. |

Human output contains the SalienceGate session identifier, selected label, idempotent write
disposition, and bounded revision count. JSON also includes the local label timestamp. That
timestamp is local ordering metadata, not independently trusted time. A chained, authenticated
anchor detects mutation and partial or tail truncation while any feedback history remains. It
cannot detect deletion of every feedback row for a session or rollback of the whole database to an
older valid copy. Errors use the common value-free capture exit contract. No output contains a
path, provider-native identifier, event digest, prompt, reasoning, command, tool input, or tool
output.

## Explicit dataset export

Feedback is not exported by capture, `report`, or `feedback`. Export requires a separate explicit
Python API call. First call `CaptureStore.list_feedback(label_freeze=...)` in one maintenance
transaction; it selects the last authenticated revision with `labeled_at < label_freeze`, rejects
silent truncation beyond the requested limit, and excludes later revisions. Pass that exact UTC
exclusive boundary to `build_capture_feedback_export_record`.

For local evidence, that factory requires the authenticated feedback record, its key-bound capture
snapshot, the recomputed snapshot-bound normalization, an exact authenticated spool rather than a
missing-spool fallback, and a closed capture report. It rebuilds the report from those inputs and
requires byte identity before deriving the three-state prediction from the headline. The label
must be no earlier than session closure and strictly precede the freeze. HMAC binding proves that
these inputs are mutually consistent; it does not prove that a caller selected sessions or snapshot
boundaries without bias. A declared E01 study therefore also commits its externally reviewed
report-selection policy in the study attestation. The separate
`build_synthetic_capture_feedback_export_record` accepts an explicit prediction but marks the row
synthetic; in-process dataset construction refuses to declare that origin as E01 evidence.

`build_capture_feedback_dataset` then fails closed unless `opt_in` is the exact boolean `True`, the
export nonce is exactly 32 bytes, and the inputs are a unique immutable tuple.
`encode_capture_feedback_dataset` provides canonical serialization. Dataset construction also adds
a key-bound HMAC over the canonical artifact. `decode_capture_feedback_dataset(...,
installation_key=...)` and `evaluate_capture_feedback_dataset(..., installation_key=...)` require
the originating installation key and reject a changed label, prediction, partition, profile,
project pseudonym, evidence declaration, digest, or tag. The SHA-256 dataset digest is a content
address; the HMAC supplies local mutation authentication. Neither proves that a human annotation
or external study declaration is true. The resulting content-free, pseudonymized dataset may contain
an export-scoped sample identifier, an opaque project stratum, the provider profile, the derived
three-state prediction, the bounded human label, a caller-supplied study partition, a pseudonymous
report binding, evidence-source metadata, opaque study-attestation commitments, and deterministic
ordering. It contains no raw capture event, short local session identifier, connection identifier,
internal session digest, filesystem path, transcript, prompt, response, reasoning, command, tool
arguments, tool output, credential, raw report digest, or exact provider-native timestamp.

The exported project and sample identifiers are export-specific, domain-separated pseudonyms, not
anonymization. They remain linkable within one export; a different explicit export nonce produces
an unlinkable identifier set. The owner must still treat the dataset as sensitive. Provider strata
may be reported directly. Project pseudonyms are used for support checks and stratified resampling,
but are omitted from the classification evaluation report. Raw data remains outside the repository;
publishing even the pseudonymized export is a separate owner decision. A recipient without the
originating key cannot use the authenticated decoder as an independent signature verifier; this local feedback protocol does
not define a public-key publication protocol.

## Classification evaluation report

Evaluation is exposed only through the Python functions `evaluate_capture_feedback_dataset`,
`encode_capture_calibration_report`, and `decode_capture_calibration_report`; there is no CLI
evaluation command. Evaluation and report decoding require the originating installation key.
Calibration reports carry a separate HMAC tag, so changing interval endpoints, reasons, counts, or
authentication claims and merely recomputing the public digest is rejected. The `calibration` name
in the serialized API is a schema name, not a claim that the heuristic emits calibrated
probabilities.

The local export-record factory uses this preregistered mapping:

| Capture conclusion | Evaluation prediction |
|---|---|
| `memory_review_suggested` | positive |
| `no_current_evidence` | negative |
| `insufficient_evidence` | system abstention |

The local export-record factory applies this mapping only after rebuilding the report from the
authenticated snapshot, recomputed normalization, and spool boundary, and binds the result to the
report digest and fixed evaluator version. Only the explicitly synthetic factory accepts a
prediction directly. Evidence-source and study-process fields remain externally reviewed
declarations rather than proof of study quality. The evaluation contract fixes every
denominator. Reports expose dataset, development, calibration, and final-test support; pooled and
provider-specific positive, negative, and uncertain support; and all nine mutually exclusive
classification cells. Sample prevalence, precision, recall, false-positive rate,
reference-abstention rate, prediction-abstention rate, and joint-abstention rate are emitted only
when their denominators are mathematically defined. Undefined estimates remain explicit JSON
`null` values rather than becoming zero, `NaN`, or an efficacy claim.

Raw numerators and denominators remain unreduced support counts. Provider strata below 30
final-test sessions are not emitted: the report exposes only the suppressed provider count and
combined suppressed support, adds an insufficiency reason, and does not disclose their label,
prediction, or confusion cells. This is a report privacy floor, not an E01 success threshold.

For the locked final-test cohort, the definitions are:

| Metric | Numerator / denominator |
|---|---|
| Sample prevalence | `memory-needed / (memory-needed + not-memory-needed)` |
| Precision | true-positive / (true-positive + false-positive) |
| Recall | true-positive / all `memory-needed` labels, including system abstentions |
| False-positive rate | false-positive / all `not-memory-needed` labels, including system abstentions |
| Reference-abstention rate | `uncertain` labels / all rows |
| Prediction-abstention rate | system abstentions / all rows |
| Joint-abstention rate | rows with `uncertain` or a system abstention / all rows |

The determinate confusion cells and system-abstention cells are disjoint. Predictions attached to
an `uncertain` reference label never become a true-positive or false-positive.

The fixed bootstrap protocol is:

- 2,000 replicates;
- public seed
  `9f4c8dc1d7f87c2bf08bfc24f9cb6bb4de27c57fa3466a7a63d7f01e13961e7e`, with
  domain-separated commitment
  `0ba351d0bc979e0f7280c10f3f36c6d4da3b9df59b4b8b0c664d504ecd89ddc3`;
- fixed-size resampling with replacement inside every provider/project cell;
- canonical cell coordinates derived from the cell's nine classification counts and duplicate
  ordinal, so changing the export nonce cannot become a second seed or support nonce shopping;
- length-prefixed SHA-256 over seed, replicate, provider, canonical cell coordinate, draw, and
  rejection attempt; the first 64 bits use unbiased rejection before modulo, with at most 16
  attempts;
- the same replicate reused for every pooled and provider metric;
- exact rational ordering with nearest ranks 50 and 1950 of 2,000;
- lower endpoints rounded down to parts per million and upper endpoints rounded up;
- no interval unless the observed raw denominator is at least 30 and all 2,000 replicate
  denominators are defined.

The same evidence produces the same metric intervals across export nonces. These percentile
intervals describe sample uncertainty under this fixed procedure; they do not repair selection
bias or validate caller-supplied study declarations. A zero-width boundary interval, including a
finite-sample false-positive interval of `[0,0]`, is marked `finite_sample_safety_bound=false` and
must not be used as E02's one-sided acceptable-upper-FPR gate.

These are deterministic classification metrics for a three-state heuristic, not probabilistic
calibration: capture emits no probability score for reliability or calibration curves to assess.

Zero examples, too few examples, missing reference classes, insufficient provider or project
coverage, and synthetic-only fixtures all produce the exact evidence status
`insufficient_real_world_evidence`. Synthetic fixtures demonstrate deterministic mechanics only.
The local evaluator retains that status even for a `DECLARED_E01` dataset satisfying the numeric
floors, because external review remains required. Every report remains `confirmatory=false` and
`decision_authority=false`. No metric, interval, threshold, or fixture result creates an activation
flag or changes runtime behavior.

## Preregistered E01 study boundary

The external E01 study may be assessed only after all of these conditions are met:

- at least 200 human-adjudicated sessions in the locked final-test partition;
- at least three projects and two providers in that partition;
- at least 30 `memory-needed` and 30 `not-memory-needed` final-test labels;
- `uncertain` retained as abstention;
- a non-empty development or tuning cohort fixed before final-test access;
- temporal separation between development or tuning data and the final test set;
- the evaluator and inclusion/exclusion rules frozen before final-test access;
- the report and snapshot selection policy frozen before final-test access;
- a label-revision cutoff frozen before final-test access, with later revisions excluded;
- external consent and preregistration evidence committed by opaque digests in the study
  attestation;
- blinding status recorded externally as `externally_attested`, `not_blinded`, or `unknown`;
- support, sample prevalence, precision, recall, false-positive rate, abstention rate, and
  deterministic confidence intervals reported pooled and by provider whenever the provider privacy
  and denominator floors are met;
- opaque project membership used for the support gate and bootstrap stratification, without project
  identifier disclosure in the evaluation report;
- no tuning on the final test set.

`DECLARED_E01`, the development/final-test partition, temporal separation, consent,
preregistration, revision cutoff, evaluator freeze, report-selection policy, and blinding are
external study declarations. Local HMACs authenticate stored and exported bytes under the
installation key; they do not prove adjudication quality, partition assignment, those process
controls, independent time, complete feedback-history preservation, or whole-database rollback
resistance. Those controls require evidence outside this local API.

The threshold for considering any later reminder must be registered before collecting the study
data, include a conservative one-sided acceptable upper bound on false positives, and require
stability across providers. The local two-sided percentile interval is descriptive and is never that
gate. Insufficient support, suppressed providers, undefined intervals, zero-width boundary
intervals, or wide intervals mean continuing the study. A reminder or other active behavior belongs
to a separate, explicitly authorized design after E01; no such path exists in the capture,
feedback, export, or classification evaluation code described here.
