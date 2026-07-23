# Research claims

This page separates software behavior that can be verified in the repository from empirical
conclusions that have not been established. It is the claim boundary for Universal Shadow Capture,
the ATIF bridge, and whole-trace Shadow analysis. The [README](../README.md) gives the shorter
project overview.

## What the implementation establishes

The repository provides:

- project-local passive capture installers for Codex, Claude Code, OpenCode, and Pi, each bound to
  one audited version contract, a closed lifecycle allowlist, and reversible owned files;
- bounded provider-callback parsing that discards unlisted content and HMAC-reduces admitted
  correlation identifiers before persistence;
- authenticated capture connection, event, health, spool, snapshot, feedback, and deletion state,
  with project-bound `connect`, `disconnect`, `status`, `sessions`, `report`, `feedback`, and
  `delete` commands;
- deterministic capture normalization, profile-constrained detector evidence, and three closed
  report headlines with explicit support, exclusions, denominators, and integrity limits;
- strict, bounded, offline ATIF parsing for two sealed field-shape profiles;
- deterministic mapping from selected structured fields to typed Shadow records;
- complete trace preflight before mutation, followed by one bounded authenticated append for the
  missing suffix;
- pre-persistence redaction, authenticated in-memory and SQLite ledgers, exact-prefix resume, and
  canonical provenance reports;
- deterministic evaluation through the four implemented Shadow detectors;
- mapping and omission diagnostics, detector outcomes, abstention reasons, and explicit
  denominators;
- a compatibility manifest that pins audited source revisions, converter identities, fixture
  identities, and the scope of each compatibility statement;
- a provider-free one-call API and an owner-private atomic CLI workflow with no model invocation or
  decision authority.

Replacement preview may open, migrate, or recover SQLite derived state, but it does not append the
accepted suffix. If the stored trace is already complete, analysis performs no append.

These are software and provenance properties. Tests can show that the code follows its frozen
contract for covered inputs. They cannot show that an input producer is honest or that an observed
signal improves an agent task.

## How to read the evidence

| Surface | Current evidence | Supported interpretation | Not supported |
|---|---|---|---|
| Codex `0.144.6`, Claude Code `2.1.204`, OpenCode `1.18.3`, and Pi `0.80.10` capture connectors | Audited upstream contracts, synthetic provider fixtures, installation/recovery tests, and installed-artifact smoke | The packaged connector handles the exact admitted callback and reversible managed-file shapes under test | Provider/source authentication, complete sessions, remote runner verification, or general future-version compatibility |
| Capture reports | Authenticated-store, spool, normalization, detector, headline, privacy, deletion, and synthetic end-to-end tests | Deterministic production of `memory_review_suggested`, `no_current_evidence`, or `insufficient_evidence` under the closed report contract | Prevalence, calibration, task prediction, reminder usefulness, or an intervention-attributable outcome |
| Local feedback and evaluator | Deterministic authenticated export, support, privacy-floor, metric, and interval tests | Mechanics of one preregistered three-state classification protocol and explicit insufficiency reasons | Accepted E01 evidence, independently verified study process, or authority to activate a reminder |
| `harbor-terminus-2/v1` | Pinned Harbor converter and sanitized consumed shapes from two pinned public goldens | The sealed mapper handles the documented ATIF-v1.6/v1.7 field shapes and omits terminal outcome text | General Terminus compatibility, producer authentication, or inferred success or failure |
| `harbor-codex/v1` | Pinned Harbor converter field shape and a fully synthetic ATIF-v1.7 fixture | The sealed mapper handles the documented `exec_command`, exact exit metadata, and `write_stdin` shapes | A pinned or generally supported Codex CLI runtime version |
| Shadow detectors | Synthetic, sanitized, and contract tests with explicit detector denominators | Deterministic behavior of the four implemented detectors on admitted evidence | Calibration, population frequency, or task prediction |
| Whole-trace repository path | Atomicity, cancellation, retry, resume, and integrity tests | Exact-prefix recovery and all-or-nothing bounded suffix mutation under the tested contract | Cross-restart whole-database rollback protection |
| ATIF command reports | Canonical encode/decode, source verification, privacy, and replacement tests | Content-addressed source, mapping, configuration, diagnostic, and nested-report commitments | A signature, producer truth proof, or standalone authenticated-ledger proof |

The [compatibility manifest](../src/saliencegate/shadow/atif_profile_compatibility.json) records the
exact upstream identities and fixture transformations. The Remember When It Matters paper and the
proactive-memory repository are research inspiration, not runtime compatibility evidence.

## Universal Capture evidence and omissions

Universal Capture observes only callbacks received from the selected project-local provider
surface. The v1 profiles fix these exact audited hosts: Codex CLI `0.144.6`, Claude Code `2.1.204`,
OpenCode `1.18.3`, and Pi `0.80.10`. Codex and Claude Code accept newer patches only within their
audited minor line as `schema_compatible_unverified_version`; OpenCode and Pi are exact-version
contracts. No connector bypasses provider project trust. Installation does not prove activation,
and status remains `installed_not_observed` until a callback is admitted.

Provider callbacks and identifiers have `source_authentication=none_same_user_untrusted`.
Timestamps have `local_observation` authority and ordering has `local_receipt_order` authority.
Capture does not read transcripts or history, persist raw content, make a model call, or gain
decision authority. Missing hooks, wholly lost transport batches, crashes before closing events,
unsupported specialized tools, provider config rejection, ambiguous outcomes, subagent or resume
boundaries, and host-version drift can make coverage incomplete. A report lists those limits rather
than inferring the missing event.

The three headlines are deliberately asymmetric. A supported signal can produce
`memory_review_suggested` despite an open window when no quarantine or integrity failure blocks the
positive observation. `no_current_evidence` requires a closed session, a clean authenticated and
drained spool, sufficient evidence for at least one applicable detector, no detected signal, and no
remaining limit. Every other bounded case is `insufficient_evidence`. None is a recommendation or
task-outcome claim.

The repository contains synthetic fixtures that exercise these branches, not a real-world sample.
The prepared Ubuntu, Windows, and macOS native connector workflows have not been run remotely from
this unpublished branch. The current assessment is therefore
`insufficient_real_world_evidence`; remote compatibility evidence and the externally reviewed E01
study are separate future authorization gates.

## Profile evidence and omissions

Both profiles declare `producer_authentication=none`. The Codex field-shape profile admits an exact
signed 32-bit integer exit status only as `producer_claimed_structured`; the Terminus 2 profile
admits no outcome evidence. Neither profile interprets output text.

| Evidence class | Codex field-shape profile | Terminus 2 field-shape profile |
|---|---|---|
| Repeated normalized action | Conditional evidence | Conditional evidence |
| Repeated unresolved failure | Conditional on admitted exit status | Omitted |
| Failed test result | Omitted | Omitted |
| Failed tool result | Conditional on admitted exit status | Omitted |
| Continuation state | `write_stdin` counted and omitted | Unterminated or unresolved terminal submissions counted and omitted |
| Complete execution session | Unsupported | Unsupported |

Only selected events from the root trajectory segment are mapped. Messages, reasoning, answers,
raw output, source IDs, tool-call IDs, metrics, copied context, and unselected arguments are
omitted. Continued and embedded-subagent trajectories are counted but not traversed. Selected
commands and working directories pass through ordinary Shadow redaction and do not appear in
reports or command summaries.

## Fixed observational boundary

Every Shadow observation and report remains:

```text
execution_mode=shadow
evidence_level=descriptive_observational
task_outcome_evidence=none
intervention_outcome_evidence=none
confirmatory=false
calibrated=false
calibration_eligible=false
decision_authority=false
representativeness_supported=false
task_efficacy_supported=false
counterfactual_effect_supported=false
```

Model calls, budget reservations, cycles, memory revisions, interventions, delivery authorizations,
deliveries, and intervention outcomes are all zero. A `flagged` disposition means only that at
least one supported deterministic detector emitted a signal for an admitted event. It is not an
instruction to invoke memory or change an agent.

Every capture report fixes `evidence_level=descriptive_observational`,
`source_authentication=none_same_user_untrusted`, `complete_execution_session_coverage=false`,
`raw_content_persisted=false`, `transcript_read=false`, `confirmatory=false`,
`decision_authority=false`, and `model_calls=0`. A capture headline does not relax any of those
fields.

## Claims not made

The repository does not claim:

- improved or preserved task success;
- token, monetary-cost, or latency reduction;
- comparative superiority over a graph-memory product, memory store, agent framework, or another
  controller;
- calibrated risk estimates or representative prevalence;
- complete agent-session, continuation, or subagent coverage;
- authenticated live-provider callbacks, provider identifiers, timestamps, or sequence;
- remote native-runner compatibility evidence for the four capture connectors;
- accepted real-world evidence from the synthetic capture headline fixture;
- authenticated ATIF producers or truthful producer metadata;
- effectiveness of a reminder, intervention, delivery, or memory revision;
- external long-horizon benchmark performance;
- compatibility with an unspecified current or future Codex or Terminus runtime.

Any efficacy or comparative statement needs a preregistered task-level evaluation with allocated
conditions, complete failures, denominators, environment and revision identities, and an
appropriate uncertainty analysis. It cannot be inferred from a Shadow report.

## Performance protocol and evidence

The tracked reference run measures implementation cost for 1,000 mapped records. It does not
measure agent quality. Five isolated measurement processes per backend ran after one warm-up process
per backend. Socket and resolver access were denied, and the report records no imported provider
module.

| Backend | Observed median | Median budget | Maximum peak RSS | RSS budget | Result |
|---|---:|---:|---:|---:|---|
| In memory | 4.449103458 s | 5 s | 185.359375 MiB | 512 MiB | Passed |
| SQLite | 7.178122875 s | 15 s | 214.0625 MiB | 512 MiB | Passed |

The measurements came from one local machine: macOS 26.5.2 on arm64, 15 logical cores, 24 GiB of
memory, and CPython 3.12.3. The runner image was unavailable and is recorded as `unspecified`. The
non-gating 250-to-1,000 median-time ratios were 5.029737155157485 in memory and
5.431701416012222 with SQLite; the v1 prefix digest is not claimed to scale linearly.

The exact output is the [reference JSON](../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json).
Its [evidence manifest](../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.manifest.json)
binds the report to the benchmark script, runtime source surface, socket guard, project inputs,
dependency lock, and toolchain. The [foundation evidence](benchmarks/foundation-evidence.md) lists
all ten measured durations and the reproduction command.

This is single-machine engineering evidence. It is not provider, efficacy, billed-cost,
cross-machine, or comparative evidence.

## Reproduction

The smallest provider-free API demonstration is:

Run from a checkout:

```bash
uv run --locked python examples/atif-shadow/one_call.py
```

The [ATIF example guide](../examples/atif-shadow/README.md) has separate commands for the Codex and
Terminus 2 field-shape profiles. The [Shadow Mode reference](reference/shadow-mode.md) defines the
API, evidence, diagnostics, and report contracts. The [CLI reference](reference/cli.md) defines
secure source handling, key initialization, replacement, summaries, and exit categories. The
[security appendix](security.md) defines the trust and privacy boundaries.
