# Shadow Mode reference

Shadow Mode analyzes agent events with pre-persistence redaction and four deterministic detectors.
The analyzed path is local: it makes no provider, model, or network call. Shadow Mode does not run
memory, change agent behavior, or hold runtime authority. A durable session authenticates its
ledger; the resulting report is content-addressed, not authenticated.

The public package supports incremental `ShadowSession` events, immutable whole-trace analysis,
and a one-call ATIF bridge. The file-oriented commands are the bounded native-NDJSON
`saliencegate shadow analyze` path and the explicit-profile `saliencegate shadow analyze-atif`
path. Runnable examples are in [`examples/shadow_asyncio.py`](../../examples/shadow_asyncio.py)
and [`examples/atif-shadow/`](../../examples/atif-shadow/README.md).

## Detector scope and dispositions

Shadow v1 implements four detector types:

| Supported detector | Observed structured evidence |
|---|---|
| `repeated_action` | Repeated normalized action evidence |
| `repeated_failure` | Repeated unresolved failure evidence |
| `test_failure` | Structured failed test results |
| `tool_error` | Structured failed tool results |

Five declared signal types are unsupported: `conflict`, `context_shift`, `irreversible_action`,
`stagnation`, and `stale_constraint`. Every observation and report lists them explicitly. They are
not treated as abstentions and do not affect a disposition.

The frozen evaluator `any-detected-signal-baseline/v1` returns one of four values over only the
supported detector scope:

| Disposition | Meaning |
|---|---|
| `flagged` | At least one applicable supported detector emitted a signal. |
| `not_flagged` | Applicable supported detectors had sufficient evidence and emitted no signal. |
| `indeterminate` | No supported signal was emitted and at least one applicable detector abstained because its evidence was incomplete. |
| `not_applicable` | No supported detector applies to this event kind. |

All four raw detector outcomes remain in the observation. A heuristic flag is descriptive output,
not a command to run memory.

## Evidence boundary

Every observation and aggregate report fixes the same boundary:

| Field | Fixed value |
|---|---|
| `execution_mode` | `shadow` |
| `evidence_level` | `descriptive_observational` |
| `task_outcome_evidence` | `none` |
| `intervention_outcome_evidence` | `none` |
| `confirmatory` | `false` |
| `calibrated` | `false` |
| `calibration_eligible` | `false` |
| `decision_authority` | `false` |
| `representativeness_supported` | `false` |
| `task_efficacy_supported` | `false` |
| `counterfactual_effect_supported` | `false` |
| `model_calls` | `0` |
| `budget_reservations` | `0` |
| `cycles_created` | `0` |
| `memory_revisions` | `0` |
| `interventions` | `0` |
| `delivery_authorizations` | `0` |
| `deliveries` | `0` |
| `intervention_outcomes` | `0` |

Rates in the report name their denominator: unique submitted events, applicable detector
evaluations, or evidence-sufficient applicable evaluations. Capture scope is caller-declared
provenance and never changes the fixed declarations `representativeness_supported=false`,
`task_efficacy_supported=false`, or `counterfactual_effect_supported=false`.

## Universal capture projection

Universal Shadow Capture applies the same deterministic signal definitions to a narrower live-event
surface. It is not transcript ingestion and it is not an automatic conversion of a provider session
into a complete `shadow-input/v1` trace. Each Codex, Claude Code, OpenCode, or Pi adapter first
validates one audited callback shape. Unlisted fields are discarded; provider session, call, tool,
workspace, subagent, and lineage values that remain useful for correlation are reduced with an
installation-key-bound HMAC before durable storage. The adapters do not call provider history,
message, transcript, RPC, or session-file APIs to fill gaps.

At report time, SalienceGate authenticates a project-bound capture snapshot and its bounded spool,
then deterministically projects admitted records into redacted internal `TraceEvent` values. Exact
action starts become action evidence only when the provider surface establishes the required
identity. Structured result or controller-failure events become outcome evidence only when the
profile assigns that event discriminator authority. Lifecycle-only events can close windows or
record coverage boundaries without becoming detector evidence. Ignored source records remain in an
explicit denominator.

The capability manifest constrains the detector matrix before evaluation. For example, Codex v1
can conditionally support repeated-action evidence but gives `PostToolUse` no success or failure
authority; Claude Code v1 supports tool-error evidence but treats pre-tool hooks as proposals;
OpenCode v1 supports tool errors and conditional repeated actions; and Pi v1 cannot distinguish an
execution failure from several pre-execution errors. Unsupported detectors remain
`not_applicable`, while conditional detectors report their omissions and exact authorized,
unresolved, and detected counts. Missing callbacks, gaps, dropped batches, ambiguous correlation,
open windows, and unverified versions cannot be repaired by inference.

Capture uses its authenticated capture store and emits `capture-session-report/v1`; it does not
write the separate Shadow ledger or claim that provider receipt order is causal order. The report's
`shadow_disposition` is summarized by one closed headline:

| Capture headline | Shadow projection boundary |
|---|---|
| `memory_review_suggested` | At least one supported signal is detected and no quarantine or integrity failure blocks the positive result. |
| `no_current_evidence` | At least one applicable detector has sufficient absence evidence in a closed window, the authenticated spool is clean and drained, no signal is detected, and no report limit remains. |
| `insufficient_evidence` | Explicit limits prevent either of the preceding results. |

These headlines inherit the observational boundary: `evidence_level=descriptive_observational`,
`confirmatory=false`, `decision_authority=false`, and `model_calls=0`. In particular,
`memory_review_suggested` does not authorize a reminder and `no_current_evidence` does not establish
that memory was unnecessary. Provider paths, selected callbacks, version policy, and exclusions are
frozen in the [integration contract](integrations.md); local storage and retention are described in
the [security model](../security.md).

## ATIF one-call API

`analyze_atif_bytes` turns one trajectory into a complete immutable report. It accepts bytes rather
than a path, requires an explicit sealed profile and caller-attested environment, owns an in-memory
session, and closes that session on success, failure, or cancellation:

Artifact-compatible after installation:

```python
import asyncio
from pathlib import Path
from uuid import UUID

from saliencegate.shadow import (
    ATIFProfile,
    ShadowEnvironmentBinding,
    analyze_atif_bytes,
)


async def main() -> None:
    report = await analyze_atif_bytes(
        Path("trajectory.json").read_bytes(),
        run_id=UUID("c0de0000-0000-4000-8000-000000000001"),
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=ShadowEnvironmentBinding(
            default_working_directory="/synthetic/workspace",
            environment_digest="e" * 64,
        ),
    )
    print(report.binding.adapter_profile_id, report.report_digest)


asyncio.run(main())
```

The library boundary does not inspect the filesystem; an application that reads a path remains
responsible for that read. When `installation_key` is omitted, the function generates a fresh key
in memory and performs no key-file or environment lookup. Separate calls are therefore not
byte-reproducible by default. A caller that needs stable report bytes supplies the same exact
`InstallationKey`; durable resume remains explicit through `ShadowSession.sqlite_for_trace` and
`ShadowAnalyzer`. Optional `redaction_policy`, `task_scope_digest`, `lineage_scope_digest`, and
`capture_manifest_digest` values pass through the same lower-level trace path.

For direct control, construct `ATIFShadowAdapter`, call `adapt_bytes`, open a trace-bound
`ShadowSession.in_memory_for_trace` or `ShadowSession.sqlite_for_trace`, and pass the resulting
`ShadowTrace` to `ShadowAnalyzer.analyze`. `sqlite_for_trace` requires an explicit installation key
and does not open or create its database until analysis reaches the first authenticated repository
operation after complete pure preflight.

For a durable session, the analyzer first requires the ledger to be an exact prefix of the trace.
It appends only the missing suffix in one atomic batch. The append is a no-op when the complete trace
is already present; a conflicting prefix fails without extending the ledger.

## Sealed ATIF profiles

Profile selection is explicit and closed:

| CLI alias | Python profile and report ID | Accepted source | Selected action | Structured outcome authority |
|---|---|---|---|---|
| `harbor-codex-v1` | `ATIFProfile.HARBOR_CODEX_V1` / `harbor-codex/v1` | ATIF-v1.7, agent `codex` | `exec_command` | Exact signed 32-bit integer `exit_code` from either sealed metadata path; `producer_claimed_structured` only |
| `harbor-terminus-2-v1` | `ATIFProfile.HARBOR_TERMINUS_2_V1` / `harbor-terminus-2/v1` | ATIF-v1.6 or ATIF-v1.7, agent `terminus-2` | non-empty LF-submitted `bash_command` | none |

Both profiles set `producer_authentication=none`. The Codex profile is supported against the pinned
Harbor converter field shape; the converter does not pin a Codex CLI runtime, so the profile makes
no Codex-version compatibility guarantee. The Terminus 2 claim covers the pinned converter and
sanitized public-golden shapes. The source paper and proactive-memory repository are research
inspiration, not runtime compatibility evidence.

The engine still exposes four supported detector types, but each profile supplies a narrower
evidence surface:

| Detector | Codex field-shape profile | Terminus 2 field-shape profile |
|---|---|---|
| `repeated_action` | conditional | conditional |
| `repeated_failure` | conditional on exact structured exit evidence | none |
| `test_failure` | none | none |
| `tool_error` | conditional on exact structured exit evidence | none |

For Codex, `write_stdin` is an ignored continuation and never inherits a preceding result. For
Terminus 2, terminal output is never searched for success or failure text. Empty waits,
unterminated keystrokes, unresolved submissions, unsupported functions, copied context, absent
evidence, and ambiguous result parents remain explicit diagnostic dispositions rather than hidden
inferences.

Only selected events in the root trajectory segment are mapped. Continued-trajectory presence and
the number of embedded subagent trajectories are reported, but those trajectories are not
traversed. Complete execution-session coverage is always false. If every selected action has an
accepted UTC timestamp, the report uses `source_utc`; if none does, it uses a frozen logical clock;
partial selected timestamps fail.

Raw ATIF bytes are not retained in the trace or ledger. Messages, reasoning, final answers, raw
terminal or tool output, source IDs, tool-call IDs, metrics, and unselected arguments are omitted.
Selected commands, working directories, and structured exit status enter normal Shadow evidence
and pass through pre-persistence redaction. Public reports and summaries do not contain the selected
command, working directory, or any source, repository, or output path.

## Incremental Python API

This asynchronous example uses a fixed run identity and passes the returned action reference into
its structured result:

Artifact-compatible after installation:

```python
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from saliencegate.security import InstallationKey
from saliencegate.shadow import ShadowSession


async def main() -> None:
    environment = sha256(b"saliencegate-shadow-example-environment-v1").hexdigest()
    async with ShadowSession.in_memory(
        run_id=UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a"),
        installation_key=InstallationKey(bytes(32)),
    ) as session:
        await session.start(
            source_event_id="run-start",
            occurred_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        )
        action = await session.action(
            source_event_id="action-1",
            occurred_at=datetime(2026, 7, 16, 10, 1, tzinfo=UTC),
            argv=("example-tool", "--check"),
            working_directory="/example",
            environment_digest=environment,
        )
        result = await session.tool_result(
            source_event_id="tool-result-1",
            occurred_at=datetime(2026, 7, 16, 10, 2, tzinfo=UTC),
            action=action.ref,
            status="failed",
            exit_status=1,
            exception_type="ExampleToolFailure",
        )

    assert result.observation.heuristic_evaluations[0].disposition.value == "flagged"


asyncio.run(main())
```

`InstallationKey(bytes(32))` is deliberately non-secret, deterministic example material for this
ephemeral in-memory session. Do not reuse it for a durable repository; a durable integration uses
its protected installation key.

`ShadowSession.sqlite(PATH, ...)` provides the same event API with an authenticated durable ledger.
The repository stores redacted events; caller payloads are never copied into observations or
aggregate reports.

## NDJSON input

Each nonblank line is one strict `shadow-input/v1` object. Timestamps are canonical RFC 3339 UTC
text ending in `Z`. This complete example starts a run, records an action and its failed result, and
then closes the run:

```jsonl
{"kind":"run_start","occurred_at":"2026-07-16T10:00:00Z","schema_version":"shadow-input/v1","source_event_id":"run-start"}
{"argv":["example-tool","--check"],"environment_digest":"9ce3329c730b3475f5f55d0183f3d167e3568f353451434fae4fa98a17215c52","kind":"action","occurred_at":"2026-07-16T10:01:00Z","schema_version":"shadow-input/v1","source_event_id":"action-1","working_directory":"/example"}
{"action_source_event_id":"action-1","exception_type":"ExampleToolFailure","exit_status":1,"kind":"tool_result","occurred_at":"2026-07-16T10:02:00Z","schema_version":"shadow-input/v1","source_event_id":"tool-result-1","status":"failed"}
{"kind":"run_end","occurred_at":"2026-07-16T10:03:00Z","schema_version":"shadow-input/v1","source_event_id":"run-end"}
```

Save the four lines above as `events.ndjson`, then install that file into a private directory and
analyze it:

Artifact-compatible after installation:

```bash
install -d -m 700 .saliencegate-shadow
install -m 600 events.ndjson .saliencegate-shadow/events.ndjson
saliencegate shadow analyze .saliencegate-shadow/events.ndjson \
  --run-id b35f05f3-555b-4f09-8996-a7b3693bb54a \
  --output .saliencegate-shadow/report.json \
  --capture-scope complete_run_declared \
  --json
```

The accepted row kinds are `run_start`, `action`, `tool_result`, `test_result`, `observation`,
`controller_error`, and `run_end`. A result names an earlier action through
`action_source_event_id`. An identical repeated source event is a retry row; it does not increase
the unique-event aggregates. Conflicting duplicates, forward parents, malformed rows, and rows
after `run_end` fail the entire command.

## Native NDJSON CLI options

The complete command surface is:

Artifact-compatible after installation:

```text
saliencegate shadow analyze TRACE --run-id UUID4 --output PATH
  [--repository :memory:|PATH]
  [--capture-scope unknown|selected_events|bounded_window|complete_run_declared]
  [--task-scope-digest SHA256]
  [--lineage-scope-digest SHA256]
  [--capture-manifest-digest SHA256]
  [--source-adapter ID]
  [--replace]
  [--json]
```

The repository defaults to `:memory:`, capture scope to `unknown`, and source adapter to
`saliencegate-shadow/v1`. `complete_run_declared` requires a final `run_end`. Task, lineage, and
capture-manifest digests are opaque lowercase SHA-256 identifiers used only for provenance and a
later safe join.

## ATIF CLI options

The ATIF command requires the source, profile, run identity, default working directory, environment
digest, and output. It never guesses a profile or the current directory:

Artifact-compatible after installation:

```text
saliencegate shadow analyze-atif TRACE
  --profile {harbor-terminus-2-v1,harbor-codex-v1}
  --run-id UUID4
  --working-directory PATH
  --environment-digest SHA256
  --output PATH
  [--repository :memory:|PATH]
  [--task-scope-digest SHA256]
  [--lineage-scope-digest SHA256]
  [--capture-manifest-digest SHA256]
  [--replace]
  [--json]
```

`capture_scope` is fixed to `selected_events`, and `source_adapter` is derived from the sealed
profile. There is no model, endpoint, API-key, provider, memory schedule, output-text parser, or
complete-capture option. `--working-directory` is the fallback for a selected action that has no
profile-supported working directory. `--environment-digest` is a lowercase caller-attested SHA-256
join value; SalienceGate does not inspect the real execution environment to derive it.

The CLI reads and adapts the complete owner-private source before it loads or creates the normal
owner-private SalienceGate installation key. The first invocation that reaches this boundary may
initialize that local key under the platform configuration root. It authenticates the ledger and is
not a provider credential. A SQLite repository supports authenticated exact-prefix resume;
`:memory:` remains the default.

Copy-paste commands for both profiles are in the
[ATIF example guide](../../examples/atif-shadow/README.md).

## Native Shadow report contract

The owner-only output is one canonical `shadow-run-report/v1` JSON object. Its field groups are:

- run ID; initial authenticated-ledger count and tags; input-byte, normalized-input,
  redaction-policy, detector-profile, and report digests;
- capture scope; optional task, lineage, and capture-manifest digests; split-metadata completeness;
- input-row, unique-event, retry, appended, pre-existing, rejected, evaluated-event, and
  observation counts;
- ordered row commitments and one payload-free observation per unique event;
- supported and unsupported signal types; detector outcomes and exact abstention reasons;
- heuristic dispositions; applicable and evidence-sufficient detector-evaluation counts;
- signal co-occurrence, event-type, and phase counts; first flagged sequence or `null`;
- every fixed evidence field listed above.

The JSON stdout form is a smaller `shadow-command-report/v1`: it contains status, run ID, the two
input digests, detector-profile digest, supported and unsupported signal types, unique-event and
retry counts, all four disposition counts, report digest, and the same fixed evidence boundary. It
contains no input, repository, or output path.

## ATIF provenance report contract

`analyze_atif_bytes`, `ShadowAnalyzer.analyze`, and `shadow analyze-atif` return or publish one
canonical `shadow-trace-report/v1`. It binds:

- the run, original source-byte digest, exact profile and configuration digests, source schema,
  timestamp mode, fixed selected-event scope, and optional provenance digests;
- complete mapping diagnostics, including source totals, every mapped and ignored disposition,
  root-segment limitations, outcome authority, and compatibility-manifest digest;
- the mapped-record digest and complete nested `shadow-run-report/v1`;
- one domain-separated outer report digest.

The wrapper is a content-addressed provenance commitment, not a signature or proof that the source
producer was truthful. `verify_shadow_trace_source` can re-adapt exact source bytes with the same
configured adapter and check the source-side binding, diagnostics, and mapped-record digest.
Authenticated-ledger verification is a separate repository operation.

The CLI's human output reports root coverage, mapping and omission counts, structured-outcome
coverage, all four heuristic dispositions, per-detector outcomes and abstentions, timestamp mode,
producer authentication, outcome authority, manifest digest, and report digest. `--json` emits the
smaller canonical `shadow-atif-command-report/v1`, not the nested report or rows. Both forms are
content- and path-free.

## Replacement and filesystem contract

The output parent must already be a private, owner-controlled directory. Native NDJSON input must be
a stable, user-owned, single-link regular file that is not group- or world-writable; ATIF input must
be owner-private. Symbolic links, hard links, changing files, unsafe ancestor traversal, and
input/repository/output aliases are rejected. A successful report is atomically published with mode
`0600`, reopened through the bounded reader, parsed again, and required to be byte-for-byte
canonical.

An existing output is preserved unless `--replace` is present. Replacement is allowed only when
the existing file is valid and exactly authorized. Native reports bind the same run, input bytes,
normalized input, capture provenance, redaction-policy tag, and detector profile. ATIF reports must
also match the complete source, profile, configuration, provenance, nested Shadow binding, initial
repository state, and outer report identity. Corrupt, unrelated, or changed reports are not
overwritten.

Exact limits are 10,000 rows, 2 MiB per encoded line, 64 MiB for the complete input, and 128 MiB
for a native canonical report. ATIF input is bounded to 64 MiB, 10,000 root steps, 10,000 total tool
calls, 10,000 total results, and 1,000 mapped Shadow records; the outer report is bounded to 130 MiB.
A platform that cannot enforce owner-only atomic publication fails closed.

HMAC integrity is not encryption. A SQLite ledger contains redacted plaintext, so filesystem access
controls and storage protection still matter.

## Exit codes

| Exit | Meaning |
|---:|---|
| `0` | Success, help, version, or a closed output pipe. |
| `2` | Syntax, schema, lifecycle, parent, collision, bound, path, alias, or replacement-authorization failure. |
| `3` | Installation-key, SQLite, secure-publication, retry-exhaustion, or repository-compatibility failure. |
| `5` | Corrupt replacement, post-publication canonical mismatch, or report-digest failure. |
| `70` | Built-in detector contract violation or unexpected internal failure. |
| `130` | Keyboard interruption. |

Errors are stable, value-free diagnostics on standard error. A failed command prints no partial
success report. An already committed exact prefix remains recoverable by rerunning the same input.
