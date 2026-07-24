# CLI reference

SalienceGate provides local commands for environment checks, an offline demo, Shadow Mode analysis,
frozen replay, algorithm replay, synthetic diagnostics, artifact inspection, and validation. These
commands do not call a live model or require an API credential. The separately named `pilot`
command is the only CLI surface that can contact a configured local OpenAI-compatible endpoint.

The installed entry point and module entry point are equivalent:

Artifact-compatible after installation:

```bash
saliencegate --help
python -m saliencegate --help
```

Long-option abbreviations are disabled. `--help` and `--version` exit successfully.

## Output and exit contract

Successful command summaries write only to standard output; explicit artifact destinations and
the documented capture lifecycle may also mutate their bounded local files. With `--json`, stdout
is one compact, canonical UTF-8 JSON object followed by a newline. Errors remain value-free text on
standard error, even when `--json` was requested; paths, payloads, endpoints, and exception details
are not echoed.

`doctor` is the one command that writes its structured report before returning a non-zero health
status. A reader closing a pipe early is treated as success. A keyboard interruption returns
130.

| Exit | Meaning |
|---:|---|
| `0` | Success, help, version, or closed output pipe |
| `2` | Invalid syntax, input, run identifier, suite, path, or destination |
| `3` | Invalid local configuration |
| `4` | Required Python or SQLite capability unavailable |
| `5` | Corrupt artifact, digest mismatch, or unmet evidence requirement |
| `70` | Unexpected internal error, reported without input values |
| `130` | Keyboard interruption |

Capture commands use the same table with these fixed categories: invalid command input exits 2;
an unsafe or unusable local configuration exits 3; a missing installed integration exits 4; and an
authenticated-store, spool, receipt, journal, or managed-file integrity failure exits 5. Capture
errors never echo project paths, provider-native identifiers, input values, or exception details.

## Passive capture lifecycle

Universal Shadow Capture supports project-local and user-global connections. The provider aliases
are exactly `codex`, `claude-code`, `opencode`, and `pi`. `setup` configures one or all providers;
`connect`, `disconnect`, and `status` select either `--project` or `--global`. Without either flag,
project commands use the current directory. `sessions`, `report`, and `feedback` always use the
current directory as the project boundary. A project-local integration takes precedence over the
matching global integration, and SalienceGate never changes provider trust settings. None of these
commands calls a model, reads a transcript, or gains decision authority.

The intended operator sequence is:

1. run `setup`, or inspect a scripted `setup ... --dry-run`, and review the managed files;
2. retain the provider's ordinary trust policy, then confirm the plan;
3. run one provider session and distinguish installation from observation with `status`;
4. select a short SalienceGate session identifier with `sessions` and inspect it with `report`;
5. run `disconnect PROVIDER` to stop admission while retaining existing evidence;
6. use `delete` only when the retained local records are no longer required.

The complete provider paths, selected callbacks, version policy, exclusions, and recovery rules are
in the [integration contract](integrations.md).

## `setup`

Artifact-compatible after installation:

```text
saliencegate setup
saliencegate setup --install-only
  [--dry-run | --yes | --confirm PHRASE]
  [--json]
saliencegate setup --provider codex|claude-code|opencode|pi|all
  [--provider codex|claude-code|opencode|pi]
  --scope project [--project PATH]
  [--dry-run | --yes | --confirm PHRASE]
  [--json]
saliencegate setup --provider codex|claude-code|opencode|pi|all
  [--provider codex|claude-code|opencode|pi]
  --scope global [--exclude PATH]...
  [--dry-run | --yes | --confirm PHRASE]
  [--json]
```

With no arguments, `setup` opens a wizard for install-only, current-project, selected-project, or
user-global setup. `all` must be used alone; in global scope it selects the detected providers that
already have a user configuration and usable host command. Explicit provider selections remain
fail-closed when unavailable. Otherwise `--provider` may be repeated. Project scope defaults to the
current directory when `--project` is omitted. Global exclusions must name existing project
directories and may be repeated.

Every form builds the complete plan before mutation. `--dry-run` stops after that plan. `--yes`
accepts it non-interactively, while `--confirm` must match the displayed scope-and-provider phrase.
JSON mutation requires one of those two explicit approvals. Setup reports that provider trust
changes are false.

## `connect`

Artifact-compatible after installation:

```text
saliencegate connect codex|claude-code|opencode|pi
  [--project PATH | --global]
  [--exclude PATH]...
  [--dry-run]
  [--json]
```

`connect` plans or installs one authenticated integration. Project scope is the default;
`--exclude` is valid only with `--global` and may be repeated. A dry-run validates the selected
scope, provider contract, collision rules, and managed-file plan without creating the installation
key, capture store, receipt, launcher, or provider configuration.

A normal project connect registers a pending local connection, publishes the managed integration,
and enables admission last. A global connect publishes into the provider's user configuration,
registers one authenticated parent, and automatically derives an authenticated child when that
provider first reports a non-excluded project. A matching project-local installation suppresses the
global route for that provider. Codex and Claude Code probe their supported live versions; OpenCode
and Pi use exact pinned versions without launching the host during connect.

The operation preserves unrelated configuration and fails closed on malformed selected config,
ownership ambiguity, an unsafe path, unexpected managed-file drift, an unsupported host version,
or provider policy that disables the selected hook surface. Repeating the same valid connection is
idempotent. A supported patch-version change creates an authenticated connection generation rather
than silently treating it as the audited baseline.

Human output reports the provider, install disposition, whether capture is enabled, and the
read-only Git review result for every managed project file. It distinguishes an absent Git work
tree, an unavailable probe, files ignored by repository rules, files Git would surface, and files
already tracked. JSON is one canonical `capture-connect/v1` record with the same bounded result and
counts. Neither mode changes `.gitignore` or includes a project path, command, identifier, or
digest. Global output instead reports managed-file and exclusion counts; its JSON schema is
`global-capture-connect/v1`.

## `disconnect`

Artifact-compatible after installation:

```text
saliencegate disconnect codex|claude-code|opencode|pi
  [--project PATH | --global]
  [--json]
```

Project-local `disconnect` disables admission, drains the authenticated spool, reverses only the
owned provider configuration or bridge files, and marks matching local connections disabled.
Global `disconnect` removes only that provider's authenticated user-global files and disables its
parent; derived project children remain as retained local evidence. Receipt and journal bindings
prevent an unrelated look-alike file from being removed. Drift or ambiguous ownership fails closed
for manual inspection.

Disconnect is the connector uninstall operation; it deliberately retains the installation key,
SQLite observations, spool boundary, disabled connection receipts, session records, feedback, and
deletion tombstones. It does not imply data deletion. JSON uses `capture-disconnect/v1`.
Global JSON uses `global-capture-disconnect/v1`.

## `status`

Artifact-compatible after installation:

```text
saliencegate status [codex|claude-code|opencode|pi]
  [--project PATH | --global]
  [--json]
```

Without a provider argument, either scope returns all four profiles. Project status drains an
available spool, authenticates the project connection and managed installation, and reports one of
`not_installed`, `installed_not_observed`, `active_observed`, `degraded`, or `drifted`. The
`installed` enum value is reserved for a future host-attested state and is not emitted by v1.
Installation alone is not observation: `installed_not_observed` remains explicit until a lifecycle
callback is actually admitted.

For each provider, the result includes connector availability, connection state, session and
quarantine counts, queued and dropped spool-event counts, the oldest short SalienceGate session
identifier, total local capture bytes, and closed drift codes. It reports neither storage paths nor
provider-native identifiers. JSON uses `capture-status/v1` with four bounded
`capture-provider-status/v1` records when no provider is selected.

Global status reports `not_installed`, `enabled`, `disabled`, or `drifted`, plus bounded project
child and exclusion counts. Its JSON schema is `global-capture-status/v1`.

## `sessions`

Artifact-compatible after installation:

```text
saliencegate sessions
  [--provider codex|claude-code|opencode|pi]
  [--state open|closed|quarantined]
  [--limit 1..100]
  [--json]
```

`sessions` drains capture state and lists only sessions bound to the current project. The default
limit is 20. Each item exposes a short SalienceGate session identifier, provider, state, bounded
event count, coverage-degraded flag, and local observation times. It never exposes the
provider-native session identifier. No configured store is a successful empty list; corrupted or
unusable existing state fails. JSON uses `capture-sessions/v1` and
`capture-session-list-item/v1`.

## `report`

Artifact-compatible after installation:

```text
saliencegate report (--latest | SESSION_ID)
  [--output PATH]
  [--replace]
  [--json]
```

`report` selects one current-project session, verifies the authenticated snapshot, drains the
matching spool boundary, normalizes only admitted structured records, and evaluates the installed
capability matrix. `--latest` selects the latest session for the current project; an explicit short
identifier must belong to that project. The complete report is canonical
`capture-session-report/v1` JSON. Human output is content-free and includes the same headline,
support, denominators, exclusions, limits, and fixed authority declarations.

The headline vocabulary is closed:

| Headline | Exact boundary |
|---|---|
| `memory_review_suggested` | At least one supported deterministic signal was observed and no quarantine or integrity failure blocks the positive conclusion. |
| `no_current_evidence` | The session is closed, the spool is authenticated and cleanly drained, an applicable detector met its absence minimum, no signal was observed, and no report limit remains. |
| `insufficient_evidence` | No supported signal was detected and an explicit limit blocks the negative conclusion, or quarantine/integrity failure blocks a positive signal; limits include open or quarantined state, gaps, drops, unavailable or pending spool, no applicable detector, insufficient applicable evidence, and unverified compatibility. |

All three remain `descriptive_observational`, `confirmatory=false`, `decision_authority=false`, and
`model_calls=0`. A positive headline is not an instruction; a negative headline is not proof that
memory was unnecessary.

`--output` atomically publishes an owner-private canonical file in an already controlled parent.
Existing output is preserved unless `--replace` is present, and replacement requires a valid report
for the same session. `--replace` without `--output` is invalid. The canonical report is bounded to
4 MiB at the command publication boundary.

## `delete`

Artifact-compatible after installation:

```text
saliencegate delete SESSION_ID [--json]
saliencegate delete --all --project PATH --confirm [--json]
```

The first form deletes one current-project session, including its captured events and feedback, and
retains a content-free authenticated tombstone so the same request is retry-safe. The second form
deletes every capture connection, session, feedback row, health row, transport receipt, and
tombstone for exactly one explicitly named project. Every provider connection for that project must
first be disabled with `disconnect`; otherwise the command exits 3 with the bounded recovery
instruction.

Both forms enable SQLite secure-delete, drain under spool maintenance, checkpoint and truncate the
WAL, and return one `capture-delete/v1` receipt. This is application-level deletion, not a guarantee
that backups, filesystem snapshots, storage-device remanence, or previously exported reports were
erased. Deletion never removes another project's records or the installation key.

## `feedback`

Artifact-compatible after installation:

```text
saliencegate feedback SESSION_ID \
  --label memory-needed|not-memory-needed|uncertain \
  [--json]
```

`feedback` records one bounded human label for a **closed** captured session in the current project.
The session must be addressed by its short SalienceGate identifier; open, quarantined,
provider-native, and cross-project sessions are rejected. Repeating the current label is an
idempotent success. Changing the label records an authenticated revision without copying captured
events or provider content.

The human result contains only the short session identifier, label, write disposition, and bounded
revision count. `--json` emits one compact canonical `capture-feedback-receipt/v1` object and also
includes the local label timestamp. Missing, open, quarantined, or wrong-project sessions exit 2;
local configuration or store contention exits 3; unavailable local capture state exits 4; and
authenticated-store corruption exits 5. Error text never echoes the session identifier, label,
project, path, or exception detail.

The command never exports a dataset or runs classification evaluation. Those are separate explicit
Python APIs described in the [feedback and evaluation reference](evaluation.md). Recording a label
does not enable a reminder, injection, or any other active behavior.

## `doctor`

Artifact-compatible after installation:

```bash
saliencegate doctor \
  [--repository PATH_OR_:memory:] \
  [--key ABSOLUTE_PATH] \
  [--endpoint HTTP_BASE_URL] \
  [--capture] \
  [--json]
```

The command checks, in order:

1. Python `>=3.11,<3.14`;
2. SQLite `>=3.24`;
3. FTS5 availability in an in-memory SQLite database;
4. repository path type and permissions;
5. installation-key path, type, size, ownership, and permissions;
6. optional endpoint syntax.

The default repository path is `saliencegate.sqlite3`; `:memory:` is also accepted. If `--key`
is omitted, the normal per-user configuration path is inspected. An explicit key path must be
absolute. An existing key must be a regular 32-byte file and, on POSIX, owner-only and owned by
the current user. Repository and key symlinks are rejected.

`--endpoint` performs syntax validation only. It accepts a credential-free `http` or `https`
base URL with a hostname and no query or fragment; it never attempts a connection. Omitting it
produces an optional `skip` result. `doctor` does not create a repository or key.

JSON uses `doctor/v1`, containing `status`, `ok`, and six `doctor-check/v1` records. A required
Python, SQLite, or FTS5 failure exits 4; another required failure exits 3.

`--capture` adds one strictly read-only project capture check. It does not open mutable SQLite or
spool handles, create a key, launch a provider, install a connector, or drain events. The added
check reports `not_configured` as an optional skip, `ready` after the store, spool, installation,
and managed files pass their read-only integrity checks, or `degraded` as a required failure. JSON
uses the enclosing `capture-doctor/v1` report.

## `demo`

Artifact-compatible after installation:

```bash
saliencegate demo [--json]
```

`demo` regenerates the 32-case StateDecayBench smoke diagnostic in memory. It validates the
eight-family and 16/16 intervene/silence geometry, evaluates every paired continuation through the
deterministic oracle, and reports one domain-separated result digest.

The command reads no fixture, repository, environment variable, credential, or Git state. It opens
no socket, imports no optional model runtime, calls no model, and writes no artifact. Its report is
always `synthetic_diagnostic`, `confirmatory: false`, and `external_claims_supported: false`. A
successful run verifies deterministic mechanics; it does not measure agent task efficacy.

JSON uses `cli-demo-report/v1` and contains exactly:

```text
schema_version, status, suite_id, evidence_level, diagnostic, synthetic,
confirmatory, external_claims_supported, external_claims_assessment,
scenario_count, family_count, intervene_count, silence_count,
oracle_passed, oracle_failed, result_digest
```

The frozen geometry is 32 scenarios, eight families, 16 intervene labels, 16 silence labels, 32
oracle passes, and zero failures. The result digest frames canonical scenario bytes before canonical
oracle-result bytes under a demo-specific domain; it is not the smoke artifact manifest digest.

An installed distribution also exposes the separate `saliencegate-review` entry point for the
StateDecayBench v2 human workflow. Its four commands, local-storage requirements, publication
warning, correction behavior, and closed generation gate are documented in the
[StateDecayBench v2 review reference](state-decay-v2-review.md).

## `shadow analyze`

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

`shadow analyze` reads a bounded, private `shadow-input/v1` NDJSON trace and atomically publishes
one owner-only canonical `shadow-run-report/v1` file. It defaults to the in-memory repository,
`capture_scope=unknown`, and `source_adapter=saliencegate-shadow/v1`. A SQLite path enables durable
authenticated-prefix recovery. The command opens no socket, calls no model, reserves no budget,
creates no memory cycle or revision, and has no delivery or decision authority.

Input kinds are `run_start`, `action`, `tool_result`, `test_result`, `observation`,
`controller_error`, and `run_end`. The first row must be `run_start`; a `run_end`, when present, is
final. `complete_run_declared` requires that terminal row. Results refer to an earlier action by
source ID. Exact repeated rows are retries and are excluded from unique-event denominators.

With SQLite, a rerun reuses an exact existing prefix and submits only missing unique events. A
complete prefix makes the ledger append a no-op. A conflicting prefix fails without an extension.

The limits are 10,000 rows, 2 MiB per encoded line, 64 MiB of complete input, and 128 MiB of
canonical report. Input must be a stable, user-owned regular file that is not group- or
world-writable. The output parent must already be private, and successful output has mode `0600`.
Symlinks, hard links, changing files, unsafe ancestors, and aliases among input, SQLite, SQLite
sidecars, and output are rejected.

Existing output is never overwritten by default. `--replace` accepts only a valid report bound to
the same run, exact input bytes, normalized input, capture provenance, redaction-policy tag, and
detector profile. Corrupt or unrelated output remains unchanged.

The report contains authenticated initial-head commitments, input and configuration digests,
capture provenance, row and unique-event denominators, ordered payload-free observations, all four
supported detector outcomes, exact abstention counts, four-state heuristic counts, co-occurrences,
event and phase counts, first flagged sequence, a self digest, and the fixed
`descriptive_observational` evidence boundary. `--json` emits the smaller canonical
`shadow-command-report/v1` with no filesystem paths or source-event IDs.

| Exit | Shadow meaning |
|---:|---|
| `0` | Success, help, version, or closed output pipe |
| `2` | Invalid syntax, input, lifecycle, parent, collision, bound, path, alias, or replacement request |
| `3` | Installation-key, SQLite, secure-publication, retry-exhaustion, or incompatible-repository failure |
| `5` | Corrupt replacement, canonical mismatch, or report-digest failure |
| `70` | Detector-contract or unexpected internal failure |
| `130` | Keyboard interruption |

See the [Shadow Mode reference](shadow-mode.md) for complete Python and NDJSON examples, detector
scope, report fields, replacement bindings, and evidence limitations.

## `shadow analyze-atif`

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

`shadow analyze-atif` adapts one bounded ATIF JSON trajectory through an explicitly selected sealed
field-shape profile, runs the same deterministic Shadow detectors, and publishes one owner-private
canonical `shadow-trace-report/v1`. It never guesses a profile or working directory. The command has
no model, endpoint, provider, API-key, memory-schedule, output-content parser, or complete-capture
option; it opens no socket and does not execute trajectory content.

The CLI aliases have these exact meanings:

| Alias | Report profile | Accepted source | Selected action | Outcome evidence |
|---|---|---|---|---|
| `harbor-codex-v1` | `harbor-codex/v1` | ATIF-v1.7, agent `codex` | `exec_command` | exact signed 32-bit integer exit metadata from a sealed path; producer-claimed |
| `harbor-terminus-2-v1` | `harbor-terminus-2/v1` | ATIF-v1.6/v1.7, agent `terminus-2` | LF-submitted `bash_command` | none |

The Codex claim covers the pinned Harbor converter field shape only; it does not guarantee a Codex
CLI runtime version. Terminus terminal text never becomes success or failure evidence. Both profiles
declare `producer_authentication=none`, select only root-segment events, report continued and
embedded-subagent presence without traversal, and set complete execution-session coverage to false.
`capture_scope` is fixed to `selected_events`.

The input must be a stable owner-private single-link regular file. Parsing and adaptation complete
before the command loads or creates the normal owner-private SalienceGate installation key. This is
local ledger-authentication state, not a provider credential; its first use may initialize the
platform-default key file. The repository defaults to `:memory:`. A private SQLite path enables
durable authenticated-prefix resume. The output parent must already be private, and output is
atomically published with mode `0600`. Source, SQLite, sidecar, and output aliases; unsafe
permissions; hard links; symlinks; and state changes fail closed.

Existing output is preserved unless `--replace` is supplied. Replacement requires exact agreement
on source, profile, adapter and detector configuration, provenance, redaction-policy tag, nested
Shadow binding, initial repository state, and complete outer report identity. With SQLite, the exact
comparison preview may open the repository, apply schema migrations, and recover authenticated
derived state. It appends no trace suffix, and a mismatch publishes no report. Analysis appends only
the missing exact trace suffix in one atomic batch. The append is a no-op when the complete trace is
already present. Corrupt or unrelated output is never overwritten.

Human output contains source totals, root-coverage declarations, mapped and ignored call/result
counts, structured-outcome coverage, all four heuristic dispositions, the profile evidence matrix,
per-detector outcome and abstention counts, the evidence-sufficient denominator, timestamp mode,
producer authentication, outcome authority, compatibility-manifest digest, report digest, and the
fixed observational boundary. `--json` emits one canonical `shadow-atif-command-report/v1`, not the
full report. Neither form contains commands, tool output, source IDs, working directories, or source,
repository, and output paths.

The principal bounds are 64 MiB of source bytes, 10,000 root steps, 10,000 tool calls, 10,000
observation results, 1,000 mapped Shadow records, and 130 MiB of canonical outer report.

| Exit | ATIF Shadow meaning |
|---:|---|
| `0` | Success, help, version, or closed output pipe |
| `2` | Invalid syntax, ATIF/profile input, bound, path, alias, or replacement request |
| `3` | Installation-key, SQLite, secure-publication, retry-exhaustion, or incompatible-repository failure |
| `5` | Corrupt replacement, canonical mismatch, or report-integrity failure |
| `70` | Internal adapter/detector contract failure |
| `130` | Keyboard interruption |

The [ATIF example guide](../../examples/atif-shadow/README.md) contains copy-paste Codex and
Terminus 2 commands for an installed wheel. The [Shadow Mode reference](shadow-mode.md) documents
the one-call API, lower-level trace sessions, omission matrix, and provenance report.

## `replay`

Artifact-compatible after installation:

```bash
saliencegate replay TRACE.jsonl \
  --output ARTIFACT_DIR \
  [--responses RESPONSES.jsonl] \
  [--replace] \
  [--json]
```

`replay` executes the frozen replay profile. It is not a configurable live-model runner. It uses
versioned JSONL trace and response fixtures, a deterministic scripted policy, an in-memory
repository, and a delivery sink that fails intentionally without an external side effect.

If `--responses` is omitted, exactly one response fixture must be discoverable:

- `<trace-directory>/<trace-stem>_responses.jsonl`; or
- when the trace directory is named `runs`,
  `<parent-of-runs>/models/<trace-stem>_responses.jsonl`.

Zero or two matching candidates are rejected. An explicit `--responses` disables discovery. The
response count cannot exceed the event count. For `N` responses, the first `event_count - N`
events use scripted silence and the last `N` use scripted invocation.

The trace manifest and every record digest are verified. The trace must contain one UUID4 run,
contiguous ordinals, unique source and event identities, parents that refer only to earlier
events, and non-decreasing timestamps. Trace and response sources must be stable regular files,
not symlinks or multiply linked files. The principal limits are 100,000 events or responses,
8 MiB per trace, and 64 MiB per response fixture.

Classification is automatic:

- all events marked `synthetic_fixture` produce `synthetic_digest_only`;
- any other valid trace produces `user_redacted`.

The CLI always exports `evidence_level: exploratory`. A clean source revision or installed
distribution can make its provenance `confirmatory_eligible`, but that does not change the
evidence level and does not satisfy `validate --require-confirmatory`.

The `cli-replay-report/v1` JSON result contains:

```text
schema_version, status, run_id, trace_digest, result_digest,
classification, confirmatory, manifest_digest, overall_content_digest, counters
```

`counters` records events, decisions, invocations, cycles, model calls, deliveries, successful
deliveries, and outcomes. Fixture model-call receipts are not live or billed calls.

An existing output is never overwritten by default. `--replace` is accepted only when the
existing tree is a completely valid replay artifact for the same run. Unrelated or corrupt data
is preserved and the command fails closed.

## `algorithm replay`

Artifact-compatible after installation:

```bash
saliencegate algorithm replay TRACE.jsonl \
  --condition CONDITION \
  [--responses RESPONSES.jsonl] \
  --output ARTIFACT_DIR \
  [--replace] \
  [--json]
```

`algorithm replay` executes the closed comparison path with a frozen, local response
fixture. It never contacts an endpoint, imports the optional live-model runtime, reads an API
credential, or performs a billed model call. It accepts only the versioned paper-two-phase prompt
bundle, the fixed `gpt-oss:20b` replay profile, and one of these condition identifiers:

- `no_memory` disables the memory algorithm and forbids `--responses`;
- `fixed_step` performs the paper two-phase cycle at fixed boundaries;
- `retrieval_always` performs deterministic retrieval followed by the Phase 1 path;
- `always_inject` performs the paper two-phase cycle and requires safe intervention selection.

The three active conditions require an explicit `--responses` path. There is no response-fixture
discovery and no condition fallback. A fixture prepared for a different condition fails because
its ordered requests and responses do not bind the selected run. `no_memory` has no model calls
and therefore has no empty response fixture.

The trajectory and response inputs must be stable, bounded, single-link regular files. Symlinks,
hardlinks, malformed records, duplicate identities, unsupported versions, incomplete response
sets, and changed files are rejected. The trajectory must contain only records classified
`synthetic_fixture`; `algorithm-artifact/v1` does not permit user data to be relabelled as
synthetic.

The command executes the fixture twice through the same closed path and publishes only when both
results are canonically identical. The resulting artifact is `synthetic_digest_only`,
`exploratory`, and never confirmatory. It contains ordered receipts and commitments, not raw
prompts, responses, reasoning, or credentials. The CLI report does not expose provider-token
totals; in the committed fixtures they are unavailable. Reported canonical token counts are
replay-attested fixture measurements and do not represent a bill.

Because frozen replay performs no inference, its hardware attestation uses deterministic
`not-applicable-replay` labels and the schema-minimum value `1` for count fields. These are
sentinels, not measurements of the host that ran the CLI.

For output-path-independent reruns, a Git checkout is conservatively recorded with
`dirty_worktree: true` even when discovery initially finds it clean. The full commit remains
recorded, but this offline CLI never claims clean-revision provenance; the Python export API can
accept an explicit exact revision attestation when that distinction is required.

The `cli-algorithm-replay-report/v1` JSON object contains:

```text
schema_version, status, condition, run_id, run_digest, result_digest,
classification, confirmatory, manifest_digest,
overall_content_digest, calls, canonical_input_tokens,
canonical_output_tokens, canonical_token_equivalents,
interventions, grounding_rejections
```

Human output reports the same condition, counts, classification, and four execution/artifact
digests without printing fixture content. An existing destination is not overwritten. `--replace`
is authorized only for a completely valid algorithm artifact with the same run and condition;
unrelated, cross-condition, or corrupt data is preserved.

Missing or unknown command-line choices exit 2 with `error: invalid command line`. Unsafe or
inconsistent trajectory, response, condition, or destination inputs also exit 2 through one
value-free algorithm diagnostic. A corrupt artifact encountered during validation exits 5 with
`error: artifact validation failed`. Input paths and parser details are never echoed.

## `benchmark`

Artifact-compatible after installation:

```bash
saliencegate benchmark state-decay-smoke \
  --output ARTIFACT_DIR \
  [--replace] \
  [--json]
```

`state-decay-smoke` is the only registered smoke suite. The command generates its dataset from
the versioned generator rather than copying the committed fixture. It then reconstructs and runs
the deterministic paired-continuation oracle before publishing the artifact.

Fixed suite properties are:

- seed `20260711`, generator `v1`, suite `v1`;
- 32 scenarios across eight families;
- four linked instances per family;
- two `intervene` and two `silence` labels per family;
- 32 expected oracle passes and zero expected failures;
- synthetic, balanced, diagnostic, and non-confirmatory.

The families are `forgotten_requirement`, `stable_environment_fact`,
`failed_prior_attempt`, `retained_diagnosis`, `neglected_subgoal`, `stale_memory`,
`conflicting_evidence`, and `irreversible_action`.

The suite does not execute an action model or measure task success, tokens, model cost, or
latency. Its output fixes `external_claims_supported: false` and
`external_claims_assessment: insufficient`.

The `cli-benchmark-report/v1` JSON object includes suite, generator, seed, balance, counts,
fixture digest, reconstructed-oracle digest, overall content digest, and manifest digest.
`--replace` is restricted to a valid artifact of the same suite, version, and fixture digest.

Maintainers regenerate the committed source fixture with:

Run from a checkout:

```bash
uv run --locked python -m saliencegate.benchmarks.state_decay.generator \
  --output benchmarks/state_decay --replace
```

That committed fixture uses `smoke_manifest.json`. The runtime artifact uses `manifest.json`, so
the committed fixture directory is not directly accepted by `saliencegate validate`.

## `inspect`

Artifact-compatible after installation:

```bash
saliencegate inspect RUN_UUID4 \
  --artifact ARTIFACT_DIR_OR_MANIFEST \
  [--json]
```

`inspect` accepts a replay artifact directory or a regular file named exactly `manifest.json`.
It does not accept a StateDecayBench artifact. The requested UUID4 must equal the manifest run
identifier.

The complete replay artifact is validated before output is produced. Inspection uses the
already-validated in-memory view and does not reopen component files. The
`cli-inspect-report/v1` output contains:

```text
schema_version, status, run_id, manifest, execution, decisions,
budgets, cycles, deliveries, outcomes, attestations
```

It exposes typed lifecycle records and digests while omitting trace payloads, memory text, model
prompts and responses, rendered reminder text, raw delivery receipts, request identifiers, and
local paths. Even for `synthetic_raw`, `synthetic.json` is never returned by inspection.

This is data minimization, not anonymity. UUIDs, timestamps, reason codes, revision identifiers,
model identifiers, and digests remain metadata and may be correlatable.

## `validate`

Artifact-compatible after installation:

```bash
saliencegate validate ARTIFACT_DIR_OR_MANIFEST \
  [--expected-digest MANIFEST_DIGEST] \
  [--require-confirmatory] \
  [--json]
```

A directory is resolved to its `manifest.json`. A file input must be a regular file named exactly
`manifest.json`; missing, renamed, or symlink inputs are rejected as invalid paths.

`--expected-digest` compares the artifact's domain-separated `manifest_digest`. It does not
accept the raw file SHA-256, `overall_content_digest`, or replay `result_digest`. Omitting it
returns `expected_digest_matched: null`; a correct value returns `true`. A malformed or different
value exits 5 rather than returning a successful report with `false`.

`--require-confirmatory` checks `evidence_level: confirmatory`, not revision eligibility. CLI replay
artifacts and algorithm-run artifacts are exploratory, while StateDecayBench artifacts are always
non-confirmatory, so all three CLI-produced families fail this option by design.

Before selecting a validator, the command reads a bounded, canonical manifest without following a
symbolic link. `artifact_kind` selects `replay_run` or `algorithm_run`; the legacy
`state-decay-smoke-manifest/v1` schema selects the benchmark validator. Unknown kinds and schemas
fail closed. This preflight does not replace the complete family-specific filesystem and digest
validation.

Replay success uses `artifact-validation-report/v1`:

```text
valid, integrity_valid, structurally_valid, expected_digest_matched,
grounding_assurance, confirmatory, manifest_digest,
overall_content_digest, component_count
```

`grounding_assurance` is `producer_attested_digest_only`: validation proves the recorded
producer attestations and cross-component bindings, not semantic entailment.

Benchmark success uses `benchmark-validation-report/v1` and reports
`assurance: deterministic_synthetic_oracle`, `confirmatory: false`, and
`external_claims_supported: false`, together with the fixture, oracle, content, and manifest
digests.

Algorithm success uses `algorithm-validation-report/v1`:

```text
valid, structurally_valid, self_consistent, expected_digest_matched,
source_result_assurance, confirmatory, manifest_digest,
overall_content_digest, component_count
```

`source_result_assurance: producer_attested` means a digest-only artifact is structurally and
cryptographically self-consistent but its source result was not reconstructed from raw content.
`recomputed_from_raw` is reserved for an explicitly classified synthetic-raw artifact whose
complete algorithm result was reprojected during validation. Both assurances are exploratory and
non-confirmatory.

For artifact layouts, digest construction, classification, validation invariants, and atomic
replacement behavior, see [Artifact reference](artifacts.md).
