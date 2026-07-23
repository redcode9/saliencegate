# Universal Shadow Capture integration contract

This page freezes the normative provider contract for Universal Shadow Capture v1 in the
unpublished local SalienceGate 0.2.0 candidate. A connector conforms only when its installed package
contains the matching audited capability manifest. Documentation alone is not evidence that a
connector is installed or has observed an event; capture status must report that separately.

The contract was audited on **2026-07-19**. The allowlists below describe native fields with
evidence authority; those fields may exist briefly in bounded memory. Provider identifiers and
native tool inputs are reduced with a domain-separated HMAC before durable storage. Every unlisted
field is ignored by the evidence adapter, and no raw field listed here is reportable. A connector
may separately validate a content-free routing envelope: Codex and Claude Code use `cwd` in memory
to authenticate the matching project installation before invoking their adapters, even when that
event gives `cwd` no evidence authority.

## Compatibility summary

| Profile | Audited host | Project-local installation | Selected lifecycle |
|---|---|---|---|
| `codex-hooks/v1` | Codex CLI `0.144.6` | `<repo>/.codex/config.toml`; the project and exact hook definition must pass Codex trust review | `SessionStart`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PreCompact`, `SubagentStart`, `SubagentStop`, `Stop` |
| `claude-code-hooks/v1` | Claude Code `2.1.204` | `<repo>/.claude/settings.local.json`; normal project-local settings trust applies | `SessionStart`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`, `PermissionDenied`, `SubagentStart`, `SubagentStop`, `Stop`, `StopFailure`, `SessionEnd` |
| `opencode-plugin/v1` | OpenCode `1.18.3`, release commit [`127bdb3`](https://github.com/anomalyco/opencode/commit/127bdb30784d508cc556c71a0f32b508a3061517) | `<repo>/.opencode/plugins/saliencegate.js`, the documented project plugin directory | `message.part.updated`, `session.idle`, `session.error`, `session.compacted`, `session.deleted`, and plugin `dispose` |
| `pi-extension/v1` | `@earendil-works/pi-coding-agent` `0.80.10`, tag commit [`8dc7883`](https://github.com/earendil-works/pi/commit/8dc78834cde4e329284cf505f9e3f99763df5529) | `<repo>/.pi/extensions/saliencegate.ts`; Pi must trust the project before loading it | `session_start`, `before_agent_start`, `tool_execution_start`, `tool_execution_end`, `agent_settled`, `session_compact`, `session_tree`, `session_shutdown` |

The native runner workflow is prepared but has not been run remotely from this unpublished branch.
Ubuntu 24.04 and Windows 2025 are **implementation ready, remote verification pending**. The
macOS 15 job is prepared as the third native boundary; remote evidence for all three runners is the
separate R01 gate. Local contract tests do not promote any platform to remotely verified status.

Installation must never bypass, pre-approve, or weaken provider trust. Project-local JavaScript,
TypeScript, hook commands, and sidecars execute with the user's operating-system authority. Users
should inspect managed files before trusting them.

Codex connect accepts the audited `0.144.6` release and newer patch releases in the same `0.144.x`
line as schema-compatible but unverified. Older releases and different minor or major lines fail
closed pending a new audit. Each accepted patch change creates a new local connection generation;
connect discovers the live version and upgrades the authenticated receipt. Read-only inspection
authenticates that receipt without launching Codex; a later connect probes again before deciding
whether an upgrade is needed. Connect also refuses a project layer that sets `features.hooks=false`,
the deprecated `features.codex_hooks=false`, or `allow_managed_hooks_only=true`; it never changes
those user or administrator policies.

Claude Code connect accepts the audited `2.1.204` release and newer patch releases in the same
`2.1.x` line as schema-compatible but unverified. Normal connect performs the bounded local
`claude --version` probe; dry-run, fixture validation, status, packaging, and artifact smoke never
launch Claude. Existing unrelated top-level settings and existing hook groups are byte-preserved in
`.claude/settings.local.json`; SalienceGate appends one distinct owned group to each selected event
array. Explicit hook disablement and malformed groups on selected event arrays fail before the
probe. Disconnect removes the authenticated owned spans and exactly restores an otherwise unchanged
config.

SalienceGate validates the hook control surface it owns, not Claude Code's entire evolving settings
schema. A malformed foreign setting can therefore cause Claude to reject that settings layer,
including the appended handlers. Connect proves an authenticated installation, while status remains
`installed_not_observed` until an actual lifecycle callback proves activation; this is the explicit
`host_rejected_foreign_settings_layer` coverage exclusion.

## Operations, recovery, and uninstall

Use the same project root for every lifecycle command:

```text
saliencegate connect PROVIDER --project PROJECT --dry-run
saliencegate connect PROVIDER --project PROJECT
saliencegate status PROVIDER --project PROJECT
saliencegate disconnect PROVIDER --project PROJECT
```

The dry-run is the review boundary. It validates the provider-specific managed-file plan and
collisions without creating a key, store, receipt, launcher, or provider file. Its read-only Git
probe reports whether those not-yet-created paths are ignored, would be surfaced by Git, are already
tracked, lie outside a work tree, or could not be classified. It never edits `.gitignore`; review
the two project-local OpenCode or Pi assets before trusting the project. A normal connect creates
owner-private operational state, registers an authenticated pending generation, publishes the
integration, and enables event admission last. It does not launch an agent session or prove that
the provider loaded the integration. `status` therefore distinguishes
`installed_not_observed` from `active_observed` and reports exact drift and degradation codes
without revealing paths.

| Provider | Managed project surface | Activation, recovery, and version boundary |
|---|---|---|
| Codex | Owned marked hook span in `.codex/config.toml` | The project and hook must pass Codex trust review. Connect probes `0.144.6` or a newer `0.144.x` patch, rejects other lines and disabling policy, and records each accepted patch as a new generation. Rerun connect after an interrupted authenticated installation; disconnect removes only the owned span and restores otherwise unchanged bytes. Hosted/specialized tools and missing callbacks remain exclusions after recovery. |
| Claude Code | One owned hook group per selected event in `.claude/settings.local.json` | Normal settings trust still applies. Connect probes `2.1.204` or a newer `2.1.x` patch and records non-baseline patches as schema-compatible but unverified. Rerun connect after an interrupted owned write; disconnect removes only authenticated owned groups. A foreign setting rejected by the host, `StopFailure` before the first prompt, and session-end/resume ambiguity remain coverage limits. |
| OpenCode | `.opencode/plugins/saliencegate.js` and `.opencode/plugins/saliencegate.bootstrap.json` | The sealed connector is exact for `1.18.3`; connect does not launch OpenCode. Reload or start the trusted project and wait for an admitted callback before expecting `active_observed`. Rerun connect to finish an interrupted authenticated bridge publication; disconnect removes exactly the bound bundle and bootstrap. A batch lost before any chunk is received and session errors without the required session identity remain undetectable exclusions. |
| Pi | `.pi/extensions/saliencegate.ts` and `.pi/extensions/saliencegate.bootstrap.json` | The project must be trusted before Pi loads the exact `0.80.10` connector. Reload or start Pi before expecting an observed status. Rerun connect to finish an interrupted authenticated bridge publication; disconnect removes exactly the bound extension and bootstrap. Missing shutdown, manual-compaction interruption, tool-less extension-triggered runs, and wholly lost batches remain exclusions. |

Operational receipts, journals, locks, and launchers live below the private SalienceGate state root,
under `integrations/<project-locator>/<provider>/`; the project locator is a SHA-256 location digest,
not a portable public project identifier. Do not copy or hand-edit those files. A retry uses their
authenticated identities to complete or reverse an interrupted operation. Unexpected drift,
ownership ambiguity, an unrelated look-alike file, or a changed receipt fails closed for manual
inspection; reconnecting cannot turn a missed callback, dropped batch, or prior gap into complete
coverage.

`disconnect` is the connector uninstall operation. It first stops admission and drains the bounded
spool, then removes only authenticated owned configuration or bridge files and disables the local
connection generation. It intentionally retains captured sessions, feedback, store health,
deletion tombstones, the SQLite database, spool boundary, and installation key. Use `sessions` and
`report` after disconnect to inspect retained evidence. Data removal is a separate explicit
`delete SESSION_ID` or, after every provider is disconnected,
`delete --all --project PROJECT --confirm` operation. There is no automatic retention period or
time-to-live.

For diagnosis, run `saliencegate doctor --capture` for a strictly read-only integrity check, then
`status`. A missing or corrupted installation key is not regenerated over existing state because a
new key could not authenticate that state. Back up the key, SQLite database, and spool together;
include the provider operational directories when the installed connectors must also be
recoverable. Restoring only one component is not a supported recovery path. HMAC recovery does not
detect rollback of all components to an older internally valid snapshot.

## Evidence field allowlists

A missing session or call-correlation field is critical: the affected evidence is omitted and
coverage degrades. Optional fields never gain authority merely because they are present.

### Codex

| Event | Evidence fields | Evidence authority |
|---|---|---|
| `SessionStart` | `session_id`, `hook_event_name` | Opens an observed window. |
| `PreToolUse` | `session_id`, `cwd`, `hook_event_name`, `tool_name`, `tool_use_id`, `tool_input` | `tool_use_id` is exact parent correlation. `cwd`, `tool_name`, and `tool_input` are HMAC-reduced into workspace and opaque action identity. |
| `PostToolUse` | `session_id`, `hook_event_name`, `tool_use_id` | Closes the correlated action, but supplies no authorized success/failure status. |
| `PermissionRequest` | `session_id`, `hook_event_name` | Validates the selected lifecycle shape but creates no semantic intake. The documented input has no `tool_use_id` or denial result. |
| `PreCompact` | `session_id`, `hook_event_name` | Validates the selected lifecycle shape but creates no semantic intake. |
| `SubagentStart`, `SubagentStop` | `session_id`, `hook_event_name`, `agent_id` | `agent_id` is a correlation input only and is HMAC-reduced. |
| `Stop` | `session_id`, `hook_event_name`, `turn_id` | When `turn_id` is present, ends one observed turn, not the task or session; without it no intake is created. |

`transcript_path` and `agent_transcript_path` are never opened. Outside `PreToolUse`, `cwd`,
`tool_name`, and `tool_input` are ignored. `source`, `turn_id` outside `Stop`, `trigger`,
`agent_type`, `tool_response`, `last_assistant_message`, `model`, `permission_mode`,
`stop_hook_active`, and every other field are ignored. In particular, no response text or shell
exit status is parsed. Hosted tools and specialized paths outside the local function-tool hook path
remain an explicit coverage exclusion.
`write_stdin` is continuation transport for an existing unified-exec call and must not create a
second action.

Codex exposes neither an admitted provider timestamp nor a sequence number. `turn_id` and
`tool_use_id` correlate records but do not establish causal order. Receipt order and time are
local observation metadata only.

### Claude Code

The installed transport requires `session_id`, absolute `cwd`, and `hook_event_name` before it
loads the adapter. `cwd` authenticates the project-local installation and is admitted as an
opaque workspace digest only for `PreToolUse`. `prompt_id` is HMAC-reduced only for required
`StopFailure` correlation; it is not a sequence number.

| Event | Additional consumed fields | Evidence authority |
|---|---|---|
| `SessionStart` | none | Opens an observed window. |
| `PreToolUse` | optional `tool_name`, `tool_use_id` | This is only a pre-hook proposal because another hook can rewrite or block it. `tool_use_id` is exact parent correlation; an optional tool name selects one closed generic proposal class. No exact executed-action identity is created. |
| `PostToolUse` | `tool_use_id` | The event discriminator is provider-claimed structured success. |
| `PostToolUseFailure` | `tool_use_id`, optional exact boolean `is_interrupt` | The event discriminator is provider-claimed structured failure; `is_interrupt` selects only the generic interrupted class. |
| `PostToolBatch` | `tool_calls[].tool_use_id` | Validates a bounded, unique reconciliation set and emits no semantic intake. A batch is not an outcome. |
| `PermissionDenied` | `tool_use_id` | The event discriminator is provider-claimed denial from the pinned auto-mode classifier path; denials outside that path are not covered. |
| `SubagentStart` | `agent_id` | The agent identifier is correlation-only and HMAC-reduced. |
| `SubagentStop` | `agent_id` | Validates the bounded selected shape but creates no semantic intake because another hook can continue the subagent. |
| `Stop` | none | Validates the bounded selected shape but creates no semantic intake because another hook can continue the turn. |
| `StopFailure` | required `prompt_id` | Emits one provider-callback-reported generic controller/API failure; raw error values and details are excluded. A failure before the first prompt has no `prompt_id`, is omitted, and degrades coverage. |
| `SessionEnd` | none | Validates the bounded selected shape but creates no semantic intake. Native session identifiers survive resume, so the capture session remains open. |

`transcript_path` and `agent_transcript_path` are never opened. `tool_response`,
`PreToolUse.tool_input`, `tool_calls[].tool_input`, the optional raw
`tool_calls[].tool_response`, `duration_ms`, the top-level `PostToolUseFailure.error` text, the
top-level `PermissionDenied.reason` text,
`error_details`, `last_assistant_message`, `background_tasks`, `session_crons`,
`stop_hook_active`, `permission_mode`, every event's optional `effort` object, and the
`SessionStart` source, model, agent type, and session title are ignored. The audited fixture uses
valid `2.1.204` `StopFailure.error` and `SessionEnd.reason` values, but the adapter does not persist
or distinguish them. Every other unlisted field is ignored as well. SalienceGate's event handlers
return no decision or context and leave normal Claude behavior unchanged.

`SessionEnd` does not close the authenticated capture session, and a resumed `SessionStart` for the
same native identifier replays the existing open session before later tool records append. An open
session can still produce a positive memory-review suggestion from observed evidence, but it cannot
support a negative claim that an event was absent. Disconnect likewise leaves already authenticated
session evidence open; the normal bounded event limit still applies.

Claude supplies no admitted timestamp or sequence field. Correlation by `prompt_id` and
`tool_use_id` is provider claimed; receipt order and time are local observation metadata only.

### OpenCode

The plugin accepts `message.part.updated` only after checking
`properties.part.type === "tool"`. For that event it consumes:

- `properties.part.sessionID` as the native session identity;
- `properties.part.type`, `callID`, `tool`, `state.status`, and `state.input`.

OpenCode's public `Event` union does not declare a top-level `id`, although the pinned runtime
currently injects one. The plugin therefore treats `id` as optional producer-event correlation,
never as sequence authority. The runtime can also add an outer `properties.sessionID`; when it is
present the plugin checks that it agrees with `properties.part.sessionID`, but it is not required.

`state.input` is critical and is canonicalized only to derive opaque action identity. An event
without it is omitted and degrades coverage. The closed state machine has this authority:

| `ToolPart.state.status` | Capture meaning |
|---|---|
| `pending`, `running` | Action observed; no outcome yet |
| `completed` | Provider-claimed structured success |
| `error` | Provider-claimed structured failure |

`session.idle` and `session.compacted` use their public `properties.sessionID` as native session
identity. `session.deleted` instead uses the public `properties.info.id`; an outer runtime-added
`properties.sessionID`, when present, is only a consistency check. Each of these session events may
carry the same runtime-added optional top-level `id`. `session.error` has no critical identity: its
`properties.sessionID` is optional; without one the bridge cannot attribute the failure and emits
no session intake. `dispose` has no payload and flushes already-reduced in-memory state.

Part and message IDs, the optional part delta, runtime-added event time, part metadata, and
`state.raw`, `state.output`, `state.error`, `state.title`, `state.metadata`, `state.time`,
`state.attachments`, and `state.compacted` are ignored. The structured `session.error` error and
all `session.deleted.properties.info` fields except `info.id` are ignored too. Non-tool part
variants are rejected by their discriminant without traversing their content. The plugin must not
call `session.messages()`, `session.get()`, or another history materializer; child sessions remain
separate by their native IDs, while parent-session metadata remains unavailable or ignored in v1.

`callID` is exact per-action parent correlation but not authentication. Neither the optional
runtime event ID nor any event or tool-state time field supplies admitted causal sequence; receipt
order and time are local observation metadata only.

For each received transport batch, authenticated chunk receipts declare the batch length and each
chunk's index. Once any chunk from a batch arrives, a missing first, middle, or final chunk is
detectable as an incomplete batch. If every chunk in a batch is lost before receipt, the receiver
has no batch record to inspect; absent a later successful flush carrying a content-free gap marker,
that wholly unobserved batch cannot be detected. This is the explicit
`fully_unobserved_transport_batch` coverage exclusion.

Under bounded in-process pressure, the plugin may replace middle records with a content-free
transport gap, but it reserves and schedules one terminal control per buffered session. If a
launcher has already attempted a terminal-bearing batch and returns an ambiguous failure, the
plugin does not replay that terminal under a fresh batch identity; the receiver's authenticated
receipt state remains the only authority for resolving an attempted delivery.

If atomic receipt admission finds the local store busy, the bounded fallback queues only the
authenticated window start, one session-stable content-free gap marker, and an observed terminal
close when present. It deliberately discards the failed chunk's middle evidence instead of
reordering records whose provider sequence is unavailable. A later spool drain therefore preserves
window validity while reporting degraded coverage; it never promotes fallback order into provider
causal authority.

If that fallback spool already has a large valid backlog, the provider-deadline path validates its
closed filename inventory but does not authenticate and sort every queued record. It instead records
the whole bounded fallback as quota-dropped behind one authenticated global degradation barrier.
Later bridge chunks remain fenced from direct admission after an ordinary drain; lifecycle
maintenance removes the barrier only after the queue is empty, so a dropped batch cannot be
silently overtaken.

The latency-sensitive hook open validates the store application and schema versions, checksummed
migration history, and exact schema inventory before admitting a target. It does not run
whole-database `quick_check` or foreign-key scans inside provider callbacks. Maintenance,
migration, and read-only audit paths retain those full-data checks.

### Pi

Every selected callback consumes the bounded provider session UUID returned by
`ctx.sessionManager.getSessionId()` as a critical common field. That UUID is bridge enrichment, not
an upstream event-object field. Each `session_start` also creates a fresh cryptographic window
discriminator, so startup, reload, new, resume, and fork observations remain distinct even when Pi
reuses a native session UUID. The Python boundary HMAC-reduces the composite identity before
durable storage, and the bridge never derives identity from a session-file path.

| Event | Consumed fields | Evidence authority |
|---|---|---|
| `session_start` | `reason` | Opens a fresh observed window; `reason` is provider claimed. |
| `before_agent_start` | none | Marks a turn boundary solely from the registered callback. The handler returns `undefined` and cannot inject or modify content. |
| `tool_execution_start` | `toolCallId`, `toolName` | Keeps only an in-memory proposal. Pi can still run a later `tool_call` hook that mutates or blocks the call, so `args` are not read and this callback has no exact executed-action authority. |
| `tool_execution_end` | `toolCallId`, `toolName`, `isError` | A bounded start/end pair with matching call and tool names emits one coarse tool-class action only when `isError=false`, which confirms that execution reached a structured successful result. Pi also emits `isError=true` for missing tools, invalid or truncated arguments, aborts, and calls blocked before execution, so that value produces a content-free ambiguity degradation instead of a failed action. |
| `agent_settled` | none | Closes a stable observed unit after retries, compaction, and queued follow-up are settled. |
| `session_compact` | `reason`, `willRetry`, `fromExtension` | Flush and coverage boundary only. |
| `session_tree` | `newLeafId`, `oldLeafId` | Nullable leaf IDs are HMAC-reduced within-window hints; no cross-window lineage or summary is claimed. |
| `session_shutdown` | `reason` | Closes the current observed window. |

The accepted reason enums are exactly `startup|reload|new|resume|fork` for `session_start`,
`manual|threshold|overflow` for `session_compact`, and `quit|reload|new|resume|fork` for
`session_shutdown`. Enum drift degrades coverage rather than being reinterpreted.

`previousSessionFile`, `targetSessionFile`, prompt text, images, `systemPrompt`,
`systemPromptOptions`, tool `args` and `result`, `compactionEntry`, `summaryEntry`, and every other
field are ignored; this includes optional `session_tree.fromExtension`. The extension must not call
`ctx.sessionManager.getEntries()`, `getTree()`, `getBranch()`, or a history/backfill API; it must not
add a session entry. It deliberately excludes
`tool_call`, `tool_result`, `session_before_*`, and other hooks that can block or rewrite behavior.
Because the selected surface cannot prove the post-hook executed input, repeated-action detection
is unsupported for Pi v1. Tool-error detection is also unsupported: the pinned runtime uses the
same `isError=true` end event for execution failures and for calls that never executed. An
ambiguous error, unmatched start, unmatched end, or conflicting tool pair is omitted and marks
coverage degraded instead of inventing an action or outcome.

Each confirmed success is buffered and transported as one adjacent start/finish group. If JSON
escaping makes either half exceed the native event envelope, the entire group becomes one
content-free `oversize` degradation; an orphaned success half is never emitted.

Pinned Pi manual compaction temporarily disconnects agent-event forwarding while it aborts active
work. A start can therefore lack an end at that boundary without indicating successful execution;
the connector emits `unmatched_start` degradation, and this interruption remains an explicit
coverage exclusion.

Shutdown closes only its fresh capture window. A later reload, resume, new session, fork, or process
restart opens another window even if the native UUID is unchanged. A crash without
`session_shutdown` leaves the previous window open and therefore incomplete. The sensitive
previous/target session-file paths are never used to infer lineage, so cross-window lineage remains
unavailable.

An extension can start a Pi run with `sendMessage({triggerTurn:true})` without causing
`before_agent_start`. Tool callbacks still open a reducible unit that `agent_settled` can close, but
a tool-less extension-triggered run is invisible to this selected surface. That case is an explicit
coverage exclusion and cannot support an absence claim.

The windowed bridge inherits the bounded chunk contract described for OpenCode. Once any chunk is
received, missing first, middle, or final chunks degrade coverage; a batch lost in full before any
receipt remains inherently undetectable. `fully_unobserved_transport_batch` and
`process_exit_without_session_shutdown` are therefore explicit Pi coverage exclusions. Normal Pi
project trust or extension loading can also prevent every callback; until a receipt proves one was
observed, status remains `installed_not_observed` and `project_trust_or_extension_loading` remains
an explicit exclusion. A boundary that produces no evidence does not create an empty receipt;
pending transport degradation is attached to the next real boundary, action, or terminal record.

Pi documents start events in assistant source order and end events in completion order when tools
run in parallel. Those delivery properties do not form a total causal sequence. Parallel calls
remain distinct by `toolCallId`; receipt order and time remain local observation metadata. The
bridge bounds the native session identifier to 16 KiB and a closed identifier alphabet. Its fresh
window discriminator is lowercase 64-hex, while positive decimal event IDs are bridge-local replay
coordinates, never provider sequence authority.

## Authority, privacy, and integrity

Every profile fixes these declarations:

| Property | Contract |
|---|---|
| `source_authentication` | `none_same_user_untrusted`; lifecycle events and provider IDs are same-user inputs, not authenticated provider attestations |
| `raw_content_persisted` | `false`; admitted native values are discarded or HMAC-reduced before durable storage |
| `transcript_read` | `false` |
| `complete_execution_session_coverage` | `false`; only received events in the selected surface are covered |
| `decision_authority` | `false`; capture cannot approve, deny, retry, stop, inject, or modify an action |
| `model_calls` | `0`; capture performs no provider, SDK, API, or model request |
| `timestamp_authority` | `local_observation`; no profile v1 admits a provider timestamp |
| `sequence_authority` | `local_receipt_order`; no profile v1 claims provider causal order |
| `rollback_detection` | `none`; there is no external monotonic anchor |

At-rest HMAC detects mutation only when the installation key remains unavailable to the mutator. It
does not encrypt data, authenticate the provider, resist a same-OS-account actor who can read the
key, or detect rollback to an older valid MACed copy. Provider session and call identifiers are
correlation inputs only and never become public identifiers.

A structured tool success or failure describes one observed execution, never the task,
conversation, or agent outcome. Missing critical fields, conflicting transitions, gaps, or
unparented results fail closed by omitting or quarantining evidence. If SQLite and the bounded spool
are both unavailable, the provider continues but coverage is incomplete; no local design can
guarantee observation when the filesystem is unavailable.

A shape-compatible host version outside the exact audited baseline may be reported only as
`schema_compatible_unverified_version`. It must never be silently promoted to the verified
baseline.

## Official source register

All sources below were reviewed on **2026-07-19**. Living documentation can advance after that
date; exact release links define pinned TypeScript contracts where available.

| Provider and version | Official sources |
|---|---|
| Codex CLI `0.144.6` | [Hooks](https://learn.chatgpt.com/docs/hooks), [CLI commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli#cli-codex-exec), [app server](https://learn.chatgpt.com/docs/app-server) |
| Claude Code `2.1.204` | [Hooks](https://code.claude.com/docs/en/hooks), [plugins](https://code.claude.com/docs/en/plugins-reference), [settings](https://code.claude.com/docs/en/settings), [sessions](https://code.claude.com/docs/en/sessions) |
| OpenCode `1.18.3` / `127bdb3` | [plugins](https://opencode.ai/docs/plugins/), [SDK sessions](https://opencode.ai/docs/sdk/#sessions), [SDK events](https://opencode.ai/docs/sdk/#events), [configuration locations](https://opencode.ai/docs/config/#locations), [v1.18.3 release](https://github.com/anomalyco/opencode/releases/tag/v1.18.3), [generated event types at `127bdb3`](https://github.com/anomalyco/opencode/blob/127bdb30784d508cc556c71a0f32b508a3061517/packages/sdk/js/src/gen/types.gen.ts), [plugin hook types at `127bdb3`](https://github.com/anomalyco/opencode/blob/127bdb30784d508cc556c71a0f32b508a3061517/packages/plugin/src/index.ts) |
| Pi `0.80.10` / `8dc7883` | [extensions](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/extensions.md), [RPC mode](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/rpc.md), [JSON mode](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/json.md), [session format](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/session-format.md), [packages](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/packages.md), [package manifest](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/package.json), [pinned event types](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/src/core/extensions/types.ts), [pinned session lifecycle](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/src/core/agent-session.ts), [pinned tool loop](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/agent/src/agent-loop.ts) |

Connector artifacts are built with Node.js `22.19.0` and npm `10.9.3`. The pin follows Pi
`0.80.10`, whose package manifest requires Node.js `>=22.19.0`; Node's official
[22.19.0 release](https://nodejs.org/en/blog/release/v22.19.0) includes npm `10.9.3`, as fixed by
the release's [npm package manifest](https://github.com/nodejs/node/blob/v22.19.0/deps/npm/package.json).
