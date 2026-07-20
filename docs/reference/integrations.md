# Universal Shadow Capture integration contract

This page freezes the normative provider contract for Universal Shadow Capture v1, targeted for
SalienceGate 0.2.0. A connector conforms only when its installed package contains the matching
audited capability manifest. Documentation alone is not evidence that a connector is installed or
has observed an event; capture status must report that separately.

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
`INSTALLED_NOT_OBSERVED` until an actual lifecycle callback proves activation; this is the explicit
`host_rejected_foreign_settings_layer` coverage exclusion.

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

- top-level `id` as a producer-event correlation input;
- `properties.sessionID` and `properties.part.sessionID`, which must agree;
- `properties.part.type`, `callID`, `tool`, `state.status`, and `state.input`.

`state.input` is canonicalized only to derive opaque action identity. The closed state machine has
this authority:

| `ToolPart.state.status` | Capture meaning |
|---|---|
| `pending`, `running` | Action observed; no outcome yet |
| `completed` | Provider-claimed structured success |
| `error` | Provider-claimed structured failure |

For `session.idle`, `session.compacted`, and `session.deleted`, only top-level `id` and
`properties.sessionID` are consumed. `session.error` consumes those fields when its optional
session ID is present; otherwise it can create only a content-free global health disposition.
`dispose` has no payload and flushes already-reduced in-memory state.

`properties.time`, part and message IDs, `state.raw`, `state.output`, `state.error`,
titles, metadata, attachments, `session.error.properties.error`, and
`session.deleted.properties.info` are ignored. Non-tool part variants are rejected by their
discriminant without traversing their content. The plugin must not call `session.messages()`,
`session.get()`, or another history materializer; parent session metadata therefore remains
unavailable in v1.

`callID` is exact per-action parent correlation but not authentication. OpenCode supplies no
admitted causal sequence. Event and tool-state time fields are excluded; receipt order and time are
local observation metadata only.

### Pi

Every selected callback consumes the bounded provider session UUID returned by
`ctx.sessionManager.getSessionId()` as a critical common field. The local transport HMAC-reduces it
before durable storage, and the bridge never derives identity from a session-file path.

| Event | Consumed fields | Evidence authority |
|---|---|---|
| `session_start` | `reason` | Opens an observed window; `reason` is provider claimed. |
| `before_agent_start` | none | Marks a turn boundary solely from the registered callback. The handler returns `undefined` and cannot inject or modify content. |
| `tool_execution_start` | `toolCallId`, `toolName`, `args` | `toolCallId` is exact parent correlation; tool name and args derive only opaque action identity. |
| `tool_execution_end` | `toolCallId`, `toolName`, `isError` | `isError=false` is provider-claimed success; `true` is provider-claimed failure. |
| `agent_settled` | none | Closes a stable observed unit after retries, compaction, and queued follow-up are settled. |
| `session_compact` | `reason`, `willRetry`, `fromExtension` | Flush and coverage boundary only. |
| `session_tree` | `newLeafId`, `oldLeafId`, `fromExtension` | Leaf IDs are HMAC-reduced lineage hints; no summary is consumed. |
| `session_shutdown` | `reason` | Closes the current observed window. |

`previousSessionFile`, `targetSessionFile`, prompt text, images, system prompts and options,
`result`, compaction entries, summary entries, and every other field are ignored. The extension
must not call `ctx.sessionManager.getEntries()`, `getTree()`, or a history/backfill API; it must
not add a session entry. It deliberately excludes `tool_call`, `tool_result`,
`session_before_*`, and other hooks that can block or rewrite behavior.

Pi documents start events in assistant source order and end events in completion order when tools
run in parallel. Those delivery properties do not form a total causal sequence. Parallel calls
remain distinct by `toolCallId`; receipt order and time remain local observation metadata.

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
| Pi `0.80.10` / `8dc7883` | [extensions](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/extensions.md), [RPC mode](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/rpc.md), [JSON mode](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/json.md), [session format](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/session-format.md), [packages](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/packages.md), [package manifest](https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/package.json) |

Connector artifacts are built with Node.js `22.19.0` and npm `10.9.3`. The pin follows Pi
`0.80.10`, whose package manifest requires Node.js `>=22.19.0`; Node's official
[22.19.0 release](https://nodejs.org/en/blog/release/v22.19.0) includes npm `10.9.3`, as fixed by
the release's [npm package manifest](https://github.com/nodejs/node/blob/v22.19.0/deps/npm/package.json).
