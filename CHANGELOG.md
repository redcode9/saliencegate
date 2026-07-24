# Changelog

This file records notable changes to SalienceGate.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.2.0 - 2026-07-25

### Added

- A typed Python package with reproducible dependency locking and public contribution, security,
  citation, conduct, and evidence policies.
- Immutable runtime records, canonical serialization, deterministic identifiers, and explicit
  boundaries between normalized input, redacted data, the authoritative ledger, and projections.
- Recursive secret redaction, installation-key HMAC tags, and an explicit opt-in for unkeyed
  digests used by synthetic benchmarks.
- Authenticated per-run ledgers and checkpoints, redaction-safe typed ingress, revision state
  machines, and deterministic projection rebuilds.
- A ledger-first SQLite repository with packaged migrations, WAL, FTS5, authenticated replay on
  open, projection repair, bounded cross-process comparison-and-swap, and crash tests.
- Recoverable memory-cycle reservations and settlements with exact model-call latency receipts.
- Deterministic memory-cycle batching with byte and token-estimate ceilings, structural
  compaction, budget-aware priority promotion, and authenticated batch manifests.
- Structured evidence envelopes and local detectors for tool errors, test failures, repeated
  actions, and unresolved repeated failures. Extraction reports identify detector versions and
  fail closed when a fingerprint cannot be established.
- Replay-stable always, never, and scripted invocation policies, plus a budget preflight wrapper
  that records budget-exhausted silence without mutating state.
- Invocation-ledger constraints that reject pre-event timestamps and competing decisions for the
  same run event, including a fail-closed SQLite migration for ambiguous legacy data.
- Citation-only memory proposals, provenance-aware grounding, bounded deterministic rendering,
  and verification receipts that reject unsupported or stale evidence.
- A durable delivery outbox with deterministic identifiers, sealed adapter capabilities,
  retry-safe deduplication, crash recovery, and one-next-decision expiry.
- Frozen replay models, strict JSONL and generic event adapters, outcome recording, and an engine
  that settles every reserved cycle.
- Canonical replay artifacts with revision evidence, component digests, cross-component digests,
  minimized inspection, strict validation, and recoverable atomic publication.
- A value-safe `saliencegate` CLI with `demo`, `doctor`, `replay`, `algorithm`, `pilot`, `benchmark`,
  `inspect`, and `validate` commands, stable machine-readable reports, and documented exit codes.
- StateDecayBench, a 32-case synthetic diagnostic with eight balanced scenario families, paired
  continuations, a deterministic oracle, and a reproducible fixture.
- The paper's fixed-step, two-phase memory algorithm alongside no-memory, deterministic-retrieval,
  and always-inject comparisons, with a frozen prompt contract, replay fixtures, and minimized
  algorithm artifacts.
- An optional OpenAI-compatible runtime and guarded `gpt-oss:20b` pilot. Live-model dependencies
  remain outside the core installation and require explicit local configuration.
- The StateDecayBench v2 research protocol, public candidate-generation contract, causal-delta
  execution, signal-fixture evaluation, and generation-authority boundary.
- An immutable StateDecayBench v2 Review Pack with 180 candidates, 900 outcome-free previews, six
  family comparisons, a seven-item checklist, append-only human review storage, and resumable
  commands.
- The `saliencegate-review` entry point for building packs, recording reviews, inspecting status,
  and projecting current envelopes without automating allocation, generation, or acceptance.
- A filesystem-free `saliencegate demo` that rebuilds and evaluates the diagnostic in memory while
  marking the result as non-confirmatory evidence.
- A provider-free Shadow Mode SDK and bounded NDJSON command with four supported deterministic
  detectors, payload-free observations, canonical descriptive reports, and no decision authority.
- Immutable whole-trace Shadow contracts, trace-bound in-memory and SQLite sessions, atomic
  authenticated batch analysis, exact-prefix repair and resume, and
  `shadow-trace-report/v1` provenance commitments.
- Sealed field-shape profiles for Harbor Terminus 2 ATIF-v1.6/v1.7 and Codex ATIF-v1.7, with
  explicit mapping and omission diagnostics, a pinned offline compatibility manifest, and
  sanitized or fully synthetic fixtures.
- The provider-free `analyze_atif_bytes` API and `saliencegate shadow analyze-atif` command, with
  explicit profile and environment binding, owner-private source and output handling, exact
  replacement identity, and content-free summaries. Codex support is limited to the pinned Harbor
  converter field shape and does not claim compatibility with a Codex CLI version.
- Locked quality gates, branch-aware coverage, dependency auditing, least-privilege CI across
  Python 3.11-3.13, installed wheel and source-distribution smoke tests, a dedicated 90-minute
  coverage job, public CLI and artifact references, and reproducible evidence reports.
- Project-local passive-capture integrations for Codex, Claude Code, OpenCode, and Pi, pinned to
  audited provider shapes and limited to selected lifecycle events.
- A content-minimizing capture store with domain-separated HMAC pseudonyms, authenticated SQLite
  records, a bounded recovery spool, and reversible installation receipts.
- Top-level commands to connect providers; inspect capture status and sessions; and report, record
  local feedback, disconnect, or delete without model calls or active reminders.
- Deterministic, content-free captured-session reports with synthetic examples for the three
  bounded headlines and an explicit descriptive-evidence boundary.
- Installed wheel and source-distribution verification for the local capture lifecycle, alongside
  offline connector bundle and network-denial checks.
- One-command, per-user installation for macOS, Linux, and Windows with a guided CLI setup.
- User-global capture for Codex, Claude Code, OpenCode, and Pi, including project exclusions,
  automatic project enrollment, status inspection, and reversible disconnects.
