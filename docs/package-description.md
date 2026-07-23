# SalienceGate

SalienceGate is a Python library for passive, local analysis of supported coding-agent lifecycle
events. Universal Shadow Capture provides project-local connectors for Codex, Claude Code,
OpenCode, and Pi. It reduces admitted fields before persistence, applies deterministic detectors,
and produces a content-free report with one of three bounded conclusions:
`memory_review_suggested`, `no_current_evidence`, or `insufficient_evidence`.

Capture calls no model, reads no transcript, persists no raw provider content, and has no authority
to approve, block, retry, inject, or change an agent action. Provider callbacks and identifiers are
same-user untrusted inputs, not authenticated attestations. Local HMAC protects stored-record
integrity while the installation key remains protected; it is not encryption and does not detect
whole-store rollback.

The package also includes incremental and whole-trace Shadow Mode APIs, sealed ATIF field-shape
profiles for Harbor Codex and Harbor Terminus 2 traces, deterministic replay and validation,
authenticated in-memory and SQLite ledgers, an offline source-paper algorithm path, and the local
StateDecayBench v2 review workflow. It is not a memory database.

Version 0.2.0 is an unpublished local candidate. It is not currently available from a public
package index. Build and review the wheel from the repository, then install that exact artifact:

```bash
python -m pip install /path/to/saliencegate-0.2.0-py3-none-any.whl
```

Only after a public package exists would `uv tool install saliencegate` replace that local-artifact
step; it is not an available installation route today.

After installation, use `saliencegate connect PROVIDER --dry-run` before enabling one reviewed,
trusted project. The `status`, `sessions`, `report`, `disconnect`, and explicit `delete` commands
provide the local operating lifecycle. Disconnecting removes the managed connector but retains
capture data; there is no automatic retention period.

The core runtime depends on Pydantic. The optional `model-runtime` extra is used only by the
separately named local OpenAI-compatible pilot path. SalienceGate requires Python 3.11, 3.12, or
3.13 and is licensed under Apache-2.0.

Synthetic examples and local tests establish software, schema, privacy, provenance, packaging, and
reproducibility contracts. They do not establish task efficacy, calibration, representative
prevalence, token or cost savings, comparative performance, provider authenticity, or reminder
usefulness. The current assessment is `insufficient_real_world_evidence`.
