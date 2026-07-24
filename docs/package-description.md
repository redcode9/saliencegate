# SalienceGate

SalienceGate passively observes coding-agent lifecycle events and turns them into bounded local
evidence. It supports Codex, Claude Code, OpenCode, and Pi, without reading transcripts or taking
control of the agent.

The README provides one-line installers for macOS, Linux, and Windows. Installation requires no
administrator access and opens the setup wizard. Connect one or all providers to the current
project, another selected project, or globally for the current user. Global setup accepts explicit
project exclusions.

Accepted provider identifiers are pseudonymized before durable storage. Prompts, responses,
reasoning, tool arguments, and tool output are excluded. Session evidence stays on the user's
machine, and capture performs no model calls or provider actions.

SalienceGate reports `memory_review_suggested`, `no_current_evidence`, or
`insufficient_evidence`. These are deterministic observations for review, not instructions or
claims of task improvement. Local HMAC protects the integrity of present records; it is not
encryption or whole-store rollback protection.

SalienceGate requires Python 3.11, 3.12, or 3.13 and is licensed under Apache-2.0.
