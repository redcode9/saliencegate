# ATIF Shadow examples

These two synthetic trajectories contain no real task, command output, session identifier, or
credential. They exercise SalienceGate's sealed Codex and Terminus 2 field-shape profiles without
calling an agent, model, or provider.

## Set up the files

The CLI accepts only an owner-private source file. On a POSIX system, run the following from the
repository root.

Run from a checkout:

```bash
install -d -m 700 .saliencegate .saliencegate/atif-shadow
install -d -m 700 .saliencegate/atif-shadow/config
install -m 600 examples/atif-shadow/codex-minimal.trajectory.json \
  .saliencegate/atif-shadow/codex.trajectory.json
install -m 600 examples/atif-shadow/terminus-minimal.trajectory.json \
  .saliencegate/atif-shadow/terminus.trajectory.json
export XDG_CONFIG_HOME="$PWD/.saliencegate/atif-shadow/config"
```

## Analyze the Codex example

Artifact-compatible after installation:

```bash
saliencegate shadow analyze-atif .saliencegate/atif-shadow/codex.trajectory.json \
  --profile harbor-codex-v1 \
  --run-id c0de0000-0000-4000-8000-000000000001 \
  --working-directory /synthetic/workspace \
  --environment-digest eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
  --output .saliencegate/atif-shadow/codex.report.json \
  --json
```

This profile maps two `exec_command` calls and exact integer exit metadata. It covers the pinned
Harbor converter field shape, not a particular Codex CLI version.

## Analyze the Terminus 2 example

Artifact-compatible after installation:

```bash
saliencegate shadow analyze-atif .saliencegate/atif-shadow/terminus.trajectory.json \
  --profile harbor-terminus-2-v1 \
  --run-id 7e2a0000-0000-4000-8000-000000000001 \
  --working-directory /synthetic/workspace \
  --environment-digest eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
  --output .saliencegate/atif-shadow/terminus.report.json \
  --json
```

This profile maps two submitted `bash_command` calls. It ignores terminal text, so the example has
no outcome evidence.

## Use the one-call API

Run from a checkout:

```bash
uv run --locked python examples/atif-shadow/one_call.py
```

The script calls `analyze_atif_bytes` for both profiles. It does not pass an `installation_key`, so
each call uses a fresh in-memory key and reads neither an environment variable nor a key file. Pass
the same explicit `InstallationKey` when you need reproducible report bytes. Durable resume uses an
explicit `ShadowSession.sqlite_for_trace` workflow.

## Read the output

The CLI may create the normal SalienceGate installation key under the configured owner-private root
on first use. That key authenticates the local ledger; it is not a provider credential.

Each output file is a canonical `shadow-trace-report/v1`. Standard output is a smaller,
content-free `shadow-atif-command-report/v1` summary. Every result is
`descriptive_observational`, has no decision authority, and provides no task-efficacy or
comparative-performance evidence.
