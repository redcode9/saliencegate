# Foundation evidence

This page records the tracked Shadow-trace performance reference for 1,000 mapped records. It
measures one implementation workload on one machine. It does not measure whether a reminder helps
an agent, compare memory policies, or establish performance on another system.

The [reference JSON](../../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json)
has SHA-256
`653c132eaaa524d44811b4016676be77a94e3fb2b2636911cf2c3381963099e6`. Its companion
[evidence manifest](../../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.manifest.json)
binds that report to the benchmark script, runtime source surface, socket guard, runtime-relevant
project metadata, dependency lock, Python version file, and captured toolchain. The manifest is
content-addressed evidence, not a signature or a producer-authentication claim.

## Measured workload

The workload used the sealed `harbor-codex/v1` mapping with two ATIF steps, 499 mapped actions, 499
mapped structured outcomes, and 1,000 mapped records. Each measured process appended all records
through one batch call and one authenticated batch mutation.

Each backend ran once for warm-up, then five times in fresh isolated processes. Socket and resolver
access were denied. The report records no imported provider module. The time gate applies to the
median of the five isolated measurements; the memory gate applies to the largest peak RSS.

## Warm-ups and all measured samples

| Backend | Sample | Duration | Peak RSS |
|---|---:|---:|---:|
| In memory | Warm-up | 4.408641667 s | 185.25 MiB |
| In memory | 1 | 4.431902167 s | 185.140625 MiB |
| In memory | 2 | 4.431001459 s | 185.359375 MiB |
| In memory | 3 | 4.518042834 s | 185.09375 MiB |
| In memory | 4 | 4.505493583 s | 185.234375 MiB |
| In memory | 5 | 4.449103458 s | 184.921875 MiB |
| SQLite | Warm-up | 7.183987542 s | 212.359375 MiB |
| SQLite | 1 | 7.168292875 s | 214.0625 MiB |
| SQLite | 2 | 7.178122875 s | 214.046875 MiB |
| SQLite | 3 | 7.17084475 s | 213.15625 MiB |
| SQLite | 4 | 7.293288 s | 213.953125 MiB |
| SQLite | 5 | 7.243114625 s | 213.875 MiB |

## Budgets and result

| Backend | Median | Median budget | Maximum peak RSS | RSS budget | Result |
|---|---:|---:|---:|---:|---|
| In memory | 4.449103458 s | 5 s | 185.359375 MiB | 512 MiB | Passed |
| SQLite | 7.178122875 s | 15 s | 214.0625 MiB | 512 MiB | Passed |

Maximum peak RSS includes the warm-up and all five measured processes. A separate, non-gating
250-record baseline produced these summaries:

| Backend | Warm-up duration | Warm-up peak RSS | Median | Maximum peak RSS | 250-to-1,000 ratio |
|---|---:|---:|---:|---:|---:|
| In memory | 0.875333833 s | 88.421875 MiB | 0.884559833 s | 88.5625 MiB | 5.029737155157485 |
| SQLite | 1.313311625 s | 93.671875 MiB | 1.321523833 s | 93.734375 MiB | 5.431701416012222 |

The baseline does not gate the run, and the v1 prefix digest is not claimed to scale linearly.

## Environment

| Item | Recorded value |
|---|---|
| Operating system | macOS 26.5.2, Darwin, arm64 |
| CPU record | `arm`, 15 logical cores |
| Memory | 24,576 MiB |
| Python | CPython 3.12.3 |
| uv | 0.11.26 |
| Runner image | Unavailable; recorded as `unspecified` |
| Network policy | Socket and resolver denied |
| Provider modules imported | None |

## Reproduce the budget assertion

Run from a checkout:

```bash
uv --cache-dir /private/tmp/saliencegate-uv-cache run --python 3.12.3 --locked python scripts/benchmark_shadow_trace.py --assert-budgets
```

Local timings will vary. A new run describes its own source tree, dependencies, interpreter,
operating system, and hardware; it does not replace the tracked report unless those inputs and the
new output are reviewed together.

## What this evidence does not show

- It does not execute an action model or test a delivered reminder.
- It does not measure provider latency, billed cost, token reduction, or task efficacy.
- It does not establish cross-machine performance or linear scaling.
- It does not compare SalienceGate with another controller, memory store, or agent framework.
- It does not turn report digests into signatures, producer truth, or semantic validation.

The broader boundary for performance and efficacy statements is in
[Research claims](../research-claims.md).
