# Foundation evidence

This page records the tracked Shadow-trace performance reference for 1,000 mapped records. It
measures one implementation workload on one machine. It does not measure whether a reminder helps
an agent, compare memory policies, or establish performance on another system.

The [reference JSON](../../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json)
has SHA-256
`6b0e2b1e9e39570210df67a56163c56b2695fc3ec6b903b2fab530274fdcf0ba`. Its companion
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

## All measured samples

| Backend | Sample | Duration | Peak RSS |
|---|---:|---:|---:|
| In memory | 1 | 4.66473575 s | 185.171875 MiB |
| In memory | 2 | 4.729587833 s | 184.8125 MiB |
| In memory | 3 | 4.57829925 s | 184.828125 MiB |
| In memory | 4 | 4.719111875 s | 185.40625 MiB |
| In memory | 5 | 4.72002375 s | 184.546875 MiB |
| SQLite | 1 | 7.427851125 s | 213.921875 MiB |
| SQLite | 2 | 7.412707417 s | 212.78125 MiB |
| SQLite | 3 | 7.398388584 s | 212.8125 MiB |
| SQLite | 4 | 7.370406125 s | 212.15625 MiB |
| SQLite | 5 | 7.507945875 s | 213.890625 MiB |

## Budgets and result

| Backend | Median | Median budget | Maximum peak RSS | RSS budget | Result |
|---|---:|---:|---:|---:|---|
| In memory | 4.719111875 s | 5 s | 185.40625 MiB | 512 MiB | Passed |
| SQLite | 7.412707417 s | 15 s | 213.921875 MiB | 512 MiB | Passed |

The 1,000-record warm-ups took 4.503220708 seconds in memory and 7.736961959 seconds with SQLite. A
separate, non-gating 250-record baseline produced median times of 0.903633125 and 1.428466542
seconds. The recorded 250-to-1,000 ratios are 5.22237592275073 and 5.189276191671558. They do not
gate the run, and the v1 prefix digest is not claimed to scale linearly.

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
