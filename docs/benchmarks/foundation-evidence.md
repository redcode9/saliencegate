# Foundation evidence

This page records the tracked Shadow-trace performance reference for 1,000 mapped records. It
measures one implementation workload on one machine. It does not measure whether a reminder helps
an agent, compare memory policies, or establish performance on another system.

The [reference JSON](../../benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json)
has SHA-256
`1a1c11f8e0491f0fcbbf80cfa31923dd8dfdf6a33b5d6c1719c75dc14e1f0cf6`. Its companion
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
| In memory | Warm-up | 4.502554042 s | 185.46875 MiB |
| In memory | 1 | 4.587372333 s | 186.59375 MiB |
| In memory | 2 | 4.9441875 s | 187.296875 MiB |
| In memory | 3 | 4.684918375 s | 186.484375 MiB |
| In memory | 4 | 4.655026 s | 185.640625 MiB |
| In memory | 5 | 5.0557965 s | 185.75 MiB |
| SQLite | Warm-up | 7.479044583 s | 214.5 MiB |
| SQLite | 1 | 7.412349542 s | 215.25 MiB |
| SQLite | 2 | 7.750414709 s | 214.890625 MiB |
| SQLite | 3 | 7.672826583 s | 214.265625 MiB |
| SQLite | 4 | 7.501124583 s | 214.3125 MiB |
| SQLite | 5 | 7.457298167 s | 213.703125 MiB |

## Budgets and result

| Backend | Median | Median budget | Maximum peak RSS | RSS budget | Result |
|---|---:|---:|---:|---:|---|
| In memory | 4.684918375 s | 5 s | 187.296875 MiB | 512 MiB | Passed |
| SQLite | 7.501124583 s | 15 s | 215.25 MiB | 512 MiB | Passed |

Maximum peak RSS includes the warm-up and all five measured processes. A separate, non-gating
250-record baseline produced these summaries:

| Backend | Warm-up duration | Warm-up peak RSS | Median | Maximum peak RSS | 250-to-1,000 ratio |
|---|---:|---:|---:|---:|---:|
| In memory | 0.955874416 s | 88.609375 MiB | 0.907012791 s | 88.84375 MiB | 5.165217537709454 |
| SQLite | 1.439151208 s | 93.890625 MiB | 1.428385625 s | 94 MiB | 5.2514702274464575 |

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
