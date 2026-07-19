# Contributing to SalienceGate

Useful contributions make a behavior more precise, add evidence that another person can reproduce,
or remove a known failure mode. Open an issue before a large change and describe the invariant you
want to change and the test that will demonstrate it.

## Development setup

You need Git, `uv`, and Python 3.11, 3.12, or 3.13.

From a checkout, create the locked development environment and run the tests:

```bash
uv sync --locked --all-extras --dev --no-install-project
uv sync --locked --all-extras --dev --no-build-isolation
uv run --locked pytest
```

The first sync installs every locked optional dependency without building the project. The second
installs SalienceGate using the build backend from that environment. CI also tests the core wheel
without optional model-runtime dependencies.

## Working on a change

Start with the smallest test that shows the missing behavior. For example:

```bash
uv run --locked pytest -q tests/test_public_docs.py
```

Make the smallest implementation that satisfies the invariant, rerun the focused test, then run
the full gate. A bug fix needs a regression test. Repository and adapter implementations should use
the shared conformance suites instead of private copies of equivalent tests.

Before committing, inspect the staged diff for secrets, generated files, and unrelated edits. Keep
format-only changes separate from behavioral changes, and use a short commit message that describes
the result.

Do not commit model weights, credentials, local databases, user traces, or generated benchmark
runs.

## Required gate order

Run the repository gates in this order:

1. `make format`
2. `make lint`
3. `make typecheck`
4. `make test`
5. `make coverage`
6. `make docs-check`
7. `make build`
8. `make audit`

`make check` runs this sequence and is the required non-interactive check before submission.

## Tests and evidence

Tests must be deterministic. Network access belongs only in explicitly marked integration tests
that are disabled by default. Property tests should include stable seeds in failure output.
Benchmark tests must distinguish schema validation, synthetic-oracle evidence, and external-task
evidence.

A reviewable benchmark result includes:

- the code revision and dirty-worktree status;
- the resolved configuration digest;
- model and runtime identifiers;
- hardware and operating system;
- the task set and seeds;
- the exact reproduction command;
- the artifact digest;
- exclusions and failed runs.

Do not change a locked benchmark label or replace a published artifact in place. Publish a new
artifact and an explicit supersession record when a result needs correction.

README and release-facing benchmark documents must keep measured results separate from targets.
Check public documentation with:

```bash
uv run --locked python scripts/check_public_docs.py
```

## Security reports

Do not open a public issue for a suspected vulnerability or leaked credential. Follow
[SECURITY.md](SECURITY.md).
