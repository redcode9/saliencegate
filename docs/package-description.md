# SalienceGate

SalienceGate is a Python library for testing when a long-running agent may need a memory reminder.
It reads native event streams or supported ATIF traces, runs deterministic detectors, and produces
an auditable report. It is not a memory database.

Shadow Mode is the shortest integration path. It observes trace data without calling a model,
reserving a budget, changing the agent's decisions, or delivering a reminder. Its reports are
descriptive evidence: they show which supported detector fired, which evidence was incomplete, and
which records were outside the current detector scope.

The package currently provides:

- incremental and whole-trace Shadow Mode APIs;
- sealed ATIF field-shape profiles for Harbor Codex and Harbor Terminus 2 traces;
- in-memory and SQLite ledgers with authenticated record integrity;
- deterministic replay, validation, and benchmark commands;
- an offline paper-algorithm path and a review workflow for StateDecayBench v2 candidates.

The package has not been published. Install a reviewed wheel built from the repository with:

```bash
python -m pip install /path/to/saliencegate-0.1.0-py3-none-any.whl
```

The core runtime depends on Pydantic. The optional `model-runtime` extra adds the HTTP and prompt
encoding dependencies used by the local OpenAI-compatible pilot path.

SalienceGate does not currently claim improved task success, lower token use, lower cost, or better
performance than another memory system. The bundled examples and diagnostics establish software,
schema, provenance, and reproducibility contracts. They do not establish agent efficacy.

SalienceGate requires Python 3.11 or newer and is licensed under Apache-2.0.
