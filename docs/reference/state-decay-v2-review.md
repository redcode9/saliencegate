# StateDecayBench v2 review protocol

The StateDecayBench v2 catalog contains 180 candidates across six visible families. Each
candidate has five outcome-free previews, producing 900 previews in total. Human review happens
before outcome allocation. The tools build and verify the review material, but they cannot supply
reviewer identity, checklist answers, rationale, a decision, or publication acceptance.

The entire review workflow is offline. It needs neither a model service nor an API credential.

## Build an immutable review pack

Use an owner-controlled parent directory. The repository's ignored `.artifacts/` directory is the
usual local location:

```bash
install -d -m 700 .artifacts
saliencegate-review build-pack \
  --output .artifacts/state-decay-v2-review-pack \
  --json
```

`build-pack` creates an immutable six-file tree. It has no `--replace` mode: a changed catalog
requires a new pack, so earlier review evidence is not overwritten. The JSON report contains
aggregate counts and binding digests, never filesystem paths or candidate content.

The publisher rejects unsafe storage. In particular, do not write the pack directly to a
world-writable directory such as `/tmp`. If temporary storage is necessary, create an owner-only
child directory first and place the pack beneath it.

## Review one candidate at a time

Start or resume the interactive review:

```bash
saliencegate-review review \
  --pack .artifacts/state-decay-v2-review-pack \
  --reviews .artifacts/state-decay-v2-reviews
```

The first submission for a reviewer ID shows the publication warning and requires this exact
confirmation:

```text
I ACCEPT PUBLICATION
```

Reviewer ID, rationale, seven checklist answers, decision, and superseded submissions in an
accepted chain can become public repository data. Do not enter secrets, private identifiers,
customer data, or anything you are unwilling to publish.

For each candidate, the command shows the family comparison, the candidate, five previews, and the
frozen seven-item checklist. Every checklist item, a rationale, and an `accepted` or `rejected`
decision are required. There is no default decision, bulk mode, non-interactive acceptance,
provider, endpoint, or model path.

Review without consulting or computing allocation, allocation rank, scenario ID, or assigned
outcome. None of those fields is present in the pack.

## Inspect review state

Read the current projection without changing it:

```bash
saliencegate-review status \
  --pack .artifacts/state-decay-v2-review-pack \
  --reviews .artifacts/state-decay-v2-reviews \
  --json
```

Status reports accepted, rejected, missing, stale-comparison, and ambiguous counts. The human gate
is complete only with exactly 180 current accepted envelopes and no missing, rejected, stale,
forked, or ambiguous head.

Review storage is locked during each transition. Submissions are append-only and use
compare-and-swap against the observed head; concurrent or stale writers fail instead of
overwriting history. An interrupted review can resume with the same command.

## Supersede a rejected or stale review

Run `review` again. The selector visits stale-comparison heads first, then rejected heads, before
moving to missing candidates. A new submission names the current head it supersedes, while earlier
submissions remain immutable audit data. A changed family comparison makes affected heads stale
and requires new checklist answers and a new decision. A changed candidate, profile,
configuration, or algorithm requires a new immutable pack under the review protocol's
invalidation rules.

An accepted current head is not automatically offered for correction. If an acceptance was entered
in error, stop and preserve the original directory. Do not delete or edit its files. Recovery
requires a fresh review directory and an explicit maintainer-selected replay; no command silently
reopens or carries forward an accepted decision.

## Build a current envelope

Project the current submission chain for one lineage key:

```bash
saliencegate-review build-envelope \
  --pack .artifacts/state-decay-v2-review-pack \
  --reviews .artifacts/state-decay-v2-reviews \
  --lineage-key KEY \
  --json
```

The command persists an immutable envelope bound to the current submission head, pack registry,
checklist, family comparison, and candidate. Building one envelope does not make the overall gate
ready.

## Generation boundary

This command surface has no `finalize`, generation, allocation, export, or readiness issuer.
Pre-custody generation remains unavailable until a later workflow binds an accepted-envelope
registry and revalidates all 180 current accepted envelopes, their chains, preview digests, and
scenario identities.

Keep local review work under `.artifacts/` and out of the repository. Generated role rows are not
review evidence; only a separately defined reviewer-authored registry can cross the publication
boundary.

## Exit codes

| Exit | Meaning |
|---:|---|
| `0` | Command completed successfully |
| `2` | Invalid command line or reviewer input |
| `5` | Corrupt, unsafe, stale, forked, or otherwise inconsistent review state |
| `70` | Unexpected internal failure reported without input values |
| `130` | Review interrupted by the operator |

The original module invocation remains supported, but `saliencegate-review` is the stable installed
entry point.
