# Security model

This appendix covers Universal Shadow Capture, the core event and memory path, authenticated
repositories, artifact publication, Shadow analysis, ATIF adaptation, the local model pilot, and
StateDecayBench review data. Vulnerability reporting and supported versions are defined in the
repository-root
[Security Policy](../SECURITY.md).

## Threat model

Treat trajectory events, memory records, tool results, model output, ATIF documents, artifact
trees, benchmark material, and reviewer-authored fields as untrusted. Inputs may be malformed,
oversized, adversarially nested, internally ambiguous, or designed to place credentials and private
data into a durable record. A structurally valid source can still contain false claims.

A local peer may attempt path aliases, ancestor or target symlink swaps, inode replacement, hard
links, permission changes, partial SQLite state, or replacement of an unrelated report or artifact.
SalienceGate checks these boundaries, but it does not defend against a process that can act as the
same operating-system user and bypass owner-only controls. Use operating-system isolation for
mutually untrusted local principals.

SalienceGate does not encrypt repositories, reports, artifacts, or review data. It also cannot make
an unsafe caller-supplied delivery callback harmless. The built-in deterministic replay sink fails
without an external side effect, and Shadow Mode has no delivery authority; applications that add a
delivery adapter own the effect boundary.

## Parsing and adaptation

- Public parsers use bounded strict schemas and reject unknown or malformed values before they
  enter authoritative state. Value-free boundary errors avoid echoing untrusted input.
- The ATIF adapter accepts exact bytes and strict UTF-8 JSON. Duplicate object keys, non-finite
  numbers, BOMs, unsupported schemas, excessive depth or collections, malformed selected calls,
  ambiguous recognized outcomes, and inconsistent timestamps fail closed.
- ATIF source bytes are bounded to 64 MiB. The parser also bounds JSON nodes and depth, root steps,
  per-step and aggregate calls and results, strings, commands, directories, and the 1,000-record
  mapped trace.
- Profile selection is explicit. Unknown producers, functions, arguments, result parents, or
  outcome paths cannot silently acquire evidence authority.
- Adaptation is pure, never executes a selected command, never searches output text for keywords,
  and completes before a repository can be opened for mutation.
- Adapter descriptors, configuration, compatibility evidence, mapped records, and reports are
  content-addressed and revalidated at their trust boundaries.

The selected execution environment is caller-attested. `default_working_directory` fills a missing
profile-supported directory; `environment_digest` is an opaque lowercase SHA-256 join value.
SalienceGate does not inspect the machine to prove either value.

## Selection, redaction, and privacy

Built-in ATIF profiles map selected events from the root trajectory segment only. They report the
presence of a continued trajectory and count embedded subagent trajectories, but do not traverse
them. Complete execution-session coverage is always false.

Raw ATIF bytes are not retained in `ShadowTrace` or persisted. Messages, reasoning, final answers,
raw terminal and tool output, source IDs, tool-call IDs, metrics, copied context, and unselected
arguments are omitted. Selected commands, working directories, and admissible structured exit
status become typed Shadow evidence and pass through pre-persistence redaction.

The core redactor covers declared sensitive fields, configured literal secrets, private-key
blocks, credential-bearing URI authorities, bearer tokens, and common credential shapes. It also
rejects ambiguous sensitive field names. Pattern redaction is defense in depth, not a complete
secret classifier. Exclude secrets at the source and configure domain-specific fields and literals.

Public trace reports, command summaries, reprs, exceptions, and logs omit selected commands,
working directories, source IDs, and source, repository, and output paths. Aggregates are not
automatically harmless: digests, counts, and provenance joins may reveal equality or activity.
Reports are owner-private by default.

Universal Capture uses a stricter provider-specific allowlist before the same general redaction
boundary. It never opens Codex or Claude transcript paths, OpenCode message history, Pi session
files, prompts, responses, reasoning, or raw tool output. Admitted provider session, call, action,
workspace, subagent, and lineage values are reduced by domain-separated HMAC before either SQLite
or the fallback spool can persist them. The store holds bounded structured event classes and local
observation metadata, not the raw native callback. Reports set `raw_content_persisted=false` and
`transcript_read=false`.

Those values are pseudonymous, not anonymous. A key-bound identifier remains linkable within the
domain and installation needed for correlation; counts, host versions, time windows, capability
exclusions, short SalienceGate session identifiers, and equality relationships can still be
sensitive. Feedback export uses separate export-nonce- and domain-separated pseudonyms, but a
recipient must still treat the resulting dataset as sensitive. No pseudonym proves that the
provider, event, timestamp, or human label was authentic.

## Outcome authority and compatibility

Both sealed profiles declare `producer_authentication=none`:

- `harbor-codex/v1` accepts an exact signed 32-bit integer exit status from two sealed metadata
  paths and labels it `producer_claimed_structured`. Output text, a completed status without an exit
  code, alternate paths, strings, booleans, floats, and conflicting recognized paths do not map.
- `harbor-terminus-2/v1` declares outcome authority `none`. Terminal text and unkeyed results never
  become success, failure, test, or exception evidence.

The compatibility manifest pins audited Harbor converter and public-golden field shapes. It does
not authenticate a producer. The Codex converter did not pin a Codex CLI runtime, so the profile
does not claim compatibility with a specific Codex version. The paper and proactive-memory
repository are research inspiration, not compatibility evidence.

## Keys, authenticated state, and recovery

The installation key is 32 random bytes stored in an owner-only regular file. It authenticates
redacted local ledger state with domain-separated HMAC-SHA-256. It is not a provider credential.
Possession of the key permits valid HMAC construction, and HMAC integrity is not encryption.

When `analyze_atif_bytes` receives no `installation_key`, it generates a fresh key in memory and
does not read environment variables or a key file. Stable report bytes across calls require the
same explicit key and all other report inputs. The CLI loads or creates its normal owner-private
installation key only after stable source read, parsing, and adaptation succeed.

`ShadowSession.sqlite_for_trace` requires an explicit key and opens storage lazily after complete
analyzer preflight. Whole-trace mutation is one bounded authenticated batch: cancellation or
failure exposes either the previous exact prefix or the complete suffix. A new process cannot by
itself detect rollback of an entire valid database to an older state; that requires an external
monotonic anchor. Protect and back up the database and key consistently.

Universal Capture uses that same installation-key boundary for distinct domains covering project
and provider identities, connection generations, captured event chains, snapshots, health,
transport receipts, feedback, deletion tombstones, integration receipts and journals, and spool
state. Domain separation prevents one valid tag from being reused as another record type. The key
does not authenticate the provider: callbacks and provider identifiers remain
`none_same_user_untrusted`. Possession of the key permits valid record construction, and replacing
the database, spool, and key together with an older internally valid set is not detected.

## Capture storage, retention, and removal

Capture resolves one per-user state root without consulting a project config file:

| Platform | State root | Installation key |
|---|---|---|
| POSIX | `$XDG_STATE_HOME/saliencegate`, or `~/.local/state/saliencegate` when `XDG_STATE_HOME` is unset | `$XDG_CONFIG_HOME/saliencegate/installation.key`, or `~/.config/saliencegate/installation.key` |
| Windows | `%LOCALAPPDATA%\SalienceGate` | `$XDG_CONFIG_HOME/saliencegate/installation.key` when that override is set; otherwise `%APPDATA%\saliencegate\installation.key`, falling back to the user home `.config` directory when `APPDATA` is absent |

The state root contains `capture.sqlite3`, its SQLite sidecars when present,
`capture-spool/`, and private provider operations below
`integrations/<project-locator>/<provider>/`. The latter can contain an authenticated receipt,
short-lived recovery journal, lock, and generated launcher. Project-local managed files are
`.codex/config.toml`, `.claude/settings.local.json`, the OpenCode plugin and bootstrap below
`.opencode/plugins/`, or the Pi extension and bootstrap below `.pi/extensions/`. These files execute
with the user's operating-system authority and must be inspected before the project is trusted.
Default launcher materialization ignores ambient `PATH` and resolves the exact capture-hook name
only inside the current Python interpreter's absolute scripts directory, after excluding the
target project and validating the executable's ownership, write permissions, single-link identity
when user-owned, and ancestry. Its interpreter boundary is already running code and is not selected
from the current working directory. The read-only Git probe similarly ignores relative,
project-local, current-working-directory, group- or world-writable, user-owned multiply-linked, and
ownership-ambiguous executable-search boundaries. Privileged-owned POSIX system executables may
have multiple immutable links and remain eligible. Either lookup fails closed when it cannot
establish a trusted absolute executable. On Windows, this deliberately means that a system-wide Git
executable owned by a privileged account rather than the current user is reported as `unavailable`.

On POSIX, private directories and regular data files are required to be owner controlled; the key
is exactly 32 bytes in an owner-only regular file. On Windows, the implementation uses native
handle and DACL checks for the managed boundary. Native Ubuntu, Windows, and macOS jobs are prepared
but remote verification remains a separate gate for this unpublished branch.

There is no automatic retention period, expiry, or time-to-live. `disconnect PROVIDER` is the
connector uninstall operation: it drains admission, reverses only authenticated owned provider
changes, removes owned bridge assets and the launcher, and leaves a disabled receipt. It retains
SQLite sessions, feedback, health, spool state, connection history, deletion tombstones, and the
installation key so existing evidence remains verifiable. Removing a Python environment or wheel
does not safely disconnect project hooks and does not delete local state. Disconnect every provider
from every affected project before uninstalling the package.

`delete SESSION_ID` removes one current-project session and its dependent records, then retains an
authenticated content-free tombstone for retry safety. `delete --all --project PATH --confirm`
requires every project connection to be disabled and removes that project's connections, sessions,
feedback, health, transport receipts, and tombstones. Both operations enable SQLite
`secure_delete`, drain the spool under maintenance, and checkpoint/truncate the WAL. They do not
erase a previously exported report or dataset, backup, filesystem snapshot, storage-device remnant,
or another project's records, and they do not delete the installation key. Remove the remaining
state root and key manually only after all projects have been disconnected and deleted and after
deciding how backups should be handled.

For recovery, treat `installation.key`, `capture.sqlite3` plus its sidecars, `capture-spool/`, and
the provider `integrations/` directories as one set. Back up and restore them together. Rerunning the
same authenticated connect or disconnect can complete its bounded journaled transition; unexpected
receipt, journal, config, bundle, launcher, permission, or generation drift fails closed.
`doctor --capture` performs a strictly read-only integrity check, while `status` may drain the
existing spool. Neither command can reconstruct missing evidence, reverse a recorded drop, or make
an incomplete session complete.

## Filesystem, artifacts, and publication

Security-sensitive CLI paths require stable regular files and controlled parents. The
implementation rejects unsafe ancestor traversal, symlinks, hard links where a single link is
required, unsupported file types or filesystems, aliases between protected roles, changed
identities, unsafe permissions, and observable swaps. SQLite authorization also binds expected
WAL, shared-memory, and journal names before use.

Artifact validators require a closed, bounded file tree; reject missing, extra, non-canonical, or
changed components; recompute content and manifest bindings; and recheck identities after reading.
Artifact SHA-256 digests are not signatures, MACs, or encryption. An expected manifest digest
received through a trusted channel is an equality anchor, not proof that the producer's semantic
claims are true.

The repository publication guard treats commit
`e3cfcf2f2c61d3917457d168733908a3adfbda41` as an immutable, pre-audited public-history anchor.
It audits the complete current index. For later history, it fails closed when the repository is
shallow, the anchor is unavailable, or the anchor is not an ancestor of `HEAD`; checks paths and
commit metadata on every reachable commit outside the anchor ancestry; and content-checks each path
entry that is absent or different at the same path in every parent. Reusing an anchored object under
a new path is therefore audited again. Only bounded UTF-8 blobs in regular Git modes `100644` and
`100755` pass the content boundary.

`shadow analyze-atif` requires a stable owner-private single-link source. Source, SQLite and its
sidecars, and output must not alias. The output parent must already be owner-controlled.
Publication is one bounded atomic create or an exact authorized replacement, produces mode `0600`,
reopens the result, decodes it, and requires exact canonical `shadow-trace-report/v1` bytes.
`--replace` checks source, profile, adapter and detector configuration, provenance, redaction
policy, nested Shadow binding, initial repository state, and report identity. A corrupt, unrelated,
or changed output is preserved rather than overwritten.

Artifact publication relies on POSIX directory descriptors and directory `fsync`; it is not
supported or verified on Windows. Python's standard SQLite module does not expose an fd-bound VFS,
so a same-user adversary can still race the narrow interval between path checks and SQLite use.
Every observable boundary is rechecked and persistent replacements fail closed.

## Network and provider boundary

Benchmark, review, Shadow, ATIF, and one-call example paths do not instantiate an HTTP client or
socket, make a model call, reserve a model budget, or import an optional provider runtime.
Deterministic replay and algorithm replay can exercise model-call and budget accounting with frozen
fixtures, but they make no provider request. Shadow and ATIF paths do not inspect provider
credential variables. The CLI key boundary may read only the local configuration-root variables
used to locate SalienceGate's installation key and the key file itself.

The separately named `saliencegate pilot paper-two-phase` command is the only CLI path that can
contact a model runtime. It accepts only a numeric loopback OpenAI-compatible endpoint, uses no API
credential, and reports bounded evidence from the fixed local diagnostic. Running that command is
an explicit network action even though traffic remains on the host. `doctor --endpoint` validates
syntax only and does not connect.

These controls support deterministic offline analysis and a narrowly scoped local pilot. They do
not establish task efficacy, calibration, representative capture, source truthfulness, producer
authentication, or decision authority.

## Review-data boundary

StateDecayBench review packs and submission chains are intended for owner-controlled local storage.
The pack contains no assigned outcomes, allocation ranks, or scenario IDs. Reviews are append-only,
but reviewer IDs, rationales, checklist answers, decisions, and superseded submissions in an
accepted chain may later become public. Do not enter confidential or identifying material.

The review CLI cannot prove that a reviewer acted independently or avoided outside allocation
information. It records explicit attestations and validates the immutable chain; those records are
audit evidence, not proof of reviewer intent. See the [review protocol](reference/state-decay-v2-review.md).
