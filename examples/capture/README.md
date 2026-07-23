# Synthetic capture headline examples

[`headline-results.json`](headline-results.json) freezes three fully synthetic
`opencode-plugin/v1` capture batches and the content-free report fields produced from them. No
provider or model was called, and none of the values came from a real session.

| Case | Admitted evidence | Report headline |
|---|---|---|
| `memory-review-suggested` | The same exact action is observed twice; both calls have structured failed outcomes. | `memory_review_suggested` |
| `no-current-evidence` | Two distinct exact actions have structured successful outcomes and the window closes cleanly. | `no_current_evidence` |
| `insufficient-evidence` | One action remains unresolved, the window stays open, and the bridge reports a transport gap. | `insufficient_evidence` |

The positive case detects two `tool_error` signals and one `repeated_action` signal. It does not
claim a repeated-failure result: `repeated_failure` is explicitly `unsupported` for this profile.
The negative headline is limited to its sufficiently observed clean window. The insufficient case
reports the open window and gap instead of treating absent evidence as a negative result.

Every report remains `descriptive_observational`, with `model_calls=0`,
`decision_authority=false`, and `confirmatory=false`. A suggested review is not a diagnosis or an
instruction, and these synthetic mechanics do not establish task efficacy, calibration, or the
value of any reminder.

The checked [capture-headline visual](../../docs/assets/readme/capture-headlines.svg) is rendered
only from the fixture:

```bash
uv run --locked python scripts/render_capture_headlines.py --check
```

To reproduce both the renderer check and the runtime adapter-to-report replay:

```bash
uv run --locked pytest -q tests/test_readme_visuals.py -k capture_headline
```
