import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import {
  buildWindowedCaptureChunks,
  normalizeWindowedCaptureEvent,
  type BootstrapBinding,
} from "@saliencegate/bridge-core";

import {
  MAX_PI_NATIVE_SESSION_ID_BYTES,
  PI_UPSTREAM_COMMIT,
  PI_UPSTREAM_VERSION,
  PiWindowReducer,
  type ReducedPiRecord,
} from "../src/index.ts";

const SESSION_ID = "019c0eaf-7b31-7000-8000-000000000001";
const WINDOW = "a".repeat(64);
const CROSS_LANGUAGE_BATCH = new URL(
  "../../../tests/fixtures/pi-cross-language-batch.json",
  import.meta.url,
);
const CROSS_LANGUAGE_BOOTSTRAP: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "pi-extension/v1",
  connection_id: `sg-${"6".repeat(48)}`,
  launcher_path: "/private/tmp/saliencegate-pi-hook",
  capability_digest: "777d82ed469eb613251b6062a088e62cf2a1bc9b714a2f69634c4eba1c86b248",
  bundle_digest: "9".repeat(64),
  receipt_mac: "a".repeat(64),
};

function reducer(): PiWindowReducer {
  return new PiWindowReducer({
    sessionID: SESSION_ID,
    windowDiscriminator: WINDOW,
  });
}

function kinds(records: readonly ReducedPiRecord[]): string[] {
  return records.map((record) => record.kind);
}

test("the structural Pi API pin is exact and does not require the provider package", () => {
  assert.equal(PI_UPSTREAM_VERSION, "0.80.10");
  assert.equal(PI_UPSTREAM_COMMIT, "8dc78834cde4e329284cf505f9e3f99763df5529");
});

test("parallel tools remain correlated when their ends interleave", () => {
  const value = reducer();

  assert.deepEqual(
    value.reduce({
      type: "tool_execution_start",
      toolCallId: "call-a",
      toolName: "read",
      args: { path: "RAW-ARGS-A" },
    }),
    [],
  );
  assert.deepEqual(
    value.reduce({
      type: "tool_execution_start",
      toolCallId: "call-b",
      toolName: "bash",
      args: { command: "RAW-ARGS-B" },
    }),
    [],
  );
  const second = value.reduce({
    type: "tool_execution_end",
    toolCallId: "call-b",
    toolName: "bash",
    result: { output: "RAW-RESULT-B" },
    isError: false,
  });
  const first = value.reduce({
    type: "tool_execution_end",
    toolCallId: "call-a",
    toolName: "read",
    result: { output: "RAW-RESULT-A" },
    isError: false,
  });

  assert.deepEqual(kinds(second), ["tool_started", "tool_finished"]);
  assert.deepEqual(kinds(first), ["tool_started", "tool_finished"]);
  assert.deepEqual(
    second.map((record) => record.event_id),
    ["1", "2"],
  );
  assert.deepEqual(
    first.map((record) => record.event_id),
    ["3", "4"],
  );
  assert.deepEqual(second[0], {
    kind: "tool_started",
    session_id: SESSION_ID,
    window_discriminator: WINDOW,
    event_id: "1",
    call_id: "call-b",
    tool: "bash",
    identity_authority: "coarse",
  });
  assert.deepEqual(second[1], {
    kind: "tool_finished",
    session_id: SESSION_ID,
    window_discriminator: WINDOW,
    event_id: "2",
    call_id: "call-b",
    outcome: "succeeded",
  });
  assert.doesNotMatch(JSON.stringify([...second, ...first]), /RAW-/);
});

test("unmatched proposals degrade before agent_settled closes the open unit", () => {
  const value = reducer();
  assert.deepEqual(value.reduce({ type: "before_agent_start", systemPromptOptions: {} }), []);
  assert.deepEqual(
    value.reduce({
      type: "tool_execution_start",
      toolCallId: "blocked-call",
      toolName: "write",
      args: { path: "RAW-BLOCKED" },
    }),
    [],
  );

  const settled = value.reduce({ type: "agent_settled" });
  assert.deepEqual(kinds(settled), ["coverage_degraded", "turn_finished"]);
  assert.equal(
    settled[0]?.kind === "coverage_degraded" && settled[0].reason,
    "unmatched_start",
  );
  assert.deepEqual(
    settled.map((record) => record.event_id),
    ["1", "2"],
  );
  assert.deepEqual(value.reduce({ type: "agent_settled" }), []);
  const lateStart = {
    type: "tool_execution_start",
    toolCallId: "blocked-call",
    toolName: "write",
    args: null,
  } as const;
  const lateEnd = {
    type: "tool_execution_end",
    toolCallId: "blocked-call",
    toolName: "write",
    result: null,
    isError: false,
  } as const;
  assert.deepEqual(value.reduce(lateStart), []);
  const late = value.reduce(lateEnd);
  assert.deepEqual(kinds(late), ["coverage_degraded"]);
  assert.equal(late[0]?.kind === "coverage_degraded" && late[0].reason, "invalid_transition");
  assert.deepEqual(value.reduce(lateEnd), []);
  assert.deepEqual(value.reduce({ type: "agent_settled" }), []);
});

test("matched upstream errors are content-free ambiguous degradations", () => {
  const cases = [
    { name: "blocked", result: { error: "RAW-BLOCKED" } },
    { name: "missing", result: { error: "RAW-MISSING-TOOL" } },
    { name: "truncated", result: { error: "RAW-TRUNCATED-CALL" } },
  ] as const;

  for (const item of cases) {
    const value = reducer();
    const start = {
      type: "tool_execution_start",
      toolCallId: `ambiguous-${item.name}`,
      toolName: "read",
      args: { secret: `RAW-${item.name}` },
    } as const;
    const end = {
      type: "tool_execution_end",
      toolCallId: `ambiguous-${item.name}`,
      toolName: "read",
      result: item.result,
      isError: true,
    } as const;

    assert.deepEqual(value.reduce(start), []);
    const degraded = value.reduce(end);
    assert.deepEqual(kinds(degraded), ["coverage_degraded"]);
    assert.equal(
      degraded[0]?.kind === "coverage_degraded" && degraded[0].reason,
      "ambiguous_error",
    );
    assert.equal(degraded[0]?.event_id, "1");
    assert.doesNotMatch(JSON.stringify(degraded), /RAW-|call_id|tool_started|tool_finished/);
    assert.deepEqual(value.reduce(end), []);
    assert.deepEqual(kinds(value.reduce({ type: "agent_settled" })), ["turn_finished"]);
  }
});

test("toolful extension-triggered runs close without before_agent_start", () => {
  const value = reducer();
  assert.deepEqual(
    value.reduce({
      type: "tool_execution_start",
      toolCallId: "extension-call",
      toolName: "read",
      args: null,
    }),
    [],
  );
  assert.deepEqual(
    kinds(
      value.reduce({
        type: "tool_execution_end",
        toolCallId: "extension-call",
        toolName: "read",
        result: null,
        isError: false,
      }),
    ),
    ["tool_started", "tool_finished"],
  );
  assert.deepEqual(kinds(value.reduce({ type: "agent_settled" })), ["turn_finished"]);
});

test("pending starts degrade before compaction, tree, and shutdown boundaries", () => {
  const cases = [
    {
      boundary: {
        type: "session_compact",
        compactionEntry: null,
        fromExtension: false,
        reason: "manual",
        willRetry: false,
      },
      finalKind: "coverage_boundary",
    },
    {
      boundary: {
        type: "session_tree",
        newLeafId: "new-leaf",
        oldLeafId: "old-leaf",
      },
      finalKind: "coverage_boundary",
    },
    {
      boundary: { type: "session_shutdown", reason: "quit" },
      finalKind: "session_finished",
    },
  ] as const;

  for (const [index, item] of cases.entries()) {
    const value = reducer();
    value.reduce({
      type: "tool_execution_start",
      toolCallId: `pending-${index}`,
      toolName: "read",
      args: null,
    });
    const records = value.reduce(item.boundary);
    assert.deepEqual(kinds(records), ["coverage_degraded", item.finalKind]);
    assert.equal(
      records[0]?.kind === "coverage_degraded" && records[0].reason,
      "unmatched_start",
    );
  }
});

test("the reducer and Python adapter share one exact canonical batch", async () => {
  const value = new PiWindowReducer({
    sessionID: "synthetic-pi-session",
    windowDiscriminator: "7".repeat(64),
  });
  const records: ReducedPiRecord[] = [];
  const push = (event: unknown): void => {
    records.push(...value.reduce(event));
  };

  push({
    type: "tool_execution_start",
    toolCallId: "successful-call",
    toolName: "read",
    args: { ignored: "RAW-SUCCESS" },
  });
  push({
    type: "tool_execution_end",
    toolCallId: "successful-call",
    toolName: "read",
    result: { ignored: "RAW-SUCCESS" },
    isError: false,
  });
  push({
    type: "tool_execution_start",
    toolCallId: "ambiguous-call",
    toolName: "bash",
    args: { ignored: "RAW-AMBIGUOUS" },
  });
  push({
    type: "tool_execution_end",
    toolCallId: "ambiguous-call",
    toolName: "bash",
    result: { ignored: "RAW-AMBIGUOUS" },
    isError: true,
  });
  push({
    type: "tool_execution_start",
    toolCallId: "interrupted-call",
    toolName: "write",
    args: { ignored: "RAW-INTERRUPTED" },
  });
  push({
    type: "session_compact",
    compactionEntry: { ignored: "RAW-COMPACTION" },
    fromExtension: false,
    reason: "manual",
    willRetry: false,
  });
  push({ type: "agent_settled" });
  push({
    type: "session_tree",
    newLeafId: null,
    oldLeafId: "old-leaf",
    summaryEntry: { ignored: "RAW-TREE" },
  });
  push({ type: "session_shutdown", reason: "quit" });

  const chunks = buildWindowedCaptureChunks({
    bootstrap: CROSS_LANGUAGE_BOOTSTRAP,
    batchID: "8".repeat(64),
    sessionID: "synthetic-pi-session",
    windowDiscriminator: "7".repeat(64),
    events: records.map((record) =>
      normalizeWindowedCaptureEvent(
        record,
        record.session_id,
        record.window_discriminator,
      ),
    ),
  });
  assert.equal(chunks.length, 1);
  assert.deepEqual(chunks[0]!.bytes, await readFile(CROSS_LANGUAGE_BATCH));
  assert.doesNotMatch(chunks[0]!.bytes.toString("utf8"), /RAW-/);
});

test("duplicate callbacks are idempotent while conflicts and unmatched ends degrade", () => {
  const value = reducer();
  const start = {
    type: "tool_execution_start",
    toolCallId: "same-call",
    toolName: "read",
    args: { ignored: true },
  } as const;
  const end = {
    type: "tool_execution_end",
    toolCallId: "same-call",
    toolName: "read",
    result: { ignored: true },
    isError: false,
  } as const;

  assert.deepEqual(value.reduce(start), []);
  assert.deepEqual(value.reduce(start), []);
  assert.deepEqual(kinds(value.reduce(end)), ["tool_started", "tool_finished"]);
  assert.deepEqual(value.reduce(end), []);
  assert.deepEqual(
    kinds(value.reduce({ ...end, isError: true })),
    ["coverage_degraded"],
  );
  assert.deepEqual(
    kinds(
      value.reduce({
        type: "tool_execution_end",
        toolCallId: "missing-call",
        toolName: "read",
        result: null,
        isError: false,
      }),
    ),
    ["coverage_degraded"],
  );

  const conflicting = reducer();
  conflicting.reduce(start);
  assert.deepEqual(
    kinds(conflicting.reduce({ ...start, toolName: "write" })),
    ["coverage_degraded"],
  );
  assert.deepEqual(
    kinds(conflicting.reduce({ ...end, toolName: "write" })),
    ["coverage_degraded"],
  );
});

test("compaction, tree navigation, retries, and shutdown have closed meanings", () => {
  const value = reducer();
  value.reduce({ type: "before_agent_start", systemPromptOptions: {} });
  const compact = value.reduce({
    type: "session_compact",
    compactionEntry: { summary: "RAW-COMPACTION" },
    fromExtension: false,
    reason: "overflow",
    willRetry: true,
  });
  value.reduce({
    type: "tool_execution_start",
    toolCallId: "retried-call",
    toolName: "read",
    args: { path: "RAW-RETRY" },
  });
  const completed = value.reduce({
    type: "tool_execution_end",
    toolCallId: "retried-call",
    toolName: "read",
    result: { text: "RAW-RESULT" },
    isError: false,
  });
  const settled = value.reduce({ type: "agent_settled" });
  const tree = value.reduce({
    type: "session_tree",
    newLeafId: "new-leaf",
    oldLeafId: "old-leaf",
    summaryEntry: { summary: "RAW-TREE" },
    fromExtension: false,
  });
  const shutdown = value.reduce({
    type: "session_shutdown",
    reason: "fork",
    targetSessionFile: "/RAW/SESSION/PATH",
  });

  assert.deepEqual(kinds(compact), ["coverage_boundary"]);
  assert.deepEqual(kinds(completed), ["tool_started", "tool_finished"]);
  assert.deepEqual(kinds(settled), ["turn_finished"]);
  assert.deepEqual(kinds(tree), ["coverage_boundary"]);
  assert.deepEqual(kinds(shutdown), ["session_finished"]);
  assert.equal(shutdown[0]?.kind === "session_finished" && shutdown[0].reason, "fork");
  assert.doesNotMatch(
    JSON.stringify([...compact, ...completed, ...settled, ...tree, ...shutdown]),
    /RAW-/,
  );
});

test("invalid lifecycle enums and fields fail closed without retaining sensitive values", () => {
  const value = reducer();
  for (const event of [
    { type: "session_compact", reason: "future", compactionEntry: { secret: "RAW" } },
    { type: "tool_execution_start", toolCallId: "", toolName: "read", args: "RAW" },
    {
      type: "tool_execution_end",
      toolCallId: "call",
      toolName: "read",
      result: "RAW",
      isError: "false",
    },
  ]) {
    const records = value.reduce(event);
    assert.deepEqual(kinds(records), ["coverage_degraded"]);
    assert.doesNotMatch(JSON.stringify(records), /RAW/);
  }

  assert.throws(
    () =>
      new PiWindowReducer({
        sessionID: "x".repeat(MAX_PI_NATIVE_SESSION_ID_BYTES + 1),
        windowDiscriminator: WINDOW,
      }),
  );
});

test("missing critical compact/tree fields and invalid shutdown keep the window incomplete", () => {
  const value = reducer();
  assert.deepEqual(
    kinds(
      value.reduce({
        type: "session_compact",
        reason: "manual",
        fromExtension: false,
      }),
    ),
    ["coverage_degraded"],
  );
  assert.deepEqual(
    kinds(value.reduce({ type: "session_tree", newLeafId: "new-leaf" })),
    ["coverage_degraded"],
  );
  assert.deepEqual(
    kinds(value.reduce({ type: "session_shutdown", reason: "future" })),
    ["coverage_degraded"],
  );
  assert.equal(value.isFinished(), false);
  assert.deepEqual(
    kinds(value.reduce({ type: "session_shutdown", reason: "quit" })),
    ["session_finished"],
  );
  assert.equal(value.isFinished(), true);
});

test("reducer state and emitted records remain bounded under call and record pressure", () => {
  const callPressure = reducer();
  for (let index = 0; index < 1_000; index += 1) {
    assert.deepEqual(
      callPressure.reduce({
        type: "tool_execution_start",
        toolCallId: `call-${index}`,
        toolName: "read",
        args: null,
      }),
      [],
    );
  }
  assert.deepEqual(
    kinds(
      callPressure.reduce({
        type: "tool_execution_start",
        toolCallId: "call-overflow",
        toolName: "read",
        args: null,
      }),
    ),
    ["coverage_degraded"],
  );

  const recordPressure = reducer();
  const emitted: ReducedPiRecord[] = [];
  for (let index = 0; index < 600; index += 1) {
    recordPressure.reduce({
      type: "tool_execution_start",
      toolCallId: `finished-${index}`,
      toolName: "read",
      args: null,
    });
    emitted.push(
      ...recordPressure.reduce({
        type: "tool_execution_end",
        toolCallId: `finished-${index}`,
        toolName: "read",
        result: null,
        isError: false,
      }),
    );
  }
  emitted.push(
    ...recordPressure.reduce({ type: "session_shutdown", reason: "quit" }),
  );

  assert.ok(emitted.length <= 997);
  assert.equal(emitted.filter((record) => record.kind === "coverage_degraded").length, 1);
  assert.equal(emitted.at(-1)?.kind, "session_finished");
  assert.deepEqual(
    emitted.map((record) => Number(record.event_id)),
    Array.from({ length: emitted.length }, (_, index) => index + 1),
  );
});
