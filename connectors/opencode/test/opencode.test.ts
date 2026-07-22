import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MAX_CAPTURE_CALL_ID_BYTES,
  MAX_CAPTURE_EVENT_BYTES,
  MAX_CAPTURE_EVENT_ID_BYTES,
  MAX_CAPTURE_TOOL_NAME_BYTES,
  encodeCanonicalJson,
} from "@saliencegate/bridge-core";

import {
  OpenCodeEventReducer,
  SerialSessionQueue,
  launcherInvocation,
  type ReducedOpenCodeRecord,
} from "../src/index.ts";

function toolEvent(input: {
  sessionID?: string;
  callID?: string;
  status: "pending" | "running" | "completed" | "error";
  eventID?: string;
  tool?: string;
  stateInput?: unknown;
}): object {
  const sessionID = input.sessionID ?? "session-one";
  const stateInput = input.stateInput ?? { path: "synthetic.txt" };
  const common = { input: stateInput };
  const state =
    input.status === "pending"
      ? { ...common, status: "pending", raw: "ignored raw" }
      : input.status === "running"
        ? { ...common, status: "running", time: { start: 1 } }
        : input.status === "completed"
          ? {
              ...common,
              status: "completed",
              output: "ignored output",
              title: "ignored title",
              metadata: { ignored: true },
              time: { start: 1, end: 2 },
            }
          : {
              ...common,
              status: "error",
              error: "ignored error",
              time: { start: 1, end: 2 },
            };
  return {
    ...(input.eventID === undefined ? {} : { id: input.eventID }),
    type: "message.part.updated",
    properties: {
      part: {
        id: "ignored-part",
        sessionID,
        messageID: "ignored-message",
        type: "tool",
        callID: input.callID ?? "call-one",
        tool: input.tool ?? "read",
        state,
      },
    },
  };
}

function kinds(records: readonly ReducedOpenCodeRecord[]): string[] {
  return records.map((item) => item.kind);
}

test("text parts are rejected after the discriminant without reading body content", () => {
  const part = new Proxy(
    { type: "text" },
    {
      get(target, property, receiver) {
        if (property !== "type") throw new Error("text body was traversed");
        return Reflect.get(target, property, receiver);
      },
      ownKeys() {
        throw new Error("text body was traversed");
      },
    },
  );
  const reducer = new OpenCodeEventReducer();

  assert.deepEqual(
    reducer.reduce({ type: "message.part.updated", properties: { part } }),
    [],
  );
});

test("pending/running/completed is monotonic and duplicate-idempotent", () => {
  const reducer = new OpenCodeEventReducer();

  assert.deepEqual(kinds(reducer.reduce(toolEvent({ status: "pending", eventID: "e1" }))), [
    "tool_started",
  ]);
  assert.deepEqual(reducer.reduce(toolEvent({ status: "pending", eventID: "e1" })), []);
  assert.deepEqual(reducer.reduce(toolEvent({ status: "running", eventID: "e2" })), []);
  assert.deepEqual(kinds(reducer.reduce(toolEvent({ status: "completed", eventID: "e3" }))), [
    "tool_finished",
  ]);
  assert.deepEqual(reducer.reduce(toolEvent({ status: "completed", eventID: "e3" })), []);
});

test("an unseen terminal state emits an observed action before its structured outcome", () => {
  const completed = new OpenCodeEventReducer().reduce(toolEvent({ status: "completed" }));
  const failed = new OpenCodeEventReducer().reduce(toolEvent({ status: "error" }));

  assert.deepEqual(kinds(completed), ["tool_started", "tool_finished"]);
  assert.equal(completed[1]?.kind === "tool_finished" && completed[1].outcome, "succeeded");
  assert.deepEqual(kinds(failed), ["tool_started", "tool_finished"]);
  assert.equal(failed[1]?.kind === "tool_finished" && failed[1].outcome, "failed");
});

test("regressive, conflicting, and reused event transitions degrade without invented outcomes", () => {
  const reducer = new OpenCodeEventReducer();
  reducer.reduce(toolEvent({ status: "completed", eventID: "same" }));

  assert.deepEqual(kinds(reducer.reduce(toolEvent({ status: "running", eventID: "later" }))), [
    "coverage_degraded",
  ]);
  assert.deepEqual(
    kinds(
      reducer.reduce(
        toolEvent({ status: "error", eventID: "same", callID: "different-call" }),
      ),
    ),
    ["coverage_degraded"],
  );
});

test("sessions with the same call ID remain separate and parent metadata is ignored", () => {
  const reducer = new OpenCodeEventReducer();
  const first = reducer.reduce(toolEvent({ sessionID: "parent", callID: "shared", status: "pending" }));
  const second = reducer.reduce(toolEvent({ sessionID: "child", callID: "shared", status: "pending" }));
  const parentMetadata = reducer.reduce({
    type: "session.created",
    properties: { info: { id: "child", parentID: "parent", title: "ignored" } },
  });

  assert.deepEqual(kinds(first), ["tool_started"]);
  assert.deepEqual(kinds(second), ["tool_started"]);
  assert.deepEqual(parentMetadata, []);
});

test("idle, error, compacted, deleted, and dispose have closed passive meanings", () => {
  const reducer = new OpenCodeEventReducer();
  reducer.reduce(toolEvent({ sessionID: "session-life", status: "pending" }));

  assert.deepEqual(kinds(reducer.reduce({ type: "session.idle", properties: { sessionID: "session-life" } })), [
    "turn_finished",
  ]);
  assert.deepEqual(
    kinds(
      reducer.reduce({
        type: "session.error",
        properties: { sessionID: "session-life", error: { name: "UnknownError", data: { message: "ignored" } } },
      }),
    ),
    ["controller_failed"],
  );
  assert.deepEqual(
    kinds(reducer.reduce({ type: "session.compacted", properties: { sessionID: "session-life" } })),
    ["coverage_boundary"],
  );
  assert.deepEqual(
    kinds(
      reducer.reduce({
        type: "session.deleted",
        properties: { info: { id: "session-life", title: "ignored" } },
      }),
    ),
    ["session_finished"],
  );
  assert.deepEqual(reducer.dispose(), []);
});

test("attributable lifecycle runtime IDs replay idempotently and collisions degrade", () => {
  const reducer = new OpenCodeEventReducer();
  reducer.reduce(toolEvent({ sessionID: "session-replay", status: "pending" }));
  const idle = {
    id: "lifecycle-idle",
    type: "session.idle",
    properties: { sessionID: "session-replay" },
  };
  const error = {
    id: "lifecycle-error",
    type: "session.error",
    properties: { sessionID: "session-replay", error: { ignored: true } },
  };
  const compacted = {
    id: "lifecycle-compacted",
    type: "session.compacted",
    properties: { sessionID: "session-replay" },
  };

  assert.deepEqual(kinds(reducer.reduce(idle)), ["turn_finished"]);
  assert.deepEqual(reducer.reduce(idle), []);
  assert.deepEqual(kinds(reducer.reduce(error)), ["controller_failed"]);
  assert.deepEqual(reducer.reduce(error), []);
  assert.deepEqual(kinds(reducer.reduce(compacted)), ["coverage_boundary"]);
  assert.deepEqual(reducer.reduce(compacted), []);
  assert.deepEqual(
    kinds(
      reducer.reduce({
        id: "lifecycle-idle",
        type: "session.compacted",
        properties: { sessionID: "session-replay" },
      }),
    ),
    ["coverage_degraded"],
  );
});

test("an uncorrelated session error is not attributed to every known session", () => {
  const reducer = new OpenCodeEventReducer();
  reducer.reduce(toolEvent({ sessionID: "known-session", status: "pending" }));

  assert.deepEqual(
    reducer.reduce({
      type: "session.error",
      properties: {
        error: { name: "UnknownError", data: { message: "ignored" } },
      },
    }),
    [],
  );
});

test("unknown event types and malformed session IDs cannot poison session capacity", () => {
  const reducer = new OpenCodeEventReducer();
  for (let index = 0; index < 256; index += 1) {
    assert.deepEqual(
      reducer.reduce({
        id: `unknown-${index}`,
        type: "session.future-event",
        properties: { sessionID: `unknown-session-${index}` },
      }),
      [],
    );
    assert.deepEqual(
      reducer.reduce({
        id: `lifecycle-${index}`,
        type: "session.idle",
        properties: { sessionID: `malformed-${index}-\ud800` },
      }),
      [],
    );
    assert.deepEqual(
      reducer.reduce(
        toolEvent({
          sessionID: `malformed-tool-${index}-\udc00`,
          callID: `call-${index}`,
          status: "pending",
        }),
      ),
      [],
    );
  }

  assert.deepEqual(
    kinds(reducer.reduce(toolEvent({ sessionID: "unpoisoned-session", status: "pending" }))),
    ["tool_started"],
  );

});

test("optional runtime IDs omit invalid correlation while malformed critical IDs degrade", () => {
  const optional = new OpenCodeEventReducer();
  const tool = optional.reduce(
    toolEvent({ sessionID: "optional-id", status: "pending", eventID: "bad-\ud800" }),
  );
  const lifecycle = optional.reduce({
    id: "bad-\udc00",
    type: "session.idle",
    properties: { sessionID: "optional-id" },
  });

  assert.deepEqual(kinds(tool), ["tool_started"]);
  assert.equal("event_id" in tool[0]!, false);
  assert.deepEqual(kinds(lifecycle), ["turn_finished"]);
  assert.equal("event_id" in lifecycle[0]!, false);

  const malformedCall = new OpenCodeEventReducer();
  assert.deepEqual(
    kinds(malformedCall.reduce(toolEvent({ status: "pending", callID: "bad-\ud800" }))),
    ["coverage_degraded"],
  );
  assert.deepEqual(malformedCall.reduce(toolEvent({ status: "pending" })), []);

  const malformedTool = new OpenCodeEventReducer();
  assert.deepEqual(
    kinds(malformedTool.reduce(toolEvent({ status: "pending", tool: "bad-\udc00" }))),
    ["coverage_degraded"],
  );

  const malformedOuter = new OpenCodeEventReducer();
  const contradictory = toolEvent({ status: "pending" }) as {
    properties: Record<string, unknown>;
  };
  contradictory.properties.sessionID = "bad-\ud800";
  assert.deepEqual(kinds(malformedOuter.reduce(contradictory)), ["coverage_degraded"]);
});

test("missing critical tool fields disable that session without exposing content", () => {
  const reducer = new OpenCodeEventReducer();
  const invalid = reducer.reduce(toolEvent({ status: "pending", callID: "" }));
  const later = reducer.reduce(toolEvent({ status: "running" }));

  assert.deepEqual(kinds(invalid), ["coverage_degraded"]);
  assert.deepEqual(later, []);
  assert.doesNotMatch(JSON.stringify(invalid), /ignored raw|synthetic\.txt/);
});

test("runtime identifiers are independently capped below the reduced-event envelope", () => {
  const eventID = "e".repeat(MAX_CAPTURE_EVENT_ID_BYTES);
  const callID = "c".repeat(MAX_CAPTURE_CALL_ID_BYTES);
  const tool = "t".repeat(MAX_CAPTURE_TOOL_NAME_BYTES);
  const accepted = new OpenCodeEventReducer().reduce(
    toolEvent({ status: "pending", eventID, callID, tool, stateInput: null }),
  );

  assert.deepEqual(kinds(accepted), ["tool_started"]);
  assert.ok(encodeCanonicalJson(accepted[0]!).byteLength <= MAX_CAPTURE_EVENT_BYTES);

  const omitted = new OpenCodeEventReducer().reduce(
    toolEvent({ status: "pending", eventID: `${eventID}x` }),
  );
  assert.deepEqual(kinds(omitted), ["tool_started"]);
  assert.equal("event_id" in omitted[0]!, false);

  for (const value of [
    toolEvent({ status: "pending", callID: `${callID}x` }),
    toolEvent({ status: "pending", tool: `${tool}x` }),
  ]) {
    assert.deepEqual(kinds(new OpenCodeEventReducer().reduce(value)), ["coverage_degraded"]);
  }
});

test("session, event-id, and emitted-record state is hard bounded", () => {
  const reducer = new OpenCodeEventReducer();
  for (let index = 0; index < 256; index += 1) {
    assert.deepEqual(
      kinds(
        reducer.reduce(
          toolEvent({ sessionID: `session-${index}`, callID: "first", status: "pending" }),
        ),
      ),
      ["tool_started"],
    );
  }
  const sessionOverflow = reducer.reduce(
    toolEvent({ sessionID: "session-over-cap", status: "pending" }),
  );
  assert.deepEqual(kinds(sessionOverflow), ["coverage_degraded"]);
  assert.equal(
    sessionOverflow[0]?.kind === "coverage_degraded" && sessionOverflow[0].reason,
    "overflow",
  );

  const eventIDs = new OpenCodeEventReducer();
  eventIDs.reduce(toolEvent({ status: "pending", eventID: "event-0" }));
  for (let index = 1; index < 4_096; index += 1) {
    assert.deepEqual(
      eventIDs.reduce(toolEvent({ status: "pending", eventID: `event-${index}` })),
      [],
    );
  }
  assert.deepEqual(
    kinds(eventIDs.reduce(toolEvent({ status: "pending", eventID: "event-over-cap" }))),
    ["coverage_degraded"],
  );
  assert.deepEqual(eventIDs.reduce(toolEvent({ status: "completed" })), []);

  const records = new OpenCodeEventReducer();
  for (let index = 0; index < 995; index += 1) {
    assert.deepEqual(
      kinds(records.reduce(toolEvent({ callID: `call-${index}`, status: "pending" }))),
      ["tool_started"],
    );
  }
  const overflow = records.reduce(toolEvent({ callID: "call-over-cap", status: "pending" }));
  assert.deepEqual(kinds(overflow), ["coverage_degraded"]);
  assert.equal(overflow[0]?.kind === "coverage_degraded" && overflow[0].reason, "overflow");
  assert.deepEqual(records.reduce(toolEvent({ callID: "call-after-cap", status: "pending" })), []);
  assert.deepEqual(
    kinds(
      records.reduce({
        type: "session.deleted",
        properties: { info: { id: "session-one" } },
      }),
    ),
    ["session_finished"],
  );
});

test("deleted sessions release the lifetime session slot after their terminal record", () => {
  const reducer = new OpenCodeEventReducer();
  for (let index = 0; index < 300; index += 1) {
    const sessionID = `sequential-session-${index}`;
    assert.deepEqual(
      kinds(reducer.reduce(toolEvent({ sessionID, callID: "call", status: "pending" }))),
      ["tool_started"],
    );
    assert.deepEqual(
      kinds(
        reducer.reduce({
          type: "session.deleted",
          properties: { info: { id: sessionID } },
        }),
      ),
      ["session_finished"],
    );
  }
  assert.deepEqual(
    reducer.reduce({
      type: "session.deleted",
      properties: { info: { id: "sequential-session-0" } },
    }),
    [],
  );
  assert.deepEqual(
    reducer.reduce(
      toolEvent({ sessionID: "sequential-session-0", callID: "late", status: "pending" }),
    ),
    [],
  );
});

test("finalized tombstones dedupe deletion IDs, detect collisions, and suppress late callbacks", () => {
  const reducer = new OpenCodeEventReducer();
  reducer.reduce(toolEvent({ sessionID: "finalized-session", status: "pending" }));
  const deleted = {
    id: "final-event",
    type: "session.deleted",
    properties: { info: { id: "finalized-session" } },
  };

  assert.deepEqual(kinds(reducer.reduce(deleted)), ["session_finished"]);
  assert.deepEqual(reducer.reduce(deleted), []);
  assert.deepEqual(
    reducer.reduce(toolEvent({ sessionID: "finalized-session", status: "completed" })),
    [],
  );
  assert.deepEqual(
    kinds(
      reducer.reduce({
        id: "final-event",
        type: "session.idle",
        properties: { sessionID: "finalized-session" },
      }),
    ),
    ["coverage_degraded"],
  );
  assert.deepEqual(
    reducer.reduce({
      id: "later-id",
      type: "session.compacted",
      properties: { sessionID: "finalized-session" },
    }),
    [],
  );
});

test("global reducer state digests large identifiers and discloses bounded budget overflow", () => {
  const reducer = new OpenCodeEventReducer();
  const sessions: string[] = [];
  let overflowed = false;

  for (let sessionIndex = 0; sessionIndex < 8 && !overflowed; sessionIndex += 1) {
    const sessionID = `budget-session-${sessionIndex}`;
    sessions.push(sessionID);
    for (let callIndex = 0; callIndex < 20; callIndex += 1) {
      const prefix = `${sessionIndex}-${callIndex}-`;
      const callID = `${prefix}${"c".repeat(MAX_CAPTURE_CALL_ID_BYTES - prefix.length)}`;
      const eventID = `${prefix}${"e".repeat(MAX_CAPTURE_EVENT_ID_BYTES - prefix.length)}`;
      const result = reducer.reduce(
        toolEvent({
          sessionID,
          callID,
          eventID,
          tool: "t".repeat(MAX_CAPTURE_TOOL_NAME_BYTES),
          status: "pending",
        }),
      );
      if (result[0]?.kind === "coverage_degraded") {
        assert.equal(result[0].reason, "overflow");
        overflowed = true;
        break;
      }
      assert.deepEqual(kinds(result), ["tool_started"]);
    }
  }
  assert.equal(overflowed, true);

  for (const sessionID of sessions) {
    reducer.reduce({ type: "session.deleted", properties: { info: { id: sessionID } } });
  }
  assert.deepEqual(
    kinds(
      reducer.reduce(
        toolEvent({ sessionID: "budget-recovered", callID: "new", status: "pending" }),
      ),
    ),
    ["tool_started"],
  );
});

test("per-session queues serialize one session while allowing independent sessions", async () => {
  const queue = new SerialSessionQueue();
  const order: string[] = [];
  let releaseFirst!: () => void;
  const gate = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });

  const first = queue.run("same", async () => {
    order.push("same-1-start");
    await gate;
    order.push("same-1-end");
  });
  const second = queue.run("same", async () => order.push("same-2"));
  const independent = queue.run("other", async () => order.push("other"));
  await independent;
  releaseFirst();
  await Promise.all([first, second]);

  assert.deepEqual(order, ["same-1-start", "other", "same-1-end", "same-2"]);
});

test("launcher invocation keeps metacharacters in data and never enables a shell", () => {
  const providerCredentials = {
    ANTHROPIC_API_KEY: "anthropic-poison",
    AZURE_OPENAI_API_KEY: "azure-poison",
    OPENAI_API_KEY: "openai-poison",
    OPENAI_ORGANIZATION: "organization-poison",
    OPENAI_ORG_ID: "organization-id-poison",
    OPENAI_PROJECT: "project-poison",
    OPENAI_PROJECT_ID: "project-id-poison",
    openai_api_key: "case-folded-api-key-poison",
  };
  const posix = launcherInvocation({
    platform: "linux",
    launcherPath: "/tmp/project & $(touch nope)/capture-hook",
    environment: { PATH: "/usr/bin", SAFE_MARKER: "preserved", ...providerCredentials },
  });
  assert.equal(posix.file, "/tmp/project & $(touch nope)/capture-hook");
  assert.deepEqual(posix.arguments, []);
  assert.equal(posix.options.shell, false);
  assert.equal(posix.options.env.SAFE_MARKER, "preserved");
  assert.equal(
    Object.keys(posix.options.env).some((key) =>
      [
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
      ].includes(key.toUpperCase()),
    ),
    false,
  );

  const windows = launcherInvocation({
    platform: "win32",
    launcherPath: "C:\\State & harmless\\capture-hook.cmd",
    environment: {
      SystemRoot: "C:\\Windows",
      PATH: "C:\\Windows\\System32",
      saliencegate_launcher: "C:\\attacker-controlled.cmd",
      ...providerCredentials,
    },
  });
  assert.equal(windows.file, "C:\\Windows\\System32\\cmd.exe");
  assert.deepEqual(windows.arguments, ["/d", "/v:off", "/s", "/c", '"%SALIENCEGATE_LAUNCHER%"']);
  assert.equal(windows.options.env.SALIENCEGATE_LAUNCHER, "C:\\State & harmless\\capture-hook.cmd");
  assert.equal("saliencegate_launcher" in windows.options.env, false);
  assert.equal(
    Object.keys(windows.options.env).some((key) => key.toUpperCase().includes("OPENAI")),
    false,
  );
  assert.equal(windows.options.shell, false);
});
