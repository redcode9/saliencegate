import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Writable } from "node:stream";
import { pathToFileURL } from "node:url";
import { test } from "node:test";

import {
  BridgeContractError,
  CAPTURE_LAUNCHER_TIMEOUT_MS,
  WindowedBatchTransport,
  encodeCanonicalJson,
  launcherInvocation,
  spawnWindowedCaptureChunk,
  type BootstrapBinding,
  type CanonicalJson,
  type WindowCoordinates,
  type WindowedCaptureChunkWrite,
} from "@saliencegate/bridge-core";

import {
  createPiExtension,
  loadPiBootstrap,
  type PiExtensionAPI,
  type PiExtensionContext,
} from "../src/index.ts";

const SESSION_ID = "019c0eaf-7b31-7000-8000-000000000001";
const WINDOW_A = "a".repeat(64);
const WINDOW_B = "b".repeat(64);
const CROSS_LANGUAGE_OVERSIZE_BATCH = new URL(
  "../../../tests/fixtures/pi-cross-language-oversize-batch.json",
  import.meta.url,
);
const crossLanguageBootstrap: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "pi-extension/v1",
  connection_id: `sg-${"6".repeat(48)}`,
  launcher_path: "/private/tmp/saliencegate-pi-hook",
  capability_digest: "777d82ed469eb613251b6062a088e62cf2a1bc9b714a2f69634c4eba1c86b248",
  bundle_digest: "9".repeat(64),
  receipt_mac: "a".repeat(64),
};

const bootstrap: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "pi-extension/v1",
  connection_id: `sg-${"1".repeat(48)}`,
  launcher_path: "/synthetic/state/pi/capture-hook",
  capability_digest: "2".repeat(64),
  bundle_digest: "3".repeat(64),
  receipt_mac: "4".repeat(64),
};

type Handler = (event: unknown, context: PiExtensionContext) => unknown;

class FakePiAPI {
  readonly handlers = new Map<string, Handler>();

  on(event: string, handler: Handler): void {
    assert.equal(this.handlers.has(event), false);
    this.handlers.set(event, handler);
  }
}

function context(
  sessionID = SESSION_ID,
  workspacePath?: string,
): PiExtensionContext {
  const manager = new Proxy(
    { getSessionId: () => sessionID },
    {
      get(target, property, receiver) {
        if (property !== "getSessionId") {
          throw new Error(`forbidden session API ${String(property)}`);
        }
        return Reflect.get(target, property, receiver);
      },
    },
  );
  const target = {
    sessionManager: manager,
    ...(workspacePath === undefined ? {} : { cwd: workspacePath }),
  };
  return new Proxy(
    target,
    {
      get(target, property, receiver) {
        if (property !== "sessionManager" && property !== "cwd") {
          throw new Error(`forbidden context field ${String(property)}`);
        }
        return Reflect.get(target, property, receiver);
      },
    },
  ) as PiExtensionContext;
}

function ignoredGetter(name: string): PropertyDescriptor {
  return {
    enumerable: true,
    get() {
      throw new Error(`${name} was traversed`);
    },
  };
}

function decode(write: WindowedCaptureChunkWrite): Record<string, unknown> {
  return JSON.parse(write.bytes.toString("utf8")) as Record<string, unknown>;
}

function events(writes: readonly WindowedCaptureChunkWrite[]): Record<string, unknown>[] {
  return writes.flatMap((write) => {
    const value = decode(write).events;
    assert.ok(Array.isArray(value));
    return value as Record<string, unknown>[];
  });
}

async function invoke(
  api: FakePiAPI,
  event: string,
  value: unknown,
  ctx = context(),
): Promise<unknown> {
  const handler = api.handlers.get(event);
  assert.ok(handler, `missing handler ${event}`);
  return await handler(value, ctx);
}

test("the extension registers exactly the eight observational callbacks and returns undefined", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  const extension = createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "5".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  });

  assert.equal(await extension(api as PiExtensionAPI), undefined);
  assert.deepEqual([...api.handlers.keys()], [
    "session_start",
    "before_agent_start",
    "tool_execution_start",
    "tool_execution_end",
    "agent_settled",
    "session_compact",
    "session_tree",
    "session_shutdown",
  ]);

  const sessionStart = Object.defineProperties(
    { type: "session_start", reason: "startup" },
    { previousSessionFile: ignoredGetter("previousSessionFile") },
  );
  const before = Object.defineProperties(
    { type: "before_agent_start" },
    {
      prompt: ignoredGetter("prompt"),
      images: ignoredGetter("images"),
      systemPrompt: ignoredGetter("systemPrompt"),
      systemPromptOptions: ignoredGetter("systemPromptOptions"),
    },
  );
  const start = Object.defineProperties(
    {
      type: "tool_execution_start",
      toolCallId: "call-one",
      toolName: "read",
    },
    { args: ignoredGetter("args") },
  );
  const end = Object.defineProperties(
    {
      type: "tool_execution_end",
      toolCallId: "call-one",
      toolName: "read",
      isError: false,
    },
    { result: ignoredGetter("result") },
  );
  const compact = Object.defineProperties(
    {
      type: "session_compact",
      reason: "overflow",
      fromExtension: false,
      willRetry: true,
    },
    { compactionEntry: ignoredGetter("compactionEntry") },
  );
  const tree = Object.defineProperties(
    { type: "session_tree", newLeafId: null, oldLeafId: null },
    { summaryEntry: ignoredGetter("summaryEntry") },
  );
  const shutdown = Object.defineProperties(
    { type: "session_shutdown", reason: "quit" },
    { targetSessionFile: ignoredGetter("targetSessionFile") },
  );

  for (const [name, value] of [
    ["session_start", sessionStart],
    ["before_agent_start", before],
    ["tool_execution_start", start],
    ["tool_execution_end", end],
    ["agent_settled", { type: "agent_settled" }],
    ["session_compact", compact],
    ["session_tree", tree],
    ["session_shutdown", shutdown],
  ] as const) {
    assert.equal(await invoke(api, name, value), undefined);
  }

  assert.equal((decode(writes[0]!).events as unknown[]).length, 0);
  const reduced = events(writes);
  assert.deepEqual(
    reduced.map((record) => record.kind),
    [
      "tool_started",
      "tool_finished",
      "turn_finished",
      "coverage_boundary",
      "coverage_boundary",
      "session_finished",
    ],
  );
  assert.ok(
    writes.every(
      (write) => decode(write).window_discriminator === WINDOW_A,
    ),
  );
  assert.doesNotMatch(Buffer.concat(writes.map((write) => write.bytes)).toString("utf8"), /RAW/);
});

test("reload, resume, new, and fork starts create distinct windows even for one native ID", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const discriminators = [WINDOW_A, WINDOW_B, "c".repeat(64), "d".repeat(64)];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "6".repeat(64),
    windowDiscriminator: () => discriminators.shift()!,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  for (const reason of ["reload", "resume", "new"] as const) {
    await invoke(api, "session_shutdown", { type: "session_shutdown", reason });
    await invoke(api, "session_start", { type: "session_start", reason });
  }
  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "fork" });

  const documents = writes.map(decode);
  const emptyStarts = documents.filter(
    (document) => Array.isArray(document.events) && document.events.length === 0,
  );
  assert.deepEqual(
    emptyStarts.map((document) => document.window_discriminator),
    [WINDOW_A, WINDOW_B, "c".repeat(64), "d".repeat(64)],
  );
  assert.ok(documents.every((document) => document.session_id === SESSION_ID));
});

test("the extension binds its absolute workspace path to every window batch", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  const workspacePath =
    process.platform === "win32"
      ? "C:\\workspace\\saliencegate"
      : "/workspace/saliencegate";
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "7".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);
  const workspaceContext = context(SESSION_ID, workspacePath);

  await invoke(
    api,
    "session_start",
    { type: "session_start", reason: "startup" },
    workspaceContext,
  );
  await invoke(
    api,
    "session_shutdown",
    { type: "session_shutdown", reason: "quit" },
    workspaceContext,
  );

  assert.ok(writes.length >= 2);
  assert.ok(
    writes.every((write) => decode(write).workspace_path === workspacePath),
  );
});

test("callbacks reject a mismatched native session without opening a foreign window", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "7".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  await invoke(
    api,
    "before_agent_start",
    { type: "before_agent_start", systemPromptOptions: {} },
    context("019c0eaf-7b31-7000-8000-000000000099"),
  );
  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "quit" });

  const reduced = events(writes);
  assert.deepEqual(
    reduced.map((record) => record.kind),
    ["coverage_degraded", "session_finished"],
  );
  assert.ok(reduced.every((record) => record.session_id === SESSION_ID));
  assert.ok(reduced.every((record) => record.window_discriminator === WINDOW_A));
});

test("bootstrap failure is fail-open and leaves all callbacks inert", async () => {
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///missing/saliencegate.bootstrap.json"),
    loadBootstrap: async () => {
      throw new BridgeContractError();
    },
  })(api as PiExtensionAPI);

  assert.equal(
    await invoke(api, "session_start", { type: "session_start", reason: "startup" }),
    undefined,
  );
  assert.equal(api.handlers.size, 8);
});

test("invalid or missing session_start reason creates no observation window", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start" });
  await invoke(api, "session_start", { type: "session_start", reason: "future" });
  await invoke(api, "before_agent_start", {
    type: "before_agent_start",
    systemPromptOptions: {},
  });
  assert.deepEqual(writes, []);
});

test("invalid shutdown emits degradation but no terminal and does not close the window", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "d".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "future" });
  assert.deepEqual(
    events(writes).map((record) => record.kind),
    ["coverage_degraded"],
  );
  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "quit" });
  const reduced = events(writes);
  assert.equal(reduced.filter((record) => record.kind === "session_finished").length, 1);
  assert.equal(reduced.at(-1)?.kind, "session_finished");
});

test("cyclic, huge, and accessor-bearing args and results are never traversed", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "e".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);
  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  await invoke(api, "before_agent_start", {
    type: "before_agent_start",
    systemPromptOptions: {},
  });

  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  const accessor = Object.defineProperty({}, "secret", ignoredGetter("nested secret"));
  const sensitive = [cycle, { payload: "RAW-HUGE".repeat(200_000) }, accessor];
  for (const [index, payload] of sensitive.entries()) {
    const start = Object.defineProperties(
      {
        type: "tool_execution_start",
        toolCallId: `opaque-call-${index}`,
        toolName: "read",
      },
      index === 2 ? { args: ignoredGetter("args") } : { args: { value: payload, enumerable: true } },
    );
    const end = Object.defineProperties(
      {
        type: "tool_execution_end",
        toolCallId: `opaque-call-${index}`,
        toolName: "read",
        isError: false,
      },
      index === 2
        ? { result: ignoredGetter("result") }
        : { result: { value: payload, enumerable: true } },
    );
    await invoke(api, "tool_execution_start", start);
    await invoke(api, "tool_execution_end", end);
  }
  await invoke(api, "agent_settled", { type: "agent_settled" });

  const encoded = Buffer.concat(writes.map((write) => write.bytes)).toString("utf8");
  assert.doesNotMatch(encoded, /RAW-HUGE|secret|payload|self/);
  assert.equal(events(writes).filter((record) => record.kind === "tool_started").length, 3);
  assert.equal(events(writes).filter((record) => record.kind === "tool_finished").length, 3);
});

test("a pending boundary flush serializes shutdown and emits one ordered terminal", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  let releaseBlocked: ((value: boolean) => void) | undefined;
  let blockNext = false;
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      if (!blockNext) return true;
      blockNext = false;
      return await new Promise<boolean>((resolve) => {
        releaseBlocked = resolve;
      });
    },
    batchID: () => "f".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  await invoke(api, "before_agent_start", {
    type: "before_agent_start",
    systemPromptOptions: {},
  });
  await invoke(api, "tool_execution_start", {
    type: "tool_execution_start",
    toolCallId: "race-call",
    toolName: "read",
    args: null,
  });
  await invoke(api, "tool_execution_end", {
    type: "tool_execution_end",
    toolCallId: "race-call",
    toolName: "read",
    result: null,
    isError: false,
  });

  blockNext = true;
  const settled = invoke(api, "agent_settled", { type: "agent_settled" });
  const shutdown = invoke(api, "session_shutdown", {
    type: "session_shutdown",
    reason: "quit",
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(events(writes).some((record) => record.kind === "session_finished"), false);
  assert.ok(releaseBlocked);
  releaseBlocked(true);
  await Promise.all([settled, shutdown]);

  const reduced = events(writes);
  assert.equal(reduced.filter((record) => record.kind === "session_finished").length, 1);
  assert.equal(reduced.at(-1)?.kind, "session_finished");
  assert.deepEqual(
    reduced.map((record) => record.event_id),
    ["1", "2", "3", "4"],
  );
});

test("no-op and disabled boundaries cannot exhaust receipts before shutdown", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "9".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  assert.equal(writes.length, 1);
  for (let index = 0; index < 1_005; index += 1) {
    await invoke(api, "agent_settled", { type: "agent_settled" });
  }
  assert.equal(writes.length, 1);

  for (let index = 0; index < 600; index += 1) {
    await invoke(api, "tool_execution_start", {
      type: "tool_execution_start",
      toolCallId: `pressure-${index}`,
      toolName: "read",
      args: null,
    });
    await invoke(api, "tool_execution_end", {
      type: "tool_execution_end",
      toolCallId: `pressure-${index}`,
      toolName: "read",
      result: null,
      isError: false,
    });
  }
  await invoke(api, "agent_settled", { type: "agent_settled" });
  const afterPressure = writes.length;
  assert.ok(afterPressure > 1);

  for (let index = 0; index < 1_005; index += 1) {
    await invoke(api, "agent_settled", { type: "agent_settled" });
  }
  assert.equal(writes.length, afterPressure);

  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "quit" });
  assert.equal(writes.length, afterPressure + 1);
  assert.equal(events(writes).at(-1)?.kind, "session_finished");
});

test("64 concurrent Pi callbacks remain ordered, complete, and provider-independent", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "a".repeat(64),
    windowDiscriminator: () => WINDOW_A,
  })(api as PiExtensionAPI);

  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  const starts = Array.from({ length: 64 }, (_, index) =>
    invoke(api, "tool_execution_start", {
      type: "tool_execution_start",
      toolCallId: `concurrent-${index}`,
      toolName: "read",
      args: null,
    }),
  );
  assert.ok((await Promise.all(starts)).every((result) => result === undefined));

  const finishes = Array.from({ length: 64 }, (_, index) =>
    invoke(api, "tool_execution_end", {
      type: "tool_execution_end",
      toolCallId: `concurrent-${index}`,
      toolName: "read",
      result: null,
      isError: false,
    }),
  );
  assert.ok((await Promise.all(finishes)).every((result) => result === undefined));
  await invoke(api, "agent_settled", { type: "agent_settled" });

  const reduced = events(writes);
  assert.equal(reduced.filter((record) => record.kind === "tool_started").length, 64);
  assert.equal(reduced.filter((record) => record.kind === "tool_finished").length, 64);
  assert.equal(reduced.filter((record) => record.kind === "coverage_degraded").length, 0);
  assert.equal(reduced.at(-1)?.kind, "turn_finished");
  assert.deepEqual(
    reduced.map((record) => record.event_id),
    Array.from({ length: 129 }, (_, index) => String(index + 1)),
  );
});

test("one-sided oversize success groups become one cross-language control", async () => {
  const writes: WindowedCaptureChunkWrite[] = [];
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => crossLanguageBootstrap,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
    batchID: () => "b".repeat(64),
    windowDiscriminator: () => "7".repeat(64),
  })(api as PiExtensionAPI);
  const ctx = context("synthetic-pi-session");

  await invoke(api, "session_start", { type: "session_start", reason: "startup" }, ctx);
  await invoke(
    api,
    "tool_execution_start",
    {
      type: "tool_execution_start",
      toolCallId: "\u0001".repeat(10_400),
      toolName: "\u0002".repeat(1_024),
      args: null,
    },
    ctx,
  );
  await invoke(
    api,
    "tool_execution_end",
    {
      type: "tool_execution_end",
      toolCallId: "\u0001".repeat(10_400),
      toolName: "\u0002".repeat(1_024),
      result: null,
      isError: false,
    },
    ctx,
  );
  await invoke(api, "agent_settled", { type: "agent_settled" }, ctx);

  assert.equal(writes.length, 2);
  assert.deepEqual(
    events(writes).map((record) => record.kind),
    ["oversize", "turn_finished"],
  );
  assert.deepEqual(writes[1]!.bytes, await readFile(CROSS_LANGUAGE_OVERSIZE_BATCH));
});

test("permanent gap poison waits for real evidence after a recovered start", async () => {
  const attempts: WindowedCaptureChunkWrite[] = [];
  let deliver = false;
  let discriminator = 0;
  let batch = 0;
  const api = new FakePiAPI();
  await createPiExtension({
    bootstrapURL: new URL("file:///synthetic/.pi/extensions/saliencegate.bootstrap.json"),
    loadBootstrap: async () => bootstrap,
    writeChunk: async (write) => {
      attempts.push(write);
      return deliver;
    },
    batchID: () => (batch++).toString(16).padStart(64, "0"),
    windowDiscriminator: () => (discriminator++).toString(16).padStart(64, "0"),
  })(api as PiExtensionAPI);

  for (let index = 0; index < 257; index += 1) {
    await invoke(api, "session_start", { type: "session_start", reason: "startup" });
    await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "quit" });
  }

  deliver = true;
  await invoke(api, "session_start", { type: "session_start", reason: "startup" });
  const afterRecoveredStart = attempts.length;
  for (let index = 0; index < 1_005; index += 1) {
    await invoke(api, "agent_settled", { type: "agent_settled" });
  }
  assert.equal(attempts.length, afterRecoveredStart);

  await invoke(api, "session_shutdown", { type: "session_shutdown", reason: "quit" });
  assert.equal(attempts.length, afterRecoveredStart + 1);
  const terminalEvents = decode(attempts.at(-1)!).events as Record<string, unknown>[];
  assert.equal(terminalEvents[0]?.reason, "transport_gap");
  assert.equal(terminalEvents.at(-1)?.kind, "session_finished");
});

test("windowed transport is byte-idempotent and reports a prior attempted gap once", async () => {
  const coordinates: WindowCoordinates = {
    sessionID: SESSION_ID,
    windowDiscriminator: WINDOW_A,
  };
  const writes: WindowedCaptureChunkWrite[] = [];
  let succeed = true;
  const transport = new WindowedBatchTransport(bootstrap, {
    writeChunk: async (write) => {
      writes.push(write);
      return succeed;
    },
    batchID: () => "8".repeat(64),
  });
  const first: CanonicalJson = {
    kind: "turn_finished",
    session_id: SESSION_ID,
    window_discriminator: WINDOW_A,
    event_id: "1",
  };

  assert.equal(await transport.flush(coordinates, [first]), "delivered");
  assert.equal(await transport.flush(coordinates, [first]), "delivered");
  assert.deepEqual(writes[0]!.bytes, writes[1]!.bytes);

  succeed = false;
  assert.equal(
    await transport.flush(coordinates, [
      { ...first, event_id: "2", kind: "coverage_boundary" },
    ]),
    "attempted_failure",
  );
  succeed = true;
  assert.equal(
    await transport.flush(coordinates, [
      { ...first, event_id: "3", kind: "turn_finished" },
    ]),
    "delivered",
  );
  const recovery = decode(writes.at(-1)!);
  const recoveryEvents = recovery.events as Record<string, unknown>[];
  assert.equal(recoveryEvents[0]?.kind, "coverage_degraded");
  assert.equal(recoveryEvents[0]?.reason, "transport_gap");
  assert.equal(recoveryEvents[0]?.window_discriminator, WINDOW_A);
  assert.equal(recoveryEvents[1]?.event_id, "3");
  assert.equal(transport.hasPendingGap(coordinates), false);
});

test("shared launcher treats metacharacters as data and absorbs spawn errors", async () => {
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
  const invocation = launcherInvocation({
    platform: "win32",
    launcherPath: "C:\\State & Data\\saliencegate.cmd",
    environment: {
      SystemRoot: "C:\\Windows",
      KEEP: "value",
      ...providerCredentials,
    },
  });
  assert.equal(invocation.file, "C:\\Windows\\System32\\cmd.exe");
  assert.deepEqual(invocation.arguments, [
    "/d",
    "/v:off",
    "/s",
    "/c",
    '""%SALIENCEGATE_LAUNCHER%""',
  ]);
  assert.equal(
    invocation.options.env.SALIENCEGATE_LAUNCHER,
    "C:\\State & Data\\saliencegate.cmd",
  );
  assert.equal(invocation.options.env.KEEP, "value");
  assert.equal(invocation.options.shell, false);
  assert.equal(invocation.options.windowsVerbatimArguments, true);
  assert.equal(
    Object.keys(invocation.options.env).some((key) =>
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

  const result = await spawnWindowedCaptureChunk(
    { invocation, bytes: Buffer.from("{}", "utf8") },
    () => {
      const child = new EventEmitter() as EventEmitter & {
        stdin: Writable;
        kill(): boolean;
      };
      child.stdin = new Writable({ write(_chunk, _encoding, callback) { callback(); } });
      child.kill = () => true;
      queueMicrotask(() => child.emit("error", new Error("synthetic spawn failure")));
      return child as never;
    },
  );
  assert.equal(result, false);
  assert.equal(CAPTURE_LAUNCHER_TIMEOUT_MS, 2_000);
});

test("Pi bootstrap verifies canonical sidecar, profile, bundle name, and digest", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "saliencegate-pi-bootstrap-"));
  t.after(async () => {
    await rm(root, { recursive: true, force: true });
  });
  const bundlePath = path.join(root, "saliencegate.ts");
  const bootstrapPath = path.join(root, "saliencegate.bootstrap.json");
  const bundle = Buffer.from("export default function () {}\n", "utf8");
  await writeFile(bundlePath, bundle, { mode: 0o600 });
  const binding: BootstrapBinding = {
    ...bootstrap,
    bundle_digest: createHash("sha256").update(bundle).digest("hex"),
  };
  await writeFile(bootstrapPath, encodeCanonicalJson(binding), { mode: 0o600 });
  await chmod(root, 0o700);

  assert.deepEqual(
    await loadPiBootstrap(pathToFileURL(bootstrapPath)),
    binding,
  );

  await writeFile(bundlePath, "tampered\n", { mode: 0o600 });
  await assert.rejects(
    loadPiBootstrap(pathToFileURL(bootstrapPath)),
    BridgeContractError,
  );
  await writeFile(bundlePath, bundle, { mode: 0o600 });
  await writeFile(
    bootstrapPath,
    encodeCanonicalJson({ ...binding, profile: "opencode-plugin/v1" }),
    { mode: 0o600 },
  );
  await assert.rejects(
    loadPiBootstrap(pathToFileURL(bootstrapPath)),
    BridgeContractError,
  );

  assert.equal((await readFile(bundlePath)).toString("utf8"), bundle.toString("utf8"));
});
