import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

import {
  encodeCanonicalJson,
  normalizeCaptureEvent,
  type BootstrapBinding,
  type CanonicalJson,
} from "@saliencegate/bridge-core";

import {
  OpenCodeBatchTransport,
  OpenCodeEventReducer,
  createOpenCodePlugin,
  loadOpenCodeBootstrap,
  spawnCaptureChunk,
  type CaptureChunkWrite,
  type SpawnChild,
  type SpawnFunction,
} from "../src/index.ts";

const VALID_BOOTSTRAP: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "opencode-plugin/v1",
  connection_id: `sg-${"1".repeat(48)}`,
  launcher_path: process.platform === "win32" ? "C:\\State\\capture-hook.cmd" : "/state/capture-hook",
  capability_digest: "2".repeat(64),
  bundle_digest: "3".repeat(64),
  receipt_mac: "4".repeat(64),
};

function toolEvent(sessionID: string, callID: string, status: "pending" | "completed"): object {
  return {
    type: "message.part.updated",
    properties: {
      part: {
        id: `part-${callID}`,
        sessionID,
        messageID: `message-${callID}`,
        type: "tool",
        callID,
        tool: "read",
        state:
          status === "pending"
            ? { status, input: { path: "synthetic.txt" }, raw: "ignored" }
            : {
                status,
                input: { path: "synthetic.txt" },
                output: "ignored",
                title: "ignored",
                metadata: {},
                time: { start: 1, end: 2 },
              },
      },
    },
  };
}

function document(write: CaptureChunkWrite): {
  session_id: string;
  events: Array<Record<string, unknown>>;
  chunk_index: number;
  chunk_count: number;
} {
  return JSON.parse(write.bytes.toString("utf8")) as {
    session_id: string;
    events: Array<Record<string, unknown>>;
    chunk_index: number;
    chunk_count: number;
  };
}

test("bootstrap loading requires canonical sidecar bytes and the exact sibling bundle digest", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "saliencegate-opencode-bootstrap-"));
  try {
    const bundlePath = path.join(directory, "saliencegate.js");
    const bootstrapPath = path.join(directory, "saliencegate.bootstrap.json");
    const bundle = Buffer.from("export default {};\n", "utf8");
    await writeFile(bundlePath, bundle, { mode: 0o600 });
    const expected = {
      ...VALID_BOOTSTRAP,
      bundle_digest: createHash("sha256").update(bundle).digest("hex"),
    };
    await writeFile(bootstrapPath, encodeCanonicalJson(expected), { mode: 0o600 });

    assert.deepEqual(
      await loadOpenCodeBootstrap(pathToFileURL(bootstrapPath)),
      expected,
    );

    await writeFile(bootstrapPath, Buffer.from(`${JSON.stringify(expected, null, 2)}\n`, "utf8"));
    await assert.rejects(loadOpenCodeBootstrap(pathToFileURL(bootstrapPath)));

    await writeFile(bootstrapPath, encodeCanonicalJson(expected));
    await writeFile(bundlePath, Buffer.from("export default { tampered: true };\n", "utf8"));
    await assert.rejects(loadOpenCodeBootstrap(pathToFileURL(bootstrapPath)));

    if (process.platform !== "win32") {
      await writeFile(bundlePath, bundle);
      await chmod(bootstrapPath, 0o644);
      await assert.rejects(loadOpenCodeBootstrap(pathToFileURL(bootstrapPath)));
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("the built asset exports the pinned host shape and constructs runtime hooks", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "saliencegate-opencode-built-shape-"));
  try {
    const source = new URL(
      "../../../src/saliencegate/integrations/assets/opencode-plugin.js",
      import.meta.url,
    );
    const bundle = await readFile(source);
    const bundlePath = path.join(directory, "saliencegate.js");
    const bootstrapPath = path.join(directory, "saliencegate.bootstrap.json");
    await writeFile(bundlePath, bundle, { mode: 0o600 });
    await writeFile(
      bootstrapPath,
      encodeCanonicalJson({
        ...VALID_BOOTSTRAP,
        bundle_digest: createHash("sha256").update(bundle).digest("hex"),
      }),
      { mode: 0o600 },
    );

    const imported = (await import(`${pathToFileURL(bundlePath).href}?shape=1`)) as {
      default?: {
        id?: unknown;
        server?: (input: unknown) => Promise<{
          event?: unknown;
          dispose?: unknown;
        }>;
      };
    };
    assert.equal(imported.default?.id, "saliencegate");
    assert.equal(typeof imported.default?.server, "function");
    const hooks = await imported.default!.server!({});
    assert.equal(typeof hooks.event, "function");
    assert.equal(typeof hooks.dispose, "function");
    await (hooks.dispose as () => Promise<void>)();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("runtime queues one session, flushes lifecycle boundaries, and never mixes sessions", async () => {
  const writes: CaptureChunkWrite[] = [];
  let releaseA!: () => void;
  const gate = new Promise<void>((resolve) => {
    releaseA = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      assert.equal(write.timeoutMS, 2_000);
      if (document(write).session_id === "session-a") await gate;
      writes.push(write);
      return true;
    },
    batchID: (() => {
      let value = 0;
      return () => (++value).toString(16).padStart(64, "0");
    })(),
  });
  const hooks = await plugin(
    new Proxy(
      {},
      {
        get() {
          throw new Error("OpenCode SDK/history input was accessed");
        },
      },
    ),
  );

  const first = hooks.event({ event: toolEvent("session-a", "call-a", "pending") });
  const flushA = hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-a" } },
  });
  const second = hooks.event({ event: toolEvent("session-b", "call-b", "completed") });
  const flushB = hooks.event({
    event: { type: "session.compacted", properties: { sessionID: "session-b" } },
  });
  await Promise.all([first, second, flushB]);

  assert.equal(writes.length, 1);
  assert.equal(document(writes[0]!).session_id, "session-b");
  releaseA();
  await flushA;

  assert.equal(writes.length, 2);
  for (const write of writes) {
    const observed = document(write);
    assert.ok(observed.events.length <= 999);
    assert.ok(observed.events.every((event) => event.session_id === observed.session_id));
  }

  await hooks.event({ event: toolEvent("session-c", "call-c", "pending") });
  await hooks.dispose();
  assert.ok(writes.some((write) => document(write).session_id === "session-c"));
});

test("error and deletion boundaries flush their matching reduced session buffers", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});

  await hooks.event({ event: toolEvent("session-error", "call-error", "pending") });
  await hooks.event({
    event: { type: "session.error", properties: { sessionID: "session-error" } },
  });
  await hooks.event({ event: toolEvent("session-delete", "call-delete", "pending") });
  await hooks.event({
    event: {
      type: "session.deleted",
      properties: { info: { id: "session-delete", title: "ignored" } },
    },
  });

  assert.deepEqual(
    writes.map((write) => document(write).session_id),
    ["session-error", "session-delete"],
  );
  assert.deepEqual(
    writes.map((write) => document(write).events.at(-1)?.kind),
    ["controller_failed", "session_finished"],
  );
});

test("finalized sessions do not emit duplicate terminal batches or reopen on late callbacks", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});
  const deleted = {
    id: "runtime-final-event",
    type: "session.deleted",
    properties: { info: { id: "runtime-finalized" } },
  };

  await hooks.event({ event: toolEvent("runtime-finalized", "call", "pending") });
  await hooks.event({ event: deleted });
  await hooks.event({ event: deleted });
  await hooks.event({ event: toolEvent("runtime-finalized", "late", "completed") });
  await hooks.event({
    event: {
      id: "runtime-final-event",
      type: "session.idle",
      properties: { sessionID: "runtime-finalized" },
    },
  });
  await hooks.dispose();

  assert.equal(writes.length, 1);
  assert.equal(document(writes[0]!).events.at(-1)?.kind, "session_finished");
});

test("an entirely failed batch propagates one content-free gap on the next flush", async () => {
  const writes: CaptureChunkWrite[] = [];
  let attempt = 0;
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      attempt += 1;
      return attempt !== 1;
    },
    batchID: () => createHash("sha256").update(String(attempt)).digest("hex"),
  });
  const hooks = await plugin({});

  await hooks.event({ event: toolEvent("session-gap", "call-one", "pending") });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-gap" } },
  });
  await hooks.event({ event: toolEvent("session-gap", "call-two", "pending") });
  await hooks.event({
    event: { type: "session.compacted", properties: { sessionID: "session-gap" } },
  });

  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events[0], {
    kind: "coverage_degraded",
    reason: "transport_gap",
    session_id: "session-gap",
  });
});

test("a later gap generation survives success of an earlier in-flight gap flush", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    announceStarted = resolve;
  });
  let batch = 0;
  const transport = new OpenCodeBatchTransport(VALID_BOOTSTRAP, {
    batchID: () => (++batch).toString(16).padStart(64, "0"),
    writeChunk: async (write) => {
      writes.push(write);
      if (writes.length === 1) {
        announceStarted();
        await gate;
      }
      return true;
    },
  });

  transport.markGap("malformed-\ud800");
  assert.deepEqual(transport.pendingSessionIDs(), []);
  transport.markGap("session-generation");
  const first = transport.flush("session-generation", [
    { kind: "turn_finished", session_id: "session-generation" },
  ]);
  await started;
  transport.markGap("session-generation");
  release();
  await first;

  assert.deepEqual(transport.pendingSessionIDs(), ["session-generation"]);
  await transport.flush("session-generation", []);
  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events, [
    {
      kind: "coverage_degraded",
      reason: "transport_gap",
      session_id: "session-generation",
    },
  ]);
  assert.deepEqual(transport.pendingSessionIDs(), []);
});

test("an empty duplicate lifecycle during a successful flush does not invent a gap", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    announceStarted = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      announceStarted();
      await gate;
      return true;
    },
  });
  const hooks = await plugin({});
  const boundary = {
    id: "in-flight-replay",
    type: "session.idle",
    properties: { sessionID: "session-no-false-gap" },
  };

  await hooks.event({ event: toolEvent("session-no-false-gap", "call", "pending") });
  const first = hooks.event({ event: boundary });
  await started;
  await hooks.event({ event: boundary });
  release();
  await first;

  assert.equal(writes.length, 1);
  assert.equal(
    document(writes[0]!).events.some((event) => event.reason === "transport_gap"),
    false,
  );
});

test("an empty duplicate lifecycle retries a gap created by its failing in-flight flush", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    announceStarted = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      if (writes.length === 1) {
        announceStarted();
        await gate;
        return false;
      }
      return true;
    },
  });
  const hooks = await plugin({});
  const boundary = {
    id: "in-flight-failure-replay",
    type: "session.idle",
    properties: { sessionID: "session-retry-gap" },
  };

  await hooks.event({ event: toolEvent("session-retry-gap", "call", "pending") });
  const first = hooks.event({ event: boundary });
  await started;
  await hooks.event({ event: boundary });
  release();
  await first;

  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events, [
    {
      kind: "coverage_degraded",
      reason: "transport_gap",
      session_id: "session-retry-gap",
    },
  ]);
});

test("an idempotent later lifecycle callback forces a pending gap-only flush", async () => {
  const writes: CaptureChunkWrite[] = [];
  let attempt = 0;
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      attempt += 1;
      return attempt > 1;
    },
  });
  const hooks = await plugin({});
  const boundary = {
    id: "idle-replay",
    type: "session.idle",
    properties: { sessionID: "session-gap-only" },
  };

  await hooks.event({ event: toolEvent("session-gap-only", "call-one", "pending") });
  await hooks.event({ event: boundary });
  await hooks.event({ event: boundary });

  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events, [
    {
      kind: "coverage_degraded",
      reason: "transport_gap",
      session_id: "session-gap-only",
    },
  ]);
});

test("dispose forces a pending gap-only flush after the session buffer is empty", async () => {
  const writes: CaptureChunkWrite[] = [];
  let attempt = 0;
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      attempt += 1;
      return attempt > 1;
    },
  });
  const hooks = await plugin({});

  await hooks.event({ event: toolEvent("session-dispose-gap", "call-one", "pending") });
  await hooks.event({
    event: {
      id: "dispose-idle",
      type: "session.idle",
      properties: { sessionID: "session-dispose-gap" },
    },
  });
  await hooks.dispose();

  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events, [
    {
      kind: "coverage_degraded",
      reason: "transport_gap",
      session_id: "session-dispose-gap",
    },
  ]);
});

test("malformed Unicode lifecycle and tool session IDs never reach transport state", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});

  await hooks.event({ event: toolEvent("tool-\ud800", "call", "pending") });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "lifecycle-\udc00" } },
  });
  await hooks.dispose();

  assert.deepEqual(writes, []);
});

test("oversize tool input is reduced before a lifecycle buffer retains it", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});
  const event = toolEvent("session-oversize", "call-oversize", "pending") as {
    properties: { part: { state: { input: unknown } } };
  };
  event.properties.part.state.input = { payload: "RAW-BUFFER-SENTINEL".repeat(10_000) };

  await hooks.event({ event });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-oversize" } },
  });

  assert.equal(writes.length, 1);
  assert.doesNotMatch(writes[0]!.bytes.toString("utf8"), /RAW-BUFFER-SENTINEL/);
  assert.equal(document(writes[0]!).events[0]?.kind, "oversize");
});

test("one in-flight session flush bounds later mail and reports a transport gap", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    announceStarted = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      if (writes.length === 1) {
        announceStarted();
        await gate;
      }
      return true;
    },
  });
  const hooks = await plugin({});

  await hooks.event({ event: toolEvent("session-mailbox", "call-first", "pending") });
  const firstFlush = hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-mailbox" } },
  });
  await started;
  const droppedFlushes: Promise<void>[] = [];
  for (let index = 0; index < 20; index += 1) {
    await hooks.event({
      event: toolEvent("session-mailbox", `call-later-${index}`, "pending"),
    });
    droppedFlushes.push(
      hooks.event({
        event: {
          type: "session.compacted",
          properties: { sessionID: "session-mailbox" },
        },
      }),
    );
  }
  release();
  await Promise.all([firstFlush, ...droppedFlushes]);
  await hooks.event({ event: toolEvent("session-mailbox", "call-final", "pending") });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-mailbox" } },
  });

  assert.equal(writes.length, 2);
  assert.deepEqual(document(writes[1]!).events[0], {
    kind: "coverage_degraded",
    reason: "transport_gap",
    session_id: "session-mailbox",
  });
});

test("an in-flight flush retains and immediately follows with a terminal control", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    announceStarted = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      if (writes.length === 1) {
        announceStarted();
        await gate;
      }
      return true;
    },
  });
  const hooks = await plugin({});
  const sessionID = "session-terminal-in-flight";

  await hooks.event({ event: toolEvent(sessionID, "call", "pending") });
  const firstFlush = hooks.event({
    event: { type: "session.idle", properties: { sessionID } },
  });
  await started;
  await hooks.event({
    event: {
      type: "session.deleted",
      properties: { info: { id: sessionID } },
    },
  });
  release();
  await firstFlush;

  assert.equal(writes.length, 2);
  assert.equal(document(writes[1]!).events[0]?.reason, "transport_gap");
  assert.equal(document(writes[1]!).events.at(-1)?.kind, "session_finished");
  await hooks.dispose();
  assert.equal(writes.length, 2);
});

test("global flush pressure drains every bounded terminal after a launcher succeeds", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  let announceFour!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const fourStarted = new Promise<void>((resolve) => {
    announceFour = resolve;
  });
  let active = 0;
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      active += 1;
      if (active === 4) announceFour();
      await gate;
      active -= 1;
      return true;
    },
  });
  const hooks = await plugin({});
  const targetSessions = Array.from(
    { length: 65 },
    (_, index) => `global-pressure-terminal-${index}`,
  );
  for (const [index, sessionID] of targetSessions.entries()) {
    await hooks.event({ event: toolEvent(sessionID, `target-call-${index}`, "pending") });
  }
  for (let index = 0; index < 64; index += 1) {
    await hooks.event({
      event: toolEvent(`global-pressure-${index}`, `call-${index}`, "pending"),
    });
  }
  const flushes = Array.from({ length: 64 }, (_, index) =>
    hooks.event({
      event: {
        type: "session.idle",
        properties: { sessionID: `global-pressure-${index}` },
      },
    }),
  );
  const deletions = targetSessions.map((sessionID) =>
    hooks.event({
      event: {
        type: "session.deleted",
        properties: { info: { id: sessionID } },
      },
    }),
  );

  await fourStarted;
  await Promise.all(deletions);
  release();
  await Promise.all(flushes);

  const terminalWrites = writes.filter(
    (write) => targetSessions.includes(document(write).session_id),
  );
  assert.equal(terminalWrites.length, targetSessions.length);
  assert.deepEqual(
    new Set(terminalWrites.map((write) => document(write).session_id)),
    new Set(targetSessions),
  );
  for (const write of terminalWrites) {
    assert.equal(document(write).events[0]?.reason, "transport_gap");
    assert.equal(document(write).events.at(-1)?.kind, "session_finished");
  }
  await hooks.dispose();
  assert.equal(
    writes.filter((write) => targetSessions.includes(document(write).session_id)).length,
    targetSessions.length,
  );
});

test("aggregate buffer pressure reserves a degraded terminal control", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});
  const sizingReducer = new OpenCodeEventReducer();
  const budget = 16 * 1024 * 1024;
  const targetSession = "aggregate-pressure-terminal";
  const targetEvent = toolEvent(targetSession, "target-call", "pending") as {
    properties: { part: { state: { input: unknown } } };
  };
  targetEvent.properties.part.state.input = { payload: "" };
  const targetProbe = new OpenCodeEventReducer().reduce(targetEvent)[0]!;
  const targetBaseBytes = encodeCanonicalJson(
    normalizeCaptureEvent(targetProbe, targetProbe.session_id),
  ).byteLength;
  targetEvent.properties.part.state.input = {
    payload: "t".repeat(64 * 1024 - targetBaseBytes),
  };
  const targetRecord = sizingReducer.reduce(targetEvent)[0]!;
  let retained = encodeCanonicalJson(
    normalizeCaptureEvent(targetRecord, targetRecord.session_id),
  ).byteLength;
  assert.equal(retained, 64 * 1024);
  const deletionEvent = {
    type: "session.deleted",
    properties: { info: { id: targetSession } },
  };
  const terminalRecord = sizingReducer.reduce(deletionEvent)[0]!;
  const terminalBytes = encodeCanonicalJson(
    normalizeCaptureEvent(terminalRecord, terminalRecord.session_id),
  ).byteLength;
  const desiredRetained = budget - terminalBytes + 1;
  const fillEvents: object[] = [];
  for (let index = 0; index < 255; index += 1) {
    const sessionID = `aggregate-pressure-${index}`;
    const event = toolEvent(sessionID, `call-${index}`, "pending") as {
      properties: { part: { state: { input: unknown } } };
    };
    event.properties.part.state.input = { payload: "" };
    const baseRecord = sizingReducer.reduce(event)[0]!;
    const baseBytes = encodeCanonicalJson(
      normalizeCaptureEvent(baseRecord, baseRecord.session_id),
    ).byteLength;
    const desiredBytes =
      index < 254 ? 64 * 1024 : desiredRetained - retained;
    const payloadBytes = desiredBytes - baseBytes;
    assert.ok(payloadBytes >= 0);
    event.properties.part.state.input = { payload: "x".repeat(payloadBytes) };
    const measuredReducer = new OpenCodeEventReducer();
    const measuredRecord = measuredReducer.reduce(event)[0]!;
    const measuredBytes = encodeCanonicalJson(
      normalizeCaptureEvent(measuredRecord, measuredRecord.session_id),
    ).byteLength;
    assert.equal(measuredBytes, desiredBytes);
    retained += measuredBytes;
    fillEvents.push(event);
  }
  assert.equal(retained, desiredRetained);

  await hooks.event({ event: targetEvent });
  for (const event of fillEvents) await hooks.event({ event });
  await hooks.event({ event: deletionEvent });

  const terminalWrite = writes.find(
    (write) => document(write).session_id === targetSession,
  );
  assert.ok(terminalWrite !== undefined);
  assert.equal(document(terminalWrite).events[0]?.reason, "transport_gap");
  assert.equal(document(terminalWrite).events.at(-1)?.kind, "session_finished");
  await hooks.dispose();
});

test("an attempted terminal launch is not retried under a fresh batch identity", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return false;
    },
  });
  const hooks = await plugin({});
  const sessionID = "attempted-terminal-failure";
  await hooks.event({ event: toolEvent(sessionID, "call", "pending") });
  await hooks.event({
    event: {
      type: "session.deleted",
      properties: { info: { id: sessionID } },
    },
  });

  assert.equal(writes.length, 1);
  assert.equal(document(writes[0]!).events.at(-1)?.kind, "session_finished");
  await hooks.dispose();

  assert.equal(
    writes.filter((write) =>
      document(write).events.some((event) => event.kind === "session_finished"),
    ).length,
    1,
  );
  assert.ok(
    writes
      .slice(1)
      .every((write) =>
        document(write).events.every((event) => event.kind !== "session_finished"),
      ),
  );
});

test("the lifetime budget leaves room for receiver, native-gap, and Busy-gap records", async () => {
  const reducer = new OpenCodeEventReducer();
  const records: CanonicalJson[] = [];
  for (let index = 0; index < 995; index += 1) {
    records.push(
      ...reducer
        .reduce(toolEvent("session-lifetime-gap", `call-${index}`, "pending"))
        .map((record) => record as unknown as CanonicalJson),
    );
  }
  records.push(
    ...reducer
      .reduce(toolEvent("session-lifetime-gap", "call-overflow", "pending"))
      .map((record) => record as unknown as CanonicalJson),
  );
  records.push(
    ...reducer
      .reduce({
        type: "session.deleted",
        properties: { info: { id: "session-lifetime-gap" } },
      })
      .map((record) => record as unknown as CanonicalJson),
  );
  assert.equal(records.length, 997);
  assert.equal((records.at(-2) as { kind?: unknown }).kind, "coverage_degraded");
  assert.equal((records.at(-1) as { kind?: unknown }).kind, "session_finished");

  const writes: CaptureChunkWrite[] = [];
  const transport = new OpenCodeBatchTransport(VALID_BOOTSTRAP, {
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  transport.markGap("session-lifetime-gap");
  await transport.flush("session-lifetime-gap", records);

  assert.equal(writes.length, 1);
  assert.equal(document(writes[0]!).events.length, 998);
  assert.equal(document(writes[0]!).events[0]?.reason, "transport_gap");
  assert.equal(document(writes[0]!).events.at(-1)?.kind, "session_finished");
  assert.equal(
    1 + document(writes[0]!).events.length + 1,
    1_000,
    "receiver start + Node events + a distinct StoreBusy gap must fit the session cap",
  );
});

test("capture launcher concurrency is globally bounded", async () => {
  let active = 0;
  let maximum = 0;
  let release!: () => void;
  let announceFour!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const fourActive = new Promise<void>((resolve) => {
    announceFour = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      if (active === 4) announceFour();
      await gate;
      active -= 1;
      return true;
    },
  });
  const hooks = await plugin({});
  const flushes: Promise<void>[] = [];
  for (let index = 0; index < 12; index += 1) {
    const sessionID = `session-concurrency-${index}`;
    await hooks.event({ event: toolEvent(sessionID, `call-${index}`, "pending") });
    flushes.push(
      hooks.event({
        event: { type: "session.idle", properties: { sessionID } },
      }),
    );
  }
  await fourActive;
  assert.equal(maximum, 4);
  release();
  await Promise.all(flushes);
  assert.equal(maximum, 4);
});

test("global launcher backpressure fails fast instead of serially blocking many sessions", async () => {
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 20));
      return true;
    },
  });
  const hooks = await plugin({});
  const flushes: Promise<void>[] = [];
  const startedAt = Date.now();
  for (let index = 0; index < 100; index += 1) {
    const sessionID = `session-backpressure-${index}`;
    await hooks.event({ event: toolEvent(sessionID, `call-${index}`, "pending") });
    flushes.push(
      hooks.event({
        event: { type: "session.idle", properties: { sessionID } },
      }),
    );
  }
  await Promise.all([...flushes, hooks.dispose()]);

  assert.ok(Date.now() - startedAt < 300);
});

test("distinct-session lifecycle floods keep pending queue state globally bounded", async () => {
  const writes: CaptureChunkWrite[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      await gate;
      return true;
    },
  });
  const hooks = await plugin({});
  const callbacks = Array.from({ length: 300 }, (_, index) =>
    hooks.event({
      event: {
        type: "session.idle",
        properties: { sessionID: `pending-flood-${index}` },
      },
    }),
  );

  await Promise.race([
    Promise.all(callbacks.slice(64)),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("bounded pending callbacks did not fail fast")), 200),
    ),
  ]);
  assert.equal(writes.length, 4);
  release();
  await Promise.all(callbacks);
  await hooks.dispose();
});

test("empty or over-cap lifecycle floods do not create transport work", async () => {
  const writes: CaptureChunkWrite[] = [];
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const hooks = await plugin({});
  const replay = {
    id: "flood-replay",
    type: "session.idle",
    properties: { sessionID: "session-flood" },
  };
  await hooks.event({ event: replay });
  assert.equal(writes.length, 1);

  await Promise.all(Array.from({ length: 500 }, () => hooks.event({ event: replay })));
  const overCap = "s".repeat(256 * 1024 + 1);
  await Promise.all(
    Array.from({ length: 100 }, () =>
      hooks.event({
        event: { type: "session.compacted", properties: { sessionID: overCap } },
      }),
    ),
  );
  await hooks.dispose();

  assert.equal(writes.length, 1);
});

test("spawn transport is shell-free, silent, bounded to two seconds, and fail-open", async () => {
  const observed: Array<{ timeout: number; shell: unknown; stdio: unknown; bytes: Buffer }> = [];
  const spawn: SpawnFunction = (file, args, options) => {
    assert.equal(file, VALID_BOOTSTRAP.launcher_path);
    assert.deepEqual(args, []);
    const child = new EventEmitter() as SpawnChild;
    const stdin = new PassThrough();
    const bytes: Buffer[] = [];
    stdin.on("data", (chunk: Buffer) => bytes.push(chunk));
    stdin.on("finish", () => {
      observed.push({
        timeout: 2_000,
        shell: options.shell,
        stdio: options.stdio,
        bytes: Buffer.concat(bytes),
      });
      queueMicrotask(() => child.emit("close", 0, null));
    });
    child.stdin = stdin;
    child.kill = () => true;
    return child;
  };

  assert.equal(
    await spawnCaptureChunk(
      {
        invocation: {
          file: VALID_BOOTSTRAP.launcher_path,
          arguments: [],
          options: {
            shell: false,
            windowsHide: true,
            env: {},
            stdio: ["pipe", "ignore", "ignore"],
          },
        },
        bytes: Buffer.from("{}", "utf8"),
      },
      spawn,
    ),
    true,
  );
  assert.equal(observed.length, 1);
  assert.equal(observed[0]!.shell, false);
  assert.deepEqual(observed[0]!.stdio, ["pipe", "ignore", "ignore"]);
  assert.deepEqual(observed[0]!.bytes, Buffer.from("{}", "utf8"));

  const throwing = (() => {
    throw new Error("spawn secret");
  }) as SpawnFunction;
  assert.equal(
    await spawnCaptureChunk(
      {
        invocation: {
          file: VALID_BOOTSTRAP.launcher_path,
          arguments: [],
          options: {
            shell: false,
            windowsHide: true,
            env: {},
            stdio: ["pipe", "ignore", "ignore"],
          },
        },
        bytes: Buffer.from("{}", "utf8"),
      },
      throwing,
    ),
    false,
  );
});

test("failed bootstrap and transport errors never escape plugin hooks", async () => {
  const disabled = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/missing/saliencegate.bootstrap.json"),
    loadBootstrap: async () => {
      throw new Error("bootstrap secret");
    },
  });
  const disabledHooks = await disabled({});
  await disabledHooks.event({ event: toolEvent("session-disabled", "call", "pending") });
  await disabledHooks.dispose();

  const transportFailure = createOpenCodePlugin({
    bootstrapURL: pathToFileURL("/unused/saliencegate.bootstrap.json"),
    loadBootstrap: async () => VALID_BOOTSTRAP,
    writeChunk: async () => {
      throw new Error("transport secret");
    },
  });
  const hooks = await transportFailure({});
  await hooks.event({ event: toolEvent("session-fail-open", "call", "pending") });
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "session-fail-open" } },
  });
  await hooks.dispose();
});

test("runtime tests never materialize an OpenCode SDK or history API", async () => {
  const source = await readFile(new URL("../src/plugin.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /session\.(?:get|messages)\s*\(/);
  assert.doesNotMatch(source, /@opencode-ai\/(?:sdk|plugin)/);
});
