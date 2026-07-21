import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BridgeContractError,
  MAX_CAPTURE_BATCH_BYTES,
  MAX_CAPTURE_SESSION_ID_BYTES,
  buildCaptureChunks,
  buildWindowedCaptureChunks,
  canonicalizeJson,
  encodeCanonicalJson,
  inspectChunkCoverage,
  inspectWindowedChunkCoverage,
  type BootstrapBinding,
} from "../src/index.ts";

const bootstrap: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "opencode-plugin/v1",
  connection_id: `sg-${"1".repeat(48)}`,
  launcher_path: "/synthetic/state/opencode/capture-hook",
  capability_digest: "2".repeat(64),
  bundle_digest: "3".repeat(64),
  receipt_mac: "4".repeat(64),
};

test("canonical JSON is recursively sorted without changing supported values", () => {
  const value = Object.assign(Object.create(null) as Record<string, unknown>, {
    z: [true, null, 1.5],
    a: { beta: "two", alpha: "one" },
  });

  assert.deepEqual(canonicalizeJson(value), {
    a: { alpha: "one", beta: "two" },
    z: [true, null, 1.5],
  });
  assert.equal(
    encodeCanonicalJson(value).toString("utf8"),
    '{"a":{"alpha":"one","beta":"two"},"z":[true,null,1.5]}',
  );
});

test("windowed capture chunks bind every document and oversize control to one window", () => {
  const piBootstrap: BootstrapBinding = {
    ...bootstrap,
    profile: "pi-extension/v1",
    launcher_path: "/synthetic/state/pi/capture-hook",
  };
  const windowDiscriminator = "5".repeat(64);
  const chunks = buildWindowedCaptureChunks({
    bootstrap: piBootstrap,
    batchID: "6".repeat(64),
    sessionID: "019c0eaf-7b31-7000-8000-000000000001",
    windowDiscriminator,
    events: [
      {
        kind: "tool_started",
        session_id: "019c0eaf-7b31-7000-8000-000000000001",
        window_discriminator: windowDiscriminator,
        event_id: "1",
        call_id: "call-one",
        tool: "read",
        identity_authority: "coarse",
        ignored: "RAW-WINDOW-SENTINEL".repeat(100_000),
      },
    ],
  });

  assert.equal(chunks.length, 1);
  assert.equal(chunks[0]!.document.window_discriminator, windowDiscriminator);
  assert.ok(chunks[0]!.bytes.byteLength <= MAX_CAPTURE_BATCH_BYTES);
  assert.doesNotMatch(chunks[0]!.bytes.toString("utf8"), /RAW-WINDOW-SENTINEL/);
  assert.deepEqual(chunks[0]!.document.events, [
    {
      kind: "oversize",
      reason: "event_limit",
      session_id: "019c0eaf-7b31-7000-8000-000000000001",
      window_discriminator: windowDiscriminator,
    },
  ]);
});

test("windowed chunk coverage detects gaps and rejects mixed discriminators", () => {
  const piBootstrap: BootstrapBinding = {
    ...bootstrap,
    profile: "pi-extension/v1",
  };
  const chunks = buildWindowedCaptureChunks({
    bootstrap: piBootstrap,
    batchID: "7".repeat(64),
    sessionID: "pi-session",
    windowDiscriminator: "8".repeat(64),
    events: Array.from({ length: 90 }, (_, index) => ({
      kind: "coverage_boundary",
      session_id: "pi-session",
      window_discriminator: "8".repeat(64),
      event_id: String(index + 1),
      padding: "z".repeat(48_000),
    })),
  }).map((item) => item.document);
  assert.ok(chunks.length >= 3);
  const withoutMiddle = chunks.filter(
    (item) => item.chunk_index !== Math.floor(chunks.length / 2),
  );
  assert.deepEqual(inspectWindowedChunkCoverage(withoutMiddle), {
    complete: false,
    missingIndexes: [Math.floor(chunks.length / 2)],
  });

  const forged = {
    ...chunks[0]!,
    window_discriminator: "9".repeat(64),
  };
  assert.throws(
    () => inspectWindowedChunkCoverage([chunks[0]!, forged]),
    BridgeContractError,
  );
  assert.throws(
    () =>
      buildWindowedCaptureChunks({
        bootstrap: piBootstrap,
        batchID: "7".repeat(64),
        sessionID: "pi-session",
        windowDiscriminator: "UPPER".repeat(13),
        events: [],
      }),
    BridgeContractError,
  );
});

test("canonicalization rejects cycles, accessors, unsupported values, and non-finite numbers", () => {
  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  const accessor = Object.defineProperty({}, "secret", {
    enumerable: true,
    get: () => "must-not-run",
  });

  for (const value of [cycle, accessor, 1n, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => canonicalizeJson(value), BridgeContractError);
  }
});

test("canonicalization preserves prototype-named keys without collision or mutation", () => {
  const value = JSON.parse('{"safe":1,"__proto__":{"polluted":true}}') as object;
  const encoded = encodeCanonicalJson(value).toString("utf8");

  assert.equal(encoded, '{"__proto__":{"polluted":true},"safe":1}');
  assert.notEqual(encoded, encodeCanonicalJson({ safe: 1 }).toString("utf8"));
  assert.equal((Object.prototype as { polluted?: boolean }).polluted, undefined);
});

test("canonicalization rejects lone UTF-16 surrogates in keys and values", () => {
  for (const value of ["\ud800", "\udc00", { ["bad-\ud800"]: true }]) {
    assert.throws(() => canonicalizeJson(value), BridgeContractError);
  }
  assert.equal(encodeCanonicalJson("paired-\ud83d\ude00").toString("utf8"), '"paired-😀"');
});

test("canonicalization enforces depth, aggregate item, and string-byte bounds", () => {
  let deep: unknown = null;
  for (let index = 0; index < 33; index += 1) deep = [deep];

  assert.throws(() => canonicalizeJson(deep), BridgeContractError);
  assert.throws(
    () => canonicalizeJson(Array.from({ length: 10_001 }, () => null)),
    BridgeContractError,
  );
  assert.throws(() => canonicalizeJson("x".repeat(1_048_577)), BridgeContractError);
});

test("capture chunks include complete bounded sequencing and deterministic bytes", () => {
  const events = Array.from({ length: 80 }, (_, index) => ({
    kind: "tool_started",
    event_id: `event-${index}`,
    call_id: `call-${index}`,
    tool: "read",
    input: { payload: "x".repeat(48_000) },
  }));

  const first = buildCaptureChunks({
    bootstrap,
    batchID: "a".repeat(64),
    sessionID: "session-one",
    events,
  });
  const second = buildCaptureChunks({
    bootstrap,
    batchID: "a".repeat(64),
    sessionID: "session-one",
    events,
  });

  assert.ok(first.length > 1);
  assert.deepEqual(
    first.map((item) => item.bytes),
    second.map((item) => item.bytes),
  );
  assert.deepEqual(
    first.map((item) => item.document.chunk_index),
    Array.from({ length: first.length }, (_, index) => index),
  );
  assert.ok(first.every((item) => item.document.chunk_count === first.length));
  assert.ok(first.every((item) => item.bytes.byteLength <= MAX_CAPTURE_BATCH_BYTES));
  assert.deepEqual(inspectChunkCoverage(first.map((item) => item.document)), {
    complete: true,
    missingIndexes: [],
  });
});

test("coverage inspection detects a missing first, middle, or tail chunk", () => {
  const chunks = buildCaptureChunks({
    bootstrap,
    batchID: "b".repeat(64),
    sessionID: "session-two",
    events: Array.from({ length: 90 }, (_, index) => ({
      kind: "tool_started",
      call_id: `call-${index}`,
      tool: "shell",
      input: { payload: "y".repeat(48_000) },
    })),
  }).map((item) => item.document);
  assert.ok(chunks.length >= 3);

  for (const missing of [0, Math.floor(chunks.length / 2), chunks.length - 1]) {
    const observed = chunks.filter((item) => item.chunk_index !== missing);
    assert.deepEqual(inspectChunkCoverage(observed), {
      complete: false,
      missingIndexes: [missing],
    });
  }
});

test("one reduced event that cannot fit is replaced by a content-free degraded control", () => {
  const chunks = buildCaptureChunks({
    bootstrap,
    batchID: "c".repeat(64),
    sessionID: "session-three",
    events: [
      {
        kind: "tool_started",
        call_id: "sensitive-call",
        tool: "write",
        input: { payload: "RAW-SENTINEL".repeat(100_000) },
      },
    ],
  });

  assert.equal(chunks.length, 1);
  assert.ok(chunks[0]!.bytes.byteLength <= MAX_CAPTURE_BATCH_BYTES);
  assert.doesNotMatch(chunks[0]!.bytes.toString("utf8"), /RAW-SENTINEL/);
  assert.deepEqual(chunks[0]!.document.events, [
    { kind: "oversize", reason: "event_limit", session_id: "session-three" },
  ]);
});

test("chunks reserve one intake slot for the provider session start", () => {
  const chunks = buildCaptureChunks({
    bootstrap,
    batchID: "d".repeat(64),
    sessionID: "session-four",
    events: Array.from({ length: 2_050 }, (_, index) => ({
      kind: "tool_started",
      session_id: "session-four",
      call_id: `call-${index}`,
      tool: "read",
    })),
  });

  assert.deepEqual(
    chunks.map((item) => item.document.events.length),
    [999, 999, 52],
  );
  assert.ok(chunks.every((item) => item.document.events.length <= 999));
});

test("chunk normalization reserves aggregate envelope item and string budgets", () => {
  const nearItemLimit = {
    kind: "tool_started",
    session_id: "session-envelope",
    input: Array.from({ length: 9_990 }, () => 0),
  };
  const chunks = buildCaptureChunks({
    bootstrap,
    batchID: "e".repeat(64),
    sessionID: "session-envelope",
    events: [nearItemLimit],
  });

  assert.deepEqual(chunks[0]!.document.events, [
    { kind: "oversize", reason: "event_limit", session_id: "session-envelope" },
  ]);
  assert.equal(
    buildCaptureChunks({
      bootstrap,
      batchID: "f".repeat(64),
      sessionID: "s".repeat(MAX_CAPTURE_SESSION_ID_BYTES),
      events: [],
    })[0]!.document.session_id.length,
    MAX_CAPTURE_SESSION_ID_BYTES,
  );
  assert.throws(
    () =>
      buildCaptureChunks({
        bootstrap,
        batchID: "0".repeat(64),
        sessionID: "s".repeat(MAX_CAPTURE_SESSION_ID_BYTES + 1),
        events: [],
      }),
    BridgeContractError,
  );
});
