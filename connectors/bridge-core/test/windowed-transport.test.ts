import assert from "node:assert/strict";
import { test } from "node:test";

import {
  WindowedBatchTransport,
  type BootstrapBinding,
  type CanonicalJson,
  type WindowCoordinates,
  type WindowedCaptureChunkWrite,
} from "../src/index.ts";

const bootstrap: BootstrapBinding = {
  schema_version: "integration-bootstrap/v1",
  profile: "pi-extension/v1",
  connection_id: `sg-${"1".repeat(48)}`,
  launcher_path:
    process.platform === "win32"
      ? "C:\\synthetic\\state\\pi\\capture-hook.cmd"
      : "/synthetic/state/pi/capture-hook",
  capability_digest: "2".repeat(64),
  bundle_digest: "3".repeat(64),
  receipt_mac: "4".repeat(64),
};

const PROVIDER_CREDENTIAL_KEYS = new Set([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID",
]);

function coordinates(index: number): WindowCoordinates {
  return {
    sessionID: `pi-session-${index}`,
    windowDiscriminator: index.toString(16).padStart(64, "0"),
  };
}

function record(window: WindowCoordinates, eventID: string): CanonicalJson {
  return {
    kind: "turn_finished",
    session_id: window.sessionID,
    window_discriminator: window.windowDiscriminator,
    event_id: eventID,
  };
}

function decodedEvents(write: WindowedCaptureChunkWrite): Record<string, unknown>[] {
  const document = JSON.parse(write.bytes.toString("utf8")) as {
    events: Record<string, unknown>[];
  };
  return document.events;
}

test("windowed transport filters provider credentials without reading hostile getters", async () => {
  const target: NodeJS.ProcessEnv = {
    SAFE_MARKER: "preserved",
    ...(process.platform === "win32" ? { SystemRoot: "C:\\Windows" } : {}),
  };
  for (const key of PROVIDER_CREDENTIAL_KEYS) target[key] = `${key}-must-not-be-read`;
  const reads: string[] = [];
  const environment = new Proxy(target, {
    get(value, property, receiver) {
      if (typeof property === "string") {
        if (PROVIDER_CREDENTIAL_KEYS.has(property.toUpperCase())) {
          throw new Error(`credential getter was read: ${property}`);
        }
        reads.push(property);
      }
      return Reflect.get(value, property, receiver) as unknown;
    },
  });
  const writes: WindowedCaptureChunkWrite[] = [];
  const transport = new WindowedBatchTransport(bootstrap, {
    environment,
    batchID: () => "b".repeat(64),
    writeChunk: async (write) => {
      writes.push(write);
      return true;
    },
  });
  const window = coordinates(901);

  assert.equal(await transport.flush(window, [record(window, "1")]), "delivered");
  assert.equal(writes.length, 1);
  assert.equal(writes[0]!.invocation.options.env.SAFE_MARKER, "preserved");
  assert.ok(reads.includes("SAFE_MARKER"));
  assert.equal(
    Object.keys(writes[0]!.invocation.options.env).some((key) =>
      PROVIDER_CREDENTIAL_KEYS.has(key.toUpperCase()),
    ),
    false,
  );
});

test("pending-gap capacity overflow permanently poisons all later valid windows", async () => {
  let deliver = false;
  const delivered: WindowedCaptureChunkWrite[] = [];
  const transport = new WindowedBatchTransport(bootstrap, {
    batchID: () => "a".repeat(64),
    writeChunk: async (write) => {
      if (deliver) delivered.push(write);
      return deliver;
    },
  });

  for (let index = 0; index < 257; index += 1) {
    const window = coordinates(index);
    assert.equal(
      await transport.flush(window, [record(window, "1")]),
      "attempted_failure",
    );
  }

  const overflowWindow = coordinates(256);
  const unrelatedWindow = coordinates(900);
  assert.equal(transport.hasPendingGap(overflowWindow), true);
  assert.equal(transport.hasPendingGap(unrelatedWindow), true);

  deliver = true;
  assert.equal(
    await transport.flush(overflowWindow, [record(overflowWindow, "2")]),
    "delivered",
  );
  assert.equal(
    await transport.flush(unrelatedWindow, [record(unrelatedWindow, "1")]),
    "delivered",
  );

  assert.equal(delivered.length, 2);
  for (const write of delivered) {
    const events = decodedEvents(write);
    assert.equal(events[0]?.kind, "coverage_degraded");
    assert.equal(events[0]?.reason, "transport_gap");
  }
  assert.equal(transport.hasPendingGap(overflowWindow), true);
  assert.equal(transport.hasPendingGap(unrelatedWindow), true);
});
