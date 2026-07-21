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
  launcher_path: "/synthetic/state/pi/capture-hook",
  capability_digest: "2".repeat(64),
  bundle_digest: "3".repeat(64),
  receipt_mac: "4".repeat(64),
};

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
