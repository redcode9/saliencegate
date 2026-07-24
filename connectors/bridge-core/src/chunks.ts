import { Buffer } from "node:buffer";

import {
  canonicalizeJson,
  encodeCanonicalJson,
  isWellFormedUnicode,
} from "./canonical.ts";
import {
  BridgeContractError,
  MAX_CAPTURE_BATCH_BYTES,
  MAX_CAPTURE_BATCH_CHUNKS,
  MAX_CAPTURE_EVENTS_PER_CHUNK,
  MAX_CAPTURE_EVENT_BYTES,
  MAX_CAPTURE_SESSION_ID_BYTES,
  type BootstrapBinding,
  type CanonicalJson,
  type CaptureBatchDocument,
  type CaptureChunk,
  type CaptureChunkCoverage,
} from "./contracts.ts";

const SHA256 = /^[0-9a-f]{64}$/;
const CONNECTION_ID = /^sg-[0-9a-f]{48}$/;
const WINDOWS_ABSOLUTE = /^[A-Za-z]:[\\/]/;
const WINDOWS_UNC_ABSOLUTE = /^\\\\[^\\]+\\[^\\]+/;
const MAX_WORKSPACE_PATH_BYTES = 4_096;

function validWorkspacePath(value: string): boolean {
  return (
    value.length > 0 &&
    isWellFormedUnicode(value) &&
    Buffer.byteLength(value, "utf8") <= MAX_WORKSPACE_PATH_BYTES &&
    !value.includes("\0") &&
    (value.startsWith("/") ||
      WINDOWS_ABSOLUTE.test(value) ||
      WINDOWS_UNC_ABSOLUTE.test(value))
  );
}

function validateBootstrap(value: BootstrapBinding): BootstrapBinding {
  const keys = Object.keys(value).sort();
  const expected = [
    "bundle_digest",
    "capability_digest",
    "connection_id",
    "launcher_path",
    "profile",
    "receipt_mac",
    "schema_version",
  ];
  if (
    JSON.stringify(keys) !== JSON.stringify(expected) ||
    value.schema_version !== "integration-bootstrap/v1" ||
    !["opencode-plugin/v1", "pi-extension/v1"].includes(value.profile) ||
    !CONNECTION_ID.test(value.connection_id) ||
    !SHA256.test(value.capability_digest) ||
    !SHA256.test(value.bundle_digest) ||
    !SHA256.test(value.receipt_mac) ||
    typeof value.launcher_path !== "string" ||
    value.launcher_path.length === 0 ||
    value.launcher_path.length > 4_096 ||
    value.launcher_path.includes("\0") ||
    !(value.launcher_path.startsWith("/") || WINDOWS_ABSOLUTE.test(value.launcher_path))
  ) {
    throw new BridgeContractError();
  }
  return canonicalizeJson(value) as BootstrapBinding;
}

export function normalizeCaptureEvent(value: unknown, sessionID: string): CanonicalJson {
  try {
    const normalized = canonicalizeJson(value);
    if (encodeCanonicalJson(normalized).byteLength > MAX_CAPTURE_EVENT_BYTES) {
      throw new BridgeContractError();
    }
    return normalized;
  } catch {
    return { kind: "oversize", reason: "event_limit", session_id: sessionID };
  }
}

function oversizeEvent(sessionID: string): CanonicalJson {
  return { kind: "oversize", reason: "event_limit", session_id: sessionID };
}

function documentFor(input: {
  bootstrap: BootstrapBinding;
  batchID: string;
  sessionID: string;
  workspacePath?: string;
  chunkIndex: number;
  chunkCount: number;
  events: readonly CanonicalJson[];
}): CaptureBatchDocument {
  return {
    schema_version: "capture-batch/v1",
    bootstrap: input.bootstrap,
    batch_id: input.batchID,
    session_id: input.sessionID,
    ...(input.workspacePath === undefined
      ? {}
      : { workspace_path: input.workspacePath }),
    chunk_index: input.chunkIndex,
    chunk_count: input.chunkCount,
    events: input.events,
  };
}

function encodedSize(input: {
  bootstrap: BootstrapBinding;
  batchID: string;
  sessionID: string;
  workspacePath?: string;
  chunkIndex: number;
  events: readonly CanonicalJson[];
}): number {
  try {
    return encodeCanonicalJson(
      documentFor({
        ...input,
        chunkCount: MAX_CAPTURE_BATCH_CHUNKS,
      }),
    ).byteLength;
  } catch (error) {
    if (error instanceof BridgeContractError) return MAX_CAPTURE_BATCH_BYTES + 1;
    throw error;
  }
}

export function buildCaptureChunks(input: {
  bootstrap: BootstrapBinding;
  batchID: string;
  sessionID: string;
  workspacePath?: string;
  events: readonly unknown[];
}): CaptureChunk[] {
  try {
    const bootstrap = validateBootstrap(input.bootstrap);
    if (
      typeof input.batchID !== "string" ||
      !SHA256.test(input.batchID) ||
      typeof input.sessionID !== "string" ||
      input.sessionID.length === 0 ||
      Buffer.byteLength(input.sessionID, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES ||
      (input.workspacePath !== undefined &&
        (typeof input.workspacePath !== "string" ||
          !validWorkspacePath(input.workspacePath))) ||
      !Array.isArray(input.events) ||
      input.events.length > 10_000
    ) {
      throw new BridgeContractError();
    }
    canonicalizeJson(input.sessionID);
    const events = input.events.map((event) => {
      const normalized = normalizeCaptureEvent(event, input.sessionID);
      return encodedSize({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        ...(input.workspacePath === undefined
          ? {}
          : { workspacePath: input.workspacePath }),
        chunkIndex: 0,
        events: [normalized],
      }) <= MAX_CAPTURE_BATCH_BYTES
        ? normalized
        : oversizeEvent(input.sessionID);
    });
    const groups: CanonicalJson[][] = [];
    let current: CanonicalJson[] = [];
    for (const event of events) {
      const candidate = [...current, event];
      if (
        current.length > 0 &&
        (current.length >= MAX_CAPTURE_EVENTS_PER_CHUNK ||
          encodedSize({
            bootstrap,
            batchID: input.batchID,
            sessionID: input.sessionID,
            ...(input.workspacePath === undefined
              ? {}
              : { workspacePath: input.workspacePath }),
            chunkIndex: groups.length,
            events: candidate,
          }) > MAX_CAPTURE_BATCH_BYTES)
      ) {
        groups.push(current);
        current = [event];
      } else {
        current = candidate;
      }
    }
    if (current.length > 0 || groups.length === 0) groups.push(current);
    if (groups.length > MAX_CAPTURE_BATCH_CHUNKS) throw new BridgeContractError();

    return groups.map((group, index) => {
      const document = documentFor({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        ...(input.workspacePath === undefined
          ? {}
          : { workspacePath: input.workspacePath }),
        chunkIndex: index,
        chunkCount: groups.length,
        events: group,
      });
      const bytes = encodeCanonicalJson(document);
      if (bytes.byteLength > MAX_CAPTURE_BATCH_BYTES) throw new BridgeContractError();
      return { document, bytes };
    });
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

export function inspectChunkCoverage(
  chunks: readonly CaptureBatchDocument[],
): CaptureChunkCoverage {
  try {
    if (!Array.isArray(chunks) || chunks.length === 0) throw new BridgeContractError();
    const first = chunks[0]!;
    if (
      !Number.isInteger(first.chunk_count) ||
      first.chunk_count < 1 ||
      first.chunk_count > MAX_CAPTURE_BATCH_CHUNKS
    ) {
      throw new BridgeContractError();
    }
    const observed = new Map<number, Buffer>();
    for (const chunk of chunks) {
      if (
        chunk.schema_version !== "capture-batch/v1" ||
        chunk.batch_id !== first.batch_id ||
        chunk.session_id !== first.session_id ||
        chunk.workspace_path !== first.workspace_path ||
        chunk.chunk_count !== first.chunk_count ||
        !Number.isInteger(chunk.chunk_index) ||
        chunk.chunk_index < 0 ||
        chunk.chunk_index >= first.chunk_count
      ) {
        throw new BridgeContractError();
      }
      const encoded = encodeCanonicalJson(chunk);
      const prior = observed.get(chunk.chunk_index);
      if (prior !== undefined && !prior.equals(encoded)) throw new BridgeContractError();
      observed.set(chunk.chunk_index, encoded);
    }
    const missingIndexes = Array.from({ length: first.chunk_count }, (_, index) => index).filter(
      (index) => !observed.has(index),
    );
    return { complete: missingIndexes.length === 0, missingIndexes };
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}
