import { Buffer } from "node:buffer";

import {
  MAX_CAPTURE_EVENT_BYTES,
  MAX_CAPTURE_SESSION_ID_BYTES,
  SerialSessionQueue,
  encodeCanonicalJson,
  isWellFormedUnicode,
  normalizeCaptureEvent,
  type BootstrapBinding,
  type CanonicalJson,
} from "@saliencegate/bridge-core";

import { loadOpenCodeBootstrap } from "./bootstrap.ts";
import { OpenCodeEventReducer, type ReducedOpenCodeRecord } from "./reducer.ts";
import {
  OpenCodeBatchTransport,
  type CaptureChunkWriter,
} from "./transport.ts";
import type { OpenCodeEvent, OpenCodePlugin } from "./upstream-types.ts";

const MAX_SESSION_BUFFER_BYTES = 512 * 1024;
const MAX_TOTAL_RETAINED_BYTES = 16 * 1024 * 1024;
const MAX_PENDING_FLUSHES_PER_SESSION = 1;
const MAX_PENDING_FLUSH_SESSIONS = 64;
const MAX_POST_PENDING_GAP_CHECKS = 256;
const MAX_DEFERRED_TERMINAL_SESSIONS = 256;
const MAX_TERMINAL_RESERVE_BYTES =
  MAX_DEFERRED_TERMINAL_SESSIONS * MAX_CAPTURE_EVENT_BYTES;
const MAX_DISPOSE_FLUSH_PASSES = 4;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactText(value: unknown): string | undefined {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    !isWellFormedUnicode(value) ||
    Buffer.byteLength(value, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES
  ) {
    return undefined;
  }
  return value;
}

function flushTargets(value: unknown): Set<string> {
  const result = new Set<string>();
  if (!isRecord(value) || typeof value.type !== "string") return result;
  const properties = value.properties;
  if (!isRecord(properties)) return result;
  if (value.type === "session.error" && properties.sessionID === undefined) return result;
  if (value.type === "session.deleted") {
    const info = properties.info;
    const sessionID = isRecord(info) ? exactText(info.id) : undefined;
    if (sessionID !== undefined) result.add(sessionID);
    return result;
  }
  if (
    value.type === "session.idle" ||
    value.type === "session.error" ||
    value.type === "session.compacted"
  ) {
    const sessionID = exactText(properties.sessionID);
    if (sessionID !== undefined) result.add(sessionID);
  }
  return result;
}

function asCanonicalRecord(record: ReducedOpenCodeRecord): CanonicalJson {
  return normalizeCaptureEvent(record, record.session_id);
}

function terminalControl(records: readonly CanonicalJson[]): CanonicalJson | undefined {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (isRecord(record) && record.kind === "session_finished") return record;
  }
  return undefined;
}

type SessionBuffer = {
  records: CanonicalJson[];
  bytes: number;
};

class OpenCodePluginRuntime {
  readonly #reducer = new OpenCodeEventReducer();
  readonly #queue = new SerialSessionQueue();
  readonly #transport: OpenCodeBatchTransport;
  readonly #buffers = new Map<string, SessionBuffer>();
  readonly #pendingFlushes = new Map<string, number>();
  readonly #postPendingGapChecks = new Set<string>();
  readonly #deferredTerminalSessions = new Set<string>();
  #retainedBytes = 0;
  #disposed = false;

  constructor(
    bootstrap: BootstrapBinding,
    options: {
      platform?: NodeJS.Platform;
      environment?: NodeJS.ProcessEnv;
      writeChunk?: CaptureChunkWriter;
      batchID?: () => string;
    },
  ) {
    this.#transport = new OpenCodeBatchTransport(bootstrap, options);
  }

  #knownSessions(): string[] {
    return [
      ...new Set([
        ...this.#buffers.keys(),
        ...this.#pendingFlushes.keys(),
        ...this.#postPendingGapChecks,
        ...this.#deferredTerminalSessions,
        ...this.#transport.pendingSessionIDs(),
      ]),
    ];
  }

  #retainOnlyTerminal(
    sessionID: string,
    buffer: SessionBuffer,
  ): CanonicalJson | undefined {
    const terminal = terminalControl(buffer.records);
    this.#buffers.delete(sessionID);
    this.#retainedBytes -= buffer.bytes;
    if (terminal === undefined) return undefined;
    const bytes = encodeCanonicalJson(terminal).byteLength;
    this.#buffers.set(sessionID, { records: [terminal], bytes });
    this.#retainedBytes += bytes;
    return terminal;
  }

  #restoreTerminal(sessionID: string, terminal: CanonicalJson): boolean {
    let buffer = this.#buffers.get(sessionID);
    if (buffer !== undefined && terminalControl(buffer.records) !== undefined) return true;
    const bytes = encodeCanonicalJson(terminal).byteLength;
    if (
      buffer !== undefined &&
      buffer.records.length > 0 &&
      buffer.bytes + bytes > MAX_SESSION_BUFFER_BYTES
    ) {
      this.#buffers.delete(sessionID);
      this.#retainedBytes -= buffer.bytes;
      this.#transport.markGap(sessionID);
      buffer = undefined;
    }
    if (
      this.#retainedBytes + bytes >
      MAX_TOTAL_RETAINED_BYTES + MAX_TERMINAL_RESERVE_BYTES
    ) {
      this.#transport.markGap(sessionID);
      return false;
    }
    if (buffer === undefined) {
      buffer = { records: [], bytes: 0 };
      this.#buffers.set(sessionID, buffer);
    }
    buffer.records.push(terminal);
    buffer.bytes += bytes;
    this.#retainedBytes += bytes;
    return true;
  }

  #deferTerminal(sessionID: string): void {
    if (
      this.#deferredTerminalSessions.has(sessionID) ||
      this.#deferredTerminalSessions.size < MAX_DEFERRED_TERMINAL_SESSIONS
    ) {
      this.#deferredTerminalSessions.add(sessionID);
    }
  }

  async #scheduleOneDeferredTerminal(): Promise<void> {
    for (const sessionID of this.#deferredTerminalSessions) {
      if (this.#pendingFlushes.has(sessionID)) continue;
      this.#deferredTerminalSessions.delete(sessionID);
      await this.#scheduleFlush(sessionID, true);
      return;
    }
    for (const [sessionID, buffer] of this.#buffers) {
      if (
        this.#pendingFlushes.has(sessionID) ||
        terminalControl(buffer.records) === undefined
      ) {
        continue;
      }
      await this.#scheduleFlush(sessionID, true);
      return;
    }
  }

  #scheduleFlush(sessionID: string, force = false): Promise<void> {
    const buffer = this.#buffers.get(sessionID);
    const records = buffer?.records ?? [];
    const bufferBytes = buffer?.bytes ?? 0;
    const hasRecords = records.length > 0;
    if (!hasRecords && !force) return Promise.resolve();
    if (
      !this.#pendingFlushes.has(sessionID) &&
      this.#pendingFlushes.size >= MAX_PENDING_FLUSH_SESSIONS
    ) {
      if (hasRecords) {
        const terminal = this.#retainOnlyTerminal(sessionID, buffer!);
        this.#transport.markGap(sessionID);
        if (terminal !== undefined) this.#deferTerminal(sessionID);
      }
      return Promise.resolve();
    }
    const pending = this.#pendingFlushes.get(sessionID) ?? 0;
    if (pending >= MAX_PENDING_FLUSHES_PER_SESSION) {
      let terminal: CanonicalJson | undefined;
      if (hasRecords) {
        terminal = this.#retainOnlyTerminal(sessionID, buffer!);
        this.#transport.markGap(sessionID);
      }
      if (!hasRecords || terminal !== undefined) {
        if (
          this.#postPendingGapChecks.has(sessionID) ||
          this.#postPendingGapChecks.size < MAX_POST_PENDING_GAP_CHECKS
        ) {
          this.#postPendingGapChecks.add(sessionID);
        } else if (terminal !== undefined) {
          this.#deferTerminal(sessionID);
        }
      }
      return Promise.resolve();
    }
    const terminal = terminalControl(records);
    if (hasRecords) this.#buffers.delete(sessionID);
    this.#pendingFlushes.set(sessionID, pending + 1);
    let flushResult: "attempted_failure" | "delivered" | "not_started" =
      "attempted_failure";
    return this.#queue
      .run(sessionID, async () => {
        flushResult = await this.#transport.flush(sessionID, records);
      })
      .finally(async () => {
        if (hasRecords) {
          this.#retainedBytes -= bufferBytes;
          if (terminal !== undefined && flushResult === "not_started") {
            if (this.#restoreTerminal(sessionID, terminal)) {
              this.#deferTerminal(sessionID);
            }
          }
        }
        const current = this.#pendingFlushes.get(sessionID);
        if (current === undefined || current <= 1) {
          this.#pendingFlushes.delete(sessionID);
        } else {
          this.#pendingFlushes.set(sessionID, current - 1);
        }
        if (this.#postPendingGapChecks.delete(sessionID)) {
          await this.#scheduleFlush(sessionID, true);
        } else if (flushResult === "delivered") {
          await this.#scheduleOneDeferredTerminal();
        }
      });
  }

  #appendRecords(
    sessionID: string,
    records: readonly CanonicalJson[],
    scheduled: Promise<void>[],
  ): void {
    for (const record of records) {
      const bytes = encodeCanonicalJson(record).byteLength;
      const isTerminal = isRecord(record) && record.kind === "session_finished";
      let buffer = this.#buffers.get(sessionID);
      if (
        buffer !== undefined &&
        buffer.records.length > 0 &&
        buffer.bytes + bytes > MAX_SESSION_BUFFER_BYTES
      ) {
        scheduled.push(this.#scheduleFlush(sessionID));
        buffer = undefined;
      }
      if (
        this.#retainedBytes + bytes >
        (isTerminal
          ? MAX_TOTAL_RETAINED_BYTES + MAX_TERMINAL_RESERVE_BYTES
          : MAX_TOTAL_RETAINED_BYTES)
      ) {
        this.#transport.markGap(sessionID);
        continue;
      }
      if (isTerminal && this.#retainedBytes + bytes > MAX_TOTAL_RETAINED_BYTES) {
        this.#transport.markGap(sessionID);
      }
      if (buffer === undefined) {
        buffer = { records: [], bytes: 0 };
        this.#buffers.set(sessionID, buffer);
      }
      buffer.records.push(record);
      buffer.bytes += bytes;
      this.#retainedBytes += bytes;
    }
  }

  async event(value: unknown): Promise<void> {
    if (this.#disposed) return;
    try {
      const records = this.#reducer.reduce(value);
      const grouped = new Map<string, CanonicalJson[]>();
      const terminalSessions = new Set<string>();
      for (const record of records) {
        if (record.kind === "session_finished") {
          terminalSessions.add(record.session_id);
        } else if (this.#reducer.isFinalized(record.session_id)) {
          continue;
        }
        const list = grouped.get(record.session_id) ?? [];
        list.push(asCanonicalRecord(record));
        grouped.set(record.session_id, list);
      }
      const targets = flushTargets(value);
      for (const sessionID of targets) {
        if (this.#reducer.isFinalized(sessionID) && !terminalSessions.has(sessionID)) {
          targets.delete(sessionID);
        }
      }
      const sessions = new Set([...grouped.keys(), ...targets]);
      const scheduled: Promise<void>[] = [];
      for (const sessionID of sessions) {
        const reduced = grouped.get(sessionID) ?? [];
        this.#appendRecords(sessionID, reduced, scheduled);
        const buffered = this.#buffers.get(sessionID);
        if (
          targets.has(sessionID) &&
          (reduced.length > 0 ||
            (buffered !== undefined && buffered.records.length > 0) ||
            this.#pendingFlushes.has(sessionID) ||
            this.#transport.hasPendingGap(sessionID))
        ) {
          scheduled.push(this.#scheduleFlush(sessionID, true));
        }
      }
      grouped.clear();
      records.length = 0;
      await Promise.all(scheduled);
    } catch {
      // OpenCode event hooks are observational and always fail open.
    }
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    try {
      await this.#queue.drain();
      for (const record of this.#reducer.dispose()) {
        const scheduled: Promise<void>[] = [];
        this.#appendRecords(record.session_id, [asCanonicalRecord(record)], scheduled);
        await Promise.all(scheduled);
      }
      for (let pass = 0; pass < MAX_DISPOSE_FLUSH_PASSES; pass += 1) {
        const sessions = this.#knownSessions();
        if (sessions.length === 0) break;
        await Promise.all(
          sessions.map((sessionID) => this.#scheduleFlush(sessionID, true)),
        );
        await this.#queue.drain();
      }
    } catch {
      // OpenCode finalizers must never affect provider shutdown.
    }
  }
}

export function createOpenCodePlugin(
  options: {
    bootstrapURL: URL;
    loadBootstrap?: (bootstrapURL: URL) => Promise<BootstrapBinding>;
    platform?: NodeJS.Platform;
    environment?: NodeJS.ProcessEnv;
    writeChunk?: CaptureChunkWriter;
    batchID?: () => string;
  },
): OpenCodePlugin {
  return async (_input: unknown) => {
    let runtime: OpenCodePluginRuntime | undefined;
    try {
      const loader = options.loadBootstrap ?? loadOpenCodeBootstrap;
      const bootstrap = await loader(options.bootstrapURL);
      runtime = new OpenCodePluginRuntime(bootstrap, options);
    } catch {
      runtime = undefined;
    }
    return {
      event: async (input: { event: unknown }) => {
        try {
          await runtime?.event(input.event as OpenCodeEvent);
        } catch {
          // Provider callbacks remain fail-open.
        }
      },
      dispose: async () => {
        try {
          await runtime?.dispose();
        } catch {
          // Provider finalization remains fail-open.
        }
      },
    };
  };
}
