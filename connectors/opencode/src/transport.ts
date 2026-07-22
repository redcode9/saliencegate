import { randomBytes } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import type { Writable } from "node:stream";

import {
  MAX_CAPTURE_SESSION_ID_BYTES,
  buildCaptureChunks,
  isWellFormedUnicode,
  type BootstrapBinding,
  type CanonicalJson,
} from "@saliencegate/bridge-core";

import { launcherInvocation, type LauncherInvocation } from "./launcher.ts";

export const CAPTURE_LAUNCHER_TIMEOUT_MS = 2_000;
const MAX_CONCURRENT_CAPTURE_LAUNCHERS = 4;
const MAX_PENDING_GAP_SESSIONS = 256;

export type SpawnChild = ChildProcess & { stdin: Writable | null };
export type SpawnFunction = (
  file: string,
  arguments_: string[],
  options: LauncherInvocation["options"],
) => SpawnChild;

export type CaptureChunkWrite = Readonly<{
  invocation: LauncherInvocation;
  bytes: Buffer;
  timeoutMS: typeof CAPTURE_LAUNCHER_TIMEOUT_MS;
}>;

export type CaptureChunkWriter = (write: CaptureChunkWrite) => Promise<boolean>;
export type CaptureFlushResult = "attempted_failure" | "delivered" | "not_started";

export async function spawnCaptureChunk(
  input: Omit<CaptureChunkWrite, "timeoutMS">,
  spawnChild: SpawnFunction = spawn as SpawnFunction,
): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    let child: SpawnChild;
    try {
      child = spawnChild(input.invocation.file, input.invocation.arguments, input.invocation.options);
    } catch {
      resolve(false);
      return;
    }

    let settled = false;
    let timedOut = false;
    let stdinFailed = false;
    const finish = (result: boolean): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill();
      } catch {
        // The provider must remain fail-open even when cleanup itself fails.
      }
      finish(false);
    }, CAPTURE_LAUNCHER_TIMEOUT_MS);
    child.once("error", () => finish(false));
    child.once("close", (code) => finish(code === 0 && !timedOut && !stdinFailed));
    if (child.stdin === null) {
      finish(false);
      return;
    }
    child.stdin.once("error", () => {
      stdinFailed = true;
    });
    try {
      child.stdin.end(input.bytes);
    } catch {
      finish(false);
    }
  });
}

type SessionRecord = CanonicalJson & { session_id: string };

function hasMatchingSession(value: CanonicalJson, sessionID: string): value is SessionRecord {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    value.session_id === sessionID
  );
}

export class OpenCodeBatchTransport {
  readonly #bootstrap: BootstrapBinding;
  readonly #invocation: LauncherInvocation;
  readonly #writeChunk: CaptureChunkWriter;
  readonly #batchID: () => string;
  readonly #pendingGaps = new Map<string, bigint>();
  #gapGeneration = 0n;
  #activeWrites = 0;

  constructor(
    bootstrap: BootstrapBinding,
    options: {
      platform?: NodeJS.Platform;
      environment?: NodeJS.ProcessEnv;
      writeChunk?: CaptureChunkWriter;
      batchID?: () => string;
    } = {},
  ) {
    this.#bootstrap = bootstrap;
    this.#invocation = launcherInvocation({
      platform: options.platform ?? process.platform,
      launcherPath: bootstrap.launcher_path,
      environment: options.environment ?? process.env,
    });
    this.#writeChunk = options.writeChunk ?? spawnCaptureChunk;
    this.#batchID = options.batchID ?? (() => randomBytes(32).toString("hex"));
  }

  pendingSessionIDs(): string[] {
    return [...this.#pendingGaps.keys()];
  }

  hasPendingGap(sessionID: string): boolean {
    return (
      typeof sessionID === "string" &&
      sessionID.length > 0 &&
      isWellFormedUnicode(sessionID) &&
      Buffer.byteLength(sessionID, "utf8") <= MAX_CAPTURE_SESSION_ID_BYTES &&
      this.#pendingGaps.has(sessionID)
    );
  }

  markGap(sessionID: string): void {
    if (
      typeof sessionID !== "string" ||
      sessionID.length === 0 ||
      !isWellFormedUnicode(sessionID) ||
      Buffer.byteLength(sessionID, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES
    ) {
      return;
    }
    if (
      this.#pendingGaps.has(sessionID) ||
      this.#pendingGaps.size < MAX_PENDING_GAP_SESSIONS
    ) {
      this.#gapGeneration += 1n;
      this.#pendingGaps.set(sessionID, this.#gapGeneration);
    }
  }

  #acquireWritePermit(): (() => void) | undefined {
    if (this.#activeWrites >= MAX_CONCURRENT_CAPTURE_LAUNCHERS) return undefined;
    this.#activeWrites += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#activeWrites -= 1;
    };
  }

  async #writeBounded(
    write: CaptureChunkWrite,
  ): Promise<"delivered" | "failed" | "not_started"> {
    const release = this.#acquireWritePermit();
    if (release === undefined) return "not_started";
    try {
      return (await this.#writeChunk(write)) ? "delivered" : "failed";
    } catch {
      return "failed";
    } finally {
      release();
    }
  }

  async flush(
    sessionID: string,
    records: readonly CanonicalJson[],
  ): Promise<CaptureFlushResult> {
    try {
      if (!records.every((record) => hasMatchingSession(record, sessionID))) {
        this.markGap(sessionID);
        return "attempted_failure";
      }
      const pendingGapGeneration = this.#pendingGaps.get(sessionID);
      const events: CanonicalJson[] = [
        ...(pendingGapGeneration !== undefined
          ? [
              {
                kind: "coverage_degraded",
                reason: "transport_gap",
                session_id: sessionID,
              } as const,
            ]
          : []),
        ...records,
      ];
      if (events.length === 0) return "delivered";
      const chunks = buildCaptureChunks({
        bootstrap: this.#bootstrap,
        batchID: this.#batchID(),
        sessionID,
        events,
      });
      let delivered = 0;
      let started = 0;
      for (const chunk of chunks) {
        const result = await this.#writeBounded({
          invocation: this.#invocation,
          bytes: chunk.bytes,
          timeoutMS: CAPTURE_LAUNCHER_TIMEOUT_MS,
        });
        if (result !== "not_started") started += 1;
        if (result === "delivered") delivered += 1;
      }
      if (started === 0) {
        this.markGap(sessionID);
        return "not_started";
      } else if (delivered === chunks.length) {
        if (this.#pendingGaps.get(sessionID) === pendingGapGeneration) {
          this.#pendingGaps.delete(sessionID);
        }
        return "delivered";
      } else {
        this.markGap(sessionID);
        return "attempted_failure";
      }
    } catch {
      this.markGap(sessionID);
      return "attempted_failure";
    }
  }
}
