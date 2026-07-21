import { spawn, type ChildProcess } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import type { Writable } from "node:stream";

import { encodeCanonicalJson, isWellFormedUnicode } from "./canonical.ts";
import {
  MAX_CAPTURE_SESSION_ID_BYTES,
  type BootstrapBinding,
  type CanonicalJson,
} from "./contracts.ts";
import { launcherInvocation, type LauncherInvocation } from "./launcher.ts";
import { buildWindowedCaptureChunks } from "./windowed-chunks.ts";

export const CAPTURE_LAUNCHER_TIMEOUT_MS = 2_000;
const MAX_CONCURRENT_CAPTURE_LAUNCHERS = 4;
const MAX_PENDING_GAP_WINDOWS = 256;
const WINDOW_DISCRIMINATOR = /^[0-9a-f]{64}$/;

export type WindowCoordinates = Readonly<{
  sessionID: string;
  windowDiscriminator: string;
}>;

export type SpawnChild = ChildProcess & { stdin: Writable | null };
export type SpawnFunction = (
  file: string,
  arguments_: string[],
  options: LauncherInvocation["options"],
) => SpawnChild;

export type WindowedCaptureChunkWrite = Readonly<{
  invocation: LauncherInvocation;
  bytes: Buffer;
  timeoutMS: typeof CAPTURE_LAUNCHER_TIMEOUT_MS;
}>;

export type WindowedCaptureChunkWriter = (
  write: WindowedCaptureChunkWrite,
) => Promise<boolean>;
export type WindowedCaptureFlushResult =
  | "attempted_failure"
  | "delivered"
  | "not_started";

function environment(value: NodeJS.ProcessEnv): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "string") result[key] = item;
  }
  return result;
}

export async function spawnWindowedCaptureChunk(
  input: Omit<WindowedCaptureChunkWrite, "timeoutMS">,
  spawnChild: SpawnFunction = spawn as SpawnFunction,
): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    let child: SpawnChild;
    try {
      child = spawnChild(
        input.invocation.file,
        input.invocation.arguments,
        input.invocation.options,
      );
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
        // Capture is observational; cleanup errors cannot affect the provider.
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

function validCoordinates(value: WindowCoordinates): boolean {
  return (
    typeof value.sessionID === "string" &&
    value.sessionID.length > 0 &&
    isWellFormedUnicode(value.sessionID) &&
    Buffer.byteLength(value.sessionID, "utf8") <= MAX_CAPTURE_SESSION_ID_BYTES &&
    typeof value.windowDiscriminator === "string" &&
    WINDOW_DISCRIMINATOR.test(value.windowDiscriminator)
  );
}

function windowKey(value: WindowCoordinates): string {
  return createHash("sha256")
    .update(
      encodeCanonicalJson({
        session_id: value.sessionID,
        window_discriminator: value.windowDiscriminator,
      }),
    )
    .digest("hex");
}

function hasMatchingWindow(value: CanonicalJson, window: WindowCoordinates): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    value.session_id === window.sessionID &&
    value.window_discriminator === window.windowDiscriminator
  );
}

type PendingGap = {
  coordinates: WindowCoordinates;
  generation: bigint;
};

export class WindowedBatchTransport {
  readonly #bootstrap: BootstrapBinding;
  readonly #invocation: LauncherInvocation;
  readonly #writeChunk: WindowedCaptureChunkWriter;
  readonly #batchID: () => string;
  readonly #pendingGaps = new Map<string, PendingGap>();
  #gapGeneration = 0n;
  #activeWrites = 0;
  #globalGapPoison = false;

  constructor(
    bootstrap: BootstrapBinding,
    options: {
      platform?: NodeJS.Platform;
      environment?: NodeJS.ProcessEnv;
      writeChunk?: WindowedCaptureChunkWriter;
      batchID?: () => string;
    } = {},
  ) {
    this.#bootstrap = bootstrap;
    this.#invocation = launcherInvocation({
      platform: options.platform ?? process.platform,
      launcherPath: bootstrap.launcher_path,
      environment: environment(options.environment ?? process.env),
    });
    this.#writeChunk = options.writeChunk ?? spawnWindowedCaptureChunk;
    this.#batchID = options.batchID ?? (() => randomBytes(32).toString("hex"));
  }

  pendingWindows(): WindowCoordinates[] {
    return [...this.#pendingGaps.values()].map((value) => value.coordinates);
  }

  hasPendingGap(coordinates: WindowCoordinates): boolean {
    return (
      validCoordinates(coordinates) &&
      (this.#globalGapPoison || this.#pendingGaps.has(windowKey(coordinates)))
    );
  }

  markGap(coordinates: WindowCoordinates): void {
    if (!validCoordinates(coordinates)) return;
    const key = windowKey(coordinates);
    if (this.#pendingGaps.has(key) || this.#pendingGaps.size < MAX_PENDING_GAP_WINDOWS) {
      this.#gapGeneration += 1n;
      this.#pendingGaps.set(key, {
        coordinates: { ...coordinates },
        generation: this.#gapGeneration,
      });
    } else {
      this.#globalGapPoison = true;
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
    write: WindowedCaptureChunkWrite,
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
    coordinates: WindowCoordinates,
    records: readonly CanonicalJson[],
    options: { force?: boolean } = {},
  ): Promise<WindowedCaptureFlushResult> {
    try {
      if (
        !validCoordinates(coordinates) ||
        !records.every((record) => hasMatchingWindow(record, coordinates))
      ) {
        this.markGap(coordinates);
        return "attempted_failure";
      }
      const key = windowKey(coordinates);
      const pending = this.#pendingGaps.get(key);
      const events: CanonicalJson[] = [
        ...(pending === undefined && !this.#globalGapPoison
          ? []
          : [
              {
                kind: "coverage_degraded",
                reason: "transport_gap",
                session_id: coordinates.sessionID,
                window_discriminator: coordinates.windowDiscriminator,
              } as const,
            ]),
        ...records,
      ];
      if (events.length === 0 && options.force !== true) return "delivered";
      const chunks = buildWindowedCaptureChunks({
        bootstrap: this.#bootstrap,
        batchID: this.#batchID(),
        sessionID: coordinates.sessionID,
        windowDiscriminator: coordinates.windowDiscriminator,
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
        this.markGap(coordinates);
        return "not_started";
      }
      if (delivered === chunks.length) {
        if (this.#pendingGaps.get(key)?.generation === pending?.generation) {
          this.#pendingGaps.delete(key);
        }
        return "delivered";
      }
      this.markGap(coordinates);
      return "attempted_failure";
    } catch {
      this.markGap(coordinates);
      return "attempted_failure";
    }
  }
}
