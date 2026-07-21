import { randomBytes } from "node:crypto";

import {
  SerialSessionQueue,
  WindowedBatchTransport,
  encodeCanonicalJson,
  normalizeWindowedCaptureEvent,
  type BootstrapBinding,
  type CanonicalJson,
  type WindowCoordinates,
  type WindowedCaptureChunkWriter,
} from "@saliencegate/bridge-core";

import { loadPiBootstrap } from "./bootstrap.ts";
import {
  PiWindowReducer,
  isValidPiNativeSessionID,
  type ReducedPiRecord,
} from "./reducer.ts";
import type {
  PiAgentSettledEvent,
  PiBeforeAgentStartEvent,
  PiCompactionReason,
  PiExtension,
  PiExtensionAPI,
  PiExtensionContext,
  PiSessionCompactEvent,
  PiSessionShutdownEvent,
  PiSessionShutdownReason,
  PiSessionStartEvent,
  PiSessionStartReason,
  PiSessionTreeEvent,
  PiToolExecutionEndEvent,
  PiToolExecutionStartEvent,
} from "./upstream-types.ts";

const MAX_SESSION_BUFFER_BYTES = 512 * 1024;
const SERIAL_QUEUE_KEY = "pi-extension-runtime";
const WINDOW_DISCRIMINATOR = /^[0-9a-f]{64}$/;
const START_REASONS = new Set<PiSessionStartReason>([
  "startup",
  "reload",
  "new",
  "resume",
  "fork",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function dataValue(value: Record<string, unknown>, key: string): unknown | undefined {
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  if (descriptor === undefined || !("value" in descriptor) || !descriptor.enumerable) {
    return undefined;
  }
  return descriptor.value;
}

function contextSessionID(value: PiExtensionContext): string | undefined {
  try {
    const sessionID = value.sessionManager.getSessionId();
    return isValidPiNativeSessionID(sessionID) ? sessionID : undefined;
  } catch {
    return undefined;
  }
}

function asCanonicalRecord(record: ReducedPiRecord): CanonicalJson {
  return normalizeWindowedCaptureEvent(
    record,
    record.session_id,
    record.window_discriminator,
  );
}

function hasTerminal(records: readonly CanonicalJson[]): boolean {
  return records.some(
    (record) =>
      typeof record === "object" &&
      record !== null &&
      !Array.isArray(record) &&
      record.kind === "session_finished",
  );
}

type ActiveWindow = {
  reducer: PiWindowReducer;
  coordinates: WindowCoordinates;
  records: CanonicalJson[];
  bytes: number;
};

class PiExtensionRuntime {
  readonly #queue = new SerialSessionQueue();
  readonly #transport: WindowedBatchTransport;
  readonly #windowDiscriminator: () => string;
  #active: ActiveWindow | undefined;

  constructor(
    bootstrap: BootstrapBinding,
    options: {
      platform?: NodeJS.Platform;
      environment?: NodeJS.ProcessEnv;
      writeChunk?: WindowedCaptureChunkWriter;
      batchID?: () => string;
      windowDiscriminator?: () => string;
    },
  ) {
    this.#transport = new WindowedBatchTransport(bootstrap, options);
    this.#windowDiscriminator =
      options.windowDiscriminator ?? (() => randomBytes(32).toString("hex"));
  }

  async #flush(active: ActiveWindow, force = false): Promise<void> {
    const records = active.records;
    if (!force && records.length === 0) return;
    const terminal = hasTerminal(records);
    active.records = [];
    active.bytes = 0;
    const result = await this.#transport.flush(active.coordinates, records, { force });
    if (result === "not_started" && terminal) {
      const retained = records.filter(
        (record) =>
          typeof record === "object" &&
          record !== null &&
          !Array.isArray(record) &&
          record.kind === "session_finished",
      );
      active.records = retained;
      active.bytes = 0;
      for (const record of retained) {
        active.bytes += encodeCanonicalJson(record).byteLength;
      }
    }
  }

  async #append(active: ActiveWindow, records: readonly ReducedPiRecord[]): Promise<void> {
    const normalized = records.map((record) => asCanonicalRecord(record));
    const oversized = normalized.find(
      (record) =>
        typeof record === "object" &&
        record !== null &&
        !Array.isArray(record) &&
        record.kind === "oversize",
    );
    const group = oversized === undefined ? normalized : [oversized];
    const sized = group.map((record) => ({
      record,
      bytes: encodeCanonicalJson(record).byteLength,
    }));
    const groupBytes = sized.reduce((total, item) => total + item.bytes, 0);
    if (
      active.records.length > 0 &&
      active.bytes + groupBytes > MAX_SESSION_BUFFER_BYTES
    ) {
      await this.#flush(active);
    }
    for (const item of sized) {
      active.records.push(item.record);
      active.bytes += item.bytes;
    }
  }

  async #degradeActive(reason: "invalid_transition" | "missing_field"): Promise<void> {
    if (this.#active === undefined) return;
    await this.#append(this.#active, this.#active.reducer.degrade(reason));
  }

  async #sessionStart(value: unknown, context: PiExtensionContext): Promise<void> {
    const sessionID = contextSessionID(context);
    if (!isRecord(value) || dataValue(value, "type") !== "session_start") {
      await this.#degradeActive("missing_field");
      if (this.#active !== undefined) await this.#flush(this.#active);
      return;
    }
    const reason = dataValue(value, "reason");
    if (
      sessionID === undefined ||
      typeof reason !== "string" ||
      !START_REASONS.has(reason as PiSessionStartReason)
    ) {
      await this.#degradeActive("missing_field");
      if (this.#active !== undefined) await this.#flush(this.#active);
      return;
    }
    if (this.#active !== undefined) {
      const active = this.#active;
      await this.#degradeActive("invalid_transition");
      await this.#flush(active);
      if (active.records.length > 0) return;
      this.#active = undefined;
    }
    let discriminator: string;
    try {
      discriminator = this.#windowDiscriminator();
    } catch {
      return;
    }
    if (!WINDOW_DISCRIMINATOR.test(discriminator)) return;
    const reducer = new PiWindowReducer({
      sessionID,
      windowDiscriminator: discriminator,
    });
    const active: ActiveWindow = {
      reducer,
      coordinates: reducer.coordinates(),
      records: [],
      bytes: 0,
    };
    this.#active = active;
    await this.#flush(active, true);
  }

  async #observedEvent(
    expectedType: string,
    value: unknown,
    context: PiExtensionContext,
    flushBoundary: boolean,
    closesWindow: boolean,
  ): Promise<void> {
    const active = this.#active;
    if (active === undefined) return;
    const sessionID = contextSessionID(context);
    if (
      sessionID !== active.coordinates.sessionID ||
      !isRecord(value) ||
      dataValue(value, "type") !== expectedType
    ) {
      await this.#append(active, active.reducer.degrade("missing_field"));
    } else {
      await this.#append(active, active.reducer.reduce(value));
    }
    if (flushBoundary) await this.#flush(active);
    if (
      closesWindow &&
      active.reducer.isFinished() &&
      active.records.length === 0
    ) {
      this.#active = undefined;
    }
  }

  sessionStart(value: unknown, context: PiExtensionContext): Promise<void> {
    return this.#queue.run(SERIAL_QUEUE_KEY, async () => {
      await this.#sessionStart(value, context);
    });
  }

  observedEvent(
    expectedType: string,
    value: unknown,
    context: PiExtensionContext,
    options: { flushBoundary?: boolean; closesWindow?: boolean } = {},
  ): Promise<void> {
    return this.#queue.run(SERIAL_QUEUE_KEY, async () => {
      await this.#observedEvent(
        expectedType,
        value,
        context,
        options.flushBoundary === true,
        options.closesWindow === true,
      );
    });
  }
}

export function createPiExtension(options: {
  bootstrapURL: URL;
  loadBootstrap?: (bootstrapURL: URL) => Promise<BootstrapBinding>;
  platform?: NodeJS.Platform;
  environment?: NodeJS.ProcessEnv;
  writeChunk?: WindowedCaptureChunkWriter;
  batchID?: () => string;
  windowDiscriminator?: () => string;
}): PiExtension {
  return async (pi: PiExtensionAPI): Promise<void> => {
    let runtime: PiExtensionRuntime | undefined;
    try {
      const loader = options.loadBootstrap ?? loadPiBootstrap;
      const bootstrap = await loader(options.bootstrapURL);
      if (bootstrap.profile === "pi-extension/v1") {
        runtime = new PiExtensionRuntime(bootstrap, options);
      }
    } catch {
      runtime = undefined;
    }

    pi.on("session_start", async (event: PiSessionStartEvent, context) => {
      try {
        await runtime?.sessionStart(event, context);
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on("before_agent_start", async (event: PiBeforeAgentStartEvent, context) => {
      try {
        await runtime?.observedEvent("before_agent_start", event, context);
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on(
      "tool_execution_start",
      async (event: PiToolExecutionStartEvent, context) => {
        try {
          await runtime?.observedEvent("tool_execution_start", event, context);
        } catch {
          // Pi lifecycle callbacks are observational and always fail open.
        }
        return undefined;
      },
    );
    pi.on("tool_execution_end", async (event: PiToolExecutionEndEvent, context) => {
      try {
        await runtime?.observedEvent("tool_execution_end", event, context);
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on("agent_settled", async (event: PiAgentSettledEvent, context) => {
      try {
        await runtime?.observedEvent("agent_settled", event, context, {
          flushBoundary: true,
        });
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on("session_compact", async (event: PiSessionCompactEvent, context) => {
      try {
        await runtime?.observedEvent("session_compact", event, context, {
          flushBoundary: true,
        });
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on("session_tree", async (event: PiSessionTreeEvent, context) => {
      try {
        await runtime?.observedEvent("session_tree", event, context, {
          flushBoundary: true,
        });
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
    pi.on("session_shutdown", async (event: PiSessionShutdownEvent, context) => {
      try {
        await runtime?.observedEvent("session_shutdown", event, context, {
          flushBoundary: true,
          closesWindow: true,
        });
      } catch {
        // Pi lifecycle callbacks are observational and always fail open.
      }
      return undefined;
    });
  };
}

// Keep the frozen enums reachable to type-level consumers without accessing any
// ignored payload fields at runtime.
void (undefined as PiCompactionReason | PiSessionShutdownReason | undefined);
