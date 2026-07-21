import { createHash } from "node:crypto";

import {
  BridgeContractError,
  MAX_CAPTURE_CALL_ID_BYTES,
  MAX_CAPTURE_TOOL_NAME_BYTES,
  encodeCanonicalJson,
  isWellFormedUnicode,
} from "@saliencegate/bridge-core";

import type { PiCompactionReason, PiSessionShutdownReason } from "./upstream-types.ts";

export const MAX_PI_NATIVE_SESSION_ID_BYTES = 16 * 1024;
export const MAX_PI_LEAF_ID_BYTES = 16 * 1024;
const MAX_CALLS_PER_WINDOW = 1_000;
const MAX_REDUCED_RECORDS_PER_WINDOW = 997;
const MAX_REDUCER_STATE_BYTES = 2 * 1024 * 1024;
const CALL_STATE_OVERHEAD_BYTES = 192;
const FINAL_CALL_STATE_BYTES = 192;
const WINDOW_DISCRIMINATOR = /^[0-9a-f]{64}$/;
const NATIVE_SESSION_ID = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/;
const COMPACTION_REASONS = new Set<PiCompactionReason>([
  "manual",
  "threshold",
  "overflow",
]);
const SHUTDOWN_REASONS = new Set<PiSessionShutdownReason>([
  "quit",
  "reload",
  "new",
  "resume",
  "fork",
]);

type CommonRecord = Readonly<{
  session_id: string;
  window_discriminator: string;
  event_id: string;
}>;

export type ReducedPiRecord = CommonRecord &
  (
    | Readonly<{
        kind: "tool_started";
        call_id: string;
        tool: string;
        identity_authority: "coarse";
      }>
    | Readonly<{
        kind: "tool_finished";
        call_id: string;
        outcome: "succeeded";
      }>
    | Readonly<{ kind: "turn_finished" }>
    | Readonly<{
        kind: "coverage_boundary";
        reason: "compaction";
        compaction_reason: PiCompactionReason;
        from_extension: boolean;
        will_retry: boolean;
      }>
    | Readonly<{
        kind: "coverage_boundary";
        reason: "tree";
        old_leaf_id: string | null;
        new_leaf_id: string | null;
      }>
    | Readonly<{
        kind: "coverage_degraded";
        reason:
          | "ambiguous_error"
          | "invalid_transition"
          | "missing_field"
          | "overflow"
          | "unmatched_start";
      }>
    | Readonly<{
        kind: "session_finished";
        reason: PiSessionShutdownReason;
      }>
  );

type ReducedPiBody =
  | Readonly<{
      kind: "tool_started";
      call_id: string;
      tool: string;
      identity_authority: "coarse";
    }>
  | Readonly<{
      kind: "tool_finished";
      call_id: string;
      outcome: "succeeded";
    }>
  | Readonly<{ kind: "turn_finished" }>
  | Readonly<{
      kind: "coverage_boundary";
      reason: "compaction";
      compaction_reason: PiCompactionReason;
      from_extension: boolean;
      will_retry: boolean;
    }>
  | Readonly<{
      kind: "coverage_boundary";
      reason: "tree";
      old_leaf_id: string | null;
      new_leaf_id: string | null;
    }>
  | Readonly<{
      kind: "coverage_degraded";
      reason:
        | "ambiguous_error"
        | "invalid_transition"
        | "missing_field"
        | "overflow"
        | "unmatched_start";
    }>
  | Readonly<{
      kind: "session_finished";
      reason: PiSessionShutdownReason;
    }>;

type PendingCall = {
  startFingerprint: string;
  retainedBytes: number;
};

type FinalCall = {
  startFingerprint?: string;
  endFingerprint?: string;
  retainedBytes: number;
};

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

function exactText(value: unknown, maximumBytes: number): string | undefined {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    !isWellFormedUnicode(value) ||
    Buffer.byteLength(value, "utf8") > maximumBytes
  ) {
    return undefined;
  }
  return value;
}

function exactLeaf(value: unknown): string | null | undefined {
  return value === null ? null : exactText(value, MAX_PI_LEAF_ID_BYTES);
}

function digest(value: unknown): string {
  return createHash("sha256").update(encodeCanonicalJson(value)).digest("hex");
}

function checkedNativeSessionID(value: unknown): string {
  const text = exactText(value, MAX_PI_NATIVE_SESSION_ID_BYTES);
  if (text === undefined || !NATIVE_SESSION_ID.test(text)) throw new BridgeContractError();
  return text;
}

export function isValidPiNativeSessionID(value: unknown): value is string {
  try {
    checkedNativeSessionID(value);
    return true;
  } catch {
    return false;
  }
}

export class PiWindowReducer {
  readonly #sessionID: string;
  readonly #windowDiscriminator: string;
  readonly #pending = new Map<string, PendingCall>();
  readonly #final = new Map<string, FinalCall>();
  #retainedStateBytes = 0;
  #recordCount = 0;
  #nextEventID = 1;
  #overflowReported = false;
  #disabled = false;
  #finished = false;
  #turnOpen = false;

  constructor(input: { sessionID: string; windowDiscriminator: string }) {
    this.#sessionID = checkedNativeSessionID(input.sessionID);
    if (!WINDOW_DISCRIMINATOR.test(input.windowDiscriminator)) {
      throw new BridgeContractError();
    }
    this.#windowDiscriminator = input.windowDiscriminator;
  }

  coordinates(): Readonly<{ sessionID: string; windowDiscriminator: string }> {
    return {
      sessionID: this.#sessionID,
      windowDiscriminator: this.#windowDiscriminator,
    };
  }

  isFinished(): boolean {
    return this.#finished;
  }

  #reserveState(bytes: number): boolean {
    if (this.#retainedStateBytes + bytes > MAX_REDUCER_STATE_BYTES) return false;
    this.#retainedStateBytes += bytes;
    return true;
  }

  #releaseState(bytes: number): void {
    this.#retainedStateBytes = Math.max(0, this.#retainedStateBytes - bytes);
  }

  #clearPending(): boolean {
    const hadPending = this.#pending.size > 0;
    for (const [key, call] of this.#pending) {
      this.#releaseState(call.retainedBytes - FINAL_CALL_STATE_BYTES);
      this.#final.set(key, {
        startFingerprint: call.startFingerprint,
        retainedBytes: FINAL_CALL_STATE_BYTES,
      });
    }
    this.#pending.clear();
    return hadPending;
  }

  #materialize(body: ReducedPiBody): ReducedPiRecord {
    const record = {
      ...body,
      session_id: this.#sessionID,
      window_discriminator: this.#windowDiscriminator,
      event_id: String(this.#nextEventID),
    } as ReducedPiRecord;
    this.#nextEventID += 1;
    this.#recordCount += 1;
    return record;
  }

  #admit(
    bodies: readonly ReducedPiBody[],
    mode: "degradation" | "normal" | "terminal" = "normal",
  ): ReducedPiRecord[] {
    if (bodies.length === 0) return [];
    const limit =
      mode === "terminal"
        ? MAX_REDUCED_RECORDS_PER_WINDOW
        : mode === "degradation"
          ? MAX_REDUCED_RECORDS_PER_WINDOW - 1
          : MAX_REDUCED_RECORDS_PER_WINDOW - 2;
    if (this.#recordCount + bodies.length <= limit) {
      return bodies.map((body) => this.#materialize(body));
    }
    if (mode !== "normal") return [];
    this.#disabled = true;
    if (!this.#overflowReported && this.#recordCount < MAX_REDUCED_RECORDS_PER_WINDOW - 1) {
      this.#overflowReported = true;
      return [
        this.#materialize({
          kind: "coverage_degraded",
          reason: "overflow",
        }),
      ];
    }
    return [];
  }

  degrade(
    reason:
      | "ambiguous_error"
      | "invalid_transition"
      | "missing_field"
      | "overflow"
      | "unmatched_start",
  ): ReducedPiRecord[] {
    if (this.#finished) return [];
    if (reason === "overflow") {
      this.#disabled = true;
      if (this.#overflowReported) return [];
      this.#overflowReported = true;
    }
    return this.#admit([{ kind: "coverage_degraded", reason }], "degradation");
  }

  #rememberFinal(key: string, value: Omit<FinalCall, "retainedBytes">): boolean {
    if (!this.#reserveState(FINAL_CALL_STATE_BYTES)) return false;
    this.#final.set(key, { ...value, retainedBytes: FINAL_CALL_STATE_BYTES });
    return true;
  }

  #toolStart(event: Record<string, unknown>): ReducedPiRecord[] {
    if (this.#disabled) return [];
    const callID = exactText(dataValue(event, "toolCallId"), MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(dataValue(event, "toolName"), MAX_CAPTURE_TOOL_NAME_BYTES);
    if (callID === undefined || tool === undefined) return this.degrade("missing_field");
    const key = digest({ callID });
    const startFingerprint = digest({ callID, tool });
    const finalized = this.#final.get(key);
    if (finalized !== undefined) {
      return finalized.startFingerprint === startFingerprint
        ? []
        : this.degrade("invalid_transition");
    }
    const prior = this.#pending.get(key);
    if (prior !== undefined) {
      if (prior.startFingerprint === startFingerprint) return [];
      this.#pending.delete(key);
      this.#releaseState(prior.retainedBytes);
      if (!this.#rememberFinal(key, {})) return this.degrade("overflow");
      return this.degrade("invalid_transition");
    }
    if (this.#pending.size + this.#final.size >= MAX_CALLS_PER_WINDOW) {
      return this.degrade("overflow");
    }
    const retainedBytes = CALL_STATE_OVERHEAD_BYTES;
    if (!this.#reserveState(retainedBytes)) return this.degrade("overflow");
    this.#pending.set(key, { startFingerprint, retainedBytes });
    this.#turnOpen = true;
    return [];
  }

  #toolEnd(event: Record<string, unknown>): ReducedPiRecord[] {
    if (this.#disabled) return [];
    const callID = exactText(dataValue(event, "toolCallId"), MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(dataValue(event, "toolName"), MAX_CAPTURE_TOOL_NAME_BYTES);
    const isError = dataValue(event, "isError");
    if (callID === undefined || tool === undefined || typeof isError !== "boolean") {
      return this.degrade("missing_field");
    }
    const key = digest({ callID });
    const startFingerprint = digest({ callID, tool });
    const endFingerprint = digest({ callID, tool, isError });
    const finalized = this.#final.get(key);
    if (finalized !== undefined) {
      if (finalized.endFingerprint === endFingerprint) return [];
      if (finalized.endFingerprint === undefined) {
        finalized.endFingerprint = endFingerprint;
      }
      return this.degrade("invalid_transition");
    }
    const pending = this.#pending.get(key);
    if (pending === undefined) {
      if (this.#pending.size + this.#final.size >= MAX_CALLS_PER_WINDOW) {
        return this.degrade("overflow");
      }
      if (!this.#rememberFinal(key, { endFingerprint })) return this.degrade("overflow");
      return this.degrade("invalid_transition");
    }
    this.#pending.delete(key);
    this.#releaseState(pending.retainedBytes);
    if (pending.startFingerprint !== startFingerprint) {
      if (!this.#rememberFinal(key, {})) return this.degrade("overflow");
      return this.degrade("invalid_transition");
    }
    if (!this.#rememberFinal(key, { startFingerprint, endFingerprint })) {
      return this.degrade("overflow");
    }
    if (isError) return this.degrade("ambiguous_error");
    return this.#admit([
      {
        kind: "tool_started",
        call_id: callID,
        tool,
        identity_authority: "coarse",
      },
      {
        kind: "tool_finished",
        call_id: callID,
        outcome: "succeeded",
      },
    ]);
  }

  #beforeAgentStart(): ReducedPiRecord[] {
    if (this.#disabled) return [];
    if (this.#turnOpen) return this.degrade("invalid_transition");
    this.#turnOpen = true;
    return [];
  }

  #agentSettled(): ReducedPiRecord[] {
    const unmatched = this.#clearPending();
    const bodies: ReducedPiBody[] = [];
    if (unmatched) bodies.push({ kind: "coverage_degraded", reason: "unmatched_start" });
    if (this.#turnOpen) bodies.push({ kind: "turn_finished" });
    this.#turnOpen = false;
    return this.#admit(bodies);
  }

  #compact(event: Record<string, unknown>): ReducedPiRecord[] {
    const reason = dataValue(event, "reason");
    const fromExtension = dataValue(event, "fromExtension");
    const willRetry = dataValue(event, "willRetry");
    if (
      typeof reason !== "string" ||
      !COMPACTION_REASONS.has(reason as PiCompactionReason) ||
      typeof fromExtension !== "boolean" ||
      typeof willRetry !== "boolean"
    ) {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    return this.#admit([
      ...(unmatched
        ? ([{ kind: "coverage_degraded", reason: "unmatched_start" }] as const)
        : []),
      {
        kind: "coverage_boundary",
        reason: "compaction",
        compaction_reason: reason as PiCompactionReason,
        from_extension: fromExtension,
        will_retry: willRetry,
      },
    ]);
  }

  #tree(event: Record<string, unknown>): ReducedPiRecord[] {
    const newLeafID = exactLeaf(dataValue(event, "newLeafId"));
    const oldLeafID = exactLeaf(dataValue(event, "oldLeafId"));
    if (newLeafID === undefined || oldLeafID === undefined) {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    return this.#admit([
      ...(unmatched
        ? ([{ kind: "coverage_degraded", reason: "unmatched_start" }] as const)
        : []),
      {
        kind: "coverage_boundary",
        reason: "tree",
        old_leaf_id: oldLeafID,
        new_leaf_id: newLeafID,
      },
    ]);
  }

  #shutdown(event: Record<string, unknown>): ReducedPiRecord[] {
    const reason = dataValue(event, "reason");
    if (
      typeof reason !== "string" ||
      !SHUTDOWN_REASONS.has(reason as PiSessionShutdownReason)
    ) {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    this.#turnOpen = false;
    const terminal: ReducedPiBody = {
      kind: "session_finished",
      reason: reason as PiSessionShutdownReason,
    };
    const bodies: ReducedPiBody[] = unmatched
      ? [{ kind: "coverage_degraded", reason: "unmatched_start" }, terminal]
      : [terminal];
    if (this.#recordCount + bodies.length > MAX_REDUCED_RECORDS_PER_WINDOW) {
      bodies.splice(0, bodies.length, terminal);
    }
    const records = this.#admit(
      bodies,
      "terminal",
    );
    this.#finished = true;
    return records;
  }

  reduce(value: unknown): ReducedPiRecord[] {
    try {
      if (!isRecord(value) || this.#finished) return [];
      const type = dataValue(value, "type");
      if (typeof type !== "string") return [];
      if (type === "before_agent_start") return this.#beforeAgentStart();
      if (type === "tool_execution_start") return this.#toolStart(value);
      if (type === "tool_execution_end") return this.#toolEnd(value);
      if (type === "agent_settled") return this.#agentSettled();
      if (type === "session_compact") return this.#compact(value);
      if (type === "session_tree") return this.#tree(value);
      if (type === "session_shutdown") return this.#shutdown(value);
      return [];
    } catch {
      return this.degrade("missing_field");
    }
  }
}
