import { createHash } from "node:crypto";

import {
  BridgeContractError,
  MAX_CAPTURE_CALL_ID_BYTES,
  MAX_CAPTURE_EVENT_ID_BYTES,
  MAX_CAPTURE_SESSION_ID_BYTES,
  MAX_CAPTURE_TOOL_NAME_BYTES,
  canonicalizeJson,
  encodeCanonicalJson,
  isWellFormedUnicode,
  type CanonicalJson,
} from "@saliencegate/bridge-core";

import type { OpenCodeEvent, OpenCodeToolStatus } from "./upstream-types.ts";

const MAX_SESSIONS = 256;
const MAX_CALLS_PER_SESSION = 1_000;
const MAX_EVENT_IDS_PER_SESSION = 4_096;
const MAX_REDUCED_RECORDS_PER_SESSION = 997;
const MAX_FINALIZED_SESSIONS = 1_024;
const MAX_FINALIZED_EVENT_IDS = 8;
const MAX_OVERFLOW_SESSION_MARKERS = 1_024;
const MAX_REDUCER_STATE_BYTES = 2 * 1024 * 1024;
const SESSION_STATE_OVERHEAD_BYTES = 256;
const CALL_STATE_OVERHEAD_BYTES = 192;
const EVENT_STATE_OVERHEAD_BYTES = 160;
const TOOL_STATES = new Set<OpenCodeToolStatus>(["pending", "running", "completed", "error"]);

export type ReducedOpenCodeRecord =
  | Readonly<{
      kind: "tool_started";
      session_id: string;
      event_id?: string;
      call_id: string;
      tool: string;
      input?: CanonicalJson;
      identity_authority: "exact" | "unavailable";
    }>
  | Readonly<{
      kind: "tool_finished";
      session_id: string;
      event_id?: string;
      call_id: string;
      outcome: "succeeded" | "failed";
    }>
  | Readonly<{
      kind: "turn_finished";
      session_id: string;
      event_id?: string;
    }>
  | Readonly<{
      kind: "controller_failed";
      session_id: string;
      event_id?: string;
    }>
  | Readonly<{
      kind: "coverage_boundary";
      session_id: string;
      event_id?: string;
    }>
  | Readonly<{
      kind: "coverage_degraded";
      session_id: string;
      reason: "invalid_transition" | "missing_field" | "overflow";
    }>
  | Readonly<{
      kind: "session_finished";
      session_id: string;
      event_id?: string;
    }>;

type CallState = {
  status: OpenCodeToolStatus;
  identity: string;
  retainedBytes: number;
};

type EventState = {
  fingerprint: string;
  retainedBytes: number;
};

type SessionState = {
  calls: Map<string, CallState>;
  events: Map<string, EventState>;
  disabled: boolean;
  deleted: boolean;
  recordCount: number;
  overflowReported: boolean;
  retainedBytes: number;
};

type FinalizedSessionState = {
  events: Map<string, string>;
  degradationReported: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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

function optionalEventID(value: unknown): string | undefined {
  return value === undefined ? undefined : exactText(value, MAX_CAPTURE_EVENT_ID_BYTES);
}

function digest(value: unknown): string {
  return createHash("sha256").update(encodeCanonicalJson(value)).digest("hex");
}

function rank(status: OpenCodeToolStatus): number {
  if (status === "pending") return 0;
  if (status === "running") return 1;
  return 2;
}

function isTerminal(status: OpenCodeToolStatus): boolean {
  return status === "completed" || status === "error";
}

export class OpenCodeEventReducer {
  readonly #sessions = new Map<string, SessionState>();
  readonly #finalizedSessions = new Map<string, FinalizedSessionState>();
  readonly #overflowSessions = new Map<string, true>();
  readonly #evictAfterReduction = new Set<string>();
  #retainedStateBytes = 0;

  #sessionKey(sessionID: string): string {
    return digest({ kind: "active_session", sessionID });
  }

  #activeSession(sessionID: string): SessionState | undefined {
    return this.#sessions.get(this.#sessionKey(sessionID));
  }

  #reserveState(bytes: number): boolean {
    if (this.#retainedStateBytes + bytes > MAX_REDUCER_STATE_BYTES) return false;
    this.#retainedStateBytes += bytes;
    return true;
  }

  #releaseState(bytes: number): void {
    this.#retainedStateBytes = Math.max(0, this.#retainedStateBytes - bytes);
  }

  #releaseSession(session: SessionState): void {
    let bytes = session.retainedBytes;
    for (const call of session.calls.values()) bytes += call.retainedBytes;
    for (const event of session.events.values()) bytes += event.retainedBytes;
    this.#releaseState(bytes);
    session.calls.clear();
    session.events.clear();
  }

  #finalizedKey(sessionID: string): string {
    return digest({ kind: "finalized_session", sessionID });
  }

  #overflowKey(sessionID: string): string {
    return digest({ kind: "session_table_overflow", sessionID });
  }

  #sessionOverflow(sessionID: string): ReducedOpenCodeRecord[] {
    const key = this.#overflowKey(sessionID);
    if (this.#overflowSessions.has(key)) {
      this.#overflowSessions.delete(key);
      this.#overflowSessions.set(key, true);
      return [];
    }
    while (this.#overflowSessions.size >= MAX_OVERFLOW_SESSION_MARKERS) {
      const oldest = this.#overflowSessions.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.#overflowSessions.delete(oldest);
    }
    this.#overflowSessions.set(key, true);
    return [{ kind: "coverage_degraded", session_id: sessionID, reason: "overflow" }];
  }

  #finalizedSession(sessionID: string): FinalizedSessionState | undefined {
    const key = this.#finalizedKey(sessionID);
    const state = this.#finalizedSessions.get(key);
    if (state === undefined) return undefined;
    this.#finalizedSessions.delete(key);
    this.#finalizedSessions.set(key, state);
    return state;
  }

  #rememberFinalized(
    sessionID: string,
    eventID: string | undefined,
    fingerprint: string,
  ): void {
    const key = this.#finalizedKey(sessionID);
    const state: FinalizedSessionState = {
      events: new Map<string, string>(),
      degradationReported: false,
    };
    if (eventID !== undefined) state.events.set(digest({ eventID }), fingerprint);
    this.#finalizedSessions.delete(key);
    while (this.#finalizedSessions.size >= MAX_FINALIZED_SESSIONS) {
      const oldest = this.#finalizedSessions.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.#finalizedSessions.delete(oldest);
    }
    this.#finalizedSessions.set(key, state);
  }

  #reduceFinalized(
    sessionID: string,
    state: FinalizedSessionState,
    eventID: string | undefined,
    fingerprint: string,
  ): ReducedOpenCodeRecord[] {
    if (eventID === undefined) return [];
    const key = digest({ eventID });
    const prior = state.events.get(key);
    if (prior === fingerprint) return [];
    if (prior !== undefined) {
      return [{ kind: "coverage_degraded", session_id: sessionID, reason: "invalid_transition" }];
    }
    if (state.events.size >= MAX_FINALIZED_EVENT_IDS) {
      return [{ kind: "coverage_degraded", session_id: sessionID, reason: "overflow" }];
    }
    state.events.set(key, fingerprint);
    return [];
  }

  #degradeFinalized(
    sessionID: string,
    reason: "invalid_transition" | "missing_field" | "overflow",
  ): ReducedOpenCodeRecord[] {
    return [{ kind: "coverage_degraded", session_id: sessionID, reason }];
  }

  isFinalized(sessionID: string): boolean {
    const checked = exactText(sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
    return checked !== undefined && this.#finalizedSession(checked) !== undefined;
  }

  #session(sessionID: string): SessionState | undefined {
    const key = this.#sessionKey(sessionID);
    const prior = this.#sessions.get(key);
    if (prior !== undefined) return prior;
    if (this.#sessions.size >= MAX_SESSIONS) return undefined;
    const retainedBytes =
      SESSION_STATE_OVERHEAD_BYTES + Buffer.byteLength(sessionID, "utf8");
    if (!this.#reserveState(retainedBytes)) return undefined;
    const overflowKey = this.#overflowKey(sessionID);
    const state: SessionState = {
      calls: new Map<string, CallState>(),
      events: new Map<string, EventState>(),
      disabled: false,
      deleted: false,
      recordCount: this.#overflowSessions.has(overflowKey) ? 1 : 0,
      overflowReported: false,
      retainedBytes,
    };
    this.#overflowSessions.delete(overflowKey);
    this.#sessions.set(key, state);
    return state;
  }

  #degrade(
    sessionID: string,
    reason: "invalid_transition" | "missing_field" | "overflow",
    disable = false,
  ): ReducedOpenCodeRecord[] {
    const session = this.#session(sessionID);
    if (session === undefined) return this.#sessionOverflow(sessionID);
    if (disable) session.disabled = true;
    return [{ kind: "coverage_degraded", session_id: sessionID, reason }];
  }

  #eventReplay(
    session: SessionState,
    eventID: string | undefined,
    fingerprint: string,
  ): "new" | "replay" | "conflict" | "overflow" {
    if (eventID === undefined) return "new";
    const key = digest({ eventID });
    const prior = session.events.get(key);
    if (prior === undefined) {
      if (session.events.size >= MAX_EVENT_IDS_PER_SESSION) return "overflow";
      const retainedBytes =
        EVENT_STATE_OVERHEAD_BYTES + Buffer.byteLength(eventID, "utf8");
      if (!this.#reserveState(retainedBytes)) return "overflow";
      session.events.set(key, { fingerprint, retainedBytes });
      return "new";
    }
    return prior.fingerprint === fingerprint ? "replay" : "conflict";
  }

  #reduceTool(event: OpenCodeEvent, properties: Record<string, unknown>): ReducedOpenCodeRecord[] {
    const part = properties.part;
    if (!isRecord(part)) return [];
    // This is the privacy boundary for all non-tool parts. No other property is
    // accessed before returning.
    if (part.type !== "tool") return [];

    const sessionID = exactText(part.sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
    if (sessionID === undefined) return [];
    const outerSessionID = properties.sessionID;
    const callID = exactText(part.callID, MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(part.tool, MAX_CAPTURE_TOOL_NAME_BYTES);
    const eventID = optionalEventID(event.id);
    const finalized = this.#finalizedSession(sessionID);
    const existing = this.#activeSession(sessionID);
    if (finalized === undefined && (existing?.disabled === true || existing?.deleted === true)) {
      return [];
    }
    if (
      outerSessionID !== undefined &&
      exactText(outerSessionID, MAX_CAPTURE_SESSION_ID_BYTES) !== sessionID
    ) {
      return finalized === undefined
        ? this.#degrade(sessionID, "missing_field", true)
        : this.#degradeFinalized(sessionID, "missing_field");
    }
    const state = part.state;
    if (callID === undefined || tool === undefined || !isRecord(state) || !("input" in state)) {
      return finalized === undefined
        ? this.#degrade(sessionID, "missing_field", true)
        : this.#degradeFinalized(sessionID, "missing_field");
    }
    const status = state.status;
    if (typeof status !== "string" || !TOOL_STATES.has(status as OpenCodeToolStatus)) {
      return finalized === undefined
        ? this.#degrade(sessionID, "missing_field", true)
        : this.#degradeFinalized(sessionID, "missing_field");
    }
    const checkedStatus = status as OpenCodeToolStatus;
    let canonicalInput: CanonicalJson | undefined;
    let identity = digest({ authority: "unavailable", callID, tool });
    try {
      canonicalInput = canonicalizeJson(state.input);
      identity = digest({ tool, input: canonicalInput });
    } catch (error) {
      if (!(error instanceof BridgeContractError)) throw error;
    }
    const fingerprint = digest({ sessionID, callID, tool, status: checkedStatus, identity });
    if (finalized !== undefined) {
      return this.#reduceFinalized(sessionID, finalized, eventID, fingerprint);
    }
    const session = existing ?? this.#session(sessionID);
    if (session === undefined) return this.#sessionOverflow(sessionID);
    const replay = this.#eventReplay(session, eventID, fingerprint);
    if (replay === "replay") return [];
    if (replay === "conflict") return this.#degrade(sessionID, "invalid_transition");
    if (replay === "overflow") return this.#degrade(sessionID, "overflow", true);

    const callKey = digest({ callID });
    const prior = session.calls.get(callKey);
    if (prior === undefined) {
      if (session.calls.size >= MAX_CALLS_PER_SESSION) {
        return this.#degrade(sessionID, "overflow", true);
      }
      const retainedBytes =
        CALL_STATE_OVERHEAD_BYTES +
        Buffer.byteLength(callID, "utf8") +
        Buffer.byteLength(tool, "utf8");
      if (!this.#reserveState(retainedBytes)) {
        return this.#degrade(sessionID, "overflow", true);
      }
      session.calls.set(callKey, { status: checkedStatus, identity, retainedBytes });
      const started: ReducedOpenCodeRecord = {
        kind: "tool_started",
        session_id: sessionID,
        ...(eventID === undefined ? {} : { event_id: eventID }),
        call_id: callID,
        tool,
        ...(canonicalInput === undefined ? {} : { input: canonicalInput }),
        identity_authority: canonicalInput === undefined ? "unavailable" : "exact",
      };
      if (!isTerminal(checkedStatus)) return [started];
      return [
        started,
        {
          kind: "tool_finished",
          session_id: sessionID,
          ...(eventID === undefined ? {} : { event_id: eventID }),
          call_id: callID,
          outcome: checkedStatus === "completed" ? "succeeded" : "failed",
        },
      ];
    }
    if (prior.identity !== identity) {
      return this.#degrade(sessionID, "invalid_transition");
    }
    if (
      rank(checkedStatus) < rank(prior.status) ||
      (isTerminal(prior.status) && checkedStatus !== prior.status)
    ) {
      return this.#degrade(sessionID, "invalid_transition");
    }
    if (checkedStatus === prior.status || (!isTerminal(checkedStatus) && rank(checkedStatus) > rank(prior.status))) {
      prior.status = checkedStatus;
      return [];
    }
    prior.status = checkedStatus;
    return [
      {
        kind: "tool_finished",
        session_id: sessionID,
        ...(eventID === undefined ? {} : { event_id: eventID }),
        call_id: callID,
        outcome: checkedStatus === "completed" ? "succeeded" : "failed",
      },
    ];
  }

  #boundedRecords(records: ReducedOpenCodeRecord[]): ReducedOpenCodeRecord[] {
    if (records.length === 0) return records;
    const grouped = new Map<string, ReducedOpenCodeRecord[]>();
    for (const record of records) {
      const values = grouped.get(record.session_id) ?? [];
      values.push(record);
      grouped.set(record.session_id, values);
    }
    const admitted: ReducedOpenCodeRecord[] = [];
    for (const [sessionID, values] of grouped) {
      const session = this.#activeSession(sessionID);
      if (session === undefined) {
        const finalized = this.#finalizedSession(sessionID);
        if (
          finalized !== undefined &&
          !finalized.degradationReported &&
          values.every((record) => record.kind === "coverage_degraded")
        ) {
          finalized.degradationReported = true;
          admitted.push(values[0]!);
        } else if (
          finalized === undefined &&
          this.#overflowSessions.has(this.#overflowKey(sessionID)) &&
          values.every((record) => record.kind === "coverage_degraded")
        ) {
          admitted.push(values[0]!);
        }
        continue;
      }
      const terminalOnly = values.every((record) => record.kind === "session_finished");
      if (session.overflowReported) {
        if (
          terminalOnly &&
          session.recordCount + values.length <= MAX_REDUCED_RECORDS_PER_SESSION
        ) {
          session.recordCount += values.length;
          admitted.push(...values);
        }
        continue;
      }
      const degradationOnly = values.every((record) => record.kind === "coverage_degraded");
      const limit = terminalOnly
        ? MAX_REDUCED_RECORDS_PER_SESSION
        : degradationOnly
          ? MAX_REDUCED_RECORDS_PER_SESSION - 1
          : MAX_REDUCED_RECORDS_PER_SESSION - 2;
      if (session.recordCount + values.length <= limit) {
        session.recordCount += values.length;
        admitted.push(...values);
        continue;
      }
      session.disabled = true;
      session.overflowReported = true;
      if (session.recordCount < MAX_REDUCED_RECORDS_PER_SESSION - 1) {
        session.recordCount += 1;
        admitted.push({
          kind: "coverage_degraded",
          session_id: sessionID,
          reason: "overflow",
        });
      }
    }
    return admitted;
  }

  #reduceValue(value: unknown): ReducedOpenCodeRecord[] {
    try {
      if (!isRecord(value)) return [];
      const event = value as OpenCodeEvent;
      const type = event.type;
      if (typeof type !== "string") return [];
      if (type === "message.part.updated") {
        const properties = event.properties;
        return isRecord(properties) ? this.#reduceTool(event, properties) : [];
      }
      if (type === "session.created" || type === "session.updated") return [];
      if (
        type !== "session.deleted" &&
        type !== "session.idle" &&
        type !== "session.error" &&
        type !== "session.compacted"
      ) {
        return [];
      }
      const properties = event.properties;
      if (!isRecord(properties)) return [];

      if (type === "session.deleted") {
        const eventID = optionalEventID(event.id);
        const info = properties.info;
        if (!isRecord(info)) return [];
        const sessionID = exactText(info.id, MAX_CAPTURE_SESSION_ID_BYTES);
        if (sessionID === undefined) return [];
        const fingerprint = digest({ sessionID, type });
        const finalized = this.#finalizedSession(sessionID);
        if (
          properties.sessionID !== undefined &&
          exactText(properties.sessionID, MAX_CAPTURE_SESSION_ID_BYTES) !== sessionID
        ) {
          return finalized === undefined
            ? this.#degrade(sessionID, "missing_field", true)
            : this.#degradeFinalized(sessionID, "missing_field");
        }
        if (finalized !== undefined) {
          return this.#reduceFinalized(sessionID, finalized, eventID, fingerprint);
        }
        const session = this.#activeSession(sessionID) ?? this.#session(sessionID);
        if (session === undefined) return this.#sessionOverflow(sessionID);
        if (session.deleted) return [];
        const replay = this.#eventReplay(session, eventID, fingerprint);
        if (replay === "replay") return [];
        if (replay === "conflict") return this.#degrade(sessionID, "invalid_transition");
        if (replay === "overflow") return this.#degrade(sessionID, "overflow", true);
        session.deleted = true;
        this.#rememberFinalized(sessionID, eventID, fingerprint);
        this.#releaseSession(session);
        this.#evictAfterReduction.add(this.#sessionKey(sessionID));
        return [
          {
            kind: "session_finished",
            session_id: sessionID,
            ...(eventID === undefined ? {} : { event_id: eventID }),
          },
        ];
      }

      if (type === "session.error" && properties.sessionID === undefined) {
        return [];
      }

      const sessionID = exactText(properties.sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
      if (sessionID === undefined) return [];
      const eventID = optionalEventID(event.id);
      const fingerprint = digest({ sessionID, type });
      const finalized = this.#finalizedSession(sessionID);
      if (finalized !== undefined) {
        return this.#reduceFinalized(sessionID, finalized, eventID, fingerprint);
      }
      const session = this.#activeSession(sessionID) ?? this.#session(sessionID);
      if (session === undefined) return this.#sessionOverflow(sessionID);
      if (session.disabled || session.deleted) return [];
      const common = {
        session_id: sessionID,
        ...(eventID === undefined ? {} : { event_id: eventID }),
      };
      let record: ReducedOpenCodeRecord | undefined;
      if (type === "session.idle") record = { kind: "turn_finished", ...common };
      if (type === "session.error") record = { kind: "controller_failed", ...common };
      if (type === "session.compacted") record = { kind: "coverage_boundary", ...common };
      if (record === undefined) return [];
      const replay = this.#eventReplay(session, eventID, fingerprint);
      if (replay === "replay") return [];
      if (replay === "conflict") return this.#degrade(sessionID, "invalid_transition");
      if (replay === "overflow") return this.#degrade(sessionID, "overflow", true);
      return [record];
    } catch {
      return [];
    }
  }

  reduce(value: unknown): ReducedOpenCodeRecord[] {
    try {
      return this.#boundedRecords(this.#reduceValue(value));
    } finally {
      for (const sessionKey of this.#evictAfterReduction) this.#sessions.delete(sessionKey);
      this.#evictAfterReduction.clear();
    }
  }

  dispose(): ReducedOpenCodeRecord[] {
    return [];
  }
}
