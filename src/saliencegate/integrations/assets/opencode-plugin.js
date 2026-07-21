export const saliencegateBootstrap = new URL("./saliencegate.bootstrap.json", import.meta.url);

// connectors/opencode/src/plugin.ts
import { Buffer as Buffer4 } from "node:buffer";

// connectors/bridge-core/src/canonical.ts
import { Buffer as Buffer2 } from "node:buffer";

// connectors/bridge-core/src/contracts.ts
var MAX_CAPTURE_BATCH_BYTES = 2 * 1024 * 1024;
var MAX_CAPTURE_EVENT_BYTES = 64 * 1024;
var MAX_CAPTURE_JSON_DEPTH = 32;
var MAX_CAPTURE_JSON_ITEMS = 1e4;
var MAX_CAPTURE_JSON_STRING_BYTES = 1024 * 1024;
var MAX_CAPTURE_BATCH_CHUNKS = 1e3;
var MAX_CAPTURE_EVENTS_PER_CHUNK = 999;
var MAX_CAPTURE_SESSION_ID_BYTES = 256 * 1024;
var MAX_CAPTURE_EVENT_ID_BYTES = 16 * 1024;
var MAX_CAPTURE_CALL_ID_BYTES = 16 * 1024;
var MAX_CAPTURE_TOOL_NAME_BYTES = 1024;
var BridgeContractError = class extends Error {
  constructor() {
    super("capture bridge contract is invalid");
    this.name = "BridgeContractError";
  }
};

// connectors/bridge-core/src/canonical.ts
function isWellFormedUnicode(value) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 55296 && code <= 56319) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 56320 && next <= 57343)) return false;
      index += 1;
    } else if (code >= 56320 && code <= 57343) {
      return false;
    }
  }
  return true;
}
function addString(value, budget) {
  if (!isWellFormedUnicode(value)) throw new BridgeContractError();
  budget.stringBytes += Buffer2.byteLength(value, "utf8");
  if (budget.stringBytes > MAX_CAPTURE_JSON_STRING_BYTES) throw new BridgeContractError();
}
function dataValue(container, key) {
  const descriptor = Object.getOwnPropertyDescriptor(container, key);
  if (descriptor === void 0 || !("value" in descriptor) || !descriptor.enumerable) {
    throw new BridgeContractError();
  }
  return descriptor.value;
}
function canonicalize(value, depth, budget) {
  budget.items += 1;
  if (budget.items > MAX_CAPTURE_JSON_ITEMS || depth > MAX_CAPTURE_JSON_DEPTH) {
    throw new BridgeContractError();
  }
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") {
    addString(value, budget);
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new BridgeContractError();
    return Object.is(value, -0) ? 0 : value;
  }
  if (typeof value !== "object") throw new BridgeContractError();
  if (budget.active.has(value)) throw new BridgeContractError();
  budget.active.add(value);
  try {
    if (Array.isArray(value)) {
      const keys2 = Object.keys(value);
      if (keys2.length !== value.length || keys2.some((key, index) => key !== String(index))) {
        throw new BridgeContractError();
      }
      return keys2.map((key) => canonicalize(dataValue(value, key), depth + 1, budget));
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) throw new BridgeContractError();
    const ownKeys = Reflect.ownKeys(value);
    if (ownKeys.some((key) => typeof key !== "string")) throw new BridgeContractError();
    const keys = ownKeys.sort();
    const result = {};
    for (const key of keys) {
      addString(key, budget);
      Object.defineProperty(result, key, {
        configurable: true,
        enumerable: true,
        value: canonicalize(dataValue(value, key), depth + 1, budget),
        writable: true
      });
    }
    return result;
  } finally {
    budget.active.delete(value);
  }
}
function canonicalizeJson(value) {
  try {
    return canonicalize(value, 0, { items: 0, stringBytes: 0, active: /* @__PURE__ */ new Set() });
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}
function encodeCanonicalJson(value) {
  try {
    return Buffer2.from(JSON.stringify(canonicalizeJson(value)), "utf8");
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

// connectors/bridge-core/src/chunks.ts
import { Buffer as Buffer3 } from "node:buffer";
var SHA256 = /^[0-9a-f]{64}$/;
var CONNECTION_ID = /^sg-[0-9a-f]{48}$/;
var WINDOWS_ABSOLUTE = /^[A-Za-z]:[\\/]/;
function validateBootstrap(value) {
  const keys = Object.keys(value).sort();
  const expected = [
    "bundle_digest",
    "capability_digest",
    "connection_id",
    "launcher_path",
    "profile",
    "receipt_mac",
    "schema_version"
  ];
  if (JSON.stringify(keys) !== JSON.stringify(expected) || value.schema_version !== "integration-bootstrap/v1" || !["opencode-plugin/v1", "pi-extension/v1"].includes(value.profile) || !CONNECTION_ID.test(value.connection_id) || !SHA256.test(value.capability_digest) || !SHA256.test(value.bundle_digest) || !SHA256.test(value.receipt_mac) || typeof value.launcher_path !== "string" || value.launcher_path.length === 0 || value.launcher_path.length > 4096 || value.launcher_path.includes("\0") || !(value.launcher_path.startsWith("/") || WINDOWS_ABSOLUTE.test(value.launcher_path))) {
    throw new BridgeContractError();
  }
  return canonicalizeJson(value);
}
function normalizeCaptureEvent(value, sessionID) {
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
function oversizeEvent(sessionID) {
  return { kind: "oversize", reason: "event_limit", session_id: sessionID };
}
function documentFor(input) {
  return {
    schema_version: "capture-batch/v1",
    bootstrap: input.bootstrap,
    batch_id: input.batchID,
    session_id: input.sessionID,
    chunk_index: input.chunkIndex,
    chunk_count: input.chunkCount,
    events: input.events
  };
}
function encodedSize(input) {
  try {
    return encodeCanonicalJson(
      documentFor({
        ...input,
        chunkCount: MAX_CAPTURE_BATCH_CHUNKS
      })
    ).byteLength;
  } catch (error) {
    if (error instanceof BridgeContractError) return MAX_CAPTURE_BATCH_BYTES + 1;
    throw error;
  }
}
function buildCaptureChunks(input) {
  try {
    const bootstrap = validateBootstrap(input.bootstrap);
    if (typeof input.batchID !== "string" || !SHA256.test(input.batchID) || typeof input.sessionID !== "string" || input.sessionID.length === 0 || Buffer3.byteLength(input.sessionID, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES || !Array.isArray(input.events) || input.events.length > 1e4) {
      throw new BridgeContractError();
    }
    canonicalizeJson(input.sessionID);
    const events = input.events.map((event) => {
      const normalized = normalizeCaptureEvent(event, input.sessionID);
      return encodedSize({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        chunkIndex: 0,
        events: [normalized]
      }) <= MAX_CAPTURE_BATCH_BYTES ? normalized : oversizeEvent(input.sessionID);
    });
    const groups = [];
    let current = [];
    for (const event of events) {
      const candidate = [...current, event];
      if (current.length > 0 && (current.length >= MAX_CAPTURE_EVENTS_PER_CHUNK || encodedSize({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        chunkIndex: groups.length,
        events: candidate
      }) > MAX_CAPTURE_BATCH_BYTES)) {
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
        chunkIndex: index,
        chunkCount: groups.length,
        events: group
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

// connectors/bridge-core/src/queue.ts
var SerialSessionQueue = class {
  #tails = /* @__PURE__ */ new Map();
  run(sessionID, operation) {
    const prior = this.#tails.get(sessionID) ?? Promise.resolve();
    const result = prior.catch(() => void 0).then(operation);
    const tail = result.then(
      () => void 0,
      () => void 0
    );
    this.#tails.set(sessionID, tail);
    void tail.finally(() => {
      if (this.#tails.get(sessionID) === tail) this.#tails.delete(sessionID);
    });
    return result;
  }
  async drain() {
    await Promise.all([...this.#tails.values()]);
  }
};

// connectors/opencode/src/bootstrap.ts
import { createHash, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
var MAX_BOOTSTRAP_BYTES = 16 * 1024;
var MAX_BUNDLE_BYTES = 2 * 1024 * 1024;
var BOOTSTRAP_NAME = "saliencegate.bootstrap.json";
var BUNDLE_NAME = "saliencegate.js";
var SHA2562 = /^[0-9a-f]{64}$/;
var CONNECTION_ID2 = /^sg-[0-9a-f]{48}$/;
var WINDOWS_ABSOLUTE2 = /^[A-Za-z]:[\\/]/;
function sameFile(first, second) {
  return first.dev === second.dev && first.ino === second.ino && first.mode === second.mode && first.size === second.size && first.mtimeMs === second.mtimeMs;
}
async function readStableRegularFile(filePath, input) {
  const before = await lstat(filePath);
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1 || before.size < input.minimum || before.size > input.maximum) {
    throw new BridgeContractError();
  }
  if (process.platform !== "win32" && ((before.mode & 63) !== 0 || typeof process.getuid === "function" && before.uid !== process.getuid())) {
    throw new BridgeContractError();
  }
  const noFollow = process.platform === "win32" ? 0 : constants.O_NOFOLLOW;
  const handle = await open(filePath, constants.O_RDONLY | noFollow);
  try {
    const opened = await handle.stat();
    if (!sameFile(before, opened) || !opened.isFile()) throw new BridgeContractError();
    const buffer = Buffer.allocUnsafe(input.maximum + 1);
    let offset = 0;
    while (offset < buffer.length) {
      const result = await handle.read(buffer, offset, buffer.length - offset, null);
      if (result.bytesRead === 0) break;
      offset += result.bytesRead;
    }
    const after = await handle.stat();
    if (!sameFile(opened, after) || offset !== opened.size || offset < input.minimum || offset > input.maximum) {
      throw new BridgeContractError();
    }
    return buffer.subarray(0, offset);
  } finally {
    await handle.close();
  }
}
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function validateBootstrap2(value) {
  if (!isRecord(value)) throw new BridgeContractError();
  const keys = Object.keys(value).sort();
  const expected = [
    "bundle_digest",
    "capability_digest",
    "connection_id",
    "launcher_path",
    "profile",
    "receipt_mac",
    "schema_version"
  ];
  const launcher = value.launcher_path;
  if (JSON.stringify(keys) !== JSON.stringify(expected) || value.schema_version !== "integration-bootstrap/v1" || value.profile !== "opencode-plugin/v1" || typeof value.connection_id !== "string" || !CONNECTION_ID2.test(value.connection_id) || typeof launcher !== "string" || launcher.length === 0 || launcher.length > 4096 || launcher.includes("\0") || !(launcher.startsWith("/") || WINDOWS_ABSOLUTE2.test(launcher)) || typeof value.capability_digest !== "string" || !SHA2562.test(value.capability_digest) || typeof value.bundle_digest !== "string" || !SHA2562.test(value.bundle_digest) || typeof value.receipt_mac !== "string" || !SHA2562.test(value.receipt_mac)) {
    throw new BridgeContractError();
  }
  return value;
}
async function loadOpenCodeBootstrap(bootstrapURL) {
  try {
    if (!(bootstrapURL instanceof URL) || bootstrapURL.protocol !== "file:" || bootstrapURL.search !== "" || bootstrapURL.hash !== "") {
      throw new BridgeContractError();
    }
    const bootstrapPath = fileURLToPath(bootstrapURL);
    if (path.basename(bootstrapPath) !== BOOTSTRAP_NAME) throw new BridgeContractError();
    const raw = await readStableRegularFile(bootstrapPath, {
      minimum: 2,
      maximum: MAX_BOOTSTRAP_BYTES
    });
    const parsed = JSON.parse(raw.toString("utf8"));
    const bootstrap = validateBootstrap2(parsed);
    const canonical = encodeCanonicalJson(bootstrap);
    if (canonical.length !== raw.length || !timingSafeEqual(canonical, raw)) {
      throw new BridgeContractError();
    }
    const bundlePath = path.join(path.dirname(bootstrapPath), BUNDLE_NAME);
    const bundle = await readStableRegularFile(bundlePath, {
      minimum: 1,
      maximum: MAX_BUNDLE_BYTES
    });
    const observedDigest = createHash("sha256").update(bundle).digest("hex");
    const expectedDigest = Buffer.from(bootstrap.bundle_digest, "ascii");
    const observedBytes = Buffer.from(observedDigest, "ascii");
    if (!timingSafeEqual(expectedDigest, observedBytes)) throw new BridgeContractError();
    return bootstrap;
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

// connectors/opencode/src/reducer.ts
import { createHash as createHash2 } from "node:crypto";
var MAX_SESSIONS = 256;
var MAX_CALLS_PER_SESSION = 1e3;
var MAX_EVENT_IDS_PER_SESSION = 4096;
var MAX_REDUCED_RECORDS_PER_SESSION = 997;
var MAX_FINALIZED_SESSIONS = 1024;
var MAX_FINALIZED_EVENT_IDS = 8;
var MAX_OVERFLOW_SESSION_MARKERS = 1024;
var MAX_REDUCER_STATE_BYTES = 2 * 1024 * 1024;
var SESSION_STATE_OVERHEAD_BYTES = 256;
var CALL_STATE_OVERHEAD_BYTES = 192;
var EVENT_STATE_OVERHEAD_BYTES = 160;
var TOOL_STATES = /* @__PURE__ */ new Set(["pending", "running", "completed", "error"]);
function isRecord2(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function exactText(value, maximumBytes) {
  if (typeof value !== "string" || value.length === 0 || !isWellFormedUnicode(value) || Buffer.byteLength(value, "utf8") > maximumBytes) {
    return void 0;
  }
  return value;
}
function optionalEventID(value) {
  return value === void 0 ? void 0 : exactText(value, MAX_CAPTURE_EVENT_ID_BYTES);
}
function digest(value) {
  return createHash2("sha256").update(encodeCanonicalJson(value)).digest("hex");
}
function rank(status) {
  if (status === "pending") return 0;
  if (status === "running") return 1;
  return 2;
}
function isTerminal(status) {
  return status === "completed" || status === "error";
}
var OpenCodeEventReducer = class {
  #sessions = /* @__PURE__ */ new Map();
  #finalizedSessions = /* @__PURE__ */ new Map();
  #overflowSessions = /* @__PURE__ */ new Map();
  #evictAfterReduction = /* @__PURE__ */ new Set();
  #retainedStateBytes = 0;
  #sessionKey(sessionID) {
    return digest({ kind: "active_session", sessionID });
  }
  #activeSession(sessionID) {
    return this.#sessions.get(this.#sessionKey(sessionID));
  }
  #reserveState(bytes) {
    if (this.#retainedStateBytes + bytes > MAX_REDUCER_STATE_BYTES) return false;
    this.#retainedStateBytes += bytes;
    return true;
  }
  #releaseState(bytes) {
    this.#retainedStateBytes = Math.max(0, this.#retainedStateBytes - bytes);
  }
  #releaseSession(session) {
    let bytes = session.retainedBytes;
    for (const call of session.calls.values()) bytes += call.retainedBytes;
    for (const event of session.events.values()) bytes += event.retainedBytes;
    this.#releaseState(bytes);
    session.calls.clear();
    session.events.clear();
  }
  #finalizedKey(sessionID) {
    return digest({ kind: "finalized_session", sessionID });
  }
  #overflowKey(sessionID) {
    return digest({ kind: "session_table_overflow", sessionID });
  }
  #sessionOverflow(sessionID) {
    const key = this.#overflowKey(sessionID);
    if (this.#overflowSessions.has(key)) {
      this.#overflowSessions.delete(key);
      this.#overflowSessions.set(key, true);
      return [];
    }
    while (this.#overflowSessions.size >= MAX_OVERFLOW_SESSION_MARKERS) {
      const oldest = this.#overflowSessions.keys().next().value;
      if (oldest === void 0) break;
      this.#overflowSessions.delete(oldest);
    }
    this.#overflowSessions.set(key, true);
    return [{ kind: "coverage_degraded", session_id: sessionID, reason: "overflow" }];
  }
  #finalizedSession(sessionID) {
    const key = this.#finalizedKey(sessionID);
    const state = this.#finalizedSessions.get(key);
    if (state === void 0) return void 0;
    this.#finalizedSessions.delete(key);
    this.#finalizedSessions.set(key, state);
    return state;
  }
  #rememberFinalized(sessionID, eventID, fingerprint) {
    const key = this.#finalizedKey(sessionID);
    const state = {
      events: /* @__PURE__ */ new Map(),
      degradationReported: false
    };
    if (eventID !== void 0) state.events.set(digest({ eventID }), fingerprint);
    this.#finalizedSessions.delete(key);
    while (this.#finalizedSessions.size >= MAX_FINALIZED_SESSIONS) {
      const oldest = this.#finalizedSessions.keys().next().value;
      if (oldest === void 0) break;
      this.#finalizedSessions.delete(oldest);
    }
    this.#finalizedSessions.set(key, state);
  }
  #reduceFinalized(sessionID, state, eventID, fingerprint) {
    if (eventID === void 0) return [];
    const key = digest({ eventID });
    const prior = state.events.get(key);
    if (prior === fingerprint) return [];
    if (prior !== void 0) {
      return [{ kind: "coverage_degraded", session_id: sessionID, reason: "invalid_transition" }];
    }
    if (state.events.size >= MAX_FINALIZED_EVENT_IDS) {
      return [{ kind: "coverage_degraded", session_id: sessionID, reason: "overflow" }];
    }
    state.events.set(key, fingerprint);
    return [];
  }
  #degradeFinalized(sessionID, reason) {
    return [{ kind: "coverage_degraded", session_id: sessionID, reason }];
  }
  isFinalized(sessionID) {
    const checked = exactText(sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
    return checked !== void 0 && this.#finalizedSession(checked) !== void 0;
  }
  #session(sessionID) {
    const key = this.#sessionKey(sessionID);
    const prior = this.#sessions.get(key);
    if (prior !== void 0) return prior;
    if (this.#sessions.size >= MAX_SESSIONS) return void 0;
    const retainedBytes = SESSION_STATE_OVERHEAD_BYTES + Buffer.byteLength(sessionID, "utf8");
    if (!this.#reserveState(retainedBytes)) return void 0;
    const overflowKey = this.#overflowKey(sessionID);
    const state = {
      calls: /* @__PURE__ */ new Map(),
      events: /* @__PURE__ */ new Map(),
      disabled: false,
      deleted: false,
      recordCount: this.#overflowSessions.has(overflowKey) ? 1 : 0,
      overflowReported: false,
      retainedBytes
    };
    this.#overflowSessions.delete(overflowKey);
    this.#sessions.set(key, state);
    return state;
  }
  #degrade(sessionID, reason, disable = false) {
    const session = this.#session(sessionID);
    if (session === void 0) return this.#sessionOverflow(sessionID);
    if (disable) session.disabled = true;
    return [{ kind: "coverage_degraded", session_id: sessionID, reason }];
  }
  #eventReplay(session, eventID, fingerprint) {
    if (eventID === void 0) return "new";
    const key = digest({ eventID });
    const prior = session.events.get(key);
    if (prior === void 0) {
      if (session.events.size >= MAX_EVENT_IDS_PER_SESSION) return "overflow";
      const retainedBytes = EVENT_STATE_OVERHEAD_BYTES + Buffer.byteLength(eventID, "utf8");
      if (!this.#reserveState(retainedBytes)) return "overflow";
      session.events.set(key, { fingerprint, retainedBytes });
      return "new";
    }
    return prior.fingerprint === fingerprint ? "replay" : "conflict";
  }
  #reduceTool(event, properties) {
    const part = properties.part;
    if (!isRecord2(part)) return [];
    if (part.type !== "tool") return [];
    const sessionID = exactText(part.sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
    if (sessionID === void 0) return [];
    const outerSessionID = properties.sessionID;
    const callID = exactText(part.callID, MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(part.tool, MAX_CAPTURE_TOOL_NAME_BYTES);
    const eventID = optionalEventID(event.id);
    const finalized = this.#finalizedSession(sessionID);
    const existing = this.#activeSession(sessionID);
    if (finalized === void 0 && (existing?.disabled === true || existing?.deleted === true)) {
      return [];
    }
    if (outerSessionID !== void 0 && exactText(outerSessionID, MAX_CAPTURE_SESSION_ID_BYTES) !== sessionID) {
      return finalized === void 0 ? this.#degrade(sessionID, "missing_field", true) : this.#degradeFinalized(sessionID, "missing_field");
    }
    const state = part.state;
    if (callID === void 0 || tool === void 0 || !isRecord2(state) || !("input" in state)) {
      return finalized === void 0 ? this.#degrade(sessionID, "missing_field", true) : this.#degradeFinalized(sessionID, "missing_field");
    }
    const status = state.status;
    if (typeof status !== "string" || !TOOL_STATES.has(status)) {
      return finalized === void 0 ? this.#degrade(sessionID, "missing_field", true) : this.#degradeFinalized(sessionID, "missing_field");
    }
    const checkedStatus = status;
    let canonicalInput;
    let identity = digest({ authority: "unavailable", callID, tool });
    try {
      canonicalInput = canonicalizeJson(state.input);
      identity = digest({ tool, input: canonicalInput });
    } catch (error) {
      if (!(error instanceof BridgeContractError)) throw error;
    }
    const fingerprint = digest({ sessionID, callID, tool, status: checkedStatus, identity });
    if (finalized !== void 0) {
      return this.#reduceFinalized(sessionID, finalized, eventID, fingerprint);
    }
    const session = existing ?? this.#session(sessionID);
    if (session === void 0) return this.#sessionOverflow(sessionID);
    const replay = this.#eventReplay(session, eventID, fingerprint);
    if (replay === "replay") return [];
    if (replay === "conflict") return this.#degrade(sessionID, "invalid_transition");
    if (replay === "overflow") return this.#degrade(sessionID, "overflow", true);
    const callKey = digest({ callID });
    const prior = session.calls.get(callKey);
    if (prior === void 0) {
      if (session.calls.size >= MAX_CALLS_PER_SESSION) {
        return this.#degrade(sessionID, "overflow", true);
      }
      const retainedBytes = CALL_STATE_OVERHEAD_BYTES + Buffer.byteLength(callID, "utf8") + Buffer.byteLength(tool, "utf8");
      if (!this.#reserveState(retainedBytes)) {
        return this.#degrade(sessionID, "overflow", true);
      }
      session.calls.set(callKey, { status: checkedStatus, identity, retainedBytes });
      const started = {
        kind: "tool_started",
        session_id: sessionID,
        ...eventID === void 0 ? {} : { event_id: eventID },
        call_id: callID,
        tool,
        ...canonicalInput === void 0 ? {} : { input: canonicalInput },
        identity_authority: canonicalInput === void 0 ? "unavailable" : "exact"
      };
      if (!isTerminal(checkedStatus)) return [started];
      return [
        started,
        {
          kind: "tool_finished",
          session_id: sessionID,
          ...eventID === void 0 ? {} : { event_id: eventID },
          call_id: callID,
          outcome: checkedStatus === "completed" ? "succeeded" : "failed"
        }
      ];
    }
    if (prior.identity !== identity) {
      return this.#degrade(sessionID, "invalid_transition");
    }
    if (rank(checkedStatus) < rank(prior.status) || isTerminal(prior.status) && checkedStatus !== prior.status) {
      return this.#degrade(sessionID, "invalid_transition");
    }
    if (checkedStatus === prior.status || !isTerminal(checkedStatus) && rank(checkedStatus) > rank(prior.status)) {
      prior.status = checkedStatus;
      return [];
    }
    prior.status = checkedStatus;
    return [
      {
        kind: "tool_finished",
        session_id: sessionID,
        ...eventID === void 0 ? {} : { event_id: eventID },
        call_id: callID,
        outcome: checkedStatus === "completed" ? "succeeded" : "failed"
      }
    ];
  }
  #boundedRecords(records) {
    if (records.length === 0) return records;
    const grouped = /* @__PURE__ */ new Map();
    for (const record of records) {
      const values = grouped.get(record.session_id) ?? [];
      values.push(record);
      grouped.set(record.session_id, values);
    }
    const admitted = [];
    for (const [sessionID, values] of grouped) {
      const session = this.#activeSession(sessionID);
      if (session === void 0) {
        const finalized = this.#finalizedSession(sessionID);
        if (finalized !== void 0 && !finalized.degradationReported && values.every((record) => record.kind === "coverage_degraded")) {
          finalized.degradationReported = true;
          admitted.push(values[0]);
        } else if (finalized === void 0 && this.#overflowSessions.has(this.#overflowKey(sessionID)) && values.every((record) => record.kind === "coverage_degraded")) {
          admitted.push(values[0]);
        }
        continue;
      }
      const terminalOnly = values.every((record) => record.kind === "session_finished");
      if (session.overflowReported) {
        if (terminalOnly && session.recordCount + values.length <= MAX_REDUCED_RECORDS_PER_SESSION) {
          session.recordCount += values.length;
          admitted.push(...values);
        }
        continue;
      }
      const degradationOnly = values.every((record) => record.kind === "coverage_degraded");
      const limit = terminalOnly ? MAX_REDUCED_RECORDS_PER_SESSION : degradationOnly ? MAX_REDUCED_RECORDS_PER_SESSION - 1 : MAX_REDUCED_RECORDS_PER_SESSION - 2;
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
          reason: "overflow"
        });
      }
    }
    return admitted;
  }
  #reduceValue(value) {
    try {
      if (!isRecord2(value)) return [];
      const event = value;
      const type = event.type;
      if (typeof type !== "string") return [];
      if (type === "message.part.updated") {
        const properties2 = event.properties;
        return isRecord2(properties2) ? this.#reduceTool(event, properties2) : [];
      }
      if (type === "session.created" || type === "session.updated") return [];
      if (type !== "session.deleted" && type !== "session.idle" && type !== "session.error" && type !== "session.compacted") {
        return [];
      }
      const properties = event.properties;
      if (!isRecord2(properties)) return [];
      if (type === "session.deleted") {
        const eventID2 = optionalEventID(event.id);
        const info = properties.info;
        if (!isRecord2(info)) return [];
        const sessionID2 = exactText(info.id, MAX_CAPTURE_SESSION_ID_BYTES);
        if (sessionID2 === void 0) return [];
        const fingerprint2 = digest({ sessionID: sessionID2, type });
        const finalized2 = this.#finalizedSession(sessionID2);
        if (properties.sessionID !== void 0 && exactText(properties.sessionID, MAX_CAPTURE_SESSION_ID_BYTES) !== sessionID2) {
          return finalized2 === void 0 ? this.#degrade(sessionID2, "missing_field", true) : this.#degradeFinalized(sessionID2, "missing_field");
        }
        if (finalized2 !== void 0) {
          return this.#reduceFinalized(sessionID2, finalized2, eventID2, fingerprint2);
        }
        const session2 = this.#activeSession(sessionID2) ?? this.#session(sessionID2);
        if (session2 === void 0) return this.#sessionOverflow(sessionID2);
        if (session2.deleted) return [];
        const replay2 = this.#eventReplay(session2, eventID2, fingerprint2);
        if (replay2 === "replay") return [];
        if (replay2 === "conflict") return this.#degrade(sessionID2, "invalid_transition");
        if (replay2 === "overflow") return this.#degrade(sessionID2, "overflow", true);
        session2.deleted = true;
        this.#rememberFinalized(sessionID2, eventID2, fingerprint2);
        this.#releaseSession(session2);
        this.#evictAfterReduction.add(this.#sessionKey(sessionID2));
        return [
          {
            kind: "session_finished",
            session_id: sessionID2,
            ...eventID2 === void 0 ? {} : { event_id: eventID2 }
          }
        ];
      }
      if (type === "session.error" && properties.sessionID === void 0) {
        return [];
      }
      const sessionID = exactText(properties.sessionID, MAX_CAPTURE_SESSION_ID_BYTES);
      if (sessionID === void 0) return [];
      const eventID = optionalEventID(event.id);
      const fingerprint = digest({ sessionID, type });
      const finalized = this.#finalizedSession(sessionID);
      if (finalized !== void 0) {
        return this.#reduceFinalized(sessionID, finalized, eventID, fingerprint);
      }
      const session = this.#activeSession(sessionID) ?? this.#session(sessionID);
      if (session === void 0) return this.#sessionOverflow(sessionID);
      if (session.disabled || session.deleted) return [];
      const common = {
        session_id: sessionID,
        ...eventID === void 0 ? {} : { event_id: eventID }
      };
      let record;
      if (type === "session.idle") record = { kind: "turn_finished", ...common };
      if (type === "session.error") record = { kind: "controller_failed", ...common };
      if (type === "session.compacted") record = { kind: "coverage_boundary", ...common };
      if (record === void 0) return [];
      const replay = this.#eventReplay(session, eventID, fingerprint);
      if (replay === "replay") return [];
      if (replay === "conflict") return this.#degrade(sessionID, "invalid_transition");
      if (replay === "overflow") return this.#degrade(sessionID, "overflow", true);
      return [record];
    } catch {
      return [];
    }
  }
  reduce(value) {
    try {
      return this.#boundedRecords(this.#reduceValue(value));
    } finally {
      for (const sessionKey of this.#evictAfterReduction) this.#sessions.delete(sessionKey);
      this.#evictAfterReduction.clear();
    }
  }
  dispose() {
    return [];
  }
};

// connectors/opencode/src/transport.ts
import { randomBytes } from "node:crypto";
import { spawn } from "node:child_process";

// connectors/opencode/src/launcher.ts
import path2 from "node:path";
function checkedEnvironment(value) {
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (key.includes("\0") || item.includes("\0")) throw new BridgeContractError();
    result[key] = item;
  }
  return result;
}
function launcherInvocation2(input) {
  try {
    if (typeof input.launcherPath !== "string" || input.launcherPath.length === 0 || input.launcherPath.length > 4096 || input.launcherPath.includes("\0")) {
      throw new BridgeContractError();
    }
    const environment2 = checkedEnvironment(input.environment);
    const options = {
      shell: false,
      windowsHide: true,
      env: environment2,
      stdio: ["pipe", "ignore", "ignore"]
    };
    if (input.platform !== "win32") {
      if (!path2.posix.isAbsolute(input.launcherPath)) throw new BridgeContractError();
      return { file: input.launcherPath, arguments: [], options };
    }
    if (!path2.win32.isAbsolute(input.launcherPath) || input.launcherPath.includes('"')) {
      throw new BridgeContractError();
    }
    const systemRoots = Object.entries(environment2).filter(
      ([key]) => key.toUpperCase() === "SYSTEMROOT"
    );
    if (systemRoots.length !== 1) throw new BridgeContractError();
    const systemRoot = systemRoots[0][1];
    if (typeof systemRoot !== "string" || !/^[A-Za-z]:\\Windows$/i.test(systemRoot) || systemRoot.includes('"')) {
      throw new BridgeContractError();
    }
    const file = path2.win32.join(systemRoot, "System32", "cmd.exe");
    const windowsEnvironment = Object.fromEntries(
      Object.entries(environment2).filter(
        ([key]) => key.toUpperCase() !== "SALIENCEGATE_LAUNCHER"
      )
    );
    return {
      file,
      arguments: ["/d", "/v:off", "/s", "/c", '"%SALIENCEGATE_LAUNCHER%"'],
      options: {
        ...options,
        env: { ...windowsEnvironment, SALIENCEGATE_LAUNCHER: input.launcherPath }
      }
    };
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

// connectors/opencode/src/transport.ts
var CAPTURE_LAUNCHER_TIMEOUT_MS = 2e3;
var MAX_CONCURRENT_CAPTURE_LAUNCHERS = 4;
var MAX_PENDING_GAP_SESSIONS = 256;
function environment(value) {
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "string") result[key] = item;
  }
  return result;
}
async function spawnCaptureChunk(input, spawnChild = spawn) {
  return await new Promise((resolve) => {
    let child;
    try {
      child = spawnChild(input.invocation.file, input.invocation.arguments, input.invocation.options);
    } catch {
      resolve(false);
      return;
    }
    let settled = false;
    let timedOut = false;
    let stdinFailed = false;
    const finish = (result) => {
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
function hasMatchingSession(value, sessionID) {
  return typeof value === "object" && value !== null && !Array.isArray(value) && value.session_id === sessionID;
}
var OpenCodeBatchTransport = class {
  #bootstrap;
  #invocation;
  #writeChunk;
  #batchID;
  #pendingGaps = /* @__PURE__ */ new Map();
  #gapGeneration = 0n;
  #activeWrites = 0;
  constructor(bootstrap, options = {}) {
    this.#bootstrap = bootstrap;
    this.#invocation = launcherInvocation2({
      platform: options.platform ?? process.platform,
      launcherPath: bootstrap.launcher_path,
      environment: environment(options.environment ?? process.env)
    });
    this.#writeChunk = options.writeChunk ?? spawnCaptureChunk;
    this.#batchID = options.batchID ?? (() => randomBytes(32).toString("hex"));
  }
  pendingSessionIDs() {
    return [...this.#pendingGaps.keys()];
  }
  hasPendingGap(sessionID) {
    return typeof sessionID === "string" && sessionID.length > 0 && isWellFormedUnicode(sessionID) && Buffer.byteLength(sessionID, "utf8") <= MAX_CAPTURE_SESSION_ID_BYTES && this.#pendingGaps.has(sessionID);
  }
  markGap(sessionID) {
    if (typeof sessionID !== "string" || sessionID.length === 0 || !isWellFormedUnicode(sessionID) || Buffer.byteLength(sessionID, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES) {
      return;
    }
    if (this.#pendingGaps.has(sessionID) || this.#pendingGaps.size < MAX_PENDING_GAP_SESSIONS) {
      this.#gapGeneration += 1n;
      this.#pendingGaps.set(sessionID, this.#gapGeneration);
    }
  }
  #acquireWritePermit() {
    if (this.#activeWrites >= MAX_CONCURRENT_CAPTURE_LAUNCHERS) return void 0;
    this.#activeWrites += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#activeWrites -= 1;
    };
  }
  async #writeBounded(write) {
    const release = this.#acquireWritePermit();
    if (release === void 0) return "not_started";
    try {
      return await this.#writeChunk(write) ? "delivered" : "failed";
    } catch {
      return "failed";
    } finally {
      release();
    }
  }
  async flush(sessionID, records) {
    try {
      if (!records.every((record) => hasMatchingSession(record, sessionID))) {
        this.markGap(sessionID);
        return "attempted_failure";
      }
      const pendingGapGeneration = this.#pendingGaps.get(sessionID);
      const events = [
        ...pendingGapGeneration !== void 0 ? [
          {
            kind: "coverage_degraded",
            reason: "transport_gap",
            session_id: sessionID
          }
        ] : [],
        ...records
      ];
      if (events.length === 0) return "delivered";
      const chunks = buildCaptureChunks({
        bootstrap: this.#bootstrap,
        batchID: this.#batchID(),
        sessionID,
        events
      });
      let delivered = 0;
      let started = 0;
      for (const chunk of chunks) {
        const result = await this.#writeBounded({
          invocation: this.#invocation,
          bytes: chunk.bytes,
          timeoutMS: CAPTURE_LAUNCHER_TIMEOUT_MS
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
};

// connectors/opencode/src/plugin.ts
var MAX_SESSION_BUFFER_BYTES = 512 * 1024;
var MAX_TOTAL_RETAINED_BYTES = 16 * 1024 * 1024;
var MAX_PENDING_FLUSHES_PER_SESSION = 1;
var MAX_PENDING_FLUSH_SESSIONS = 64;
var MAX_POST_PENDING_GAP_CHECKS = 256;
var MAX_DEFERRED_TERMINAL_SESSIONS = 256;
var MAX_TERMINAL_RESERVE_BYTES = MAX_DEFERRED_TERMINAL_SESSIONS * MAX_CAPTURE_EVENT_BYTES;
var MAX_DISPOSE_FLUSH_PASSES = 4;
function isRecord3(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function exactText2(value) {
  if (typeof value !== "string" || value.length === 0 || !isWellFormedUnicode(value) || Buffer4.byteLength(value, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES) {
    return void 0;
  }
  return value;
}
function flushTargets(value) {
  const result = /* @__PURE__ */ new Set();
  if (!isRecord3(value) || typeof value.type !== "string") return result;
  const properties = value.properties;
  if (!isRecord3(properties)) return result;
  if (value.type === "session.error" && properties.sessionID === void 0) return result;
  if (value.type === "session.deleted") {
    const info = properties.info;
    const sessionID = isRecord3(info) ? exactText2(info.id) : void 0;
    if (sessionID !== void 0) result.add(sessionID);
    return result;
  }
  if (value.type === "session.idle" || value.type === "session.error" || value.type === "session.compacted") {
    const sessionID = exactText2(properties.sessionID);
    if (sessionID !== void 0) result.add(sessionID);
  }
  return result;
}
function asCanonicalRecord(record) {
  return normalizeCaptureEvent(record, record.session_id);
}
function terminalControl(records) {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (isRecord3(record) && record.kind === "session_finished") return record;
  }
  return void 0;
}
var OpenCodePluginRuntime = class {
  #reducer = new OpenCodeEventReducer();
  #queue = new SerialSessionQueue();
  #transport;
  #buffers = /* @__PURE__ */ new Map();
  #pendingFlushes = /* @__PURE__ */ new Map();
  #postPendingGapChecks = /* @__PURE__ */ new Set();
  #deferredTerminalSessions = /* @__PURE__ */ new Set();
  #retainedBytes = 0;
  #disposed = false;
  constructor(bootstrap, options) {
    this.#transport = new OpenCodeBatchTransport(bootstrap, options);
  }
  #knownSessions() {
    return [
      .../* @__PURE__ */ new Set([
        ...this.#buffers.keys(),
        ...this.#pendingFlushes.keys(),
        ...this.#postPendingGapChecks,
        ...this.#deferredTerminalSessions,
        ...this.#transport.pendingSessionIDs()
      ])
    ];
  }
  #retainOnlyTerminal(sessionID, buffer) {
    const terminal = terminalControl(buffer.records);
    this.#buffers.delete(sessionID);
    this.#retainedBytes -= buffer.bytes;
    if (terminal === void 0) return void 0;
    const bytes = encodeCanonicalJson(terminal).byteLength;
    this.#buffers.set(sessionID, { records: [terminal], bytes });
    this.#retainedBytes += bytes;
    return terminal;
  }
  #restoreTerminal(sessionID, terminal) {
    let buffer = this.#buffers.get(sessionID);
    if (buffer !== void 0 && terminalControl(buffer.records) !== void 0) return true;
    const bytes = encodeCanonicalJson(terminal).byteLength;
    if (buffer !== void 0 && buffer.records.length > 0 && buffer.bytes + bytes > MAX_SESSION_BUFFER_BYTES) {
      this.#buffers.delete(sessionID);
      this.#retainedBytes -= buffer.bytes;
      this.#transport.markGap(sessionID);
      buffer = void 0;
    }
    if (this.#retainedBytes + bytes > MAX_TOTAL_RETAINED_BYTES + MAX_TERMINAL_RESERVE_BYTES) {
      this.#transport.markGap(sessionID);
      return false;
    }
    if (buffer === void 0) {
      buffer = { records: [], bytes: 0 };
      this.#buffers.set(sessionID, buffer);
    }
    buffer.records.push(terminal);
    buffer.bytes += bytes;
    this.#retainedBytes += bytes;
    return true;
  }
  #deferTerminal(sessionID) {
    if (this.#deferredTerminalSessions.has(sessionID) || this.#deferredTerminalSessions.size < MAX_DEFERRED_TERMINAL_SESSIONS) {
      this.#deferredTerminalSessions.add(sessionID);
    }
  }
  async #scheduleOneDeferredTerminal() {
    for (const sessionID of this.#deferredTerminalSessions) {
      if (this.#pendingFlushes.has(sessionID)) continue;
      this.#deferredTerminalSessions.delete(sessionID);
      await this.#scheduleFlush(sessionID, true);
      return;
    }
    for (const [sessionID, buffer] of this.#buffers) {
      if (this.#pendingFlushes.has(sessionID) || terminalControl(buffer.records) === void 0) {
        continue;
      }
      await this.#scheduleFlush(sessionID, true);
      return;
    }
  }
  #scheduleFlush(sessionID, force = false) {
    const buffer = this.#buffers.get(sessionID);
    const records = buffer?.records ?? [];
    const bufferBytes = buffer?.bytes ?? 0;
    const hasRecords = records.length > 0;
    if (!hasRecords && !force) return Promise.resolve();
    if (!this.#pendingFlushes.has(sessionID) && this.#pendingFlushes.size >= MAX_PENDING_FLUSH_SESSIONS) {
      if (hasRecords) {
        const terminal2 = this.#retainOnlyTerminal(sessionID, buffer);
        this.#transport.markGap(sessionID);
        if (terminal2 !== void 0) this.#deferTerminal(sessionID);
      }
      return Promise.resolve();
    }
    const pending = this.#pendingFlushes.get(sessionID) ?? 0;
    if (pending >= MAX_PENDING_FLUSHES_PER_SESSION) {
      let terminal2;
      if (hasRecords) {
        terminal2 = this.#retainOnlyTerminal(sessionID, buffer);
        this.#transport.markGap(sessionID);
      }
      if (!hasRecords || terminal2 !== void 0) {
        if (this.#postPendingGapChecks.has(sessionID) || this.#postPendingGapChecks.size < MAX_POST_PENDING_GAP_CHECKS) {
          this.#postPendingGapChecks.add(sessionID);
        } else if (terminal2 !== void 0) {
          this.#deferTerminal(sessionID);
        }
      }
      return Promise.resolve();
    }
    const terminal = terminalControl(records);
    if (hasRecords) this.#buffers.delete(sessionID);
    this.#pendingFlushes.set(sessionID, pending + 1);
    let flushResult = "attempted_failure";
    return this.#queue.run(sessionID, async () => {
      flushResult = await this.#transport.flush(sessionID, records);
    }).finally(async () => {
      if (hasRecords) {
        this.#retainedBytes -= bufferBytes;
        if (terminal !== void 0 && flushResult === "not_started") {
          if (this.#restoreTerminal(sessionID, terminal)) {
            this.#deferTerminal(sessionID);
          }
        }
      }
      const current = this.#pendingFlushes.get(sessionID);
      if (current === void 0 || current <= 1) {
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
  #appendRecords(sessionID, records, scheduled) {
    for (const record of records) {
      const bytes = encodeCanonicalJson(record).byteLength;
      const isTerminal2 = isRecord3(record) && record.kind === "session_finished";
      let buffer = this.#buffers.get(sessionID);
      if (buffer !== void 0 && buffer.records.length > 0 && buffer.bytes + bytes > MAX_SESSION_BUFFER_BYTES) {
        scheduled.push(this.#scheduleFlush(sessionID));
        buffer = void 0;
      }
      if (this.#retainedBytes + bytes > (isTerminal2 ? MAX_TOTAL_RETAINED_BYTES + MAX_TERMINAL_RESERVE_BYTES : MAX_TOTAL_RETAINED_BYTES)) {
        this.#transport.markGap(sessionID);
        continue;
      }
      if (isTerminal2 && this.#retainedBytes + bytes > MAX_TOTAL_RETAINED_BYTES) {
        this.#transport.markGap(sessionID);
      }
      if (buffer === void 0) {
        buffer = { records: [], bytes: 0 };
        this.#buffers.set(sessionID, buffer);
      }
      buffer.records.push(record);
      buffer.bytes += bytes;
      this.#retainedBytes += bytes;
    }
  }
  async event(value) {
    if (this.#disposed) return;
    try {
      const records = this.#reducer.reduce(value);
      const grouped = /* @__PURE__ */ new Map();
      const terminalSessions = /* @__PURE__ */ new Set();
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
      const sessions = /* @__PURE__ */ new Set([...grouped.keys(), ...targets]);
      const scheduled = [];
      for (const sessionID of sessions) {
        const reduced = grouped.get(sessionID) ?? [];
        this.#appendRecords(sessionID, reduced, scheduled);
        const buffered = this.#buffers.get(sessionID);
        if (targets.has(sessionID) && (reduced.length > 0 || buffered !== void 0 && buffered.records.length > 0 || this.#pendingFlushes.has(sessionID) || this.#transport.hasPendingGap(sessionID))) {
          scheduled.push(this.#scheduleFlush(sessionID, true));
        }
      }
      grouped.clear();
      records.length = 0;
      await Promise.all(scheduled);
    } catch {
    }
  }
  async dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    try {
      await this.#queue.drain();
      for (const record of this.#reducer.dispose()) {
        const scheduled = [];
        this.#appendRecords(record.session_id, [asCanonicalRecord(record)], scheduled);
        await Promise.all(scheduled);
      }
      for (let pass = 0; pass < MAX_DISPOSE_FLUSH_PASSES; pass += 1) {
        const sessions = this.#knownSessions();
        if (sessions.length === 0) break;
        await Promise.all(
          sessions.map((sessionID) => this.#scheduleFlush(sessionID, true))
        );
        await this.#queue.drain();
      }
    } catch {
    }
  }
};
function createOpenCodePlugin(options) {
  return async (_input) => {
    let runtime;
    try {
      const loader = options.loadBootstrap ?? loadOpenCodeBootstrap;
      const bootstrap = await loader(options.bootstrapURL);
      runtime = new OpenCodePluginRuntime(bootstrap, options);
    } catch {
      runtime = void 0;
    }
    return {
      event: async (input) => {
        try {
          await runtime?.event(input.event);
        } catch {
        }
      },
      dispose: async () => {
        try {
          await runtime?.dispose();
        } catch {
        }
      }
    };
  };
}

// opencode-runtime-entry.ts
var plugin = {
  id: "saliencegate",
  server: createOpenCodePlugin({ bootstrapURL: saliencegateBootstrap })
};
var opencode_runtime_entry_default = plugin;
export {
  opencode_runtime_entry_default as default
};
