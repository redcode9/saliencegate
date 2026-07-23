export const saliencegateBootstrap = new URL("./saliencegate.bootstrap.json", import.meta.url);

// connectors/pi/src/extension.ts
import { randomBytes as randomBytes2 } from "node:crypto";

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

// connectors/bridge-core/src/launcher.ts
import path from "node:path";
var PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS = Object.freeze([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID"
]);
var PROVIDER_CREDENTIAL_ENVIRONMENT_KEY_SET = new Set(
  PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS
);
function copyLauncherEnvironment(value) {
  try {
    const result = {};
    for (const key of Object.keys(value)) {
      if (key.includes("\0")) throw new BridgeContractError();
      if (PROVIDER_CREDENTIAL_ENVIRONMENT_KEY_SET.has(key.toUpperCase())) continue;
      const item = Reflect.get(value, key);
      if (typeof item !== "string") continue;
      if (item.includes("\0")) throw new BridgeContractError();
      Object.defineProperty(result, key, {
        configurable: true,
        enumerable: true,
        value: item,
        writable: true
      });
    }
    return result;
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}
function launcherInvocation(input) {
  try {
    if (typeof input.launcherPath !== "string" || input.launcherPath.length === 0 || input.launcherPath.length > 4096 || input.launcherPath.includes("\0")) {
      throw new BridgeContractError();
    }
    const environment = copyLauncherEnvironment(input.environment);
    const options = {
      shell: false,
      windowsHide: true,
      env: environment,
      stdio: ["pipe", "ignore", "ignore"]
    };
    if (input.platform !== "win32") {
      if (!path.posix.isAbsolute(input.launcherPath)) throw new BridgeContractError();
      return { file: input.launcherPath, arguments: [], options };
    }
    if (!path.win32.isAbsolute(input.launcherPath) || input.launcherPath.includes('"')) {
      throw new BridgeContractError();
    }
    const systemRoots = Object.entries(environment).filter(
      ([key]) => key.toUpperCase() === "SYSTEMROOT"
    );
    if (systemRoots.length !== 1) throw new BridgeContractError();
    const systemRoot = systemRoots[0][1];
    if (typeof systemRoot !== "string" || !/^[A-Za-z]:\\Windows$/i.test(systemRoot) || systemRoot.includes('"')) {
      throw new BridgeContractError();
    }
    const file = path.win32.join(systemRoot, "System32", "cmd.exe");
    const windowsEnvironment = Object.fromEntries(
      Object.entries(environment).filter(
        ([key]) => key.toUpperCase() !== "SALIENCEGATE_LAUNCHER"
      )
    );
    return {
      file,
      arguments: ["/d", "/v:off", "/s", "/c", '""%SALIENCEGATE_LAUNCHER%""'],
      options: {
        ...options,
        env: { ...windowsEnvironment, SALIENCEGATE_LAUNCHER: input.launcherPath },
        windowsVerbatimArguments: true
      }
    };
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

// connectors/bridge-core/src/windowed-chunks.ts
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
function hasWindowCoordinates(value, sessionID, windowDiscriminator) {
  return typeof value === "object" && value !== null && !Array.isArray(value) && value.session_id === sessionID && value.window_discriminator === windowDiscriminator;
}
function oversizeEvent(sessionID, windowDiscriminator) {
  return {
    kind: "oversize",
    reason: "event_limit",
    session_id: sessionID,
    window_discriminator: windowDiscriminator
  };
}
function normalizeWindowedCaptureEvent(value, sessionID, windowDiscriminator) {
  let normalized;
  try {
    normalized = canonicalizeJson(value);
    if (encodeCanonicalJson(normalized).byteLength > MAX_CAPTURE_EVENT_BYTES) {
      return oversizeEvent(sessionID, windowDiscriminator);
    }
  } catch {
    return oversizeEvent(sessionID, windowDiscriminator);
  }
  if (!hasWindowCoordinates(normalized, sessionID, windowDiscriminator)) {
    throw new BridgeContractError();
  }
  return normalized;
}
function documentFor(input) {
  return {
    schema_version: "capture-batch/v1",
    bootstrap: input.bootstrap,
    batch_id: input.batchID,
    session_id: input.sessionID,
    window_discriminator: input.windowDiscriminator,
    chunk_index: input.chunkIndex,
    chunk_count: input.chunkCount,
    events: input.events
  };
}
function encodedSize(input) {
  try {
    return encodeCanonicalJson(
      documentFor({ ...input, chunkCount: MAX_CAPTURE_BATCH_CHUNKS })
    ).byteLength;
  } catch (error) {
    if (error instanceof BridgeContractError) return MAX_CAPTURE_BATCH_BYTES + 1;
    throw error;
  }
}
function buildWindowedCaptureChunks(input) {
  try {
    const bootstrap = validateBootstrap(input.bootstrap);
    if (typeof input.batchID !== "string" || !SHA256.test(input.batchID) || typeof input.sessionID !== "string" || input.sessionID.length === 0 || Buffer3.byteLength(input.sessionID, "utf8") > MAX_CAPTURE_SESSION_ID_BYTES || typeof input.windowDiscriminator !== "string" || !SHA256.test(input.windowDiscriminator) || !Array.isArray(input.events) || input.events.length > 1e4) {
      throw new BridgeContractError();
    }
    canonicalizeJson(input.sessionID);
    const events = input.events.map((event) => {
      const normalized = normalizeWindowedCaptureEvent(
        event,
        input.sessionID,
        input.windowDiscriminator
      );
      return encodedSize({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        windowDiscriminator: input.windowDiscriminator,
        chunkIndex: 0,
        events: [normalized]
      }) <= MAX_CAPTURE_BATCH_BYTES ? normalized : oversizeEvent(input.sessionID, input.windowDiscriminator);
    });
    const groups = [];
    let current = [];
    for (const event of events) {
      const candidate = [...current, event];
      if (current.length > 0 && (current.length >= MAX_CAPTURE_EVENTS_PER_CHUNK || encodedSize({
        bootstrap,
        batchID: input.batchID,
        sessionID: input.sessionID,
        windowDiscriminator: input.windowDiscriminator,
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
        windowDiscriminator: input.windowDiscriminator,
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

// connectors/bridge-core/src/windowed-transport.ts
import { spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
var CAPTURE_LAUNCHER_TIMEOUT_MS = 2e3;
var MAX_CONCURRENT_CAPTURE_LAUNCHERS = 4;
var MAX_PENDING_GAP_WINDOWS = 256;
var WINDOW_DISCRIMINATOR = /^[0-9a-f]{64}$/;
async function spawnWindowedCaptureChunk(input, spawnChild = spawn) {
  return await new Promise((resolve) => {
    let child;
    try {
      child = spawnChild(
        input.invocation.file,
        input.invocation.arguments,
        input.invocation.options
      );
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
function validCoordinates(value) {
  return typeof value.sessionID === "string" && value.sessionID.length > 0 && isWellFormedUnicode(value.sessionID) && Buffer.byteLength(value.sessionID, "utf8") <= MAX_CAPTURE_SESSION_ID_BYTES && typeof value.windowDiscriminator === "string" && WINDOW_DISCRIMINATOR.test(value.windowDiscriminator);
}
function windowKey(value) {
  return createHash("sha256").update(
    encodeCanonicalJson({
      session_id: value.sessionID,
      window_discriminator: value.windowDiscriminator
    })
  ).digest("hex");
}
function hasMatchingWindow(value, window) {
  return typeof value === "object" && value !== null && !Array.isArray(value) && value.session_id === window.sessionID && value.window_discriminator === window.windowDiscriminator;
}
var WindowedBatchTransport = class {
  #bootstrap;
  #invocation;
  #writeChunk;
  #batchID;
  #pendingGaps = /* @__PURE__ */ new Map();
  #gapGeneration = 0n;
  #activeWrites = 0;
  #globalGapPoison = false;
  constructor(bootstrap, options = {}) {
    this.#bootstrap = bootstrap;
    this.#invocation = launcherInvocation({
      platform: options.platform ?? process.platform,
      launcherPath: bootstrap.launcher_path,
      environment: options.environment ?? process.env
    });
    this.#writeChunk = options.writeChunk ?? spawnWindowedCaptureChunk;
    this.#batchID = options.batchID ?? (() => randomBytes(32).toString("hex"));
  }
  pendingWindows() {
    return [...this.#pendingGaps.values()].map((value) => value.coordinates);
  }
  hasPendingGap(coordinates) {
    return validCoordinates(coordinates) && (this.#globalGapPoison || this.#pendingGaps.has(windowKey(coordinates)));
  }
  markGap(coordinates) {
    if (!validCoordinates(coordinates)) return;
    const key = windowKey(coordinates);
    if (this.#pendingGaps.has(key) || this.#pendingGaps.size < MAX_PENDING_GAP_WINDOWS) {
      this.#gapGeneration += 1n;
      this.#pendingGaps.set(key, {
        coordinates: { ...coordinates },
        generation: this.#gapGeneration
      });
    } else {
      this.#globalGapPoison = true;
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
  async flush(coordinates, records, options = {}) {
    try {
      if (!validCoordinates(coordinates) || !records.every((record) => hasMatchingWindow(record, coordinates))) {
        this.markGap(coordinates);
        return "attempted_failure";
      }
      const key = windowKey(coordinates);
      const pending = this.#pendingGaps.get(key);
      const events = [
        ...pending === void 0 && !this.#globalGapPoison ? [] : [
          {
            kind: "coverage_degraded",
            reason: "transport_gap",
            session_id: coordinates.sessionID,
            window_discriminator: coordinates.windowDiscriminator
          }
        ],
        ...records
      ];
      if (events.length === 0 && options.force !== true) return "delivered";
      const chunks = buildWindowedCaptureChunks({
        bootstrap: this.#bootstrap,
        batchID: this.#batchID(),
        sessionID: coordinates.sessionID,
        windowDiscriminator: coordinates.windowDiscriminator,
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
};

// connectors/pi/src/bootstrap.ts
import { createHash as createHash2, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import path2 from "node:path";
import { fileURLToPath } from "node:url";
var MAX_BOOTSTRAP_BYTES = 16 * 1024;
var MAX_BUNDLE_BYTES = 2 * 1024 * 1024;
var BOOTSTRAP_NAME = "saliencegate.bootstrap.json";
var BUNDLE_NAME = "saliencegate.ts";
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
  if (JSON.stringify(keys) !== JSON.stringify(expected) || value.schema_version !== "integration-bootstrap/v1" || value.profile !== "pi-extension/v1" || typeof value.connection_id !== "string" || !CONNECTION_ID2.test(value.connection_id) || typeof launcher !== "string" || launcher.length === 0 || launcher.length > 4096 || launcher.includes("\0") || !(launcher.startsWith("/") || WINDOWS_ABSOLUTE2.test(launcher)) || typeof value.capability_digest !== "string" || !SHA2562.test(value.capability_digest) || typeof value.bundle_digest !== "string" || !SHA2562.test(value.bundle_digest) || typeof value.receipt_mac !== "string" || !SHA2562.test(value.receipt_mac)) {
    throw new BridgeContractError();
  }
  return value;
}
async function loadPiBootstrap(bootstrapURL) {
  try {
    if (!(bootstrapURL instanceof URL) || bootstrapURL.protocol !== "file:" || bootstrapURL.search !== "" || bootstrapURL.hash !== "") {
      throw new BridgeContractError();
    }
    const bootstrapPath = fileURLToPath(bootstrapURL);
    if (path2.basename(bootstrapPath) !== BOOTSTRAP_NAME) throw new BridgeContractError();
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
    const bundlePath = path2.join(path2.dirname(bootstrapPath), BUNDLE_NAME);
    const bundle = await readStableRegularFile(bundlePath, {
      minimum: 1,
      maximum: MAX_BUNDLE_BYTES
    });
    const observedDigest = createHash2("sha256").update(bundle).digest("hex");
    const expectedDigest = Buffer.from(bootstrap.bundle_digest, "ascii");
    const observedBytes = Buffer.from(observedDigest, "ascii");
    if (!timingSafeEqual(expectedDigest, observedBytes)) throw new BridgeContractError();
    return bootstrap;
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

// connectors/pi/src/reducer.ts
import { createHash as createHash3 } from "node:crypto";
var MAX_PI_NATIVE_SESSION_ID_BYTES = 16 * 1024;
var MAX_PI_LEAF_ID_BYTES = 16 * 1024;
var MAX_CALLS_PER_WINDOW = 1e3;
var MAX_REDUCED_RECORDS_PER_WINDOW = 997;
var MAX_REDUCER_STATE_BYTES = 2 * 1024 * 1024;
var CALL_STATE_OVERHEAD_BYTES = 192;
var FINAL_CALL_STATE_BYTES = 192;
var WINDOW_DISCRIMINATOR2 = /^[0-9a-f]{64}$/;
var NATIVE_SESSION_ID = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/;
var COMPACTION_REASONS = /* @__PURE__ */ new Set([
  "manual",
  "threshold",
  "overflow"
]);
var SHUTDOWN_REASONS = /* @__PURE__ */ new Set([
  "quit",
  "reload",
  "new",
  "resume",
  "fork"
]);
function isRecord2(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function dataValue2(value, key) {
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  if (descriptor === void 0 || !("value" in descriptor) || !descriptor.enumerable) {
    return void 0;
  }
  return descriptor.value;
}
function exactText(value, maximumBytes) {
  if (typeof value !== "string" || value.length === 0 || !isWellFormedUnicode(value) || Buffer.byteLength(value, "utf8") > maximumBytes) {
    return void 0;
  }
  return value;
}
function exactLeaf(value) {
  return value === null ? null : exactText(value, MAX_PI_LEAF_ID_BYTES);
}
function digest(value) {
  return createHash3("sha256").update(encodeCanonicalJson(value)).digest("hex");
}
function checkedNativeSessionID(value) {
  const text = exactText(value, MAX_PI_NATIVE_SESSION_ID_BYTES);
  if (text === void 0 || !NATIVE_SESSION_ID.test(text)) throw new BridgeContractError();
  return text;
}
function isValidPiNativeSessionID(value) {
  try {
    checkedNativeSessionID(value);
    return true;
  } catch {
    return false;
  }
}
var PiWindowReducer = class {
  #sessionID;
  #windowDiscriminator;
  #pending = /* @__PURE__ */ new Map();
  #final = /* @__PURE__ */ new Map();
  #retainedStateBytes = 0;
  #recordCount = 0;
  #nextEventID = 1;
  #overflowReported = false;
  #disabled = false;
  #finished = false;
  #turnOpen = false;
  constructor(input) {
    this.#sessionID = checkedNativeSessionID(input.sessionID);
    if (!WINDOW_DISCRIMINATOR2.test(input.windowDiscriminator)) {
      throw new BridgeContractError();
    }
    this.#windowDiscriminator = input.windowDiscriminator;
  }
  coordinates() {
    return {
      sessionID: this.#sessionID,
      windowDiscriminator: this.#windowDiscriminator
    };
  }
  isFinished() {
    return this.#finished;
  }
  #reserveState(bytes) {
    if (this.#retainedStateBytes + bytes > MAX_REDUCER_STATE_BYTES) return false;
    this.#retainedStateBytes += bytes;
    return true;
  }
  #releaseState(bytes) {
    this.#retainedStateBytes = Math.max(0, this.#retainedStateBytes - bytes);
  }
  #clearPending() {
    const hadPending = this.#pending.size > 0;
    for (const [key, call] of this.#pending) {
      this.#releaseState(call.retainedBytes - FINAL_CALL_STATE_BYTES);
      this.#final.set(key, {
        startFingerprint: call.startFingerprint,
        retainedBytes: FINAL_CALL_STATE_BYTES
      });
    }
    this.#pending.clear();
    return hadPending;
  }
  #materialize(body) {
    const record = {
      ...body,
      session_id: this.#sessionID,
      window_discriminator: this.#windowDiscriminator,
      event_id: String(this.#nextEventID)
    };
    this.#nextEventID += 1;
    this.#recordCount += 1;
    return record;
  }
  #admit(bodies, mode = "normal") {
    if (bodies.length === 0) return [];
    const limit = mode === "terminal" ? MAX_REDUCED_RECORDS_PER_WINDOW : mode === "degradation" ? MAX_REDUCED_RECORDS_PER_WINDOW - 1 : MAX_REDUCED_RECORDS_PER_WINDOW - 2;
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
          reason: "overflow"
        })
      ];
    }
    return [];
  }
  degrade(reason) {
    if (this.#finished) return [];
    if (reason === "overflow") {
      this.#disabled = true;
      if (this.#overflowReported) return [];
      this.#overflowReported = true;
    }
    return this.#admit([{ kind: "coverage_degraded", reason }], "degradation");
  }
  #rememberFinal(key, value) {
    if (!this.#reserveState(FINAL_CALL_STATE_BYTES)) return false;
    this.#final.set(key, { ...value, retainedBytes: FINAL_CALL_STATE_BYTES });
    return true;
  }
  #toolStart(event) {
    if (this.#disabled) return [];
    const callID = exactText(dataValue2(event, "toolCallId"), MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(dataValue2(event, "toolName"), MAX_CAPTURE_TOOL_NAME_BYTES);
    if (callID === void 0 || tool === void 0) return this.degrade("missing_field");
    const key = digest({ callID });
    const startFingerprint = digest({ callID, tool });
    const finalized = this.#final.get(key);
    if (finalized !== void 0) {
      return finalized.startFingerprint === startFingerprint ? [] : this.degrade("invalid_transition");
    }
    const prior = this.#pending.get(key);
    if (prior !== void 0) {
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
  #toolEnd(event) {
    if (this.#disabled) return [];
    const callID = exactText(dataValue2(event, "toolCallId"), MAX_CAPTURE_CALL_ID_BYTES);
    const tool = exactText(dataValue2(event, "toolName"), MAX_CAPTURE_TOOL_NAME_BYTES);
    const isError = dataValue2(event, "isError");
    if (callID === void 0 || tool === void 0 || typeof isError !== "boolean") {
      return this.degrade("missing_field");
    }
    const key = digest({ callID });
    const startFingerprint = digest({ callID, tool });
    const endFingerprint = digest({ callID, tool, isError });
    const finalized = this.#final.get(key);
    if (finalized !== void 0) {
      if (finalized.endFingerprint === endFingerprint) return [];
      if (finalized.endFingerprint === void 0) {
        finalized.endFingerprint = endFingerprint;
      }
      return this.degrade("invalid_transition");
    }
    const pending = this.#pending.get(key);
    if (pending === void 0) {
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
        identity_authority: "coarse"
      },
      {
        kind: "tool_finished",
        call_id: callID,
        outcome: "succeeded"
      }
    ]);
  }
  #beforeAgentStart() {
    if (this.#disabled) return [];
    if (this.#turnOpen) return this.degrade("invalid_transition");
    this.#turnOpen = true;
    return [];
  }
  #agentSettled() {
    const unmatched = this.#clearPending();
    const bodies = [];
    if (unmatched) bodies.push({ kind: "coverage_degraded", reason: "unmatched_start" });
    if (this.#turnOpen) bodies.push({ kind: "turn_finished" });
    this.#turnOpen = false;
    return this.#admit(bodies);
  }
  #compact(event) {
    const reason = dataValue2(event, "reason");
    const fromExtension = dataValue2(event, "fromExtension");
    const willRetry = dataValue2(event, "willRetry");
    if (typeof reason !== "string" || !COMPACTION_REASONS.has(reason) || typeof fromExtension !== "boolean" || typeof willRetry !== "boolean") {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    return this.#admit([
      ...unmatched ? [{ kind: "coverage_degraded", reason: "unmatched_start" }] : [],
      {
        kind: "coverage_boundary",
        reason: "compaction",
        compaction_reason: reason,
        from_extension: fromExtension,
        will_retry: willRetry
      }
    ]);
  }
  #tree(event) {
    const newLeafID = exactLeaf(dataValue2(event, "newLeafId"));
    const oldLeafID = exactLeaf(dataValue2(event, "oldLeafId"));
    if (newLeafID === void 0 || oldLeafID === void 0) {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    return this.#admit([
      ...unmatched ? [{ kind: "coverage_degraded", reason: "unmatched_start" }] : [],
      {
        kind: "coverage_boundary",
        reason: "tree",
        old_leaf_id: oldLeafID,
        new_leaf_id: newLeafID
      }
    ]);
  }
  #shutdown(event) {
    const reason = dataValue2(event, "reason");
    if (typeof reason !== "string" || !SHUTDOWN_REASONS.has(reason)) {
      return this.degrade("missing_field");
    }
    const unmatched = this.#clearPending();
    this.#turnOpen = false;
    const terminal = {
      kind: "session_finished",
      reason
    };
    const bodies = unmatched ? [{ kind: "coverage_degraded", reason: "unmatched_start" }, terminal] : [terminal];
    if (this.#recordCount + bodies.length > MAX_REDUCED_RECORDS_PER_WINDOW) {
      bodies.splice(0, bodies.length, terminal);
    }
    const records = this.#admit(
      bodies,
      "terminal"
    );
    this.#finished = true;
    return records;
  }
  reduce(value) {
    try {
      if (!isRecord2(value) || this.#finished) return [];
      const type = dataValue2(value, "type");
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
};

// connectors/pi/src/extension.ts
var MAX_SESSION_BUFFER_BYTES = 512 * 1024;
var SERIAL_QUEUE_KEY = "pi-extension-runtime";
var WINDOW_DISCRIMINATOR3 = /^[0-9a-f]{64}$/;
var START_REASONS = /* @__PURE__ */ new Set([
  "startup",
  "reload",
  "new",
  "resume",
  "fork"
]);
function isRecord3(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function dataValue3(value, key) {
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  if (descriptor === void 0 || !("value" in descriptor) || !descriptor.enumerable) {
    return void 0;
  }
  return descriptor.value;
}
function contextSessionID(value) {
  try {
    const sessionID = value.sessionManager.getSessionId();
    return isValidPiNativeSessionID(sessionID) ? sessionID : void 0;
  } catch {
    return void 0;
  }
}
function asCanonicalRecord(record) {
  return normalizeWindowedCaptureEvent(
    record,
    record.session_id,
    record.window_discriminator
  );
}
function hasTerminal(records) {
  return records.some(
    (record) => typeof record === "object" && record !== null && !Array.isArray(record) && record.kind === "session_finished"
  );
}
var PiExtensionRuntime = class {
  #queue = new SerialSessionQueue();
  #transport;
  #windowDiscriminator;
  #active;
  constructor(bootstrap, options) {
    this.#transport = new WindowedBatchTransport(bootstrap, options);
    this.#windowDiscriminator = options.windowDiscriminator ?? (() => randomBytes2(32).toString("hex"));
  }
  async #flush(active, force = false) {
    const records = active.records;
    if (!force && records.length === 0) return;
    const terminal = hasTerminal(records);
    active.records = [];
    active.bytes = 0;
    const result = await this.#transport.flush(active.coordinates, records, { force });
    if (result === "not_started" && terminal) {
      const retained = records.filter(
        (record) => typeof record === "object" && record !== null && !Array.isArray(record) && record.kind === "session_finished"
      );
      active.records = retained;
      active.bytes = 0;
      for (const record of retained) {
        active.bytes += encodeCanonicalJson(record).byteLength;
      }
    }
  }
  async #append(active, records) {
    const normalized = records.map((record) => asCanonicalRecord(record));
    const oversized = normalized.find(
      (record) => typeof record === "object" && record !== null && !Array.isArray(record) && record.kind === "oversize"
    );
    const group = oversized === void 0 ? normalized : [oversized];
    const sized = group.map((record) => ({
      record,
      bytes: encodeCanonicalJson(record).byteLength
    }));
    const groupBytes = sized.reduce((total, item) => total + item.bytes, 0);
    if (active.records.length > 0 && active.bytes + groupBytes > MAX_SESSION_BUFFER_BYTES) {
      await this.#flush(active);
    }
    for (const item of sized) {
      active.records.push(item.record);
      active.bytes += item.bytes;
    }
  }
  async #degradeActive(reason) {
    if (this.#active === void 0) return;
    await this.#append(this.#active, this.#active.reducer.degrade(reason));
  }
  async #sessionStart(value, context) {
    const sessionID = contextSessionID(context);
    if (!isRecord3(value) || dataValue3(value, "type") !== "session_start") {
      await this.#degradeActive("missing_field");
      if (this.#active !== void 0) await this.#flush(this.#active);
      return;
    }
    const reason = dataValue3(value, "reason");
    if (sessionID === void 0 || typeof reason !== "string" || !START_REASONS.has(reason)) {
      await this.#degradeActive("missing_field");
      if (this.#active !== void 0) await this.#flush(this.#active);
      return;
    }
    if (this.#active !== void 0) {
      const active2 = this.#active;
      await this.#degradeActive("invalid_transition");
      await this.#flush(active2);
      if (active2.records.length > 0) return;
      this.#active = void 0;
    }
    let discriminator;
    try {
      discriminator = this.#windowDiscriminator();
    } catch {
      return;
    }
    if (!WINDOW_DISCRIMINATOR3.test(discriminator)) return;
    const reducer = new PiWindowReducer({
      sessionID,
      windowDiscriminator: discriminator
    });
    const active = {
      reducer,
      coordinates: reducer.coordinates(),
      records: [],
      bytes: 0
    };
    this.#active = active;
    await this.#flush(active, true);
  }
  async #observedEvent(expectedType, value, context, flushBoundary, closesWindow) {
    const active = this.#active;
    if (active === void 0) return;
    const sessionID = contextSessionID(context);
    if (sessionID !== active.coordinates.sessionID || !isRecord3(value) || dataValue3(value, "type") !== expectedType) {
      await this.#append(active, active.reducer.degrade("missing_field"));
    } else {
      await this.#append(active, active.reducer.reduce(value));
    }
    if (flushBoundary) await this.#flush(active);
    if (closesWindow && active.reducer.isFinished() && active.records.length === 0) {
      this.#active = void 0;
    }
  }
  sessionStart(value, context) {
    return this.#queue.run(SERIAL_QUEUE_KEY, async () => {
      await this.#sessionStart(value, context);
    });
  }
  observedEvent(expectedType, value, context, options = {}) {
    return this.#queue.run(SERIAL_QUEUE_KEY, async () => {
      await this.#observedEvent(
        expectedType,
        value,
        context,
        options.flushBoundary === true,
        options.closesWindow === true
      );
    });
  }
};
function createPiExtension(options) {
  return async (pi) => {
    let runtime;
    try {
      const loader = options.loadBootstrap ?? loadPiBootstrap;
      const bootstrap = await loader(options.bootstrapURL);
      if (bootstrap.profile === "pi-extension/v1") {
        runtime = new PiExtensionRuntime(bootstrap, options);
      }
    } catch {
      runtime = void 0;
    }
    pi.on("session_start", async (event, context) => {
      try {
        await runtime?.sessionStart(event, context);
      } catch {
      }
      return void 0;
    });
    pi.on("before_agent_start", async (event, context) => {
      try {
        await runtime?.observedEvent("before_agent_start", event, context);
      } catch {
      }
      return void 0;
    });
    pi.on(
      "tool_execution_start",
      async (event, context) => {
        try {
          await runtime?.observedEvent("tool_execution_start", event, context);
        } catch {
        }
        return void 0;
      }
    );
    pi.on("tool_execution_end", async (event, context) => {
      try {
        await runtime?.observedEvent("tool_execution_end", event, context);
      } catch {
      }
      return void 0;
    });
    pi.on("agent_settled", async (event, context) => {
      try {
        await runtime?.observedEvent("agent_settled", event, context, {
          flushBoundary: true
        });
      } catch {
      }
      return void 0;
    });
    pi.on("session_compact", async (event, context) => {
      try {
        await runtime?.observedEvent("session_compact", event, context, {
          flushBoundary: true
        });
      } catch {
      }
      return void 0;
    });
    pi.on("session_tree", async (event, context) => {
      try {
        await runtime?.observedEvent("session_tree", event, context, {
          flushBoundary: true
        });
      } catch {
      }
      return void 0;
    });
    pi.on("session_shutdown", async (event, context) => {
      try {
        await runtime?.observedEvent("session_shutdown", event, context, {
          flushBoundary: true,
          closesWindow: true
        });
      } catch {
      }
      return void 0;
    });
  };
}

// pi-runtime-entry.ts
var pi_runtime_entry_default = createPiExtension({ bootstrapURL: saliencegateBootstrap });
export {
  pi_runtime_entry_default as default
};
