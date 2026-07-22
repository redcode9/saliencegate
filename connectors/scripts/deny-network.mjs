import { registerHooks } from "node:module";

const BLOCKED_MODULE_ROOTS = new Set([
  "_http_agent",
  "_http_client",
  "_http_common",
  "_http_incoming",
  "_http_outgoing",
  "_http_server",
  "_tls_common",
  "_tls_wrap",
  "cluster",
  "dgram",
  "dns",
  "http",
  "http2",
  "https",
  "inspector",
  "net",
  "tls",
  "undici",
  "ws",
]);
const BLOCKED_BINDINGS = new Set([
  "cares_wrap",
  "http_parser",
  "tcp_wrap",
  "tls_wrap",
  "udp_wrap",
]);
const NETWORK_DENIAL_MARKER = Symbol.for("saliencegate.network-denial/v1");

function normalizedModuleRoot(specifier) {
  const normalized = specifier.startsWith("node:") ? specifier.slice(5) : specifier;
  return normalized.split("/", 1)[0];
}

function isBlockedModule(specifier) {
  return (
    typeof specifier === "string" &&
    BLOCKED_MODULE_ROOTS.has(normalizedModuleRoot(specifier))
  );
}

function denialError(target) {
  const error = new Error(`network access disabled by SalienceGate: ${target}`);
  error.code = "ERR_SALIENCEGATE_NETWORK_DISABLED";
  return error;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (isBlockedModule(specifier)) throw denialError(`module ${specifier}`);
    return nextResolve(specifier, context);
  },
});

const originalGetBuiltinModule = process.getBuiltinModule.bind(process);
Object.defineProperty(process, "getBuiltinModule", {
  configurable: false,
  enumerable: false,
  writable: false,
  value(specifier) {
    if (isBlockedModule(specifier)) {
      throw denialError(`builtin module ${specifier}`);
    }
    return originalGetBuiltinModule(specifier);
  },
});

const originalBinding = process.binding.bind(process);
Object.defineProperty(process, "binding", {
  configurable: false,
  enumerable: false,
  writable: false,
  value(name) {
    if (BLOCKED_BINDINGS.has(name)) throw denialError(`binding ${name}`);
    return originalBinding(name);
  },
});

async function deniedFetch() {
  throw denialError("global fetch");
}

class NetworkDeniedWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor() {
    throw denialError("global WebSocket");
  }
}

class NetworkDeniedEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;

  constructor() {
    throw denialError("global EventSource");
  }
}

Object.defineProperties(globalThis, {
  fetch: {
    configurable: false,
    enumerable: true,
    writable: false,
    value: deniedFetch,
  },
  WebSocket: {
    configurable: false,
    enumerable: true,
    writable: false,
    value: NetworkDeniedWebSocket,
  },
  EventSource: {
    configurable: false,
    enumerable: true,
    writable: false,
    value: NetworkDeniedEventSource,
  },
  [NETWORK_DENIAL_MARKER]: {
    configurable: false,
    enumerable: false,
    writable: false,
    value: true,
  },
});
