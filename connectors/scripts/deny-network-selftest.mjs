import assert from "node:assert/strict";
import { createRequire } from "node:module";

const REQUIRED_BLOCKED_MODULES = [
  "_http_agent",
  "_http_client",
  "_http_common",
  "_http_incoming",
  "_http_outgoing",
  "_http_server",
  "_tls_common",
  "_tls_wrap",
  "net",
  "tls",
  "http",
  "https",
  "dgram",
  "dns",
  "dns/promises",
  "http2",
  "cluster",
  "inspector",
];
const DENIAL_CODE = "ERR_SALIENCEGATE_NETWORK_DISABLED";
const require = createRequire(import.meta.url);

function isDenial(error) {
  return error instanceof Error && error.code === DENIAL_CODE;
}

assert.equal(globalThis[Symbol.for("saliencegate.network-denial/v1")], true);

for (const moduleName of REQUIRED_BLOCKED_MODULES) {
  for (const specifier of [moduleName, `node:${moduleName}`]) {
    await assert.rejects(import(specifier), isDenial, `dynamic import escaped: ${specifier}`);
    assert.throws(() => require(specifier), isDenial, `require escaped: ${specifier}`);
    assert.throws(
      () => process.getBuiltinModule(specifier),
      isDenial,
      `process.getBuiltinModule escaped: ${specifier}`,
    );
  }
}

await assert.rejects(
  globalThis.fetch("https://network-denial.invalid/"),
  isDenial,
  "global fetch escaped",
);
assert.throws(
  () => new globalThis.WebSocket("wss://network-denial.invalid/"),
  isDenial,
  "global WebSocket escaped",
);
assert.throws(
  () => new globalThis.EventSource("https://network-denial.invalid/"),
  isDenial,
  "global EventSource escaped",
);
for (const binding of ["cares_wrap", "http_parser", "tcp_wrap", "tls_wrap", "udp_wrap"]) {
  assert.throws(() => process.binding(binding), isDenial, `process.binding escaped: ${binding}`);
}

process.stdout.write(
  '{"blocked_bindings":5,"blocked_module_specifiers":36,"event_source":true,"fetch":true,"schema_version":"connector-network-denial-selftest/v1","web_socket":true}\n',
);
