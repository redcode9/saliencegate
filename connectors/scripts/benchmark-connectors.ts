import path from "node:path";
import { performance } from "node:perf_hooks";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";

import {
  encodeCanonicalJson,
  type BootstrapBinding,
} from "../bridge-core/src/index.ts";
import { createOpenCodePlugin } from "../opencode/src/index.ts";
import {
  createPiExtension,
  type PiExtensionAPI,
  type PiExtensionContext,
} from "../pi/src/index.ts";

export const SYNCHRONOUS_CALLBACK_P95_BUDGET_MS = 10;
export const WARM_LOCAL_FLUSH_P95_BUDGET_MS = 500;
export const SYNCHRONOUS_CALLBACK_SAMPLES = 200;
export const WARM_LOCAL_FLUSH_SAMPLES = 30;
const WARMUP_SAMPLES = 20;
const PI_SESSION_ID = "019c0eaf-7b31-7000-8000-000000000001";

type TimedMeasurement = Readonly<{
  launcher_processes: number;
  synchronous_callback_p95_ms: number;
  warm_local_flush_p95_ms: number;
}>;

type TimedResult = TimedMeasurement & Readonly<{ passed: boolean }>;

export type ConnectorBenchmarkReport = Readonly<{
  budgets_ms: {
    synchronous_callback_p95: number;
    warm_local_flush_p95: number;
  };
  measurements: {
    opencode: TimedResult;
    pi: TimedResult;
  };
  passed: boolean;
  samples: {
    synchronous_callbacks: number;
    warm_local_flushes: number;
  };
  schema_version: "connector-runtime-benchmark/v1";
  transport: "provider-free-offline-subprocess-stdin";
}>;

type LauncherFixture = Readonly<{
  close: () => Promise<void>;
  count: () => Promise<number>;
  environment: NodeJS.ProcessEnv;
  launcherPath: string;
}>;

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

function launcherEnvironment(): NodeJS.ProcessEnv {
  const denialPreload = new URL("./deny-network.mjs", import.meta.url).href;
  if (process.platform === "win32") {
    return {
      NODE_OPTIONS: `--import=${denialPreload}`,
      PATH: process.env.PATH ?? "C:\\Windows\\System32",
      SystemRoot: process.env.SystemRoot ?? "C:\\Windows",
    };
  }
  return {
    NODE_OPTIONS: `--import=${denialPreload}`,
    PATH: process.env.PATH ?? "/usr/bin:/bin",
  };
}

async function createLauncherFixture(): Promise<LauncherFixture> {
  const directory = await mkdtemp(
    path.join(tmpdir(), "saliencegate-connector-benchmark-"),
  );
  const receiverPath = path.join(directory, "receive-batch.mjs");
  const counterPath = path.join(directory, "accepted-batches.txt");
  const receiver = `
import { appendFile } from "node:fs/promises";

const providerCredentialKeys = new Set([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID",
]);
if (globalThis[Symbol.for("saliencegate.network-denial/v1")] !== true) {
  throw new Error("network denial preload missing in benchmark launcher");
}
if (Object.keys(process.env).some((key) => providerCredentialKeys.has(key.toUpperCase()))) {
  throw new Error("provider credential reached benchmark launcher");
}
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const document = JSON.parse(Buffer.concat(chunks).toString("utf8"));
if (
  document.schema_version !== "capture-batch/v1" ||
  !Array.isArray(document.events) ||
  typeof document.session_id !== "string"
) {
  throw new Error("invalid benchmark capture batch");
}
await appendFile(${JSON.stringify(counterPath)}, "1", { encoding: "ascii" });
`;
  await writeFile(receiverPath, receiver, { mode: 0o600 });

  let launcherPath: string;
  if (process.platform === "win32") {
    if (process.execPath.includes('"') || receiverPath.includes('"')) {
      throw new Error("unsupported quote in Windows benchmark launcher path");
    }
    launcherPath = path.join(directory, "capture-hook.cmd");
    await writeFile(
      launcherPath,
      `@echo off\r\n"${process.execPath}" "${receiverPath}"\r\n`,
      { mode: 0o600 },
    );
  } else {
    launcherPath = path.join(directory, "capture-hook");
    await writeFile(
      launcherPath,
      `#!/bin/sh\nexec ${shellQuote(process.execPath)} ${shellQuote(receiverPath)}\n`,
      { mode: 0o700 },
    );
    await chmod(launcherPath, 0o700);
  }

  return {
    close: async () => {
      await rm(directory, { force: true, recursive: true });
    },
    count: async () => {
      try {
        return (await readFile(counterPath, "ascii")).length;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
        throw error;
      }
    },
    environment: launcherEnvironment(),
    launcherPath,
  };
}

function bootstrap(
  profile: BootstrapBinding["profile"],
  launcherPath: string,
): BootstrapBinding {
  return {
    schema_version: "integration-bootstrap/v1",
    profile,
    connection_id: `sg-${"1".repeat(48)}`,
    launcher_path: launcherPath,
    capability_digest: "2".repeat(64),
    bundle_digest: "3".repeat(64),
    receipt_mac: "4".repeat(64),
  };
}

function roundedMilliseconds(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

export function percentile95(samples: readonly number[]): number {
  if (samples.length === 0) throw new Error("benchmark requires samples");
  const ordered = [...samples].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(ordered.length * 0.95) - 1);
  return roundedMilliseconds(ordered[index]!);
}

async function synchronousReturnDuration(
  callback: () => Promise<void>,
): Promise<number> {
  const started = performance.now();
  const completion = callback();
  const elapsed = performance.now() - started;
  await completion;
  return elapsed;
}

async function completionDuration(callback: () => Promise<void>): Promise<number> {
  const started = performance.now();
  await callback();
  return performance.now() - started;
}

function openCodeToolEvent(callID: string): object {
  return {
    type: "message.part.updated",
    properties: {
      part: {
        id: `part-${callID}`,
        sessionID: "benchmark-opencode",
        messageID: `message-${callID}`,
        type: "tool",
        callID,
        tool: "read",
        state: { status: "pending", input: null },
      },
    },
  };
}

async function benchmarkOpenCode(
  fixture: LauncherFixture,
): Promise<TimedMeasurement> {
  const initialLauncherCount = await fixture.count();
  const plugin = createOpenCodePlugin({
    bootstrapURL: pathToFileURL(fixture.launcherPath),
    loadBootstrap: async () =>
      bootstrap("opencode-plugin/v1", fixture.launcherPath),
    environment: fixture.environment,
    batchID: () => "5".repeat(64),
  });
  const hooks = await plugin({});

  for (let index = 0; index < WARMUP_SAMPLES; index += 1) {
    await hooks.event({ event: openCodeToolEvent(`warmup-${index}`) });
  }
  const callbackSamples: number[] = [];
  for (let index = 0; index < SYNCHRONOUS_CALLBACK_SAMPLES; index += 1) {
    callbackSamples.push(
      await synchronousReturnDuration(async () => {
        await hooks.event({ event: openCodeToolEvent(`callback-${index}`) });
      }),
    );
  }
  await hooks.event({
    event: { type: "session.idle", properties: { sessionID: "benchmark-opencode" } },
  });

  const flushSamples: number[] = [];
  for (let index = 0; index < WARM_LOCAL_FLUSH_SAMPLES; index += 1) {
    await hooks.event({ event: openCodeToolEvent(`flush-${index}`) });
    flushSamples.push(
      await completionDuration(async () => {
        await hooks.event({
          event: {
            type: "session.idle",
            properties: { sessionID: "benchmark-opencode" },
          },
        });
      }),
    );
  }
  await hooks.dispose();
  const launcherProcesses = (await fixture.count()) - initialLauncherCount;
  if (launcherProcesses !== WARM_LOCAL_FLUSH_SAMPLES + 1) {
    throw new Error("OpenCode benchmark launcher did not accept every flush");
  }
  return {
    launcher_processes: launcherProcesses,
    synchronous_callback_p95_ms: percentile95(callbackSamples),
    warm_local_flush_p95_ms: percentile95(flushSamples),
  };
}

type PiHandler = (event: unknown, context: PiExtensionContext) => unknown;

class BenchmarkPiAPI {
  readonly handlers = new Map<string, PiHandler>();

  on(event: string, handler: PiHandler): void {
    this.handlers.set(event, handler);
  }
}

async function invokePi(
  api: BenchmarkPiAPI,
  event: string,
  value: unknown,
  context: PiExtensionContext,
): Promise<void> {
  const handler = api.handlers.get(event);
  if (handler === undefined) throw new Error(`missing Pi benchmark handler: ${event}`);
  await handler(value, context);
}

async function benchmarkPi(fixture: LauncherFixture): Promise<TimedMeasurement> {
  const initialLauncherCount = await fixture.count();
  const api = new BenchmarkPiAPI();
  const context = {
    sessionManager: { getSessionId: () => PI_SESSION_ID },
  } as PiExtensionContext;
  await createPiExtension({
    bootstrapURL: pathToFileURL(fixture.launcherPath),
    loadBootstrap: async () => bootstrap("pi-extension/v1", fixture.launcherPath),
    environment: fixture.environment,
    batchID: () => "6".repeat(64),
    windowDiscriminator: () => "7".repeat(64),
  })(api as unknown as PiExtensionAPI);
  await invokePi(
    api,
    "session_start",
    { type: "session_start", reason: "startup" },
    context,
  );

  for (let index = 0; index < WARMUP_SAMPLES; index += 1) {
    await invokePi(
      api,
      "tool_execution_start",
      {
        type: "tool_execution_start",
        toolCallId: `warmup-${index}`,
        toolName: "read",
      },
      context,
    );
    await invokePi(
      api,
      "tool_execution_end",
      {
        type: "tool_execution_end",
        toolCallId: `warmup-${index}`,
        toolName: "read",
        isError: false,
      },
      context,
    );
  }
  const callbackSamples: number[] = [];
  for (let index = 0; index < SYNCHRONOUS_CALLBACK_SAMPLES; index += 1) {
    const event = {
      type: "tool_execution_start",
      toolCallId: `callback-${index}`,
      toolName: "read",
    };
    callbackSamples.push(
      await synchronousReturnDuration(async () => {
        await invokePi(api, "tool_execution_start", event, context);
      }),
    );
    await invokePi(
      api,
      "tool_execution_end",
      { ...event, type: "tool_execution_end", isError: false },
      context,
    );
  }
  await invokePi(api, "agent_settled", { type: "agent_settled" }, context);

  const flushSamples: number[] = [];
  for (let index = 0; index < WARM_LOCAL_FLUSH_SAMPLES; index += 1) {
    const event = {
      type: "tool_execution_start",
      toolCallId: `flush-${index}`,
      toolName: "read",
    };
    await invokePi(api, "tool_execution_start", event, context);
    await invokePi(
      api,
      "tool_execution_end",
      { ...event, type: "tool_execution_end", isError: false },
      context,
    );
    flushSamples.push(
      await completionDuration(async () => {
        await invokePi(api, "agent_settled", { type: "agent_settled" }, context);
      }),
    );
  }
  await invokePi(
    api,
    "session_shutdown",
    { type: "session_shutdown", reason: "quit" },
    context,
  );
  const launcherProcesses = (await fixture.count()) - initialLauncherCount;
  if (launcherProcesses !== WARM_LOCAL_FLUSH_SAMPLES + 3) {
    throw new Error("Pi benchmark launcher did not accept every flush");
  }
  return {
    launcher_processes: launcherProcesses,
    synchronous_callback_p95_ms: percentile95(callbackSamples),
    warm_local_flush_p95_ms: percentile95(flushSamples),
  };
}

export async function benchmarkConnectors(): Promise<ConnectorBenchmarkReport> {
  const fixture = await createLauncherFixture();
  try {
    const openCodeMeasurement = await benchmarkOpenCode(fixture);
    const piMeasurement = await benchmarkPi(fixture);
    const opencode = {
      ...openCodeMeasurement,
      passed: measurementPassed(openCodeMeasurement),
    };
    const pi = { ...piMeasurement, passed: measurementPassed(piMeasurement) };
    return {
      budgets_ms: {
        synchronous_callback_p95: SYNCHRONOUS_CALLBACK_P95_BUDGET_MS,
        warm_local_flush_p95: WARM_LOCAL_FLUSH_P95_BUDGET_MS,
      },
      measurements: {
        opencode,
        pi,
      },
      passed: opencode.passed && pi.passed,
      samples: {
        synchronous_callbacks: SYNCHRONOUS_CALLBACK_SAMPLES,
        warm_local_flushes: WARM_LOCAL_FLUSH_SAMPLES,
      },
      schema_version: "connector-runtime-benchmark/v1",
      transport: "provider-free-offline-subprocess-stdin",
    };
  } finally {
    await fixture.close();
  }
}

function measurementPassed(measurement: TimedMeasurement): boolean {
  return (
    measurement.synchronous_callback_p95_ms <=
      SYNCHRONOUS_CALLBACK_P95_BUDGET_MS &&
    measurement.warm_local_flush_p95_ms <= WARM_LOCAL_FLUSH_P95_BUDGET_MS
  );
}

export function assertBenchmarkBudgets(report: ConnectorBenchmarkReport): void {
  for (const [provider, measurement] of Object.entries(report.measurements)) {
    if (
      measurement.synchronous_callback_p95_ms >
      SYNCHRONOUS_CALLBACK_P95_BUDGET_MS
    ) {
      throw new Error(`${provider} synchronous callback p95 exceeded budget`);
    }
    if (measurement.warm_local_flush_p95_ms > WARM_LOCAL_FLUSH_P95_BUDGET_MS) {
      throw new Error(`${provider} warm local flush p95 exceeded budget`);
    }
    if (!measurement.passed) throw new Error(`${provider} benchmark did not pass`);
  }
  if (!report.passed) throw new Error("connector runtime benchmark did not pass");
}

async function main(): Promise<void> {
  const arguments_ = process.argv.slice(2);
  if (arguments_.some((value) => value !== "--assert-budgets")) {
    throw new Error("usage: benchmark-connectors.ts [--assert-budgets]");
  }
  const report = await benchmarkConnectors();
  if (arguments_.includes("--assert-budgets")) assertBenchmarkBudgets(report);
  process.stdout.write(encodeCanonicalJson(report));
  process.stdout.write("\n");
}

const entrypoint = process.argv[1];
if (
  entrypoint !== undefined &&
  pathToFileURL(path.resolve(entrypoint)).href === import.meta.url
) {
  await main();
}
