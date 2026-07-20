import { createHash, timingSafeEqual } from "node:crypto";
import { constants, type Stats } from "node:fs";
import { lstat, open } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  BridgeContractError,
  encodeCanonicalJson,
  type BootstrapBinding,
} from "@saliencegate/bridge-core";

const MAX_BOOTSTRAP_BYTES = 16 * 1024;
const MAX_BUNDLE_BYTES = 2 * 1024 * 1024;
const BOOTSTRAP_NAME = "saliencegate.bootstrap.json";
const BUNDLE_NAME = "saliencegate.js";
const SHA256 = /^[0-9a-f]{64}$/;
const CONNECTION_ID = /^sg-[0-9a-f]{48}$/;
const WINDOWS_ABSOLUTE = /^[A-Za-z]:[\\/]/;

function sameFile(first: Stats, second: Stats): boolean {
  return (
    first.dev === second.dev &&
    first.ino === second.ino &&
    first.mode === second.mode &&
    first.size === second.size &&
    first.mtimeMs === second.mtimeMs
  );
}

async function readStableRegularFile(
  filePath: string,
  input: { minimum: number; maximum: number },
): Promise<Buffer> {
  const before = await lstat(filePath);
  if (
    !before.isFile() ||
    before.isSymbolicLink() ||
    before.nlink !== 1 ||
    before.size < input.minimum ||
    before.size > input.maximum
  ) {
    throw new BridgeContractError();
  }
  if (
    process.platform !== "win32" &&
    ((before.mode & 0o077) !== 0 ||
      (typeof process.getuid === "function" && before.uid !== process.getuid()))
  ) {
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
    if (
      !sameFile(opened, after) ||
      offset !== opened.size ||
      offset < input.minimum ||
      offset > input.maximum
    ) {
      throw new BridgeContractError();
    }
    return buffer.subarray(0, offset);
  } finally {
    await handle.close();
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateBootstrap(value: unknown): BootstrapBinding {
  if (!isRecord(value)) throw new BridgeContractError();
  const keys = Object.keys(value).sort();
  const expected = [
    "bundle_digest",
    "capability_digest",
    "connection_id",
    "launcher_path",
    "profile",
    "receipt_mac",
    "schema_version",
  ];
  const launcher = value.launcher_path;
  if (
    JSON.stringify(keys) !== JSON.stringify(expected) ||
    value.schema_version !== "integration-bootstrap/v1" ||
    value.profile !== "opencode-plugin/v1" ||
    typeof value.connection_id !== "string" ||
    !CONNECTION_ID.test(value.connection_id) ||
    typeof launcher !== "string" ||
    launcher.length === 0 ||
    launcher.length > 4_096 ||
    launcher.includes("\0") ||
    !(launcher.startsWith("/") || WINDOWS_ABSOLUTE.test(launcher)) ||
    typeof value.capability_digest !== "string" ||
    !SHA256.test(value.capability_digest) ||
    typeof value.bundle_digest !== "string" ||
    !SHA256.test(value.bundle_digest) ||
    typeof value.receipt_mac !== "string" ||
    !SHA256.test(value.receipt_mac)
  ) {
    throw new BridgeContractError();
  }
  return value as BootstrapBinding;
}

export async function loadOpenCodeBootstrap(bootstrapURL: URL): Promise<BootstrapBinding> {
  try {
    if (
      !(bootstrapURL instanceof URL) ||
      bootstrapURL.protocol !== "file:" ||
      bootstrapURL.search !== "" ||
      bootstrapURL.hash !== ""
    ) {
      throw new BridgeContractError();
    }
    const bootstrapPath = fileURLToPath(bootstrapURL);
    if (path.basename(bootstrapPath) !== BOOTSTRAP_NAME) throw new BridgeContractError();
    const raw = await readStableRegularFile(bootstrapPath, {
      minimum: 2,
      maximum: MAX_BOOTSTRAP_BYTES,
    });
    const parsed = JSON.parse(raw.toString("utf8")) as unknown;
    const bootstrap = validateBootstrap(parsed);
    const canonical = encodeCanonicalJson(bootstrap);
    if (canonical.length !== raw.length || !timingSafeEqual(canonical, raw)) {
      throw new BridgeContractError();
    }

    const bundlePath = path.join(path.dirname(bootstrapPath), BUNDLE_NAME);
    const bundle = await readStableRegularFile(bundlePath, {
      minimum: 1,
      maximum: MAX_BUNDLE_BYTES,
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
