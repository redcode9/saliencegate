import { Buffer } from "node:buffer";

import {
  BridgeContractError,
  MAX_CAPTURE_JSON_DEPTH,
  MAX_CAPTURE_JSON_ITEMS,
  MAX_CAPTURE_JSON_STRING_BYTES,
  type CanonicalJson,
} from "./contracts.ts";

type Budget = {
  items: number;
  stringBytes: number;
  active: Set<object>;
};

export function isWellFormedUnicode(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  return true;
}

function addString(value: string, budget: Budget): void {
  if (!isWellFormedUnicode(value)) throw new BridgeContractError();
  budget.stringBytes += Buffer.byteLength(value, "utf8");
  if (budget.stringBytes > MAX_CAPTURE_JSON_STRING_BYTES) throw new BridgeContractError();
}

function dataValue(container: object, key: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(container, key);
  if (descriptor === undefined || !("value" in descriptor) || !descriptor.enumerable) {
    throw new BridgeContractError();
  }
  return descriptor.value;
}

function canonicalize(value: unknown, depth: number, budget: Budget): CanonicalJson {
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
      const keys = Object.keys(value);
      if (
        keys.length !== value.length ||
        keys.some((key, index) => key !== String(index))
      ) {
        throw new BridgeContractError();
      }
      return keys.map((key) => canonicalize(dataValue(value, key), depth + 1, budget));
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) throw new BridgeContractError();
    const ownKeys = Reflect.ownKeys(value);
    if (ownKeys.some((key) => typeof key !== "string")) throw new BridgeContractError();
    const keys = (ownKeys as string[]).sort();
    const result: Record<string, CanonicalJson> = {};
    for (const key of keys) {
      addString(key, budget);
      Object.defineProperty(result, key, {
        configurable: true,
        enumerable: true,
        value: canonicalize(dataValue(value, key), depth + 1, budget),
        writable: true,
      });
    }
    return result;
  } finally {
    budget.active.delete(value);
  }
}

export function canonicalizeJson(value: unknown): CanonicalJson {
  try {
    return canonicalize(value, 0, { items: 0, stringBytes: 0, active: new Set<object>() });
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

export function encodeCanonicalJson(value: unknown): Buffer {
  try {
    return Buffer.from(JSON.stringify(canonicalizeJson(value)), "utf8");
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}
