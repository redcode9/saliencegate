export const MAX_CAPTURE_BATCH_BYTES = 2 * 1024 * 1024;
export const MAX_CAPTURE_EVENT_BYTES = 64 * 1024;
export const MAX_CAPTURE_JSON_DEPTH = 32;
export const MAX_CAPTURE_JSON_ITEMS = 10_000;
export const MAX_CAPTURE_JSON_STRING_BYTES = 1024 * 1024;
export const MAX_CAPTURE_BATCH_CHUNKS = 1_000;
export const MAX_CAPTURE_EVENTS_PER_CHUNK = 999;
export const MAX_CAPTURE_SESSION_ID_BYTES = 256 * 1024;
export const MAX_CAPTURE_EVENT_ID_BYTES = 16 * 1024;
export const MAX_CAPTURE_CALL_ID_BYTES = 16 * 1024;
export const MAX_CAPTURE_TOOL_NAME_BYTES = 1024;

export class BridgeContractError extends Error {
  constructor() {
    super("capture bridge contract is invalid");
    this.name = "BridgeContractError";
  }
}

export type CanonicalJson =
  | null
  | boolean
  | number
  | string
  | CanonicalJson[]
  | { [key: string]: CanonicalJson };

export type BootstrapBinding = Readonly<{
  schema_version: "integration-bootstrap/v1";
  profile: "opencode-plugin/v1" | "pi-extension/v1";
  connection_id: string;
  launcher_path: string;
  capability_digest: string;
  bundle_digest: string;
  receipt_mac: string;
}>;

export type CaptureBatchDocument = Readonly<{
  schema_version: "capture-batch/v1";
  bootstrap: BootstrapBinding;
  batch_id: string;
  session_id: string;
  workspace_path?: string;
  chunk_index: number;
  chunk_count: number;
  events: readonly CanonicalJson[];
}>;

export type CaptureChunk = Readonly<{
  document: CaptureBatchDocument;
  bytes: Buffer;
}>;

export type CaptureChunkCoverage = Readonly<{
  complete: boolean;
  missingIndexes: number[];
}>;
