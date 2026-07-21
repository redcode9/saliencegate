// Frozen structurally from @earendil-works/pi-coding-agent 0.80.10
// (8dc78834cde4e329284cf505f9e3f99763df5529). The provider package is not a
// runtime or development dependency of the standalone extension.
export const PI_UPSTREAM_VERSION = "0.80.10" as const;
export const PI_UPSTREAM_COMMIT =
  "8dc78834cde4e329284cf505f9e3f99763df5529" as const;

export type PiSessionStartReason = "startup" | "reload" | "new" | "resume" | "fork";
export type PiSessionShutdownReason = "quit" | "reload" | "new" | "resume" | "fork";
export type PiCompactionReason = "manual" | "threshold" | "overflow";

export type PiSessionStartEvent = Readonly<{
  type: "session_start";
  reason: PiSessionStartReason;
  previousSessionFile?: string;
}>;

export type PiBeforeAgentStartEvent = Readonly<{
  type: "before_agent_start";
  prompt: string;
  images?: unknown;
  systemPrompt: string;
  systemPromptOptions: unknown;
}>;

export type PiToolExecutionStartEvent = Readonly<{
  type: "tool_execution_start";
  toolCallId: string;
  toolName: string;
  args: unknown;
}>;

export type PiToolExecutionEndEvent = Readonly<{
  type: "tool_execution_end";
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError: boolean;
}>;

export type PiAgentSettledEvent = Readonly<{ type: "agent_settled" }>;

export type PiSessionCompactEvent = Readonly<{
  type: "session_compact";
  compactionEntry: unknown;
  fromExtension: boolean;
  reason: PiCompactionReason;
  willRetry: boolean;
}>;

export type PiSessionTreeEvent = Readonly<{
  type: "session_tree";
  newLeafId: string | null;
  oldLeafId: string | null;
  summaryEntry?: unknown;
  fromExtension?: boolean;
}>;

export type PiSessionShutdownEvent = Readonly<{
  type: "session_shutdown";
  reason: PiSessionShutdownReason;
  targetSessionFile?: string;
}>;

export type PiObservedEvent =
  | PiBeforeAgentStartEvent
  | PiToolExecutionStartEvent
  | PiToolExecutionEndEvent
  | PiAgentSettledEvent
  | PiSessionCompactEvent
  | PiSessionTreeEvent
  | PiSessionShutdownEvent;

export type PiExtensionContext = Readonly<{
  sessionManager: Readonly<{
    getSessionId(): string;
  }>;
}>;

export type PiExtensionHandler<Event> = (
  event: Event,
  context: PiExtensionContext,
) => Promise<void> | void;

export interface PiExtensionAPI {
  on(event: "session_start", handler: PiExtensionHandler<PiSessionStartEvent>): void;
  on(
    event: "before_agent_start",
    handler: PiExtensionHandler<PiBeforeAgentStartEvent>,
  ): void;
  on(
    event: "tool_execution_start",
    handler: PiExtensionHandler<PiToolExecutionStartEvent>,
  ): void;
  on(
    event: "tool_execution_end",
    handler: PiExtensionHandler<PiToolExecutionEndEvent>,
  ): void;
  on(event: "agent_settled", handler: PiExtensionHandler<PiAgentSettledEvent>): void;
  on(event: "session_compact", handler: PiExtensionHandler<PiSessionCompactEvent>): void;
  on(event: "session_tree", handler: PiExtensionHandler<PiSessionTreeEvent>): void;
  on(
    event: "session_shutdown",
    handler: PiExtensionHandler<PiSessionShutdownEvent>,
  ): void;
}

export type PiExtension = (api: PiExtensionAPI) => Promise<void> | void;
