// Frozen from OpenCode 1.18.3 (127bdb30784d508cc556c71a0f32b508a3061517).
// Runtime events also carry an undeclared optional top-level `id`.
export type OpenCodeToolStatus = "pending" | "running" | "completed" | "error";

export type OpenCodeEvent = Readonly<{
  id?: unknown;
  type?: unknown;
  properties?: unknown;
}>;

export type OpenCodeHooks = Readonly<{
  event: (input: { event: unknown }) => Promise<void>;
  dispose: () => Promise<void>;
}>;

export type OpenCodePlugin = (input: unknown) => Promise<OpenCodeHooks>;
