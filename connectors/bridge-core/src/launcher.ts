import path from "node:path";

import { BridgeContractError } from "./contracts.ts";

export const PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS = Object.freeze([
  "ANTHROPIC_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORGANIZATION",
  "OPENAI_ORG_ID",
  "OPENAI_PROJECT",
  "OPENAI_PROJECT_ID",
] as const);

const PROVIDER_CREDENTIAL_ENVIRONMENT_KEY_SET = new Set<string>(
  PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS,
);

export type LauncherEnvironment = Readonly<
  Record<string, string | undefined>
>;

export type LauncherInvocation = Readonly<{
  file: string;
  arguments: string[];
  options: {
    shell: false;
    windowsHide: true;
    windowsVerbatimArguments?: true;
    env: Record<string, string>;
    stdio: ["pipe", "ignore", "ignore"];
  };
}>;

export function copyLauncherEnvironment(
  value: LauncherEnvironment,
): Record<string, string> {
  try {
    const result: Record<string, string> = {};
    for (const key of Object.keys(value)) {
      if (key.includes("\0")) throw new BridgeContractError();
      if (PROVIDER_CREDENTIAL_ENVIRONMENT_KEY_SET.has(key.toUpperCase())) continue;
      const item = Reflect.get(value, key) as unknown;
      if (typeof item !== "string") continue;
      if (item.includes("\0")) throw new BridgeContractError();
      Object.defineProperty(result, key, {
        configurable: true,
        enumerable: true,
        value: item,
        writable: true,
      });
    }
    return result;
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}

export function launcherInvocation(input: {
  platform: NodeJS.Platform;
  launcherPath: string;
  environment: LauncherEnvironment;
}): LauncherInvocation {
  try {
    if (
      typeof input.launcherPath !== "string" ||
      input.launcherPath.length === 0 ||
      input.launcherPath.length > 4_096 ||
      input.launcherPath.includes("\0")
    ) {
      throw new BridgeContractError();
    }
    const environment = copyLauncherEnvironment(input.environment);
    const options = {
      shell: false as const,
      windowsHide: true as const,
      env: environment,
      stdio: ["pipe", "ignore", "ignore"] as ["pipe", "ignore", "ignore"],
    };
    if (input.platform !== "win32") {
      if (!path.posix.isAbsolute(input.launcherPath)) throw new BridgeContractError();
      return { file: input.launcherPath, arguments: [], options };
    }

    if (!path.win32.isAbsolute(input.launcherPath) || input.launcherPath.includes('"')) {
      throw new BridgeContractError();
    }
    const systemRoots = Object.entries(environment).filter(
      ([key]) => key.toUpperCase() === "SYSTEMROOT",
    );
    if (systemRoots.length !== 1) throw new BridgeContractError();
    const systemRoot = systemRoots[0]![1];
    if (
      typeof systemRoot !== "string" ||
      !/^[A-Za-z]:\\Windows$/i.test(systemRoot) ||
      systemRoot.includes('"')
    ) {
      throw new BridgeContractError();
    }
    const file = path.win32.join(systemRoot, "System32", "cmd.exe");
    const windowsEnvironment = Object.fromEntries(
      Object.entries(environment).filter(
        ([key]) => key.toUpperCase() !== "SALIENCEGATE_LAUNCHER",
      ),
    );
    return {
      file,
      arguments: ["/d", "/v:off", "/s", "/c", '""%SALIENCEGATE_LAUNCHER%""'],
      options: {
        ...options,
        env: { ...windowsEnvironment, SALIENCEGATE_LAUNCHER: input.launcherPath },
        windowsVerbatimArguments: true,
      },
    };
  } catch (error) {
    if (error instanceof BridgeContractError) throw error;
    throw new BridgeContractError();
  }
}
