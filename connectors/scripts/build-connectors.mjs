import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const assetDirectory = path.join(root, "src", "saliencegate", "integrations", "assets");
const bootstrapReference = "./saliencegate.bootstrap.json";
const binding = `export const saliencegateBootstrap = new URL("${bootstrapReference}", import.meta.url);\n`;
const manifestPaths = [
  "package.json",
  "connectors/bridge-core/package.json",
  "connectors/opencode/package.json",
  "connectors/pi/package.json",
];
const connectors = [
  {
    name: "OpenCode",
    directory: "opencode",
    asset: "opencode-plugin.js",
    sourcefile: "opencode-runtime-entry.ts",
    entry: `
      import { createOpenCodePlugin } from "./connectors/opencode/src/plugin.ts";
      declare const saliencegateBootstrap: URL;
      const plugin = {
        id: "saliencegate",
        server: createOpenCodePlugin({ bootstrapURL: saliencegateBootstrap }),
      };
      export default plugin;
    `,
  },
  {
    name: "Pi",
    directory: "pi",
    asset: "pi-extension.js",
    sourcefile: "pi-runtime-entry.ts",
    entry: `
      import { createPiExtension } from "./connectors/pi/src/extension.ts";
      declare const saliencegateBootstrap: URL;
      export default createPiExtension({ bootstrapURL: saliencegateBootstrap });
    `,
  },
];

async function bundle(connector) {
  const outputDirectory = path.join(root, "connectors", ".build", connector.directory);
  const result = await build({
    absWorkingDir: root,
    banner: { js: binding.trimEnd() },
    bundle: true,
    charset: "utf8",
    entryNames: "saliencegate",
    format: "esm",
    legalComments: "none",
    logLevel: "silent",
    metafile: true,
    minify: false,
    outdir: outputDirectory,
    platform: "node",
    sourcemap: false,
    stdin: {
      contents: connector.entry,
      loader: "ts",
      resolveDir: root,
      sourcefile: connector.sourcefile,
    },
    target: "node22.19",
    treeShaking: true,
    write: false,
  });
  assert.equal(result.outputFiles.length, 1);
  const bytes = Buffer.from(result.outputFiles[0].contents);
  const text = bytes.toString("utf8");
  assert.ok(text.startsWith(binding));
  assert.equal(text.split(binding).length - 1, 1);
  assert.equal(text.split("new URL(").length - 1, 1);
  assert.equal(text.split("import.meta.url").length - 1, 1);
  assert.equal(text.split(bootstrapReference).length - 1, 1);
  assert.doesNotMatch(text, /sourceMappingURL/);
  assert.doesNotMatch(text, /@saliencegate\//);
  for (const output of Object.values(result.metafile.outputs)) {
    assert.ok(output.imports.every((item) => item.path.startsWith("node:")));
  }
  return { bytes, metafile: result.metafile, outputDirectory };
}

async function readJson(relativePath) {
  return JSON.parse(await readFile(path.join(root, relativePath), "utf8"));
}

function runtimeDependencies(manifest) {
  return {
    ...(manifest.dependencies ?? {}),
    ...(manifest.optionalDependencies ?? {}),
    ...(manifest.peerDependencies ?? {}),
  };
}

async function auditRuntimeDependencies(metafiles) {
  const manifests = new Map();
  for (const manifestPath of manifestPaths) {
    manifests.set(manifestPath, await readJson(manifestPath));
  }
  const rootManifest = manifests.get("package.json");
  assert.equal(rootManifest.packageManager, "npm@10.9.3");
  assert.deepEqual(rootManifest.engines, { node: "22.19.0", npm: "10.9.3" });
  assert.deepEqual(runtimeDependencies(rootManifest), {});

  const workspaceManifests = [...manifests.entries()].filter(
    ([manifestPath]) => manifestPath !== "package.json",
  );
  const workspaceVersions = new Map(
    workspaceManifests.map(([, manifest]) => [manifest.name, manifest.version]),
  );
  for (const [manifestPath, manifest] of workspaceManifests) {
    for (const [dependency, version] of Object.entries(runtimeDependencies(manifest))) {
      assert.equal(
        workspaceVersions.get(dependency),
        version,
        `${manifestPath} has external or version-skewed runtime dependency ${dependency}`,
      );
    }
  }

  const lock = await readJson("package-lock.json");
  assert.equal(lock.lockfileVersion, 3);
  assert.equal(lock.name, rootManifest.name);
  assert.equal(lock.version, rootManifest.version);
  for (const [location, entry] of Object.entries(lock.packages)) {
    if (!location.startsWith("node_modules/") || entry.link === true) continue;
    assert.equal(entry.dev, true, `production dependency leaked into lock: ${location}`);
  }
  for (const [manifestPath, manifest] of workspaceManifests) {
    const workspacePath = path.posix.dirname(manifestPath);
    const lockEntry = lock.packages[workspacePath];
    assert.ok(lockEntry, `workspace missing from lock: ${workspacePath}`);
    assert.deepEqual(runtimeDependencies(lockEntry), runtimeDependencies(manifest));
  }

  let inputCount = 0;
  let externalImportCount = 0;
  for (const metafile of metafiles) {
    const inputs = Object.keys(metafile.inputs);
    inputCount += inputs.length;
    assert.ok(inputs.length > 0);
    assert.ok(
      inputs.every(
        (input) => !input.replaceAll("\\", "/").split("/").includes("node_modules"),
      ),
      "bundle metafile contains an external package input",
    );
    for (const output of Object.values(metafile.outputs)) {
      externalImportCount += output.imports.length;
      assert.ok(
        output.imports.every(
          (item) => item.external === true && item.path.startsWith("node:"),
        ),
        "bundle metafile contains an external non-builtin runtime import",
      );
    }
  }
  return {
    bundle_external_builtin_imports: externalImportCount,
    bundle_inputs: inputCount,
    external_runtime_dependencies: 0,
    schema_version: "connector-runtime-dependency-audit/v1",
  };
}

const allowedArguments = new Set(["--audit-runtime", "--check"]);
assert.ok(
  process.argv.slice(2).every((argument) => allowedArguments.has(argument)),
  "usage: build-connectors.mjs [--check] [--audit-runtime]",
);
const metafiles = [];
for (const connector of connectors) {
  const first = await bundle(connector);
  const second = await bundle(connector);
  assert.deepEqual(
    first.bytes,
    second.bytes,
    `${connector.name} bundle output is not deterministic`,
  );
  assert.deepEqual(
    first.metafile,
    second.metafile,
    `${connector.name} bundle metafile is not deterministic`,
  );
  metafiles.push(first.metafile);
  const assetPath = path.join(assetDirectory, connector.asset);
  if (process.argv.includes("--check")) {
    assert.deepEqual(
      Buffer.from(await readFile(assetPath)),
      first.bytes,
      `embedded ${connector.name} asset is stale`,
    );
  } else {
    await mkdir(first.outputDirectory, { recursive: true });
    await mkdir(assetDirectory, { recursive: true });
    await writeFile(
      path.join(first.outputDirectory, "saliencegate.js"),
      first.bytes,
      { mode: 0o600 },
    );
    await writeFile(assetPath, first.bytes, { mode: 0o644 });
  }
}

if (process.argv.includes("--audit-runtime")) {
  process.stdout.write(`${JSON.stringify(await auditRuntimeDependencies(metafiles))}\n`);
}
