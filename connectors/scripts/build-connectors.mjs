import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const assetDirectory = path.join(root, "src", "saliencegate", "integrations", "assets");
const bootstrapReference = "./saliencegate.bootstrap.json";
const binding = `export const saliencegateBootstrap = new URL("${bootstrapReference}", import.meta.url);\n`;
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
    target: "es2023",
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
  return { bytes, outputDirectory };
}

for (const connector of connectors) {
  const first = await bundle(connector);
  const second = await bundle(connector);
  assert.deepEqual(
    first.bytes,
    second.bytes,
    `${connector.name} bundle output is not deterministic`,
  );
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
