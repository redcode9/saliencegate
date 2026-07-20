import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const outputDirectory = path.join(root, "connectors", ".build", "opencode");
const outputPath = path.join(outputDirectory, "saliencegate.js");
const assetDirectory = path.join(root, "src", "saliencegate", "integrations", "assets");
const assetPath = path.join(assetDirectory, "opencode-plugin.js");
const bootstrapReference = "./saliencegate.bootstrap.json";
const binding = `export const saliencegateBootstrap = new URL("${bootstrapReference}", import.meta.url);\n`;
const entry = `
  import { createOpenCodePlugin } from "./connectors/opencode/src/plugin.ts";
  declare const saliencegateBootstrap: URL;
  const plugin = {
    id: "saliencegate",
    server: createOpenCodePlugin({ bootstrapURL: saliencegateBootstrap }),
  };
  export default plugin;
`;

async function bundle() {
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
      contents: entry,
      loader: "ts",
      resolveDir: root,
      sourcefile: "opencode-runtime-entry.ts",
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
  return bytes;
}

const first = await bundle();
const second = await bundle();
assert.deepEqual(first, second, "OpenCode bundle output is not deterministic");

if (process.argv.includes("--check")) {
  assert.deepEqual(
    Buffer.from(await readFile(assetPath)),
    first,
    "embedded OpenCode asset is stale",
  );
} else {
  await mkdir(outputDirectory, { recursive: true });
  await mkdir(assetDirectory, { recursive: true });
  await writeFile(outputPath, first, { mode: 0o600 });
  await writeFile(assetPath, first, { mode: 0o644 });
}
