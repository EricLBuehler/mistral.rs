#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import {
  ensureDir,
  normalizePath,
  parseArgs,
  toRelativePath,
  writeJson,
} from "./lib/preview-utils.mjs";

function run(command, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(stderr || stdout || `Command failed: ${command}`));
      }
    });
  });
}

async function buildWithEsbuildApi(options) {
  const esbuild = await import("esbuild");
  await esbuild.build(options);
}

async function buildWithNpx(entryFile, outputFile) {
  await run("npx", [
    "-y",
    "esbuild",
    entryFile,
    "--bundle",
    "--format=esm",
    "--platform=browser",
    `--outfile=${outputFile}`,
  ]);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const entry = args.entry || args.input;
  if (!entry) {
    throw new Error("Missing required argument: --entry <path>");
  }

  const entryPath = normalizePath(entry);
  const artifactId = args["artifact-id"] || "react-preview";
  const outputDir = normalizePath(args.outdir, path.join("dist", "previews", artifactId));
  const outputFile = path.join(outputDir, args.outfile || "preview.js");
  const htmlFile = path.join(outputDir, args["html-file"] || "preview.html");
  const reportFile = path.join(outputDir, args["report-file"] || "compile-report.json");

  await ensureDir(outputDir);

  const report = {
    status: "failed",
    artifact_id: artifactId,
    entry: toRelativePath(entryPath),
    output_js: toRelativePath(outputFile),
    output_html: toRelativePath(htmlFile),
    compiled_at: new Date().toISOString(),
    compiler: null,
    error: null,
  };

  try {
    await buildWithEsbuildApi({
      entryPoints: [entryPath],
      bundle: true,
      format: "esm",
      platform: "browser",
      outfile: outputFile,
    });
    report.compiler = "esbuild-api";
  } catch (error) {
    try {
      await buildWithNpx(entryPath, outputFile);
      report.compiler = "npx-esbuild";
    } catch (fallbackError) {
      report.error = `${error.message}\n--- fallback ---\n${fallbackError.message}`;
      await writeJson(reportFile, report);
      console.error("❌ Failed to compile TSX preview");
      console.error(report.error);
      process.exit(2);
      return;
    }
  }

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>React Preview</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="./${path.basename(outputFile)}"></script>
</body>
</html>
`;

  await fs.writeFile(htmlFile, html, "utf8");
  report.status = "success";
  await writeJson(reportFile, report);

  console.log("✅ TSX preview compilation complete");
  console.log(`- JS: ${toRelativePath(outputFile)}`);
  console.log(`- HTML: ${toRelativePath(htmlFile)}`);
  console.log(`- Report: ${toRelativePath(reportFile)}`);
}

main().catch(async (error) => {
  console.error(`❌ compile-tsx-preview failed: ${error.message}`);
  process.exit(2);
});
