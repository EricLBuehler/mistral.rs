#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import {
  ensureDir,
  fileUrlFromPath,
  normalizePath,
  parseArgs,
  readJsonIfExists,
  toBool,
  toRelativePath,
  withTimeout,
  writeJson,
} from "./lib/preview-utils.mjs";

function detectHtmxUsage(html) {
  return /(?:\shx-[a-z-]+=)|(?:\bhtmx\b)/i.test(html);
}

function hasHtmxScriptTag(html) {
  return /<script[^>]+src=["'][^"']*htmx[^"']*["'][^>]*>/i.test(html);
}

async function injectHtmxRuntime({
  html,
  outputDir,
  htmxMode,
  localRuntimePath,
  networkEnabled,
}) {
  const result = {
    html,
    runtime_source: "none",
    runtime_detail: "not-required",
    htmx_required: false,
  };

  const htmxRequired = detectHtmxUsage(html);
  result.htmx_required = htmxRequired;
  if (!htmxRequired || hasHtmxScriptTag(html)) {
    if (htmxRequired) {
      result.runtime_detail = "script-already-present";
    }
    return result;
  }

  const localExists = await fs
    .access(localRuntimePath)
    .then(() => true)
    .catch(() => false);

  if (htmxMode !== "network" && localExists) {
    const vendorDir = path.join(outputDir, "vendor");
    await ensureDir(vendorDir);
    const copiedRuntime = path.join(vendorDir, "htmx.min.js");
    await fs.copyFile(localRuntimePath, copiedRuntime);
    result.html = html.replace(
      "</body>",
      `  <script src="./vendor/htmx.min.js"></script>\n</body>`,
    );
    result.runtime_source = "local";
    result.runtime_detail = toRelativePath(localRuntimePath);
    return result;
  }

  if (networkEnabled || htmxMode === "network") {
    result.html = html.replace(
      "</body>",
      '  <script src="https://unpkg.com/htmx.org@1.9.12"></script>\n</body>',
    );
    result.runtime_source = "network";
    result.runtime_detail = "https://unpkg.com/htmx.org@1.9.12";
    return result;
  }

  throw new Error(
    "HTMX runtime required but local runtime was not found and network mode is disabled",
  );
}

function updateManifestWithPreview(manifest, run) {
  if (!manifest.preview) {
    manifest.preview = { required: true, runs: [] };
  }
  if (!Array.isArray(manifest.preview.runs)) {
    manifest.preview.runs = [];
  }

  const existingIdx = manifest.preview.runs.findIndex(
    (item) => item.artifact_id === run.artifact_id,
  );
  if (existingIdx >= 0) {
    manifest.preview.runs[existingIdx] = run;
  } else {
    manifest.preview.runs.push(run);
  }

  if (!Array.isArray(manifest.variants)) {
    manifest.variants = [];
  }
  const variantName = `${run.artifact_id}-preview`;
  const variant = {
    name: variantName,
    files: [run.html, run.screenshot, run.report],
  };
  const variantIdx = manifest.variants.findIndex((item) => item.name === variantName);
  if (variantIdx >= 0) {
    manifest.variants[variantIdx] = variant;
  } else {
    manifest.variants.push(variant);
  }
}

async function writeReportAndManifest({
  reportFile,
  report,
  manifestPath,
  artifactType,
}) {
  await writeJson(reportFile, report);

  if (!manifestPath) {
    return;
  }

  const manifest =
    (await readJsonIfExists(manifestPath, null)) ??
    {
      artifact_type: artifactType || "ui",
      variants: [],
      generated_at: new Date().toISOString(),
    };

  updateManifestWithPreview(manifest, {
    artifact_id: report.artifact_id,
    status: report.status,
    html: report.preview_html,
    screenshot: report.screenshot,
    report: report.preview_report,
    runtime_source: report.runtime_source,
    runtime_detail: report.runtime_detail,
  });
  manifest.generated_at = new Date().toISOString();
  await writeJson(manifestPath, manifest);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const input = args.input || args.html;
  if (!input) {
    throw new Error("Missing required argument: --input <preview-html>");
  }

  const artifactId = args["artifact-id"] || "preview";
  const artifactType = args["artifact-type"] || "ui";
  const outputDir = normalizePath(args["output-dir"], path.join("dist", "previews", artifactId));
  const screenshotName = args["screenshot-name"] || "screenshot.png";
  const reportName = args["report-name"] || "preview-report.json";
  const previewName = args["preview-name"] || "preview.html";
  const timeoutMs = Number(args["timeout-ms"] || 20000);
  const width = Number(args.width || 1280);
  const height = Number(args.height || 720);
  const fullPage = toBool(args["full-page"], true);
  const softFail = toBool(args["soft-fail"], true);
  const networkEnabled = toBool(args["network-enabled"], false);
  const htmxMode = (args["htmx-mode"] || "local").toLowerCase();
  const localRuntimePath = normalizePath(args["htmx-local-path"], "assets/vendor/htmx.min.js");
  const manifestPath = args.manifest ? normalizePath(args.manifest) : null;

  await ensureDir(outputDir);
  const inputPath = normalizePath(input);
  const previewHtmlPath = path.join(outputDir, previewName);
  const screenshotPath = path.join(outputDir, screenshotName);
  const reportPath = path.join(outputDir, reportName);

  const report = {
    artifact_id: artifactId,
    artifact_type: artifactType,
    status: "failed",
    generated_at: new Date().toISOString(),
    source_html: toRelativePath(inputPath),
    preview_html: toRelativePath(previewHtmlPath),
    screenshot: toRelativePath(screenshotPath),
    preview_report: toRelativePath(reportPath),
    runtime_source: "none",
    runtime_detail: "none",
    diagnostics: {
      console: [],
      page_errors: [],
      request_failures: [],
    },
    error: null,
  };

  try {
    let html = await fs.readFile(inputPath, "utf8");
    const injected = await injectHtmxRuntime({
      html,
      outputDir,
      htmxMode,
      localRuntimePath,
      networkEnabled,
    });
    html = injected.html;
    report.runtime_source = injected.runtime_source;
    report.runtime_detail = injected.runtime_detail;
    report.htmx_required = injected.htmx_required;

    await fs.writeFile(previewHtmlPath, html, "utf8");
  } catch (error) {
    report.error = error.message;
    await writeReportAndManifest({
      reportFile: reportPath,
      report,
      manifestPath,
      artifactType,
    });
    console.error(`❌ Failed preparing preview HTML: ${error.message}`);
    process.exit(2);
    return;
  }

  let chromium = null;
  try {
    const playwright = await import("playwright");
    chromium = playwright.chromium;
  } catch (error) {
    report.status = "skipped";
    report.error =
      "Playwright is unavailable. Install dependencies (npm install) or use browser_renderer tool.";
    await writeReportAndManifest({
      reportFile: reportPath,
      report,
      manifestPath,
      artifactType,
    });

    if (softFail) {
      console.warn(`⚠️ ${report.error}`);
      process.exit(0);
      return;
    }

    process.exit(2);
    return;
  }

  let browser = null;
  try {
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width, height } });
    const page = await context.newPage();

    page.on("console", (message) => {
      report.diagnostics.console.push({
        type: message.type(),
        text: message.text(),
      });
    });
    page.on("pageerror", (error) => {
      report.diagnostics.page_errors.push(error.message);
    });
    page.on("requestfailed", (request) => {
      report.diagnostics.request_failures.push({
        url: request.url(),
        failure: request.failure()?.errorText || "unknown",
      });
    });

    const targetUrl = fileUrlFromPath(previewHtmlPath);
    await withTimeout(
      () => page.goto(targetUrl, { waitUntil: "networkidle", timeout: timeoutMs }),
      timeoutMs + 1000,
      "browser preview navigation",
    );

    await page.screenshot({
      path: screenshotPath,
      fullPage,
    });

    report.status = "success";
  } catch (error) {
    report.status = "failed";
    report.error = error.message;
    if (!softFail) {
      console.error(`❌ Preview render failed: ${error.message}`);
    } else {
      console.warn(`⚠️ Preview render failed: ${error.message}`);
    }
  } finally {
    if (browser) {
      await browser.close();
    }
  }

  await writeReportAndManifest({
    reportFile: reportPath,
    report,
    manifestPath,
    artifactType,
  });

  if (report.status === "failed" && !softFail) {
    process.exit(2);
    return;
  }

  if (report.status === "success") {
    console.log("✅ Browser preview rendering complete");
  } else {
    console.log("ℹ️ Browser preview completed with non-success status");
  }
}

main().catch((error) => {
  console.error(`❌ render-preview failed: ${error.message}`);
  process.exit(2);
});
