import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { copyText } from "../src/clipboard.js";

const websiteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distDir = path.join(websiteDir, "dist");
const assetsDir = path.join(distDir, "assets");

async function readAssets(extension) {
  const names = (await readdir(assetsDir)).filter((name) => name.endsWith(extension));
  return Promise.all(names.map((name) => readFile(path.join(assetsDir, name), "utf8")));
}

test("builds the focused landing page", async () => {
  const html = await readFile(path.join(distDir, "index.html"), "utf8");
  const scripts = (await readAssets(".js")).join("\n");
  const styles = (await readAssets(".css")).join("\n");

  assert.match(html, /mistral<span>\.rs<\/span>/);
  assert.ok(html.includes("Fast, flexible LLM inference"));
  assert.ok(html.includes("Go from install to inference in one command."));
  assert.ok(html.includes("mistralrs serve -m Qwen/Qwen3.8-27B --quant 4"));
  assert.ok(html.includes("mistralrs run -m google/gemma-4-E4B-it --quant 4"));
  assert.ok(html.includes("50+ model architectures"));
  assert.ok(html.includes("https://docs.mistralrs.dev/quickstart/"));
  assert.ok(html.includes("install from source"));
  assert.doesNotMatch(html, /MIT licensed|not affiliated with Mistral AI/);
  assert.ok(scripts.includes("curl -fsSL https://mistralrs.dev/install.sh | sh"));
  assert.ok(scripts.includes("irm https://mistralrs.dev/install.ps1 | iex"));
  assert.ok(styles.includes("SFMono-Regular"));
  assert.ok(styles.includes("overflow-x:hidden"));
  assert.doesNotMatch(`${html}\n${scripts}\n${styles}`, /Geist|fonts\.googleapis\.com/);
  assert.doesNotMatch(`${html}\n${scripts}`, /mistralrs\.(?:sh|ps1)/);
});

test("copies the root installers into the Cloudflare Pages output", async () => {
  const outputShell = await readFile(path.join(distDir, "install.sh"));
  const outputPowerShell = await readFile(path.join(distDir, "install.ps1"));
  const sourceShell = await readFile(path.join(websiteDir, "..", "install.sh"));
  const sourcePowerShell = await readFile(path.join(websiteDir, "..", "install.ps1"));
  const headers = await readFile(path.join(distDir, "_headers"), "utf8");

  assert.ok(outputShell.equals(sourceShell));
  assert.ok(outputPowerShell.equals(sourcePowerShell));
  assert.match(headers, /\/assets\/\*/);
  assert.match(headers, /\/install\.sh/);
  assert.match(headers, /\/install\.ps1/);
  await assert.rejects(access(path.join(websiteDir, "public", "install.sh")));
  await assert.rejects(access(path.join(websiteDir, "public", "install.ps1")));
  await assert.rejects(access(path.join(distDir, "server")));
  await assert.rejects(access(path.join(distDir, "client")));
});

test("builds the blog from release posts", async () => {
  const distBlogDir = path.join(distDir, "blog");
  const indexHtml = await readFile(path.join(distBlogDir, "index.html"), "utf8");
  const landingHtml = await readFile(path.join(distDir, "index.html"), "utf8");
  const css = await readFile(path.join(distDir, "blog.css"), "utf8");

  assert.ok(css.includes(".post-body"));
  assert.ok(landingHtml.includes('href="/blog/"'));
  assert.ok(indexHtml.includes("up to 19.5% higher serving throughput than vLLM"));
  assert.ok(indexHtml.includes("By Eric Buehler"));

  const slugMatch = indexHtml.match(/href="([^"]+\.html)"/);
  assert.ok(slugMatch, "blog index should link at least one post");
  const slug = slugMatch[1].replace(/\.html$/, "");
  const postHtml = await readFile(path.join(distBlogDir, slugMatch[1]), "utf8");
  const figure = await readFile(
    path.join(distBlogDir, "assets", slug, "target_throughput.png"),
  );
  const dflashFigure = await readFile(
    path.join(distBlogDir, "assets", slug, "dflash_throughput.png"),
  );

  assert.ok(postHtml.includes("<table>"));
  assert.ok(postHtml.includes("4,914.05"));
  assert.ok(postHtml.includes("9.7-15.5%"));
  assert.ok(postHtml.includes("DFlash2 widens the gap"));
  assert.ok(postHtml.includes("2,993.44"));
  assert.ok(postHtml.includes(`src="assets/${slug}/dflash_throughput.png"`));
  assert.ok(postHtml.includes(`src="assets/${slug}/target_throughput.png"`));
  assert.ok(postHtml.includes('property="og:title"'));
  assert.ok(postHtml.includes("<span class=\"post-author\">Eric Buehler</span>"));
  assert.ok(postHtml.includes("releases/v0.9.3/report.md"));
  assert.ok(figure.length > 0);
  assert.ok(dflashFigure.length > 0);
});

test("copies with the Clipboard API when available", async () => {
  const writes = [];
  const copied = await copyText("install command", {
    clipboard: { writeText: async (text) => writes.push(text) },
    document: undefined,
  });

  assert.equal(copied, true);
  assert.deepEqual(writes, ["install command"]);
});

test("falls back when the Clipboard API rejects the write", async () => {
  let selected = false;
  let removed = false;
  let copyHandler;
  let copiedText;
  const textarea = {
    setAttribute() {},
    style: {},
    select() {
      selected = true;
    },
    remove() {
      removed = true;
    },
  };
  const document = {
    body: { append() {} },
    createElement: () => textarea,
    addEventListener: (event, handler) => {
      if (event === "copy") copyHandler = handler;
    },
    removeEventListener: (event, handler) => {
      if (event === "copy" && handler === copyHandler) copyHandler = undefined;
    },
    execCommand: (command) => {
      copyHandler({
        clipboardData: { setData: (_, text) => (copiedText = text) },
        preventDefault() {},
      });
      return command === "copy";
    },
  };
  const copied = await copyText("install command", {
    clipboard: {
      writeText: async () => {
        throw new Error("clipboard unavailable");
      },
    },
    document,
  });

  assert.equal(copied, true);
  assert.equal(textarea.value, "install command");
  assert.equal(copiedText, "install command");
  assert.equal(selected, true);
  assert.equal(removed, true);
  assert.equal(copyHandler, undefined);
});

test("reports a copy failure when neither method works", async () => {
  const copied = await copyText("install command", {
    clipboard: undefined,
    document: undefined,
  });

  assert.equal(copied, false);
});
