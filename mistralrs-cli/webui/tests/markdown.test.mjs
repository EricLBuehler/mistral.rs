import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";

let server;
let renderMarkdown;

before(async () => {
  server = await createServer({
    server: { middlewareMode: true, ws: false, watch: null },
  });
  ({ renderMarkdown } = await server.ssrLoadModule("/src/lib/utils/markdown.ts"));
});

after(async () => {
  await server?.close();
});

test("renders inline and display math alongside markdown", () => {
  const html = renderMarkdown("**Result:** $x^2$\n\n$$\n\\frac{1}{2}\n$$");
  assert.match(html, /<strong>Result:<\/strong>/);
  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex-display"/);
  assert.match(html, /<mfrac>/);
});

test("keeps math delimiters literal in code", () => {
  const html = renderMarkdown("`$x$`\n\n```text\n$$x$$\n```");
  assert.match(html, /<code>\$x\$<\/code>/);
  assert.match(html, /\$\$x\$\$/);
  assert.doesNotMatch(html, /class="katex/);
});

test("tolerates incomplete model output and escapes invalid math", () => {
  assert.match(renderMarkdown("Result: $x +"), /Result: \$x \+/);
  const html = renderMarkdown("$\\frac{<img src=x onerror=alert(1)>}$");
  assert.match(html, /katex-error/);
  assert.doesNotMatch(html, /<img\b/);
});

test("blocks untrusted math URLs, images, and HTML", () => {
  const html = renderMarkdown(String.raw`$\href{javascript:alert(1)}{click}$

$\includegraphics{https://example.com/tracker.png}$

$\htmlStyle{background:url(https://example.com/tracker.png)}{x}$`);
  assert.doesNotMatch(html, /<a\b|<img\b|style="[^"]*url\(/);
});

test("continues rejecting raw HTML and executable markdown links", () => {
  const html = renderMarkdown('<script>alert(1)</script>\n\n[click](javascript:alert%281%29)');
  assert.doesNotMatch(html, /<script\b|href="javascript:/);
  assert.match(html, /href="#"/);
});
