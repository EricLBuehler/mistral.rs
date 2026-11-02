import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";

let server;
let renderMarkdown;
let renderMarkdownBlocks;

before(async () => {
  server = await createServer({
    server: { middlewareMode: true, ws: false, watch: null },
  });
  ({ renderMarkdown, renderMarkdownBlocks } = await server.ssrLoadModule("/src/lib/utils/markdown.ts"));
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

const DOC = `# Title

Some **bold** text with $x^2$ and a [ref][r].

- one
- two

\`\`\`python
print("hi")
\`\`\`

[r]: https://example.com
`;

test("block rendering matches whole-document rendering", () => {
  assert.equal(renderMarkdownBlocks(DOC, false).join(""), renderMarkdown(DOC));
});

test("streaming every prefix ends in the same output as a full render", () => {
  let last;
  for (let i = 1; i <= DOC.length; i++) last = renderMarkdownBlocks(DOC.slice(0, i), true);
  assert.equal(renderMarkdownBlocks(DOC, false).join(""), renderMarkdown(DOC));
  assert.ok(last.join("").includes("print"));
});

test("an unclosed code fence is shown as escaped plain text while streaming", () => {
  const blocks = renderMarkdownBlocks("intro\n\n```js\nif (a < b) {", true);
  const tail = blocks.at(-1);
  assert.match(tail, /if \(a &lt; b\) \{/);
  assert.doesNotMatch(tail, /hljs-keyword/);
});

test("code block language is escaped and copy buttons carry no inline handlers", () => {
  const html = renderMarkdown('```x"><img src=x onerror=alert(1)>\ncode\n```');
  assert.doesNotMatch(html, /<img\b|onclick=/);
});
