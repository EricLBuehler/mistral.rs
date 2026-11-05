import { Marked, type Token, type Tokens, type TokensList } from "marked";
import { markedHighlight } from "marked-highlight";
import markedKatex from "marked-katex-extension";
import hljs from "highlight.js";

// Reject schemes that can execute script. Allow http(s), mailto, and relative paths.
function safeHref(href: string): string {
  if (/^[#/]/.test(href)) return href;
  try {
    const url = new URL(href);
    if (["http:", "https:", "mailto:"].includes(url.protocol)) return href;
  } catch {
    // not a URL we can parse; fall through to "#"
  }
  return "#";
}

function escapeAttr(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

const COPY_LABEL = "Copy";
const COPIED_LABEL = "Copied!";
const COPIED_RESET_MS = 1500;
const BLOCK_CACHE_MAX = 2000;

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function codeBlockHtml(highlighted: string, lang?: string): string {
  const language = lang ? escapeAttr(lang) : "";
  const langLabel = language ? `<span class="code-lang">${language}</span>` : "";
  return `<div class="code-block-wrapper">
      <div class="code-block-header">
        ${langLabel}
        <button type="button" class="copy-btn">${COPY_LABEL}</button>
      </div>
      <pre><code class="hljs${language ? ` language-${language}` : ""}">${highlighted}</code></pre>
    </div>`;
}

// Custom renderer for links and code blocks
const renderer = {
  link({ href, text }: { href: string; text: string }) {
    const safe = escapeAttr(safeHref(href));
    return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  },
  // Raw HTML in model output is dropped to avoid XSS (no DOMPurify dependency).
  html(): string {
    return "";
  },
  // `text` is ALREADY highlighted by markedHighlight; do NOT re-highlight
  code({ text, lang }: { text: string; lang?: string }) {
    return codeBlockHtml(text, lang);
  },
};

const marked = new Marked();

// Order matters: highlight first, then katex, then renderer
marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code: string, lang: string) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
  }),
);

marked.use(
  markedKatex({
    throwOnError: false,
    nonStandard: true,
  }),
);

marked.use({ renderer });

marked.setOptions({
  breaks: true,
  gfm: true,
});

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}

const blockCache = new Map<string, string>();

type Links = TokensList["links"];

// lexer + parser skip the extensions' walkTokens (that's where markedHighlight highlights), so run it here
function parseToken(token: Token, links: Links): string {
  const walk = marked.defaults.walkTokens;
  if (walk) marked.walkTokens([token], walk);
  return marked.parser(Object.assign([token], { links })) as string;
}

function renderToken(token: Token, links: Links): string {
  const key = Object.keys(links).length ? `${token.raw}\0${JSON.stringify(links)}` : token.raw;
  const cached = blockCache.get(key);
  if (cached != null) return cached;
  const html = parseToken(token, links);
  if (blockCache.size >= BLOCK_CACHE_MAX) blockCache.delete(blockCache.keys().next().value!);
  blockCache.set(key, html);
  return html;
}

// Finished top-level blocks are cached by source, so a streaming reply only re-renders its last block.
// An unclosed code fence skips highlighting until it closes.
export function renderMarkdownBlocks(text: string, streaming: boolean): string[] {
  const tokens = marked.lexer(text);
  const out: string[] = [];
  tokens.forEach((token, i) => {
    const isTail = streaming && i === tokens.length - 1;
    if (isTail && token.type === "code") {
      const code = token as Tokens.Code;
      out.push(codeBlockHtml(escapeHtml(code.text), code.lang || undefined));
    } else if (isTail) {
      out.push(parseToken(token, tokens.links));
    } else {
      out.push(renderToken(token, tokens.links));
    }
  });
  return out;
}

// One delegated listener instead of inline onclick strings in rendered HTML.
export function installCopyHandler(root: Document = document) {
  root.addEventListener("click", (e) => {
    const btn = (e.target as Element | null)?.closest?.(".copy-btn");
    const code = btn?.closest(".code-block-wrapper")?.querySelector("code");
    if (!btn || !code) return;
    navigator.clipboard.writeText(code.textContent ?? "").then(() => {
      btn.textContent = COPIED_LABEL;
      setTimeout(() => (btn.textContent = COPY_LABEL), COPIED_RESET_MS);
    });
  });
}
