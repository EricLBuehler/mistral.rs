import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import markedKatex from "marked-katex-extension";
import hljs from "highlight.js";

const marked = new Marked(
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

// KaTeX for LaTeX math rendering
marked.use(
  markedKatex({
    throwOnError: false,
    displayMode: false,
  }),
);

marked.setOptions({
  breaks: true,
  gfm: true,
});

// Custom renderer for links (open in new tab) and code blocks (copy button wrapper)
const renderer = new marked.Renderer();

renderer.link = function ({ href, text }: { href: string; text: string }) {
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
};

// `text` is ALREADY highlighted by markedHighlight — do NOT re-highlight
renderer.code = function ({
  text,
  lang,
}: {
  text: string;
  lang?: string;
}) {
  const language = lang || "";
  const langLabel = language
    ? `<span class="code-lang">${language}</span>`
    : "";

  return `<div class="code-block-wrapper">
    <div class="code-block-header">
      ${langLabel}
      <button class="copy-btn" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)})">Copy</button>
    </div>
    <pre><code class="hljs${language ? ` language-${language}` : ""}">${text}</code></pre>
  </div>`;
};

marked.use({ renderer });

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}
