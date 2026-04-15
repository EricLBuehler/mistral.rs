import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import markedKatex from "marked-katex-extension";
import hljs from "highlight.js";

// Custom renderer for links and code blocks
const renderer = {
  link({ href, text }: { href: string; text: string }) {
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  },
  // `text` is ALREADY highlighted by markedHighlight — do NOT re-highlight
  code({ text, lang }: { text: string; lang?: string }) {
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
