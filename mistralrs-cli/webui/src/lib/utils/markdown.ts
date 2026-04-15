import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
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

marked.setOptions({
  breaks: true,
  gfm: true,
});

// Custom renderer for links (open in new tab) and code blocks (copy button)
const renderer = new marked.Renderer();

renderer.link = function ({ href, text }: { href: string; text: string }) {
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
};

renderer.code = function ({
  text,
  lang,
}: {
  text: string;
  lang?: string;
}) {
  const language = lang || "";
  const highlighted =
    language && hljs.getLanguage(language)
      ? hljs.highlight(text, { language }).value
      : hljs.highlightAuto(text).value;

  const langLabel = language
    ? `<span class="code-lang">${language}</span>`
    : "";

  return `<div class="code-block-wrapper">
    <div class="code-block-header">
      ${langLabel}
      <button class="copy-btn" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)})">Copy</button>
    </div>
    <pre><code class="hljs${language ? ` language-${language}` : ""}">${highlighted}</code></pre>
  </div>`;
};

marked.use({ renderer });

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}
