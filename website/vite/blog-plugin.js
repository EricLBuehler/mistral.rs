import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const SITE_ORIGIN = "https://mistralrs.dev";
const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]),
  );
}

function stripQuotes(value) {
  if (
    value.length >= 2 &&
    ((value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'")))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function parseFrontmatter(raw, label) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!match) {
    throw new Error(`blog post ${label}: missing leading frontmatter block`);
  }
  const data = {};
  for (const line of match[1].split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separator = trimmed.indexOf(":");
    if (separator === -1) {
      throw new Error(`blog post ${label}: bad frontmatter line "${trimmed}"`);
    }
    const key = trimmed.slice(0, separator).trim();
    const value = trimmed.slice(separator + 1).trim();
    if (value.startsWith("[") && value.endsWith("]")) {
      data[key] = value
        .slice(1, -1)
        .split(",")
        .map((part) => stripQuotes(part.trim()))
        .filter(Boolean);
    } else {
      data[key] = stripQuotes(value);
    }
  }
  return { data, body: raw.slice(match[0].length) };
}

function truncateToWords(value, max) {
  if (value.length <= max) return value;
  const cut = value.slice(0, max + 1);
  const space = cut.lastIndexOf(" ");
  return `${cut.slice(0, space > 0 ? space : max).trimEnd()}...`;
}

function extractDescription(markdown) {
  const paragraphs = [];
  let current = [];
  const flush = () => {
    if (current.length) paragraphs.push(current.join(" "));
    current = [];
  };
  for (const line of markdown.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    if (/^(#|\||```|>|-{3,}|\*{3,}|[-*+] |\d+\.\s)/.test(trimmed)) {
      flush();
      continue;
    }
    current.push(trimmed);
  }
  flush();
  const plain = (paragraphs.find((p) => p.length > 60) ?? paragraphs[0] ?? "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return truncateToWords(plain, 200);
}

function parseDate(value, label) {
  const match = value?.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) {
    throw new Error(`blog post ${label}: date must be YYYY-MM-DD, got "${value}"`);
  }
  const [, year, month, day] = match;
  if (month < "01" || month > "12" || day < "01" || day > "31") {
    throw new Error(`blog post ${label}: invalid date "${value}"`);
  }
  return { raw: value, display: `${MONTHS[Number(month) - 1]} ${Number(day)}, ${year}` };
}

async function collectImages(html, postDir, releasesDir, slug) {
  const assets = [];
  const used = new Set();
  const pattern = /<img\b[^>]*?\bsrc="([^"]+)"[^>]*>/g;
  let match;
  while ((match = pattern.exec(html))) {
    const src = match[1];
    if (src.startsWith("/") || src.startsWith("data:") || /^(https?:)?\/\//.test(src)) {
      continue;
    }
    const resolved = path.resolve(postDir, src);
    if (!resolved.startsWith(path.resolve(releasesDir) + path.sep)) {
      throw new Error(`blog post ${slug}: image "${src}" must live under releases/`);
    }
    const base = path.basename(resolved);
    const fileName = `blog/assets/${slug}/${base}`;
    if (used.has(fileName)) {
      throw new Error(`blog post ${slug}: duplicate image name "${base}"`);
    }
    used.add(fileName);
    assets.push({ fileName, source: await readFile(resolved) });
    html = html.replaceAll(`src="${src}"`, `src="assets/${slug}/${base}"`);
  }
  return { html, assets };
}

function stripLeadingH1(markdown) {
  const lines = markdown.split(/\r?\n/);
  let index = 0;
  while (index < lines.length && !lines[index].trim()) index += 1;
  if (index < lines.length && /^#[ \t]+/.test(lines[index])) {
    lines.splice(index, 1);
  }
  return lines.join("\n");
}

async function loadPost(postPath, releasesDir) {
  const version = path.basename(path.dirname(postPath));
  const raw = await readFile(postPath, "utf8");
  const { data, body: frontmatterBody } = parseFrontmatter(raw, version);
  if (!data.title) throw new Error(`blog post ${version}: frontmatter needs a title`);
  if (!data.slug) throw new Error(`blog post ${version}: frontmatter needs a slug`);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(data.slug)) {
    throw new Error(`blog post ${version}: slug "${data.slug}" must be kebab-case`);
  }
  const date = parseDate(data.date, version);
  const tags = data.tags ?? [];
  const body = stripLeadingH1(frontmatterBody);
  const rendered = marked.parse(body, { gfm: true });
  const { html, assets } = await collectImages(
    rendered,
    path.dirname(postPath),
    releasesDir,
    data.slug,
  );
  return {
    version,
    slug: data.slug,
    title: data.title,
    author: data.author ?? "",
    date,
    tags,
    description: extractDescription(body),
    bodyHtml: html.trim(),
    assets,
  };
}

function metaTags({ title, description, url, ogType }) {
  const image = `${SITE_ORIGIN}/og.png`;
  return [
    `<meta name="description" content="${escapeHtml(description)}" />`,
    `<meta property="og:type" content="${ogType}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:site_name" content="mistral.rs" />`,
    `<meta property="og:title" content="${escapeHtml(title)}" />`,
    `<meta property="og:description" content="${escapeHtml(description)}" />`,
    `<meta property="og:image" content="${image}" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${escapeHtml(title)}" />`,
    `<meta name="twitter:description" content="${escapeHtml(description)}" />`,
    `<meta name="twitter:image" content="${image}" />`,
  ].join("\n    ");
}

function page({ title, head, navLabel, navHref, main }) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#070605" />
    ${head}
    <title>${escapeHtml(title)}</title>
    <link rel="stylesheet" href="/blog.css" />
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to content</a>
    <header class="blog-header">
      <a class="blog-home" href="/">mistral<span>.rs</span></a>
      <nav class="blog-nav" aria-label="Blog">
        <a href="${navHref}">${navLabel}</a>
      </nav>
    </header>
    ${main}
  </body>
</html>
`;
}

function postPage(post) {
  const url = `${SITE_ORIGIN}/blog/${post.slug}.html`;
  const tags = post.tags.length
    ? `<span class="post-tags">${escapeHtml(post.tags.join(" / "))}</span>`
    : "";
  const author = post.author ? `<span class="post-author">${escapeHtml(post.author)}</span>` : "";
  const main = `<main id="main" class="blog-main">
      <article class="post">
        <header class="post-head">
          <h1 class="post-title">${escapeHtml(post.title)}</h1>
          <p class="post-meta">
            ${author}
            <time datetime="${post.date.raw}">${post.date.display}</time>
            ${tags}
          </p>
        </header>
        <div class="post-body">${post.bodyHtml}</div>
        <footer class="post-foot">
          <a href="/blog/">All posts</a>
        </footer>
      </article>
    </main>`;
  return page({
    title: `${post.title} | mistral.rs blog`,
    head: metaTags({ title: post.title, description: post.description, url, ogType: "article" }),
    navLabel: "All posts",
    navHref: "/blog/",
    main,
  });
}

function postListItem(post) {
  const parts = [];
  if (post.author) parts.push(`By ${escapeHtml(post.author)}`);
  parts.push(post.date.display);
  if (post.tags.length) parts.push(escapeHtml(post.tags.join(", ")));
  const meta = parts.join(" | ");
  return `<li class="post-list-item">
          <a class="post-list-link" href="${post.slug}.html">
            <h2 class="post-list-title">${escapeHtml(post.title)}</h2>
            <p class="post-meta"><span>${meta}</span></p>
            <p class="post-list-excerpt">${escapeHtml(post.description)}</p>
          </a>
        </li>`;
}

function indexPage(posts) {
  const url = `${SITE_ORIGIN}/blog/`;
  const items = posts.map(postListItem).join("\n        ");
  const list = items
    ? `  <ul class="post-list">\n        ${items}\n      </ul>`
    : `  <p class="post-list-empty">No posts yet.</p>`;
  const main = `<main id="main" class="blog-main">
      <h1 class="blog-title">Blog</h1>
      ${list}
    </main>`;
  return page({
    title: "Blog | mistral.rs",
    head: metaTags({
      title: "mistral.rs blog",
      description: "Benchmarks and release notes for mistral.rs.",
      url,
      ogType: "website",
    }),
    navLabel: "Home",
    navHref: "/",
    main,
  });
}

export function mistralrsBlog() {
  return {
    name: "mistralrs-blog",
    apply: "build",
    async buildStart() {
      const releasesDir = path.resolve(
        path.dirname(fileURLToPath(import.meta.url)),
        "..",
        "..",
        "releases",
      );
      let entries = [];
      try {
        entries = await readdir(releasesDir, { withFileTypes: true });
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
      const posts = [];
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const postPath = path.join(releasesDir, entry.name, "post.md");
        try {
          await stat(postPath);
        } catch {
          continue;
        }
        posts.push(await loadPost(postPath, releasesDir));
      }
      posts.sort(
        (a, b) => b.date.raw.localeCompare(a.date.raw) || b.version.localeCompare(a.version),
      );
      const slugs = new Set();
      for (const post of posts) {
        if (slugs.has(post.slug)) {
          throw new Error(`blog post ${post.version}: duplicate slug "${post.slug}"`);
        }
        slugs.add(post.slug);
      }

      const cssSource = await readFile(new URL("../src/blog.css", import.meta.url));
      this.emitFile({ type: "asset", fileName: "blog.css", source: cssSource });
      this.emitFile({ type: "asset", fileName: "blog/index.html", source: indexPage(posts) });
      for (const post of posts) {
        for (const asset of post.assets) {
          this.emitFile({ type: "asset", fileName: asset.fileName, source: asset.source });
        }
        this.emitFile({ type: "asset", fileName: `blog/${post.slug}.html`, source: postPage(post) });
      }
    },
  };
}
