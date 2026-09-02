# mistral.rs landing page

The static one-page site for [mistralrs.dev](https://mistralrs.dev), built with
Vite for Cloudflare Pages.

## Local development

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

The development server prints its local URL when it starts.

## Cloudflare Pages

- Root directory: `website`
- Build command: `npm run build`
- Build output directory: `dist`

The Vite build copies the repository's root `install.sh` and `install.ps1` into
the output directory. Those root files are the only installer sources.

## Blog

`/blog/` is generated at build time from `releases/<version>/post.md` files in
the repository root. Each post needs a frontmatter block with `title`, `date` (YYYY-MM-DD), and
`slug` (kebab-case, unique across posts), plus optional `author` and `tags`.
Relative image references resolve against the post's directory and
are copied to `/blog/assets/<slug>/`, so figures should live next to the
post (e.g. `releases/<version>/figures/`). The post list is sorted by date,
newest first, and each page gets Open Graph/Twitter meta with a description
auto-extracted from the first substantial paragraph.

Publishing a post is committing `post.md` (plus any figures); the Cloudflare
Pages build emits `blog/index.html`, `blog/<slug>.html`, `blog.css`, and the
asset images. There is no runtime JavaScript on blog pages.

## Validation

```bash
npm run check
npm test
```
