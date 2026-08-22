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

## Validation

```bash
npm run check
npm test
```
