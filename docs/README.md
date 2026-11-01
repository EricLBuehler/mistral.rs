# Documentation

The Astro/Starlight site lives in `src/content/docs`. Run `npm install` once, then `npm run dev` to preview or `npm run build` to validate the site.

## Where content belongs

- **Guides:** a task, a runnable starting point, and the limits needed to complete it. Link to full examples instead of copying every variant.
- **Reference:** exact flags, fields, formats, compatibility, and error behavior.
- **Developer:** architecture, tensor layouts, kernel selection, scheduling, and contributor setup.
- **Examples:** runnable source, rendered from the repository examples.

Give each explanation one main home and link to it. Keep HTTP, Python, and Rust examples aligned. Preserve important restrictions when shortening a page. Add model discovery links when introducing a new task, rather than expanding the family-notes page into another tutorial.

## Generated content

Update the source rather than editing generated pages:

| Pages | Source |
| --- | --- |
| Supported models | `mistralrs-core/src/model_metadata.rs` |
| Python reference | `mistralrs-pyo3/mistralrs.pyi` and `scripts/render_pyi.py` |
| CLI reference | CLI definitions and `regenerate_cli_reference` |
| HTTP schema | Server API definitions and `openapi.json` |
| Examples | Repository example files and `scripts/render_examples.py` |

`npm run generate` refreshes all generated content. `npm run build` refreshes only the Python reference before building; regenerate other affected pages explicitly. Check internal links and section anchors after moving content. Sidebar entries and old-page redirects live in `astro.config.mjs`.
