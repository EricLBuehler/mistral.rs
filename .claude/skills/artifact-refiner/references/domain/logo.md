# Logo Refiner Module

## Brand Constraint Loading

Load brand constraints from the user's provided brand guide files, or infer reasonable defaults if no brand guide exists.

If brand guide files are available, extract:

- Primary brand color (hex)
- Dark mode primary variant
- Neutral background palette
- Typography hierarchy (heading + body fonts)
- Logo usage guidelines (spacing, minimum sizes)

If no brand guide exists, generate a cohesive brand system based on the user's stated preferences and target audience.

## Template

Use `assets/templates/logo-showcase.template.html` for the final showcase output.
Replace `{{VARIABLE_NAME}}` placeholders with generated values.

## Generation Requirements

1. **Generate SVG first** — Vector source of truth
2. **Rasterize to PNG** in standard sizes: 16, 32, 48, 64, 128, 192, 256, 512
3. **Generate variants**:
   - Wordmark (full logo with text)
   - Icon (symbol only)
   - App icon (square, rounded corners ready)
   - Favicon (16x16, 32x32)
4. **Generate showcase HTML** referencing all generated assets
5. **Update `artifact_manifest.json`** with all generated files and their paths

## Deterministic Execution Triggers

Use code interpreter or e2b sandbox for:
- SVG → PNG rasterization
- Image resizing / format conversion
- HTML template population
- File size / dimension validation

## Common Constraints

| Constraint | Severity | Validation |
|---|---|---|
| SVG must be valid XML | blocking | Parse with XML parser |
| All PNG sizes must match spec | blocking | Check pixel dimensions |
| Colors must match brand palette | high | Hex value comparison |
| Showcase HTML must render | medium | File existence + valid HTML |
| Manifest must reference all files | blocking | Cross-reference file list |