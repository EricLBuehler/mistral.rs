# Content Refinement Example

This example demonstrates PMPO refinement of a blog post, from rough draft to polished content.

## Scenario

**User request**: "Refine my blog post about PMPO methodology. Fix heading structure, improve readability, add SEO metadata."

## Input

See [input.md](./input.md) — a rough blog post draft with inconsistent headings, long paragraphs, and no metadata.

## PMPO Flow

### Iteration 1

1. **Specify**: Defined constraints — single H1, heading hierarchy, paragraph length ≤ 4 sentences, meta description required
2. **Plan**: Staged approach — normalize headings → split paragraphs → add metadata → generate report
3. **Execute**: Applied all transformations
4. **Reflect**: Headings fixed ✅, paragraphs split ✅, meta description missing ❌
5. **Persist**: State saved, decision = continue

### Iteration 2

1. **Plan**: Add meta description and Open Graph tags
2. **Execute**: Generated SEO metadata
3. **Reflect**: All constraints satisfied ✅
4. **Persist**: Decision = terminate

## Output

See [output.html](./output.html) — the refined content report generated from the content report template.
