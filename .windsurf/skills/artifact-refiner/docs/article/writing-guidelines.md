# Writing Guidelines by Platform

This document provides guidance for adapting the article to different publishing platforms. The core content and transparency principles remain the same; what changes is tone, structure, length, and audience assumptions.

---

## General Principles (All Platforms)

1. **Transparency first** — Be explicit that an AI coding assistant (Antigravity/Claude) performed the bulk of the implementation work. Name the tool. Describe what you directed vs. what it generated.
2. **Show, don't tell** — Use code snippets, directory trees, audit tables, and before/after comparisons. Readers should see evidence, not just claims.
3. **Link the repo** — Every version should link to [github.com/GQAdonis/artifact-refiner-skill](https://github.com/GQAdonis/artifact-refiner-skill) so readers can verify every claim.
4. **Dual audience** — Write for both AI practitioners and technically curious non-practitioners. Explain jargon on first use.
5. **MIT license callout** — Mention that the project is open source (MIT) and invite contributions.

---

## Platform: travisjames.ai (Personal Blog)

### Audience
Developers, AI engineers, and technical leaders who follow your work. They expect depth and are there specifically for your perspective.

### Tone
First-person, candid, technical but accessible. You're talking to peers. Use "I" freely. Share what surprised you, what didn't work on the first try, and what you learned.

### Length
**3,000–5,000 words**. This is your home — take the space you need.

### Structure
- Use the full 7-part outline
- Include all code snippets, directory trees, and audit tables
- Embed screenshots or terminal recordings if available
- Include the complete before/after audit score table
- End with a detailed "What's Next" section showing your roadmap

### Unique Elements
- **Behind-the-scenes narration** — Include your internal reasoning: "At this point I had to decide whether to define subagents as YAML or Markdown. I chose Markdown because..."
- **Conversation excerpts** — Quote specific prompts you gave the AI and summarize what it returned. This is the transparency differentiator.
- **Full audit table** — Embed the 10-dimension scoring table with reasoning columns.
- **Cross-links** — Link to related posts on PMPO theory, your other skills work, or your development philosophy.

### SEO
- Title tag: "Building a 10/10 AI Skill: PMPO Artifact Refiner | Travis James"
- Meta description: "How I used AI to architect, audit, and perfect an open-source Claude Code plugin for iterative artifact refinement."
- Target keywords: AI skills, Claude plugin, PMPO, artifact refinement, AI coding assistant, open source

### Call to Action
"Star the repo, try `/refine-logo` in Claude Code, and tell me what domain adapter you'd want next."

---

## Platform: Medium.com

### Audience
Broad technical audience. Mix of AI enthusiasts, senior engineers, product managers, and curious generalists. Many are skimmers — they'll read headings, bold text, and code blocks.

### Tone
Second-person where useful ("Imagine you have a skill that scores 7.2/10..."), polished but not academic. Avoid bullet-heavy walls of text. Medium rewards storytelling.

### Length
**1,800–2,500 words**. Medium's sweet spot. Cut ruthlessly.

### Structure
- **Consolidate Parts 1–2** into a short "The Problem" + "The Starting Score" section
- **Compress Part 3** (PMPO) into 2–3 paragraphs with a diagram or bullet summary
- **Part 4** (the overhaul) is the core — but pick 3–4 workstreams to highlight, not all 7. Recommended: WS3 (architecture), WS4 (subagents), WS5 (hooks), WS6 (slash commands)
- **Skip Part 5** (verification) — mention it in one sentence
- **Part 6** (meta-observation) becomes the emotional payoff — expand it slightly
- **Part 7** becomes a single closing paragraph

### What to Cut
- Detailed directory trees (link to repo instead)
- Full audit tables (show just the summary: "7.2 → 10.0")
- In-depth constraint schema examples
- The content refinement example (keep logo example only)

### Unique Elements
- **Hook opening** — Start with the punchline: "I scored my own AI skill a 7.2 out of 10. Then I used an AI to fix everything it was missing."
- **One visual** — Include the before/after score summary as an image or formatted table
- **Subheadings every 200–300 words** — Medium readers scan

### Medium-Specific
- Publish in a relevant publication (Towards AI, Better Programming, The Startup)
- Use Medium's code block formatting (triple backtick)
- Add 5 tags: AI, Software Engineering, Open Source, Claude, Developer Tools
- Friend link for non-subscribers

### Call to Action
"The full skill is open source at [GitHub link]. Fork it, try it, or propose a new domain adapter."

---

## Platform: LinkedIn

### Audience
Professional network. Mix of engineering leaders, product managers, recruiters, and AI-curious professionals. They're scrolling a feed — you have 2–3 seconds to earn a stop.

### Tone
Professional but human. First-person. Conversational. Avoid jargon unless immediately explained. LinkedIn rewards vulnerability and lessons learned over pure technical depth.

### Length
**800–1,200 words** for a long-form post. Alternatively, a **300-word post** linking to the full article on travisjames.ai.

### Structure (Long-Form Post)

```
[Hook — 1 sentence, bold or provocative]

[Problem — 2–3 sentences about why AI skills are fragile]

[What I did — 3–4 paragraphs covering the audit, the overhaul, and the AI collaboration]

[The meta-lesson — 2 paragraphs about transparency and human-AI collaboration]

[CTA — 1 paragraph with repo link]
```

### Structure (Short Post → Link)

```
I just open-sourced an AI skill that scores 10/10 across every dimension of quality.

It took an AI to get it there.

[2–3 sentences about what the skill does]
[1 sentence about the audit process]
[1 sentence about the AI collaboration]

Full writeup on my blog → [link]
GitHub → [link]

#AI #OpenSource #SoftwareEngineering #CodingAssistant
```

### What to Include
- The 7.2 → 10.0 score jump (concrete numbers perform well on LinkedIn)
- The "I used AI to improve an AI skill" angle (meta-narrative hooks)
- The transparency angle: "Here's exactly what the AI did vs. what I directed"
- The open-source callout

### What to Cut
- All code snippets (link instead)
- Technical details about PMPO phases
- Directory structure details
- Audit dimension breakdowns

### LinkedIn-Specific
- Use line breaks generously (LinkedIn's mobile formatting is brutal with long paragraphs)
- Include 3–5 hashtags: #AI #OpenSource #SoftwareEngineering #ClaudeCode #DeveloperTools
- Tag relevant people or companies if appropriate
- Post between 8–10 AM weekdays for best engagement

### Call to Action
"The repo is MIT licensed and ready for contributions. What domain adapter would you want first? Drop it in the comments."

---

## Platform Comparison Summary

| Aspect | travisjames.ai | Medium | LinkedIn |
|---|---|---|---|
| Length | 3,000–5,000 words | 1,800–2,500 words | 800–1,200 words |
| Tone | Candid, technical, first-person | Polished, storytelling | Professional, conversational |
| Code snippets | Extensive | Selective (3–4 key ones) | None (link instead) |
| Audit detail | Full 10-dimension table | Summary score only | Score jump number only |
| PMPO depth | Complete methodology | 2–3 paragraph summary | One sentence |
| Workstream coverage | All 7 | 3–4 highlights | High-level overview |
| Transparency emphasis | Conversation excerpts | Mentioned, not quoted | Central hook |
| Visual assets | Screenshots, trees, tables | 1 key visual | None or 1 image |
| CTA | Star + try + request features | Fork + try | Comment + contribute |
