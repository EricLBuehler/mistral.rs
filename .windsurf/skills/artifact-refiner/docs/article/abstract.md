# Building a 10/10 AI Skill: How I Used AI to Architect an AI Refinement Engine

## Abstract

What does it look like when you use an AI coding assistant to build, audit, and perfect a skill that teaches AI how to iteratively refine artifacts? This article documents the full journey — from a rough 7.2/10-scored skill to a perfect 10/10 across ten dimensions of quality.

The **Artifact Refiner** is an open-source Claude Code plugin built on PMPO (Prometheus Meta-Prompting Orchestration), a structured methodology for iterative, constraint-driven artifact creation. Unlike traditional prompting, PMPO separates cognition from computation, persists state to disk between sessions, and uses explicit convergence criteria to decide when work is done.

This piece traces the complete development arc:

1. **The initial audit** — scoring the skill across 10 dimensions and identifying gaps in spec compliance, architecture, documentation, tooling, and extensibility
2. **The 7-workstream overhaul** — restructuring directories, defining 5 specialized subagents, implementing lifecycle hooks, integrating sandboxed code execution (e2b MCP), building 7 slash commands, and eliminating documentation duplication
3. **The verification loop** — automated tests for YAML frontmatter, JSON schema validity, file reference integrity, and naming convention compliance
4. **The meta-observation** — using an AI assistant (Antigravity) to do the work, while documenting the process transparently for the community

The article is written for developers, AI engineers, and technical leaders who want to understand what production-quality AI skills look like, how to audit them systematically, and how human-AI collaboration can produce artifacts that neither could efficiently produce alone.

**Repository**: [github.com/GQAdonis/artifact-refiner-skill](https://github.com/GQAdonis/artifact-refiner-skill)  
**Author**: Travis James — [travisjames.ai](https://travisjames.ai)  
**License**: MIT
