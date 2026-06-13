# Docs restructure plan

Sign-off artifact for the onboarding project. Execution is one-fell-swoop per section below; this
document is the review surface. Sources: 15-agent prose audit + 9-agent examples/spikes round
(full reports preserved; ask Claude for the raw synthesis files).

## Goal and principles

- Problem-first IA (vLLM-style): **User Guide / Developer Guide / Reference**. Genres (tutorial,
  concept, lookup) become page *shape*, not navigation.
- **One fact, one home.** Every cross-page contradiction the audit found occurred where a fact had
  multiple homes. Pages link; they do not restate.
- Answer pages: explanation + runnable example inline, depth below, collapsibles for advanced.
- Generated where a source of truth exists: CLI from clap, HTTP from OpenAPI, Python from the pyi.
- ~100 prose pages -> ~52. 185 examples -> ~109.

## Locked decisions

1. MCP-server mode gets **ported to the unified CLI** (it exists only in the deprecated binary
   today). The 4 pages documenting it stay, rewritten against the ported implementation.
2. **Dockerfiles migrate to the unified CLI** and CI publishes correct tags. docker.md is rewritten
   against the new images; README's broken `:latest` instructions are replaced.
3. One-swoop cutover; Starlight redirects map keeps every old URL working.
4. Verification gates replace eyeball review (list at the end).

## New information architecture

```
User Guide
  Quickstart                  (tabbed Metal/CUDA/Windows install + doctor + first run + first serve)
  Serving
    OpenAI-compatible API     (canonical serve page)
    Anthropic Messages API
    Responses API notes       (inside openai-compatibility reference)
    Web UI
    Multiple models & config file
    Coding agents (Claude Code, Codex)
    Structured output         (NEW page: response_format, json_schema, grammars, all 3 surfaces)
  Models
    Run any model             (NEW page: HF ids, local files, GGUF, --quant)
    Model family notes        (merged walkthroughs; quirks, /think tags, MatFormer)
    Multimodal input          (merged vision/video/audio input + Python variants)
    Video setup (FFmpeg)
    Speech & ASR
    Image generation
    Embeddings
    Block diffusion (DiffusionGemma)
  Quantization
    Quantize a model          (canonical: narrative + decision guide + concepts + auto-tune)
    UQFF
    Online calibration
  Agents & tools
    Tool calling              (absorbs strict + loop config)
    Code execution            (absorbs design internals)
    Web search
    Permissions & approvals   (NEW page: split from agentic-runtime)
    Agentic runtime for apps
    Sessions                  (absorbs python session guide + splicing facts)
    MCP: connect a server
    MCP: serve over MCP       (rewritten after the port)
  Python SDK                  (getting started + streaming/async + generated reference children)
  Rust SDK                    (getting started + streaming + axum + NEW in-tree reference page)
  Performance & scaling
    Paged attention           (guide + how-it-works; absorbs flash-attention para + CUDA graphs)
    Speculative decoding (MTP) (absorbs gemma4-mtp)
    Distributed inference     (merged: modes intro, TP, NCCL multi-node, ring, device-mapping notes)
    Topology
    Throughput tuning         (NEW page: --max-seqs, prefix caching, scheduling)
  Deploy
    Docker                    (rewritten against migrated images)
    Production checklist      (+ /metrics, readiness)
Developer Guide
  Architecture
  MoE backends
  Multimodal pipeline
  Session splicing internals
  Build from source
  Contributing
Reference
  CLI                         (GENERATED from clap; golden test)
  HTTP API                    (GENERATED OpenAPI core + hand-written SSE/prose page)
  OpenAI compatibility
  TOML config (+ flag mapping column, [sandbox], thinking)
  Environment variables
  MCP config schema
  Sandbox
  Quantization types          (AFQ claim fixed)
  Supported models            (+ model-notes absorbed)
  UQFF format
  Cargo features
  Troubleshooting
  Python API                  (generated, regen in release checklist)
```

## Engineering workstreams (pre-cutover)

| # | Work | Notes |
|---|---|---|
| E1 | Port MCP-server mode to unified CLI | `--mcp-port` / `POST /mcp` from deprecated binary; docs for it blocked until this lands |
| E2 | Dockerfile migration to unified CLI + CI tag fix | both docker.md commands fail today; `:latest` never published |
| E3 | OpenAPI registration debt | ~7 one-line path registrations, annotations for files/approvals/metrics/tune/system routes, CalibrationStatus ToSchema (3 edits), doc comments on request structs (raises 15% description coverage) |
| E4 | CLI docgen module + golden test | spike code ready (preserved as patch); golden test = CI fails on flag/docs drift |
| E5 | OpenAPI dump test + starlight-openapi wiring | spike code ready; plugin peer-deps verified against docs site |
| E6 | render_pyi.py into release checklist/CI | the calibration drift was a process gap (PR #2210 fixed the instance) |
| E7 | Examples auto-render hook | walk examples trees, header comment -> blurb, emit docs pages per topic |

## Page disposition

Verdicts: K keep, M merge into target, D delete, R rewrite. All audit conflicts resolved
(conservative: merge over delete; merge direction decided per-fact against code, not per-genre).

| Page | V | Target / notes |
|---|---|---|
| index.mdx | R | Landing; re-point cards; fix tune claim |
| start-here.md | D | runtime-mode info -> quickstart |
| README.md | R | slim front door; corrected ghcr -> docker.md; SDK snippets -> links |
| tutorials/index.md | D | |
| tutorials/01-install-and-run.md | K | becomes Quickstart; canonical install one-liner home |
| tutorials/02-serve-an-api.md | M | serving quickstart core; HF-auth walkthrough gets own section |
| tutorials/03-python-sdk.md | M | Python getting-started; wheel table single home |
| tutorials/04-rust-sdk.md | M | Rust getting-started |
| tutorials/05-build-an-agent.md | R | flagship agent narrative; fix set_max_tool_rounds + sandbox claim |
| tutorials/06-quantize-a-model.md | M | into canonical Quantization page (keep beginner arc + sizing math) |
| guides/index.md | D | |
| guides/agents/index.md | R | keep three-layer framing sentence only |
| guides/agents/tool-calling-basics.md | K | anchor; absorbs strict + configure-tool-loop |
| guides/agents/strict-tool-calling.md | M | one canonical "strict by default" statement |
| guides/agents/configure-tool-loop.md | M | loop-config section |
| guides/agents/enable-code-execution.md | K | absorbs code-execution-design internals; schemas -> pointers |
| guides/agents/web-search.md | K | as-is |
| guides/agents/agentic-runtime.md | R | split: NEW Permissions & Approvals page; wire schemas cede to http-api |
| guides/agents/persist-sessions.md | K | single sessions home; fix content-matching contradiction |
| guides/agents/connect-mcp-server.md | K | absorbs MCP_QUICK_START.md |
| guides/agents/expose-as-mcp.md | R | rewritten against the E1 port (was delete; decision reversed it) |
| guides/serve/index.md | D | |
| guides/serve/http-server.md | M | logging + auth paragraphs -> TOML config reference |
| guides/serve/multiple-models.md | K | endpoint bodies -> http-api links |
| guides/serve/openai-compatible-apis.md | M | examples index survives; rest covered elsewhere |
| guides/serve/openai-responses-api.md | M | unique bits -> openai-compatibility Responses section |
| guides/serve/anthropic-messages-api.md | K | canonical Anthropic home |
| guides/serve/coding-agents.md | K | pin upstream-client claims to refs |
| guides/serve/with-web-ui.md | K | single UI home |
| guides/perf/index.md | D | |
| guides/perf/pick-a-quantization.md | M | canonical Quantization page; add --quant auto |
| guides/perf/auto-tune.md | M | section of Quantization page; fix "benchmarks your hardware" myth |
| guides/perf/use-flash-attention.md | M | one disambiguation paragraph -> PA page; rest dies |
| guides/perf/use-paged-attention.md | M | one PA page with explanation twin |
| guides/perf/use-cuda-graphs.md | K | section of PA page (~70% unique) |
| guides/perf/multi-gpu-distributed.md | M | intro of Distributed page |
| guides/perf/multi-gpu-tensor-parallel.md | K | core of Distributed page |
| guides/perf/multi-node-nccl.md | M | Distributed section; env tables -> environment-variables only |
| guides/perf/multi-machine-ring.md | M | Distributed "Ring backend" section |
| guides/perf/topology.md | K | sole YAML-spec home |
| guides/perf/speculative-decoding.md | M | one MTP page |
| guides/perf/gemma4-mtp.md | M | Gemma 4 section of MTP page |
| guides/perf/use-uqff.md | K | flag enumeration -> CLI reference links |
| guides/perf/online-calibration.md | K | as shipped |
| guides/models/index.md | D | |
| guides/models/text-model-walkthroughs.md | R | merged family-notes page; tables -> supported-models |
| guides/models/vision-model-walkthroughs.md | R | same page; media sections -> Multimodal Input |
| guides/models/use-block-diffusion.md | K | single DiffusionGemma home |
| guides/models/use-embeddings.md | K | add encoding_format/dimensions |
| guides/models/use-image-generation.md | K | fix dict-vs-attribute snippet |
| guides/models/use-speech-models.md | K | wav/pcm contract lives once |
| guides/models/use-vision-input.md | M | into Multimodal Input page |
| guides/models/video-setup.md | K | FFmpeg anchor |
| guides/install/index.md | D | |
| guides/install/linux-cuda.md | M | Quickstart Linux tab; feature table -> cargo-features |
| guides/install/macos-metal.md | M | Quickstart macOS tab; AFQ fact -> quantization home |
| guides/install/windows.md | M | Quickstart Windows tab; fix both wrong tips |
| guides/install/from-source.md | K | Developer Guide > Build from source |
| guides/deploy/index.md | D | |
| guides/deploy/docker.md | R | against E2 migrated images; keep K8s/healthcheck |
| guides/deploy/production-checklist.md | R | true checklist; add /metrics |
| guides/python/index.md | D | |
| guides/python/streaming.md | K | cut blocking half (tutorial 03 owns it) |
| guides/python/multimodal-input.md | M | Python variants into Multimodal Input page |
| guides/python/agentic-session.md | M | into Sessions page |
| guides/rust/index.md | D | |
| guides/rust/streaming.md | K | fix Model-Clone bug |
| guides/rust/embed-in-axum.md | K | fix dep listing |
| guides/customize/index.md | D | |
| guides/customize/lora-adapters.md | K | fix reversed --xlora requires claim |
| guides/customize/anymoe.md | M | into consolidated Python types reference |
| guides/customize/matformer.md | M | family-notes MatFormer section |
| guides/customize/chat-templates.md | R | five false claims; rebuild line-by-line against paths.rs |
| guides/customize/sampling.md | R | rebuild against sampler.rs; delete invented [sampling] TOML |
| explanation/index.md | D | |
| explanation/architecture.md | K | Developer Guide spine |
| explanation/agentic-loop.md | M | into agentic-runtime page |
| explanation/session-memory.md | K | Developer Guide; sole splicing home |
| explanation/quantization-tradeoffs.md | M | concepts into Quantization page (it is the CORRECT AFQ source) |
| explanation/paged-attention.md | M | how-it-works section of PA page |
| explanation/mla.md | M | into Dev Guide model notes; add GLM-4.7-Flash |
| explanation/moe-backends.md | K | Dev Guide |
| explanation/multimodal-pipeline.md | K | sole encoder-cache documentation |
| explanation/code-execution-design.md | M | internals -> code-execution page; isolation -> sandbox.md |
| explanation/device-mapping.md | M | notes into Distributed page; fix terminology |
| reference/index.md | R | regenerate post-IA |
| reference/cli.md | R | REPLACED by generated pages (E4); interactions/examples prose kept as hand-written intro |
| reference/cli-toml-config.md | K | + [sandbox], thinking, flag-mapping column; fix [server] MCP rows per E1 |
| reference/server-config.md | M | into cli-toml-config + generated CLI |
| reference/http-api.md | R | split: generated OpenAPI core (E5) + hand-written SSE/prose page |
| reference/openai-compatibility.md | K | absorbs Responses restrictions |
| reference/mcp-config-schema.md | K | |
| reference/model-notes.md | M | into supported-models |
| reference/quantization-types.md | K | fix stale AFQ-Metal-only claim |
| reference/sandbox.md | K | absorbs threat model; sheds duplicated snippets |
| reference/supported-models.md | K | + notes column; add GraniteMoeHybrid |
| reference/troubleshooting.md | K | symptom pages earn duplication; link canonical homes |
| reference/uqff-format.md | K | |
| reference/cargo-features.md | K | single feature-table home |
| reference/environment-variables.md | K | offline how-to -> User Guide page |
| reference/python/* | K | generated; regen only (E6). Docstring/generator quality pass DEFERRED to the Python API redesign |

New pages: Run any model (GGUF/local files), Structured output, Permissions & Approvals,
Throughput tuning, Rust SDK reference (in-tree), Installation tabs inside Quickstart.

## Examples consolidation

185 -> 109 (-41%). Rust 53->45, Python 67->33, Server 56->26, root strays 9->5. Trees reorganize
to mirror the IA above (topic dirs; per-language leaves only where the surface differs). Full
cluster-by-cluster table in the saved examples report; headline merges: 28-file vision-chat clone
family -> 3 parameterized survivors; per-model text boilerplate -> 1 per surface + model table in
docs; structured-output examples consolidated per surface.

Fix-before-render ledger: 10 broken examples (wrong enums/archs), ~20 stale/misleading (swapped
AnyMoE expert types, GBNF-vs-regex header, dead `adapters` field, 14x dead log_response helper),
header backfill for keepers with thin/missing doc comments.

### Surfacing model

- CLI examples are in-context only: inline shell blocks on answer pages (one-liners do not get
  repo files); the generated CLI reference is the exhaustive layer behind them.
- Answer pages carry surface tabs in a fixed order, `CLI | HTTP | Python | Rust` (omitting tabs
  that do not apply), each tab a short snippet with a "Full example" link to the canonical repo
  example for that surface.
- The E7 hook renders the reorganized example trees into a browsable, searchable Examples section
  grouped by the same topics; rendered example pages link back to their answer page.
- Discoverability depths for any task: tab snippet (90% case) -> full repo example -> generated
  reference.

## Verification gates (cutover PR must pass all)

1. `astro build` clean (docs site compiles, all MDX valid)
2. Dead-link scan over built site (internal links + anchors)
3. Redirect coverage: every pre-cutover URL returns a page (scripted list -> redirects map)
4. CLI golden test: generated pages match committed pages
5. OpenAPI dump matches committed openapi.json; render_pyi diff-clean
6. `cargo check -p mistralrs --features metal --examples` (all surviving Rust examples compile)
7. Python examples AST-checked against mistralrs.pyi symbols
8. Grep gates: no `--mcp-port` references until E1 lands; no legacy `mistralrs-server` invocations
   in docs; killed-page filenames absent from prose

## Execution sequence

- **Wave 0 (engineering, parallel, lands first):** E1 MCP port, E2 Docker migration, E3 OpenAPI
  registration, E4/E5 generators, E6 checklist, E7 render hook. Each is a normal reviewed PR.
- **Wave 1 (content swoop, agent fan-out against this frozen plan):** IA + moves + merges +
  rewrites + new pages + redirects, one PR, gated by the list above. chat-templates and sampling
  rebuilt line-by-line against code by dedicated agents with verification prompts.
- **Wave 2 (examples swoop):** tree reorg + deletions + fixes + header backfill + auto-render,
  one PR, gated by 6-7.
- Quiet window: avoid colliding with release announcements; docs changes on master freeze during
  the swoop branches' lifetime (days, not weeks).

## Out of scope (recorded, not planned)

systemd/bare-metal deployment guide; llms.txt surfacing of canonical pages; generating
environment-variables.md from a const registry; CLI help-string polish pass (value-name casing);
Python API docs quality (pyi docstrings + generator polish) -- deferred until the planned Python
API redesign, since generated docs follow the API for free once it changes.
