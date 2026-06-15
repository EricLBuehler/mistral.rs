import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightOpenAPI, { openAPISidebarGroups } from 'starlight-openapi';

// Deployed at https://ericlbuehler.github.io/mistral.rs/
// Adjust `site` + `base` if we move to docs.mistral.rs.
export default defineConfig({
  site: 'https://ericlbuehler.github.io',
  base: '/mistral.rs',
  // Allow access over Tailscale.
  vite: { preview: { allowedHosts: ['.ts.net'] } },
  redirects: {
    // Entry + quickstart
    '/start-here/': '/mistral.rs/quickstart/',
    '/tutorials/': '/mistral.rs/quickstart/',
    '/tutorials/01-install-and-run/': '/mistral.rs/quickstart/',
    '/tutorials/02-serve-an-api/': '/mistral.rs/quickstart/',
    '/tutorials/03-python-sdk/': '/mistral.rs/guides/python/getting-started/',
    '/tutorials/04-rust-sdk/': '/mistral.rs/guides/rust/getting-started/',
    '/tutorials/05-build-an-agent/': '/mistral.rs/guides/agents/build-an-agent/',
    '/tutorials/06-quantize-a-model/': '/mistral.rs/guides/quantization/quantize-a-model/',
    '/guides/': '/mistral.rs/',
    '/guides/install/': '/mistral.rs/quickstart/',
    '/guides/install/linux-cuda/': '/mistral.rs/quickstart/',
    '/guides/install/macos-metal/': '/mistral.rs/quickstart/',
    '/guides/install/windows/': '/mistral.rs/quickstart/',
    '/guides/install/from-source/': '/mistral.rs/developer/from-source/',
    // Serving
    '/guides/serve/': '/mistral.rs/guides/serve/openai-compatible-apis/',
    '/guides/serve/http-server/': '/mistral.rs/guides/serve/openai-compatible-apis/',
    '/guides/serve/openai-responses-api/': '/mistral.rs/reference/openai-compatibility/#responses-api',
    // Models
    '/guides/models/': '/mistral.rs/guides/models/run-any-model/',
    '/guides/models/text-model-walkthroughs/': '/mistral.rs/guides/models/model-family-notes/',
    '/guides/models/vision-model-walkthroughs/': '/mistral.rs/guides/models/model-family-notes/',
    '/guides/models/use-vision-input/': '/mistral.rs/guides/models/multimodal-input/',
    '/guides/python/multimodal-input/': '/mistral.rs/guides/models/multimodal-input/',
    // Quantization
    '/guides/perf/pick-a-quantization/': '/mistral.rs/guides/quantization/quantize-a-model/',
    '/guides/perf/auto-tune/': '/mistral.rs/guides/quantization/quantize-a-model/',
    '/guides/perf/use-uqff/': '/mistral.rs/guides/quantization/uqff/',
    '/guides/perf/online-calibration/': '/mistral.rs/guides/quantization/online-calibration/',
    '/explanation/quantization-tradeoffs/': '/mistral.rs/guides/quantization/quantize-a-model/',
    // Agents & tools
    '/guides/agents/strict-tool-calling/': '/mistral.rs/guides/agents/tool-calling-basics/',
    '/guides/agents/configure-tool-loop/': '/mistral.rs/guides/agents/tool-calling-basics/',
    '/guides/python/agentic-session/': '/mistral.rs/guides/agents/persist-sessions/',
    '/explanation/agentic-loop/': '/mistral.rs/guides/agents/agentic-runtime/',
    '/explanation/code-execution-design/': '/mistral.rs/reference/sandbox/',
    // Performance + deploy
    '/guides/perf/': '/mistral.rs/guides/perf/paged-attention/',
    '/guides/perf/use-paged-attention/': '/mistral.rs/guides/perf/paged-attention/',
    '/guides/perf/use-flash-attention/': '/mistral.rs/guides/perf/paged-attention/',
    '/guides/perf/use-cuda-graphs/': '/mistral.rs/guides/perf/paged-attention/',
    '/explanation/paged-attention/': '/mistral.rs/guides/perf/paged-attention/',
    '/guides/perf/multi-gpu-distributed/': '/mistral.rs/guides/perf/distributed-inference/',
    '/guides/perf/multi-gpu-tensor-parallel/': '/mistral.rs/guides/perf/distributed-inference/',
    '/guides/perf/multi-node-nccl/': '/mistral.rs/guides/perf/distributed-inference/',
    '/guides/perf/multi-machine-ring/': '/mistral.rs/guides/perf/distributed-inference/',
    '/explanation/device-mapping/': '/mistral.rs/guides/perf/distributed-inference/',
    '/guides/perf/gemma4-mtp/': '/mistral.rs/guides/perf/speculative-decoding/',
    '/guides/deploy/': '/mistral.rs/guides/deploy/docker/',
    // SDKs + customize ( /guides/python/ , /guides/rust/ , /guides/customize/ keep real index pages)
    '/guides/customize/anymoe/': '/mistral.rs/guides/customize/lora-adapters/',
    '/guides/customize/matformer/': '/mistral.rs/guides/models/model-family-notes/#matformer',
    '/explanation/mla/': '/mistral.rs/guides/models/model-family-notes/',
    // Reference + developer
    '/reference/server-config/': '/mistral.rs/reference/cli-toml-config/',
    '/reference/model-notes/': '/mistral.rs/reference/supported-models/',
    '/explanation/': '/mistral.rs/developer/',
    '/explanation/architecture/': '/mistral.rs/developer/architecture/',
    '/explanation/moe-backends/': '/mistral.rs/developer/moe-backends/',
    '/explanation/multimodal-pipeline/': '/mistral.rs/developer/multimodal-pipeline/',
    '/explanation/session-memory/': '/mistral.rs/developer/session-memory/',
  },
  integrations: [
    starlight({
      title: 'mistral.rs',
      description: 'Fast, flexible LLM inference engine written in Rust.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/EricLBuehler/mistral.rs' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/SZrecqK8qw' },
      ],
      editLink: {
        baseUrl: 'https://github.com/EricLBuehler/mistral.rs/edit/master/docs/',
      },
      // openapi.json is refreshed by `cargo test -p mistralrs-server-core dump_openapi_json`
      plugins: [
        starlightOpenAPI([
          {
            base: 'reference/http-api-generated',
            schema: './openapi.json',
            label: 'HTTP API (generated)',
          },
        ]),
      ],
      sidebar: [
        {
          label: 'Quickstart',
          slug: 'quickstart',
        },
        {
          label: 'User Guide',
          items: [
            {
              label: 'Serving',
              collapsed: true,
              items: [
                'guides/serve/openai-compatible-apis',
                'guides/serve/anthropic-messages-api',
                'guides/serve/structured-output',
                'guides/serve/with-web-ui',
                'guides/serve/multiple-models',
                'guides/serve/coding-agents',
              ],
            },
            {
              label: 'Models',
              collapsed: true,
              items: [
                'guides/models/run-any-model',
                'guides/models/model-family-notes',
                'guides/models/multimodal-input',
                'guides/models/video-setup',
                'guides/models/use-speech-models',
                'guides/models/use-image-generation',
                'guides/models/use-embeddings',
                'guides/models/use-block-diffusion',
              ],
            },
            {
              label: 'Quantization',
              collapsed: true,
              items: [
                'guides/quantization/quantize-a-model',
                'guides/quantization/uqff',
                'guides/quantization/online-calibration',
              ],
            },
            {
              label: 'Agents & tools',
              collapsed: true,
              items: [
                'guides/agents',
                'guides/agents/build-an-agent',
                'guides/agents/tool-calling-basics',
                'guides/agents/enable-code-execution',
                'guides/agents/enable-shell',
                'guides/agents/skills',
                'guides/agents/file-inputs',
                'guides/agents/web-search',
                'guides/agents/permissions-and-approvals',
                'guides/agents/agentic-runtime',
                'guides/agents/persist-sessions',
                'guides/agents/connect-mcp-server',
                'guides/agents/expose-as-mcp',
              ],
            },
            {
              label: 'Python SDK',
              collapsed: true,
              items: [
                'guides/python',
                'guides/python/getting-started',
                'guides/python/streaming',
              ],
            },
            {
              label: 'Rust SDK',
              collapsed: true,
              items: [
                'guides/rust',
                'guides/rust/getting-started',
                'guides/rust/streaming',
                'guides/rust/embed-in-axum',
              ],
            },
            {
              label: 'Customize',
              collapsed: true,
              items: [
                'guides/customize',
                'guides/customize/chat-templates',
                'guides/customize/sampling',
                'guides/customize/lora-adapters',
              ],
            },
            {
              label: 'Performance & scaling',
              collapsed: true,
              items: [
                'guides/perf/paged-attention',
                'guides/perf/speculative-decoding',
                'guides/perf/distributed-inference',
                'guides/perf/topology',
                'guides/perf/throughput-tuning',
              ],
            },
            {
              label: 'Deploy',
              collapsed: true,
              items: [
                'guides/deploy/docker',
                'guides/deploy/production-checklist',
              ],
            },
          ],
        },
        {
          label: 'Examples',
          collapsed: true,
          autogenerate: { directory: 'examples' },
        },
        {
          label: 'Reference',
          collapsed: true,
          autogenerate: { directory: 'reference' },
        },
        ...openAPISidebarGroups,
        {
          label: 'Developer Guide',
          collapsed: true,
          autogenerate: { directory: 'developer' },
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
