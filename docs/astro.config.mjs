import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightOpenAPI, { openAPISidebarGroups } from 'starlight-openapi';

// Deployed at https://docs.mistralrs.dev/
export default defineConfig({
  site: 'https://docs.mistralrs.dev',
  base: '/',
  // Allow access over Tailscale.
  vite: { preview: { allowedHosts: ['.ts.net'] } },
  redirects: {
    // Entry + quickstart
    '/start-here/': '/quickstart/',
    '/tutorials/': '/quickstart/',
    '/tutorials/01-install-and-run/': '/quickstart/',
    '/tutorials/02-serve-an-api/': '/quickstart/',
    '/tutorials/03-python-sdk/': '/guides/python/getting-started/',
    '/tutorials/04-rust-sdk/': '/guides/rust/getting-started/',
    '/tutorials/05-build-an-agent/': '/guides/agents/build-an-agent/',
    '/tutorials/06-quantize-a-model/': '/guides/quantization/quantize-a-model/',
    '/guides/': '/',
    '/guides/install/': '/quickstart/',
    '/guides/install/linux-cuda/': '/quickstart/',
    '/guides/install/macos-metal/': '/quickstart/',
    '/guides/install/windows/': '/quickstart/',
    '/guides/install/from-source/': '/developer/from-source/',
    // Serving
    '/guides/serve/': '/guides/serve/openai-compatible-apis/',
    '/guides/serve/http-server/': '/guides/serve/openai-compatible-apis/',
    '/guides/serve/openai-responses-api/': '/reference/openai-compatibility/#responses-api',
    // Models
    '/guides/models/': '/guides/models/run-any-model/',
    '/guides/models/text-model-walkthroughs/': '/guides/models/model-family-notes/',
    '/guides/models/vision-model-walkthroughs/': '/guides/models/model-family-notes/',
    '/guides/models/use-vision-input/': '/guides/models/multimodal-input/',
    '/guides/python/multimodal-input/': '/guides/models/multimodal-input/',
    // Quantization
    '/guides/perf/pick-a-quantization/': '/guides/quantization/quantize-a-model/',
    '/guides/perf/auto-tune/': '/guides/quantization/quantize-a-model/',
    '/guides/perf/use-uqff/': '/guides/quantization/uqff/',
    '/guides/perf/online-calibration/': '/guides/quantization/online-calibration/',
    '/explanation/quantization-tradeoffs/': '/guides/quantization/quantize-a-model/',
    // Agents & tools
    '/guides/agents/strict-tool-calling/': '/guides/agents/tool-calling-basics/',
    '/guides/agents/configure-tool-loop/': '/guides/agents/tool-calling-basics/',
    '/guides/python/agentic-session/': '/guides/agents/persist-sessions/',
    '/explanation/agentic-loop/': '/guides/agents/agentic-runtime/',
    '/explanation/code-execution-design/': '/reference/sandbox/',
    // Performance + deploy
    '/guides/perf/': '/guides/perf/paged-attention/',
    '/guides/perf/use-paged-attention/': '/guides/perf/paged-attention/',
    '/guides/perf/use-flash-attention/': '/guides/perf/paged-attention/',
    '/guides/perf/use-cuda-graphs/': '/guides/perf/paged-attention/',
    '/explanation/paged-attention/': '/guides/perf/paged-attention/',
    '/guides/perf/multi-gpu-distributed/': '/guides/perf/distributed-inference/',
    '/guides/perf/multi-gpu-tensor-parallel/': '/guides/perf/distributed-inference/',
    '/guides/perf/multi-node-nccl/': '/guides/perf/distributed-inference/',
    '/guides/perf/multi-machine-ring/': '/guides/perf/distributed-inference/',
    '/explanation/device-mapping/': '/guides/perf/distributed-inference/',
    '/guides/perf/gemma4-mtp/': '/guides/perf/speculative-decoding/',
    '/guides/deploy/': '/guides/deploy/docker/',
    // SDKs + customize ( /guides/python/ , /guides/rust/ , /guides/customize/ keep real index pages)
    '/guides/customize/anymoe/': '/guides/customize/lora-adapters/',
    '/guides/customize/matformer/': '/guides/models/model-family-notes/#matformer',
    '/examples/python/lora-zephyr/': '/examples/python/lora/',
    '/explanation/mla/': '/guides/models/model-family-notes/',
    // Reference + developer
    '/reference/server-config/': '/reference/cli-toml-config/',
    '/reference/model-notes/': '/reference/supported-models/',
    '/explanation/': '/developer/',
    '/explanation/architecture/': '/developer/architecture/',
    '/explanation/moe-backends/': '/developer/moe-backends/',
    '/explanation/multimodal-pipeline/': '/developer/multimodal-pipeline/',
    '/explanation/session-memory/': '/developer/session-memory/',
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
                'guides/models/run-gguf',
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
                'guides/deploy/observability',
                'guides/deploy/production-checklist',
              ],
            },
          ],
        },
        {
          label: 'Examples',
          collapsed: true,
          items: [{ autogenerate: { directory: 'examples', collapsed: true } }],
        },
        {
          label: 'Reference',
          collapsed: true,
          items: [{ autogenerate: { directory: 'reference', collapsed: true } }],
        },
        ...openAPISidebarGroups,
        {
          label: 'Developer Guide',
          collapsed: true,
          items: [{ autogenerate: { directory: 'developer', collapsed: true } }],
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
