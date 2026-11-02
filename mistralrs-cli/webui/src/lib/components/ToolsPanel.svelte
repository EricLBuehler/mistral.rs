<script lang="ts">
  import { settingsStore } from "../stores/settings.svelte";
  import { modelStore } from "../stores/models.svelte";
  import * as api from "../services/api";
  import type { McpToolInfo } from "../services/api";
  import { ICON_BUTTON, PANEL_HEADING } from "../ui";
  import Wrench from "@lucide/svelte/icons/wrench";
  import X from "@lucide/svelte/icons/x";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import Globe from "@lucide/svelte/icons/globe";
  import Code from "@lucide/svelte/icons/code";
  import Terminal from "@lucide/svelte/icons/terminal";
  import RefreshCw from "@lucide/svelte/icons/refresh-cw";
  import Zap from "@lucide/svelte/icons/zap";

  const ROW = "flex items-center justify-between rounded-md border border-line bg-background px-2.5 py-1.5";
  const NOTE = "rounded-md border border-line bg-background px-2.5 py-2 text-xs text-muted";

  let capabilitiesOpen = $state(true);
  let mcpOpen = $state(true);

  let mcpTools = $state<McpToolInfo[]>([]);
  let mcpLoading = $state(true);
  let mcpError = $state<string | null>(null);

  async function loadMcpTools() {
    mcpLoading = true;
    mcpError = null;
    try {
      const { tools } = await api.listMcpTools();
      mcpTools = tools;
    } catch (e) {
      mcpError = String(e);
    } finally {
      mcpLoading = false;
    }
  }

  // Load when the panel opens
  $effect(() => {
    if (settingsStore.toolsOpen) {
      loadMcpTools();
    }
  });
</script>

<div class="flex h-full flex-col">
  <div class="flex items-center justify-between border-b border-line px-4 py-2">
    <div class="flex items-center gap-2">
      <Wrench class="h-4 w-4 text-muted" />
      <h2 class="text-sm font-semibold text-foreground">Tools</h2>
    </div>
    <button
      class={ICON_BUTTON}
      onclick={() => (settingsStore.toolsOpen = false)}
      aria-label="Close tools panel"
    >
      <X class="h-4 w-4" />
    </button>
  </div>

  <div class="flex-1 space-y-4 overflow-y-auto p-4">
    <details bind:open={capabilitiesOpen} class="group">
      <summary class="flex cursor-pointer select-none items-center gap-1.5 {PANEL_HEADING} hover:text-foreground">
        <ChevronRight class="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
        Capabilities
      </summary>
      <div class="mt-2 space-y-1.5 text-sm">
        {#snippet capability(enabled: boolean)}
          {#if enabled}
            <span class="flex items-center gap-1.5 text-xs text-foreground">
              <span class="h-1.5 w-1.5 rounded-full bg-success"></span>
              enabled
            </span>
          {:else}
            <span class="text-xs text-muted">off</span>
          {/if}
        {/snippet}
        <div class={ROW}>
          <div class="flex items-center gap-1.5">
            <Globe class="h-3.5 w-3.5 text-muted" />
            <span class="text-foreground">Web Search</span>
          </div>
          {@render capability(modelStore.capabilities.search_enabled)}
        </div>
        <div class={ROW}>
          <div class="flex items-center gap-1.5">
            <Code class="h-3.5 w-3.5 text-muted" />
            <span class="text-foreground">Code Execution</span>
          </div>
          {@render capability(modelStore.capabilities.code_execution_enabled)}
        </div>
        <div class={ROW}>
          <div class="flex items-center gap-1.5">
            <Terminal class="h-3.5 w-3.5 text-muted" />
            <span class="text-foreground">Shell</span>
          </div>
          {@render capability(modelStore.capabilities.shell_enabled)}
        </div>
        {#if modelStore.capabilities.tool_dispatch_url}
          <div class="rounded-md border border-line bg-background px-2.5 py-1.5 text-xs">
            <div class="text-muted">Tool Dispatch URL</div>
            <div class="mt-0.5 truncate font-mono text-muted-strong" title={modelStore.capabilities.tool_dispatch_url}>
              {modelStore.capabilities.tool_dispatch_url}
            </div>
          </div>
        {/if}
      </div>
    </details>

    <details bind:open={mcpOpen} class="group">
      <summary class="flex cursor-pointer select-none items-center gap-1.5 {PANEL_HEADING} hover:text-foreground">
        <ChevronRight class="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
        MCP Tools
        {#if !mcpLoading && !mcpError}
          <span class="ml-1 rounded-full border border-line-strong px-1.5 font-mono text-[10px] font-normal normal-case tracking-normal text-muted">
            {mcpTools.length}
          </span>
        {/if}
        <button
          class="ml-auto inline-flex h-6 w-6 items-center justify-center rounded-md text-muted hover:bg-surface-raised hover:text-foreground"
          onclick={(e) => { e.stopPropagation(); e.preventDefault(); loadMcpTools(); }}
          title="Refresh"
          aria-label="Refresh MCP tools"
        >
          <RefreshCw class="h-3.5 w-3.5 {mcpLoading ? 'animate-spin' : ''}" />
        </button>
      </summary>
      <div class="mt-2 space-y-1.5">
        {#if mcpLoading}
          <div class={NOTE}>
            Loading...
          </div>
        {:else if mcpError}
          <div class="rounded-md border border-danger/30 bg-danger-soft px-2.5 py-2 text-xs text-danger">
            {mcpError}
          </div>
        {:else if mcpTools.length === 0}
          <div class={NOTE}>
            No MCP servers configured.
            <span class="mt-1 block">
              Run with <code class="font-mono text-muted-strong">--mcp-config &lt;file&gt;</code> to connect MCP servers.
            </span>
          </div>
        {:else}
          {#each mcpTools as tool}
            <details class="group/tool rounded-md border border-line bg-background">
              <summary class="flex cursor-pointer select-none items-center gap-1.5 px-2.5 py-1.5 text-sm">
                <ChevronRight class="h-3.5 w-3.5 shrink-0 text-muted transition-transform group-open/tool:rotate-90" />
                <Zap class="h-3.5 w-3.5 shrink-0 text-muted" />
                <span class="truncate font-mono text-xs text-foreground" title={tool.name}>{tool.name}</span>
              </summary>
              {#if tool.description}
                <div class="border-t border-line px-2.5 py-2 text-xs leading-relaxed text-muted">
                  {tool.description}
                </div>
              {/if}
            </details>
          {/each}
        {/if}
      </div>
    </details>
  </div>
</div>
