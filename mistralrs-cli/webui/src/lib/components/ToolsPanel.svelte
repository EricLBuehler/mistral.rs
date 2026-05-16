<script lang="ts">
  import { settingsStore } from "../stores/settings.svelte";
  import { modelStore } from "../stores/models.svelte";
  import * as api from "../services/api";
  import type { McpToolInfo } from "../services/api";

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
  <!-- Header -->
  <div class="flex items-center justify-between border-b border-gray-200 px-4 py-2.5 dark:border-gray-800">
    <div class="flex items-center gap-2">
      <svg class="h-4 w-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
      </svg>
      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Tools</h2>
    </div>
    <button
      class="rounded-md p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      onclick={() => (settingsStore.toolsOpen = false)}
      aria-label="Close tools panel"
    >
      <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>

  <!-- Sections -->
  <div class="flex-1 space-y-4 overflow-y-auto p-4">
    <!-- Capabilities -->
    <details bind:open={capabilitiesOpen} class="group">
      <summary class="flex cursor-pointer items-center gap-2 text-xs font-semibold uppercase tracking-wider text-gray-500 select-none dark:text-gray-400">
        <svg class="h-3.5 w-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
        </svg>
        Capabilities
      </summary>
      <div class="mt-2 space-y-1.5 text-sm">
        <div class="flex items-center justify-between rounded-md bg-gray-50 px-2.5 py-1.5 dark:bg-gray-800/50">
          <div class="flex items-center gap-1.5">
            <svg class="h-3.5 w-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span class="text-gray-700 dark:text-gray-300">Web Search</span>
          </div>
          {#if modelStore.capabilities.search_enabled}
            <span class="text-xs text-green-600 dark:text-green-400">enabled</span>
          {:else}
            <span class="text-xs text-gray-400">off</span>
          {/if}
        </div>
        <div class="flex items-center justify-between rounded-md bg-gray-50 px-2.5 py-1.5 dark:bg-gray-800/50">
          <div class="flex items-center gap-1.5">
            <svg class="h-3.5 w-3.5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <span class="text-gray-700 dark:text-gray-300">Code Execution</span>
          </div>
          {#if modelStore.capabilities.code_execution_enabled}
            <span class="text-xs text-purple-600 dark:text-purple-400">enabled</span>
          {:else}
            <span class="text-xs text-gray-400">off</span>
          {/if}
        </div>
        {#if modelStore.capabilities.tool_dispatch_url}
          <div class="rounded-md bg-gray-50 px-2.5 py-1.5 text-xs dark:bg-gray-800/50">
            <div class="text-gray-500 dark:text-gray-400">Tool Dispatch URL</div>
            <div class="mt-0.5 truncate font-mono text-gray-700 dark:text-gray-300" title={modelStore.capabilities.tool_dispatch_url}>
              {modelStore.capabilities.tool_dispatch_url}
            </div>
          </div>
        {/if}
      </div>
    </details>

    <!-- MCP Tools -->
    <details bind:open={mcpOpen} class="group">
      <summary class="flex cursor-pointer items-center gap-2 text-xs font-semibold uppercase tracking-wider text-gray-500 select-none dark:text-gray-400">
        <svg class="h-3.5 w-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
        </svg>
        MCP Tools
        {#if !mcpLoading && !mcpError}
          <span class="ml-1 rounded-full bg-gray-200 px-1.5 py-0.5 text-[10px] font-normal normal-case text-gray-600 dark:bg-gray-700 dark:text-gray-400">
            {mcpTools.length}
          </span>
        {/if}
        <button
          class="ml-auto rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 dark:hover:text-gray-300"
          onclick={(e) => { e.stopPropagation(); e.preventDefault(); loadMcpTools(); }}
          title="Refresh"
          aria-label="Refresh MCP tools"
        >
          <svg class="h-3 w-3 {mcpLoading ? 'animate-spin' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </summary>
      <div class="mt-2 space-y-1.5">
        {#if mcpLoading}
          <div class="rounded-md bg-gray-50 px-2.5 py-2 text-xs text-gray-500 dark:bg-gray-800/50 dark:text-gray-400">
            Loading…
          </div>
        {:else if mcpError}
          <div class="rounded-md bg-red-50 px-2.5 py-2 text-xs text-red-600 dark:bg-red-900/20 dark:text-red-400">
            {mcpError}
          </div>
        {:else if mcpTools.length === 0}
          <div class="rounded-md bg-gray-50 px-2.5 py-2 text-xs text-gray-500 dark:bg-gray-800/50 dark:text-gray-400">
            No MCP servers configured.
            <span class="block mt-1 text-gray-400 dark:text-gray-500">
              Run with <code class="font-mono">--mcp-config &lt;file&gt;</code> to connect MCP servers.
            </span>
          </div>
        {:else}
          {#each mcpTools as tool}
            <details class="group/tool rounded-md bg-gray-50 dark:bg-gray-800/50">
              <summary class="flex cursor-pointer items-center gap-1.5 px-2.5 py-1.5 text-sm select-none">
                <svg class="h-3 w-3 shrink-0 transition-transform group-open/tool:rotate-90 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
                <svg class="h-3.5 w-3.5 shrink-0 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span class="truncate font-mono text-xs text-gray-700 dark:text-gray-300" title={tool.name}>{tool.name}</span>
              </summary>
              {#if tool.description}
                <div class="border-t border-gray-200 px-2.5 py-2 text-xs text-gray-600 dark:border-gray-700 dark:text-gray-400">
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
