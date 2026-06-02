<script lang="ts">
  import { settingsStore } from "../stores/settings.svelte";
  import { modelStore } from "../stores/models.svelte";
  import type { AgentPermission } from "../types";

  let samplingOpen = $state(false);
  const approvalModes: AgentPermission[] = ["auto", "ask", "deny"];

  function handleChange() {
    settingsStore.persist();
  }
</script>

<div class="flex flex-col gap-4 p-4">
  <div class="flex items-center justify-between">
    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Settings</h2>
    <button
      class="rounded-md p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      onclick={() => (settingsStore.settingsOpen = false)}
      aria-label="Close settings"
    >
      <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>

  <!-- Tool toggles -->
  <div class="space-y-3">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Tools</h3>

    <!-- Web Search toggle -->
    <label class="flex items-center justify-between {modelStore.capabilities.search_enabled ? '' : 'opacity-50'}">
      <div class="flex items-center gap-2">
        <svg class="h-4 w-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span class="text-sm text-gray-700 dark:text-gray-300">Web Search</span>
        {#if !modelStore.capabilities.search_enabled}
          <span class="text-xs text-gray-400">(--enable-search)</span>
        {/if}
      </div>
      <input
        type="checkbox"
        class="h-4 w-4 rounded border-gray-300 text-blue-600 accent-blue-600"
        bind:checked={settingsStore.enableSearch}
        onchange={handleChange}
        disabled={!modelStore.capabilities.search_enabled}
      />
    </label>

    <!-- Code Execution toggle -->
    <label class="flex items-center justify-between {modelStore.capabilities.code_execution_enabled ? '' : 'opacity-50'}">
      <div class="flex items-center gap-2">
        <svg class="h-4 w-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        <span class="text-sm text-gray-700 dark:text-gray-300">Code Execution</span>
        {#if !modelStore.capabilities.code_execution_enabled}
          <span class="text-xs text-gray-400">(--enable-code-execution)</span>
        {/if}
      </div>
      <input
        type="checkbox"
        class="h-4 w-4 rounded border-gray-300 text-blue-600 accent-blue-600"
        bind:checked={settingsStore.enableCodeExecution}
        onchange={handleChange}
        disabled={!modelStore.capabilities.code_execution_enabled}
      />
    </label>

    <!-- Thinking toggle -->
    <label class="flex items-center justify-between">
      <div class="flex items-center gap-2">
        <svg class="h-4 w-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span class="text-sm text-gray-700 dark:text-gray-300">Thinking</span>
      </div>
      <input
        type="checkbox"
        class="h-4 w-4 rounded border-gray-300 text-blue-600 accent-blue-600"
        bind:checked={settingsStore.enableThinking}
        onchange={handleChange}
      />
    </label>

    <!-- Tool dispatch URL -->
    {#if modelStore.capabilities.tool_dispatch_url}
      <div>
        <span class="block text-xs text-gray-500 dark:text-gray-400 mb-1">Tool Dispatch URL</span>
        <div class="rounded-md bg-gray-100 px-2 py-1.5 text-xs font-mono text-gray-600 dark:bg-gray-800 dark:text-gray-400">
          {modelStore.capabilities.tool_dispatch_url}
        </div>
      </div>
    {/if}

    <div class="space-y-1.5">
      <div class="flex items-center justify-between">
        <span class="text-sm text-gray-700 dark:text-gray-300">Tool approval</span>
        <span class="text-xs text-gray-400 dark:text-gray-500">agent_permission</span>
      </div>
      <div class="grid grid-cols-3 overflow-hidden rounded-lg border border-gray-200 bg-gray-100 p-0.5 dark:border-gray-700 dark:bg-gray-800">
        {#each approvalModes as mode}
          <button
            type="button"
            class="rounded-md px-2 py-1.5 text-xs font-medium transition-colors {settingsStore.agentPermission === mode ? 'bg-white text-gray-900 shadow-sm dark:bg-gray-700 dark:text-gray-100' : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'}"
            onclick={() => {
              settingsStore.agentPermission = mode;
              handleChange();
            }}
            aria-pressed={settingsStore.agentPermission === mode}
          >
            {mode}
          </button>
        {/each}
      </div>
    </div>
  </div>

  <div class="border-t border-gray-200 dark:border-gray-700"></div>

  <!-- System Prompt -->
  <div>
    <h3 class="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">System Prompt</h3>
    <textarea
      class="w-full resize-none rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm outline-none placeholder:text-gray-400 focus:border-blue-400 focus:ring-1 focus:ring-blue-400 dark:border-gray-600 dark:bg-gray-800 dark:placeholder:text-gray-500"
      rows="3"
      placeholder="Enter system prompt..."
      bind:value={settingsStore.systemPrompt}
      onchange={handleChange}
    ></textarea>
  </div>

  <div class="border-t border-gray-200 dark:border-gray-700"></div>

  <!-- Sampling Parameters (collapsed) -->
  <details bind:open={samplingOpen} class="group">
    <summary class="flex cursor-pointer items-center gap-2 text-xs font-semibold uppercase tracking-wider text-gray-500 select-none dark:text-gray-400">
      <svg class="h-3.5 w-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
      </svg>
      Sampling Parameters
    </summary>

    <div class="mt-3 space-y-4">
      <!-- Temperature -->
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class="text-xs text-gray-600 dark:text-gray-400">Temperature</span>
          <span class="text-xs font-mono text-gray-500">{settingsStore.temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          bind:value={settingsStore.temperature}
          onchange={handleChange}
          class="w-full accent-blue-600"
        />
      </div>

      <!-- Top P -->
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class="text-xs text-gray-600 dark:text-gray-400">Top P</span>
          <span class="text-xs font-mono text-gray-500">{settingsStore.topP.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          bind:value={settingsStore.topP}
          onchange={handleChange}
          class="w-full accent-blue-600"
        />
      </div>

      <!-- Top K -->
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class="text-xs text-gray-600 dark:text-gray-400">Top K</span>
          <span class="text-xs font-mono text-gray-500">{settingsStore.topK}</span>
        </div>
        <input
          type="range"
          min="0"
          max="200"
          step="1"
          bind:value={settingsStore.topK}
          onchange={handleChange}
          class="w-full accent-blue-600"
        />
      </div>

      <!-- Max Tokens -->
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class="text-xs text-gray-600 dark:text-gray-400">Max Tokens</span>
          <input
            type="number"
            min="1"
            max="131072"
            bind:value={settingsStore.maxTokens}
            onchange={handleChange}
            class="w-20 rounded border border-gray-300 bg-gray-50 px-2 py-0.5 text-right text-xs outline-none focus:border-blue-400 dark:border-gray-600 dark:bg-gray-800"
          />
        </div>
      </div>

      <!-- Repetition Penalty -->
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class="text-xs text-gray-600 dark:text-gray-400">Repetition Penalty</span>
          <span class="text-xs font-mono text-gray-500">{settingsStore.repetitionPenalty.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="1"
          max="2"
          step="0.05"
          bind:value={settingsStore.repetitionPenalty}
          onchange={handleChange}
          class="w-full accent-blue-600"
        />
      </div>
    </div>
  </details>
</div>
