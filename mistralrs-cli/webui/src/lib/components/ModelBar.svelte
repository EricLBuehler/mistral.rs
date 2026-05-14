<script lang="ts">
  import { modelStore } from "../stores/models.svelte";
  import { settingsStore } from "../stores/settings.svelte";
  import * as api from "../services/api";
  import type { UiModelInfo } from "../types";

  let dropdownOpen = $state(false);
  let dropdownEl = $state<HTMLDivElement | null>(null);

  let connection = $derived(`${window.location.hostname}:${window.location.port || (window.location.protocol === "https:" ? "443" : "80")}`);

  /** Compact in/out modality string for the dropdown row. */
  function modalityRow(model: UiModelInfo): string | null {
    const inp = model.input_modalities ?? [];
    const out = model.output_modalities ?? [];
    if (inp.length === 0 && out.length === 0) return null;
    return `${inp.join("+") || "—"} → ${out.join("+") || "—"}`;
  }

  async function pickModel(name: string) {
    dropdownOpen = false;
    if (name === modelStore.selectedModel) return;
    modelStore.selectModel(name);
    try {
      await api.selectModel(name);
    } catch (e) {
      console.error("Failed to switch model:", e);
    }
  }

  function onDocClick(e: MouseEvent) {
    if (dropdownEl && !dropdownEl.contains(e.target as Node)) {
      dropdownOpen = false;
    }
  }

  $effect(() => {
    if (dropdownOpen) {
      document.addEventListener("click", onDocClick);
      return () => document.removeEventListener("click", onDocClick);
    }
  });
</script>

<header class="flex h-12 shrink-0 items-center gap-3 border-b border-gray-200 bg-white px-4 dark:border-gray-800 dark:bg-gray-950">
  <button
    class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
    onclick={() => (settingsStore.sidebarOpen = !settingsStore.sidebarOpen)}
    aria-label="Toggle sidebar"
  >
    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  </button>

  <!-- Model dropdown chip -->
  <div class="relative" bind:this={dropdownEl}>
    <button
      class="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 px-2.5 py-1 text-sm hover:border-gray-300 dark:border-gray-800 dark:bg-gray-900 dark:hover:border-gray-700"
      onclick={() => (dropdownOpen = !dropdownOpen)}
      aria-haspopup="listbox"
      aria-expanded={dropdownOpen}
    >
      {#if modelStore.selectedModel}
        <span class="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
          {modelStore.selectedModelKind}
        </span>
        <span class="font-mono text-xs text-gray-800 dark:text-gray-200">{modelStore.selectedModel}</span>
      {:else}
        <span class="text-xs text-gray-400">No model selected</span>
      {/if}
      <svg class="h-3 w-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 9l6 6 6-6" />
      </svg>
    </button>

    {#if dropdownOpen}
      <div
        class="absolute left-0 top-full z-50 mt-1 max-h-96 min-w-full overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-900"
        role="listbox"
      >
        {#if modelStore.models.length === 0}
          <div class="whitespace-nowrap px-3 py-2 text-sm text-gray-400">No models loaded</div>
        {:else}
          {#each modelStore.models as model (model.name)}
            {@const detail = modalityRow(model)}
            <button
              class="flex w-full items-start gap-2 whitespace-nowrap px-3 py-2 text-left text-sm hover:bg-gray-50 dark:hover:bg-gray-800 {model.name === modelStore.selectedModel ? 'bg-blue-50 dark:bg-blue-900/20' : ''}"
              onclick={() => pickModel(model.name)}
              role="option"
              aria-selected={model.name === modelStore.selectedModel}
            >
              <span class="mt-0.5 rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                {model.kind}
              </span>
              <div class="flex flex-1 flex-col">
                <span class="font-mono text-xs">{model.name}</span>
                {#if detail}
                  <span class="font-mono text-[10.5px] text-gray-400 dark:text-gray-500">{detail}</span>
                {/if}
              </div>
              {#if model.name === modelStore.selectedModel}
                <svg class="mt-0.5 h-3.5 w-3.5 shrink-0 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
              {/if}
            </button>
          {/each}
        {/if}
      </div>
    {/if}
  </div>

  <!-- Inline sampling params (all active) -->
  <div class="hidden items-center gap-3 font-mono text-[11px] text-gray-500 md:flex dark:text-gray-400">
    {#if settingsStore.temperature != null}
      <span><span class="text-gray-400 dark:text-gray-500">temp</span> {settingsStore.temperature}</span>
    {/if}
    {#if settingsStore.topP != null}
      <span><span class="text-gray-400 dark:text-gray-500">top_p</span> {settingsStore.topP}</span>
    {/if}
    {#if settingsStore.topK}
      <span><span class="text-gray-400 dark:text-gray-500">top_k</span> {settingsStore.topK}</span>
    {/if}
    {#if settingsStore.repetitionPenalty}
      <span><span class="text-gray-400 dark:text-gray-500">rep</span> {settingsStore.repetitionPenalty}</span>
    {/if}
    {#if settingsStore.maxTokens}
      <span><span class="text-gray-400 dark:text-gray-500">max</span> {settingsStore.maxTokens}</span>
    {/if}
    {#if settingsStore.enableThinking}
      <span class="text-amber-600 dark:text-amber-400">thinking</span>
    {/if}
  </div>

  <!-- Active tool indicators -->
  <div class="ml-auto flex items-center gap-1.5">
    {#if settingsStore.enableSearch && modelStore.capabilities.search_enabled}
      <span class="flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/40 dark:text-green-300" title="Web search active">
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        Search
      </span>
    {/if}
    {#if settingsStore.enableCodeExecution && modelStore.capabilities.code_execution_enabled}
      <span class="flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700 dark:bg-purple-900/40 dark:text-purple-300" title="Code execution active">
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        Code
      </span>
    {/if}
    <span class="hidden items-center gap-1.5 rounded-full border border-gray-200 px-2 py-0.5 font-mono text-[11px] text-gray-500 md:inline-flex dark:border-gray-700 dark:text-gray-400" title="Server endpoint">
      <span class="h-1.5 w-1.5 rounded-full bg-green-500"></span>
      {connection}
    </span>
  </div>

  <button
    class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
    onclick={() => (settingsStore.toolsOpen = !settingsStore.toolsOpen)}
    aria-label="Tools panel"
    title="Tools"
  >
    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
    </svg>
  </button>

  <button
    class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
    onclick={() => (settingsStore.settingsOpen = !settingsStore.settingsOpen)}
    aria-label="Settings"
  >
    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  </button>
</header>
