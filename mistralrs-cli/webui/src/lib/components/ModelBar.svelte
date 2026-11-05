<script lang="ts">
  import { modelStore } from "../stores/models.svelte";
  import { settingsStore } from "../stores/settings.svelte";
  import { chatStore } from "../stores/chat.svelte";
  import * as api from "../services/api";
  import type { UiModelInfo } from "../types";
  import ParamPopover from "./ParamPopover.svelte";
  import Brand from "./Brand.svelte";
  import { ICON_BUTTON } from "../ui";
  import PanelLeft from "@lucide/svelte/icons/panel-left";
  import ChevronDown from "@lucide/svelte/icons/chevron-down";
  import Check from "@lucide/svelte/icons/check";
  import Globe from "@lucide/svelte/icons/globe";
  import Code from "@lucide/svelte/icons/code";
  import Terminal from "@lucide/svelte/icons/terminal";
  import Sun from "@lucide/svelte/icons/sun";
  import Moon from "@lucide/svelte/icons/moon";
  import Monitor from "@lucide/svelte/icons/monitor";
  import Wrench from "@lucide/svelte/icons/wrench";
  import Settings from "@lucide/svelte/icons/settings";

  type ParamKey = "temp" | "top_p" | "top_k" | "rep" | "max";

  const CHIP =
    "inline-flex items-center gap-1 rounded-full border border-line-strong px-2 py-0.5 text-xs text-muted-strong";

  let dropdownOpen = $state(false);
  let dropdownEl = $state<HTMLDivElement | null>(null);
  let openParam = $state<ParamKey | null>(null);

  let modelDefaults = $derived(
    modelStore.models.find((m) => m.name === modelStore.selectedModel)?.generation_defaults,
  );

  let connection = $derived(`${window.location.hostname}:${window.location.port || (window.location.protocol === "https:" ? "443" : "80")}`);

  /** Compact in/out modality string for the dropdown row. */
  function modalityRow(model: UiModelInfo): string | null {
    const inp = model.input_modalities ?? [];
    const out = model.output_modalities ?? [];
    if (inp.length === 0 && out.length === 0) return null;
    return `${inp.join(" + ") || "none"} -> ${out.join(" + ") || "none"}`;
  }

  async function pickModel(name: string) {
    dropdownOpen = false;
    if (name === modelStore.selectedModel || chatStore.isStreaming) return;
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

<header class="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-background px-3">
  <button
    class={ICON_BUTTON}
    onclick={() => (settingsStore.sidebarOpen = !settingsStore.sidebarOpen)}
    aria-label="Toggle sidebar"
    title="Toggle sidebar"
  >
    <PanelLeft class="h-4 w-4" />
  </button>

  <Brand class="mr-2 hidden text-[15px] sm:inline" />

  <div class="relative min-w-0" bind:this={dropdownEl}>
    <button
      class="flex min-w-0 items-center gap-2 rounded-md border border-line-strong bg-surface px-2.5 py-1 text-sm hover:border-muted disabled:cursor-not-allowed disabled:opacity-60"
      onclick={() => (dropdownOpen = !dropdownOpen)}
      disabled={chatStore.isStreaming}
      title={chatStore.isStreaming ? "Can't switch models while a reply is streaming" : "Choose model"}
      aria-haspopup="listbox"
      aria-expanded={dropdownOpen}
    >
      {#if modelStore.selectedModel}
        <span class="truncate font-mono text-xs text-foreground">{modelStore.selectedModel}</span>
        <span class="hidden text-[11px] uppercase tracking-wide text-muted sm:inline">{modelStore.selectedModelKind}</span>
      {:else}
        <span class="text-xs text-muted">No model selected</span>
      {/if}
      <ChevronDown class="h-3.5 w-3.5 shrink-0 text-muted" />
    </button>

    {#if dropdownOpen}
      <div
        class="absolute left-0 top-full z-50 mt-1 max-h-96 min-w-full overflow-y-auto rounded-md border border-line-strong bg-surface p-1 shadow-lg shadow-black/10"
        role="listbox"
      >
        {#if modelStore.models.length === 0}
          <div class="whitespace-nowrap px-3 py-2 text-sm text-muted">No models loaded</div>
        {:else}
          {#each modelStore.models as model (model.name)}
            {@const detail = modalityRow(model)}
            {@const selected = model.name === modelStore.selectedModel}
            <button
              class="flex w-full items-start gap-3 whitespace-nowrap rounded px-2.5 py-2 text-left hover:bg-surface-raised {selected ? 'bg-surface-raised' : ''}"
              onclick={() => pickModel(model.name)}
              role="option"
              aria-selected={selected}
            >
              <div class="flex flex-1 flex-col gap-0.5">
                <span class="font-mono text-xs text-foreground">{model.name}</span>
                <span class="text-[11px] text-muted">{model.kind}{detail ? ` . ${detail}` : ""}</span>
              </div>
              {#if selected}
                <Check class="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" />
              {/if}
            </button>
          {/each}
        {/if}
      </div>
    {/if}
  </div>

  <div class="hidden items-center font-mono text-xs text-muted lg:flex">
    {#snippet param(key: ParamKey, label: string, value: number | null | undefined)}
      <div class="relative">
        <button
          class="rounded px-1.5 py-1 hover:bg-surface-raised {openParam === key ? 'bg-surface-raised' : ''}"
          onclick={() => (openParam = openParam === key ? null : key)}
          title="Edit {label}"
        >
          {label} <span class="text-foreground">{value ?? "-"}</span>
        </button>
        {#if openParam === key}
          {#if key === "temp"}
            <ParamPopover
              label="temperature"
              value={settingsStore.temperature}
              min={0} max={2} step={0.05}
              defaultValue={modelDefaults?.temperature ?? 0.7}
              onCommit={(v) => { if (v != null) { settingsStore.temperature = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "top_p"}
            <ParamPopover
              label="top_p"
              value={settingsStore.topP}
              min={0} max={1} step={0.01}
              defaultValue={modelDefaults?.top_p ?? 0.9}
              onCommit={(v) => { if (v != null) { settingsStore.topP = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "top_k"}
            <ParamPopover
              label="top_k"
              value={settingsStore.topK}
              min={0} step={1}
              defaultValue={modelDefaults?.top_k ?? 40}
              onCommit={(v) => { if (v != null) { settingsStore.topK = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "rep"}
            <ParamPopover
              label="repetition penalty"
              value={settingsStore.repetitionPenalty}
              min={0.5} max={2} step={0.01}
              defaultValue={modelDefaults?.repetition_penalty ?? 1.1}
              onCommit={(v) => { if (v != null) { settingsStore.repetitionPenalty = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "max"}
            <ParamPopover
              label="max tokens"
              value={settingsStore.maxTokens}
              min={1} step={1}
              defaultValue={modelDefaults?.max_tokens ?? 2048}
              onCommit={(v) => { if (v != null) { settingsStore.maxTokens = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {/if}
        {/if}
      </div>
    {/snippet}

    {@render param("temp", "temp", settingsStore.temperature)}
    {@render param("top_p", "top_p", settingsStore.topP)}
    {@render param("top_k", "top_k", settingsStore.topK)}
    {@render param("rep", "rep", settingsStore.repetitionPenalty)}
    {@render param("max", "max", settingsStore.maxTokens)}

    <button
      class="rounded px-1.5 py-1 hover:bg-surface-raised"
      onclick={() => (settingsStore.settingsOpen = true)}
      title="Set reasoning effort"
    >
      reasoning <span class={settingsStore.reasoningEffort === "off" ? "text-muted" : "text-accent"}>{settingsStore.reasoningEffort || "default"}</span>
    </button>
  </div>

  <div class="ml-auto flex items-center gap-1.5">
    {#if settingsStore.enableSearch && modelStore.capabilities.search_enabled}
      <span class={CHIP} title="Web search is on"><Globe class="h-3 w-3" />Search</span>
    {/if}
    {#if settingsStore.enableCodeExecution && modelStore.capabilities.code_execution_enabled}
      <span class={CHIP} title="Code execution is on"><Code class="h-3 w-3" />Code</span>
    {/if}
    {#if settingsStore.enableShell && modelStore.capabilities.shell_enabled}
      <span class={CHIP} title="Shell is on"><Terminal class="h-3 w-3" />Shell</span>
    {/if}
    <span class="hidden items-center gap-1.5 px-1 font-mono text-xs text-muted md:inline-flex" title="Server endpoint">
      <span class="h-1.5 w-1.5 rounded-full bg-success"></span>
      {connection}
    </span>
  </div>

  <button
    class={ICON_BUTTON}
    onclick={() => settingsStore.cycleTheme()}
    aria-label="Theme: {settingsStore.theme}"
    title="Theme: {settingsStore.theme}"
  >
    {#if settingsStore.theme === "light"}
      <Sun class="h-4 w-4" />
    {:else if settingsStore.theme === "dark"}
      <Moon class="h-4 w-4" />
    {:else}
      <Monitor class="h-4 w-4" />
    {/if}
  </button>

  <button
    class={ICON_BUTTON}
    onclick={() => (settingsStore.toolsOpen = !settingsStore.toolsOpen)}
    aria-label="Tools"
    title="Tools"
  >
    <Wrench class="h-4 w-4" />
  </button>

  <button
    class={ICON_BUTTON}
    onclick={() => (settingsStore.settingsOpen = !settingsStore.settingsOpen)}
    aria-label="Settings"
    title="Settings"
  >
    <Settings class="h-4 w-4" />
  </button>
</header>
