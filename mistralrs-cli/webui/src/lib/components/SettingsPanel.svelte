<script lang="ts">
  import { settingsStore } from "../stores/settings.svelte";
  import { modelStore } from "../stores/models.svelte";
  import type { AgentPermission } from "../types";
  import { FIELD, ICON_BUTTON, PANEL_HEADING } from "../ui";
  import X from "@lucide/svelte/icons/x";
  import Globe from "@lucide/svelte/icons/globe";
  import Code from "@lucide/svelte/icons/code";
  import Terminal from "@lucide/svelte/icons/terminal";
  import Lightbulb from "@lucide/svelte/icons/lightbulb";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";

  const ROW_LABEL = "text-sm text-foreground";
  const FLAG_HINT = "font-mono text-[11px] text-muted";
  const CHECKBOX = "h-4 w-4 accent-[var(--accent)]";
  const SLIDER = "w-full accent-[var(--accent)]";
  const PARAM_LABEL = "text-xs text-muted-strong";
  const PARAM_VALUE = "font-mono text-xs text-muted";

  let samplingOpen = $state(false);
  const approvalModes: AgentPermission[] = ["auto", "ask", "deny"];

  function handleChange() {
    settingsStore.persist();
  }
</script>

<div class="flex flex-col gap-4 p-4">
  <div class="flex items-center justify-between">
    <h2 class="text-sm font-semibold text-foreground">Settings</h2>
    <button
      class={ICON_BUTTON}
      onclick={() => (settingsStore.settingsOpen = false)}
      aria-label="Close settings"
    >
      <X class="h-4 w-4" />
    </button>
  </div>

  <div class="space-y-3">
    <h3 class={PANEL_HEADING}>Tools</h3>

    <label class="flex items-center justify-between {modelStore.capabilities.search_enabled ? '' : 'opacity-50'}">
      <div class="flex items-center gap-2">
        <Globe class="h-4 w-4 text-muted" />
        <span class={ROW_LABEL}>Web Search</span>
        {#if !modelStore.capabilities.search_enabled}
          <span class={FLAG_HINT}>(--enable-search)</span>
        {/if}
      </div>
      <input
        type="checkbox"
        class={CHECKBOX}
        bind:checked={settingsStore.enableSearch}
        onchange={handleChange}
        disabled={!modelStore.capabilities.search_enabled}
      />
    </label>

    <label class="flex items-center justify-between {modelStore.capabilities.code_execution_enabled ? '' : 'opacity-50'}">
      <div class="flex items-center gap-2">
        <Code class="h-4 w-4 text-muted" />
        <span class={ROW_LABEL}>Code Execution</span>
        {#if !modelStore.capabilities.code_execution_enabled}
          <span class={FLAG_HINT}>(--enable-code-execution)</span>
        {/if}
      </div>
      <input
        type="checkbox"
        class={CHECKBOX}
        bind:checked={settingsStore.enableCodeExecution}
        onchange={handleChange}
        disabled={!modelStore.capabilities.code_execution_enabled}
      />
    </label>

    <label class="flex items-center justify-between {modelStore.capabilities.shell_enabled ? '' : 'opacity-50'}">
      <div class="flex items-center gap-2">
        <Terminal class="h-4 w-4 text-muted" />
        <span class={ROW_LABEL}>Shell</span>
        {#if !modelStore.capabilities.shell_enabled}
          <span class={FLAG_HINT}>(--enable-shell)</span>
        {/if}
      </div>
      <input
        type="checkbox"
        class={CHECKBOX}
        bind:checked={settingsStore.enableShell}
        onchange={handleChange}
        disabled={!modelStore.capabilities.shell_enabled}
      />
    </label>

    <label class="flex items-center justify-between">
      <div class="flex items-center gap-2">
        <Lightbulb class="h-4 w-4 text-muted" />
        <span class={ROW_LABEL}>Reasoning effort</span>
      </div>
      <select
        class="{FIELD} px-2 py-1 text-xs"
        bind:value={settingsStore.reasoningEffort}
        onchange={handleChange}
      >
        <option value="">Default (on)</option>
        <option value="off">Off</option>
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="xhigh">XHigh</option>
      </select>
    </label>

    {#if modelStore.capabilities.tool_dispatch_url}
      <div>
        <span class="mb-1 block text-xs text-muted">Tool Dispatch URL</span>
        <div class="break-all rounded-md border border-line bg-background px-2 py-1.5 font-mono text-xs text-muted-strong">
          {modelStore.capabilities.tool_dispatch_url}
        </div>
      </div>
    {/if}

    <div class="space-y-1.5">
      <div class="flex items-center justify-between">
        <span class={ROW_LABEL}>Tool approval</span>
        <span class={FLAG_HINT}>agent_permission</span>
      </div>
      <div class="grid grid-cols-3 gap-0.5 rounded-md border border-line-strong bg-background p-0.5">
        {#each approvalModes as mode}
          <button
            type="button"
            class="rounded px-2 py-1 text-xs font-medium transition-colors {settingsStore.agentPermission === mode ? 'bg-surface-raised text-foreground' : 'text-muted hover:text-foreground'}"
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

  <div class="border-t border-line"></div>

  <div>
    <h3 class="mb-2 {PANEL_HEADING}">System Prompt</h3>
    <textarea
      class="{FIELD} w-full resize-none px-3 py-2 text-sm"
      rows="3"
      placeholder="Enter system prompt..."
      bind:value={settingsStore.systemPrompt}
      onchange={handleChange}
    ></textarea>
  </div>

  <div class="border-t border-line"></div>

  <details bind:open={samplingOpen} class="group">
    <summary class="flex cursor-pointer select-none items-center gap-1.5 {PANEL_HEADING} hover:text-foreground">
      <ChevronRight class="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
      Sampling Parameters
    </summary>

    <div class="mt-3 space-y-4">
      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class={PARAM_LABEL}>Temperature</span>
          <span class={PARAM_VALUE}>{settingsStore.temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          bind:value={settingsStore.temperature}
          onchange={handleChange}
          class={SLIDER}
        />
      </div>

      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class={PARAM_LABEL}>Top P</span>
          <span class={PARAM_VALUE}>{settingsStore.topP.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          bind:value={settingsStore.topP}
          onchange={handleChange}
          class={SLIDER}
        />
      </div>

      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class={PARAM_LABEL}>Top K</span>
          <span class={PARAM_VALUE}>{settingsStore.topK}</span>
        </div>
        <input
          type="range"
          min="0"
          max="200"
          step="1"
          bind:value={settingsStore.topK}
          onchange={handleChange}
          class={SLIDER}
        />
      </div>

      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class={PARAM_LABEL}>Max Tokens</span>
          <input
            type="number"
            min="1"
            max="131072"
            bind:value={settingsStore.maxTokens}
            onchange={handleChange}
            class="{FIELD} w-20 px-2 py-0.5 text-right font-mono text-xs"
          />
        </div>
      </div>

      <div>
        <div class="mb-1 flex items-center justify-between">
          <span class={PARAM_LABEL}>Repetition Penalty</span>
          <span class={PARAM_VALUE}>{settingsStore.repetitionPenalty.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="1"
          max="2"
          step="0.05"
          bind:value={settingsStore.repetitionPenalty}
          onchange={handleChange}
          class={SLIDER}
        />
      </div>
    </div>
  </details>
</div>
