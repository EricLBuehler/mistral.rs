<script lang="ts">
  import type { ShellData } from "../types";
  import { TOOL_CARD } from "../ui";
  import Terminal from "@lucide/svelte/icons/terminal";
  import Copy from "@lucide/svelte/icons/copy";
  import Check from "@lucide/svelte/icons/check";
  import CircleX from "@lucide/svelte/icons/circle-x";

  const SECTION_LABEL = "flex items-center gap-1.5 border-t border-line px-3 py-1 font-mono text-[11px] text-muted";

  let { data, phase }: { data: ShellData; phase: "calling" | "complete" } = $props();

  let copyState = $state<"idle" | "copied">("idle");
  let commandText = $derived(data.commands.join("\n"));
  let commandsForDisplay = $derived(data.commands.map((command) => `$ ${command}`).join("\n"));
  let failed = $derived(
    phase === "complete" &&
      (data.timed_out === true ||
        data.exit_code != null && data.exit_code !== 0 ||
        data.status === "failed")
  );
  let statusLabel = $derived(() => {
    if (phase === "calling") return "Running...";
    if (data.timed_out) return "Timed out";
    if (data.status) return data.status;
    if (data.exit_code != null) return data.exit_code === 0 ? "Completed" : "Failed";
    return "Completed";
  });

  async function copyCommands(e: Event) {
    e.stopPropagation();
    if (!commandText) return;
    try {
      await navigator.clipboard.writeText(commandText);
      copyState = "copied";
      setTimeout(() => (copyState = "idle"), 1200);
    } catch {
      // ignore
    }
  }
</script>

<div class={TOOL_CARD}>
  <div class="flex items-center gap-2 px-3 py-1.5">
    <Terminal class="h-3.5 w-3.5 text-muted" />
    <span class="text-xs font-medium text-muted-strong">Shell</span>
    <div class="ml-auto flex items-center gap-1.5 text-xs {failed ? 'text-danger' : 'text-muted'}">
      {#if phase === "calling"}
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent"></span>
      {:else if failed}
        <CircleX class="h-3.5 w-3.5" />
      {:else}
        <Check class="h-3.5 w-3.5 text-success" />
      {/if}
      <span>{statusLabel()}</span>
      {#if data.exit_code != null}
        <span class="font-mono text-[11px]">exit {data.exit_code}</span>
      {/if}
    </div>
  </div>

  {#if data.commands.length}
    <div class="border-t border-line">
      <div class="flex items-center">
        <div class="flex flex-1 items-center gap-1.5 px-3 py-1 text-muted">
          <span class="text-xs">Commands</span>
          <span class="font-mono text-[11px]">({data.commands.length})</span>
        </div>
        <button
          class="flex items-center gap-1 self-stretch border-l border-line px-3 py-1 text-xs text-muted-strong hover:bg-surface-raised hover:text-foreground"
          onclick={copyCommands}
          title="Copy commands"
          aria-label="Copy commands"
        >
          {#if copyState === "copied"}
            <Check class="h-3.5 w-3.5 text-success" />
            <span>Copied</span>
          {:else}
            <Copy class="h-3.5 w-3.5" />
            <span>Copy</span>
          {/if}
        </button>
      </div>
      <div class="border-t border-line bg-background px-3 py-2">
        <pre class="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-code">{commandsForDisplay}</pre>
      </div>
    </div>
  {/if}

  {#if data.working_directory || data.status || data.timed_out != null}
    <div class="flex flex-wrap gap-x-3 gap-y-1 border-t border-line px-3 py-1.5 font-mono text-[11px] text-muted">
      {#if data.working_directory}
        <span class="truncate">cwd {data.working_directory}</span>
      {/if}
      {#if data.status}
        <span>status {data.status}</span>
      {/if}
      {#if data.timed_out != null}
        <span>timeout {data.timed_out ? "yes" : "no"}</span>
      {/if}
    </div>
  {/if}

  {#if data.stdout}
    <div class={SECTION_LABEL}>stdout</div>
    <div class="bg-background px-3 py-2">
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-code">{data.stdout}</pre>
    </div>
  {/if}

  {#if data.stderr}
    <div class={SECTION_LABEL}>stderr</div>
    <div class="bg-background px-3 py-2">
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-danger">{data.stderr}</pre>
    </div>
  {/if}
</div>
