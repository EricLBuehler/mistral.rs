<script lang="ts">
  import type { ShellData } from "../types";

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

<div class="overflow-hidden rounded-xl border border-teal-200 dark:border-teal-800/50">
  <div class="flex items-center gap-2 bg-teal-50 px-3 py-1.5 dark:bg-teal-900/20">
    <svg class="h-4 w-4 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 17l6-6-6-6M12 19h8" />
    </svg>
    <span class="text-xs font-medium text-teal-700 dark:text-teal-300">Shell</span>
    <div class="ml-auto flex items-center gap-2 text-xs {failed ? 'text-red-500 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'}">
      {#if phase === "calling"}
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-teal-400"></span>
      {/if}
      <span>{statusLabel()}</span>
      {#if data.exit_code != null}
        <span>exit {data.exit_code}</span>
      {/if}
    </div>
  </div>

  {#if data.commands.length}
    <div class="border-t border-gray-700">
      <div class="flex items-center bg-gray-900">
        <div class="flex flex-1 items-center gap-1.5 px-3 py-1">
          <span class="text-xs font-medium text-gray-400">Commands</span>
          <span class="text-xs text-gray-500">({data.commands.length})</span>
        </div>
        <button
          class="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-gray-200"
          onclick={copyCommands}
          title="Copy commands"
          aria-label="Copy commands"
        >
          {#if copyState === "copied"}
            <svg class="h-3.5 w-3.5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            <span>Copied</span>
          {:else}
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Copy</span>
          {/if}
        </button>
      </div>
      <div class="bg-gray-950 px-3 py-2">
        <pre class="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-200">{commandsForDisplay}</pre>
      </div>
    </div>
  {/if}

  {#if data.working_directory || data.status || data.timed_out != null}
    <div class="flex flex-wrap gap-x-3 gap-y-1 border-t border-gray-200 bg-gray-50 px-3 py-1.5 font-mono text-[11px] text-gray-500 dark:border-gray-700 dark:bg-gray-900/60 dark:text-gray-400">
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
    <div class="border-t border-gray-200 dark:border-gray-700">
      <div class="flex items-center gap-1.5 bg-gray-100 px-3 py-1 dark:bg-gray-800/80">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400">stdout</span>
      </div>
      <div class="bg-gray-50 px-3 py-2 dark:bg-gray-900">
        <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-700 dark:text-gray-300">{data.stdout}</pre>
      </div>
    </div>
  {/if}

  {#if data.stderr}
    <div class="border-t border-red-200 dark:border-red-800/50">
      <div class="flex items-center gap-1.5 bg-red-50 px-3 py-1 dark:bg-red-900/30">
        <span class="text-xs font-medium text-red-600 dark:text-red-400">stderr</span>
      </div>
      <div class="bg-red-50/50 px-3 py-2 dark:bg-red-900/10">
        <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-red-700 dark:text-red-300">{data.stderr}</pre>
      </div>
    </div>
  {/if}
</div>
