<script lang="ts">
  import type { CustomToolData } from "../types";

  let { data, phase, toolName }: { data: CustomToolData; phase: "calling" | "complete"; toolName: string } = $props();
</script>

<details class="group rounded-xl border border-amber-200 dark:border-amber-800/50">
  <summary class="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm select-none">
    <svg class="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
    </svg>
    <svg class="h-4 w-4 shrink-0 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
    </svg>
    <span class="text-sm text-gray-600 dark:text-gray-300">
      {toolName}
    </span>
    {#if phase === "calling"}
      <span class="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400"></span>
    {:else}
      <span class="ml-auto text-xs text-gray-400">done</span>
    {/if}
  </summary>
  <div class="space-y-2 border-t border-amber-200 px-3 py-2 dark:border-amber-800/50">
    {#if data.arguments}
      <div>
        <div class="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">Arguments</div>
        <pre class="whitespace-pre-wrap rounded-lg bg-gray-50 p-2 font-mono text-xs text-gray-700 dark:bg-gray-900 dark:text-gray-300">{data.arguments}</pre>
      </div>
    {/if}
    {#if data.content}
      <div>
        <div class="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">Result</div>
        <pre class="whitespace-pre-wrap rounded-lg bg-gray-50 p-2 font-mono text-xs text-gray-700 dark:bg-gray-900 dark:text-gray-300">{data.content}</pre>
      </div>
    {/if}
  </div>
</details>
