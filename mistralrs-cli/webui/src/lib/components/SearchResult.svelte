<script lang="ts">
  import type { WebSearchData } from "../types";

  let { data, phase }: { data: WebSearchData; phase: "calling" | "complete" } = $props();
</script>

<details class="group rounded-xl border border-green-200 dark:border-green-800/50">
  <summary class="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm select-none">
    <svg class="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
    </svg>
    <svg class="h-4 w-4 shrink-0 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
    <!-- Always show query -->
    <span class="text-sm text-gray-600 dark:text-gray-300">
      {#if data.query}
        "{data.query}"
      {:else}
        Web search
      {/if}
    </span>
    <!-- Status: spinner while calling, result count badge when complete -->
    {#if phase === "calling"}
      <span class="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-green-400"></span>
    {:else if data.results_count != null}
      <span class="ml-auto rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/40 dark:text-green-300">
        {data.results_count} result{data.results_count !== 1 ? "s" : ""}
      </span>
    {/if}
  </summary>
  <div class="border-t border-green-200 px-3 py-2 text-xs text-gray-500 dark:border-green-800/50 dark:text-gray-400">
    {#if data.query}
      <p><strong>Query:</strong> {data.query}</p>
    {/if}
    {#if data.results_count != null}
      <p><strong>Results:</strong> {data.results_count}</p>
    {/if}
  </div>
</details>
