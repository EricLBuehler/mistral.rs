<script lang="ts">
  import type { WebSearchData } from "../types";
  import { TOOL_CARD } from "../ui";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import Globe from "@lucide/svelte/icons/globe";

  let { data, phase }: { data: WebSearchData; phase: "calling" | "complete" } = $props();
</script>

<details class="group {TOOL_CARD}">
  <summary class="flex cursor-pointer select-none items-center gap-2 px-3 py-1.5 text-xs text-muted hover:text-foreground">
    <ChevronRight class="h-3.5 w-3.5 shrink-0 transition-transform group-open:rotate-90" />
    <Globe class="h-3.5 w-3.5 shrink-0" />
    <span class="min-w-0 truncate text-muted-strong">
      {#if data.query}
        "{data.query}"
      {:else}
        Web search
      {/if}
    </span>
    {#if phase === "calling"}
      <span class="ml-auto h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent"></span>
    {:else if data.results_count != null}
      <span class="ml-auto shrink-0 font-mono text-[11px] text-muted">
        {data.results_count} result{data.results_count !== 1 ? "s" : ""}
      </span>
    {/if}
  </summary>
  <div class="space-y-0.5 border-t border-line px-3 py-2 text-xs text-muted">
    {#if data.query}
      <p><strong class="font-medium text-muted-strong">Query:</strong> {data.query}</p>
    {/if}
    {#if data.results_count != null}
      <p><strong class="font-medium text-muted-strong">Results:</strong> {data.results_count}</p>
    {/if}
  </div>
</details>
