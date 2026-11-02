<script lang="ts">
  import type { CustomToolData } from "../types";
  import { TOOL_CARD } from "../ui";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import Wrench from "@lucide/svelte/icons/wrench";
  import Check from "@lucide/svelte/icons/check";

  let { data, phase, toolName }: { data: CustomToolData; phase: "calling" | "complete"; toolName: string } = $props();

  const BLOCK_LABEL = "mb-1 font-mono text-[11px] text-muted";
  const BLOCK = "whitespace-pre-wrap rounded-md border border-line bg-background p-2 font-mono text-xs text-code";
</script>

<details class="group {TOOL_CARD}">
  <summary class="flex cursor-pointer select-none items-center gap-2 px-3 py-1.5 text-xs text-muted hover:text-foreground">
    <ChevronRight class="h-3.5 w-3.5 shrink-0 transition-transform group-open:rotate-90" />
    <Wrench class="h-3.5 w-3.5 shrink-0" />
    <span class="min-w-0 truncate font-mono text-muted-strong">
      {toolName}
    </span>
    {#if phase === "calling"}
      <span class="ml-auto h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent"></span>
    {:else}
      <span class="ml-auto flex shrink-0 items-center gap-1 text-muted">
        <Check class="h-3.5 w-3.5 text-success" />
        done
      </span>
    {/if}
  </summary>
  <div class="space-y-2 border-t border-line px-3 py-2">
    {#if data.arguments}
      <div>
        <div class={BLOCK_LABEL}>Arguments</div>
        <pre class={BLOCK}>{data.arguments}</pre>
      </div>
    {/if}
    {#if data.content}
      <div>
        <div class={BLOCK_LABEL}>Result</div>
        <pre class={BLOCK}>{data.content}</pre>
      </div>
    {/if}
  </div>
</details>
