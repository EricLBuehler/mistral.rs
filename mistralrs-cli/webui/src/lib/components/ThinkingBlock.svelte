<script lang="ts">
  import ChevronRight from "@lucide/svelte/icons/chevron-right";

  let { reasoning, streaming = false }: { reasoning: string; streaming?: boolean } = $props();

  const CHARS_PER_TOKEN_ESTIMATE = 4;

  let open = $state(false);
  let tail = $state<HTMLDivElement | null>(null);

  let tokenEstimate = $derived(Math.ceil(reasoning.length / CHARS_PER_TOKEN_ESTIMATE));

  function toggle() {
    open = !open;
  }

  $effect(() => {
    void reasoning;
    if (streaming && tail) tail.scrollTop = tail.scrollHeight;
  });
</script>

<div class="text-sm">
  <button
    class="flex items-center gap-1.5 rounded py-0.5 text-muted hover:text-foreground"
    onclick={toggle}
    aria-expanded={open}
  >
    <ChevronRight class="h-3.5 w-3.5 transition-transform {open ? 'rotate-90' : ''}" />
    {#if streaming}
      <span class="text-muted-strong">Thinking</span>
      <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent"></span>
    {:else}
      <span>Thought</span>
      <span class="font-mono text-xs tabular-nums">~{tokenEstimate} tok</span>
    {/if}
  </button>
  {#if open}
    <div
      bind:this={tail}
      class="mt-1.5 ml-[7px] overflow-y-auto border-l border-line-strong pl-4 max-h-[28rem]"
    >
      <p class="whitespace-pre-wrap leading-relaxed text-muted">{reasoning}</p>
    </div>
  {/if}
</div>
