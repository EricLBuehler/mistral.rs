<script lang="ts">
  let { reasoning, streaming = false }: { reasoning: string; streaming?: boolean } = $props();

  // Estimate token count (rough: ~4 chars per token)
  let tokenEstimate = $derived(Math.ceil(reasoning.length / 4));
</script>

<details class="group rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900/50">
  <summary class="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm text-gray-500 select-none hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
    <svg class="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
    </svg>
    <svg class="h-4 w-4 shrink-0 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
    {#if streaming}
      <span>Thinking...</span>
      <span class="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400"></span>
    {:else}
      <span>Thinking</span>
      <span class="ml-auto text-xs text-gray-400 dark:text-gray-500">~{tokenEstimate} tokens</span>
    {/if}
  </summary>
  <div class="border-t border-gray-200 px-3 py-2 dark:border-gray-700">
    <pre class="whitespace-pre-wrap text-xs leading-relaxed text-gray-600 dark:text-gray-400">{reasoning}</pre>
  </div>
</details>
