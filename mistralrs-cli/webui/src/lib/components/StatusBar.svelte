<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import { modelStore } from "../stores/models.svelte";

  let elapsedTick = $state(0);
  $effect(() => {
    if (!chatStore.isStreaming || chatStore.streamingStart == null) return;
    const t = setInterval(() => (elapsedTick = elapsedTick + 1), 100);
    return () => clearInterval(t);
  });

  let elapsedSec = $derived(() => {
    void elapsedTick;
    if (!chatStore.isStreaming || chatStore.streamingStart == null) return null;
    return ((performance.now() - chatStore.streamingStart) / 1000).toFixed(1);
  });
</script>

<footer class="flex h-7 shrink-0 items-center gap-4 border-t border-gray-200 bg-gray-50 px-4 font-mono text-[11px] text-gray-500 dark:border-gray-800 dark:bg-gray-900/50 dark:text-gray-400">
  <div class="flex items-center gap-4">
    {#if chatStore.isStreaming}
      <span class="flex items-center gap-1.5 text-green-600 dark:text-green-400">
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500"></span>
        streaming
      </span>
      <span><span class="text-gray-400 dark:text-gray-500">tok/s</span> <span class="text-gray-700 dark:text-gray-200">{chatStore.streamingTokRate.toFixed(1)}</span></span>
      <span><span class="text-gray-400 dark:text-gray-500">tok</span> <span class="text-gray-700 dark:text-gray-200">{chatStore.streamingTokens}</span></span>
      <span><span class="text-gray-400 dark:text-gray-500">elapsed</span> <span class="text-gray-700 dark:text-gray-200">{elapsedSec()}s</span></span>
    {:else}
      <span class="flex items-center gap-1.5">
        <span class="h-1.5 w-1.5 rounded-full bg-gray-400 dark:bg-gray-600"></span>
        idle
      </span>
    {/if}
  </div>

  <div class="ml-auto flex items-center gap-4">
    {#if modelStore.selectedModel}
      <span><span class="text-gray-400 dark:text-gray-500">model</span> <span class="text-gray-700 dark:text-gray-200">{modelStore.selectedModel}</span></span>
    {/if}
    {#if modelStore.models.length > 0}
      <span><span class="text-gray-400 dark:text-gray-500">loaded</span> <span class="text-gray-700 dark:text-gray-200">{modelStore.models.length}</span></span>
    {/if}
  </div>
</footer>
