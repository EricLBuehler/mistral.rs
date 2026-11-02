<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
</script>

{#if chatStore.saveError}
  <div class="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200" role="alert">
    <span class="min-w-0 flex-1 truncate" title={chatStore.saveError}>{chatStore.saveError}. Changes since then are not saved yet.</span>
    <button
      class="shrink-0 rounded-md border border-amber-300 px-2 py-0.5 text-xs font-medium hover:bg-amber-100 dark:border-amber-800 dark:hover:bg-amber-900/40"
      onclick={() => chatStore.retrySaves()}
    >Retry</button>
  </div>
{/if}

{#if chatStore.lastError}
  <div class="flex items-center gap-3 border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300" role="alert">
    <span class="min-w-0 flex-1 truncate" title={chatStore.lastError}>{chatStore.lastError}</span>
    <button
      class="shrink-0 rounded-md px-2 py-0.5 text-xs font-medium hover:bg-red-100 dark:hover:bg-red-900/40"
      onclick={() => (chatStore.lastError = null)}
      aria-label="Dismiss"
    >Dismiss</button>
  </div>
{/if}
