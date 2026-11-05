<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
</script>

{#if chatStore.saveError}
  <div class="flex items-center gap-3 border-b border-line bg-warning-soft px-4 py-2 text-sm text-warning" role="alert">
    <span class="min-w-0 flex-1 truncate" title={chatStore.saveError}>{chatStore.saveError}. Changes since then are not saved yet.</span>
    <button
      class="shrink-0 rounded-md border border-current/30 px-2 py-0.5 text-xs font-medium hover:bg-current/10"
      onclick={() => chatStore.retrySaves()}
    >Retry</button>
  </div>
{/if}

{#if chatStore.lastError}
  <div class="flex items-center gap-3 border-b border-line bg-danger-soft px-4 py-2 text-sm text-danger" role="alert">
    <span class="min-w-0 flex-1 truncate" title={chatStore.lastError}>{chatStore.lastError}</span>
    <button
      class="shrink-0 rounded-md px-2 py-0.5 text-xs font-medium hover:bg-current/10"
      onclick={() => (chatStore.lastError = null)}
      aria-label="Dismiss"
    >Dismiss</button>
  </div>
{/if}
