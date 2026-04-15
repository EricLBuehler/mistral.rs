<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import UserMessage from "./UserMessage.svelte";
  import AssistantMessage from "./AssistantMessage.svelte";

  let container: HTMLDivElement;
  let shouldAutoScroll = $state(true);

  function handleScroll() {
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    // Auto-scroll if user is near the bottom (within 100px)
    shouldAutoScroll = scrollHeight - scrollTop - clientHeight < 100;
  }

  $effect(() => {
    // Trigger on any message or streaming content change
    void chatStore.messages.length;
    void chatStore.streamingContent;
    void chatStore.streamingBlocks.length;

    if (shouldAutoScroll && container) {
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  });
</script>

<div
  class="flex-1 overflow-y-auto"
  bind:this={container}
  onscroll={handleScroll}
>
  <div class="mx-auto max-w-3xl px-4 py-6">
    {#if chatStore.messages.length === 0 && !chatStore.isStreaming}
      <!-- Empty state -->
      <div class="flex h-full flex-col items-center justify-center pt-20 text-center">
        <div class="mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 p-4 text-white shadow-lg">
          <svg class="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        </div>
        <h2 class="text-lg font-semibold text-gray-700 dark:text-gray-300">Start a conversation</h2>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">Type a message below to begin</p>
      </div>
    {:else}
      <div class="space-y-4">
        {#each chatStore.messages as message, i (i)}
          {#if message.role === "user"}
            <UserMessage {message} />
          {:else if message.role === "assistant"}
            <AssistantMessage {message} />
          {/if}
        {/each}

        <!-- Streaming message -->
        {#if chatStore.isStreaming}
          <AssistantMessage
            message={{
              role: "assistant",
              content: chatStore.streamingContent,
              blocks: chatStore.streamingBlocks.length
                ? chatStore.streamingBlocks
                : undefined,
            }}
            streaming={true}
          />
        {/if}
      </div>
    {/if}
  </div>
</div>
