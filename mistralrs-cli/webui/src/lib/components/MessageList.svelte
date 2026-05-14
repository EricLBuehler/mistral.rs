<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import UserMessage from "./UserMessage.svelte";
  import AssistantMessage from "./AssistantMessage.svelte";

  let container: HTMLDivElement;
  let userScrolledAway = $state(false);
  let lastScrollTop = 0;

  function handleScroll() {
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // If user scrolled UP at all while not near the bottom, detach
    if (scrollTop < lastScrollTop && distanceFromBottom > 80) {
      userScrolledAway = true;
    }

    // If user scrolled back near the bottom, re-attach
    if (distanceFromBottom < 30) {
      userScrolledAway = false;
    }

    lastScrollTop = scrollTop;
  }

  $effect(() => {
    // Trigger on content changes
    void chatStore.messages.length;
    void chatStore.streamingBlocks;
    void chatStore.streamingBlocks.length;

    if (!userScrolledAway && container) {
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  });

  // When a new message is sent, always scroll to bottom
  $effect(() => {
    if (chatStore.isStreaming) {
      userScrolledAway = false;
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
        <h2 class="text-lg font-semibold text-gray-700 dark:text-gray-300">Start a conversation</h2>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">Type a message below to begin</p>
      </div>
    {:else}
      <div class="space-y-4">
        {#each chatStore.messages as message (message.id)}
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
              id: "__streaming__",
              parentId: null,
              role: "assistant",
              content: "",
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

  <!-- Scroll-to-bottom button when detached -->
  {#if userScrolledAway && chatStore.isStreaming}
    <button
      class="fixed bottom-28 left-1/2 z-20 -translate-x-1/2 rounded-full bg-gray-800 px-3 py-1.5 text-xs text-white shadow-lg transition-colors hover:bg-gray-700 dark:bg-gray-600 dark:hover:bg-gray-500"
      onclick={() => {
        userScrolledAway = false;
        if (container) container.scrollTop = container.scrollHeight;
      }}
    >
      <svg class="mr-1 inline h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
      </svg>
      Scroll to bottom
    </button>
  {/if}
</div>
