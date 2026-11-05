<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import UserMessage from "./UserMessage.svelte";
  import AssistantMessage from "./AssistantMessage.svelte";
  import ArrowDown from "@lucide/svelte/icons/arrow-down";

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

<div class="relative min-h-0 flex-1">
  <div class="h-full overflow-y-auto" bind:this={container} onscroll={handleScroll}>
    <div class="mx-auto max-w-3xl space-y-8 px-4 pt-8 pb-10">
      {#each chatStore.messages as message (message.id)}
        {#if message.role === "user"}
          <UserMessage {message} />
        {:else if message.role === "assistant"}
          <AssistantMessage {message} />
        {/if}
      {/each}

      {#if chatStore.isStreaming}
        <AssistantMessage
          message={{
            id: "__streaming__",
            parentId: null,
            role: "assistant",
            content: "",
            blocks: chatStore.streamingBlocks.length ? chatStore.streamingBlocks : undefined,
          }}
          streaming={true}
        />
      {/if}
    </div>
  </div>

  {#if userScrolledAway}
    <button
      class="absolute bottom-3 left-1/2 z-20 inline-flex h-8 w-8 -translate-x-1/2 items-center justify-center rounded-full border border-line-strong bg-surface text-muted-strong shadow-md shadow-black/10 hover:text-foreground"
      onclick={() => {
        userScrolledAway = false;
        if (container) container.scrollTop = container.scrollHeight;
      }}
      aria-label="Scroll to bottom"
      title="Scroll to bottom"
    >
      <ArrowDown class="h-4 w-4" />
    </button>
  {/if}
</div>
