<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import type { ChatSearchHit, ChatSummary } from "../types";
  import { highlightSegments } from "../utils/search";
  import { HOVER_REVEAL } from "../ui";
  import Plus from "@lucide/svelte/icons/plus";
  import Search from "@lucide/svelte/icons/search";
  import Pencil from "@lucide/svelte/icons/pencil";
  import Trash2 from "@lucide/svelte/icons/trash-2";
  import { onMount } from "svelte";

  let editingId = $state<string | null>(null);
  let editTitle = $state("");

  let visibleChats = $derived<(ChatSummary | ChatSearchHit)[]>(
    chatStore.searchResults ?? chatStore.chats,
  );

  function onSearchKey(e: KeyboardEvent) {
    if (e.key === "Escape") chatStore.setSearchQuery("");
  }

  onMount(() => {
    chatStore.refreshChats();
  });

  function startRename(id: string, currentTitle: string) {
    editingId = id;
    editTitle = currentTitle || "";
  }

  async function saveRename(id: string) {
    const title = editTitle.trim();
    editingId = null;
    if (title) await chatStore.renameChat(id, title);
  }

  const LABEL_CHARS = 40;

  function getChatLabel(chat: ChatSummary): string {
    if (chat.title) return chat.title;
    if (chat.preview) {
      return chat.preview.length > LABEL_CHARS ? `${chat.preview.slice(0, LABEL_CHARS)}...` : chat.preview;
    }
    return `Chat ${chat.id}`;
  }
</script>

<div class="flex h-full flex-col">
  <div class="space-y-2 p-3">
    <button
      onclick={() => chatStore.newChat()}
      class="flex w-full items-center gap-2 rounded-md border border-line-strong bg-background px-3 py-1.5 text-sm text-foreground hover:bg-surface-raised"
    >
      <Plus class="h-4 w-4 text-muted" />
      New chat
    </button>

    <div class="relative">
      <Search class="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
      <input
        type="search"
        placeholder="Search chats"
        aria-label="Search chats"
        class="w-full rounded-md border border-line bg-background py-1.5 pl-8 pr-2 text-sm text-foreground placeholder:text-muted focus:border-accent focus:outline-none"
        value={chatStore.searchQuery}
        oninput={(e) => chatStore.setSearchQuery(e.currentTarget.value)}
        onkeydown={onSearchKey}
      />
    </div>
  </div>

  <nav class="flex-1 overflow-y-auto px-2 pb-3" aria-label="Chats">
    {#if chatStore.searchResults?.length === 0}
      <p class="px-2 py-6 text-center text-xs text-muted">No chats match "{chatStore.searchQuery.trim()}"</p>
    {:else if !chatStore.searchResults && chatStore.chats.length === 0}
      <p class="px-2 py-6 text-center text-xs text-muted">No chats yet</p>
    {/if}
    <ul class="space-y-px">
      {#each visibleChats as chat (chat.id)}
        {@const isActive = chatStore.currentChatId === chat.id}
        <li
          class="group relative flex items-center rounded-md text-sm {isActive ? 'bg-surface-raised' : 'hover:bg-surface-raised/60'}"
        >
          {#if isActive}
            <span class="absolute top-2 bottom-2 left-0 w-0.5 rounded-full bg-accent" aria-hidden="true"></span>
          {/if}
          {#if editingId === chat.id}
            <input
              class="m-1 flex-1 rounded border border-accent bg-background px-2 py-1 text-sm text-foreground outline-none"
              bind:value={editTitle}
              aria-label="Chat title"
              onkeydown={(e: KeyboardEvent) => {
                if (e.key === "Enter") saveRename(chat.id);
                if (e.key === "Escape") editingId = null;
              }}
              onblur={() => saveRename(chat.id)}
            />
          {:else}
            <button
              class="flex min-w-0 flex-1 flex-col items-start gap-0.5 px-3 py-2 text-left"
              onclick={() => chatStore.loadChat(chat.id)}
              aria-current={isActive ? "page" : undefined}
            >
              <span class="w-full truncate {isActive ? 'text-foreground' : 'text-muted-strong'}">{getChatLabel(chat)}</span>
              {#if "snippet" in chat}
                <span class="line-clamp-2 w-full text-xs text-muted">
                  {#each highlightSegments(chat.snippet, chatStore.searchQuery) as seg, k (k)}
                    {#if seg.match}<mark class="rounded-sm bg-accent-soft px-0.5 text-foreground">{seg.text}</mark>{:else}{seg.text}{/if}
                  {/each}
                </span>
              {/if}
              <span class="w-full truncate font-mono text-[11px] text-muted">{chat.model}</span>
            </button>
            <div class="flex shrink-0 pr-1.5 {HOVER_REVEAL}">
              <button
                class="rounded p-1 text-muted hover:bg-background hover:text-foreground"
                onclick={() => startRename(chat.id, getChatLabel(chat))}
                title="Rename"
                aria-label="Rename chat"
              >
                <Pencil class="h-3.5 w-3.5" />
              </button>
              <button
                class="rounded p-1 text-muted hover:bg-background hover:text-danger"
                onclick={() => chatStore.deleteChat(chat.id)}
                title="Delete"
                aria-label="Delete chat"
              >
                <Trash2 class="h-3.5 w-3.5" />
              </button>
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  </nav>
</div>
