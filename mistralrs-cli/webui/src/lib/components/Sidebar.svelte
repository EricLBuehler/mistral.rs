<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import type { ChatSearchHit, ChatSummary } from "../types";
  import { highlightSegments } from "../utils/search";
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
  <!-- New chat button -->
  <div class="p-3">
    <button
      onclick={() => chatStore.newChat()}
      class="flex w-full items-center justify-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
    >
      <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
      </svg>
      New Chat
    </button>
  </div>

  <div class="px-3 pb-3">
    <div class="relative">
      <svg class="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M17 11a6 6 0 11-12 0 6 6 0 0112 0z" />
      </svg>
      <input
        type="search"
        placeholder="Search chats"
        aria-label="Search chats"
        class="w-full rounded-lg border border-gray-300 bg-white py-1.5 pl-8 pr-2 text-sm text-gray-800 placeholder-gray-400 focus:border-blue-400 focus:outline-none dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:placeholder-gray-500"
        value={chatStore.searchQuery}
        oninput={(e) => chatStore.setSearchQuery(e.currentTarget.value)}
        onkeydown={onSearchKey}
      />
    </div>
  </div>

  <div class="border-b border-gray-200 dark:border-gray-700"></div>

  <!-- Chat list -->
  <div class="flex-1 overflow-y-auto p-2">
    <div class="space-y-0.5">
      {#if chatStore.searchResults?.length === 0}
        <div class="px-2 py-6 text-center text-xs text-gray-400 dark:text-gray-500">No chats match "{chatStore.searchQuery.trim()}"</div>
      {/if}
      {#each visibleChats as chat (chat.id)}
        {@const isActive = chatStore.currentChatId === chat.id}
        <div
          class="group flex items-center rounded-lg px-2 py-1.5 text-sm transition-colors
          {isActive
            ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200'
            : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'}"
        >
          {#if editingId === chat.id}
            <input
              class="flex-1 rounded border border-blue-300 bg-white px-1.5 py-0.5 text-sm dark:border-blue-600 dark:bg-gray-800"
              bind:value={editTitle}
              onkeydown={(e: KeyboardEvent) => {
                if (e.key === "Enter") saveRename(chat.id);
                if (e.key === "Escape") editingId = null;
              }}
              onblur={() => saveRename(chat.id)}
            />
          {:else}
            <button
              class="flex min-w-0 flex-1 flex-col items-start text-left"
              onclick={() => chatStore.loadChat(chat.id)}
            >
              <span class="w-full truncate">{getChatLabel(chat)}</span>
              {#if "snippet" in chat}
                <span class="line-clamp-2 w-full text-xs text-gray-500 dark:text-gray-400">
                  {#each highlightSegments(chat.snippet, chatStore.searchQuery) as seg, k (k)}
                    {#if seg.match}<mark class="rounded-sm bg-yellow-200 px-0.5 text-gray-900 dark:bg-yellow-500/40 dark:text-gray-100">{seg.text}</mark>{:else}{seg.text}{/if}
                  {/each}
                </span>
              {/if}
              <span class="w-full truncate font-mono text-[10.5px] {isActive ? 'text-blue-600 dark:text-blue-300' : 'text-gray-400 dark:text-gray-500'}">
                {chat.model}
              </span>
            </button>
            <div class="flex shrink-0 opacity-0 group-hover:opacity-100">
              <button
                class="rounded p-0.5 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
                onclick={() => startRename(chat.id, getChatLabel(chat))}
                title="Rename"
              >
                <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
              <button
                class="rounded p-0.5 text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400"
                onclick={() => chatStore.deleteChat(chat.id)}
                title="Delete"
              >
                <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          {/if}
        </div>
      {/each}
    </div>
  </div>
</div>
