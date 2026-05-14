<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import * as api from "../services/api";
  import type { ChatFile } from "../types";
  import { onMount } from "svelte";

  let chats = $state<ChatFile[]>([]);
  let editingId = $state<string | null>(null);
  let editTitle = $state("");

  async function refreshChats() {
    try {
      const result = await api.listChats();
      chats = result.chats.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
    } catch {
      // ignore
    }
  }

  onMount(() => {
    refreshChats();
  });

  async function handleNewChat() {
    await chatStore.newChat();
    await refreshChats();
  }

  async function handleLoadChat(id: string) {
    await chatStore.loadChat(id);
    await refreshChats();
  }

  async function handleDeleteChat(id: string) {
    await chatStore.deleteChat(id);
    await refreshChats();
  }

  function startRename(id: string, currentTitle: string) {
    editingId = id;
    editTitle = currentTitle || "";
  }

  async function saveRename(id: string) {
    if (editTitle.trim()) {
      await api.renameChat(id, editTitle.trim());
      await refreshChats();
    }
    editingId = null;
  }

  function getChatLabel(chat: ChatFile): string {
    if (chat.title) return chat.title;
    const firstMsg = chat.messages.find((m) => m.role === "user");
    if (firstMsg) {
      return firstMsg.content.slice(0, 40) + (firstMsg.content.length > 40 ? "..." : "");
    }
    return `Chat ${chat.id}`;
  }

  function turnCount(chat: ChatFile): number {
    return chat.messages.filter((m) => m.role === "user").length;
  }
</script>

<div class="flex h-full flex-col">
  <!-- New chat button -->
  <div class="p-3">
    <button
      onclick={handleNewChat}
      class="flex w-full items-center justify-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
    >
      <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
      </svg>
      New Chat
    </button>
  </div>

  <div class="border-b border-gray-200 dark:border-gray-700"></div>

  <!-- Chat list -->
  <div class="flex-1 overflow-y-auto p-2">
    <div class="space-y-0.5">
      {#each chats as chat}
        {@const isActive = chatStore.currentChatId === chat.id}
        {@const turns = turnCount(chat)}
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
                if (e.key === "Enter") saveRename(chat.id!);
                if (e.key === "Escape") editingId = null;
              }}
              onblur={() => saveRename(chat.id!)}
            />
          {:else}
            <button
              class="flex min-w-0 flex-1 flex-col items-start text-left"
              onclick={() => handleLoadChat(chat.id!)}
            >
              <span class="w-full truncate">{getChatLabel(chat)}</span>
              <span class="w-full truncate font-mono text-[10.5px] {isActive ? 'text-blue-600 dark:text-blue-300' : 'text-gray-400 dark:text-gray-500'}">
                {chat.model}{turns > 0 ? ` · ${turns} turn${turns === 1 ? '' : 's'}` : ''}
              </span>
            </button>
            <div class="flex shrink-0 opacity-0 group-hover:opacity-100">
              <button
                class="rounded p-0.5 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
                onclick={() => startRename(chat.id!, getChatLabel(chat))}
                title="Rename"
              >
                <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
              <button
                class="rounded p-0.5 text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400"
                onclick={() => handleDeleteChat(chat.id!)}
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
