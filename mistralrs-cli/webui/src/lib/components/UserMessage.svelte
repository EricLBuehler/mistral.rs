<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";
  import { chatStore } from "../stores/chat.svelte";

  let { message }: { message: DisplayMessage } = $props();
  let copied = $state(false);
  let editing = $state(false);
  let editValue = $state("");
  let editTextarea = $state<HTMLTextAreaElement | null>(null);

  let sibInfo = $derived(chatStore.siblingInfo(message.id));

  function downloadFile(url: string, fallbackName: string) {
    const a = document.createElement("a");
    a.href = url;
    const fromUrl = url.split("/").pop()?.split("?")[0];
    a.download = fromUrl || fallbackName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  async function copyContent() {
    try {
      await navigator.clipboard.writeText(message.content);
      copied = true;
      setTimeout(() => (copied = false), 1200);
    } catch {}
  }

  function startEdit() {
    editValue = message.content;
    editing = true;
    setTimeout(() => editTextarea?.focus(), 0);
  }

  async function saveEdit() {
    const next = editValue.trim();
    editing = false;
    if (!next || next === message.content) return;
    await chatStore.editMessage(message.id, next);
  }

  function cancelEdit() {
    editing = false;
  }

  function onEditKey(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      saveEdit();
    } else if (e.key === "Escape") {
      cancelEdit();
    }
  }

  function prevBranch() {
    if (!sibInfo) return;
    const next = sibInfo.siblings[(sibInfo.index - 1 + sibInfo.total) % sibInfo.total];
    chatStore.switchBranch(next);
  }

  function nextBranch() {
    if (!sibInfo) return;
    const next = sibInfo.siblings[(sibInfo.index + 1) % sibInfo.total];
    chatStore.switchBranch(next);
  }
</script>

<div class="group flex justify-end">
  <div class="relative max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-white shadow-sm">
    <div class="absolute -left-20 top-1.5 flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
      <button
        class="rounded-md p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-800 dark:hover:text-gray-200"
        onclick={startEdit}
        title="Edit message"
        aria-label="Edit message"
      >
        <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
      </button>
      <button
        class="rounded-md p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-800 dark:hover:text-gray-200"
        onclick={copyContent}
        title="Copy message"
        aria-label="Copy message"
      >
        {#if copied}
          <svg class="h-3.5 w-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
        {:else}
          <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        {/if}
      </button>
    </div>
    <!-- Images -->
    {#if message.images?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.images as img, i}
          <div class="group relative">
            <img
              src={img}
              alt="Attached"
              class="max-h-40 rounded-lg object-contain"
            />
            <button
              class="absolute right-1 top-1 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover:opacity-100"
              onclick={() => downloadFile(img, `image-${i + 1}.png`)}
              title="Download image"
              aria-label="Download image"
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Videos (rendered as looping GIF-like clips; .gif files use <img> since <video> can't play them) -->
    {#if message.videos?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.videos as vid, i}
          {@const isGif = vid.toLowerCase().endsWith(".gif")}
          <div class="group relative">
            {#if isGif}
              <img
                src={vid}
                alt="GIF {i + 1}"
                class="max-h-48 rounded-lg"
              />
            {:else}
              <video
                src={vid}
                autoplay
                loop
                muted
                playsinline
                class="max-h-48 rounded-lg"
              ></video>
            {/if}
            <button
              class="absolute right-1 top-1 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover:opacity-100"
              onclick={() => downloadFile(vid, isGif ? `video-${i + 1}.gif` : `video-${i + 1}.mp4`)}
              title={isGif ? "Download GIF" : "Download video"}
              aria-label={isGif ? "Download GIF" : "Download video"}
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Text content -->
    {#if editing}
      <textarea
        bind:this={editTextarea}
        bind:value={editValue}
        onkeydown={onEditKey}
        class="w-full min-w-[40ch] resize-y rounded-md border border-blue-300/60 bg-blue-700 px-2 py-1 text-sm text-white outline-none focus:border-white/40"
        rows="3"
      ></textarea>
      <div class="mt-1 flex items-center justify-end gap-2 text-[11px]">
        <span class="text-blue-200/70">⌘↩ save · esc cancel</span>
        <button class="rounded bg-blue-700 px-2 py-0.5 hover:bg-blue-800" onclick={cancelEdit}>Cancel</button>
        <button class="rounded bg-white text-blue-700 px-2 py-0.5 font-medium hover:bg-blue-100" onclick={saveEdit}>Save</button>
      </div>
    {:else}
      <div class="text-sm leading-relaxed [&_a]:underline [&_code]:rounded [&_code]:bg-blue-500/30 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_pre]:my-1 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-blue-500/20 [&_pre]:p-2">
        {@html renderMarkdown(message.content)}
      </div>
    {/if}
  </div>
</div>

{#if sibInfo && !editing}
  <div class="-mt-1 flex justify-end pr-1">
    <div class="flex items-center gap-1 font-mono text-[11px] text-gray-400 dark:text-gray-500">
      <button class="rounded px-1 hover:bg-gray-100 dark:hover:bg-gray-800" onclick={prevBranch} aria-label="Previous branch">‹</button>
      <span>{sibInfo.index + 1}/{sibInfo.total}</span>
      <button class="rounded px-1 hover:bg-gray-100 dark:hover:bg-gray-800" onclick={nextBranch} aria-label="Next branch">›</button>
    </div>
  </div>
{/if}
