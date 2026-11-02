<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";
  import { chatStore } from "../stores/chat.svelte";
  import { MESSAGE_ACTION, HOVER_REVEAL, BUTTON_PRIMARY, BUTTON_SECONDARY } from "../ui";
  import Copy from "@lucide/svelte/icons/copy";
  import Check from "@lucide/svelte/icons/check";
  import Pencil from "@lucide/svelte/icons/pencil";
  import Download from "@lucide/svelte/icons/download";
  import ChevronLeft from "@lucide/svelte/icons/chevron-left";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import FileText from "@lucide/svelte/icons/file-text";

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

{#snippet downloadButton(url: string, name: string, label: string)}
  <button
    class="absolute right-1.5 top-1.5 rounded bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover/media:opacity-100"
    onclick={() => downloadFile(url, name)}
    title={label}
    aria-label={label}
  >
    <Download class="h-3.5 w-3.5" />
  </button>
{/snippet}

<div class="group flex flex-col items-end gap-1">
  <div class="max-w-[80%] {editing ? 'w-full' : ''} rounded-lg bg-surface-raised px-4 py-2.5 text-foreground">
    {#if message.images?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.images as img, i}
          <div class="group/media relative">
            <img src={img} alt="Attachment {i + 1}" class="max-h-48 rounded-md border border-line object-contain" />
            {@render downloadButton(img, `image-${i + 1}.png`, "Download image")}
          </div>
        {/each}
      </div>
    {/if}

    <!-- .gif files use <img> since <video> can't play them -->
    {#if message.videos?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.videos as vid, i}
          {@const isGif = vid.toLowerCase().endsWith(".gif")}
          <div class="group/media relative">
            {#if isGif}
              <img src={vid} alt="GIF {i + 1}" class="max-h-48 rounded-md border border-line" />
            {:else}
              <video src={vid} autoplay loop muted playsinline class="max-h-48 rounded-md border border-line"></video>
            {/if}
            {@render downloadButton(vid, isGif ? `video-${i + 1}.gif` : `video-${i + 1}.mp4`, isGif ? "Download GIF" : "Download video")}
          </div>
        {/each}
      </div>
    {/if}

    {#if message.audios?.length}
      <div class="mb-2 flex flex-col gap-2">
        {#each message.audios as aud, i (i)}
          <audio src={aud} controls class="h-9 max-w-full"></audio>
        {/each}
      </div>
    {/if}

    {#if message.files?.length}
      <div class="mb-2 flex flex-col gap-1.5">
        {#each message.files as file, i (i)}
          <details class="group/file rounded-md border border-line bg-background">
            <summary class="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 text-xs">
              <FileText class="h-3.5 w-3.5 shrink-0 text-muted" />
              <span class="truncate font-mono text-muted-strong">{file.name}</span>
              <ChevronRight class="ml-auto h-3.5 w-3.5 shrink-0 text-muted transition-transform group-open/file:rotate-90" />
            </summary>
            <pre class="max-h-72 overflow-auto border-t border-line px-3 py-2 font-mono text-xs text-code">{file.text}</pre>
          </details>
        {/each}
      </div>
    {/if}

    {#if editing}
      <textarea
        bind:this={editTextarea}
        bind:value={editValue}
        onkeydown={onEditKey}
        class="w-full resize-y rounded-md border border-line-strong bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-accent"
        rows="3"
        aria-label="Edit message"
      ></textarea>
      <div class="mt-2 flex items-center justify-end gap-2">
        <span class="mr-auto text-xs text-muted">Ctrl+Enter to save, Esc to cancel. The original stays as a branch.</span>
        <button class={BUTTON_SECONDARY} onclick={cancelEdit}>Cancel</button>
        <button class={BUTTON_PRIMARY} onclick={saveEdit}>Save and send</button>
      </div>
    {:else}
      <div class="markdown-content text-[15px]">
        {@html renderMarkdown(message.content)}
      </div>
    {/if}
  </div>

  {#if !editing}
    <div class="flex items-center gap-0.5">
      {#if sibInfo}
        <div class="flex items-center font-mono text-xs text-muted">
          <button class={MESSAGE_ACTION} onclick={prevBranch} aria-label="Previous branch" title="Previous branch"><ChevronLeft class="h-3.5 w-3.5" /></button>
          <span class="tabular-nums">{sibInfo.index + 1}/{sibInfo.total}</span>
          <button class={MESSAGE_ACTION} onclick={nextBranch} aria-label="Next branch" title="Next branch"><ChevronRight class="h-3.5 w-3.5" /></button>
        </div>
      {/if}
      <div class="flex items-center gap-0.5 {HOVER_REVEAL}">
        <button class={MESSAGE_ACTION} onclick={startEdit} disabled={chatStore.isStreaming} title="Edit (creates a new branch)" aria-label="Edit message">
          <Pencil class="h-3.5 w-3.5" />
        </button>
        <button class={MESSAGE_ACTION} onclick={copyContent} title="Copy message" aria-label="Copy message">
          {#if copied}<Check class="h-3.5 w-3.5 text-success" />{:else}<Copy class="h-3.5 w-3.5" />{/if}
        </button>
      </div>
    </div>
  {/if}
</div>
