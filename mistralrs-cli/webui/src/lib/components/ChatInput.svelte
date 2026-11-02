<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import { modelStore } from "../stores/models.svelte";
  import * as api from "../services/api";
  import ArrowUp from "@lucide/svelte/icons/arrow-up";
  import Square from "@lucide/svelte/icons/square";
  import Paperclip from "@lucide/svelte/icons/paperclip";
  import X from "@lucide/svelte/icons/x";

  let { placeholder = "Message mistral.rs" }: { placeholder?: string } = $props();

  let textareaEl: HTMLTextAreaElement;
  let inputValue = $state("");
  let pendingImages = $state<{ file: File; url: string; uploadedUrl?: string }[]>([]);
  let pendingVideos = $state<{ file: File; url: string; uploadedUrl?: string }[]>([]);
  let isDragging = $state(false);

  function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  function autoResize() {
    if (!textareaEl) return;
    textareaEl.style.height = "auto";
    textareaEl.style.height = Math.min(textareaEl.scrollHeight, 200) + "px";
  }

  async function handleSend() {
    if (chatStore.isStreaming) return;
    const content = inputValue.trim();
    if (!content && !pendingImages.length && !pendingVideos.length) return;

    // Encode as data URLs; the /v1/chat/completions endpoint only accepts
    // http/https/data sources, not the server's relative `uploads/...` paths.
    const imageUrls: string[] = [];
    for (const img of pendingImages) {
      if (!img.uploadedUrl) {
        try {
          img.uploadedUrl = await fileToDataUrl(img.file);
        } catch (e) {
          console.error("Failed to read image:", e);
          continue;
        }
      }
      if (img.uploadedUrl) {
        imageUrls.push(img.uploadedUrl);
      }
    }

    const videoUrls: string[] = [];
    for (const vid of pendingVideos) {
      if (!vid.uploadedUrl) {
        try {
          vid.uploadedUrl = await fileToDataUrl(vid.file);
        } catch (e) {
          console.error("Failed to read video:", e);
          continue;
        }
      }
      if (vid.uploadedUrl) {
        videoUrls.push(vid.uploadedUrl);
      }
    }

    const draft = { text: inputValue, images: pendingImages, videos: pendingVideos };
    inputValue = "";
    pendingImages = [];
    pendingVideos = [];
    if (textareaEl) {
      textareaEl.style.height = "auto";
    }

    const sent = await chatStore.sendMessage(
      content,
      imageUrls.length ? imageUrls : undefined,
      videoUrls.length ? videoUrls : undefined,
    );
    if (!sent) {
      inputValue = draft.text;
      pendingImages = draft.images;
      pendingVideos = draft.videos;
      autoResize();
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleStop() {
    chatStore.stopStreaming();
  }

  async function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files?.length) return;
    for (const file of input.files) {
      await addFile(file);
    }
    input.value = "";
  }

  async function addFile(file: File) {
    if (file.type.startsWith("video/") || file.name.endsWith(".gif")) {
      const url = URL.createObjectURL(file);
      pendingVideos = [...pendingVideos, { file, url }];
    } else if (file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      pendingImages = [...pendingImages, { file, url }];
    } else if (file.type.startsWith("audio/")) {
      try {
        await api.uploadAudio(file);
      } catch (e) {
        console.error("Audio upload failed:", e);
      }
    } else {
      try {
        await api.uploadText(file);
      } catch (e) {
        console.error("Text upload failed:", e);
      }
    }
  }

  function removeImage(index: number) {
    URL.revokeObjectURL(pendingImages[index].url);
    pendingImages = pendingImages.filter((_, i) => i !== index);
  }

  function removeVideo(index: number) {
    URL.revokeObjectURL(pendingVideos[index].url);
    pendingVideos = pendingVideos.filter((_, i) => i !== index);
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    const zone = e.currentTarget as HTMLElement;
    if (!zone.contains(e.relatedTarget as Node | null)) isDragging = false;
  }

  async function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
    if (e.dataTransfer?.files) {
      for (const file of e.dataTransfer.files) {
        await addFile(file);
      }
    }
  }
</script>

{#snippet removeButton(onclick: () => void, label: string)}
  <button
    class="absolute right-1 top-1 rounded bg-black/60 p-0.5 text-white hover:bg-black/80"
    {onclick}
    aria-label={label}
    title={label}
  >
    <X class="h-3 w-3" />
  </button>
{/snippet}

<div
  class="relative rounded-lg border bg-surface shadow-sm shadow-black/5 transition-colors {isDragging ? 'border-accent' : 'border-line-strong focus-within:border-muted'}"
  role="region"
  aria-label="Message composer"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  {#if isDragging}
    <div class="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-accent-soft/90 text-sm font-medium text-accent">
      Drop files to attach
    </div>
  {/if}

  {#if pendingImages.length > 0 || pendingVideos.length > 0}
    <div class="flex flex-wrap gap-2 px-3 pt-3">
      {#each pendingImages as img, i}
        <div class="relative h-16 w-16 overflow-hidden rounded-md border border-line">
          <img src={img.url} alt="Attachment preview" class="h-full w-full object-cover" />
          {@render removeButton(() => removeImage(i), "Remove image")}
        </div>
      {/each}
      {#each pendingVideos as vid, i}
        {@const isGif = vid.file.name.toLowerCase().endsWith(".gif")}
        <div class="relative h-16 w-24 overflow-hidden rounded-md border border-line">
          {#if isGif}
            <img src={vid.url} alt="GIF preview" class="h-full w-full object-cover" />
          {:else}
            <video src={vid.url} class="h-full w-full object-cover" muted></video>
          {/if}
          <span class="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] text-white">{isGif ? "GIF" : "Video"}</span>
          {@render removeButton(() => removeVideo(i), "Remove video")}
        </div>
      {/each}
    </div>
  {/if}

  <textarea
    bind:this={textareaEl}
    bind:value={inputValue}
    oninput={autoResize}
    onkeydown={handleKeydown}
    placeholder={modelStore.selectedModel ? placeholder : "No model loaded"}
    rows="1"
    disabled={!modelStore.selectedModel}
    aria-label="Message"
    class="block max-h-[200px] w-full resize-none bg-transparent px-4 pt-3 pb-1 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted disabled:opacity-50"
  ></textarea>

  <div class="flex items-center gap-2 px-2 pb-2">
    {#if modelStore.isMultimodal}
      <label class="inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-muted hover:bg-surface-raised hover:text-foreground" title="Attach files">
        <Paperclip class="h-4 w-4" />
        <input
          type="file"
          class="hidden"
          accept="image/*,video/*,audio/*,.gif,.mp4,.avi,.mov,.mkv,.webm,.m4v,.txt,.md,.py,.js,.ts,.json,.csv,.xml,.yaml,.yml,.toml,.html,.css"
          multiple
          onchange={handleFileSelect}
          aria-label="Attach files"
        />
      </label>
    {/if}

    <span class="ml-1 hidden text-xs text-muted sm:inline">Enter to send, Shift+Enter for a new line</span>

    {#if chatStore.isStreaming}
      <button
        onclick={handleStop}
        class="ml-auto inline-flex h-8 w-8 items-center justify-center rounded-md bg-foreground text-background hover:opacity-85"
        title="Stop generating"
        aria-label="Stop generating"
      >
        <Square class="h-3.5 w-3.5 fill-current" />
      </button>
    {:else}
      <button
        onclick={handleSend}
        disabled={!modelStore.selectedModel || (!inputValue.trim() && !pendingImages.length && !pendingVideos.length)}
        class="ml-auto inline-flex h-8 w-8 items-center justify-center rounded-md bg-fill text-on-fill hover:opacity-90 disabled:bg-surface-raised disabled:text-muted"
        title="Send message"
        aria-label="Send message"
      >
        <ArrowUp class="h-4 w-4" />
      </button>
    {/if}
  </div>
</div>
