<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import { modelStore } from "../stores/models.svelte";
  import * as api from "../services/api";

  let textareaEl: HTMLTextAreaElement;
  let inputValue = $state("");
  let pendingImages = $state<{ file: File; url: string; uploadedUrl?: string }[]>([]);
  let pendingVideos = $state<{ file: File; url: string; uploadedUrl?: string }[]>([]);
  let isDragging = $state(false);

  function autoResize() {
    if (!textareaEl) return;
    textareaEl.style.height = "auto";
    textareaEl.style.height = Math.min(textareaEl.scrollHeight, 200) + "px";
  }

  async function handleSend() {
    if (chatStore.isStreaming) return;
    const content = inputValue.trim();
    if (!content && !pendingImages.length && !pendingVideos.length) return;

    // Upload pending images
    const imageUrls: string[] = [];
    for (const img of pendingImages) {
      if (!img.uploadedUrl) {
        try {
          const result = await api.uploadImage(img.file);
          img.uploadedUrl = result.url;
        } catch (e) {
          console.error("Upload failed:", e);
          continue;
        }
      }
      if (img.uploadedUrl) {
        imageUrls.push(img.uploadedUrl);
      }
    }

    // Upload pending videos
    const videoUrls: string[] = [];
    for (const vid of pendingVideos) {
      if (!vid.uploadedUrl) {
        try {
          const result = await api.uploadVideo(vid.file);
          vid.uploadedUrl = result.url;
        } catch (e) {
          console.error("Video upload failed:", e);
          continue;
        }
      }
      if (vid.uploadedUrl) {
        videoUrls.push(vid.uploadedUrl);
      }
    }

    inputValue = "";
    pendingImages = [];
    pendingVideos = [];
    if (textareaEl) {
      textareaEl.style.height = "auto";
    }

    await chatStore.sendMessage(
      content,
      imageUrls.length ? imageUrls : undefined,
      videoUrls.length ? videoUrls : undefined,
    );
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

  function handleDragLeave() {
    isDragging = false;
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

<div
  class="shrink-0 border-t border-gray-200 bg-white px-4 py-3 dark:border-gray-800 dark:bg-gray-950"
  role="region"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  <div class="mx-auto max-w-3xl">
    <!-- Media previews -->
    {#if pendingImages.length > 0 || pendingVideos.length > 0}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each pendingImages as img, i}
          <div class="relative h-16 w-16 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
            <img src={img.url} alt="Upload preview" class="h-full w-full object-cover" />
            <button
              class="absolute right-0 top-0 rounded-bl bg-black/60 p-0.5 text-white hover:bg-black/80"
              onclick={() => removeImage(i)}
              aria-label="Remove image"
            >
              <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        {/each}
        {#each pendingVideos as vid, i}
          {@const isGif = vid.file.name.toLowerCase().endsWith(".gif")}
          <div class="relative h-16 w-24 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
            {#if isGif}
              <img src={vid.url} alt="GIF preview" class="h-full w-full object-cover" />
            {:else}
              <video src={vid.url} class="h-full w-full object-cover" muted></video>
            {/if}
            <div class="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 py-0.5 text-[9px] text-white">{isGif ? "GIF" : "Video"}</div>
            <button
              class="absolute right-0 top-0 rounded-bl bg-black/60 p-0.5 text-white hover:bg-black/80"
              onclick={() => removeVideo(i)}
              aria-label="Remove video"
            >
              <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Drag overlay indicator -->
    {#if isDragging}
      <div class="mb-2 rounded-lg border-2 border-dashed border-blue-400 bg-blue-50 p-4 text-center text-sm text-blue-600 dark:border-blue-500 dark:bg-blue-900/20 dark:text-blue-400">
        Drop files here
      </div>
    {/if}

    <!-- Input area -->
    <div class="flex items-start gap-2">
      <!-- File upload button -->
      {#if modelStore.isMultimodal}
        <label class="shrink-0 cursor-pointer rounded-lg p-2 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 dark:hover:text-gray-300">
          <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
          <input
            type="file"
            class="hidden"
            accept="image/*,video/*,audio/*,.gif,.mp4,.avi,.mov,.mkv,.webm,.m4v,.txt,.md,.py,.js,.ts,.json,.csv,.xml,.yaml,.yml,.toml,.html,.css"
            multiple
            onchange={handleFileSelect}
          />
        </label>
      {/if}

      <!-- Textarea -->
      <div class="relative min-w-0 flex-1">
        <textarea
          bind:this={textareaEl}
          bind:value={inputValue}
          oninput={autoResize}
          onkeydown={handleKeydown}
          placeholder="Type a message..."
          rows="1"
          disabled={!modelStore.selectedModel}
          class="w-full resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-gray-400 focus:border-blue-400 focus:bg-white focus:ring-1 focus:ring-blue-400 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:placeholder:text-gray-500 dark:focus:border-blue-500 dark:focus:bg-gray-900"
        ></textarea>
      </div>

      <!-- Send/Stop button -->
      {#if chatStore.isStreaming}
        <button
          onclick={handleStop}
          class="shrink-0 rounded-xl bg-red-500 p-2.5 text-white transition-colors hover:bg-red-600"
          title="Stop generating"
        >
          <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
        </button>
      {:else}
        <button
          onclick={handleSend}
          disabled={!modelStore.selectedModel || (!inputValue.trim() && !pendingImages.length)}
          class="shrink-0 rounded-xl bg-blue-600 p-2.5 text-white transition-colors hover:bg-blue-700 disabled:opacity-40 disabled:hover:bg-blue-600"
          title="Send message"
        >
          <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19V5m-7 7l7-7 7 7" />
          </svg>
        </button>
      {/if}
    </div>

    <p class="mt-1.5 text-center text-xs text-gray-400 dark:text-gray-500">
      Press Enter to send, Shift+Enter for new line
    </p>
  </div>
</div>
