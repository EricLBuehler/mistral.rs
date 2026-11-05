<script lang="ts">
  import { chatStore } from "../stores/chat.svelte";
  import { modelStore } from "../stores/models.svelte";
  import type { TextAttachment } from "../types";
  import ArrowUp from "@lucide/svelte/icons/arrow-up";
  import Square from "@lucide/svelte/icons/square";
  import Paperclip from "@lucide/svelte/icons/paperclip";
  import X from "@lucide/svelte/icons/x";
  import FileText from "@lucide/svelte/icons/file-text";
  import AudioLines from "@lucide/svelte/icons/audio-lines";

  let { placeholder = "Message mistral.rs" }: { placeholder?: string } = $props();

  let textareaEl: HTMLTextAreaElement;
  let inputValue = $state("");
  type PendingMedia = { file: File; url: string };

  const MAX_TEXT_FILE_BYTES = 1024 * 1024;
  const TEXT_FILE_EXTENSIONS =
    ".txt,.md,.py,.js,.ts,.json,.csv,.xml,.yaml,.yml,.toml,.html,.css,.rs,.go,.c,.h,.cpp,.java,.sh,.log";
  const IMAGE_ACCEPT = "image/*";
  const VIDEO_ACCEPT = "video/*,.gif,.mp4,.avi,.mov,.mkv,.webm,.m4v";
  const AUDIO_ACCEPT = "audio/*";

  let pendingImages = $state<PendingMedia[]>([]);
  let pendingVideos = $state<PendingMedia[]>([]);
  let pendingAudios = $state<PendingMedia[]>([]);
  let pendingFiles = $state<TextAttachment[]>([]);
  let isDragging = $state(false);

  let canImage = $derived(modelStore.acceptsInput("vision"));
  let canVideo = $derived(modelStore.acceptsInput("video"));
  let canAudio = $derived(modelStore.acceptsInput("audio"));
  let accept = $derived(
    [TEXT_FILE_EXTENSIONS, canImage && IMAGE_ACCEPT, canVideo && VIDEO_ACCEPT, canAudio && AUDIO_ACCEPT]
      .filter(Boolean)
      .join(","),
  );
  let hasAttachments = $derived(
    pendingImages.length + pendingVideos.length + pendingAudios.length + pendingFiles.length > 0,
  );

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

  async function dataUrls(list: PendingMedia[]): Promise<string[] | undefined> {
    if (!list.length) return undefined;
    return Promise.all(list.map((m) => fileToDataUrl(m.file)));
  }

  async function handleSend() {
    if (chatStore.isStreaming) return;
    const content = inputValue.trim();
    if (!content && !hasAttachments) return;

    // Data URLs because /v1/chat/completions only accepts http/https/data sources, not `uploads/...` paths.
    let images, videos, audios;
    try {
      [images, videos, audios] = await Promise.all([
        dataUrls(pendingImages),
        dataUrls(pendingVideos),
        dataUrls(pendingAudios),
      ]);
    } catch (e) {
      chatStore.lastError = `Couldn't read an attachment: ${e instanceof Error ? e.message : String(e)}`;
      return;
    }

    const draft = { text: inputValue, images: pendingImages, videos: pendingVideos, audios: pendingAudios, files: pendingFiles };
    const files = pendingFiles.length ? pendingFiles : undefined;
    inputValue = "";
    pendingImages = [];
    pendingVideos = [];
    pendingAudios = [];
    pendingFiles = [];
    if (textareaEl) {
      textareaEl.style.height = "auto";
    }

    const sent = await chatStore.sendMessage(content, { images, videos, audios, files });
    if (!sent) {
      inputValue = draft.text;
      pendingImages = draft.images;
      pendingVideos = draft.videos;
      pendingAudios = draft.audios;
      pendingFiles = draft.files;
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

  async function addTextFile(file: File) {
    if (file.size > MAX_TEXT_FILE_BYTES) {
      chatStore.lastError = `${file.name} is too large to attach (limit ${MAX_TEXT_FILE_BYTES / 1024 / 1024} MB)`;
      return;
    }
    const text = await file.text();
    if (text.includes("\u0000")) {
      chatStore.lastError = `${file.name} doesn't look like a text file`;
      return;
    }
    pendingFiles = [...pendingFiles, { name: file.name, text }];
  }

  async function addFile(file: File) {
    const isVideo = file.type.startsWith("video/") || file.name.toLowerCase().endsWith(".gif");
    const isImage = !isVideo && file.type.startsWith("image/");
    const isAudio = file.type.startsWith("audio/");
    const media = (list: PendingMedia[]) => [...list, { file, url: URL.createObjectURL(file) }];
    if (isVideo && canVideo) pendingVideos = media(pendingVideos);
    else if (isImage && canImage) pendingImages = media(pendingImages);
    else if (isAudio && canAudio) pendingAudios = media(pendingAudios);
    else if (isVideo || isImage || isAudio) chatStore.lastError = `${modelStore.selectedModel} can't take ${file.name}`;
    else await addTextFile(file);
  }

  function removeImage(index: number) {
    URL.revokeObjectURL(pendingImages[index].url);
    pendingImages = pendingImages.filter((_, i) => i !== index);
  }

  function removeVideo(index: number) {
    URL.revokeObjectURL(pendingVideos[index].url);
    pendingVideos = pendingVideos.filter((_, i) => i !== index);
  }

  function removeAudio(index: number) {
    URL.revokeObjectURL(pendingAudios[index].url);
    pendingAudios = pendingAudios.filter((_, i) => i !== index);
  }

  function removeFile(index: number) {
    pendingFiles = pendingFiles.filter((_, i) => i !== index);
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

{#snippet chip(Icon: typeof FileText, name: string, onclick: () => void, label: string)}
  <div class="flex h-16 max-w-48 items-center gap-2 rounded-md border border-line bg-background pl-2.5 pr-1">
    <Icon class="h-4 w-4 shrink-0 text-muted" />
    <span class="truncate font-mono text-xs text-muted-strong" title={name}>{name}</span>
    <button class="shrink-0 rounded p-0.5 text-muted hover:bg-surface-raised hover:text-foreground" {onclick} aria-label={label} title={label}>
      <X class="h-3 w-3" />
    </button>
  </div>
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

  {#if hasAttachments}
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
      {#each pendingAudios as aud, i}
        {@render chip(AudioLines, aud.file.name, () => removeAudio(i), "Remove audio")}
      {/each}
      {#each pendingFiles as f, i}
        {@render chip(FileText, f.name, () => removeFile(i), "Remove file")}
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
    {#if modelStore.selectedModel}
      <label class="inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-muted hover:bg-surface-raised hover:text-foreground" title="Attach files">
        <Paperclip class="h-4 w-4" />
        <input
          type="file"
          class="hidden"
          {accept}
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
        disabled={!modelStore.selectedModel || (!inputValue.trim() && !hasAttachments)}
        class="ml-auto inline-flex h-8 w-8 items-center justify-center rounded-md bg-fill text-on-fill hover:opacity-90 disabled:bg-surface-raised disabled:text-muted"
        title="Send message"
        aria-label="Send message"
      >
        <ArrowUp class="h-4 w-4" />
      </button>
    {/if}
  </div>
</div>
