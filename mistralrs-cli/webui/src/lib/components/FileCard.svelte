<script lang="ts">
  import type { File as ProducedFile } from "../types";

  let { file }: { file: ProducedFile } = $props();

  const PREVIEW_LINES = 10;
  const PREVIEW_CHARS = 1024;

  function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
  }

  function getApiBase(): string {
    const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
    try {
      return new URL("../", new URL(base, window.location.origin)).pathname;
    } catch {
      return "/";
    }
  }

  // Prefer the inline body persisted in the chat sidecar -- the server's in-memory
  // FileStore expires after 30 minutes and is wiped on restart, so /v1/files/{id}/content
  // 404s after a reload unless the file is still resident.
  let inlineBlobUrl = $derived.by(() => {
    if (typeof file.text === "string") {
      const blob = new Blob([file.text], {
        type: file.mime_type ?? "text/plain;charset=utf-8",
      });
      return URL.createObjectURL(blob);
    }
    if (typeof file.data_base64 === "string") {
      try {
        const binary = atob(file.data_base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes.buffer as ArrayBuffer], {
          type: file.mime_type ?? "application/octet-stream",
        });
        return URL.createObjectURL(blob);
      } catch {
        return null;
      }
    }
    return null;
  });

  // Revoke the previous object URL when the file (and thus the derived blob URL) changes.
  $effect(() => {
    const url = inlineBlobUrl;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  });

  let downloadHref = $derived(
    inlineBlobUrl ?? file.url ?? `${getApiBase()}v1/files/${file.id}/content`,
  );

  let mime = $derived((file.mime_type ?? "").toLowerCase());
  let isImage = $derived(mime.startsWith("image/") && mime !== "image/svg+xml");
  let isVideo = $derived(mime.startsWith("video/"));
  let isAudio = $derived(mime.startsWith("audio/"));
  let hasText = $derived(typeof file.text === "string" || typeof file.preview === "string");
  let isErr = $derived(!!file.error);

  let imgSrc = $derived(() => {
    if (!isImage) return null;
    if (file.data_base64) return `data:${file.mime_type ?? "image/png"};base64,${file.data_base64}`;
    return downloadHref;
  });

  let mediaSrc = $derived(() => {
    if (file.data_base64) return `data:${file.mime_type ?? "application/octet-stream"};base64,${file.data_base64}`;
    return downloadHref;
  });

  let previewText = $derived(() => {
    const src = file.text ?? file.preview ?? "";
    if (!src) return "";
    const sliced = src.length > PREVIEW_CHARS ? src.slice(0, PREVIEW_CHARS) : src;
    const lines = sliced.split("\n");
    if (lines.length > PREVIEW_LINES) {
      return lines.slice(0, PREVIEW_LINES).join("\n");
    }
    return sliced;
  });

  let truncated = $derived(() => {
    const src = file.text ?? file.preview ?? "";
    return src.length > PREVIEW_CHARS || src.split("\n").length > PREVIEW_LINES;
  });

  let expanded = $state(false);
  let expandedImage = $state<string | null>(null);

  function downloadFile(e: MouseEvent) {
    e.preventDefault();
    const a = document.createElement("a");
    a.href = downloadHref;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
</script>

{#if isErr}
  <div class="inline-flex items-center gap-2 rounded-full border border-red-300 bg-red-50 px-3 py-1 text-xs text-red-700 dark:border-red-700/60 dark:bg-red-900/20 dark:text-red-300">
    <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
    </svg>
    <span class="font-medium">missing: {file.name}</span>
    <span class="text-red-500/80 dark:text-red-400/80">({file.error?.code})</span>
  </div>
{:else}
  <div class="overflow-hidden rounded-xl border border-emerald-200 dark:border-emerald-800/50">
    <!-- Header -->
    <div class="flex items-center gap-2 bg-emerald-50 px-3 py-1.5 dark:bg-emerald-900/20">
      <svg class="h-4 w-4 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <span class="text-xs font-medium text-emerald-800 dark:text-emerald-200">{file.name}</span>
      <span class="text-xs text-emerald-700/70 dark:text-emerald-300/70">
        {file.format ?? file.mime_type ?? "file"} &middot; {formatBytes(file.bytes)}
      </span>
      <a
        href={downloadHref}
        onclick={downloadFile}
        class="ml-auto inline-flex items-center gap-1 rounded-md bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800 hover:bg-emerald-200 dark:bg-emerald-900/40 dark:text-emerald-200 dark:hover:bg-emerald-900/60"
        title="Download"
      >
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        Download
      </a>
    </div>

    <!-- Body -->
    {#if isImage}
      <div class="bg-gray-50 p-3 dark:bg-gray-900/50">
        <button
          class="block w-full text-center"
          onclick={() => expandedImage = imgSrc()}
          title="Click to expand"
          aria-label="Expand image"
        >
          <img
            src={imgSrc() ?? ""}
            alt={file.name}
            class="mx-auto max-h-64 max-w-full rounded-md object-contain shadow-sm"
          />
        </button>
      </div>
    {:else if isVideo}
      <div class="bg-gray-50 p-3 dark:bg-gray-900/50">
        <video
          src={mediaSrc()}
          controls
          class="mx-auto max-h-96 max-w-full rounded-md shadow-sm"
        >
          <track kind="captions" />
        </video>
      </div>
    {:else if isAudio}
      <div class="bg-gray-50 p-3 dark:bg-gray-900/50">
        <audio src={mediaSrc()} controls class="w-full">
          <track kind="captions" />
        </audio>
      </div>
    {:else if hasText}
      <div class="bg-gray-50 dark:bg-gray-900">
        <pre class="overflow-x-auto whitespace-pre-wrap break-words px-3 py-2 font-mono text-xs leading-relaxed text-gray-700 dark:text-gray-300">{expanded ? (file.text ?? file.preview ?? "") : previewText()}</pre>
        {#if truncated()}
          <button
            class="w-full border-t border-gray-200 bg-gray-100 px-3 py-1 text-xs text-gray-600 hover:bg-gray-200 dark:border-gray-700 dark:bg-gray-800/80 dark:text-gray-300 dark:hover:bg-gray-800"
            onclick={() => expanded = !expanded}
          >
            {expanded ? "Collapse" : "Show more"}
          </button>
        {/if}
      </div>
    {:else}
      <div class="bg-gray-50 px-3 py-2 text-xs text-gray-500 dark:bg-gray-900 dark:text-gray-400">
        Binary content. Use Download to fetch.
      </div>
    {/if}
  </div>
{/if}

<!-- Expanded image overlay -->
{#if expandedImage}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-8 backdrop-blur-sm">
    <button
      class="absolute inset-0"
      onclick={() => expandedImage = null}
      aria-label="Close expanded image"
    ></button>
    <img
      src={expandedImage}
      alt={file.name}
      class="relative max-h-full max-w-full rounded-lg shadow-2xl"
    />
    <div class="absolute right-4 top-4 flex gap-2">
      <button
        class="rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
        onclick={downloadFile}
        title="Download"
        aria-label="Download"
      >
        <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
      </button>
      <button
        class="rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
        onclick={() => expandedImage = null}
        title="Close"
        aria-label="Close"
      >
        <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  </div>
{/if}
