<script lang="ts">
  import type { File as ProducedFile } from "../types";
  import { TOOL_CARD } from "../ui";
  import FileText from "@lucide/svelte/icons/file-text";
  import TriangleAlert from "@lucide/svelte/icons/triangle-alert";
  import Download from "@lucide/svelte/icons/download";
  import X from "@lucide/svelte/icons/x";

  let { file }: { file: ProducedFile } = $props();

  const PREVIEW_LINES = 10;
  const PREVIEW_CHARS = 1024;
  const OVERLAY_BUTTON = "rounded-md bg-black/50 p-2 text-white hover:bg-black/70";

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
  <div class="inline-flex items-center gap-2 rounded-md border border-danger/30 bg-danger-soft px-2.5 py-1 text-xs text-danger">
    <TriangleAlert class="h-3.5 w-3.5" />
    <span class="font-medium">missing: {file.name}</span>
    <span class="font-mono text-[11px] opacity-80">({file.error?.code})</span>
  </div>
{:else}
  <div class={TOOL_CARD}>
    <div class="flex items-center gap-2 px-3 py-1.5">
      <FileText class="h-3.5 w-3.5 shrink-0 text-muted" />
      <span class="min-w-0 truncate font-mono text-xs text-foreground">{file.name}</span>
      <span class="shrink-0 font-mono text-[11px] text-muted">
        {file.format ?? file.mime_type ?? "file"} &middot; {formatBytes(file.bytes)}
      </span>
      <a
        href={downloadHref}
        onclick={downloadFile}
        class="ml-auto inline-flex shrink-0 items-center gap-1 rounded-md border border-line-strong px-2 py-0.5 text-xs text-muted-strong hover:bg-surface-raised hover:text-foreground"
        title="Download"
      >
        <Download class="h-3.5 w-3.5" />
        Download
      </a>
    </div>

    {#if isImage}
      <div class="border-t border-line bg-background p-3">
        <button
          class="block w-full text-center"
          onclick={() => expandedImage = imgSrc()}
          title="Click to expand"
          aria-label="Expand image"
        >
          <img
            src={imgSrc() ?? ""}
            alt={file.name}
            class="mx-auto max-h-64 max-w-full rounded-md border border-line object-contain"
          />
        </button>
      </div>
    {:else if isVideo}
      <div class="border-t border-line bg-background p-3">
        <video
          src={mediaSrc()}
          controls
          class="mx-auto max-h-96 max-w-full rounded-md border border-line"
        >
          <track kind="captions" />
        </video>
      </div>
    {:else if isAudio}
      <div class="border-t border-line bg-background p-3">
        <audio src={mediaSrc()} controls class="w-full">
          <track kind="captions" />
        </audio>
      </div>
    {:else if hasText}
      <div class="border-t border-line bg-background">
        <pre class="overflow-x-auto whitespace-pre-wrap break-words px-3 py-2 font-mono text-xs leading-relaxed text-code">{expanded ? (file.text ?? file.preview ?? "") : previewText()}</pre>
        {#if truncated()}
          <button
            class="w-full border-t border-line bg-surface px-3 py-1 text-xs text-muted hover:bg-surface-raised hover:text-foreground"
            onclick={() => expanded = !expanded}
          >
            {expanded ? "Collapse" : "Show more"}
          </button>
        {/if}
      </div>
    {:else}
      <div class="border-t border-line px-3 py-2 text-xs text-muted">
        Binary content. Use Download to fetch.
      </div>
    {/if}
  </div>
{/if}

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
      class="relative max-h-full max-w-full rounded-md"
    />
    <div class="absolute right-4 top-4 flex gap-2">
      <button
        class={OVERLAY_BUTTON}
        onclick={downloadFile}
        title="Download"
        aria-label="Download"
      >
        <Download class="h-4 w-4" />
      </button>
      <button
        class={OVERLAY_BUTTON}
        onclick={() => expandedImage = null}
        title="Close"
        aria-label="Close"
      >
        <X class="h-4 w-4" />
      </button>
    </div>
  </div>
{/if}
