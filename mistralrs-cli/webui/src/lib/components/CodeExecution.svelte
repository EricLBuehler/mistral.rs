<script lang="ts">
  import type { CodeExecutionData } from "../types";
  import hljs from "highlight.js";
  import VideoFrames from "./VideoFrames.svelte";

  let { data, phase }: { data: CodeExecutionData; phase: "calling" | "complete" } = $props();

  let videoFrameCount = $derived(
    data.video_frames_base64?.length ?? data.video_frame_count ?? 0
  );

  const PREVIEW_LINES = 10;

  let codeLines = $derived(data.code?.split("\n") ?? []);
  let totalLines = $derived(codeLines.length);
  let needsExpand = $derived(totalLines > PREVIEW_LINES);
  let previewCode = $derived(
    needsExpand ? codeLines.slice(0, PREVIEW_LINES).join("\n") : (data.code ?? "")
  );
  let previewLineCount = $derived(needsExpand ? PREVIEW_LINES : totalLines);

  let highlightedPreview = $derived(
    previewCode ? hljs.highlight(previewCode, { language: "python" }).value : ""
  );
  let highlightedFull = $derived(
    data.code ? hljs.highlight(data.code, { language: "python" }).value : ""
  );

  let executionTimeFormatted = $derived(() => {
    if (data.execution_time_ms == null) return null;
    if (data.execution_time_ms < 1000) return `${data.execution_time_ms}ms`;
    return `${(data.execution_time_ms / 1000).toFixed(2)}s`;
  });

  let codeExpanded = $state(false);
  let expandedImage = $state<string | null>(null);
  let copyState = $state<"idle" | "copied">("idle");

  function lineNumberGutter(n: number): string {
    let s = "";
    for (let i = 1; i <= n; i++) s += i + (i < n ? "\n" : "");
    return s;
  }

  async function copyCode(e: Event) {
    e.stopPropagation();
    if (!data.code) return;
    try {
      await navigator.clipboard.writeText(data.code);
      copyState = "copied";
      setTimeout(() => (copyState = "idle"), 1200);
    } catch {
      // ignore
    }
  }
</script>

<div class="overflow-hidden rounded-xl border border-purple-200 dark:border-purple-800/50">
  <!-- Header -->
  <div class="flex items-center gap-2 bg-purple-50 px-3 py-1.5 dark:bg-purple-900/20">
    <svg class="h-4 w-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
    <span class="text-xs font-medium text-purple-700 dark:text-purple-300">Python</span>
    {#if phase === "calling"}
      <span class="ml-auto flex items-center gap-1 text-xs text-purple-500">
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-purple-400"></span>
        Running...
      </span>
    {:else}
      <div class="ml-auto flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
        {#if data.images_base64?.length}
          <span class="flex items-center gap-1">
            <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            {data.images_base64.length}
          </span>
        {/if}
        {#if videoFrameCount}
          <span class="flex items-center gap-1">
            <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            {videoFrameCount}
          </span>
        {/if}
        {#if executionTimeFormatted()}
          <span>{executionTimeFormatted()}</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Code section -->
  {#if data.code}
    <div class="border-t border-gray-700">
      <div class="flex items-center bg-gray-900">
        <button
          class="flex flex-1 cursor-pointer items-center gap-1.5 px-3 py-1 text-left"
          onclick={() => codeExpanded = !codeExpanded}
        >
          <svg class="h-3 w-3 text-gray-400 transition-transform {codeExpanded ? 'rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
          <span class="text-xs font-medium text-gray-400">Code</span>
          <span class="text-xs text-gray-500">({totalLines} lines)</span>
        </button>
        <button
          class="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-gray-200"
          onclick={copyCode}
          title="Copy code"
          aria-label="Copy code"
        >
          {#if copyState === "copied"}
            <svg class="h-3.5 w-3.5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            <span>Copied</span>
          {:else}
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Copy</span>
          {/if}
        </button>
      </div>
      {#if codeExpanded}
        <div class="code-exec-pane flex bg-gray-950 py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right text-xs leading-relaxed text-gray-600">{lineNumberGutter(totalLines)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedFull}</code></pre>
          </div>
        </div>
      {:else if needsExpand}
        <div class="code-exec-pane relative flex bg-gray-950 py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right text-xs leading-relaxed text-gray-600">{lineNumberGutter(previewLineCount)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedPreview}</code></pre>
          </div>
          <div class="pointer-events-none absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-gray-950 to-transparent"></div>
        </div>
      {:else}
        <div class="code-exec-pane flex bg-gray-950 py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right text-xs leading-relaxed text-gray-600">{lineNumberGutter(totalLines)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedFull}</code></pre>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- stdout -->
  {#if data.stdout}
    <div class="border-t border-gray-200 dark:border-gray-700">
      <div class="flex items-center gap-1.5 bg-gray-100 px-3 py-1 dark:bg-gray-800/80">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400">stdout</span>
      </div>
      <div class="bg-gray-50 px-3 py-2 dark:bg-gray-900">
        <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-700 dark:text-gray-300">{data.stdout}</pre>
      </div>
    </div>
  {/if}

  <!-- stderr -->
  {#if data.stderr}
    <div class="border-t border-red-200 dark:border-red-800/50">
      <div class="flex items-center gap-1.5 bg-red-50 px-3 py-1 dark:bg-red-900/30">
        <span class="text-xs font-medium text-red-600 dark:text-red-400">stderr</span>
      </div>
      <div class="bg-red-50/50 px-3 py-2 dark:bg-red-900/10">
        <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-red-700 dark:text-red-300">{data.stderr}</pre>
      </div>
    </div>
  {/if}

  <!-- exception -->
  {#if data.exception}
    <div class="border-t border-red-300 dark:border-red-700">
      <div class="flex items-center gap-1.5 bg-red-100 px-3 py-1 dark:bg-red-900/40">
        <svg class="h-3.5 w-3.5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
        <span class="text-xs font-medium text-red-600 dark:text-red-400">Exception</span>
      </div>
      <div class="bg-red-50/50 px-3 py-2 dark:bg-red-900/10">
        <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-red-700 dark:text-red-300">{data.exception}</pre>
      </div>
    </div>
  {/if}

  <!-- Images -->
  {#if data.images_base64?.length}
    <div class="border-t border-gray-200 dark:border-gray-700">
      <div class="flex items-center gap-1.5 bg-gray-100 px-3 py-1 dark:bg-gray-800/80">
        <svg class="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400">
          Images ({data.images_base64.length})
        </span>
      </div>
      <div class="bg-gray-50 p-3 dark:bg-gray-900/50">
        <div class="flex flex-wrap justify-center gap-3">
          {#each data.images_base64 as img, i}
            <div class="group relative overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-gray-700 dark:bg-gray-800">
              <button
                class="block"
                onclick={() => expandedImage = `data:image/png;base64,${img}`}
                title="Click to expand"
                aria-label="Expand image"
              >
                <img
                  src="data:image/png;base64,{img}"
                  alt="Output {i + 1}"
                  class="max-h-64 max-w-xs object-contain"
                />
                <div class="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-all group-hover:bg-black/10 group-hover:opacity-100">
                  <svg class="h-6 w-6 text-white drop-shadow-md" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                  </svg>
                </div>
              </button>
              <button
                class="absolute right-1 top-1 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover:opacity-100"
                onclick={(e) => {
                  e.stopPropagation();
                  const a = document.createElement("a");
                  a.href = `data:image/png;base64,${img}`;
                  a.download = `output-${i + 1}.png`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }}
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
      </div>
    </div>
  {/if}

  <!-- Video frames (animated playback) -->
  {#if data.video_frames_base64?.length}
    <VideoFrames frames={data.video_frames_base64} />
  {/if}
</div>

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
      alt="Expanded output"
      class="relative max-h-full max-w-full rounded-lg shadow-2xl"
    />
    <div class="absolute right-4 top-4 flex gap-2">
      <button
        class="rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
        onclick={() => {
          if (!expandedImage) return;
          const a = document.createElement("a");
          a.href = expandedImage;
          a.download = "output.png";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }}
        title="Download image"
        aria-label="Download image"
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

<style>
  :global(.code-exec-pane .hljs) {
    background: transparent;
    padding: 0;
  }
</style>
