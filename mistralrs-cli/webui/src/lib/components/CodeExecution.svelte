<script lang="ts">
  import type { CodeExecutionData } from "../types";
  import hljs from "highlight.js";

  let { data, phase }: { data: CodeExecutionData; phase: "calling" | "complete" } = $props();

  const PREVIEW_LINES = 10;

  let codeLines = $derived(data.code?.split("\n") ?? []);
  let totalLines = $derived(codeLines.length);
  let needsExpand = $derived(totalLines > PREVIEW_LINES);
  let previewCode = $derived(
    needsExpand ? codeLines.slice(0, PREVIEW_LINES).join("\n") : (data.code ?? "")
  );

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
        {#if executionTimeFormatted()}
          <span>{executionTimeFormatted()}</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Code section -->
  {#if data.code}
    <div class="border-t border-gray-700">
      <div class="flex items-center gap-1.5 bg-gray-900 px-3 py-1">
        <span class="text-xs font-medium text-gray-400">Code</span>
        <span class="text-xs text-gray-500">({totalLines} lines)</span>
      </div>
      <div class="overflow-x-auto bg-gray-950 px-3 py-2">
        {#if codeExpanded || !needsExpand}
          <pre class="text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedFull}</code></pre>
        {:else}
          <pre class="text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedPreview}</code></pre>
          <button
            class="mt-1 flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300"
            onclick={() => codeExpanded = true}
          >
            <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
            Show all {totalLines} lines
          </button>
        {/if}
      </div>
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
            <button
              class="group relative overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-gray-700 dark:bg-gray-800"
              onclick={() => expandedImage = `data:image/png;base64,${img}`}
              title="Click to expand"
            >
              <img
                src="data:image/png;base64,{img}"
                alt="Output {i + 1}"
                class="max-h-64 max-w-xs object-contain"
              />
              <div class="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-all group-hover:bg-black/10 group-hover:opacity-100">
                <svg class="h-6 w-6 text-white drop-shadow-md" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                </svg>
              </div>
            </button>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>

<!-- Expanded image overlay -->
{#if expandedImage}
  <button
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-8 backdrop-blur-sm"
    onclick={() => expandedImage = null}
    aria-label="Close expanded image"
  >
    <img
      src={expandedImage}
      alt="Expanded output"
      class="max-h-full max-w-full rounded-lg shadow-2xl"
    />
    <div class="absolute right-4 top-4 rounded-full bg-black/50 p-2 text-white">
      <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </div>
  </button>
{/if}
