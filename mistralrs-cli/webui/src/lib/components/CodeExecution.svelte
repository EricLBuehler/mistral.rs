<script lang="ts">
  import type { CodeExecutionData } from "../types";
  import hljs from "highlight.js";

  let { data, phase }: { data: CodeExecutionData; phase: "calling" | "complete" } = $props();

  let highlightedCode = $derived(
    data.code
      ? hljs.highlight(data.code, { language: "python" }).value
      : ""
  );

  let executionTimeFormatted = $derived(() => {
    if (data.execution_time_ms == null) return null;
    if (data.execution_time_ms < 1000) return `${data.execution_time_ms}ms`;
    return `${(data.execution_time_ms / 1000).toFixed(2)}s`;
  });
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
      <!-- Stats bar like CLI output -->
      <div class="ml-auto flex items-center gap-3 text-xs text-gray-400 dark:text-gray-500">
        {#if data.images_base64?.length}
          <span>{data.images_base64.length} image{data.images_base64.length !== 1 ? "s" : ""}</span>
        {/if}
        {#if executionTimeFormatted()}
          <span>{executionTimeFormatted()}</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Code -->
  {#if data.code}
    <div class="overflow-x-auto bg-gray-950 p-3">
      <pre class="text-xs leading-relaxed"><code class="hljs language-python">{@html highlightedCode}</code></pre>
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
    <div class="border-t border-purple-200 bg-white p-3 dark:border-purple-800/50 dark:bg-gray-900">
      <div class="grid gap-2" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
        {#each data.images_base64 as img, i}
          <img
            src="data:image/png;base64,{img}"
            alt="Output image {i + 1}"
            class="w-full rounded-lg border border-gray-200 dark:border-gray-700"
          />
        {/each}
      </div>
    </div>
  {/if}
</div>
