<script lang="ts">
  import type { CodeExecutionData } from "../types";
  import hljs from "highlight.js";

  let { data, phase }: { data: CodeExecutionData; phase: "calling" | "complete" } = $props();

  let highlightedCode = $derived(
    data.code
      ? hljs.highlight(data.code, { language: "python" }).value
      : ""
  );
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
    {:else if data.execution_time_ms != null}
      <span class="ml-auto text-xs text-gray-400">{data.execution_time_ms}ms</span>
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
    <div class="border-t border-purple-200 bg-gray-50 px-3 py-2 dark:border-purple-800/50 dark:bg-gray-900">
      <div class="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">Output</div>
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-700 dark:text-gray-300">{data.stdout}</pre>
    </div>
  {/if}

  <!-- stderr / exception -->
  {#if data.stderr || data.exception}
    <div class="border-t border-red-200 bg-red-50 px-3 py-2 dark:border-red-800/50 dark:bg-red-900/20">
      <div class="mb-1 text-xs font-medium text-red-600 dark:text-red-400">
        {data.exception ? "Error" : "stderr"}
      </div>
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-red-700 dark:text-red-300">{data.exception || data.stderr}</pre>
    </div>
  {/if}

  <!-- Images -->
  {#if data.images_base64?.length}
    <div class="border-t border-purple-200 bg-white p-3 dark:border-purple-800/50 dark:bg-gray-900">
      <div class="flex flex-wrap gap-2">
        {#each data.images_base64 as img}
          <img
            src="data:image/png;base64,{img}"
            alt="Code output"
            class="max-w-full rounded-lg border border-gray-200 dark:border-gray-700"
          />
        {/each}
      </div>
    </div>
  {/if}
</div>
