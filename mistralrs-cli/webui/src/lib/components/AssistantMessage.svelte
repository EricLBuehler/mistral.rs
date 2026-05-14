<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";
  import ThinkingBlock from "./ThinkingBlock.svelte";
  import CodeExecution from "./CodeExecution.svelte";
  import SearchResult from "./SearchResult.svelte";
  import CustomTool from "./CustomTool.svelte";
  import FileCard from "./FileCard.svelte";

  let { message, streaming = false }: { message: DisplayMessage; streaming?: boolean } = $props();

  let renderedBlocks = $derived(message.blocks ?? []);

  function finishReasonStyle(reason: string): { label: string; color: string } {
    switch (reason) {
      case "stop":
        return { label: "stop", color: "text-gray-400 dark:text-gray-500" };
      case "length":
        return { label: "max tokens reached", color: "text-amber-500 dark:text-amber-400" };
      case "tool_calls":
        return { label: "tool call", color: "text-blue-500 dark:text-blue-400" };
      case "content_filter":
        return { label: "content filter", color: "text-red-500 dark:text-red-400" };
      default:
        return { label: reason, color: "text-gray-400 dark:text-gray-500" };
    }
  }

  function fmtElapsed(ms?: number): string | null {
    if (ms == null) return null;
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function fmtDecodeRate(tokens?: number, elapsedMs?: number, ttftMs?: number): string | null {
    if (!tokens || !elapsedMs) return null;
    const decodeMs = elapsedMs - (ttftMs ?? 0);
    if (decodeMs <= 0) return null;
    const r = tokens / (decodeMs / 1000);
    return `${r.toFixed(1)} tok/s`;
  }
</script>

<div class="flex justify-start">
  <div class="max-w-[85%] space-y-2">
    <!-- Ordered blocks: content, reasoning, and tool calls in arrival sequence -->
    {#each renderedBlocks as block, i (i)}
      {#if block.type === "reasoning"}
        <ThinkingBlock reasoning={block.content} streaming={streaming && block === renderedBlocks[renderedBlocks.length - 1]} />
      {:else if block.type === "tool_call"}
        {#if block.data.data.tool_type === "code_execution"}
          <CodeExecution data={block.data.data} phase={block.data.phase} />
        {:else if block.data.data.tool_type === "web_search"}
          <SearchResult data={block.data.data} phase={block.data.phase} />
        {:else if block.data.data.tool_type === "custom"}
          <CustomTool data={block.data.data} phase={block.data.phase} toolName={block.data.tool_name} />
        {/if}
      {:else if block.type === "content"}
        <div class="rounded-2xl rounded-bl-md bg-gray-100 px-4 py-2.5 shadow-sm dark:bg-gray-800">
          <div class="markdown-content text-sm leading-relaxed text-gray-900 dark:text-gray-100">
            {@html renderMarkdown(block.content)}
          </div>
        </div>
      {:else if block.type === "file"}
        <FileCard file={block.data} />
      {/if}
    {/each}

    <!-- "Working..." spinner: shown until end/stop signal received -->
    {#if streaming}
      <div class="flex items-center gap-2 px-1 text-xs text-gray-500 dark:text-gray-400">
        <svg class="h-3.5 w-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v3a5 5 0 00-5 5H4z"></path>
        </svg>
        <span>Working...</span>
      </div>
    {/if}

    <!-- Per-message footer (only shown after streaming completes) -->
    {#if !streaming && (message.finishReason || message.tokens || message.elapsedMs)}
      <div class="flex flex-wrap items-center gap-x-3 gap-y-1 px-1 font-mono text-[11px] text-gray-400 dark:text-gray-500">
        {#if message.tokens}
          <span><span class="text-gray-500 dark:text-gray-400">{message.tokens}</span> tok</span>
        {/if}
        {#if fmtDecodeRate(message.tokens, message.elapsedMs, message.ttftMs)}
          <span class="text-gray-500 dark:text-gray-400">{fmtDecodeRate(message.tokens, message.elapsedMs, message.ttftMs)}</span>
        {/if}
        {#if fmtElapsed(message.ttftMs)}
          <span><span class="text-gray-400 dark:text-gray-500">ttft</span> {fmtElapsed(message.ttftMs)}</span>
        {/if}
        {#if fmtElapsed(message.elapsedMs)}
          <span>{fmtElapsed(message.elapsedMs)}</span>
        {/if}
        {#if message.finishReason}
          {@const style = finishReasonStyle(message.finishReason)}
          <span class="{style.color}">· {style.label}</span>
        {/if}
      </div>
    {/if}
  </div>
</div>
