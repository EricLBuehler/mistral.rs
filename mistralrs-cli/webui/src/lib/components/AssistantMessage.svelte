<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";
  import ThinkingBlock from "./ThinkingBlock.svelte";
  import CodeExecution from "./CodeExecution.svelte";
  import SearchResult from "./SearchResult.svelte";
  import CustomTool from "./CustomTool.svelte";

  let { message, streaming = false }: { message: DisplayMessage; streaming?: boolean } = $props();
</script>

<div class="flex justify-start">
  <div class="max-w-[85%] space-y-2">
    <!-- Ordered blocks: reasoning and tool calls in sequence -->
    {#if message.blocks?.length}
      {#each message.blocks as block}
        {#if block.type === "reasoning"}
          <ThinkingBlock reasoning={block.content} streaming={streaming && block === message.blocks![message.blocks!.length - 1]} />
        {:else if block.type === "tool_call"}
          {#if block.data.data.tool_type === "code_execution"}
            <CodeExecution data={block.data.data} phase={block.data.phase} />
          {:else if block.data.data.tool_type === "web_search"}
            <SearchResult data={block.data.data} phase={block.data.phase} />
          {:else if block.data.data.tool_type === "custom"}
            <CustomTool data={block.data.data} phase={block.data.phase} toolName={block.data.tool_name} />
          {/if}
        {/if}
      {/each}
    {/if}

    <!-- Main content -->
    {#if message.content}
      <div class="rounded-2xl rounded-bl-md bg-gray-100 px-4 py-2.5 shadow-sm dark:bg-gray-800">
        <div class="markdown-content text-sm leading-relaxed text-gray-900 dark:text-gray-100">
          {@html renderMarkdown(message.content)}
        </div>
      </div>
    {/if}

    <!-- Streaming cursor when nothing has appeared yet -->
    {#if streaming && !message.content && !message.blocks?.length}
      <div class="rounded-2xl rounded-bl-md bg-gray-100 px-4 py-3 dark:bg-gray-800">
        <div class="flex items-center gap-1.5">
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]"></span>
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]"></span>
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"></span>
        </div>
      </div>
    {/if}
  </div>
</div>
