<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdownBlocks } from "../utils/markdown";
  import ThinkingBlock from "./ThinkingBlock.svelte";
  import CodeExecution from "./CodeExecution.svelte";
  import ShellTool from "./ShellTool.svelte";
  import SearchResult from "./SearchResult.svelte";
  import CustomTool from "./CustomTool.svelte";
  import FileCard from "./FileCard.svelte";
  import AgentApproval from "./AgentApproval.svelte";
  import { chatStore } from "../stores/chat.svelte";
  import { MESSAGE_ACTION } from "../ui";
  import Copy from "@lucide/svelte/icons/copy";
  import Check from "@lucide/svelte/icons/check";
  import RefreshCw from "@lucide/svelte/icons/refresh-cw";
  import ChevronLeft from "@lucide/svelte/icons/chevron-left";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import CircleAlert from "@lucide/svelte/icons/circle-alert";

  let { message, streaming = false }: { message: DisplayMessage; streaming?: boolean } = $props();

  let renderedBlocks = $derived(message.blocks ?? []);
  let copied = $state(false);
  let sibInfo = $derived(message.id ? chatStore.siblingInfo(message.id) : null);

  function regenerate() {
    if (!message.id) return;
    chatStore.regenerateAssistant(message.id);
  }

  function prevBranch() {
    if (!sibInfo) return;
    const next = sibInfo.siblings[(sibInfo.index - 1 + sibInfo.total) % sibInfo.total];
    chatStore.switchBranch(next);
  }

  function nextBranch() {
    if (!sibInfo) return;
    const next = sibInfo.siblings[(sibInfo.index + 1) % sibInfo.total];
    chatStore.switchBranch(next);
  }

  function textForCopy(): string {
    const parts: string[] = [];
    for (const b of renderedBlocks) {
      if (b.type === "content") parts.push(b.content);
      else if (b.type === "reasoning") parts.push(`[reasoning]\n${b.content}`);
    }
    return parts.join("\n\n").trim() || message.content;
  }

  async function copyContent() {
    try {
      await navigator.clipboard.writeText(textForCopy());
      copied = true;
      setTimeout(() => (copied = false), 1200);
    } catch {
      // ignore
    }
  }

  // Only reasons worth calling out; a normal "stop" stays silent.
  function finishReasonNote(reason: string): { label: string; tone: string } | null {
    switch (reason) {
      case "stop":
      case "tool_calls":
        return null;
      case "length":
        return { label: "hit max tokens", tone: "text-warning" };
      case "content_filter":
        return { label: "content filter", tone: "text-danger" };
      default:
        return { label: reason, tone: "text-muted" };
    }
  }

  function fmtElapsed(ms?: number): string | null {
    if (ms == null) return null;
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function fmtDecodeRate(m: DisplayMessage): string | null {
    if (m.decodeTokPerSec) return `${m.decodeTokPerSec.toFixed(1)} tok/s`;
    if (!m.tokens || !m.elapsedMs) return null;
    const decodeMs = m.elapsedMs - (m.ttftMs ?? 0);
    if (decodeMs <= 0) return null;
    return `~${(m.tokens / (decodeMs / 1000)).toFixed(1)} tok/s`;
  }

  let decodeRate = $derived(fmtDecodeRate(message));
  let lastBlock = $derived(renderedBlocks[renderedBlocks.length - 1]);
  let waiting = $derived(streaming && lastBlock?.type !== "content" && lastBlock?.type !== "reasoning");
  let finishNote = $derived(message.finishReason ? finishReasonNote(message.finishReason) : null);
</script>

<div class="group w-full">
  <div class="space-y-3">
    {#each renderedBlocks as block, i (i)}
      {#if block.type === "reasoning"}
        <ThinkingBlock reasoning={block.content} streaming={streaming && i === renderedBlocks.length - 1} />
      {:else if block.type === "tool_call"}
        {#if block.data.data.tool_type === "code_execution"}
          <CodeExecution data={block.data.data} phase={block.data.phase} />
        {:else if block.data.data.tool_type === "shell"}
          <ShellTool data={block.data.data} phase={block.data.phase} />
        {:else if block.data.data.tool_type === "web_search"}
          <SearchResult data={block.data.data} phase={block.data.phase} />
        {:else if block.data.data.tool_type === "custom"}
          <CustomTool data={block.data.data} phase={block.data.phase} toolName={block.data.tool_name} />
        {/if}
      {:else if block.type === "content"}
        <div class="markdown-content text-[15px]">
          {#each renderMarkdownBlocks(block.content, streaming && i === renderedBlocks.length - 1) as html, j (j)}
            {@html html}
          {/each}
        </div>
      {:else if block.type === "file"}
        <FileCard file={block.data} />
      {:else if block.type === "approval"}
        <AgentApproval data={block.data} />
      {/if}
    {/each}

    {#if message.error}
      <div class="flex gap-2.5 rounded-md border border-danger/30 bg-danger-soft px-3 py-2.5 text-sm" role="alert">
        <CircleAlert class="mt-0.5 h-4 w-4 shrink-0 text-danger" />
        <div class="min-w-0">
          <div class="font-medium text-danger">Request failed</div>
          <div class="mt-0.5 whitespace-pre-wrap break-words font-mono text-xs text-muted-strong">{message.error}</div>
        </div>
      </div>
    {/if}

    {#if waiting}
      <div class="flex h-6 items-center gap-1" role="status" aria-label="Waiting for the model">
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent"></span>
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent [animation-delay:150ms]"></span>
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent [animation-delay:300ms]"></span>
      </div>
    {/if}
  </div>

  {#if !streaming && (message.finishReason || message.tokens || message.elapsedMs || message.error)}
    <div class="mt-2 flex flex-wrap items-center gap-x-1 gap-y-1">
      {#if sibInfo}
        <div class="flex items-center font-mono text-xs text-muted">
          <button class={MESSAGE_ACTION} onclick={prevBranch} aria-label="Previous branch" title="Previous branch"><ChevronLeft class="h-3.5 w-3.5" /></button>
          <span class="tabular-nums">{sibInfo.index + 1}/{sibInfo.total}</span>
          <button class={MESSAGE_ACTION} onclick={nextBranch} aria-label="Next branch" title="Next branch"><ChevronRight class="h-3.5 w-3.5" /></button>
        </div>
      {/if}
      <button class={MESSAGE_ACTION} onclick={copyContent} title="Copy message" aria-label="Copy message">
        {#if copied}<Check class="h-3.5 w-3.5 text-success" />{:else}<Copy class="h-3.5 w-3.5" />{/if}
      </button>
      <button class={MESSAGE_ACTION} onclick={regenerate} disabled={chatStore.isStreaming} title="Regenerate (creates a new branch)" aria-label="Regenerate response">
        <RefreshCw class="h-3.5 w-3.5" />
      </button>

      <div class="ml-2 flex flex-wrap items-center gap-x-3 font-mono text-xs tabular-nums text-muted">
        {#if decodeRate}<span>{decodeRate}</span>{/if}
        {#if message.tokens}<span>{message.tokens} tok</span>{/if}
        {#if fmtElapsed(message.ttftMs)}<span>ttft {fmtElapsed(message.ttftMs)}</span>{/if}
        {#if message.promptTokens}
          <span title="Prompt tokens{message.cachedPromptTokens ? `, ${message.cachedPromptTokens} reused from the prefix cache` : ''}">
            prompt {message.promptTokens}{#if message.cachedPromptTokens}<span class="text-accent"> ({message.cachedPromptTokens} cached)</span>{/if}
          </span>
        {/if}
        {#if finishNote}<span class={finishNote.tone}>{finishNote.label}</span>{/if}
      </div>
    </div>
  {/if}
</div>
