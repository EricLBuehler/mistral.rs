<script lang="ts">
  import type { CodeExecutionData } from "../types";
  import hljs from "highlight.js";
  import VideoFrames from "./VideoFrames.svelte";
  import { TOOL_CARD } from "../ui";
  import Code from "@lucide/svelte/icons/code";
  import ImageIcon from "@lucide/svelte/icons/image";
  import Video from "@lucide/svelte/icons/video";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import Copy from "@lucide/svelte/icons/copy";
  import Check from "@lucide/svelte/icons/check";
  import TriangleAlert from "@lucide/svelte/icons/triangle-alert";
  import ZoomIn from "@lucide/svelte/icons/zoom-in";
  import Download from "@lucide/svelte/icons/download";
  import X from "@lucide/svelte/icons/x";

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

  const SECTION_LABEL = "flex items-center gap-1.5 border-t border-line px-3 py-1 font-mono text-[11px] text-muted";
  const OVERLAY_BUTTON = "rounded-md bg-black/50 p-2 text-white hover:bg-black/70";

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

<div class={TOOL_CARD}>
  <div class="flex items-center gap-2 px-3 py-1.5">
    <Code class="h-3.5 w-3.5 text-muted" />
    <span class="text-xs font-medium text-muted-strong">Python</span>
    {#if phase === "calling"}
      <span class="ml-auto flex items-center gap-1.5 text-xs text-muted">
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent"></span>
        Running...
      </span>
    {:else}
      <div class="ml-auto flex items-center gap-2.5 font-mono text-[11px] text-muted">
        {#if data.images_base64?.length}
          <span class="flex items-center gap-1">
            <ImageIcon class="h-3 w-3" />
            {data.images_base64.length}
          </span>
        {/if}
        {#if videoFrameCount}
          <span class="flex items-center gap-1">
            <Video class="h-3 w-3" />
            {videoFrameCount}
          </span>
        {/if}
        {#if executionTimeFormatted()}
          <span>{executionTimeFormatted()}</span>
        {/if}
      </div>
    {/if}
  </div>

  {#if data.code}
    <div class="border-t border-line">
      <div class="flex items-center">
        <button
          class="flex flex-1 cursor-pointer items-center gap-1.5 px-3 py-1 text-left text-muted hover:text-foreground"
          onclick={() => codeExpanded = !codeExpanded}
        >
          <ChevronRight class="h-3.5 w-3.5 transition-transform {codeExpanded ? 'rotate-90' : ''}" />
          <span class="text-xs">Code</span>
          <span class="font-mono text-[11px]">({totalLines} lines)</span>
        </button>
        <button
          class="flex items-center gap-1 self-stretch border-l border-line px-3 py-1 text-xs text-muted-strong hover:bg-surface-raised hover:text-foreground"
          onclick={copyCode}
          title="Copy code"
          aria-label="Copy code"
        >
          {#if copyState === "copied"}
            <Check class="h-3.5 w-3.5 text-success" />
            <span>Copied</span>
          {:else}
            <Copy class="h-3.5 w-3.5" />
            <span>Copy</span>
          {/if}
        </button>
      </div>
      {#if codeExpanded}
        <div class="code-exec-pane flex border-t border-line bg-background py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right font-mono text-xs leading-relaxed text-muted/60">{lineNumberGutter(totalLines)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 font-mono text-xs leading-relaxed text-code"><code class="hljs language-python">{@html highlightedFull}</code></pre>
          </div>
        </div>
      {:else if needsExpand}
        <div class="code-exec-pane relative flex border-t border-line bg-background py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right font-mono text-xs leading-relaxed text-muted/60">{lineNumberGutter(previewLineCount)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 font-mono text-xs leading-relaxed text-code"><code class="hljs language-python">{@html highlightedPreview}</code></pre>
          </div>
          <div class="pointer-events-none absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-background to-transparent"></div>
        </div>
      {:else}
        <div class="code-exec-pane flex border-t border-line bg-background py-2">
          <pre class="flex-shrink-0 select-none px-3 text-right font-mono text-xs leading-relaxed text-muted/60">{lineNumberGutter(totalLines)}</pre>
          <div class="min-w-0 flex-1 overflow-x-auto">
            <pre class="pr-3 font-mono text-xs leading-relaxed text-code"><code class="hljs language-python">{@html highlightedFull}</code></pre>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  {#if data.stdout}
    <div class={SECTION_LABEL}>stdout</div>
    <div class="bg-background px-3 py-2">
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-code">{data.stdout}</pre>
    </div>
  {/if}

  {#if data.stderr}
    <div class={SECTION_LABEL}>stderr</div>
    <div class="bg-background px-3 py-2">
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-danger">{data.stderr}</pre>
    </div>
  {/if}

  {#if data.exception}
    <div class="flex items-center gap-1.5 border-t border-line bg-danger-soft px-3 py-1 font-mono text-[11px] text-danger">
      <TriangleAlert class="h-3.5 w-3.5" />
      Exception
    </div>
    <div class="bg-background px-3 py-2">
      <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-danger">{data.exception}</pre>
    </div>
  {/if}

  {#if data.images_base64?.length}
    <div class={SECTION_LABEL}>
      <ImageIcon class="h-3.5 w-3.5" />
      Images ({data.images_base64.length})
    </div>
    <div class="bg-background p-3">
      <div class="flex flex-wrap justify-center gap-3">
        {#each data.images_base64 as img, i}
          <div class="group relative overflow-hidden rounded-md border border-line-strong bg-surface">
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
                <ZoomIn class="h-6 w-6 text-white drop-shadow-md" />
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
              <Download class="h-3.5 w-3.5" />
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if data.video_frames_base64?.length}
    <VideoFrames frames={data.video_frames_base64} />
  {/if}
</div>

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
      class="relative max-h-full max-w-full rounded-md"
    />
    <div class="absolute right-4 top-4 flex gap-2">
      <button
        class={OVERLAY_BUTTON}
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

<style>
  :global(.code-exec-pane .hljs) {
    background: transparent;
    padding: 0;
  }
</style>
