<script lang="ts">
  import { onDestroy } from "svelte";
  import { FIELD } from "../ui";
  import Video from "@lucide/svelte/icons/video";
  import Play from "@lucide/svelte/icons/play";
  import Pause from "@lucide/svelte/icons/pause";
  import Download from "@lucide/svelte/icons/download";

  const CONTROL = "inline-flex items-center rounded-md border border-line-strong px-2 py-1 text-muted-strong hover:bg-surface-raised hover:text-foreground";

  let { frames }: { frames: string[] } = $props();

  let currentFrame = $state(0);
  let playing = $state(true);
  let fps = $state(10);
  let intervalId: ReturnType<typeof setInterval> | null = null;

  function tick() {
    currentFrame = (currentFrame + 1) % frames.length;
  }

  function startPlaying() {
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(tick, 1000 / fps);
  }

  function stopPlaying() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  $effect(() => {
    if (playing && frames.length > 1) {
      startPlaying();
    } else {
      stopPlaying();
    }
    return () => stopPlaying();
  });

  onDestroy(() => stopPlaying());

  function togglePlayPause() {
    playing = !playing;
  }

  function downloadCurrentFrame() {
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${frames[currentFrame]}`;
    a.download = `frame-${currentFrame + 1}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  async function downloadAsGif() {
    // Build a simple animated PNG-like sequence by zipping frames into a single download
    // For true GIF we'd need a JS encoder. Instead, give the user the first frame.
    // Better: trigger one download per frame is annoying. Just download a zip-like manifest? No.
    // Simplest practical solution: download each frame sequentially (rare action).
    for (let i = 0; i < frames.length; i++) {
      const a = document.createElement("a");
      a.href = `data:image/png;base64,${frames[i]}`;
      a.download = `frame-${String(i + 1).padStart(4, "0")}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Small delay so browsers don't choke
      await new Promise((r) => setTimeout(r, 50));
    }
  }
</script>

<div class="border-t border-line">
  <div class="flex items-center gap-1.5 px-3 py-1 font-mono text-[11px] text-muted">
    <Video class="h-3.5 w-3.5" />
    <span>
      Animation ({frames.length} frame{frames.length !== 1 ? "s" : ""})
    </span>
    <span class="ml-auto tabular-nums">
      {currentFrame + 1} / {frames.length}
    </span>
  </div>
  <div class="bg-background p-3">
    <div class="flex flex-col items-center gap-2">
      <div class="relative">
        <img
          src="data:image/png;base64,{frames[currentFrame]}"
          alt="Frame {currentFrame + 1}"
          class="max-h-80 max-w-full rounded-md border border-line bg-white"
        />
      </div>

      <div class="flex flex-wrap items-center justify-center gap-3 text-xs text-muted">
        <button
          class={CONTROL}
          onclick={togglePlayPause}
          title={playing ? "Pause" : "Play"}
          aria-label={playing ? "Pause" : "Play"}
        >
          {#if playing}
            <Pause class="h-3.5 w-3.5" />
          {:else}
            <Play class="h-3.5 w-3.5" />
          {/if}
        </button>

        <input
          type="range"
          min="0"
          max={frames.length - 1}
          bind:value={currentFrame}
          oninput={() => (playing = false)}
          class="w-32 accent-[var(--accent)]"
          aria-label="Frame slider"
        />

        <label class="flex items-center gap-1">
          <span class="font-mono text-[11px]">FPS</span>
          <input
            type="number"
            min="1"
            max="60"
            bind:value={fps}
            onchange={() => playing && startPlaying()}
            class="{FIELD} w-12 px-1 py-0.5 text-right font-mono text-xs"
          />
        </label>

        <button
          class={CONTROL}
          onclick={downloadCurrentFrame}
          title="Download current frame"
        >
          <Download class="h-3.5 w-3.5" />
        </button>
        <button
          class="{CONTROL} text-[11px]"
          onclick={downloadAsGif}
          title="Download all frames as PNG sequence"
        >
          All frames
        </button>
      </div>
    </div>
  </div>
</div>
