<script lang="ts">
  import { onDestroy } from "svelte";

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

<div class="border-t border-gray-200 dark:border-gray-700">
  <div class="flex items-center gap-1.5 bg-gray-100 px-3 py-1 dark:bg-gray-800/80">
    <svg class="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
    <span class="text-xs font-medium text-gray-500 dark:text-gray-400">
      Animation ({frames.length} frame{frames.length !== 1 ? "s" : ""})
    </span>
    <span class="ml-auto text-xs text-gray-400 dark:text-gray-500">
      {currentFrame + 1} / {frames.length}
    </span>
  </div>
  <div class="bg-gray-50 p-3 dark:bg-gray-900/50">
    <div class="flex flex-col items-center gap-2">
      <div class="relative">
        <img
          src="data:image/png;base64,{frames[currentFrame]}"
          alt="Frame {currentFrame + 1}"
          class="max-h-80 max-w-full rounded-lg border border-gray-200 bg-white dark:border-gray-700"
        />
      </div>

      <!-- Controls -->
      <div class="flex items-center gap-3 text-xs text-gray-600 dark:text-gray-400">
        <button
          class="rounded-md bg-gray-200 px-2 py-1 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600"
          onclick={togglePlayPause}
          title={playing ? "Pause" : "Play"}
          aria-label={playing ? "Pause" : "Play"}
        >
          {#if playing}
            <svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="5" width="4" height="14" rx="1" />
              <rect x="14" y="5" width="4" height="14" rx="1" />
            </svg>
          {:else}
            <svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          {/if}
        </button>

        <input
          type="range"
          min="0"
          max={frames.length - 1}
          bind:value={currentFrame}
          oninput={() => (playing = false)}
          class="w-32 accent-blue-600"
          aria-label="Frame slider"
        />

        <label class="flex items-center gap-1">
          <span>FPS</span>
          <input
            type="number"
            min="1"
            max="60"
            bind:value={fps}
            onchange={() => playing && startPlaying()}
            class="w-12 rounded border border-gray-300 bg-white px-1 py-0.5 text-right text-xs dark:border-gray-600 dark:bg-gray-800"
          />
        </label>

        <button
          class="rounded-md bg-gray-200 px-2 py-1 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600"
          onclick={downloadCurrentFrame}
          title="Download current frame"
        >
          <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
        <button
          class="rounded-md bg-gray-200 px-2 py-1 text-[10px] hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600"
          onclick={downloadAsGif}
          title="Download all frames as PNG sequence"
        >
          All frames
        </button>
      </div>
    </div>
  </div>
</div>
