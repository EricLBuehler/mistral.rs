<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";

  let { message }: { message: DisplayMessage } = $props();

  function downloadFile(url: string, fallbackName: string) {
    const a = document.createElement("a");
    a.href = url;
    // Try to extract filename from URL
    const fromUrl = url.split("/").pop()?.split("?")[0];
    a.download = fromUrl || fallbackName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
</script>

<div class="flex justify-end">
  <div class="max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-white shadow-sm">
    <!-- Images -->
    {#if message.images?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.images as img, i}
          <div class="group relative">
            <img
              src={img}
              alt="Attached"
              class="max-h-40 rounded-lg object-contain"
            />
            <button
              class="absolute right-1 top-1 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover:opacity-100"
              onclick={() => downloadFile(img, `image-${i + 1}.png`)}
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
    {/if}

    <!-- Videos (rendered as looping GIF-like clips; .gif files use <img> since <video> can't play them) -->
    {#if message.videos?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.videos as vid, i}
          {@const isGif = vid.toLowerCase().endsWith(".gif")}
          <div class="group relative">
            {#if isGif}
              <img
                src={vid}
                alt="GIF {i + 1}"
                class="max-h-48 rounded-lg"
              />
            {:else}
              <video
                src={vid}
                autoplay
                loop
                muted
                playsinline
                class="max-h-48 rounded-lg"
              ></video>
            {/if}
            <button
              class="absolute right-1 top-1 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity hover:bg-black/80 group-hover:opacity-100"
              onclick={() => downloadFile(vid, isGif ? `video-${i + 1}.gif` : `video-${i + 1}.mp4`)}
              title={isGif ? "Download GIF" : "Download video"}
              aria-label={isGif ? "Download GIF" : "Download video"}
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Text content -->
    <div class="text-sm leading-relaxed [&_a]:underline [&_code]:rounded [&_code]:bg-blue-500/30 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_pre]:my-1 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-blue-500/20 [&_pre]:p-2">
      {@html renderMarkdown(message.content)}
    </div>
  </div>
</div>
