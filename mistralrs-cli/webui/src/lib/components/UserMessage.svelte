<script lang="ts">
  import type { DisplayMessage } from "../types";
  import { renderMarkdown } from "../utils/markdown";

  let { message }: { message: DisplayMessage } = $props();
</script>

<div class="flex justify-end">
  <div class="max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-white shadow-sm">
    <!-- Images -->
    {#if message.images?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.images as img}
          <img
            src={img}
            alt="Attached"
            class="max-h-40 rounded-lg object-contain"
          />
        {/each}
      </div>
    {/if}

    <!-- Videos (rendered as looping GIF-like clips) -->
    {#if message.videos?.length}
      <div class="mb-2 flex flex-wrap gap-2">
        {#each message.videos as vid}
          <video
            src={vid}
            autoplay
            loop
            muted
            playsinline
            class="max-h-48 rounded-lg"
          ></video>
        {/each}
      </div>
    {/if}

    <!-- Text content -->
    <div class="text-sm leading-relaxed [&_a]:underline [&_code]:rounded [&_code]:bg-blue-500/30 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_pre]:my-1 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-blue-500/20 [&_pre]:p-2">
      {@html renderMarkdown(message.content)}
    </div>
  </div>
</div>
