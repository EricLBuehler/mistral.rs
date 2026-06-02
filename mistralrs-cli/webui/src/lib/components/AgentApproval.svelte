<script lang="ts">
  import type { AgentToolApprovalBlock } from "../types";
  import { chatStore } from "../stores/chat.svelte";

  let { data }: { data: AgentToolApprovalBlock } = $props();

  let denyMessage = $state("");
  let showArguments = $state(false);

  let argumentsJson = $derived(JSON.stringify(data.arguments ?? {}, null, 2));
  let isBusy = $derived(data.status === "submitting");
  let isResolved = $derived(data.status === "approved" || data.status === "denied");
  let showDetailsToggle = $derived(
    data.tool.source !== "built_in" || data.tool.kind !== "code_execution"
  );

  function approve(rememberForSession = false) {
    chatStore.resolveAgentApproval(data.approval_id, "approve", rememberForSession);
  }

  function deny() {
    chatStore.resolveAgentApproval(
      data.approval_id,
      "deny",
      false,
      denyMessage.trim() || undefined,
    );
  }
</script>

<div class="w-full overflow-hidden rounded-xl border border-amber-200 bg-white shadow-sm dark:border-amber-800/60 dark:bg-gray-900">
  <div class="flex flex-wrap items-center gap-2 bg-amber-50 px-3 py-2 dark:bg-amber-900/20">
    <svg class="h-4 w-4 shrink-0 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v3.75m0 3.75h.01M4.217 19.5h15.566a1.5 1.5 0 001.299-2.25L13.299 3.75a1.5 1.5 0 00-2.598 0L2.918 17.25A1.5 1.5 0 004.217 19.5z" />
    </svg>
    <div class="min-w-0 flex-1">
      <div class="truncate text-sm font-medium text-amber-800 dark:text-amber-200">
        Approval required: {data.tool.label}
      </div>
      <div class="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-amber-700/80 dark:text-amber-300/80">
        <span>{data.tool.source}</span>
        <span>{data.tool.kind}</span>
        <span>round {data.round}</span>
        {#if data.status !== "pending"}
          <span>{data.status}</span>
        {/if}
      </div>
    </div>
  </div>

  {#if data.error}
    <div class="border-t border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-800/50 dark:bg-red-900/20 dark:text-red-300">
      {data.error}
    </div>
  {/if}

  <div class="space-y-2 border-t border-gray-200 px-3 py-2 dark:border-gray-800">
    {#if showDetailsToggle}
      <button
        type="button"
        class="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-100"
        onclick={() => (showArguments = !showArguments)}
      >
        <svg class="h-3.5 w-3.5 transition-transform {showArguments ? 'rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
        </svg>
        Details
      </button>
    {/if}

    {#if showDetailsToggle && showArguments}
      <pre class="max-h-56 overflow-auto rounded-md bg-gray-100 p-2 font-mono text-[11px] leading-relaxed text-gray-700 dark:bg-gray-950 dark:text-gray-300">{argumentsJson}</pre>
    {/if}

    {#if !isResolved}
      <input
        class="w-full rounded-md border border-gray-200 bg-gray-50 px-2 py-1.5 text-xs text-gray-700 outline-none placeholder:text-gray-400 focus:border-amber-400 focus:ring-1 focus:ring-amber-400 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-200 dark:placeholder:text-gray-500"
        placeholder="Optional deny message"
        bind:value={denyMessage}
        disabled={isBusy}
      />
    {/if}

    <div class="flex flex-wrap items-center gap-2">
      {#if isResolved}
        <div class="text-xs text-gray-500 dark:text-gray-400">
          {#if data.status === "approved"}
            Approved{data.remember_for_session ? " for this session" : ""}.
          {:else}
            Denied{data.message ? `: ${data.message}` : "."}
          {/if}
        </div>
      {:else}
        <button
          type="button"
          class="inline-flex items-center gap-1 rounded-md bg-green-600 px-2.5 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-60"
          onclick={() => approve(false)}
          disabled={isBusy}
        >
          Approve
        </button>
        <button
          type="button"
          class="inline-flex items-center gap-1 rounded-md border border-green-200 px-2.5 py-1.5 text-xs font-medium text-green-700 hover:bg-green-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-green-800 dark:text-green-300 dark:hover:bg-green-900/20"
          onclick={() => approve(true)}
          disabled={isBusy}
        >
          Always
        </button>
        <button
          type="button"
          class="inline-flex items-center gap-1 rounded-md border border-red-200 px-2.5 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-red-800 dark:text-red-300 dark:hover:bg-red-900/20"
          onclick={deny}
          disabled={isBusy}
        >
          Deny
        </button>
        {#if isBusy}
          <span class="text-xs text-gray-500 dark:text-gray-400">Submitting...</span>
        {/if}
      {/if}
    </div>
  </div>
</div>
