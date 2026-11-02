<script lang="ts">
  import type { AgentToolApprovalBlock } from "../types";
  import { chatStore } from "../stores/chat.svelte";
  import { BUTTON_PRIMARY, BUTTON_SECONDARY, FIELD } from "../ui";
  import ShieldAlert from "@lucide/svelte/icons/shield-alert";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";

  let { data }: { data: AgentToolApprovalBlock } = $props();

  let denyMessage = $state("");
  let showArguments = $state(false);

  function getShellCommands(args: Record<string, unknown>): string[] {
    const commands = args.commands;
    if (!Array.isArray(commands)) return [];
    return commands.filter((command): command is string => typeof command === "string");
  }

  let argumentsJson = $derived(JSON.stringify(data.arguments ?? {}, null, 2));
  let shellCommands = $derived(
    data.tool.kind === "shell" ? getShellCommands(data.arguments ?? {}) : []
  );
  let shellCommandText = $derived(shellCommands.map((command) => `$ ${command}`).join("\n"));
  let isBusy = $derived(data.status === "submitting");
  let isResolved = $derived(data.status === "approved" || data.status === "denied");
  let showDetailsToggle = $derived(
    data.tool.source !== "built_in" ||
      (data.tool.kind !== "code_execution" && data.tool.kind !== "shell")
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

<div class="w-full overflow-hidden rounded-md border border-line-strong border-l-2 border-l-accent bg-surface">
  <div class="flex flex-wrap items-center gap-2 px-3 py-2">
    <ShieldAlert class="h-4 w-4 shrink-0 text-accent" />
    <div class="min-w-0 flex-1">
      <div class="truncate text-sm font-medium text-foreground">
        Approval required: {data.tool.label}
      </div>
      <div class="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[11px] text-muted">
        <span>{data.tool.source}</span>
        <span>{data.tool.kind}</span>
        <span>round {data.round}</span>
        {#if data.status !== "pending"}
          <span class="text-muted-strong">{data.status}</span>
        {/if}
      </div>
    </div>
  </div>

  {#if data.error}
    <div class="border-t border-danger/30 bg-danger-soft px-3 py-2 text-xs text-danger">
      {data.error}
    </div>
  {/if}

  <div class="space-y-2 border-t border-line px-3 py-2.5">
    {#if shellCommands.length}
      <div class="overflow-hidden rounded-md border border-line">
        <div class="border-b border-line px-3 py-1 font-mono text-[11px] text-muted">Commands</div>
        <pre class="max-h-44 overflow-auto bg-background px-3 py-2 font-mono text-xs leading-relaxed text-code">{shellCommandText}</pre>
      </div>
    {/if}

    {#if showDetailsToggle}
      <button
        type="button"
        class="flex items-center gap-1.5 text-xs text-muted hover:text-foreground"
        onclick={() => (showArguments = !showArguments)}
      >
        <ChevronRight class="h-3.5 w-3.5 transition-transform {showArguments ? 'rotate-90' : ''}" />
        Details
      </button>
    {/if}

    {#if showDetailsToggle && showArguments}
      <pre class="max-h-56 overflow-auto rounded-md border border-line bg-background p-2 font-mono text-[11px] leading-relaxed text-code">{argumentsJson}</pre>
    {/if}

    {#if !isResolved}
      <input
        class="{FIELD} w-full px-2 py-1.5 text-xs"
        placeholder="Optional deny message"
        bind:value={denyMessage}
        disabled={isBusy}
      />
    {/if}

    <div class="flex flex-wrap items-center gap-2">
      {#if isResolved}
        <div class="text-xs text-muted">
          {#if data.status === "approved"}
            Approved{data.remember_for_session ? " for this session" : ""}.
          {:else}
            Denied{data.message ? `: ${data.message}` : "."}
          {/if}
        </div>
      {:else}
        <button
          type="button"
          class="{BUTTON_PRIMARY} disabled:cursor-not-allowed"
          onclick={() => approve(false)}
          disabled={isBusy}
        >
          Approve
        </button>
        <button
          type="button"
          class="{BUTTON_SECONDARY} disabled:cursor-not-allowed"
          onclick={() => approve(true)}
          disabled={isBusy}
          title="Approve and remember for this session"
        >
          Always
        </button>
        <button
          type="button"
          class="{BUTTON_SECONDARY} hover:text-danger disabled:cursor-not-allowed"
          onclick={deny}
          disabled={isBusy}
        >
          Deny
        </button>
        {#if isBusy}
          <span class="text-xs text-muted">Submitting...</span>
        {/if}
      {/if}
    </div>
  </div>
</div>
