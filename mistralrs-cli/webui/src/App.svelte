<script lang="ts">
  import ModelBar from "./lib/components/ModelBar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import MessageList from "./lib/components/MessageList.svelte";
  import ChatInput from "./lib/components/ChatInput.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import ToolsPanel from "./lib/components/ToolsPanel.svelte";
  import Notice from "./lib/components/Notice.svelte";
  import Brand from "./lib/components/Brand.svelte";
  import { chatStore } from "./lib/stores/chat.svelte";
  import { BUTTON_SECONDARY } from "./lib/ui";
  import { modelStore } from "./lib/stores/models.svelte";
  import { settingsStore } from "./lib/stores/settings.svelte";
  import * as api from "./lib/services/api";
  import { onMount } from "svelte";
  import { installCopyHandler } from "./lib/utils/markdown";

  let startupError = $state<string | null>(null);
  let starting = $state(true);

  async function init() {
    starting = true;
    startupError = null;
    try {
      const { models } = await api.listModels();
      modelStore.models = models;

      const settings = await api.getSettings();
      if (settings.model) {
        modelStore.selectModel(settings.model);
      } else if (models.length > 0) {
        modelStore.selectModel(models[0].name);
      }
      if (settings.defaults) {
        settingsStore.applyDefaults(settings.defaults);
      }

      const caps = await api.getCapabilities();
      modelStore.capabilities = caps;
      if (caps.search_enabled) settingsStore.enableSearch = true;
      if (caps.code_execution_enabled) settingsStore.enableCodeExecution = true;
      if (caps.shell_enabled) settingsStore.enableShell = true;
    } catch (e) {
      startupError = e instanceof Error ? e.message : String(e);
    } finally {
      starting = false;
    }
  }

  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  let systemDark = $state(darkQuery.matches);

  $effect(() => {
    const dark = settingsStore.theme === "dark" || (settingsStore.theme === "system" && systemDark);
    document.documentElement.classList.toggle("dark", dark);
  });

  onMount(() => {
    const onChange = (e: MediaQueryListEvent) => (systemDark = e.matches);
    darkQuery.addEventListener("change", onChange);
    installCopyHandler();
    init();
    return () => darkQuery.removeEventListener("change", onChange);
  });

  let empty = $derived(chatStore.messages.length === 0 && !chatStore.isStreaming);
</script>

{#if startupError}
  <div class="flex h-screen items-center justify-center bg-background p-6">
    <div class="max-w-md space-y-4 text-center" role="alert">
      <Brand class="text-2xl" />
      <h1 class="text-base font-medium text-foreground">Can't reach the server</h1>
      <p class="break-words font-mono text-xs text-muted">{startupError}</p>
      <button class={BUTTON_SECONDARY} onclick={init} disabled={starting}>
        {starting ? "Retrying..." : "Retry"}
      </button>
    </div>
  </div>
{:else}
<div class="flex h-screen flex-col overflow-hidden bg-background text-foreground">
  <ModelBar />
  <Notice />

  <div class="flex min-h-0 flex-1">
    {#if settingsStore.sidebarOpen}
      <button
        class="fixed inset-0 z-30 bg-black/40 md:hidden"
        onclick={() => (settingsStore.sidebarOpen = false)}
        aria-label="Close sidebar"
      ></button>
      <div class="fixed top-12 bottom-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-surface md:hidden">
        <Sidebar />
      </div>
      <div class="hidden w-64 shrink-0 flex-col border-r border-line bg-surface md:flex">
        <Sidebar />
      </div>
    {/if}

    <main class="flex min-w-0 flex-1 flex-col">
      {#if empty}
        <div class="flex flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-[10vh]">
          <div class="w-full max-w-3xl">
            <div class="mb-8 text-center">
              <Brand class="text-4xl sm:text-5xl" />
              <p class="mt-3 text-muted">
                {#if modelStore.selectedModel}
                  Chatting with <span class="font-mono text-sm text-muted-strong">{modelStore.selectedModel}</span>
                {:else}
                  No model loaded
                {/if}
              </p>
            </div>
            <ChatInput />
          </div>
        </div>
      {:else}
        <MessageList />
        <div class="mx-auto w-full max-w-3xl shrink-0 px-4 pb-4">
          <ChatInput />
        </div>
      {/if}
    </main>

    {#if settingsStore.settingsOpen}
      <button
        class="fixed inset-0 z-30 bg-black/40"
        onclick={() => (settingsStore.settingsOpen = false)}
        aria-label="Close settings"
      ></button>
      <div class="fixed right-0 top-12 bottom-0 z-40 w-80 overflow-y-auto border-l border-line bg-surface shadow-xl shadow-black/20">
        <SettingsPanel />
      </div>
    {/if}

    {#if settingsStore.toolsOpen}
      <button
        class="fixed inset-0 z-30 bg-black/40"
        onclick={() => (settingsStore.toolsOpen = false)}
        aria-label="Close tools"
      ></button>
      <div class="fixed right-0 top-12 bottom-0 z-40 w-96 overflow-hidden border-l border-line bg-surface shadow-xl shadow-black/20">
        <ToolsPanel />
      </div>
    {/if}
  </div>

</div>
{/if}
