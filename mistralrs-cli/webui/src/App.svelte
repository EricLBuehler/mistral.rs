<script lang="ts">
  import ModelBar from "./lib/components/ModelBar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import MessageList from "./lib/components/MessageList.svelte";
  import ChatInput from "./lib/components/ChatInput.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import ToolsPanel from "./lib/components/ToolsPanel.svelte";
  import StatusBar from "./lib/components/StatusBar.svelte";
  import Notice from "./lib/components/Notice.svelte";
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
</script>

{#if startupError}
  <div class="flex h-screen items-center justify-center bg-white p-6 dark:bg-gray-950">
    <div class="max-w-md space-y-3 text-center" role="alert">
      <h1 class="text-base font-semibold text-gray-900 dark:text-gray-100">Can't reach the mistral.rs server</h1>
      <p class="break-words font-mono text-xs text-gray-500 dark:text-gray-400">{startupError}</p>
      <button
        class="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-60 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
        onclick={init}
        disabled={starting}
      >{starting ? "Retrying..." : "Retry"}</button>
    </div>
  </div>
{:else}
<div class="flex h-screen flex-col overflow-hidden">
  <ModelBar />
  <Notice />

  <div class="flex min-h-0 flex-1">
    {#if settingsStore.sidebarOpen}
      <button
        class="fixed inset-0 z-30 bg-black/30 md:hidden"
        onclick={() => (settingsStore.sidebarOpen = false)}
        aria-label="Close sidebar"
      ></button>
    {/if}

    {#if settingsStore.sidebarOpen}
      <div
        class="fixed top-12 bottom-7 z-40 flex w-72 flex-col border-r border-gray-200 bg-gray-50 md:hidden dark:border-gray-800 dark:bg-gray-900"
      >
        <Sidebar />
      </div>
      <div
        class="hidden w-72 shrink-0 flex-col border-r border-gray-200 bg-gray-50 md:flex dark:border-gray-800 dark:bg-gray-900"
      >
        <Sidebar />
      </div>
    {/if}

    <div class="flex min-w-0 flex-1 flex-col">
      <MessageList />
      <ChatInput />
    </div>

    {#if settingsStore.settingsOpen}
      <button
        class="fixed inset-0 z-30 bg-black/30"
        onclick={() => (settingsStore.settingsOpen = false)}
        aria-label="Close settings"
      ></button>
      <div class="fixed right-0 top-12 bottom-7 z-40 w-80 overflow-y-auto border-l border-gray-200 bg-white shadow-lg dark:border-gray-800 dark:bg-gray-950">
        <SettingsPanel />
      </div>
    {/if}

    {#if settingsStore.toolsOpen}
      <button
        class="fixed inset-0 z-30 bg-black/30"
        onclick={() => (settingsStore.toolsOpen = false)}
        aria-label="Close tools"
      ></button>
      <div class="fixed right-0 top-12 bottom-7 z-40 w-96 overflow-hidden border-l border-gray-200 bg-white shadow-lg dark:border-gray-800 dark:bg-gray-950">
        <ToolsPanel />
      </div>
    {/if}
  </div>

  <StatusBar />
</div>
{/if}
