<script lang="ts">
  import ModelBar from "./lib/components/ModelBar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import MessageList from "./lib/components/MessageList.svelte";
  import ChatInput from "./lib/components/ChatInput.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import ToolsPanel from "./lib/components/ToolsPanel.svelte";
  import StatusBar from "./lib/components/StatusBar.svelte";
  import { chatStore } from "./lib/stores/chat.svelte";
  import { modelStore } from "./lib/stores/models.svelte";
  import { settingsStore } from "./lib/stores/settings.svelte";
  import * as api from "./lib/services/api";
  import { onMount } from "svelte";

  onMount(async () => {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }

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

      try {
        const caps = await api.getCapabilities();
        modelStore.capabilities = caps;
        if (caps.search_enabled) settingsStore.enableSearch = true;
        if (caps.code_execution_enabled) settingsStore.enableCodeExecution = true;
      } catch {
        // Capabilities endpoint might not exist on older servers
      }
    } catch (e) {
      console.error("Failed to initialize:", e);
    }
  });
</script>

<div class="flex h-screen flex-col overflow-hidden">
  <ModelBar />

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
