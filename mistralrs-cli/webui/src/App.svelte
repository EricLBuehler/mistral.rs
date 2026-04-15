<script lang="ts">
  import ModelBar from "./lib/components/ModelBar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import MessageList from "./lib/components/MessageList.svelte";
  import ChatInput from "./lib/components/ChatInput.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import { chatStore } from "./lib/stores/chat.svelte";
  import { modelStore } from "./lib/stores/models.svelte";
  import { settingsStore } from "./lib/stores/settings.svelte";
  import * as api from "./lib/services/api";
  import { onMount } from "svelte";

  // Initialize on mount
  onMount(async () => {
    // Detect dark mode
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }

    try {
      // Load models
      const { models } = await api.listModels();
      modelStore.models = models;

      // Load settings (includes selected model)
      const settings = await api.getSettings();
      if (settings.model) {
        modelStore.selectModel(settings.model);
      } else if (models.length > 0) {
        modelStore.selectModel(models[0].name);
      }

      // Apply generation defaults
      if (settings.defaults) {
        settingsStore.applyDefaults(settings.defaults);
      }

      // Load capabilities
      try {
        const caps = await api.getCapabilities();
        modelStore.capabilities = caps;
        // If features are enabled server-side, turn on the toggles
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
    <!-- Sidebar overlay for mobile -->
    {#if settingsStore.sidebarOpen}
      <button
        class="fixed inset-0 z-30 bg-black/30 md:hidden"
        onclick={() => (settingsStore.sidebarOpen = false)}
        aria-label="Close sidebar"
      ></button>
    {/if}

    <!-- Sidebar -->
    <div
      class="fixed z-40 flex h-[calc(100vh-3rem)] w-72 flex-col border-r border-gray-200 bg-gray-50 transition-transform duration-200 md:relative md:z-auto md:translate-x-0 dark:border-gray-800 dark:bg-gray-900
      {settingsStore.sidebarOpen ? 'translate-x-0' : '-translate-x-full'}"
    >
      <Sidebar />
    </div>

    <!-- Main chat area -->
    <div class="flex min-w-0 flex-1 flex-col">
      <MessageList />
      <ChatInput />
    </div>

    <!-- Settings panel overlay -->
    {#if settingsStore.settingsOpen}
      <button
        class="fixed inset-0 z-30 bg-black/30"
        onclick={() => (settingsStore.settingsOpen = false)}
        aria-label="Close settings"
      ></button>
      <div class="fixed right-0 top-12 bottom-0 z-40 w-80 overflow-y-auto border-l border-gray-200 bg-white shadow-lg dark:border-gray-800 dark:bg-gray-950">
        <SettingsPanel />
      </div>
    {/if}
  </div>
</div>
