import type { AgentPermission } from "../types";

const STORAGE_KEY = "mistralrs_settings";

interface StoredSettings {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repetitionPenalty: number;
  systemPrompt: string;
  enableSearch: boolean;
  enableCodeExecution: boolean;
  agentPermission: AgentPermission;
  enableThinking: boolean;
}

// Stale keys from previous UI versions — clean them up on load
const LEGACY_KEYS = ["mistralrs-settings", "theme"];

function loadStored(): Partial<StoredSettings> {
  // Clean up legacy keys from prior UI versions
  for (const key of LEGACY_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch {
      // ignore
    }
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore
  }
  return {};
}

function save(s: StoredSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

const stored = loadStored();

class SettingsStore {
  temperature = $state(stored.temperature ?? 0.7);
  topP = $state(stored.topP ?? 0.9);
  topK = $state(stored.topK ?? 40);
  maxTokens = $state(stored.maxTokens ?? 8192);
  repetitionPenalty = $state(stored.repetitionPenalty ?? 1.1);
  systemPrompt = $state(stored.systemPrompt ?? "");
  enableSearch = $state(stored.enableSearch ?? false);
  enableCodeExecution = $state(stored.enableCodeExecution ?? false);
  agentPermission = $state<AgentPermission>(stored.agentPermission ?? "auto");
  enableThinking = $state(stored.enableThinking ?? true);
  sidebarOpen = $state(window.innerWidth >= 768);
  settingsOpen = $state(false);
  toolsOpen = $state(false);

  persist() {
    save({
      temperature: this.temperature,
      topP: this.topP,
      topK: this.topK,
      maxTokens: this.maxTokens,
      repetitionPenalty: this.repetitionPenalty,
      systemPrompt: this.systemPrompt,
      enableSearch: this.enableSearch,
      enableCodeExecution: this.enableCodeExecution,
      agentPermission: this.agentPermission,
      enableThinking: this.enableThinking,
    });
  }

  applyDefaults(defaults: {
    temperature?: number | null;
    top_p?: number | null;
    top_k?: number | null;
    max_tokens?: number | null;
    repetition_penalty?: number | null;
    system_prompt?: string | null;
  }) {
    // Only apply defaults if user hasn't saved custom settings
    if (!localStorage.getItem(STORAGE_KEY)) {
      if (defaults.temperature != null) this.temperature = defaults.temperature;
      if (defaults.top_p != null) this.topP = defaults.top_p;
      if (defaults.top_k != null) this.topK = defaults.top_k;
      if (defaults.max_tokens != null) this.maxTokens = defaults.max_tokens;
      if (defaults.repetition_penalty != null)
        this.repetitionPenalty = defaults.repetition_penalty;
      if (defaults.system_prompt) this.systemPrompt = defaults.system_prompt;
    }
  }
}

export const settingsStore = new SettingsStore();
