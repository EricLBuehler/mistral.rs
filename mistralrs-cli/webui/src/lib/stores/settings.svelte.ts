import type { AgentPermission, ReasoningEffort } from "../types";

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
  enableShell: boolean;
  agentPermission: AgentPermission;
  reasoningEffort: ReasoningEffort | "";
  enableThinking?: boolean;
}

export type Theme = "system" | "light" | "dark";
const THEME_ORDER: Theme[] = ["system", "light", "dark"];
// Separate from STORAGE_KEY so picking a theme doesn't mask the model's generation defaults.
const THEME_KEY = "mistralrs_theme";

function loadTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t && (THEME_ORDER as string[]).includes(t)) return t as Theme;
  } catch {
    // ignore
  }
  return "system";
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

const FLOAT_PARAM_PRECISION = 6;

// Older servers widened f32 defaults to f64 (1.100000023841858), and saved settings may still hold them.
function tidy(value: number | undefined): number | undefined {
  return value == null ? value : Number(value.toPrecision(FLOAT_PARAM_PRECISION));
}

const stored = loadStored();

class SettingsStore {
  temperature = $state(tidy(stored.temperature) ?? 0.7);
  topP = $state(tidy(stored.topP) ?? 0.9);
  topK = $state(stored.topK ?? 40);
  maxTokens = $state(stored.maxTokens ?? 8192);
  repetitionPenalty = $state(tidy(stored.repetitionPenalty) ?? 1.1);
  systemPrompt = $state(stored.systemPrompt ?? "");
  enableSearch = $state(stored.enableSearch ?? false);
  enableCodeExecution = $state(stored.enableCodeExecution ?? false);
  enableShell = $state(stored.enableShell ?? false);
  agentPermission = $state<AgentPermission>(stored.agentPermission ?? "auto");
  reasoningEffort = $state<ReasoningEffort | "">(
    stored.reasoningEffort ?? (stored.enableThinking === false ? "off" : ""),
  );
  theme = $state<Theme>(loadTheme());
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
      enableShell: this.enableShell,
      agentPermission: this.agentPermission,
      reasoningEffort: this.reasoningEffort,
    });
  }

  cycleTheme() {
    this.theme = THEME_ORDER[(THEME_ORDER.indexOf(this.theme) + 1) % THEME_ORDER.length];
    try {
      localStorage.setItem(THEME_KEY, this.theme);
    } catch {
      // ignore
    }
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
