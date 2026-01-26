// Settings management functionality

// Default settings (will be updated from server)
let serverDefaults = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  max_tokens: 2048,
  repetition_penalty: 1.1,
  system_prompt: null
};

// Current user settings (localStorage overrides)
let userSettings = {
  temperature: null,
  top_p: null,
  top_k: null,
  max_tokens: null,
  repetition_penalty: null,
  system_prompt: null
};

// Whether search is enabled on the server
let searchEnabledOnServer = false;

/**
 * Load settings from server and localStorage
 */
async function loadSettings() {
  try {
    const res = await fetch(apiUrl('api/settings'));
    if (res.ok) {
      const data = await res.json();
      serverDefaults = data.defaults;
      searchEnabledOnServer = data.search_enabled;

      // Hide web search controls if not enabled on server
      const webSearchCard = document.getElementById('webSearchCard');
      if (webSearchCard) {
        webSearchCard.style.display = searchEnabledOnServer ? 'block' : 'none';
      }
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }

  // Load user overrides from localStorage
  const stored = localStorage.getItem('mistralrs_settings');
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      userSettings = { ...userSettings, ...parsed };
    } catch (e) {
      console.error('Failed to parse stored settings:', e);
    }
  }

  // Update UI with current values
  updateSettingsUI();
}

/**
 * Save user settings to localStorage
 */
function saveSettings() {
  localStorage.setItem('mistralrs_settings', JSON.stringify(userSettings));
}

/**
 * Get effective value for a setting (user override or server default)
 */
function getSetting(key) {
  if (userSettings[key] !== null && userSettings[key] !== undefined) {
    return userSettings[key];
  }
  return serverDefaults[key];
}

/**
 * Update a setting value
 */
function setSetting(key, value) {
  userSettings[key] = value;
  saveSettings();

  // If system prompt changed, notify the WebSocket
  if (key === 'system_prompt' && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ set_system_prompt: value }));
  }
}

/**
 * Reset a setting to server default
 */
function resetSetting(key) {
  userSettings[key] = null;
  saveSettings();
  updateSettingsUI();
}

/**
 * Reset all settings to defaults
 */
function resetAllSettings() {
  userSettings = {
    temperature: null,
    top_p: null,
    top_k: null,
    max_tokens: null,
    repetition_penalty: null,
    system_prompt: null
  };
  saveSettings();
  updateSettingsUI();

  // Notify WebSocket of system prompt reset
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ set_system_prompt: serverDefaults.system_prompt }));
  }
}

/**
 * Update the settings UI to reflect current values
 */
function updateSettingsUI() {
  // Temperature
  const tempInput = document.getElementById('settingTemperature');
  const tempValue = document.getElementById('temperatureValue');
  if (tempInput && tempValue) {
    tempInput.value = getSetting('temperature');
    tempValue.textContent = getSetting('temperature').toFixed(2);
  }

  // Top P
  const topPInput = document.getElementById('settingTopP');
  const topPValue = document.getElementById('topPValue');
  if (topPInput && topPValue) {
    topPInput.value = getSetting('top_p');
    topPValue.textContent = getSetting('top_p').toFixed(2);
  }

  // Top K
  const topKInput = document.getElementById('settingTopK');
  if (topKInput) {
    topKInput.value = getSetting('top_k');
  }

  // Max Tokens
  const maxTokensInput = document.getElementById('settingMaxTokens');
  if (maxTokensInput) {
    maxTokensInput.value = getSetting('max_tokens');
  }

  // Repetition Penalty
  const repPenInput = document.getElementById('settingRepetitionPenalty');
  const repPenValue = document.getElementById('repetitionPenaltyValue');
  if (repPenInput && repPenValue) {
    repPenInput.value = getSetting('repetition_penalty');
    repPenValue.textContent = getSetting('repetition_penalty').toFixed(2);
  }

  // System Prompt
  const sysPromptInput = document.getElementById('settingSystemPrompt');
  if (sysPromptInput) {
    sysPromptInput.value = getSetting('system_prompt') || '';
  }
}

/**
 * Get generation params object for sending with messages
 */
function getGenerationParams() {
  return {
    temperature: getSetting('temperature'),
    top_p: getSetting('top_p'),
    top_k: getSetting('top_k'),
    max_tokens: getSetting('max_tokens'),
    repetition_penalty: getSetting('repetition_penalty')
  };
}

/**
 * Initialize settings panel event handlers
 */
function initSettingsHandlers() {
  // Settings toggle button
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsPanel = document.getElementById('settingsPanel');

  if (settingsBtn && settingsPanel) {
    settingsBtn.addEventListener('click', () => {
      settingsPanel.classList.toggle('hidden');
      settingsBtn.classList.toggle('active');
    });
  }

  // Temperature slider
  const tempInput = document.getElementById('settingTemperature');
  const tempValue = document.getElementById('temperatureValue');
  if (tempInput && tempValue) {
    tempInput.addEventListener('input', () => {
      const val = parseFloat(tempInput.value);
      tempValue.textContent = val.toFixed(2);
      setSetting('temperature', val);
    });
  }

  // Top P slider
  const topPInput = document.getElementById('settingTopP');
  const topPValue = document.getElementById('topPValue');
  if (topPInput && topPValue) {
    topPInput.addEventListener('input', () => {
      const val = parseFloat(topPInput.value);
      topPValue.textContent = val.toFixed(2);
      setSetting('top_p', val);
    });
  }

  // Top K input
  const topKInput = document.getElementById('settingTopK');
  if (topKInput) {
    topKInput.addEventListener('change', () => {
      const val = parseInt(topKInput.value, 10);
      if (!isNaN(val) && val > 0) {
        setSetting('top_k', val);
      }
    });
  }

  // Max Tokens input
  const maxTokensInput = document.getElementById('settingMaxTokens');
  if (maxTokensInput) {
    maxTokensInput.addEventListener('change', () => {
      const val = parseInt(maxTokensInput.value, 10);
      if (!isNaN(val) && val > 0) {
        setSetting('max_tokens', val);
      }
    });
  }

  // Repetition Penalty slider
  const repPenInput = document.getElementById('settingRepetitionPenalty');
  const repPenValue = document.getElementById('repetitionPenaltyValue');
  if (repPenInput && repPenValue) {
    repPenInput.addEventListener('input', () => {
      const val = parseFloat(repPenInput.value);
      repPenValue.textContent = val.toFixed(2);
      setSetting('repetition_penalty', val);
    });
  }

  // System Prompt textarea
  const sysPromptInput = document.getElementById('settingSystemPrompt');
  if (sysPromptInput) {
    let debounceTimer;
    sysPromptInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        const val = sysPromptInput.value.trim() || null;
        setSetting('system_prompt', val);
      }, 500);
    });
  }

  // Reset button
  const resetBtn = document.getElementById('resetSettingsBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (confirm('Reset all settings to defaults?')) {
        resetAllSettings();
      }
    });
  }
}

/**
 * Initialize settings module
 */
async function initSettings() {
  await loadSettings();
  initSettingsHandlers();
}
