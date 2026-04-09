// Main application initialization and coordination

// Get DOM elements that are used across modules
const log = document.getElementById('log');
const input = document.getElementById('input');
const form = document.getElementById('form');
const modelSelect = document.getElementById('modelSelect');
const clearBtn = document.getElementById('clearBtn');
const newChatBtn = document.getElementById('newChatBtn');
const renameBtn = document.getElementById('renameBtn');
const deleteBtn = document.getElementById('deleteBtn');
const chatList = document.getElementById('chatList');
const imageInput = document.getElementById('imageInput');
const imageLabel = document.getElementById('imageLabel');
const mainArea = document.getElementById('main');

let initRetryCount = 0;
const MAX_INIT_RETRIES = 3;

/**
 * Show an initialization error banner in the #log area.
 * @param {Error} err - The error that was caught.
 * @param {number} retryNum - Current retry number (1-based), or -1 if no more retries.
 */
function showInitError(err, retryNum) {
  const logEl = document.getElementById('log');
  if (!logEl) return;
  // Remove any existing init error
  const existing = logEl.querySelector('.init-error');
  if (existing) existing.remove();

  const errorDiv = document.createElement('div');
  errorDiv.className = 'init-error';
  if (retryNum === -1) {
    errorDiv.innerHTML = `<strong>Initialization Failed</strong><br>The server is unreachable after multiple attempts. Please check that the server is running and <a href="javascript:location.reload()">refresh the page</a>.<br><small>${err.message}</small>`;
  } else {
    errorDiv.innerHTML = `<strong>Connection Error</strong><br>Failed to connect to the server. Retrying in 5s (attempt ${retryNum}/${MAX_INIT_RETRIES})...<br><small>${err.message}</small>`;
  }
  logEl.appendChild(errorDiv);
}

/**
 * Initialize the entire application
 */
async function initApp() {
  try {
    // Initialize theme (must be first to prevent flash)
    initTheme();
    initThemeToggle();

    // Initialize WebSocket connection
    initWebSocket();

    // Initialize settings (loads from server and localStorage)
    await initSettings();

    // Initialize all modules
    initModelSelection();
    initChatHandlers();
    initMessageSending();
    initUI();

    // Load initial data
    await refreshModels();
  } catch (err) {
    console.error('App initialization failed:', err);
    initRetryCount++;
    if (initRetryCount <= MAX_INIT_RETRIES) {
      showInitError(err, initRetryCount);
      setTimeout(initApp, 5000);
    } else {
      showInitError(err, -1); // -1 = no more retries
    }
  }
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
