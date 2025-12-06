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

/**
 * Initialize the entire application
 */
async function initApp() {
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
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
