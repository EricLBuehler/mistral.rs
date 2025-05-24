// WebSocket handling and message streaming

// WebSocket constants and state
const CLEAR_CMD = "__CLEAR__";
let ws;
let assistantBuf = '';
let assistantDiv = null;
let currentSpinner = null;

/**
 * Initialize WebSocket connection
 */
function initWebSocket() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  
  ws.addEventListener('message', handleWebSocketMessage);
  
  return ws;
}

/**
 * ✅ IMPROVEMENT: Helper function to manage spinner state
 */
function showSpinner() {
  // Remove existing spinner if any
  hideSpinner();
  
  const log = document.getElementById('log');
  const spinnerEl = document.createElement('div');
  spinnerEl.classList.add('spinner');
  spinnerEl.id = 'spinner';
  log.appendChild(spinnerEl);
  currentSpinner = spinnerEl;
}

/**
 * ✅ IMPROVEMENT: Helper function to hide spinner
 */
function hideSpinner() {
  if (currentSpinner) {
    currentSpinner.remove();
    currentSpinner = null;
  }
  // Also clean up any orphaned spinners
  const existingSpinner = document.getElementById('spinner');
  if (existingSpinner) {
    existingSpinner.remove();
  }
}

/**
 * Handle incoming WebSocket messages
 */
function handleWebSocketMessage(ev) {
  const log = document.getElementById('log');
  
  if (ev.data === '[Context cleared]') { 
    pendingClear = false; 
    hideSpinner();
    return; 
  }
  
  if (ev.data === 'Cannot clear while assistant is replying.') { 
    pendingClear = false; 
    alert(ev.data); 
    return; 
  }
  
  if (!assistantDiv) {
    hideSpinner();
    assistantDiv = append('', 'assistant');
  }
  
  assistantBuf += ev.data;
  assistantDiv.innerHTML = renderMarkdown(assistantBuf);
  addCopyBtns(assistantDiv); 
  fixLinks(assistantDiv);
  log.scrollTop = log.scrollHeight;
}

/**
 * Send a message through WebSocket
 */
function sendMessage() {
  const input = document.getElementById('input');
  const msg = input.value.trim();
  
  if (!msg) return;
  
  if (ws.readyState !== WebSocket.OPEN) {
    alert('Connection lost. Please refresh the page.');
    return;
  }
  
  append(renderMarkdown(msg), 'user');
  assistantBuf = ''; 
  assistantDiv = null;
  
  showSpinner();
  
  ws.send(msg);
  input.value = ''; 
  
  // Trigger textarea resize
  const event = new Event('input');
  input.dispatchEvent(event);
}

/**
 * Initialize message sending functionality
 */
function initMessageSending() {
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  
  form.addEventListener('submit', ev => {
    ev.preventDefault();
    sendMessage();
  });
  
  input.addEventListener('keydown', ev => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      if (ev.ctrlKey || ev.metaKey) {
        ev.preventDefault();
        sendMessage();
      } else {
        ev.preventDefault();
      }
    }
  });
  
  ws.addEventListener('close', () => {
    hideSpinner();
    console.warn('WebSocket connection closed');
  });
  
  ws.addEventListener('error', (error) => {
    hideSpinner();
    console.error('WebSocket error:', error);
  });
}
