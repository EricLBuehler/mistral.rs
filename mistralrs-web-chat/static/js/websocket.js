// WebSocket handling and message streaming

// WebSocket constants and state
const CLEAR_CMD = "__CLEAR__";
let ws;
let assistantBuf = '';
let assistantDiv = null;

/**
 * Initialize WebSocket connection
 */
function initWebSocket() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  
  ws.addEventListener('message', handleWebSocketMessage);
  
  return ws;
}

/**
 * Handle incoming WebSocket messages
 */
function handleWebSocketMessage(ev) {
  const log = document.getElementById('log');
  
  if (ev.data === '[Context cleared]') { 
    pendingClear = false; 
    return; 
  }
  
  if (ev.data === 'Cannot clear while assistant is replying.') { 
    pendingClear = false; 
    alert(ev.data); 
    return; 
  }
  
  if (!assistantDiv) {
    // remove inline spinner when first assistant data arrives
    const spinner = document.getElementById('spinner');
    if (spinner) spinner.remove();
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
  
  append(renderMarkdown(msg), 'user');
  assistantBuf = ''; 
  assistantDiv = null;
  ws.send(msg);
  // dynamically add spinner in log area
  const log = document.getElementById('log');
  const spinnerEl = document.createElement('div');
  spinnerEl.classList.add('spinner');
  spinnerEl.id = 'spinner';
  log.appendChild(spinnerEl);
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
}
