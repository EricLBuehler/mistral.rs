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
  // Auto-scroll only if the user is already near the bottom
  const threshold = 20;
  if (log.scrollHeight - log.clientHeight - log.scrollTop < threshold) {
    log.scrollTop = log.scrollHeight;
  }
}

/**
 * Check if there are any uploaded files (images or text)
 */
function hasUploadedFiles() {
  const imageContainer = document.getElementById('image-container');
  const textContainer = document.getElementById('text-files-container');
  const hasImages = imageContainer.querySelectorAll('.image-preview-container').length > 0;
  const hasTextFiles = textContainer.querySelectorAll('.text-file-preview').length > 0;
  return hasImages || hasTextFiles;
}

/**
 * Send a message through WebSocket
 */
function sendMessage() {
  const input = document.getElementById('input');
  let msg = input.value.trim();
  // If speech model is selected, handle text-to-speech via HTTP
  const modelSelect = document.getElementById('modelSelect');
  const kind = models[modelSelect.value];
  if (kind === 'speech') {
    if (!msg) return;
    // Display user prompt in chat log
    append(renderMarkdown(msg), 'user');
    // Clear input field
    input.value = '';
    // Show spinner during generation
    showSpinner();
    fetch('/api/generate_speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: msg }),
    })
      .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      })
      .then(data => {
        hideSpinner();
        const div = append('', 'assistant');
        const link = document.createElement('a');
        link.href = data.url;
        link.textContent = 'Download WAV file';
        link.download = '';
        div.appendChild(link);
      })
      .catch(err => {
        hideSpinner();
        console.error(err);
        alert('Speech generation failed');
      });
    return;
  }
  
  if (!msg && !hasUploadedFiles()) return;
  
  if (ws.readyState !== WebSocket.OPEN) {
    alert('Connection lost. Please refresh the page.');
    return;
  }
  
  // Check if there are any uploaded text files to inject
  const textFilesContainer = document.getElementById('text-files-container');
  const textFiles = textFilesContainer.querySelectorAll('.text-file-preview');
  
  // Inject text file contents into the message
  if (textFiles.length > 0) {
    let fileContents = '';
    textFiles.forEach(preview => {
      const filename = preview.dataset.filename;
      const content = preview.dataset.content;
      // Wrap file content in a fenced code block to preserve literal text
      fileContents += `
--- ${filename} ---
\`\`\`text
${content}
\`\`\`
--- End of ${filename} ---
`;
    });
    
    if (msg) {
      msg = msg + fileContents;
    } else {
      msg = 'Here are the uploaded files:' + fileContents;
    }
  }
  
  // Create the user message div
  const userDiv = append(renderMarkdown(msg), 'user');
  
  // Check if there are any images in the image-container
  const imageContainer = document.getElementById('image-container');
  const imageContainers = imageContainer.querySelectorAll('.image-preview-container');
  
  if (imageContainers.length > 0) {
    // Send images to server context (once per message)
    imageContainers.forEach(container => {
      const url = container.dataset.uploadUrl;
      if (url) {
        ws.send(JSON.stringify({ image: url }));
      }
    });

    // Add thumbnails to the user message
    const imgWrap = document.createElement('div');
    imgWrap.className = 'chat-images';
    imgWrap.style.display = 'flex';
    imgWrap.style.flexWrap = 'wrap';
    imgWrap.style.gap = '1rem';

    imageContainers.forEach(container => {
      const img = container.querySelector('img');
      if (img) {
        const thumbnail = document.createElement('img');
        thumbnail.src = img.src;
        thumbnail.className = 'chat-preview';
        imgWrap.appendChild(thumbnail);
      }
    });

    userDiv.appendChild(imgWrap);
  }
  
  assistantBuf = ''; 
  assistantDiv = null;
  
  showSpinner();
  
  ws.send(msg);
  input.value = ''; 
  
  // Clear uploaded files after sending
  clearImagePreviews();
  clearTextFilePreviews();
  
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
