// WebSocket handling and message streaming

// WebSocket constants and state
const CLEAR_CMD = "__CLEAR__";
let ws;
let assistantBuf = '';
let assistantDiv = null;
let currentSpinner = null;

/**
 * Initialize WebSocket connection with reconnection support
 */
function initWebSocket() {
  ws = new WebSocket(wsUrl('ws'));

  ws.addEventListener('open', () => {
    console.log('WebSocket connected');
    // Send system prompt if set
    if (typeof getSetting === 'function') {
      const sysPrompt = getSetting('system_prompt');
      if (sysPrompt) {
        ws.send(JSON.stringify({ set_system_prompt: sysPrompt }));
      }
    }
    // Restore pending chat context first, otherwise re-send current chat ID
    if (typeof flushPendingChatRestore === 'function') {
      flushPendingChatRestore();
    } else if (typeof currentChatId !== 'undefined' && currentChatId) {
      ws.send(JSON.stringify({ chat_id: currentChatId }));
    }
  });

  ws.addEventListener('message', handleWebSocketMessage);

  ws.addEventListener('error', (e) => {
    console.error('WebSocket error:', e);
  });

  ws.addEventListener('close', (e) => {
    console.log('WebSocket closed:', e.code, e.reason);
    hideSpinner();
    // Auto-reconnect after a delay (unless it was a clean close)
    if (e.code !== 1000) {
      setTimeout(() => {
        console.log('Attempting to reconnect...');
        initWebSocket();
      }, 2000);
    }
  });

  return ws;
}

/**
 * Show an error message in the chat log
 */
function showError(message) {
  const logEl = document.getElementById('log');
  const errorDiv = document.createElement('div');
  errorDiv.className = 'msg error-msg';
  errorDiv.innerHTML = `<strong>Error:</strong> ${message}`;
  logEl.appendChild(errorDiv);
  logEl.scrollTop = logEl.scrollHeight;
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
  const logEl = document.getElementById('log');

  // Handle system messages
  if (ev.data === '[Context cleared]') {
    pendingClear = false;
    hideSpinner();
    return;
  }

  if (ev.data === '[System prompt updated]') {
    console.log('System prompt updated on server');
    return;
  }

  if (ev.data === 'Cannot clear while assistant is replying.') {
    pendingClear = false;
    alert(ev.data);
    return;
  }

  // Handle error messages from server
  if (ev.data.startsWith('Error:') || ev.data.startsWith('No model selected') || ev.data.startsWith('Selected model not found')) {
    hideSpinner();
    showError(ev.data);
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
  if (logEl.scrollHeight - logEl.clientHeight - logEl.scrollTop < threshold) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

/**
 * Check if there are any uploaded files (images or text)
 */
function hasUploadedFiles() {
  const imageContainer = document.getElementById('image-container');
  const textContainer = document.getElementById('text-files-container');
  const audioContainer = document.getElementById('audio-container');
  const hasImages = imageContainer.querySelectorAll('.image-preview-container').length > 0;
  const hasTextFiles = textContainer.querySelectorAll('.text-file-preview').length > 0;
  const hasAudios = audioContainer.querySelectorAll('.audio-preview-container').length > 0;
  return hasImages || hasTextFiles || hasAudios;
}

/**
 * Check if any uploads are still in progress
 */
function hasPendingUploads() {
  const imageContainer = document.getElementById('image-container');
  const audioContainer = document.getElementById('audio-container');
  const imageContainers = imageContainer.querySelectorAll('.image-preview-container');
  const audioContainers = audioContainer.querySelectorAll('.audio-preview-container');

  const hasPendingImages = Array.from(imageContainers).some(container =>
    container.dataset.uploading === 'true' || !container.dataset.uploadPath
  );
  const hasPendingAudio = Array.from(audioContainers).some(container =>
    container.dataset.uploading === 'true' || !container.dataset.uploadPath
  );

  return hasPendingImages || hasPendingAudio;
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
    fetch(apiUrl('api/generate_speech'), {
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

  if (hasPendingUploads()) {
    alert('Please wait for file uploads to finish before sending.');
    return;
  }
  
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

  // Check for audio attachments
  const audioContainerDiv = document.getElementById('audio-container');
  const audioContainers = audioContainerDiv.querySelectorAll('.audio-preview-container');
  
  if (imageContainers.length > 0) {
    // Send images to server context (once per message)
    imageContainers.forEach(container => {
      const path = container.dataset.uploadPath || container.dataset.uploadUrl;
      if (path) {
        ws.send(JSON.stringify({ image: path }));
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

  // ----- Handle audio attachments -----
  if (audioContainers.length > 0) {
    // Send audio URLs first
    audioContainers.forEach(container => {
      const path = container.dataset.uploadPath || container.dataset.uploadUrl;
      if (path) {
        ws.send(JSON.stringify({ audio: path }));
      }
    });

    // Add audio players to user message
    const audioWrap = document.createElement('div');
    audioWrap.className = 'chat-audios';
    audioWrap.style.display = 'flex';
    audioWrap.style.flexDirection = 'column';
    audioWrap.style.gap = '1rem';

    audioContainers.forEach(container => {
      const audioEl = container.querySelector('audio');
      if (audioEl) {
        const clone = audioEl.cloneNode(true);
        clone.controls = true;
        clone.style.maxWidth = '240px';
        audioWrap.appendChild(clone);
      }
    });
    userDiv.appendChild(audioWrap);
  }
  
  assistantBuf = ''; 
  assistantDiv = null;
  
  showSpinner();
  
  // Build message payload with generation params
  const payload = {
    content: msg,
    generation_params: getGenerationParams()
  };

  // Add web search options if enabled
  const enableSearch = document.getElementById('enableSearch')?.checked;
  if (enableSearch) {
    const opts = {};
    const sizeSelect = document.getElementById('searchContextSize');
    if (sizeSelect) {
      const sizeValue = sizeSelect.value;
      if (sizeValue) opts.search_context_size = sizeValue;
    }
    payload.web_search_options = opts;
  }

  ws.send(JSON.stringify(payload));
  input.value = ''; 
  
  // Clear uploaded files after sending
  clearImagePreviews();
  clearTextFilePreviews();
  clearAudioPreviews();
  
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
