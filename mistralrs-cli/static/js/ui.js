// UI interactions and behaviors

const TEXT_FILE_MIME_TYPES = new Set([
  'text/plain', 'text/markdown', 'text/csv', 'text/xml', 'text/yaml', 'text/css', 'text/html',
  'application/json', 'application/javascript', 'application/xml', 'application/yaml',
  'application/x-python', 'application/x-rust', 'application/x-sh', 'application/octet-stream'
]);

const TEXT_FILE_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'py', 'js', 'ts', 'jsx', 'tsx', 'rs', 'html', 'htm', 'css', 'scss', 'sass', 'less',
  'json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf',
  'c', 'cpp', 'cc', 'cxx', 'h', 'hpp', 'hxx', 'java', 'kt', 'swift', 'go', 'rb', 'php',
  // GPU and shader languages
  'cu', 'cuh', 'cl', 'ptx', 'glsl', 'vert', 'frag', 'geom', 'comp', 'tesc', 'tese', 'hlsl', 'metal', 'wgsl',
  // Shell and scripts
  'sh', 'bash', 'zsh', 'fish', 'ps1', 'bat', 'cmd', 'sql', 'log', 'csv', 'tsv',
  'dockerfile', 'makefile', 'gitignore', 'dockerignore', 'editorconfig', 'env', 'htaccess', 'lock',
  'swift', 'kt', 'scala', 'r', 'clj', 'cljs', 'hs', 'elm', 'ex', 'exs', 'erl', 'fs', 'fsx', 'ml', 'mli',
  'vue', 'svelte', 'astro', 'lua', 'nim', 'zig', 'd', 'dart', 'jl', 'pl', 'pm', 'tcl',
  'readme', 'license', 'changelog', 'vagrantfile', 'gemfile', 'rakefile'
]);

function getFileExtension(filename) {
  const parts = filename.toLowerCase().split('.');
  return parts.length > 1 ? parts.pop() : parts[0];
}

function isAllowedTextFile(file) {
  const ext = getFileExtension(file.name || '');
  if (ext && TEXT_FILE_EXTENSIONS.has(ext)) return true;
  if (file.type && file.type.startsWith('text/')) return true;
  if (TEXT_FILE_MIME_TYPES.has(file.type)) return true;
  if (!file.type || file.type === 'application/octet-stream') {
    return ext && TEXT_FILE_EXTENSIONS.has(ext);
  }
  return false;
}

/**
 * Initialize textarea auto-resize functionality
 */
function initTextareaResize() {
  const input = document.getElementById('input');
  // ✅ FIX: Make consistent with CSS max-height: calc(1.4em * 15)
  const maxH = parseFloat(getComputedStyle(input).lineHeight) * 15;
  
  function fit() { 
    input.style.height = 'auto'; 
    input.style.height = Math.min(input.scrollHeight, maxH) + 'px'; 
  }
  
  input.addEventListener('input', fit); 
  fit();
}

/**
 * Handle image uploads
 */
async function handleImageUpload(file) {
  // ✅ IMPROVEMENT: Validate file type before processing
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file');
    return;
  }
  
  // ✅ IMPROVEMENT: Validate file size (50MB limit to match backend)
  const maxSize = 50 * 1024 * 1024; // 50MB
  if (file.size > maxSize) {
    alert('Image file is too large. Maximum size is 50MB.');
    return;
  }
  
  const objectUrl = URL.createObjectURL(file);
  const preview = createImagePreview(objectUrl);
  preview.dataset.uploading = 'true';
  document.getElementById('image-container').appendChild(preview);
  
  const fd = new FormData(); 
  fd.append('image', file);
  
  try {
    const r = await fetch(apiUrl('api/upload_image'), { method: 'POST', body: fd });
    if (r.ok) {
      const j = await r.json();
      // Record the server upload path + display URL for use on send/render
      preview.dataset.uploadPath = j.path || j.url || '';
      preview.dataset.displayUrl = j.url || '';
      preview.dataset.uploading = 'false';

      const img = preview.querySelector('img');
      if (img && preview.dataset.displayUrl) {
        img.onload = () => URL.revokeObjectURL(objectUrl);
        img.src = preview.dataset.displayUrl;
      }
    } else {
      const errorText = await r.text();
      alert(`Upload failed: ${errorText}`);
      preview.remove();
    }
  } catch (error) {
    alert(`Upload failed: ${error.message}`);
    preview.remove();
  }
}

/**
 * Create a text file preview element
 */
function createTextFilePreview(filename, content, size) {
  const preview = document.createElement('div');
  preview.className = 'text-file-preview';
  
  // File header with name, size, and remove button
  const header = document.createElement('div');
  header.className = 'file-header';
  
  const fileName = document.createElement('div');
  fileName.className = 'file-name';
  fileName.textContent = filename;
  fileName.title = filename; // Show full name on hover
  
  const fileSize = document.createElement('div');
  fileSize.className = 'file-size';
  fileSize.textContent = formatFileSize(size);
  
  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove-btn';
  removeBtn.textContent = '×';
  removeBtn.title = 'Remove file';
  removeBtn.addEventListener('click', () => {
    preview.remove();
  });
  
  header.appendChild(fileName);
  header.appendChild(fileSize);
  header.appendChild(removeBtn);
  
  // File content preview (truncated)
  const contentDiv = document.createElement('div');
  contentDiv.className = 'file-content';
  const truncatedContent = content.length > 500 ? content.substring(0, 500) + '...' : content;
  contentDiv.textContent = truncatedContent;
  
  preview.appendChild(header);
  preview.appendChild(contentDiv);
  
  // Store the full content as a data attribute
  preview.dataset.content = content;
  preview.dataset.filename = filename;
  
  return preview;
}

/**
 * Format file size in human readable format
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Handle text file uploads
 */
async function handleTextUpload(file) {
  // ✅ IMPROVEMENT: Validate file type before processing
  if (!isAllowedTextFile(file)) {
    alert('Please select a text or code file');
    return;
  }
  
  // ✅ IMPROVEMENT: Validate file size (10MB limit to match backend)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    alert('Text file is too large. Maximum size is 10MB.');
    return;
  }
  
  try {
    const content = await file.text();
    const preview = createTextFilePreview(file.name, content, file.size);
    document.getElementById('text-files-container').appendChild(preview);
  } catch (error) {
    alert(`Failed to read file: ${error.message}`);
  }
}

/**
 * Initialize image upload functionality
 */
function initImageUpload() {
  const imageInput = document.getElementById('imageInput');
  
  imageInput.addEventListener('change', async () => {
    const f = imageInput.files[0]; 
    if (!f) return;
    
    await handleImageUpload(f);
    imageInput.value = '';
  });
}

/**
 * Initialize text file upload functionality
 */
function initTextUpload() {
  const textInput = document.getElementById('textInput');
  
  textInput.addEventListener('change', async () => {
    const f = textInput.files[0]; 
    if (!f) return;
    
    await handleTextUpload(f);
    textInput.value = '';
  });
}

/**
 * Initialize drag and drop functionality
 */
function initDragAndDrop() {
  const mainArea = document.getElementById('main');
  
  mainArea.addEventListener('dragover', e => {
    e.preventDefault();
    mainArea.classList.add('drag-over');
  });
  
  mainArea.addEventListener('dragleave', e => {
    e.preventDefault();
    mainArea.classList.remove('drag-over');
  });
  
  mainArea.addEventListener('drop', async e => {
    e.preventDefault(); 
    mainArea.classList.remove('drag-over');
    
    const files = Array.from(e.dataTransfer.files);
    
    // Handle image files
    const imageFile = files.find(f => f.type.startsWith('image/'));
    if (imageFile) {
      await handleImageUpload(imageFile);
      return;
    }

    // Handle audio files
    const audioFile = files.find(f => f.type.startsWith('audio/'));
    if (audioFile) {
      await handleAudioUpload(audioFile);
      return;
    }
    
    // Handle text files
    const textFile = files.find(isAllowedTextFile);
    
    if (textFile) {
      await handleTextUpload(textFile);
      return;
    }
    
    // No supported file type found
    alert('Please drop an image, audio, or text/code file');
  });
}

/**
 * Initialize stop button functionality: abort assistant streaming
 */
function initStopButton() {
  const stopBtn = document.getElementById('stopBtn');
  if (!stopBtn) return;
  stopBtn.addEventListener('click', async () => {
    // Save partial assistant response
    if (typeof assistantBuf !== 'undefined' && assistantBuf) {
      if (typeof currentChatId !== 'undefined' && currentChatId) {
        try {
          await fetch(apiUrl('api/append_message'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: currentChatId, role: 'assistant', content: assistantBuf })
          });
        } catch (e) {
          console.error('Failed to append partial message:', e);
        }
      }
    }
    // Close and reset WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    hideSpinner();
    assistantBuf = '';
    assistantDiv = null;
    // Reconnect WebSocket for further messages
    initWebSocket();
  });
}

/**
 * Initialize web search controls: toggle visibility of search options
 */
function initWebSearchControls() {
  const checkbox = document.getElementById('enableSearch');
  const options = document.getElementById('webSearchOptions');
  if (!checkbox || !options) return;
  checkbox.addEventListener('change', () => {
    options.hidden = !checkbox.checked;
  });
}

/**
 * Initialize all UI interactions
 */
function initUI() {
  initTextareaResize();
  initImageUpload();
  initTextUpload();
  initAudioUpload();
  initDragAndDrop();
  initWebSearchControls();
  initStopButton();
}

// ---------------------- Audio Upload ----------------------

async function handleAudioUpload(file) {
  if (!file.type.startsWith('audio/')) {
    alert('Please select an audio file');
    return;
  }
  // Limit 50MB
  const maxSize = 50 * 1024 * 1024;
  if (file.size > maxSize) {
    alert('Audio file is too large. Maximum size is 50MB.');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  const preview = createAudioPreview(objectUrl);
  preview.dataset.uploading = 'true';
  document.getElementById('audio-container').appendChild(preview);

  const fd = new FormData();
  fd.append('audio', file);

  try {
    const r = await fetch(apiUrl('api/upload_audio'), { method: 'POST', body: fd });
    if (r.ok) {
      const j = await r.json();
      preview.dataset.uploadPath = j.path || j.url || '';
      preview.dataset.displayUrl = j.url || '';
      preview.dataset.uploading = 'false';

      const audio = preview.querySelector('audio');
      if (audio && preview.dataset.displayUrl) {
        audio.onloadeddata = () => URL.revokeObjectURL(objectUrl);
        audio.src = preview.dataset.displayUrl;
      }
    } else {
      const errText = await r.text();
      alert(`Upload failed: ${errText}`);
      preview.remove();
    }
  } catch (e) {
    alert(`Upload failed: ${e.message}`);
    preview.remove();
  }
}

function initAudioUpload() {
  const audioInput = document.getElementById('audioInput');
  if (!audioInput) return;

  audioInput.addEventListener('change', async () => {
    const f = audioInput.files[0];
    if (!f) return;
    await handleAudioUpload(f);
    audioInput.value = '';
  });
}
