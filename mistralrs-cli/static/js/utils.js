// Utility functions for markdown rendering, DOM manipulation, etc.

const UI_BASE = (() => {
  const baseTag = document.querySelector('base');
  if (baseTag) {
    const href = baseTag.getAttribute('href');
    if (href) return href.endsWith('/') ? href : `${href}/`;
  }
  const path = window.location.pathname;
  return path.endsWith('/') ? path : `${path}/`;
})();

function apiUrl(path) {
  return `${UI_BASE}${path.replace(/^\/+/, '')}`;
}

function wsUrl(path) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${UI_BASE}${path.replace(/^\/+/, '')}`;
}

/**
 * Render markdown to HTML with proper escaping
 */
function renderMarkdown(src) {
  // Helper: escape & < > once
  const escape = s =>
    s.replace(/&(?![a-zA-Z0-9#]+;)/g,'&amp;')
     .replace(/</g,'&lt;')
     .replace(/>/g,'&gt;');

  // Split the markdown into segments that are either
  // (1) fenced code blocks  ``` ... ```
  // (2) inline code  `...`
  // (3) normal text  (everything else)
  //
  // We only escape (3).
  const pattern = /(```[\s\S]*?```|`[^`]*`)/g;
  const escaped = src.split(pattern).map(seg=>{
    // segments that start with back-tick are code, keep raw
    return seg.startsWith('`') ? seg : escape(seg);
  }).join('');

  return marked.parse(escaped);
}

/**
 * Append HTML content to the chat log
 */
function append(html, cls = '') {
  const d = document.createElement('div');
  if (cls) d.className = cls;
  d.innerHTML = html;
  addCopyBtns(d); 
  fixLinks(d);
  log.appendChild(d); 
  log.scrollTop = log.scrollHeight; 
  return d;
}

/**
 * Add copy buttons to code blocks
 */
function addCopyBtns(el) {
  el.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.copy-btn')) return; // avoid duplicates
    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.className = 'copy-btn';

    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const codeEl = pre.querySelector('code') || pre;
      const text = codeEl.innerText.trim();

      try {
        await navigator.clipboard.writeText(text); // modern clipboard API
      } catch {
        /* Fallback for older browsers */
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try { document.execCommand('copy'); } catch (_) {}
        document.body.removeChild(ta);
      }

      btn.textContent = '✔';
      setTimeout(() => (btn.textContent = 'Copy'), 800);
    });

    pre.appendChild(btn);
  });
}

/**
 * Fix links to open in new tab
 */
function fixLinks(el) { 
  el.querySelectorAll('a[href]').forEach(a => {
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
  }); 
}

/**
 * Clear image previews from the image container
 */
function clearImagePreviews() {
  const imgContainer = document.getElementById('image-container');
  if (imgContainer) imgContainer.innerHTML = '';
}

/**
 * Clear text file previews from the text files container
 */
function clearTextFilePreviews() {
  const textContainer = document.getElementById('text-files-container');
  if (textContainer) textContainer.innerHTML = '';
}

/**
 * Clear audio previews from the audio container
 */
function clearAudioPreviews() {
  const audioContainer = document.getElementById('audio-container');
  if (audioContainer) audioContainer.innerHTML = '';
}

/**
 * Update image input visibility based on model kind
 */
function updateImageVisibility(kind) {
  const imageLabel = document.getElementById('imageLabel');
  const imageInput = document.getElementById('imageInput');
  const audioLabel = document.getElementById('audioLabel');
  const audioInput = document.getElementById('audioInput');
  const textLabel = document.getElementById('textLabel');
  const textInput = document.getElementById('textInput');
  
  const isVision = (kind === 'vision');
  const isText = (kind === 'text');

  // Show audio upload for vision models as well (covers audio-enabled models)
  audioLabel.style.display = isVision ? 'inline-block' : 'none';
  if (!isVision) audioInput.value = '';
  
  // Toggle image upload only for vision models
  imageLabel.style.display = isVision ? 'inline-block' : 'none';
  if (!isVision) imageInput.value = '';
  
  // Toggle file upload only for text models
  textLabel.style.display = isText ? 'inline-block' : 'none';
  if (!isText) textInput.value = '';
}

/**
 * Create an image preview element with delete button
 */
function createImagePreview(imageSrc) {
  const container = document.createElement('div');
  container.className = 'image-preview-container';
  container.style.position = 'relative';
  container.style.display = 'inline-block';
  
  const img = document.createElement('img');
  img.src = imageSrc;
  img.className = 'chat-preview';
  
  const removeBtn = document.createElement('button');
  removeBtn.className = 'image-remove-btn';
  removeBtn.textContent = '×';
  removeBtn.title = 'Remove image';
  removeBtn.style.position = 'absolute';
  removeBtn.style.top = '5px';
  removeBtn.style.right = '5px';
  removeBtn.style.background = 'rgba(0,0,0,0.7)';
  removeBtn.style.color = 'white';
  removeBtn.style.border = 'none';
  removeBtn.style.borderRadius = '50%';
  removeBtn.style.width = '20px';
  removeBtn.style.height = '20px';
  removeBtn.style.cursor = 'pointer';
  removeBtn.style.fontSize = '12px';
  removeBtn.style.display = 'flex';
  removeBtn.style.alignItems = 'center';
  removeBtn.style.justifyContent = 'center';
  
  removeBtn.addEventListener('click', () => {
    container.remove();
  });
  
  removeBtn.addEventListener('mouseenter', () => {
    removeBtn.style.background = '#ff4444';
  });
  
  removeBtn.addEventListener('mouseleave', () => {
    removeBtn.style.background = 'rgba(0,0,0,0.7)';
  });
  
  container.appendChild(img);
  container.appendChild(removeBtn);

  // Store the display URL; upload path will be filled after server upload
  container.dataset.displayUrl = imageSrc;

  return container;
}

/**
 * Create an audio preview element with delete button
 */
function createAudioPreview(audioSrc) {
  const container = document.createElement('div');
  container.className = 'audio-preview-container';
  container.style.position = 'relative';
  container.style.display = 'inline-block';

  const audio = document.createElement('audio');
  audio.src = audioSrc;
  audio.controls = true;
  audio.style.maxWidth = '240px';

  const removeBtn = document.createElement('button');
  removeBtn.className = 'audio-remove-btn';
  removeBtn.textContent = '×';
  removeBtn.title = 'Remove audio';
  removeBtn.style.position = 'absolute';
  removeBtn.style.top = '5px';
  removeBtn.style.right = '5px';
  removeBtn.style.background = 'rgba(0,0,0,0.7)';
  removeBtn.style.color = 'white';
  removeBtn.style.border = 'none';
  removeBtn.style.borderRadius = '50%';
  removeBtn.style.width = '20px';
  removeBtn.style.height = '20px';
  removeBtn.style.cursor = 'pointer';
  removeBtn.style.fontSize = '12px';

  removeBtn.addEventListener('click', () => {
    container.remove();
  });

  removeBtn.addEventListener('mouseenter', () => {
    removeBtn.style.background = '#ff4444';
  });

  removeBtn.addEventListener('mouseleave', () => {
    removeBtn.style.background = 'rgba(0,0,0,0.7)';
  });

  container.appendChild(audio);
  container.appendChild(removeBtn);

  container.dataset.displayUrl = audioSrc;

  return container;
}
