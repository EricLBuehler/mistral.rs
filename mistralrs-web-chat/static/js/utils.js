// Utility functions for markdown rendering, DOM manipulation, etc.

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
 * Update image input visibility based on model kind
 */
function updateImageVisibility(kind) {
  const imageLabel = document.getElementById('imageLabel');
  const imageInput = document.getElementById('imageInput');
  
  imageLabel.style.display = (kind === 'vision') ? 'inline-block' : 'none';
  if (kind !== 'vision') imageInput.value = '';
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

  // Store the upload URL for sending to the server
  container.dataset.uploadUrl = imageSrc;

  return container;
}
