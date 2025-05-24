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
 * Update image input visibility based on model kind
 */
function updateImageVisibility(kind) {
  const imageLabel = document.getElementById('imageLabel');
  const imageInput = document.getElementById('imageInput');
  
  imageLabel.style.display = (kind === 'vision') ? 'inline-block' : 'none';
  if (kind !== 'vision') imageInput.value = '';
}
