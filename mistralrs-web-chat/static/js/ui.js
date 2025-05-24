// UI interactions and behaviors

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
  
  const img = document.createElement('img'); 
  img.src = URL.createObjectURL(file); 
  img.classList.add('chat-preview');
  document.getElementById('image-container').appendChild(img);
  
  const fd = new FormData(); 
  fd.append('image', file);
  
  try {
    const r = await fetch('/api/upload_image', { method: 'POST', body: fd });
    
    if (r.ok && ws.readyState === WebSocket.OPEN) {
      const j = await r.json();
      ws.send(JSON.stringify({ image: j.url }));
    } else {
      // ✅ IMPROVEMENT: Better error handling
      const errorText = await r.text();
      alert(`Upload failed: ${errorText}`);
      // Remove the preview image on error
      img.remove();
    }
  } catch (error) {
    alert(`Upload failed: ${error.message}`);
    img.remove();
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
    
    const f = Array.from(e.dataTransfer.files).find(f => f.type.startsWith('image/')); 
    if (!f) {
      // ✅ IMPROVEMENT: Better feedback for non-image files
      alert('Please drop an image file');
      return;
    }
    
    await handleImageUpload(f);
  });
}

/**
 * Initialize all UI interactions
 */
function initUI() {
  initTextareaResize();
  initImageUpload();
  initDragAndDrop();
}
