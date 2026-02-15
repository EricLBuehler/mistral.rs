// Model management functionality

// Global model state
const models = Object.create(null); // name -> kind
let prevModel = null;

/**
 * Refresh the list of available models
 */
async function refreshModels() {
  const res = await fetch('/api/list_models');
  const data = await res.json();
  const modelSelect = document.getElementById('modelSelect');
  
  modelSelect.innerHTML = '';
  Object.keys(models).forEach(k => delete models[k]);
  
  data.models.forEach(m => {
    models[m.name] = m.kind;
    const opt = document.createElement('option');
    opt.value = m.name;
    // Prefix icon based on model kind
    let prefix = '📝 ';
    if (m.kind === 'vision') prefix = '🖼️ ';
    else if (m.kind === 'speech') prefix = '🔊 ';
    opt.textContent = prefix + m.name;
    modelSelect.appendChild(opt);
  });
  
  if (modelSelect.options.length) {
    modelSelect.selectedIndex = 0;
    prevModel = modelSelect.value;
    updateImageVisibility(models[prevModel]);
    await selectModel(prevModel, false);
  }
  
  await refreshChatList();
  if (!currentChatId) {
    document.getElementById('newChatBtn').click();
  }
}

/**
 * Select a model on the server
 */
async function selectModel(name, notify = true) {
  await fetch('/api/select_model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
}

/**
 * Handle model selection changes
 */
function initModelSelection() {
  const modelSelect = document.getElementById('modelSelect');
  
  modelSelect.addEventListener('change', async () => {
    const name = modelSelect.value;
    if (name === prevModel) return;
    if (!maybeClearChat()) { 
      modelSelect.value = prevModel; 
      return; 
    }
    updateImageVisibility(models[name]);
    clearImagePreviews();
    clearTextFilePreviews();
    prevModel = name;
    await selectModel(name);
  });
}
