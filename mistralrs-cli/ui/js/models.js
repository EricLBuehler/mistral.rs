// Model management functionality

// Global model state
const models = Object.create(null); // name -> kind
let prevModel = null;

/**
 * Refresh the list of available models
 */
async function refreshModels() {
  const res = await fetch(apiUrl('api/list_models'));
  const data = await res.json();
  const modelSelect = document.getElementById('modelSelect');
  
  const prevValue = modelSelect.value;
  modelSelect.innerHTML = '';
  Object.keys(models).forEach(k => delete models[k]);
  
  data.models.forEach(m => {
    models[m.name] = m.kind;
    const opt = document.createElement('option');
    opt.value = m.name;
    const statusIcon = {
      'loaded': '\u25CF',
      'pending': '\u25CB',
      'reloading': '\u27F3',
      'unloaded': '\u25CB'
    }[m.status] || '';
    opt.textContent = m.status && m.status !== 'loaded' 
      ? `${statusIcon} ${m.name} (${m.status})`
      : m.name;
    opt.dataset.status = m.status || 'loaded';
    modelSelect.appendChild(opt);
  });
  
  // Restore previous selection if still available
  if (prevValue && [...modelSelect.options].some(o => o.value === prevValue)) {
    modelSelect.value = prevValue;
  }
  
  if (modelSelect.options.length && !modelSelect.value) {
    modelSelect.selectedIndex = 0;
  }
  
  prevModel = modelSelect.value;
  if (prevModel) {
    updateImageVisibility(models[prevModel]);
    await selectModel(prevModel, false);
  }
  
  await refreshChatList();
  if (!currentChatId) {
    document.getElementById('newChatBtn').click();
  }

  // Periodically refresh model list to catch status changes
  if (!refreshModels._intervalSet) {
    refreshModels._intervalSet = true;
    setInterval(refreshModels, 30000);
  }
}

/**
 * Select a model on the server
 */
async function selectModel(name, notify = true) {
  await fetch(apiUrl('api/select_model'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  await loadSettings();
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
    clearAudioPreviews();
    prevModel = name;
    await selectModel(name);
  });
}
