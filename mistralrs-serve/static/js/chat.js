// Chat management functionality

// Global chat state
let currentChatId = null;
let pendingClear = false;
let pendingChatRestore = null;

function sendChatRestore(chatId, messages) {
  if (typeof ws === 'undefined' || ws.readyState !== WebSocket.OPEN) {
    pendingChatRestore = { chatId, messages };
    return;
  }

  ws.send(JSON.stringify({ chat_id: chatId }));
  messages.forEach(m => {
    ws.send(JSON.stringify({
      restore: {
        role: m.role,
        content: m.content,
        images: m.images || []
      }
    }));
  });

  pendingChatRestore = null;
}

function flushPendingChatRestore() {
  if (pendingChatRestore) {
    sendChatRestore(pendingChatRestore.chatId, pendingChatRestore.messages);
  }
}

window.flushPendingChatRestore = flushPendingChatRestore;

/**
 * Refresh the chat list in the sidebar
 */
async function refreshChatList() {
  const res = await fetch(apiUrl('api/list_chats'));
  const data = await res.json();
  const chatList = document.getElementById('chatList');
  
  // Sort chats by creation timestamp, newest first
  data.chats.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  chatList.innerHTML = '';
  
  data.chats.forEach((c, idx) => {
    const li = document.createElement('li');
    li.dataset.id = c.id;
    li.onclick = () => loadChat(c.id);
    
    // Highlight active chat
    if (c.id === currentChatId) {
      li.classList.add('active');
    }

    // Determine label: use title if set, otherwise reverse-number by creation order
    const display = c.title && c.title.trim()
      ? c.title
      : `Chat #${data.chats.length - idx}`;

    const titleDiv = document.createElement('div');
    titleDiv.className = 'chat-title';
    titleDiv.textContent = display;

    const dateDiv = document.createElement('div');
    dateDiv.className = 'chat-date';
    dateDiv.textContent = new Date(c.created_at).toLocaleString();

    // Delete button (visible on hover)
    const delBtn = document.createElement('button');
    delBtn.className = 'chat-delete-btn';
    delBtn.title = 'Delete chat';
    delBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    delBtn.onclick = (ev) => {
      ev.stopPropagation();
      if (!confirm('Delete this chat permanently?')) return;
      fetch(apiUrl('api/delete_chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: c.id })
      }).then(res => {
        if (!res.ok) { alert('Failed to delete chat'); return; }
        if (currentChatId === c.id) {
          currentChatId = null;
          document.getElementById('log').innerHTML = '';
        }
        refreshChatList();
      });
    };

    li.appendChild(titleDiv);
    li.appendChild(dateDiv);
    li.appendChild(delBtn);
    chatList.appendChild(li);
  });
}

/**
 * Find the most recent blank chat for a given model
 */
/**
 * Find the most recent blank chat for a given model.
 * Uses message_count from list_chats to avoid N+1 HTTP requests.
 */
async function findBlankChat(model) {
  const res = await fetch(apiUrl('api/list_chats'));
  if (!res.ok) return null;
  const data = await res.json();

  // Sort by creation time, newest first
  data.chats.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

  for (const chat of data.chats) {
    // Check if it's the same model and has no messages
    if (chat.model === model && (chat.message_count || 0) === 0) {
      return chat.id;
    }
  }

  return null;
}

/**
 * Load a specific chat
 */
async function loadChat(id) {
  if (!maybeClearChat(true)) return;
  
  const res = await fetch(apiUrl('api/load_chat'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id })
  });
  
  if (!res.ok) { 
    alert('Failed to load chat'); 
    return; 
  }
  
  const data = await res.json();
  const log = document.getElementById('log');
  const modelSelect = document.getElementById('modelSelect');
  const chatList = document.getElementById('chatList');
  
  currentChatId = data.id;
  document.querySelectorAll('#chatList li').forEach(li => li.classList.remove('active'));
  const activeLi = document.querySelector(`#chatList li[data-id="${id}"]`);
  if (activeLi) activeLi.classList.add('active');
  
  if (data.model && models[data.model]) {
    modelSelect.value = data.model;
    prevModel = data.model;
    updateImageVisibility(models[data.model]);
    await selectModel(data.model);
  }
  
  log.innerHTML = '';
  clearImagePreviews();
  clearTextFilePreviews();
  clearAudioPreviews();
  
  data.messages.forEach(m => {
    // ---- render text ----
    const div = append(renderMarkdown(m.content),
                       m.role === 'user' ? 'user' : 'assistant');

    // ---- render any saved images ----
    if (m.images && m.images.length) {
      const imgWrap = document.createElement('div');
      imgWrap.className = 'chat-images';
      imgWrap.style.display = 'flex';
      imgWrap.style.flexWrap = 'wrap';
      imgWrap.style.gap = '1rem';
      m.images.forEach(src => {
        const im = document.createElement('img');
        im.src = src;
        im.className = 'chat-preview';
        imgWrap.appendChild(im);
      });
      div.appendChild(imgWrap);
    }

  });

  // Restore server-side context for this chat
  sendChatRestore(id, data.messages);
}

/**
 * Clear the current chat with optional confirmation
 */
function maybeClearChat(skipConfirm = false) {
  const log = document.getElementById('log');
  
  if (log.children.length === 0) return true;
  if (skipConfirm || confirm('Clear the current draft conversation?')) {
    pendingClear = true;
    if (ws.readyState === WebSocket.OPEN) ws.send(CLEAR_CMD);
    log.innerHTML = '';
    return true;
  }
  return false;
}

/**
 * Initialize chat management event handlers
 */
function initChatHandlers() {
  const newChatBtn = document.getElementById('newChatBtn');
  const clearBtn = document.getElementById('clearBtn');
  const renameBtn = document.getElementById('renameBtn');
  const deleteBtn = document.getElementById('deleteBtn');

  newChatBtn.addEventListener('click', async () => {
    if (!prevModel) { 
      alert('Select a model first'); 
      return; 
    }
    
    // Check if there's already a blank chat with the same model
    const blankChatId = await findBlankChat(prevModel);
    
    if (blankChatId) {
      // Clear current UI state before loading
      document.getElementById('log').innerHTML = '';
      clearImagePreviews();
      clearTextFilePreviews();
      clearAudioPreviews();
      
      // Load the existing blank chat instead of creating a new one
      await loadChat(blankChatId);
      await refreshChatList();
      return;
    }
    
    // No blank chat found, create a new one
    const res = await fetch(apiUrl('api/new_chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: prevModel })
    });
    if (!res.ok) { 
      alert('Failed to create new chat'); 
      return; 
    }
    const { id } = await res.json();
    // Load and activate the new chat
    await loadChat(id);
    await refreshChatList();
  });

  clearBtn.addEventListener('click', () => {
    const log = document.getElementById('log');
    if (log.children.length === 0) return;
    if (!confirm('Clear the chat history?')) return;
    pendingClear = true;
    if (ws.readyState === WebSocket.OPEN) ws.send(CLEAR_CMD);
    log.innerHTML = '';
    clearImagePreviews();
    clearTextFilePreviews();
    clearAudioPreviews();
  });

  renameBtn.addEventListener('click', async () => {
    if (!currentChatId) { 
      alert('No chat selected'); 
      return; 
    }
    const newTitle = prompt('Enter new chat name:', '');
    if (newTitle && newTitle.trim()) {
      const res = await fetch(apiUrl('api/rename_chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: currentChatId, title: newTitle.trim() })
      });
      if (res.ok) {
        await refreshChatList();
      } else {
        alert('Failed to rename chat');
      }
    }
  });

  deleteBtn.addEventListener('click', async () => {
    if (!currentChatId) { 
      alert('No chat selected'); 
      return; 
    }
    if (!confirm('Delete this chat permanently?')) return;
    const res = await fetch(apiUrl('api/delete_chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: currentChatId })
    });
    if (!res.ok) { 
      alert('Failed to delete chat'); 
      return; 
    }
    currentChatId = null;
    document.getElementById('log').innerHTML = '';
    clearImagePreviews();
    clearTextFilePreviews();
    clearAudioPreviews();
    await refreshChatList();
    
    // Move to newest chat if any, otherwise create a fresh one
    const chatList = document.getElementById('chatList');
    const firstLi = chatList.querySelector('li');
    if (firstLi) {
      loadChat(firstLi.dataset.id);
    } else if (prevModel) {
      newChatBtn.click();
    }
  });
}
