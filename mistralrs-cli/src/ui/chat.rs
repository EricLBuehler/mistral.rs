use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::fs;

use crate::ui::handlers::api::MessageStats;
use crate::ui::types::{AppState, ChatFile, ChatMessage};

/// Append a chat message to the specified chat file.
#[allow(clippy::too_many_arguments)]
pub async fn append_chat_message(
    app: &Arc<AppState>,
    chat_id: &str,
    id: Option<String>,
    parent_id: Option<String>,
    role: &str,
    content: &str,
    images: Option<Vec<String>>,
    videos: Option<Vec<String>>,
    blocks: Option<serde_json::Value>,
    finish_reason: Option<String>,
    stats: MessageStats,
) -> Result<()> {
    if content.trim_start().starts_with("{\"restore\":") {
        return Ok(());
    }
    let path = format!("{}/{}.json", app.chats_dir, chat_id);
    let data = match fs::read(&path).await {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let mut chat: ChatFile = serde_json::from_slice(&data)?;
    let msg_id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    chat.messages.push(ChatMessage {
        id: Some(msg_id.clone()),
        parent_id,
        role: role.into(),
        content: content.into(),
        images,
        videos,
        blocks,
        finish_reason,
        elapsed_ms: stats.elapsed_ms,
        ttft_ms: stats.ttft_ms,
        tokens: stats.tokens,
        model: stats.model,
        session_id: stats.session_id,
    });
    chat.tail = Some(msg_id);
    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
}

/// Replace content of a single message. Drops every descendant in the tree.
pub async fn edit_chat_message(
    app: &Arc<AppState>,
    chat_id: &str,
    message_id: &str,
    new_content: &str,
) -> Result<()> {
    let path = format!("{}/{}.json", app.chats_dir, chat_id);
    let data = fs::read(&path).await?;
    let mut chat: ChatFile = serde_json::from_slice(&data)?;

    // Compute the set of descendants of `message_id` and remove them.
    let descendants = descendants_of(&chat, message_id);
    chat.messages
        .retain(|m| m.id.as_deref().is_none_or(|id| !descendants.contains(id)));

    // Rewrite the target's content.
    for m in chat.messages.iter_mut() {
        if m.id.as_deref() == Some(message_id) {
            m.content = new_content.to_string();
            break;
        }
    }
    chat.tail = Some(message_id.to_string());

    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
}

/// Set the active tail (selected branch leaf) for the chat.
pub async fn set_chat_tail(app: &Arc<AppState>, chat_id: &str, tail: Option<String>) -> Result<()> {
    let path = format!("{}/{}.json", app.chats_dir, chat_id);
    let data = fs::read(&path).await?;
    let mut chat: ChatFile = serde_json::from_slice(&data)?;
    chat.tail = tail;
    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
}

fn descendants_of(chat: &ChatFile, root: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut frontier = vec![root.to_string()];
    while let Some(node) = frontier.pop() {
        for m in chat.messages.iter() {
            if m.parent_id.as_deref() == Some(node.as_str()) {
                if let Some(id) = &m.id {
                    if out.insert(id.clone()) {
                        frontier.push(id.clone());
                    }
                }
            }
        }
    }
    out
}
