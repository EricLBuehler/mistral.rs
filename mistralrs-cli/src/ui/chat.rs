use anyhow::Result;
use std::sync::Arc;
use tokio::fs;

use crate::ui::types::{AppState, ChatFile, ChatMessage};

/// Append a chat message to the specified chat file.
pub async fn append_chat_message(
    app: &Arc<AppState>,
    chat_id: &str,
    role: &str,
    content: &str,
    images: Option<Vec<String>>,
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
    chat.messages.push(ChatMessage {
        role: role.into(),
        content: content.into(),
        images,
    });
    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
}
