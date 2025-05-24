use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use tokio::fs;

use crate::types::{AppState, ChatFile, ChatMessage};

pub async fn append_chat_message(
    app: &Arc<AppState>,
    role: &str,
    content: &str,
    images: Option<Vec<String>>,
) -> Result<()> {
    // Ignore replay helpers sent from the front‑end
    if content.trim_start().starts_with("{\"restore\":") {
        return Ok(());
    }
    let chat_opt = app.current_chat.read().await.clone();
    let Some(chat_id) = chat_opt else {
        return Ok(());
    };
    let path = format!("{}/{}.json", app.chats_dir, chat_id);

    let mut chat: ChatFile = if let Ok(data) = fs::read(&path).await {
        serde_json::from_slice(&data).unwrap_or(ChatFile {
            title: None,
            model: app.current.read().await.clone().unwrap_or_default(),
            kind: String::new(),
            created_at: Utc::now().to_rfc3339(),
            messages: Vec::new(),
        })
    } else {
        ChatFile {
            title: None,
            model: app.current.read().await.clone().unwrap_or_default(),
            kind: String::new(),
            created_at: Utc::now().to_rfc3339(),
            messages: Vec::new(),
        }
    };

    chat.messages.push(ChatMessage {
        role: role.into(),
        content: content.into(),
        images,
    });
    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
}
