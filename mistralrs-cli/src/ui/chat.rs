use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::fs;

use crate::ui::types::{AppState, ChatFile, ChatMessage};

const TMP_SUFFIX: &str = "tmp";

pub async fn read_chat(path: &Path) -> Result<ChatFile> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("chat not found: {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

// rename is atomic on the same filesystem, so a crash mid-write never leaves a truncated chat file
pub async fn write_chat(path: &Path, chat: &ChatFile) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".");
    tmp.push(TMP_SUFFIX);
    fs::write(&tmp, serde_json::to_vec_pretty(chat)?).await?;
    fs::rename(&tmp, path).await?;
    Ok(())
}

pub async fn update_chat<F>(app: &Arc<AppState>, chat_id: &str, f: F) -> Result<()>
where
    F: FnOnce(&mut ChatFile),
{
    let path = app.chat_path(chat_id).context("invalid chat id")?;
    let _guard = app.chat_write_lock.lock().await;
    let mut chat = read_chat(&path).await?;
    f(&mut chat);
    write_chat(&path, &chat).await
}

pub async fn append_chat_message(
    app: &Arc<AppState>,
    chat_id: &str,
    msg: ChatMessage,
) -> Result<()> {
    if msg.content.trim_start().starts_with("{\"restore\":") {
        return Ok(());
    }
    update_chat(app, chat_id, |chat| upsert_message(chat, msg)).await
}

pub async fn set_chat_tail(app: &Arc<AppState>, chat_id: &str, tail: Option<String>) -> Result<()> {
    update_chat(app, chat_id, |chat| chat.tail = tail).await
}

// client retries resend the same message id, so replace instead of duplicating
fn upsert_message(chat: &mut ChatFile, mut msg: ChatMessage) {
    let id = msg
        .id
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone();
    match chat
        .messages
        .iter_mut()
        .find(|m| m.id.as_deref() == Some(id.as_str()))
    {
        Some(existing) => *existing = msg,
        None => chat.messages.push(msg),
    }
    chat.tail = Some(id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat() -> ChatFile {
        ChatFile {
            title: None,
            model: "m".into(),
            kind: "text".into(),
            created_at: "2026-11-01T00:00:00Z".into(),
            messages: Vec::new(),
            session_id: None,
            tail: None,
        }
    }

    fn msg(id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            id: Some(id.into()),
            role: "user".into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn upsert_replaces_retried_message() {
        let mut c = chat();
        upsert_message(&mut c, msg("a", "first"));
        upsert_message(&mut c, msg("b", "second"));
        upsert_message(&mut c, msg("a", "retried"));
        assert_eq!(c.messages.len(), 2);
        assert_eq!(c.messages[0].content, "retried");
        assert_eq!(c.tail.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn write_chat_leaves_no_tmp_file() {
        let dir = std::env::temp_dir().join(format!("mistralrs-ui-chat-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).await.unwrap();
        let path = dir.join("chat_1.json");
        let mut c = chat();
        upsert_message(&mut c, msg("a", "hi"));
        write_chat(&path, &c).await.unwrap();
        assert_eq!(read_chat(&path).await.unwrap().messages[0].content, "hi");
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("chat_1.json")]);
        fs::remove_dir_all(&dir).await.unwrap();
    }
}
