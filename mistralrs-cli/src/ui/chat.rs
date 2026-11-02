use anyhow::{Context, Result};
use chrono::Utc;
use regex::{Regex, RegexBuilder};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::MutexGuard;
use tracing::error;

use crate::ui::types::{
    chat_id_from_filename, AppState, ChatFile, ChatIndexEntry, ChatMessage, ChatSearchHit,
    ChatSummary,
};

const TMP_SUFFIX: &str = "tmp";
const PREVIEW_CHARS: usize = 80;
const SNIPPET_CONTEXT_CHARS: usize = 40;
const MAX_SEARCH_TERMS: usize = 8;
const SNIPPET_ELLIPSIS: &str = "...";

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

pub fn summarize(id: String, chat: &ChatFile, updated_at: String) -> ChatSummary {
    let preview = chat
        .messages
        .iter()
        .find(|m| m.role == "user" && !m.content.trim().is_empty())
        .map(|m| m.content.trim().chars().take(PREVIEW_CHARS).collect());
    ChatSummary {
        id,
        title: chat.title.clone(),
        model: chat.model.clone(),
        created_at: chat.created_at.clone(),
        updated_at,
        preview,
    }
}

fn index_entry(id: &str, chat: &ChatFile, mtime: std::time::SystemTime) -> ChatIndexEntry {
    let updated_at = chrono::DateTime::<Utc>::from(mtime).to_rfc3339();
    let mut text = chat.title.clone().unwrap_or_default();
    for m in &chat.messages {
        text.push('\n');
        text.push_str(&m.content);
    }
    ChatIndexEntry {
        mtime,
        summary: summarize(id.to_string(), chat, updated_at),
        text,
    }
}

/// Rescans chats_dir, re-reading only files whose mtime changed, and returns the up-to-date index.
pub async fn refresh_chat_index(
    app: &Arc<AppState>,
) -> Result<MutexGuard<'_, HashMap<String, ChatIndexEntry>>> {
    let mut entries = fs::read_dir(&app.chats_dir).await?;
    let mut index = app.chat_index.lock().await;
    let mut seen = HashSet::new();
    while let Some(entry) = entries.next_entry().await? {
        let filename = entry.file_name().to_string_lossy().to_string();
        let Some(id) = chat_id_from_filename(&filename) else {
            continue;
        };
        let Ok(mtime) = entry.metadata().await.and_then(|m| m.modified()) else {
            continue;
        };
        seen.insert(id.to_string());
        if index.get(id).is_some_and(|e| e.mtime == mtime) {
            continue;
        }
        match read_chat(&entry.path()).await {
            Ok(chat) => {
                index.insert(id.to_string(), index_entry(id, &chat, mtime));
            }
            Err(e) => {
                index.remove(id);
                error!("skipping unreadable chat {}: {}", filename, e);
            }
        }
    }
    index.retain(|id, _| seen.contains(id));
    Ok(index)
}

/// Case-insensitive terms; a chat matches when every term appears in its title or messages.
pub fn search_terms(query: &str) -> Vec<Regex> {
    query
        .split_whitespace()
        .take(MAX_SEARCH_TERMS)
        .filter_map(|t| {
            RegexBuilder::new(&regex::escape(t))
                .case_insensitive(true)
                .build()
                .ok()
        })
        .collect()
}

pub fn search_entry(entry: &ChatIndexEntry, terms: &[Regex]) -> Option<ChatSearchHit> {
    if terms.is_empty() || !terms.iter().all(|t| t.is_match(&entry.text)) {
        return None;
    }
    let first = terms[0].find(&entry.text)?;
    Some(ChatSearchHit {
        summary: entry.summary.clone(),
        snippet: snippet(&entry.text, first.start(), first.end()),
    })
}

fn snippet(text: &str, start: usize, end: usize) -> String {
    let from = text[..start]
        .char_indices()
        .rev()
        .nth(SNIPPET_CONTEXT_CHARS - 1)
        .map_or(0, |(i, _)| i);
    let to = text[end..]
        .char_indices()
        .nth(SNIPPET_CONTEXT_CHARS)
        .map_or(text.len(), |(i, _)| end + i);
    let body = text[from..to]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let lead = if from > 0 { SNIPPET_ELLIPSIS } else { "" };
    let trail = if to < text.len() {
        SNIPPET_ELLIPSIS
    } else {
        ""
    };
    format!("{lead}{body}{trail}")
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

    #[test]
    fn summary_previews_first_user_message() {
        let mut c = chat();
        upsert_message(&mut c, msg("a", &"x".repeat(200)));
        let s = summarize("chat_1".into(), &c, "t".into());
        assert_eq!(s.preview.unwrap().len(), PREVIEW_CHARS);
    }

    fn entry(text: &str) -> ChatIndexEntry {
        let c = chat();
        ChatIndexEntry {
            mtime: std::time::SystemTime::UNIX_EPOCH,
            summary: summarize("chat_1".into(), &c, "t".into()),
            text: text.into(),
        }
    }

    #[test]
    fn search_requires_every_term_case_insensitively() {
        let e = entry("How do I tune the KV Cache for long prompts?");
        assert!(search_entry(&e, &search_terms("kv PROMPTS")).is_some());
        assert!(search_entry(&e, &search_terms("kv gpu")).is_none());
        assert!(search_entry(&e, &search_terms("   ")).is_none());
        assert!(search_entry(&e, &search_terms("c++")).is_none());
    }

    #[test]
    fn snippet_trims_context_on_char_boundaries() {
        let text = format!("{}needle{}", "\u{e9}".repeat(100), "\u{fc}".repeat(100));
        let hit = search_entry(&entry(&text), &search_terms("NEEDLE")).unwrap();
        let expected = format!(
            "...{}needle{}...",
            "\u{e9}".repeat(SNIPPET_CONTEXT_CHARS),
            "\u{fc}".repeat(SNIPPET_CONTEXT_CHARS)
        );
        assert_eq!(hit.snippet, expected);
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
