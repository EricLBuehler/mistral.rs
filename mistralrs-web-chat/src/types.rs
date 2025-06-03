use clap::Parser;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::models::LoadedModel;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Cli {
    /// In-situ quantization to apply, defaults to 6-bit.
    #[arg(long = "isq")]
    pub isq: Option<String>,

    /// Repeated flag for text‑only models
    #[arg(long = "text-model")]
    pub text_models: Vec<String>,

    /// Repeated flag for vision models
    #[arg(long = "vision-model")]
    pub vision_models: Vec<String>,

    /// Repeated flag for speech models
    #[arg(long = "speech-model")]
    pub speech_models: Vec<String>,
    /// Enable web search tool (requires embedding model)
    #[arg(long)]
    pub enable_search: bool,
    /// Hugging Face model ID for search embeddings (default: SnowflakeArcticEmbedL if --enable-search)
    #[arg(long = "search-bert-model")]
    pub search_bert_model: Option<String>,

    /// Port to listen on (default: 8080)
    #[arg(long = "port")]
    pub port: Option<u16>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatFile {
    #[serde(default)]
    pub title: Option<String>,
    pub model: String,
    pub kind: String,
    pub created_at: String,
    pub messages: Vec<ChatMessage>,
}

pub struct AppState {
    pub models: IndexMap<String, LoadedModel>,
    pub current: RwLock<Option<String>>,
    pub chats_dir: String,
    /// Directory for storing generated speech wav files
    pub speech_dir: String,
    pub current_chat: RwLock<Option<String>>,
    pub next_chat_id: RwLock<u32>,
}

// Request/Response types
#[derive(Deserialize)]
pub struct SelectRequest {
    pub name: String,
}

#[derive(Deserialize)]
pub struct NewChatRequest {
    pub model: String,
}

#[derive(Deserialize)]
pub struct DeleteChatRequest {
    pub id: String,
}

#[derive(Deserialize)]
pub struct LoadChatRequest {
    pub id: String,
}

#[derive(Deserialize)]
pub struct RenameChatRequest {
    pub id: String,
    pub title: String,
}
