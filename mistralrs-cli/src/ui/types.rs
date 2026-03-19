use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use mistralrs::SearchEmbeddingModel;

#[derive(Clone, Serialize)]
pub struct UiModelInfo {
    pub name: String,
    pub kind: String,
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

/// Default generation parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
    pub system_prompt: Option<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 2048,
            repetition_penalty: 1.1,
            system_prompt: None,
        }
    }
}

pub struct AppState {
    pub model: mistralrs::Model,
    pub models: IndexMap<String, UiModelInfo>,
    pub current: RwLock<Option<String>>,
    pub chats_dir: String,
    /// Directory for storing generated speech wav files
    pub speech_dir: String,
    pub current_chat: RwLock<Option<String>>,
    pub next_chat_id: RwLock<u32>,
    /// Default generation parameters
    pub default_params: GenerationParams,
    /// Whether web search is enabled
    pub search_enabled: bool,
    /// Search embedding model to use (if enabled)
    pub search_embedding_model: Option<SearchEmbeddingModel>,
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
