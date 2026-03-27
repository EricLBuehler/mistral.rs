use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use mistralrs::{ModelGenerationDefaults, SearchEmbeddingModel};

#[derive(Clone, Serialize)]
pub struct UiModelInfo {
    pub name: String,
    pub kind: String,
    pub generation_defaults: GenerationParams,
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
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub system_prompt: Option<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: Some(2048),
            repetition_penalty: Some(1.1),
            system_prompt: None,
        }
    }
}

impl GenerationParams {
    pub fn empty() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            repetition_penalty: None,
            system_prompt: None,
        }
    }

    pub fn from_model_defaults(defaults: Option<&ModelGenerationDefaults>) -> Self {
        let Some(defaults) = defaults else {
            return Self::default();
        };

        let mut params = Self::empty();

        if defaults.do_sample == Some(false) {
            params.temperature = Some(0.0);
            params.top_k = Some(1);
            params.top_p = Some(1.0);
        }

        if let Some(temperature) = defaults.temperature {
            params.temperature = Some(temperature);
        }
        if let Some(top_p) = defaults.top_p {
            params.top_p = Some(top_p);
        }
        if let Some(top_k) = defaults.top_k.filter(|top_k| *top_k > 0) {
            params.top_k = Some(top_k);
        }
        if let Some(max_tokens) = defaults.max_new_tokens {
            params.max_tokens = Some(max_tokens);
        }
        if let Some(repetition_penalty) = defaults.repetition_penalty {
            params.repetition_penalty = Some(repetition_penalty);
        }

        params
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
