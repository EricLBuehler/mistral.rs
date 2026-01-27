use clap::Parser;
use indexmap::IndexMap;
use mistralrs::SearchEmbeddingModel;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::models::LoadedModel;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// In-situ quantization to apply. Defaults to Q6K on CPU, AFQ6 on Metal.
    /// Options: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    /// HQQ1, HQQ2, HQQ3, HQQ4, HQQ8, AFQ2, AFQ4, AFQ6, AFQ8
    #[arg(long = "isq")]
    pub isq: Option<String>,

    /// Repeated flag for text-only models (HuggingFace model ID or local path)
    #[arg(long = "text-model")]
    pub text_models: Vec<String>,

    /// Repeated flag for vision models (HuggingFace model ID or local path)
    #[arg(long = "vision-model")]
    pub vision_models: Vec<String>,

    /// Repeated flag for speech models (HuggingFace model ID or local path)
    #[arg(long = "speech-model")]
    pub speech_models: Vec<String>,

    /// Enable web search tool (requires embedding model)
    #[arg(long)]
    pub enable_search: bool,

    /// Built-in search embedding model to load (e.g., `embedding_gemma`)
    #[arg(long = "search-embedding-model")]
    pub search_embedding_model: Option<SearchEmbeddingModel>,

    /// Port to listen on (default: 1234)
    #[arg(long = "port", short = 'p')]
    pub port: Option<u16>,

    /// IP address to serve on (default: 0.0.0.0)
    #[arg(long = "host")]
    pub host: Option<String>,

    /// Use CPU only (disable GPU acceleration)
    #[arg(long)]
    pub cpu: bool,

    /// Maximum sequence length for context. If not specified, uses model default.
    #[arg(long = "max-seq-len")]
    pub max_seq_len: Option<usize>,

    /// Default temperature for generation (0.0-2.0). Default: 0.7
    #[arg(long, default_value = "0.7")]
    pub temperature: f64,

    /// Default top_p for generation (0.0-1.0). Default: 0.9
    #[arg(long = "top-p", default_value = "0.9")]
    pub top_p: f64,

    /// Default top_k for generation. Default: 40
    #[arg(long = "top-k", default_value = "40")]
    pub top_k: usize,

    /// Default max tokens to generate. Default: 2048
    #[arg(long = "max-tokens", default_value = "2048")]
    pub max_tokens: usize,

    /// Default repetition penalty (1.0 = no penalty). Default: 1.1
    #[arg(long = "repetition-penalty", default_value = "1.1")]
    pub repetition_penalty: f32,

    /// Default system prompt for all chats
    #[arg(long = "system-prompt")]
    pub system_prompt: Option<String>,
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
    pub models: IndexMap<String, LoadedModel>,
    pub current: RwLock<Option<String>>,
    pub chats_dir: String,
    /// Directory for storing generated speech wav files
    pub speech_dir: String,
    pub current_chat: RwLock<Option<String>>,
    pub next_chat_id: RwLock<u32>,
    /// Default generation parameters from CLI
    pub default_params: GenerationParams,
    /// Whether web search is enabled
    pub search_enabled: bool,
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
