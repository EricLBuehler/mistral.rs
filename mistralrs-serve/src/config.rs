//! Plain config structs for the server.
//! These are the public interface — mistralrs-cli converts clap args into these.

use std::path::PathBuf;

use mistralrs::SearchEmbeddingModel;
use mistralrs_core::{PagedCacheType, TokenSource};

/// HTTP server configuration
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub ui: bool,
}

/// Runtime inference options
pub struct RuntimeConfig {
    pub max_seqs: usize,
    pub no_kv_cache: bool,
    pub prefix_cache_n: usize,
    pub enable_search: bool,
    pub search_embedding_model: Option<SearchEmbeddingModel>,
    pub chat_template: Option<PathBuf>,
    pub jinja_explicit: Option<PathBuf>,
}

/// Global options (token source, seed, logging)
pub struct GlobalConfig {
    pub token_source: TokenSource,
    pub seed: Option<u64>,
    pub log: Option<PathBuf>,
}

/// Paged attention settings (extracted from ModelType by the CLI)
pub struct PagedAttnConfig {
    pub paged_attn: Option<bool>,
    pub paged_attn_gpu_mem: Option<usize>,
    pub paged_attn_gpu_mem_usage: Option<f32>,
    pub paged_ctxt_len: Option<usize>,
    pub paged_attn_block_size: Option<usize>,
    pub paged_cache_type: PagedCacheType,
}

/// Device settings (extracted from ModelType by the CLI)
pub struct DeviceConfig {
    pub cpu: bool,
    pub device_layers: Option<Vec<String>>,
}
