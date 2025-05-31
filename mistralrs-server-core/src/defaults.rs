/// Provides the default values used for the server core
/// For example, they can be used as CLI args, etc.
pub const DEVICE: Option<candle_core::Device> = None;
pub const SEED: Option<u64> = None;
pub const LOG: Option<String> = None;
pub const TRUNCATE_SEQUENCE: bool = false;
pub const MODEL: Option<mistralrs_core::ModelSelected> = None;
pub const MAX_SEQS: usize = 16;
pub const NO_KV_CACHE: bool = false;
pub const CHAT_TEMPLATE: Option<String> = None;
pub const JINJA_EXPLICIT: Option<String> = None;
pub const INTERACTIVE_MODE: bool = false;
pub const PREFIX_CACHE_N: usize = 16;
pub const NUM_DEVICE_LAYERS: Option<Vec<String>> = None;
pub const IN_SITU_QUANT: Option<String> = None;
pub const PAGED_ATTN_GPU_MEM: Option<usize> = None;
pub const PAGED_ATTN_GPU_MEM_USAGE: Option<f32> = None;
pub const PAGED_CTXT_LEN: Option<usize> = None;
pub const PAGED_ATTN_BLOCK_SIZE: Option<usize> = None;
pub const NO_PAGED_ATTN: bool = false;
pub const PAGED_ATTN: bool = false;
pub const PROMPT_CHUNKSIZE: Option<usize> = None;
pub const CPU: bool = false;
pub const ENABLE_SEARCH: bool = false;
pub const SEARCH_BERT_MODEL: Option<String> = None;
pub const ENABLE_THINKING: bool = false;
pub const DEFAULT_TOKEN_SOURCE: mistralrs_core::TokenSource =
    mistralrs_core::TokenSource::CacheToken;

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb
pub const MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;
