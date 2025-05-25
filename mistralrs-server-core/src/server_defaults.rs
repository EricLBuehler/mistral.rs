/// Provides the default values used for the server core
/// For example, they can be used as CLI args, etc.
pub const TRUNCATE_SEQUENCE: bool = false;
pub const MAX_SEQS: usize = 16;
pub const NO_KV_CACHE: bool = false;
pub const INTERACTIVE_MODE: bool = false;
pub const PREFIX_CACHE_N: usize = 16;
pub const NO_PAGED_ATTN: bool = false;
pub const PAGED_ATTN: bool = false;
pub const CPU: bool = false;
pub const ENABLE_SEARCH: bool = false;
pub const ENABLE_THINKING: bool = false;
pub const DEFAULT_TOKEN_SOURCE: mistralrs_core::TokenSource =
    mistralrs_core::TokenSource::CacheToken;

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb
pub const MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;

pub fn default_none<T>() -> Option<T> {
    None
}

// Helper function for placeholder model (used in Default impl)
pub fn placeholder_model() -> crate::ModelSelected {
    crate::ModelSelected::Toml {
        file: String::from("/this/is/just/a/placeholder"),
    }
}
