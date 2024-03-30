//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

use candle_core::Tensor;

/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
pub mod block_engine;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
pub mod cache_engine;

pub const _PAD_SLOT_ID: i64 = -1;

pub struct InputMetadata {
    pub prompt_lens: Vec<usize>,
    pub max_context_len: Option<usize>,
    pub block_tables: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub slot_mapping: Tensor,
    pub is_prompt: bool,
    pub kv_cache_dtype: String,
}

impl InputMetadata {
    /// prompt_lens: Lengths of prompts.
    /// slot_mapping: The address to write the new KV to of each token.
    /// context_lens: the length of attention context for each generation token.
    /// max_context_len: The maximum context length.
    /// block_tables: The block tables. (Seq id -> list of physical block)
    /// kv_cache_dtype: KV cache datatype (auto or fp8_e5m2)
    pub fn new(
        prompt_lens: Vec<usize>,
        max_context_len: Option<usize>,
        block_tables: Option<Tensor>,
        context_lens: Option<Tensor>,
        slot_mapping: Tensor,
        kv_cache_dtype: String,
    ) -> Self {
        let is_prompt = !prompt_lens.is_empty();
        Self {
            prompt_lens,
            max_context_len,
            block_tables,
            context_lens,
            slot_mapping,
            is_prompt,
            kv_cache_dtype,
        }
    }
}

#[allow(dead_code)]
pub struct PreparedInputs {
    pub tokens: Tensor,
    pub positions: Tensor,
    pub metadata: InputMetadata,
}
