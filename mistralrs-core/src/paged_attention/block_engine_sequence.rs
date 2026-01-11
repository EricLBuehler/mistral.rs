use super::{BlockRef, LogicalTokenBlock};

pub trait BlockEngineSequence {
    fn blocks_to_add_new_tok(&self) -> usize;
    fn take_physical_blocks_prefill(&mut self) -> Option<Vec<BlockRef>>;
    fn get_id(&self) -> usize;
    fn logical_token_blocks(&self) -> &[LogicalTokenBlock];
    /// Returns the previous count
    fn increment_waitlist_count(&mut self) -> usize;
    /// Set the number of prefix tokens that are cached (KV already computed).
    fn set_prefix_cache_len(&mut self, len: usize);
    /// Get the block size for this sequence
    fn block_size(&self) -> usize;
}
