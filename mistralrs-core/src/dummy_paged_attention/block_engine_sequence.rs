use std::sync::Arc;

use super::{LogicalTokenBlock, PhysicalTokenBlock};

pub trait BlockEngineSequence {
    fn blocks_to_add_new_tok(&self) -> usize;
    fn take_physical_blocks_prefill(&mut self) -> Option<Vec<Arc<PhysicalTokenBlock>>>;
    fn get_id(&self) -> usize;
    fn logical_token_blocks(&self) -> &[LogicalTokenBlock];
    /// Returns the previous count
    fn increment_waitlist_count(&mut self) -> usize;
}
