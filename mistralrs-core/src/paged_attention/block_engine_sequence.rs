use super::LogicalTokenBlock;

pub trait BlockEngineSequence {
    fn blocks_to_add_new_tok(&self) -> usize;
    fn physical_blocks_prefill(&self) -> &Option<Vec<usize>>;
    fn get_id(&self) -> usize;
    fn logical_token_blocks(&self) -> &[LogicalTokenBlock];
}
