pub trait BlockEngineSequence {
    fn blocks_to_add_new_tok(&self) -> usize;
    fn get_id(&self) -> usize;
}

pub trait BlockEngineSequenceGroup {
    fn get_total_logical_token_blocks(&self) -> usize;
    fn total_blocks_to_add_new_tok(&self) -> usize;
    fn seq_ids(&self) -> &[usize];
}
