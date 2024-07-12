pub trait BlockEngineSequence {
    fn blocks_to_add_new_tok(&self) -> usize;
    fn get_id(&self) -> usize;
    fn get_logical_token_blocks(&self) -> usize;
}
