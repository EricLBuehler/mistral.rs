use std::collections::HashMap;

pub trait SequenceGroup {
    fn get_total_logical_token_blocks(&self) -> usize;
    fn total_blocks_to_add_new_tok(&self) -> usize;
    fn get_seqs(&self) -> &HashMap<usize, impl Sequence>;
}

pub trait Sequence {
    fn get_id(&self) -> usize;
    fn blocks_to_add_new_tok(&self) -> usize;
}
