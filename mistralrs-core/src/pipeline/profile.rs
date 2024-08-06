use crate::{PagedAttentionConfig, Pipeline};

pub fn profile(
    pipeline: &dyn Pipeline,
    block_size: Option<usize>,
    max_num_batched_tokens: Option<usize>,
) -> PagedAttentionConfig {
    todo!()
}
