use std::collections::HashMap;

use candle_core::Tensor;

/// # Safety
/// Unsafe due to passing pointers
pub unsafe fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) {
    todo!()
}

pub fn swap_blocks(src: Tensor, dst: &mut Tensor, block_mapping: HashMap<usize, usize>) {
    todo!()
}
