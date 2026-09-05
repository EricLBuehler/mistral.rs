#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::paged_attention;

const NUM_HEADS: usize = 40;
const KV_HEADS: usize = 8;
const HEAD_SIZE: usize = 128;
const BLOCK_SIZE: usize = 32;
const MAX_CONTEXT: usize = 2048;
const CACHE_VECTOR_BYTES: usize = 16;
const CASES: [(usize, usize); 6] = [
    (16, 2048),
    (16, 1152),
    (15, 2048),
    (14, 512),
    (16, 2048),
    (32, 1024),
];
const TOLERANCE: f32 = 0.01;

#[test]
fn paged_attention_handles_context_length_reordering() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let num_blocks = MAX_CONTEXT / BLOCK_SIZE;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let vector_width = CACHE_VECTOR_BYTES / dtype.size_in_bytes();
        let keys = Tensor::zeros(
            (
                num_blocks,
                KV_HEADS,
                HEAD_SIZE / vector_width,
                BLOCK_SIZE,
                vector_width,
            ),
            dtype,
            &device,
        )?;
        let values = Tensor::ones(
            (num_blocks, KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
            dtype,
            &device,
        )?;
        for (batch, context_len) in CASES {
            let query = Tensor::zeros((batch, NUM_HEADS, HEAD_SIZE), dtype, &device)?;
            let blocks = Tensor::from_vec(
                (0..num_blocks as u32)
                    .cycle()
                    .take(batch * num_blocks)
                    .collect(),
                (batch, num_blocks),
                &device,
            )?;
            let lengths = Tensor::from_vec(vec![context_len as u32; batch], batch, &device)?;
            let output = paged_attention(
                &query,
                None,
                None,
                &keys,
                &values,
                &blocks,
                &lengths,
                None,
                context_len,
                1.0 / (HEAD_SIZE as f32).sqrt(),
                1.0,
                None,
            )?;
            let output = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                output.iter().all(|value| (value - 1.0).abs() < TOLERANCE),
                "{dtype:?}, batch {batch}, context {context_len}"
            );
        }
    }
    Ok(())
}
