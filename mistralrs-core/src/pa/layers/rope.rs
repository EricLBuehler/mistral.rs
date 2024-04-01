use std::sync::{Arc, OnceLock};

use candle_core::{DType, Device, Tensor};

use crate::pa::kernels::rope::{apply_rotary_embedding, compute_cos_sin_cache};

fn get_cos_sin_cache(
    base: f32,
    rotary_dim: usize,
    max_position_embeddings: usize,
    dtype: DType,
    device: &Device,
) -> Arc<Tensor> {
    static CACHE: OnceLock<Arc<Tensor>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let t = compute_cos_sin_cache(base, rotary_dim, max_position_embeddings, dtype, device)
                .unwrap();
            Arc::new(t)
        })
        .clone()
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    head_size: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    is_neox: bool,
    // pos_encoding: PosEncoding,
    // cos_sin_cache: Tensor,
    // cos: Tensor,
    // sin: Tensor,
    pub(crate) cos_sin_cache: Arc<Tensor>,
}

// fn compute_cos_sin_cache(
//     rotary_dim: usize,
//     base: f32,
//     max_position_embeddings: usize,
//     device: &Device,
//     dtype: DType,
// ) -> candle_core::Result<Tensor> {
//     let inv_freq: Vec<_> = (0..rotary_dim)
//         .step_by(2)
//         .map(|i| 1f32 / base.powf(i as f32 / rotary_dim as f32))
//         .collect();
//     let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
//     let t = Tensor::arange(0, max_position_embeddings as u32, device)?
//         .to_dtype(DType::F32)?
//         .reshape((max_position_embeddings, 1))?;
//     let inv_freq_n = inv_freq.elem_count();
//     let inv_freq = inv_freq.reshape((1, inv_freq_n))?;
//     let freqs = t.matmul(&inv_freq)?;
//     let cos = freqs.cos()?;
//     let sin = freqs.sin()?;
//     let last = cos.dims().len() - 1;
//     Tensor::cat(&[cos, sin], last)
//     // freqs = torch.einsum("i,j -> ij", t, inv_freq)
//     // cos = freqs.cos()
//     // sin = freqs.sin()
//     // cache = torch.cat((cos, sin), dim=-1)
// }

impl RotaryEmbedding {
    pub fn new(
        device: &Device,
        dtype: DType,
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f32,
        is_neox_style: bool,
    ) -> candle_core::Result<Self> {
        // get_cos_sin_cache(base, rotary_dim, max_position_embeddings, dtype, device)
        // let cos_sin_cache = compute_cos_sin_cache(
        //     rotary_dim,
        //     rope_theta,
        //     max_position_embeddings,
        //     device,
        //     dtype,
        // )?;

        // tracing::info!("cos_sin_cache shape:{:?}", cos_sin_cache.shape());
        // Create inv freqs
        // let inv_freqs = candle_rotary::inv_freqs(rotary_dim, 10000f32, device)?;
        // // Create an over-sized cos sin cache like you would usually do
        // let (cos, sin) = candle_rotary::cos_sin(32, &inv_freqs, dtype)?;

        Ok(Self {
            head_size,
            rotary_dim,
            max_position_embeddings,
            rope_theta,
            is_neox: is_neox_style,
            cos_sin_cache: get_cos_sin_cache(
                10000f32,
                rotary_dim,
                max_position_embeddings,
                dtype,
                device,
            ),
        })
    }
    pub fn forward(
        &self,
        position: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> candle_core::Result<()> {
        // // q,k shape  //[batch_size, seq_len, num_heads * head_size]
        let (_b_sz, _seq_len, _hidden_size) = query.dims3()?;
        // let fwd_q = query.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_size))?;
        // let fwd_k = key.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_size))?;
        apply_rotary_embedding(
            position,
            // &fwd_q,
            // &fwd_k,
            query,
            key,
            &self.cos_sin_cache.as_ref(),
            self.head_size,
            self.is_neox,
        )?;

        Ok(())
    }
}

#[test]
fn test_() -> candle_core::Result<()> {
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::IndexOp;
    let cuda_dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new("/data/dev/rust/lmsf/test_qkv")? };
    let qkv = st.load("qkv", &cuda_dev)?;
    let q = qkv.i((.., .., 0..4096))?;
    let k = qkv.i((.., .., 4096..8192))?;
    let _v = qkv.i((.., .., 8192..))?;
    println!("before q:{}", q.to_string());
    println!("before k:{}", k.to_string());
    let position = Tensor::from_slice(&[9_i64], (1, 1), &cuda_dev)?;
    let rotary_emb = RotaryEmbedding::new(&cuda_dev, DType::F16, 128, 128, 4096, 10000.0, true)?;
    rotary_emb.forward(&position, &q, &k)?;
    println!("after q:{}", q.to_string());
    println!("after k:{}", k.to_string());
    Ok(())
}
