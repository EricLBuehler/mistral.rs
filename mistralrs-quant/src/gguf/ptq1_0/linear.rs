//! CPU linear over PTQ1_0 blocks read straight from the GGUF mmap, applying the Hadamard fold to activations.

use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use super::cuda::PackedWeights;
use super::{cpu::packed_matmul, dequantize_row, PTQ1_0_BLOCK_BYTES, PTQ1_0_BLOCK_ELEMS};
use crate::gguf::{archive::GgufArchive, hadamard::RowTransform};
use crate::{
    IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde, UnquantLinear,
};

#[derive(Debug)]
pub struct Ptq1_0Linear {
    archive: Arc<GgufArchive>,
    name: String,
    out_dim: usize,
    in_dim: usize,
    transform: Option<RowTransform>,
    bias: Option<Tensor>,
    dtype: DType,
    device: Device,
    #[cfg(feature = "cuda")]
    gpu: Option<PackedWeights>,
}

impl Ptq1_0Linear {
    pub fn new(
        archive: Arc<GgufArchive>,
        name: &str,
        transform: Option<RowTransform>,
        bias: Option<Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = archive.tensor_info(name)?.shape().to_vec();
        let &[out_dim, in_dim] = shape.as_slice() else {
            candle_core::bail!("PTQ1_0 linear `{name}` must be rank 2, got {shape:?}");
        };
        if !in_dim.is_multiple_of(PTQ1_0_BLOCK_ELEMS) {
            candle_core::bail!("PTQ1_0 linear `{name}` width {in_dim} is not a multiple of 128");
        }
        #[cfg(feature = "cuda")]
        let gpu = if device.is_cuda() {
            let bytes = archive.tensor_data(name)?.bytes();
            Some(PackedWeights::upload(bytes, transform.as_ref(), device)?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        if !device.is_cpu() {
            candle_core::bail!(
                "PTQ1_0 packed linear `{name}` needs the cuda feature for {device:?}"
            );
        }
        Ok(Self {
            archive,
            name: name.to_string(),
            out_dim,
            in_dim,
            transform,
            bias,
            dtype,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            gpu,
        })
    }

    fn row_bytes(&self) -> usize {
        self.in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES
    }

    fn weight_bytes(&self) -> Result<&[u8]> {
        Ok(self.archive.tensor_data(&self.name)?.bytes())
    }

    fn require_cpu(tensor: &Tensor) -> Result<()> {
        if !tensor.device().is_cpu() {
            candle_core::bail!("PTQ1_0 packed linear only runs on CPU activations");
        }
        Ok(())
    }
}

impl QuantMethod for Ptq1_0Linear {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("PTQ1_0 linears are only built from a GGUF archive")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        let mut data = vec![0f32; self.out_dim * self.in_dim];
        let row_bytes = self.row_bytes();
        data.par_chunks_mut(self.in_dim)
            .zip(self.weight_bytes()?.par_chunks(row_bytes))
            .for_each(|(dst, src)| dequantize_row(src, dst));
        if let Some(transform) = &self.transform {
            transform.unfold_weight(&mut data);
        }
        Tensor::from_vec(data, (self.out_dim, self.in_dim), &Device::Cpu)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            let y = gpu.matmul(a, self.out_dim, self.in_dim)?;
            return match &self.bias {
                Some(bias) => y.broadcast_add(&bias.to_dtype(y.dtype())?),
                None => Ok(y),
            };
        }
        Self::require_cpu(a)?;
        if self
            .transform
            .as_ref()
            .is_some_and(RowTransform::is_inverse)
        {
            candle_core::bail!(
                "PTQ1_0 linear `{}`: inverse-role tensors only support embedding lookups",
                self.name
            );
        }
        let dims = a.dims().to_vec();
        if dims.last() != Some(&self.in_dim) {
            candle_core::bail!("PTQ1_0 linear `{}` got input shape {dims:?}", self.name);
        }
        let tokens = a.elem_count() / self.in_dim;
        let mut xt = a
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if let Some(transform) = &self.transform {
            xt.par_chunks_mut(self.in_dim)
                .for_each_init(Vec::new, |scratch, row| transform.apply(row, scratch));
        }
        let y = packed_matmul(self.weight_bytes()?, self.out_dim, self.in_dim, &xt, tokens);
        let mut y = Tensor::from_vec(y, (tokens, self.out_dim), &Device::Cpu)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }
        let mut out_dims = dims;
        *out_dims.last_mut().expect("checked non-empty above") = self.out_dim;
        y.to_dtype(a.dtype())?.reshape(out_dims)
    }

    fn embedding_forward(&self, ids: &Tensor, output_dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            return gpu.embedding(ids, self.in_dim, output_dtype);
        }
        Self::require_cpu(ids)?;
        let dims = ids.dims().to_vec();
        let flat = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
        let bytes = self.weight_bytes()?;
        let row_bytes = self.row_bytes();
        let mut out = vec![0f32; flat.len() * self.in_dim];
        out.par_chunks_mut(self.in_dim)
            .zip(flat.par_iter())
            .for_each_init(Vec::new, |scratch, (dst, id)| {
                let start = *id as usize * row_bytes;
                dequantize_row(&bytes[start..start + row_bytes], dst);
                if let Some(transform) = &self.transform {
                    transform.apply(dst, scratch);
                }
            });
        let mut out_dims = dims;
        out_dims.push(self.in_dim);
        Tensor::from_vec(out, out_dims, &Device::Cpu)?.to_dtype(output_dtype)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.device.clone())
    }

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dtype,
            self.device.clone(),
            vec![self.out_dim, self.in_dim],
            request,
            false,
        ))
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("PTQ1_0 packed linear does not support LoRA deltas")
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let dense = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.dequantize_w()?,
            self.bias.clone(),
        )))?;
        Arc::new(dense).apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl QuantizedSerde for Ptq1_0Linear {
    fn name(&self) -> &'static str {
        "ptq1_0"
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{
        super::cpu::{K_TILE_BLOCKS, ROWS_PER_TASK, TOKEN_BLOCK, TOKEN_TILE},
        *,
    };
    #[cfg(feature = "cuda")]
    use crate::gguf::hadamard::HadamardRole;
    use crate::gguf::ptq1_0::encode_block;

    const INT8_REL_ERR: f32 = 2e-2; // int8 activations, one scale per 128 columns
    const LCG_MUL: u64 = 6364136223846793005;

    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(LCG_MUL)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    fn synthetic(out_dim: usize, in_dim: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
        let mut s = 7u64;
        let mut bytes = Vec::new();
        for _ in 0..out_dim * in_dim / PTQ1_0_BLOCK_ELEMS {
            let mut codes = [0u8; PTQ1_0_BLOCK_ELEMS];
            codes.iter_mut().for_each(|c| *c = (lcg(&mut s) % 3) as u8);
            bytes.extend_from_slice(&encode_block(
                &codes,
                0.01 + (lcg(&mut s) % 100) as f32 * 1e-4,
            ));
        }
        let x = (0..tokens * in_dim)
            .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        (bytes, x)
    }

    #[test]
    fn packed_matmul_matches_dequantized_dense() {
        let wide = (K_TILE_BLOCKS + 3) * PTQ1_0_BLOCK_ELEMS;
        for in_dim in [3 * PTQ1_0_BLOCK_ELEMS, wide] {
            for tokens in [1, 2, TOKEN_BLOCK + 1, TOKEN_TILE + TOKEN_BLOCK + 1] {
                check_against_dense(in_dim, tokens);
            }
        }
    }

    fn check_against_dense(in_dim: usize, tokens: usize) {
        let out_dim = ROWS_PER_TASK * 2 + 5;
        let (bytes, x) = synthetic(out_dim, in_dim, tokens);
        let mut w = vec![0f32; out_dim * in_dim];
        w.chunks_mut(in_dim)
            .zip(bytes.chunks(in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES))
            .for_each(|(d, s)| dequantize_row(s, d));
        let got = packed_matmul(&bytes, out_dim, in_dim, &x, tokens);
        let (mut err, mut norm) = (0f32, 0f32);
        for t in 0..tokens {
            for r in 0..out_dim {
                let want: f32 = (0..in_dim)
                    .map(|c| w[r * in_dim + c] * x[t * in_dim + c])
                    .sum();
                let g = got[t * out_dim + r];
                err += (g - want).powi(2);
                norm += want.powi(2);
            }
        }
        assert!(
            err.sqrt() <= INT8_REL_ERR * norm.sqrt(),
            "in_dim {in_dim} tokens {tokens}: rel err {}",
            err.sqrt() / norm.sqrt()
        );
    }

    #[test]
    #[ignore = "timing"]
    fn packed_matmul_speed() {
        // the last shape is ~156 MB of weights so it streams from DRAM instead of L3
        for (out_dim, in_dim, tokens) in [
            (5120, 17408, 1),
            (5120, 17408, 16),
            (5120, 17408, 64),
            (5120, 17408, 256),
            (40960, 17408, 1),
        ] {
            let (bytes, x) = synthetic(out_dim, in_dim, tokens);
            packed_matmul(&bytes, out_dim, in_dim, &x, tokens);
            let start = Instant::now();
            let reps = 5;
            for _ in 0..reps {
                std::hint::black_box(packed_matmul(&bytes, out_dim, in_dim, &x, tokens));
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            let gbps = bytes.len() as f64 / secs / 1e9;
            eprintln!(
                "tokens {tokens} rows {out_dim}: {:.1} ms, {gbps:.2} GB/s of weights",
                secs * 1e3
            );
        }
        let (out_dim, in_dim) = (40960, 17408);
        let (bytes, x) = synthetic(out_dim, in_dim, 1);
        let path = std::env::temp_dir().join("ptq1_0_speed.bin");
        std::fs::write(&path, &bytes).unwrap();
        let map = unsafe { memmap2::Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
        packed_matmul(&map, out_dim, in_dim, &x, 1);
        let start = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            std::hint::black_box(packed_matmul(&map, out_dim, in_dim, &x, 1));
        }
        let secs = start.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "tokens 1 rows {out_dim} mmap: {:.1} ms, {:.2} GB/s of weights",
            secs * 1e3,
            bytes.len() as f64 / secs / 1e9
        );
        std::fs::remove_file(&path).ok();
        let bytes = vec![1u8; 1 << 28];
        let start = Instant::now();
        let sum: u64 = bytes
            .par_chunks(1 << 20)
            .map(|c| c.iter().map(|b| *b as u64).sum::<u64>())
            .sum();
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "memory read reference: {:.2} GB/s (sum {sum})",
            bytes.len() as f64 / secs / 1e9
        );
    }

    #[test]
    #[ignore = "timing"]
    fn packed_matmul_gap_overhead() {
        let (out_dim, in_dim) = (5120, 17408);
        let (bytes, x) = synthetic(out_dim, in_dim, 1);
        packed_matmul(&bytes, out_dim, in_dim, &x, 1);
        for gap_us in [0u64, 20, 100, 300] {
            let reps = 200;
            let mut busy = std::time::Duration::ZERO;
            for _ in 0..reps {
                let start = Instant::now();
                std::hint::black_box(packed_matmul(&bytes, out_dim, in_dim, &x, 1));
                busy += start.elapsed();
                let gap = Instant::now();
                while gap.elapsed() < std::time::Duration::from_micros(gap_us) {
                    std::hint::spin_loop();
                }
            }
            eprintln!(
                "gap {gap_us} us: {:.0} us per matmul",
                busy.as_secs_f64() * 1e6 / reps as f64
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_matches_cpu_packed() -> Result<()> {
        use crate::gguf::ptq1_0::cuda::PackedWeights;

        let dev = Device::new_cuda(0)?;
        let cases = [
            (384, 1, false),
            (2048, 1, true),
            (2048, 3, true),
            (2048, 4, false),
            (2048, 5, false),
            (2048, 12, false),
            (2048, 40, true),
            (3072, 300, true),
            (45056, 1, false), // staged activations exceed the shared-memory budget
            (45056, 4, false),
        ];
        for (in_dim, tokens, folded) in cases {
            let out_dim = ROWS_PER_TASK * 2 + 5;
            let (bytes, x) = synthetic(out_dim, in_dim, tokens);
            let transform = folded
                .then(|| RowTransform::for_test(HadamardRole::Fold, in_dim, 11, in_dim > 2048));
            let gpu = PackedWeights::upload(&bytes, transform.as_ref(), &dev)?;
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let input = Tensor::from_vec(x.clone(), (tokens, in_dim), &dev)?.to_dtype(dtype)?;
                let rounded = input
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut xt = rounded.clone();
                if let Some(transform) = &transform {
                    xt.chunks_mut(in_dim)
                        .for_each(|row| transform.apply(row, &mut Vec::new()));
                }
                let want = packed_matmul(&bytes, out_dim, in_dim, &xt, tokens);
                let got = gpu
                    .matmul(&input, out_dim, in_dim)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let norm = want.iter().map(|w| w * w).sum::<f32>().sqrt();
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (g - w).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    err / norm < INT8_REL_ERR,
                    "{dtype:?} in {in_dim} tokens {tokens} folded {folded}: {}",
                    err / norm
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_embedding_matches_cpu() -> Result<()> {
        use crate::gguf::ptq1_0::cuda::PackedWeights;

        let dev = Device::new_cuda(0)?;
        let (vocab, width) = (50, 3072);
        let (bytes, _) = synthetic(vocab, width, 1);
        let transform = RowTransform::for_test(HadamardRole::Inverse, width, 5, false);
        let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
        let ids = vec![3u32, 0, 49, 7, 7, 21];
        let row_bytes = width / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES;
        let mut want = Vec::new();
        for id in &ids {
            let mut row = vec![0f32; width];
            let start = *id as usize * row_bytes;
            dequantize_row(&bytes[start..start + row_bytes], &mut row);
            transform.apply(&mut row, &mut Vec::new());
            want.extend(row);
        }
        let input = Tensor::from_vec(ids, (2, 3), &dev)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let got = gpu.embedding(&input, width, dtype)?;
            assert_eq!(got.dims(), [2, 3, width]);
            let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let norm = want.iter().map(|w| w * w).sum::<f32>().sqrt();
            let err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).powi(2))
                .sum::<f32>()
                .sqrt();
            let tol = if dtype == DType::F32 { 1e-4 } else { 1e-2 };
            assert!(err / norm < tol, "{dtype:?}: {}", err / norm);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_matmul_speed() -> Result<()> {
        use crate::gguf::ptq1_0::cuda::PackedWeights;

        const REPS: usize = 50;
        let dev = Device::new_cuda(0)?;
        let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            f()?;
            dev.synchronize()?;
            let start = Instant::now();
            for _ in 0..REPS {
                f()?;
            }
            dev.synchronize()?;
            Ok(start.elapsed().as_secs_f64() / REPS as f64)
        };

        eprintln!("production kernel, 1 token, GB/s of weights");
        let shapes = [
            (17408, 5120, "ffn gate/up"),
            (5120, 17408, "ffn down"),
            (10240, 5120, "gdn qkv"),
            (6144, 5120, "gdn gate"),
            (5120, 6144, "ssm out"),
            (12288, 5120, "attn q"),
            (1024, 5120, "attn k/v"),
            (248320, 5120, "output"),
        ];
        for (out_dim, in_dim, label) in shapes {
            let (bytes, x) = synthetic(out_dim, in_dim, 1);
            let transform = RowTransform::for_test(HadamardRole::Fold, in_dim, 11, false);
            let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
            let input = Tensor::from_vec(x, (1, in_dim), &dev)?.to_dtype(DType::BF16)?;
            let secs = time(&|| gpu.matmul(&input, out_dim, in_dim))?;
            eprintln!(
                "{out_dim:>6} x {in_dim:<5} {label:<12} {:>6.1} MB  {:5.0}",
                bytes.len() as f64 / 1e6,
                bytes.len() as f64 / secs / 1e9
            );
        }

        let (out_dim, in_dim) = (5120, 17408);
        let (bytes, _) = synthetic(out_dim, in_dim, 1);
        let transform = RowTransform::for_test(HadamardRole::Fold, in_dim, 11, false);
        let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
        for tokens in [1, 2, 3, 4, 8, 59, 256] {
            let x = vec![0.5f32; tokens * in_dim];
            let input = Tensor::from_vec(x, (tokens, in_dim), &dev)?.to_dtype(DType::BF16)?;
            let secs = time(&|| gpu.matmul(&input, out_dim, in_dim))?;
            eprintln!(
                "production kernel, tokens {tokens}: {:.3} ms, {:.1} GB/s of weights",
                secs * 1e3,
                bytes.len() as f64 / secs / 1e9
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_batch_size_invariance() -> Result<()> {
        use crate::gguf::ptq1_0::cuda::PackedWeights;

        let (out_dim, in_dim, max_tokens) = (2048, 5120, 40);
        let dev = Device::new_cuda(0)?;
        let (bytes, x) = synthetic(out_dim, in_dim, max_tokens);
        let transform = RowTransform::for_test(HadamardRole::Fold, in_dim, 11, false);
        let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
        let run = |first: usize, rows: usize| -> Result<Vec<f32>> {
            let data = x[first * in_dim..(first + rows) * in_dim].to_vec();
            let input = Tensor::from_vec(data, (rows, in_dim), &dev)?;
            gpu.matmul(&input, out_dim, in_dim)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let alone = (0..max_tokens)
            .map(|t| run(t, 1))
            .collect::<Result<Vec<_>>>()?;
        for tokens in [2, 3, 4, 8, 15, 16, 40] {
            let batch = run(0, tokens)?;
            let (mut differing, mut worst) = (0, 0f32);
            for (t, want) in alone.iter().take(tokens).enumerate() {
                let got = &batch[t * out_dim..(t + 1) * out_dim];
                let diff = got
                    .iter()
                    .zip(want)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0f32, f32::max);
                differing += usize::from(
                    got.iter()
                        .zip(want)
                        .any(|(g, w)| g.to_bits() != w.to_bits()),
                );
                worst = worst.max(diff);
            }
            eprintln!(
                "batch invariance: {tokens:>2} tokens, {differing} of {tokens} rows differ from the single-token result, max abs diff {worst:.2e}"
            );
        }
        Ok(())
    }
}
