use super::*;
use candle_core::{cuda::cudarc::driver::sys, Var};

const BF16_ABSOLUTE_TOLERANCE: f32 = 0.006;
const F16_ABSOLUTE_TOLERANCE: f32 = 0.0007;
const EXTRA_CACHE_PAGES: usize = 7;
const RANDOM_MULTIPLIER: u64 = 6364136223846793005;
const RANDOM_INCREMENT: u64 = 1442695040888963407;

#[derive(Clone, Copy)]
struct Case {
    sequences: usize,
    query_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    ratio: usize,
    topk: usize,
    capacity: usize,
}

struct Inputs {
    case: Case,
    dtype: DType,
    q: Var,
    key: Var,
    value: Var,
    tables: Var,
    lens: Var,
    selected: Var,
    counts: Var,
    query: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    pages: Vec<i32>,
    lengths: Vec<u32>,
    selection: Vec<i32>,
    selected_counts: Vec<i32>,
}

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(value).to_f32(),
        DType::F16 => f16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

fn value(index: usize, seed: usize, dtype: DType) -> f32 {
    rounded(
        (((index * 73 + index / 17 + seed * 251) % 2039) as f32 - 1019.0) / 1024.0,
        dtype,
    )
}

impl Inputs {
    fn new(case: Case, dtype: DType, device: &Device, query_device: &Device) -> Result<Self> {
        let tokens = case.sequences * case.query_len;
        let table_pages = case.capacity.div_ceil(case.block_size);
        let pool_pages = case.sequences * table_pages + EXTRA_CACHE_PAGES;
        let cache_shape = (pool_pages, case.kv_heads, case.block_size, case.head_dim);
        let cache_len = pool_pages * case.kv_heads * case.block_size * case.head_dim;
        Ok(Self {
            case,
            dtype,
            q: Var::zeros((tokens, case.heads, case.head_dim), dtype, query_device)?,
            key: Var::zeros(cache_shape, dtype, device)?,
            value: Var::zeros(cache_shape, dtype, device)?,
            tables: Var::zeros((case.sequences, table_pages), DType::I32, device)?,
            lens: Var::zeros(case.sequences, DType::U32, device)?,
            selected: Var::zeros((tokens, case.topk), DType::I32, device)?,
            counts: Var::zeros(tokens, DType::I32, device)?,
            query: vec![0.0; tokens * case.heads * case.head_dim],
            keys: vec![f32::NAN; cache_len],
            values: vec![f32::NAN; cache_len],
            pages: vec![0; case.sequences * table_pages],
            lengths: vec![0; case.sequences],
            selection: vec![0; tokens * case.topk],
            selected_counts: vec![0; tokens],
        })
    }

    fn update(&mut self, context: usize, request: usize) -> Result<()> {
        let case = self.case;
        let table_pages = case.capacity.div_ceil(case.block_size);
        let pool_pages = case.sequences * table_pages + EXTRA_CACHE_PAGES;
        let mut permutation = (0..pool_pages).collect::<Vec<_>>();
        let mut random = request as u64 + 701;
        for index in (1..pool_pages).rev() {
            random = random
                .wrapping_mul(RANDOM_MULTIPLIER)
                .wrapping_add(RANDOM_INCREMENT);
            permutation.swap(index, random as usize % (index + 1));
        }
        self.keys.fill(f32::NAN);
        self.values.fill(f32::NAN);
        for (index, query) in self.query.iter_mut().enumerate() {
            *query = value(index, request + 1, self.dtype);
        }
        for seq in 0..case.sequences {
            let length = context.saturating_sub(seq * case.query_len);
            self.lengths[seq] = length as u32;
            for page in 0..table_pages {
                self.pages[seq * table_pages + page] = permutation[seq * table_pages + page] as i32;
            }
            for position in 0..length {
                let physical = self.pages[seq * table_pages + position / case.block_size] as usize;
                for head in 0..case.kv_heads {
                    let base = ((physical * case.kv_heads + head) * case.block_size
                        + position % case.block_size)
                        * case.head_dim;
                    for dim in 0..case.head_dim {
                        let logical = ((seq * case.capacity + position) * case.kv_heads + head)
                            * case.head_dim
                            + dim;
                        self.keys[base + dim] = value(logical, request + 3, self.dtype);
                        self.values[base + dim] = value(logical, request + 7, self.dtype);
                    }
                }
            }
        }
        for token in 0..case.sequences * case.query_len {
            let sequence = token / case.query_len;
            let position = self.lengths[sequence] as isize - case.query_len as isize
                + (token % case.query_len) as isize;
            let blocks = if position < 0 {
                0
            } else {
                (position as usize + 1) / case.ratio
            };
            let count = blocks.min(case.topk);
            self.selected_counts[token] = count as i32;
            for index in 0..case.topk {
                self.selection[token * case.topk + index] = if index < count {
                    ((index + request * 7 + token) % blocks) as i32
                } else {
                    -1
                };
            }
        }
        for (var, data) in [
            (&self.q, &self.query),
            (&self.key, &self.keys),
            (&self.value, &self.values),
        ] {
            var.set(
                &Tensor::from_vec(data.clone(), var.shape(), var.device())?.to_dtype(self.dtype)?,
            )?;
        }
        self.tables.set(&Tensor::from_vec(
            self.pages.clone(),
            self.tables.shape(),
            self.tables.device(),
        )?)?;
        self.lens.set(&Tensor::from_vec(
            self.lengths.clone(),
            self.lens.shape(),
            self.lens.device(),
        )?)?;
        self.selected.set(&Tensor::from_vec(
            self.selection.clone(),
            self.selected.shape(),
            self.selected.device(),
        )?)?;
        self.counts.set(&Tensor::from_vec(
            self.selected_counts.clone(),
            self.counts.shape(),
            self.counts.device(),
        )?)?;
        Ok(())
    }

    fn run(&self, sparse: bool, device: &CudaDevice) -> Result<Tensor> {
        let paged = QsaPaged {
            block_tables: &self.tables,
            max_blocks_per_seq: self.tables.dim(1)?,
            block_size: self.case.block_size,
        };
        try_cutile_qsa_decode(
            &QsaDecodeArgs {
                q: &self.q,
                key_cache: &self.key,
                value_cache: &self.value,
                paged: &paged,
                kv_lens: &self.lens,
                selection: sparse.then_some((&self.selected, &self.counts)),
                q_len: self.case.query_len,
                n_kv_heads: self.case.kv_heads,
                ratio: self.case.ratio,
                topk: self.case.topk,
                scale: (self.case.head_dim as f32).sqrt().recip(),
            },
            device,
        )?
        .ok_or_else(|| candle_core::Error::msg("supported QSA decode shape was declined"))
    }

    fn reference(&self, sparse: bool) -> Vec<f32> {
        let case = self.case;
        let table_pages = self.pages.len() / case.sequences;
        let mut output = vec![0.0; self.query.len()];
        for token in 0..case.sequences * case.query_len {
            let seq = token / case.query_len;
            let position = self.lengths[seq] as isize - case.query_len as isize
                + (token % case.query_len) as isize;
            if position < 0 {
                continue;
            }
            let length = position as usize + 1;
            let mut items = Vec::new();
            if sparse {
                for &block in &self.selection
                    [token * case.topk..token * case.topk + self.selected_counts[token] as usize]
                {
                    items.extend(block as usize * case.ratio..(block as usize + 1) * case.ratio);
                }
                items.extend(length / case.ratio * case.ratio..length);
            } else {
                items.extend(0..length);
            }
            for head in 0..case.heads {
                let kv_head = head / (case.heads / case.kv_heads);
                let base = |position: usize| {
                    ((self.pages[seq * table_pages + position / case.block_size] as usize
                        * case.kv_heads
                        + kv_head)
                        * case.block_size
                        + position % case.block_size)
                        * case.head_dim
                };
                let query = &self.query[(token * case.heads + head) * case.head_dim..];
                let logits = items
                    .iter()
                    .map(|&position| {
                        (0..case.head_dim)
                            .map(|dim| {
                                f64::from(query[dim]) * f64::from(self.keys[base(position) + dim])
                            })
                            .sum::<f64>()
                            / (case.head_dim as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = logits
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f64>();
                for dim in 0..case.head_dim {
                    output[(token * case.heads + head) * case.head_dim + dim] = if total > 0.0 {
                        (items
                            .iter()
                            .zip(&weights)
                            .map(|(&position, weight)| {
                                weight * f64::from(self.values[base(position) + dim])
                            })
                            .sum::<f64>()
                            / total) as f32
                    } else {
                        0.0
                    };
                }
            }
        }
        output
    }

    fn check(&self, output: &Tensor, sparse: bool) -> Result<()> {
        let actual = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = self.reference(sparse);
        let tolerance = if self.dtype == DType::F16 {
            F16_ABSOLUTE_TOLERANCE
        } else {
            BF16_ABSOLUTE_TOLERANCE
        };
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                got.is_finite() && (got - want).abs() < tolerance,
                "{:?} sparse={sparse} index={index}: {got} vs {want}",
                self.dtype
            );
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires a CUDA device with cuTile support"]
fn qsa_decode_graph_tracks_metadata_and_matches_independent_attention() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let cases = [
        (
            Case {
                sequences: 1,
                query_len: 1,
                heads: 24,
                kv_heads: 2,
                head_dim: 256,
                block_size: 32,
                ratio: 4,
                topk: 512,
                capacity: 8195,
            },
            [2048, 2051, 2052, 8195],
        ),
        (
            Case {
                sequences: 1,
                query_len: 2,
                heads: 6,
                kv_heads: 2,
                head_dim: 256,
                block_size: 16,
                ratio: 4,
                topk: 23,
                capacity: 160,
            },
            [0, 1, 130, 159],
        ),
        (
            Case {
                sequences: 2,
                query_len: 1,
                heads: 8,
                kv_heads: 2,
                head_dim: 128,
                block_size: 16,
                ratio: 2,
                topk: 32,
                capacity: 129,
            },
            [1, 31, 64, 129],
        ),
    ];
    for (case, contexts) in cases {
        warm_decode(
            QsaWarmShape {
                ratio: case.ratio,
                topk: case.topk,
                n_q_heads: case.heads,
                n_kv_heads: case.kv_heads,
                block_size: case.block_size,
            },
            cuda,
        )?;
        for dtype in [DType::BF16, DType::F16] {
            let mut inputs = Inputs::new(case, dtype, &device, &device)?;
            for sparse in [false, true] {
                inputs.update(contexts[0], 0)?;
                let compiled = cutile::tile_kernel::jit_compile_count();
                inputs.check(&inputs.run(sparse, cuda)?, sparse)?;
                assert_eq!(
                    cutile::tile_kernel::jit_compile_count(),
                    compiled,
                    "startup warmup missed a decode specialization"
                );
                device.synchronize()?;
                unsafe { stream.context().disable_event_tracking() };
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .map_err(candle_core::Error::wrap)?;
                let output = inputs.run(sparse, cuda);
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                unsafe { stream.context().enable_event_tracking() };
                let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
                let output = output?;
                assert_eq!(
                    cutile::tile_kernel::jit_compile_count(),
                    compiled,
                    "capture compiled a new QSA specialization"
                );
                for (request, context) in contexts.into_iter().enumerate() {
                    inputs.update(context, request)?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    inputs.check(&output, sparse)?;
                    assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device with cuTile support"]
fn qsa_decode_orders_secondary_stream_inputs_and_preserves_fallbacks() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let query_device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let case = Case {
        sequences: 1,
        query_len: 1,
        heads: 3,
        kv_heads: 1,
        head_dim: 128,
        block_size: 16,
        ratio: 4,
        topk: 16,
        capacity: 83,
    };
    for dtype in [DType::BF16, DType::F16] {
        let mut inputs = Inputs::new(case, dtype, &device, &query_device)?;
        for request in 0..3 {
            inputs.update(case.capacity - request, request)?;
            inputs.check(&inputs.run(true, cuda)?, true)?;
        }
        let paged = QsaPaged {
            block_tables: &inputs.tables,
            max_blocks_per_seq: inputs.tables.dim(1)?,
            block_size: case.block_size,
        };
        let float_q = inputs.q.to_dtype(DType::F32)?;
        let wrong_cache = inputs.key.reshape((1, ()))?;
        for (q, key) in [
            (&float_q, inputs.key.as_tensor()),
            (inputs.q.as_tensor(), &wrong_cache),
        ] {
            assert!(try_cutile_qsa_decode(
                &QsaDecodeArgs {
                    q,
                    key_cache: key,
                    value_cache: &inputs.value,
                    paged: &paged,
                    kv_lens: &inputs.lens,
                    selection: Some((&inputs.selected, &inputs.counts)),
                    q_len: 1,
                    n_kv_heads: case.kv_heads,
                    ratio: case.ratio,
                    topk: case.topk,
                    scale: (case.head_dim as f32).sqrt().recip(),
                },
                cuda
            )?
            .is_none());
        }
    }
    Ok(())
}

#[test]
fn qsa_decode_index_and_alignment_guards_reject_unsupported_views() -> Result<()> {
    assert_eq!(
        indexed_product(&[MAX_INDEXED_ELEMENTS]),
        Some(MAX_INDEXED_ELEMENTS)
    );
    assert_eq!(indexed_product(&[MAX_INDEXED_ELEMENTS, 2]), None);
    assert_eq!(indexed_product(&[usize::MAX, usize::MAX]), None);
    assert_eq!(indexed_product(&[32, 24, 48, 256]), Some(9_437_184));
    for dtype in [DType::BF16, DType::F16, DType::I32, DType::U32] {
        let input = Tensor::zeros(32, dtype, &Device::Cpu)?;
        let aligned_offset = POINTER_ALIGNMENT / dtype.size_in_bytes();
        assert!(aligned_operand(&input.narrow(0, aligned_offset, 1)?));
        assert!(!aligned_operand(&input.narrow(0, 1, 1)?));
        assert!(!aligned_operand(&input.reshape((4, 8))?.t()?));
    }
    Ok(())
}
