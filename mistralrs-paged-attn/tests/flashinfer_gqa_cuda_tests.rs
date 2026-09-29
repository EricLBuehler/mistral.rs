#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Storage, Tensor};
use mistralrs_paged_attn::{
    flashinfer_decode, FlashInferDecodeScratch, KvCacheScales, DEFAULT_FP8_KV_CACHE_SCALES, USE_FP8,
};
use std::sync::Mutex;

const GROUP_SIZES: [usize; 6] = [3, 4, 5, 6, 7, 8];
const HEAD_DIMS: [usize; 2] = [64, 128];
const BATCH: usize = 2;
const KV_HEADS: usize = 3;
const PAGE_SIZE: usize = 16;
const CONTEXTS: [usize; BATCH] = [29, 67];
const REPLAY_CONTEXTS: [usize; BATCH] = [17, 65];
const CHUNK_SIZE: usize = 32;
const WINDOW_LEFT: usize = 11;
const SOFT_CAP: f32 = 0.7;
const FP8_SCALES: KvCacheScales = KvCacheScales { k: 0.75, v: 0.625 };
const BF16_TOLERANCE: f32 = 0.006;
const F16_TOLERANCE: f32 = 0.002;
const F32_TOLERANCE: f32 = 0.00003;
static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

struct Fixture {
    group_size: usize,
    head_dim: usize,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    indptr: Tensor,
    indices: Tensor,
    last_page_len: Tensor,
    request_indices: Tensor,
    tile_indices: Tensor,
    o_indptr: Tensor,
    chunk_size: Tensor,
    valid_mask: Tensor,
    tmp_v: Tensor,
    tmp_s: Tensor,
    scales: KvCacheScales,
    query_values: Vec<f32>,
    key_values: Vec<f32>,
    value_values: Vec<f32>,
    page_indices: Vec<usize>,
    page_indptr: Vec<usize>,
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()
}

fn copy_i32_metadata(source: &Tensor, destination: &Tensor) -> Result<()> {
    use candle_core::cuda::cudarc::driver::{result, DevicePtr};

    let (source_storage, source_layout) = source.storage_and_layout();
    let (destination_storage, destination_layout) = destination.storage_and_layout();
    let Storage::Cuda(source_storage) = &*source_storage else {
        unreachable!()
    };
    let Storage::Cuda(destination_storage) = &*destination_storage else {
        unreachable!()
    };
    let source_view = source_storage
        .as_cuda_slice::<i32>()?
        .slice(source_layout.start_offset()..source_layout.start_offset() + BATCH);
    let destination_view = destination_storage
        .as_cuda_slice::<i32>()?
        .slice(destination_layout.start_offset()..destination_layout.start_offset() + BATCH);
    let stream = destination.device().as_cuda_device()?.cuda_stream();
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let (source_ptr, _source_guard) = source_view.device_ptr(&stream);
    let (destination_ptr, _destination_guard) = destination_view.device_ptr(&stream);
    // Both fixture buffers stay alive through replay on this stream, preserving the captured addresses.
    unsafe {
        result::memcpy_dtod_async(
            destination_ptr,
            source_ptr,
            BATCH * std::mem::size_of::<i32>(),
            stream.cu_stream(),
        )
    }
    .map_err(candle_core::Error::wrap)
}

fn data(length: usize, multiplier: usize, offset: usize) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * multiplier + offset) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

impl Fixture {
    fn new(
        device: &Device,
        group_size: usize,
        head_dim: usize,
        dtype: DType,
        cache_dtype: DType,
        split: bool,
    ) -> Result<Self> {
        let heads = group_size * KV_HEADS;
        let query = Tensor::from_vec(
            data(BATCH * heads * head_dim, 29, 13),
            (BATCH, heads, head_dim),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        let mut page_indptr = vec![0];
        for context in CONTEXTS {
            page_indptr.push(page_indptr.last().unwrap() + context.div_ceil(PAGE_SIZE));
        }
        let num_pages = *page_indptr.last().unwrap();
        let page_indices: Vec<_> = (0..num_pages).rev().collect();
        let cache_shape = (num_pages, KV_HEADS, PAGE_SIZE, head_dim);
        let cache_length = num_pages * KV_HEADS * PAGE_SIZE * head_dim;
        let key = Tensor::from_vec(data(cache_length, 17, 7), cache_shape, &Device::Cpu)?
            .to_dtype(cache_dtype)?
            .to_device(device)?;
        let value = Tensor::from_vec(data(cache_length, 43, 31), cache_shape, &Device::Cpu)?
            .to_dtype(cache_dtype)?
            .to_device(device)?;
        let mut request_indices = Vec::new();
        let mut tile_indices = Vec::new();
        let mut o_indptr = vec![0i32];
        for (request, context) in CONTEXTS.into_iter().enumerate() {
            let chunks = if split {
                context.div_ceil(CHUNK_SIZE)
            } else {
                1
            };
            for tile in 0..chunks {
                request_indices.push(request as i32);
                tile_indices.push(tile as i32);
            }
            o_indptr.push(request_indices.len() as i32);
        }
        let mut valid_mask = vec![1u8; request_indices.len()];
        if split {
            request_indices.push(0);
            tile_indices.push(0);
            valid_mask.push(0);
        }
        let padded = request_indices.len();
        Ok(Self {
            group_size,
            head_dim,
            query_values: values(&query)?,
            key_values: values(&key)?,
            value_values: values(&value)?,
            query,
            key,
            value,
            indptr: Tensor::from_vec(
                page_indptr.iter().map(|value| *value as i32).collect(),
                BATCH + 1,
                device,
            )?,
            indices: Tensor::from_vec(
                page_indices.iter().map(|value| *value as i32).collect(),
                num_pages,
                device,
            )?,
            last_page_len: Self::last_page_lengths(device, CONTEXTS)?,
            request_indices: Tensor::from_vec(request_indices, padded, device)?,
            tile_indices: Tensor::from_vec(tile_indices, padded, device)?,
            o_indptr: Tensor::from_vec(o_indptr, BATCH + 1, device)?,
            chunk_size: Tensor::new(&[CHUNK_SIZE as i32], device)?,
            valid_mask: Tensor::from_vec(valid_mask, padded, device)?,
            tmp_v: Tensor::zeros((padded, heads, head_dim), dtype, device)?,
            tmp_s: Tensor::zeros((padded, heads), DType::F32, device)?,
            scales: if cache_dtype == DType::F8E4M3 {
                FP8_SCALES
            } else {
                DEFAULT_FP8_KV_CACHE_SCALES
            },
            page_indices,
            page_indptr,
        })
    }

    fn last_page_lengths(device: &Device, contexts: [usize; BATCH]) -> Result<Tensor> {
        Tensor::from_vec(
            contexts
                .into_iter()
                .map(|context| ((context - 1) % PAGE_SIZE + 1) as i32)
                .collect(),
            BATCH,
            device,
        )
    }

    fn decode(&self, window: Option<usize>, soft_cap: Option<f32>) -> Result<Tensor> {
        flashinfer_decode(
            &self.query,
            &self.key,
            &self.value,
            self.scales,
            &self.indptr,
            &self.indices,
            &self.last_page_len,
            &self.request_indices,
            &self.tile_indices,
            &self.o_indptr,
            &self.chunk_size,
            &self.valid_mask,
            1.0 / (self.head_dim as f32).sqrt(),
            window,
            soft_cap,
            Some(FlashInferDecodeScratch {
                tmp_v: &self.tmp_v,
                tmp_s: &self.tmp_s,
            }),
        )
    }

    fn reference(
        &self,
        contexts: [usize; BATCH],
        sign: f32,
        window: Option<usize>,
        soft_cap: Option<f32>,
    ) -> Vec<f32> {
        let heads = self.group_size * KV_HEADS;
        let mut output = vec![0.0; BATCH * heads * self.head_dim];
        let scale = self.scales.k as f64 / (self.head_dim as f64).sqrt();
        for (batch, context) in contexts.into_iter().enumerate() {
            let start = window.map_or(0, |window| context.saturating_sub(window + 1));
            for head in 0..heads {
                let kv_head = head / self.group_size;
                let q_base = (batch * heads + head) * self.head_dim;
                let mut logits = Vec::new();
                let mut offsets = Vec::new();
                for token in start..context {
                    let page = self.page_indices[self.page_indptr[batch] + token / PAGE_SIZE];
                    let base = ((page * KV_HEADS + kv_head) * PAGE_SIZE + token % PAGE_SIZE)
                        * self.head_dim;
                    let mut logit = (0..self.head_dim)
                        .map(|dim| {
                            (self.query_values[q_base + dim] * sign) as f64
                                * self.key_values[base + dim] as f64
                        })
                        .sum::<f64>()
                        * scale;
                    if let Some(cap) = soft_cap {
                        logit = (logit / cap as f64).tanh() * cap as f64;
                    }
                    logits.push(logit);
                    offsets.push(base);
                }
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = logits.iter().map(|logit| (logit - max).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for dim in 0..self.head_dim {
                    output[q_base + dim] = (weights
                        .iter()
                        .zip(&offsets)
                        .map(|(weight, base)| weight * self.value_values[base + dim] as f64)
                        .sum::<f64>()
                        * self.scales.v as f64
                        / denominator) as f32;
                }
            }
        }
        output
    }

    fn assert_close(&self, actual: &Tensor, expected: &[f32]) -> Result<()> {
        let tolerance = match actual.dtype() {
            DType::BF16 => BF16_TOLERANCE,
            DType::F16 => F16_TOLERANCE,
            DType::F32 => F32_TOLERANCE,
            _ => unreachable!(),
        };
        for (index, (actual, expected)) in values(actual)?.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "G={} D={} dtype={:?}/{:?} index={index}: {actual} != {expected}",
                self.group_size,
                self.head_dim,
                self.query.dtype(),
                self.key.dtype(),
            );
        }
        Ok(())
    }
}

#[test]
fn flashinfer_gqa_matches_dense_reference() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for group_size in GROUP_SIZES {
        for head_dim in HEAD_DIMS {
            for dtype in [DType::F16, DType::BF16, DType::F32] {
                let cache_dtypes = if USE_FP8 {
                    vec![dtype, DType::F8E4M3]
                } else {
                    vec![dtype]
                };
                for cache_dtype in cache_dtypes {
                    for split in [false, true] {
                        let fixture =
                            Fixture::new(&device, group_size, head_dim, dtype, cache_dtype, split)?;
                        for (window, soft_cap) in [
                            (None, None),
                            (Some(WINDOW_LEFT), None),
                            (None, Some(SOFT_CAP)),
                            (Some(WINDOW_LEFT), Some(SOFT_CAP)),
                        ] {
                            fixture.assert_close(
                                &fixture.decode(window, soft_cap)?,
                                &fixture.reference(CONTEXTS, 1.0, window, soft_cap),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn flashinfer_gqa_replays_changed_queries_and_context_tails() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    for group_size in [5, 7] {
        for head_dim in HEAD_DIMS {
            let fixture = Fixture::new(
                &device,
                group_size,
                head_dim,
                DType::BF16,
                DType::BF16,
                true,
            )?;
            let source = fixture.query.copy()?;
            let alternate = source.neg()?;
            let original_lengths = Fixture::last_page_lengths(&device, CONTEXTS)?;
            let alternate_lengths = Fixture::last_page_lengths(&device, REPLAY_CONTEXTS)?;
            fixture.decode(Some(WINDOW_LEFT), Some(SOFT_CAP))?;
            device.synchronize()?;
            let tracking = stream.context().is_event_tracking();
            if tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            if let Err(error) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                return Err(candle_core::Error::msg(error.to_string()));
            }
            let captured = fixture.decode(Some(WINDOW_LEFT), Some(SOFT_CAP));
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = captured?;
            let graph = graph
                .map_err(|error| candle_core::Error::msg(error.to_string()))?
                .ok_or_else(|| candle_core::Error::msg("FlashInfer capture produced no graph"))?;
            for changed in [false, true, false] {
                fixture
                    .query
                    .slice_set(if changed { &alternate } else { &source }, 0, 0)?;
                copy_i32_metadata(
                    if changed {
                        &alternate_lengths
                    } else {
                        &original_lengths
                    },
                    &fixture.last_page_len,
                )?;
                graph
                    .launch()
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                device.synchronize()?;
                fixture.assert_close(
                    &output,
                    &fixture.reference(
                        if changed { REPLAY_CONTEXTS } else { CONTEXTS },
                        if changed { -1.0 } else { 1.0 },
                        Some(WINDOW_LEFT),
                        Some(SOFT_CAP),
                    ),
                )?;
            }
        }
    }
    Ok(())
}
