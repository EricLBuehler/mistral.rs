use std::sync::{Arc, Mutex};

use candle_core::{quantized::k_quants::QK8_0, Context, DType, Result, Shape, Tensor, D};
use mistralrs_quant::{dequantize_8bit, quantize_inplace_8bit, KvQuantizeOp8Bit};

use crate::sequence::Sequence;

use super::{CacheManager, CacheManagerMixin, MetadataMixin};

#[derive(Debug, Clone)]
pub struct SingleQuantizedCache {
    // qs is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a QuantizedKvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub qs: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
    pub latest_shape: Option<Shape>,
}

impl SingleQuantizedCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            qs: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
            latest_shape: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn qs(&self) -> &Option<Tensor> {
        &self.qs
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.qs.as_ref() {
            None => None,
            Some(qs) => {
                let mut latest_shape = self
                    .latest_shape
                    .as_ref()
                    .context("Expected latest_shape!")?
                    .dims()
                    .to_vec();
                latest_shape[self.dim] = self.current_seq_len;

                let n_blocks = latest_shape.iter().product::<usize>() / QK8_0;
                // dbg!(&n_blocks);
                let qs = KvQuantizeOp8Bit::narrow_qs(&qs, 0, n_blocks)?;
                Some(dequantize_8bit(&qs, DType::BF16)?.reshape(latest_shape)?)
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.qs = None;
    }

    pub fn set_len(&mut self, len: usize) {
        self.current_seq_len = len;
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.qs.get_or_insert_with.

        if self.qs.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let qs_numel = KvQuantizeOp8Bit::compute_qs_elem_count_(shape.iter().product())?;
            let qs = Tensor::zeros(qs_numel, DType::U8, src.device())?;
            self.qs = Some(qs);
        };

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            todo!()
        }
        let qs = self.qs.as_mut().unwrap();

        let mut shape_offset = src.dims().to_vec();
        shape_offset[self.dim] = self.current_seq_len;

        let start_block = shape_offset.iter().product::<usize>() / QK8_0;
        let n_blocks = src.elem_count() / QK8_0;

        // dbg!(&start_block);
        // dbg!(&n_blocks);
        let mut qs =
            KvQuantizeOp8Bit::narrow_qs(&qs, start_block, n_blocks)?;
        quantize_inplace_8bit(&mut qs, src)?;

        // dbg!(&src.shape());
        self.latest_shape = Some(src.shape().clone());

        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    pub k: SingleQuantizedCache,
    pub v: SingleQuantizedCache,
}

impl QuantizedKvCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        let k = SingleQuantizedCache::new(dim, max_seq_len, capacity_seq_len);
        let v = SingleQuantizedCache::new(dim, max_seq_len, capacity_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &SingleQuantizedCache {
        &self.k
    }

    pub fn v_cache(&self) -> &SingleQuantizedCache {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut SingleQuantizedCache {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut SingleQuantizedCache {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    pub fn append_sliding_window(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let (mut k, mut v) = self.append(k, v)?;

        if let Some(sliding_window) = sliding_window {
            assert_eq!(self.k.dim, 2);
            let kv_seq_len = k.dim(2)?;
            if kv_seq_len > sliding_window {
                k = k.narrow(2, kv_seq_len - (sliding_window - 1), sliding_window - 1)?;
                v = v.narrow(2, kv_seq_len - (sliding_window - 1), sliding_window - 1)?;
                if let Some(mut mask) = mask.cloned() {
                    let mask_len = mask.dim(1)?;
                    mask = mask.narrow(1, mask_len - (sliding_window - 1), sliding_window - 1)?;
                    mask = Tensor::cat(
                        &[&mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                        D::Minus1,
                    )?;
                    return Ok((k, v, Some(mask)));
                }
            }
        }
        Ok((k, v, mask.cloned()))
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        self.k.append(&k)?;
        self.v.append(&v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    pub fn set_len(&mut self, len: usize) {
        self.k.set_len(len);
        self.v.set_len(len);
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedCache(pub Vec<QuantizedKvCache>);

impl QuantizedCache {
    /// The number of tokens to grow the cache by
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self(vec![
            QuantizedKvCache::new(
                2,
                max_seq_len,
                Self::CACHE_GROW_SIZE
            );
            len
        ])))
    }
}

pub struct QuantizedCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for QuantizedCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        _modify_draft_cache: bool,
    ) {
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();
        // Use this for the various parameters. Assumes all seqs are from one model.
        let template_cache_dim = pipeline.cache().quantized().0[0].k.dim;
        let template_cache_csl = pipeline.cache().quantized().0[0].k.current_seq_len;
        let template_cache_msl = pipeline.cache().quantized().0[0].k.max_seq_len;
        let template_cache_capsl = pipeline.cache().quantized().0[0].k.capacity_seq_len;

        let template_cache_k_shape = pipeline.cache().quantized().0[0].k.latest_shape.clone();
        let template_cache_v_shape = pipeline.cache().quantized().0[0].v.latest_shape.clone();

        'outer: for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();
            for seq in &mut *seqs {
                let src_cache = seq.quantized_cache();
                let cache = src_cache.get(layer).unwrap();
                // This case for llama 3.2 vision cross attn
                if cache.is_none() {
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue 'outer;
                }
                let cache = cache
                    .as_ref()
                    .expect("Not handling completions in `clone_in_cache`.");
                k_vec.push(cache.k.qs.clone().unwrap());
                v_vec.push(cache.v.qs.clone().unwrap());
            }
            new_k_cache.push(Some(if k_vec.len() > 1 {
                Tensor::cat(&k_vec, 0).unwrap()
            } else {
                k_vec[0].clone()
            }));
            new_v_cache.push(Some(if v_vec.len() > 1 {
                Tensor::cat(&v_vec, 0).unwrap()
            } else {
                v_vec[0].clone()
            }));
        }
        let mut caches = Vec::new();
        for (k_cache, v_cache) in new_k_cache.into_iter().zip(new_v_cache) {
            caches.push(QuantizedKvCache {
                k: SingleQuantizedCache {
                    qs: k_cache.map(|x| x.contiguous().unwrap()),
                    dim: template_cache_dim,
                    current_seq_len: template_cache_csl,
                    max_seq_len: template_cache_msl,
                    capacity_seq_len: template_cache_capsl,
                    latest_shape: template_cache_k_shape.clone(),
                },
                v: SingleQuantizedCache {
                    qs: v_cache.map(|x| x.contiguous().unwrap()),
                    dim: template_cache_dim,
                    current_seq_len: template_cache_csl,
                    max_seq_len: template_cache_msl,
                    capacity_seq_len: template_cache_capsl,
                    latest_shape: template_cache_v_shape.clone(),
                },
            });
        }
        *pipeline.cache().quantized() = QuantizedCache(caches);
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], _modify_draft_cache: bool) {
        let all_cache = pipeline.cache().quantized();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.k().unwrap().is_none() {
                continue;
            }

            let k_cache = cache.k.qs.clone().unwrap();
            let v_cache = cache.v.qs.clone().unwrap();

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let output_cache = seq.quantized_cache();
                let seq_cache = &mut output_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();
                *seq_cache = Some(QuantizedKvCache {
                    k: SingleQuantizedCache {
                        qs: Some(k),
                        dim: cache.k.dim,
                        current_seq_len: cache.k.current_seq_len,
                        max_seq_len: cache.k.max_seq_len,
                        capacity_seq_len: cache.k.capacity_seq_len,
                        latest_shape: cache.k.latest_shape.clone(),
                    },
                    v: SingleQuantizedCache {
                        qs: Some(v),
                        dim: cache.v.dim,
                        current_seq_len: cache.v.current_seq_len,
                        max_seq_len: cache.v.max_seq_len,
                        capacity_seq_len: cache.v.capacity_seq_len,
                        latest_shape: cache.v.latest_shape.clone(),
                    },
                });
            }
        }
    }
    fn set_none_cache(
        &self,
        pipeline: &T,
        _seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        // Use this for the various parameters. Assumes all seqs are from one model.
        let template_cache_dim = pipeline.cache().quantized().0[0].k.dim;
        let template_cache_msl = pipeline.cache().quantized().0[0].k.max_seq_len;
        let template_cache_capsl = pipeline.cache().quantized().0[0].k.capacity_seq_len;

        let template_cache_k_shape = pipeline.cache().quantized().0[0].k.latest_shape.clone();
        let template_cache_v_shape = pipeline.cache().quantized().0[0].v.latest_shape.clone();

        for layer in pipeline.cache().quantized().0.iter_mut() {
            let cache = QuantizedKvCache {
                k: SingleQuantizedCache {
                    qs: None,
                    dim: template_cache_dim,
                    current_seq_len: 0,
                    max_seq_len: template_cache_msl,
                    capacity_seq_len: template_cache_capsl,
                    latest_shape: template_cache_k_shape.clone(),
                },
                v: SingleQuantizedCache {
                    qs: None,
                    dim: template_cache_dim,
                    current_seq_len: 0,
                    max_seq_len: template_cache_msl,
                    capacity_seq_len: template_cache_capsl,
                    latest_shape: template_cache_v_shape.clone(),
                },
            };
            *layer = cache;
        }
    }
}
