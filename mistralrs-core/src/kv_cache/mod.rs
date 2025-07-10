use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{Result, Tensor, D};

use crate::{
    get_mut_arcmutex,
    pipeline::{CacheManagerMixin, MetadataMixin},
    sequence::Sequence,
};

mod full_cache;
mod rotating_cache;
mod single_cache;

pub use full_cache::{EitherCache, LayerCaches};
pub use rotating_cache::RotatingCache;
pub use single_cache::SingleCache;

pub trait CacheManager<T: CacheManagerMixin + MetadataMixin + ?Sized> {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    );
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    );
}

#[derive(Debug, Clone)]
pub enum KvCache {
    Normal { k: SingleCache, v: SingleCache },
    Rotating { k: RotatingCache, v: RotatingCache },
}

impl KvCache {
    pub fn new_normal(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        let k = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        let v = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        Self::Normal { k, v }
    }

    pub fn new_rotating(dim: usize, sliding_window: usize, capacity_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        let v = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        Self::Rotating { k, v }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (out_k, out_v) = match self {
            Self::Normal { k: kc, v: vc } => {
                kc.append(&k)?;
                vc.append(&v)?;
                (kc.current_data()?, vc.current_data()?)
            }
            Self::Rotating { k: kc, v: vc } => {
                let out_k = kc.append(&k)?;
                let out_v = vc.append(&v)?;
                (Some(out_k), Some(out_v))
            }
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                }
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                match self {
                    Self::Normal { v, .. } => shape[v.dim] = 0,
                    Self::Rotating { v, .. } => shape[v.dim] = 0,
                }
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Normal { k, .. } => k.current_seq_len(),
            Self::Rotating { k, .. } => k.current_seq_len(),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Normal { k, v } => {
                k.reset();
                v.reset();
            }
            Self::Rotating { k, v } => {
                k.reset();
                v.reset();
            }
        }
    }

    /// Returns Ok if the length reassignment was successful, otherwise returns Err.
    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
        }
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating { .. })
    }
}

#[derive(Debug, Clone)]
pub struct NormalCache(pub Vec<KvCache>);

#[derive(Debug)]
pub enum NormalCacheType {
    Normal { max_seq_len: usize },
    SlidingWindow { window: usize },
}

impl NormalCache {
    /// The number of tokens to grow the cache by
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self(vec![
            KvCache::new_normal(
                2,
                max_seq_len,
                Self::CACHE_GROW_SIZE
            );
            len
        ])))
    }

    pub fn new_sliding(
        len: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
    ) -> Arc<Mutex<Self>> {
        match sliding_window {
            Some(sliding_window) => Arc::new(Mutex::new(Self(vec![
                KvCache::new_rotating(
                    2,
                    sliding_window,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
            None => Arc::new(Mutex::new(Self(vec![
                KvCache::new_normal(
                    2,
                    max_seq_len,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
        }
    }

    pub fn from_types(types: Vec<NormalCacheType>) -> Arc<Mutex<Self>> {
        let mut caches = Vec::new();
        for ty in types {
            match ty {
                NormalCacheType::Normal { max_seq_len } => {
                    caches.push(KvCache::new_normal(2, max_seq_len, Self::CACHE_GROW_SIZE));
                }
                NormalCacheType::SlidingWindow { window } => {
                    caches.push(KvCache::new_rotating(2, window, Self::CACHE_GROW_SIZE));
                }
            }
        }
        Arc::new(Mutex::new(Self(caches)))
    }
}

pub struct NormalCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for NormalCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();

        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            // Preallocate combined k and v caches across all sequences, avoiding Tensor::cat copies
            let batch_len = seqs.len();
            // Use the first sequence as template
            let (first_k, first_v) = {
                let src_cache = if modify_draft_cache {
                    seqs[0].normal_draft_cache()
                } else {
                    seqs[0].normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    // This is hit in gemma3n for the shared kv cache
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue;
                };
                match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                }
            };
            // Build dims for batched cache
            let mut dims_k = first_k.dims().to_vec();
            let mut dims_v = first_v.dims().to_vec();
            dims_k[0] *= batch_len;
            dims_v[0] *= batch_len;
            let batch_k = Tensor::zeros(dims_k.clone(), first_k.dtype(), first_k.device()).unwrap();
            let batch_v = Tensor::zeros(dims_v.clone(), first_v.dtype(), first_v.device()).unwrap();
            // Fill each sequence's cache slice
            for (i, seq) in seqs.iter_mut().enumerate() {
                let src_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    // Skip for shared kv cache layers in models like gemma3n
                    continue;
                };
                let (src_k, src_v) = match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                };
                let offset = i * first_k.dims()[0];
                batch_k.slice_set(&src_k, 0, offset).unwrap();
                batch_v.slice_set(&src_v, 0, offset).unwrap();
            }
            new_k_cache.push(Some(batch_k));
            new_v_cache.push(Some(batch_v));
        }

        let seq0_cache = if modify_draft_cache {
            &*seqs[0].normal_draft_cache()
        } else {
            &*seqs[0].normal_cache()
        };

        let mut caches = Vec::new();
        for (layer_idx, (k_cache, v_cache)) in new_k_cache.into_iter().zip(new_v_cache).enumerate()
        {
            // Use this for the various parameters. Assumes all seqs are from one model.
            let Some(cache_ref) = seq0_cache[layer_idx].as_ref() else {
                // This is hit in gemma3n for the shared kv cache - create dummy cache
                // These layers don't have their own cache because they share another layer's cache
                caches.push(KvCache::Normal {
                    k: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                    v: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                });
                continue;
            };
            match cache_ref {
                KvCache::Normal { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Normal {
                        k: SingleCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                        v: SingleCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                    });
                }
                KvCache::Rotating { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_offset = old_k.offset;
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: template_cache_capsl,
                        },
                        v: RotatingCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: template_cache_capsl,
                        },
                    });
                }
            }
        }
        *pipeline.cache().normal() = NormalCache(caches);
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let all_cache = pipeline.cache().normal();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.k().unwrap().is_none() {
                continue;
            }

            let (k_cache, v_cache) = match cache {
                KvCache::Normal { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::Rotating { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
            };

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let output_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let seq_cache = &mut output_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();

                match cache {
                    KvCache::Normal {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Normal {
                            k: SingleCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: SingleCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::Rotating {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Rotating {
                            k: RotatingCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                offset: cache_k.offset,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: RotatingCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                offset: cache_v.offset,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                }
            }
        }
    }
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if seqs.iter().any(|seq| seq.preallocated_cache().is_none()) {
            for layer in pipeline.cache().normal().0.iter_mut() {
                layer.reset();
            }
            return;
        }

        let layer_devices = if let Some(device_mapper) = pipeline.device_mapper() {
            let mut layer_devices = Vec::new();
            for layer in 0..device_mapper.num_device_mapping_layers() {
                let device = device_mapper.device_for(layer, false).cloned();
                layer_devices.push(device.expect("Internal bug, layer out of range!"));
            }
            Some(layer_devices)
        } else {
            None
        };

        let old_caches = pipeline.cache().normal().0.clone();

        for (layer_idx, layer) in pipeline.cache().normal().0.iter_mut().enumerate() {
            if !load_preallocated_cache {
                layer.reset();
                continue;
            }

            let mut k_caches = Vec::new();
            let mut v_caches = Vec::new();
            for seq in seqs.iter_mut() {
                let (mut k_preallocated_cache, mut v_preallocated_cache) =
                    (*seq.preallocated_cache().as_ref().unwrap()).clone();
                if let Some(layer_devices) = &layer_devices {
                    let layer_dev = &layer_devices[layer_idx];
                    k_preallocated_cache = k_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                    v_preallocated_cache = v_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                }
                k_caches.push(k_preallocated_cache);
                v_caches.push(v_preallocated_cache);
            }
            let k_cache = if k_caches.len() > 1 {
                Tensor::cat(&k_caches, 0).unwrap()
            } else {
                k_caches[0].clone()
            };
            let v_cache = if v_caches.len() > 1 {
                Tensor::cat(&v_caches, 0).unwrap()
            } else {
                v_caches[0].clone()
            };

            // Use this for the various parameters. Assumes all seqs are from one model.
            match &old_caches[layer_idx] {
                KvCache::Normal { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    let cache = KvCache::Normal {
                        k: SingleCache {
                            all_data: Some(k_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                        v: SingleCache {
                            all_data: Some(v_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                    };
                    *layer = cache;
                }
                KvCache::Rotating { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    // Rotating cache is not preallocated.
                    let cache = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                        v: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                    };
                    *layer = cache;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<LayerCaches>>,
    xlora_cache: Option<Arc<Mutex<LayerCaches>>>,
    draft_cache: Arc<Mutex<LayerCaches>>,
    scalings_cache: Option<Arc<Mutex<Option<Tensor>>>>,
}

impl Cache {
    pub(crate) fn new(len: usize, is_xlora: bool) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
            xlora_cache: if is_xlora {
                Some(Arc::new(Mutex::new(vec![None; len])))
            } else {
                None
            },
            draft_cache: Arc::new(Mutex::new(vec![None; len])),
            scalings_cache: if is_xlora {
                Some(Arc::new(Mutex::new(None)))
            } else {
                None
            },
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.cache)
    }

    pub(crate) fn draft_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.draft_cache)
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn xlora_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.xlora_cache.as_ref().expect("No X-LoRA cache."))
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn get_scalings_cache(&self) -> MutexGuard<'_, Option<Tensor>> {
        get_mut_arcmutex!(self
            .scalings_cache
            .as_ref()
            .expect("No X-LoRA scalings cache."))
    }

    pub(crate) fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    /// Update the KV cache and return (k,v)
    pub(crate) fn update_kv_cache(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
        slow_cat: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match &*cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if !slow_cat {
                    let k = candle_nn::ops::kvconcat(k_cache, &k, 2)?.contiguous()?;
                    let v = candle_nn::ops::kvconcat(v_cache, &v, 2)?.contiguous()?;
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                    (k, v)
                }
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?))
    }

    /// Update the KV cache and return (k,v,attn_mask)
    pub(crate) fn update_kv_cache_sliding_window(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
        slow_cat: bool,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let (k, v, attention_mask) = match cache.clone() {
            None => (k, v, attention_mask.cloned()),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = attention_mask.cloned();
                if let Some(sliding_window) = sliding_window {
                    let kv_seq_len = prev_k.dim(2)?;
                    if kv_seq_len > sliding_window {
                        prev_k = prev_k.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        prev_v = prev_v.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        if let Some(ref mut mask) = mask {
                            let mask_len = mask.dim(1)?;
                            *mask = mask.narrow(
                                1,
                                mask_len - (sliding_window - 1),
                                sliding_window - 1,
                            )?;
                            *mask = Tensor::cat(
                                &[&*mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                                D::Minus1,
                            )?;
                        }
                    }
                }
                let (k, v) = if !slow_cat {
                    let k = candle_nn::ops::kvconcat(&prev_k, &k, 2)?;
                    let v = candle_nn::ops::kvconcat(&prev_v, &v, 2)?;
                    (k, v)
                } else {
                    let k = Tensor::cat(&[prev_k, k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[prev_v, v], 2)?.contiguous()?;
                    (k, v)
                };
                (k, v, mask)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?, attention_mask))
    }
}

pub struct FullCacheManager;

enum SeqCache {
    Normal,
    XLora,
    Draft,
}

fn clone_in_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    src: SeqCache,
) {
    let mut new_cache = Vec::new();
    'outer: for layer in 0..num_hidden_layers {
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();
        for seq in &mut *seqs {
            let src_cache = match src {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let cache = src_cache.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.is_none() {
                new_cache.push(None);
                continue 'outer;
            }
            let cache = cache
                .as_ref()
                .expect("Not handling completions in `clone_in_cache`.");
            k_vec.push(cache.0.clone());
            v_vec.push(cache.1.clone());
        }
        new_cache.push(Some((
            if k_vec.len() > 1 {
                Tensor::cat(&k_vec, 0).unwrap()
            } else {
                k_vec[0].clone()
            },
            if v_vec.len() > 1 {
                Tensor::cat(&v_vec, 0).unwrap()
            } else {
                v_vec[0].clone()
            },
        )));
    }
    *cache = new_cache;
}

fn clone_out_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    target: SeqCache,
) {
    for layer in 0..num_hidden_layers {
        let cache = cache.get(layer).unwrap();
        // This case for llama 3.2 vision cross attn
        if cache.is_none() {
            continue;
        }

        let k_cache = cache.as_ref().unwrap().0.clone();
        let v_cache = cache.as_ref().unwrap().1.clone();

        let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(k_caches.len(), seqs.len());
        let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(v_caches.len(), seqs.len());

        for (seq_i, seq) in seqs.iter_mut().enumerate() {
            let output_cache = match target {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let seq_cache = &mut output_cache[layer];
            let k = k_caches.get(seq_i).unwrap().clone();
            let v = v_caches.get(seq_i).unwrap().clone();
            *seq_cache = Some((k, v));
        }
    }
}

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for FullCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_in_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            pipeline
                .cache()
                .full()
                .get_scalings_cache()
                .clone_from(seqs[0].scaling_cache());
        }
    }

    fn clone_out_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_out_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            seqs[0]
                .scaling_cache()
                .clone_from(&pipeline.cache().full().get_scalings_cache());
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        _seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.get_metadata().num_hidden_layers {
            new_cache.push(None);
        }
        pipeline.cache().full().lock().clone_from(&new_cache);
        if modify_draft_cache {
            pipeline.cache().full().draft_lock().clone_from(&new_cache);
        }
        if pipeline.cache().full().is_xlora() {
            *pipeline.cache().full().xlora_lock() = new_cache;
        }
    }
}
