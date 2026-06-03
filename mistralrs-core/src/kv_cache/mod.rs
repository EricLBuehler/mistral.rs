use crate::attention::AttentionMask;
use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{Result, Tensor, D};

use crate::{
    get_mut_arcmutex,
    pipeline::{CacheManagerMixin, MetadataMixin},
    sequence::Sequence,
};

mod full_cache;
mod hybrid_cache;
mod rotating_cache;
mod single_cache;

pub use full_cache::{EitherCache, LayerCaches};
pub use hybrid_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    RecurrentStateSnapshot,
};
pub use rotating_cache::{RotatingCache, RotatingCacheSnapshot};
pub use single_cache::{SingleCache, SingleCacheSnapshot};

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
    Shared { owner: usize },
}

#[derive(Debug, Clone)]
pub enum KvCacheSnapshot {
    Normal {
        k: SingleCacheSnapshot,
        v: SingleCacheSnapshot,
    },
    Rotating {
        k: RotatingCacheSnapshot,
        v: RotatingCacheSnapshot,
    },
    Shared {
        owner: usize,
    },
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

    pub fn new_shared(owner: usize) -> Self {
        Self::Shared { owner }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
            Self::Shared { .. } => Ok(None),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
            Self::Shared { .. } => Ok(None),
        }
    }

    /// Return the K tensor from the last `append()` call.
    ///
    /// For Normal caches this is identical to `k()`. For Rotating caches it
    /// returns the full (retained + new) tensor that `append()` produced,
    /// which during prefill may be larger than the internal sliding-window
    /// buffer returned by `k()`.  Shared KV layers must use this instead of
    /// `k()` so they see the same K/V the donor used for its own attention.
    pub fn appended_k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => Ok(k.last_append_result().cloned()),
            Self::Shared { .. } => Ok(None),
        }
    }

    /// Same as [`appended_k`](Self::appended_k) but for the V tensor.
    pub fn appended_v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => Ok(v.last_append_result().cloned()),
            Self::Shared { .. } => Ok(None),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Metal fast-path: fuse the K and V slice_set calls into one kernel.
        // Skip if inputs aren't already contiguous; the slow path will fix that.
        #[cfg(feature = "metal")]
        if k.device().is_metal() && k.is_contiguous() && v.is_contiguous() {
            #[allow(clippy::collapsible_match)]
            match self {
                Self::Normal { k: kc, v: vc } => {
                    if try_kv_append_dual_metal(kc, vc, k, v)? {
                        let out_k = kc.current_data()?;
                        let out_v = vc.current_data()?;
                        return Ok((out_k.unwrap(), out_v.unwrap()));
                    }
                }
                Self::Rotating { k: kc, v: vc } => {
                    if let Some((rk, rv)) = try_kv_append_rotating_metal(kc, vc, k, v)? {
                        return Ok((rk, rv));
                    }
                }
                _ => {}
            }
        }
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
            Self::Shared { owner } => {
                candle_core::bail!(
                    "attempted to append KV data to shared cache owned by layer {owner}"
                );
            }
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                    Self::Shared { .. } => unreachable!(),
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
                    Self::Shared { .. } => unreachable!(),
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
            Self::Shared { .. } => 0,
        }
    }

    pub fn snapshot(&self) -> Result<KvCacheSnapshot> {
        match self {
            Self::Normal { k, v } => Ok(KvCacheSnapshot::Normal {
                k: k.snapshot(),
                v: v.snapshot(),
            }),
            Self::Rotating { k, v } => Ok(KvCacheSnapshot::Rotating {
                k: k.snapshot()?,
                v: v.snapshot()?,
            }),
            Self::Shared { owner } => Ok(KvCacheSnapshot::Shared { owner: *owner }),
        }
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &KvCacheSnapshot,
        base_len: usize,
        append_len: usize,
        max_context_len: usize,
    ) -> bool {
        if base_len + append_len > max_context_len {
            return false;
        }
        match (self, snapshot) {
            (
                Self::Normal { k, v },
                KvCacheSnapshot::Normal {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                k.current_seq_len() == base_len
                    && v.current_seq_len() == base_len
                    && k_snapshot.current_seq_len == base_len
                    && v_snapshot.current_seq_len == base_len
                    && k.can_append_from_snapshot(k_snapshot, append_len)
                    && v.can_append_from_snapshot(v_snapshot, append_len)
            }
            (
                Self::Rotating { k, v },
                KvCacheSnapshot::Rotating {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                k.current_seq_len() == base_len
                    && v.current_seq_len() == base_len
                    && k_snapshot.current_seq_len == base_len
                    && v_snapshot.current_seq_len == base_len
                    && k.can_append_from_snapshot(k_snapshot, append_len)
                    && v.can_append_from_snapshot(v_snapshot, append_len)
            }
            (
                Self::Shared { owner },
                KvCacheSnapshot::Shared {
                    owner: snapshot_owner,
                },
            ) => owner == snapshot_owner,
            _ => false,
        }
    }

    pub fn restore_after_speculative_append(
        &mut self,
        snapshot: &KvCacheSnapshot,
        post_forward_layer: Option<&KvCache>,
        keep_len: usize,
        row_idx: usize,
        batch_len: usize,
    ) -> Result<()> {
        match (self, snapshot) {
            (Self::Normal { k, v }, KvCacheSnapshot::Normal { .. }) => {
                k.rollback_to(keep_len)?;
                v.rollback_to(keep_len)?;
            }
            (
                Self::Rotating { k, v },
                KvCacheSnapshot::Rotating {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                let Some(KvCache::Rotating {
                    k: post_k,
                    v: post_v,
                }) = post_forward_layer
                else {
                    candle_core::bail!(
                        "rotating cache speculative rollback requires post-forward rotating layer"
                    );
                };
                let accepted_k = post_k.accepted_append_from_batched_append(
                    k_snapshot, keep_len, row_idx, batch_len,
                )?;
                let accepted_v = post_v.accepted_append_from_batched_append(
                    v_snapshot, keep_len, row_idx, batch_len,
                )?;
                *k = RotatingCache::restore_from_snapshot(k_snapshot, accepted_k, keep_len)?;
                *v = RotatingCache::restore_from_snapshot(v_snapshot, accepted_v, keep_len)?;
            }
            (
                Self::Shared { owner },
                KvCacheSnapshot::Shared {
                    owner: snapshot_owner,
                },
            ) => {
                *owner = *snapshot_owner;
            }
            (layer, KvCacheSnapshot::Shared { owner }) => {
                *layer = KvCache::Shared { owner: *owner };
            }
            _ => {
                candle_core::bail!("kv-cache speculative rollback snapshot kind mismatch");
            }
        }
        Ok(())
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
            Self::Shared { .. } => {}
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
            Self::Shared { .. } => Ok(()),
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
            Self::Shared { .. } => Ok(()),
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating { .. })
    }

    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared { .. })
    }
}

#[derive(Debug, Clone)]
pub struct NormalCache(pub Vec<KvCache>);

#[derive(Debug)]
pub enum NormalCacheType {
    Normal { max_seq_len: usize },
    SlidingWindow { window: usize },
    Shared { owner: usize },
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
                NormalCacheType::Shared { owner } => {
                    caches.push(KvCache::new_shared(owner));
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
                    KvCache::Shared { .. } => {
                        new_k_cache.push(None);
                        new_v_cache.push(None);
                        continue;
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
                    continue;
                };
                let (src_k, src_v) = match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Shared { .. } => continue,
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
                let mut cache = pipeline.cache().normal().0[layer_idx].clone();
                cache.reset();
                caches.push(cache);
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
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                            last_append_result: None,
                        },
                        v: RotatingCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                            last_append_result: None,
                        },
                    });
                }
                KvCache::Shared { owner } => {
                    caches.push(KvCache::Shared { owner: *owner });
                }
            }
        }
        *pipeline.cache().normal() = NormalCache(caches);
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let all_cache = pipeline.cache().normal();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();
            if let KvCache::Shared { owner } = cache {
                for seq in seqs.iter_mut() {
                    let output_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    output_cache[layer] = Some(KvCache::Shared { owner: *owner });
                }
                continue;
            }
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
                KvCache::Shared { .. } => unreachable!(),
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
                                capacity_seq_len: cache_k.capacity_seq_len,
                                last_append_result: None,
                            },
                            v: RotatingCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                                last_append_result: None,
                            },
                        });
                    }
                    KvCache::Shared { .. } => unreachable!(),
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

        let layer_devices = pipeline.device_mapper().map(|device_mapper| {
            let total_layers = pipeline.cache().normal().0.len();
            let mut layer_devices = Vec::with_capacity(total_layers);
            for layer in 0..total_layers {
                let device = device_mapper
                    .device_for(layer, false)
                    .cloned()
                    .expect("Internal bug, layer out of range!");
                layer_devices.push(device);
            }
            layer_devices
        });

        let old_caches = pipeline.cache().normal().0.clone();

        for (layer_idx, layer) in pipeline.cache().normal().0.iter_mut().enumerate() {
            if !load_preallocated_cache {
                layer.reset();
                continue;
            }

            match &old_caches[layer_idx] {
                KvCache::Rotating { k, .. } => {
                    *layer = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: None,
                            dim: k.dim,
                            current_seq_len: 0,
                            max_seq_len: k.max_seq_len,
                            capacity_seq_len: k.capacity_seq_len,
                            last_append_result: None,
                        },
                        v: RotatingCache {
                            all_data: None,
                            dim: k.dim,
                            current_seq_len: 0,
                            max_seq_len: k.max_seq_len,
                            capacity_seq_len: k.capacity_seq_len,
                            last_append_result: None,
                        },
                    };
                    continue;
                }
                KvCache::Shared { owner } => {
                    *layer = KvCache::Shared { owner: *owner };
                    continue;
                }
                KvCache::Normal { .. } => {}
            }

            let mut k_caches = Vec::new();
            let mut v_caches = Vec::new();
            let mut missing_preallocated = false;
            for seq in seqs.iter_mut() {
                let Some((mut k_preallocated_cache, mut v_preallocated_cache)) = seq
                    .preallocated_cache()
                    .and_then(|cache| cache.get(layer_idx))
                    .cloned()
                    .flatten()
                else {
                    missing_preallocated = true;
                    break;
                };
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
            if missing_preallocated {
                layer.reset();
                continue;
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
                KvCache::Rotating { .. } | KvCache::Shared { .. } => unreachable!(),
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
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match &*cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k, v)
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
        attention_mask: &AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let mask_tensor = match attention_mask {
            AttentionMask::Custom(t) => Some(t.clone()),
            _ => None,
        };
        let (k, v, attention_mask) = match cache.clone() {
            None => (k, v, mask_tensor),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = mask_tensor;
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
                let (k, v) = {
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

/// Cache manager for hybrid models (attention + recurrent layers).
///
/// This implements vLLM-style continuous batching:
/// - Attention layers: Standard KV cache batching (cat on clone_in, chunk on clone_out)
/// - Recurrent layers: Pool-based state management with indexed access
///
/// Each sequence has a `recurrent_state_idx` pointing to its slot in the
/// state pool. The forward pass builds a `state_indices` tensor from these
/// indices and uses gather/scatter operations.
pub struct HybridCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for HybridCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        let mut hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();

        // Build state_indices for recurrent layers from sequences' recurrent_state_idx
        // Find the device from the first recurrent layer's pool
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });

        // Ensure every sequence has a recurrent slot when using hybrid cache.
        let mut state_index_allocation_failed = false;
        let mut newly_allocated = Vec::new();
        for (seq_idx, seq) in seqs.iter_mut().enumerate() {
            if seq.recurrent_state_idx().is_none() {
                if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                    seq.set_recurrent_state_idx(Some(slot_idx));
                    newly_allocated.push((seq_idx, slot_idx));
                } else {
                    tracing::warn!(
                        "Failed to allocate recurrent state slot for sequence {}, hybrid forward will fail for this batch.",
                        seq.id()
                    );
                    state_index_allocation_failed = true;
                    break;
                }
            }
        }
        if state_index_allocation_failed {
            for (seq_idx, slot_idx) in newly_allocated {
                seqs[seq_idx].set_recurrent_state_idx(None);
                hybrid_cache.free_seq(slot_idx);
            }
        }

        if let Some(device) = recurrent_device {
            if state_index_allocation_failed {
                hybrid_cache.set_state_indices(None);
            } else {
                // Build state_indices tensor from sequences
                let mut indices = Vec::with_capacity(seqs.len());
                for seq in seqs.iter() {
                    if let Some(idx) = seq.recurrent_state_idx() {
                        #[allow(clippy::cast_possible_truncation)]
                        indices.push(idx as u32);
                    } else {
                        tracing::warn!(
                            "Sequence {} missing recurrent_state_idx during hybrid clone_in_cache.",
                            seq.id()
                        );
                        hybrid_cache.set_state_indices(None);
                        return;
                    }
                }
                if let Ok(state_indices) = Tensor::from_vec(indices.clone(), (seqs.len(),), &device)
                {
                    hybrid_cache.set_state_indices_with_host(Some(state_indices), Some(indices));
                } else {
                    hybrid_cache.set_state_indices(None);
                }
            }
        }

        // For attention layers, we still need to batch KV caches
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get_mut(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                // Batch KV caches from sequences (same as NormalCacheManager)
                let mut k_tensors = Vec::new();
                let mut v_tensors = Vec::new();
                let mut template_cache: Option<KvCache> = None;

                for seq in seqs.iter_mut() {
                    let seq_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    if let Some(Some(ref kv)) = seq_cache.get(layer_idx) {
                        if template_cache.is_none() {
                            template_cache = Some(kv.clone());
                        }
                        if let (Ok(Some(k)), Ok(Some(v))) = (kv.k(), kv.v()) {
                            k_tensors.push(k);
                            v_tensors.push(v);
                        }
                    }
                }

                if !k_tensors.is_empty() {
                    // cat/clone of narrow'd views may be non-contiguous;
                    // all_data must be contiguous for slice_set in SingleCache::append.
                    let batched_k = if k_tensors.len() > 1 {
                        Tensor::cat(&k_tensors, 0).unwrap()
                    } else {
                        k_tensors[0].contiguous().unwrap()
                    };
                    let batched_v = if v_tensors.len() > 1 {
                        Tensor::cat(&v_tensors, 0).unwrap()
                    } else {
                        v_tensors[0].contiguous().unwrap()
                    };

                    if let Some(ref template) = template_cache {
                        match (template, kv_cache) {
                            (KvCache::Normal { k: tk, .. }, KvCache::Normal { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            (KvCache::Rotating { k: tk, .. }, KvCache::Rotating { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            _ => {}
                        }
                    }
                }
            }
            // For recurrent layers: No copying needed!
            // The pool is accessed directly via state_indices during forward.
        }
    }

    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();
        let num_seqs = seqs.len();

        // For attention layers, split batched KV caches back to sequences
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.k(), kv_cache.v()) {
                    let k_chunks = k.chunk(num_seqs, 0).unwrap();
                    let v_chunks = v.chunk(num_seqs, 0).unwrap();

                    for (seq_idx, seq) in seqs.iter_mut().enumerate() {
                        // chunk() returns non-contiguous views; all_data must be contiguous.
                        let seq_k = k_chunks.get(seq_idx).unwrap().contiguous().unwrap();
                        let seq_v = v_chunks.get(seq_idx).unwrap().contiguous().unwrap();

                        let seq_cache = if modify_draft_cache {
                            seq.normal_draft_cache()
                        } else {
                            seq.normal_cache()
                        };

                        // Initialize cache if needed
                        if seq_cache.get(layer_idx).is_none() || seq_cache[layer_idx].is_none() {
                            while seq_cache.len() <= layer_idx {
                                seq_cache.push(None);
                            }
                            seq_cache[layer_idx] = Some(kv_cache.clone());
                        }

                        if let Some(ref mut seq_kv) = seq_cache[layer_idx] {
                            match (kv_cache, seq_kv) {
                                (KvCache::Normal { k: src_k, .. }, KvCache::Normal { k, v }) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                (
                                    KvCache::Rotating { k: src_k, .. },
                                    KvCache::Rotating { k, v },
                                ) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // For recurrent layers: No splitting needed!
            // The pool was updated in-place during forward via scatter operations.
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        // Reset attention KV caches in sequences
        for seq in seqs.iter_mut() {
            let seq_cache = if modify_draft_cache {
                seq.normal_draft_cache()
            } else {
                seq.normal_cache()
            };
            for kv in seq_cache.iter_mut().flatten() {
                kv.reset();
            }
        }
        // Reset the hybrid cache (including recurrent state pools)
        let mut hybrid_cache = pipeline.cache().hybrid();
        hybrid_cache.reset();

        // Build state_indices so the forward pass can access recurrent pool states.
        // Sequences already have slots allocated from add_request.
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });
        if let Some(device) = recurrent_device {
            #[allow(clippy::cast_possible_truncation)]
            let indices: Vec<u32> = seqs
                .iter()
                .filter_map(|seq| seq.recurrent_state_idx().map(|idx| idx as u32))
                .collect();
            if indices.len() == seqs.len() {
                if let Ok(state_indices) = Tensor::from_vec(indices.clone(), (seqs.len(),), &device)
                {
                    hybrid_cache.set_state_indices_with_host(Some(state_indices), Some(indices));
                }
            }
        }
    }
}

#[cfg(feature = "metal")]
fn try_kv_append_dual_metal(
    kc: &mut single_cache::SingleCache,
    vc: &mut single_cache::SingleCache,
    k_src: &Tensor,
    v_src: &Tensor,
) -> Result<bool> {
    use candle_core::{backend::BackendStorage, Storage};

    // Layout requirements: dim=2, rank=4, source [b=1, n_kv, src_seq, head_dim],
    // dst (cache) [b=1, n_kv, max_seq, head_dim], both BF16/F16/F32.
    if kc.dim != 2 || vc.dim != 2 {
        return Ok(false);
    }
    if k_src.rank() != 4 || v_src.rank() != 4 {
        return Ok(false);
    }
    if !matches!(
        k_src.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16 | candle_core::DType::F32
    ) {
        return Ok(false);
    }
    if k_src.dtype() != v_src.dtype() {
        return Ok(false);
    }
    if k_src.shape() != v_src.shape() {
        return Ok(false);
    }
    let (b, n_kv, src_seq, head_dim) = k_src.dims4()?;
    if b != 1 {
        return Ok(false);
    }
    if kc.current_seq_len + src_seq > kc.capacity_seq_len {
        return Ok(false);
    }
    if vc.current_seq_len + src_seq > vc.capacity_seq_len {
        return Ok(false);
    }
    if kc.current_seq_len != vc.current_seq_len {
        return Ok(false);
    }
    if kc.all_data.is_none() || vc.all_data.is_none() {
        // First call: let the slow path allocate the cache buffers.
        return Ok(false);
    }
    let k_dst = kc.all_data.as_ref().unwrap();
    let v_dst = vc.all_data.as_ref().unwrap();
    if k_dst.shape() != v_dst.shape() {
        return Ok(false);
    }
    let max_seq = k_dst.dim(2)?;
    if k_dst.dims4()? != (b, n_kv, max_seq, head_dim) {
        return Ok(false);
    }
    if !k_dst.is_contiguous() || !v_dst.is_contiguous() {
        return Ok(false);
    }

    let (k_src_s, k_src_l) = k_src.storage_and_layout();
    let (v_src_s, v_src_l) = v_src.storage_and_layout();
    let (k_dst_s, _) = k_dst.storage_and_layout();
    let (v_dst_s, _) = v_dst.storage_and_layout();
    let (
        Storage::Metal(k_src_m),
        Storage::Metal(v_src_m),
        Storage::Metal(k_dst_m),
        Storage::Metal(v_dst_m),
    ) = (&*k_src_s, &*v_src_s, &*k_dst_s, &*v_dst_s)
    else {
        return Ok(false);
    };

    let device = k_src_m.device().clone();
    let encoder = device.command_encoder()?;
    encoder.set_label("kv-append-dual");

    mistralrs_quant::metal_kernels::call_kv_append_dual(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        k_src.dtype(),
        k_src_m.buffer(),
        k_src_l.start_offset() * k_src.dtype().size_in_bytes(),
        v_src_m.buffer(),
        v_src_l.start_offset() * v_src.dtype().size_in_bytes(),
        k_dst_m.buffer(),
        v_dst_m.buffer(),
        head_dim,
        n_kv,
        src_seq,
        max_seq,
        kc.current_seq_len,
    )
    .map_err(candle_core::Error::wrap)?;

    kc.current_seq_len += src_seq;
    vc.current_seq_len += src_seq;
    Ok(true)
}

#[cfg(feature = "metal")]
fn try_kv_append_rotating_metal(
    kc: &mut rotating_cache::RotatingCache,
    vc: &mut rotating_cache::RotatingCache,
    k_src: &Tensor,
    v_src: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::{backend::BackendStorage, Storage};

    // Decode steady-state only: window is already full, one new token at a time.
    // Anything else falls back so the existing shift-based code handles it.
    if kc.dim != 2 || vc.dim != 2 {
        return Ok(None);
    }
    if k_src.rank() != 4 || v_src.rank() != 4 {
        return Ok(None);
    }
    if !matches!(
        k_src.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16 | candle_core::DType::F32
    ) || k_src.dtype() != v_src.dtype()
    {
        return Ok(None);
    }
    if k_src.shape() != v_src.shape() {
        return Ok(None);
    }
    let (b, n_kv, src_seq, head_dim) = k_src.dims4()?;
    if b != 1 || src_seq != 1 {
        return Ok(None);
    }
    // Window must be allocated and already full so we can use the buffer as a
    // circular window without breaking shared-KV / prefill paths.
    if kc.all_data.is_none() || vc.all_data.is_none() {
        return Ok(None);
    }
    if kc.current_seq_len < kc.max_seq_len || vc.current_seq_len < vc.max_seq_len {
        return Ok(None);
    }
    if kc.current_seq_len != vc.current_seq_len {
        return Ok(None);
    }
    let k_dst = kc.all_data.as_ref().unwrap().clone();
    let v_dst = vc.all_data.as_ref().unwrap().clone();
    if !k_dst.is_contiguous() || !v_dst.is_contiguous() {
        return Ok(None);
    }
    let max_seq = kc.max_seq_len;
    if k_dst.dims4()? != (b, n_kv, max_seq, head_dim) {
        return Ok(None);
    }

    // Write the new token to slot (current_seq_len) % max_seq, overwriting the
    // oldest entry. The attention math is order-invariant (RoPE is in K), and
    // the returned buffer is just the full window.
    let slot = kc.current_seq_len % max_seq;

    {
        let (k_src_s, k_src_l) = k_src.storage_and_layout();
        let (v_src_s, v_src_l) = v_src.storage_and_layout();
        let (k_dst_s, _) = k_dst.storage_and_layout();
        let (v_dst_s, _) = v_dst.storage_and_layout();
        let (
            Storage::Metal(k_src_m),
            Storage::Metal(v_src_m),
            Storage::Metal(k_dst_m),
            Storage::Metal(v_dst_m),
        ) = (&*k_src_s, &*v_src_s, &*k_dst_s, &*v_dst_s)
        else {
            return Ok(None);
        };

        let device = k_src_m.device().clone();
        let encoder = device.command_encoder()?;
        encoder.set_label("kv-append-rotating");

        mistralrs_quant::metal_kernels::call_kv_append_dual(
            device.device(),
            &encoder,
            &mistralrs_quant::metal_kernels::Kernels::new(),
            k_src.dtype(),
            k_src_m.buffer(),
            k_src_l.start_offset() * k_src.dtype().size_in_bytes(),
            v_src_m.buffer(),
            v_src_l.start_offset() * v_src.dtype().size_in_bytes(),
            k_dst_m.buffer(),
            v_dst_m.buffer(),
            head_dim,
            n_kv,
            src_seq,
            max_seq,
            slot,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    kc.current_seq_len += src_seq;
    vc.current_seq_len += src_seq;
    kc.last_append_result = Some(k_dst.clone());
    vc.last_append_result = Some(v_dst.clone());
    Ok(Some((k_dst, v_dst)))
}
