use candle_core::Tensor;

use crate::Pipeline;

use super::CacheManager;

pub struct DefaultCacheManager;

impl CacheManager for DefaultCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &mut dyn Pipeline,
        seqs: &mut [&mut crate::sequence::Sequence],
    ) {
        let mut new_cache = Vec::new();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();
            for seq in &mut *seqs {
                let seq_cache = &*seq.cache();
                let cache = seq_cache.get(layer).unwrap();
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
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().has_no_kv_cache {
            let mut new_cache = Vec::new();
            for layer in 0..pipeline.get_metadata().num_hidden_layers {
                let mut k_vec = Vec::new();
                let mut v_vec = Vec::new();
                for seq in &mut *seqs {
                    let seq_cache = &*seq.xlora_cache();
                    let cache = seq_cache.get(layer).unwrap();
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
            *pipeline.cache().xlora_lock() = new_cache;
        }
        if pipeline.get_metadata().is_xlora {
            *pipeline.cache().get_scalings_cache() = seqs[0].scaling_cache().clone();
        }
        *pipeline.cache().lock() = new_cache;
    }

    fn clone_out_cache(
        &self,
        pipeline: &mut dyn Pipeline,
        seqs: &mut [&mut crate::sequence::Sequence],
    ) {
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = pipeline.cache().lock();
            let cache = cache.get(layer).unwrap();
            let k_cache = cache.as_ref().unwrap().0.clone();
            let v_cache = cache.as_ref().unwrap().1.clone();

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let seq_cache = seq.cache();
                let seq_cache = &mut seq_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();
                *seq_cache = Some((k, v));
            }
            if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().has_no_kv_cache {
                let cache = pipeline.cache().xlora_lock();
                let cache = cache.get(layer).unwrap();
                let k_cache = cache.as_ref().unwrap().0.clone();
                let v_cache = cache.as_ref().unwrap().1.clone();

                let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(k_caches.len(), seqs.len());
                let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(v_caches.len(), seqs.len());

                for (seq_i, seq) in seqs.iter_mut().enumerate() {
                    let seq_cache = seq.xlora_cache();
                    let seq_cache = &mut seq_cache[layer];
                    let k = k_caches.get(seq_i).unwrap().clone();
                    let v = v_caches.get(seq_i).unwrap().clone();
                    *seq_cache = Some((k, v));
                }
            }
            if pipeline.get_metadata().is_xlora {
                *seqs[0].scaling_cache() = pipeline.cache().get_scalings_cache().clone();
            }
        }
    }

    fn set_none_cache(&self, pipeline: &mut dyn Pipeline) {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.get_metadata().num_hidden_layers {
            new_cache.push(None);
        }
        *pipeline.cache().lock() = new_cache.clone();
        if pipeline.cache().is_xlora() {
            *pipeline.cache().xlora_lock() = new_cache;
        }
    }
}
