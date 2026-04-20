use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

use super::codec::KvCacheCodec;
use super::{Cache, HybridCache, NormalCache};

pub type LayerCaches = Vec<Option<(Tensor, Tensor)>>;

#[derive(Debug, Clone)]
pub enum EitherCache {
    Normal(Arc<Mutex<NormalCache>>),
    Full(Cache),
    Hybrid(Arc<Mutex<HybridCache>>),
}

impl EitherCache {
    /// Panics otherwise!
    pub fn full(&self) -> &Cache {
        match self {
            Self::Full(full) => full,
            Self::Normal(_) => panic!("Got normal cache, expected full cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected full cache."),
        }
    }

    /// Panics otherwise!
    pub fn normal(&self) -> MutexGuard<'_, NormalCache> {
        match self {
            Self::Normal(normal) => normal.lock().unwrap(),
            Self::Full(_) => panic!("Got full cache, expected normal cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected normal cache."),
        }
    }

    /// Panics otherwise!
    pub fn hybrid(&self) -> MutexGuard<'_, HybridCache> {
        match self {
            Self::Hybrid(hybrid) => hybrid.lock().unwrap(),
            Self::Normal(_) => panic!("Got normal cache, expected hybrid cache."),
            Self::Full(_) => panic!("Got full cache, expected hybrid cache."),
        }
    }

    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    /// Install a `KvCacheCodec` on every attention-layer KV cache reachable
    /// from this cache handle. Returns the number of layers where the codec
    /// was installed.
    ///
    /// - `Normal`: installs on every `KvCache` layer (both K and V).
    /// - `Hybrid`: installs on attention layers only; recurrent layers have
    ///   no codec concept and are skipped.
    /// - `Full`: not layered — returns 0.
    ///
    /// Safe to call multiple times; each call replaces the previously
    /// installed codec. Call *before* inference starts — switching mid-stream
    /// leaves a mix of encoded / unencoded data in the buffer.
    pub fn set_kv_cache_codec(&self, codec: Arc<dyn KvCacheCodec>) -> usize {
        match self {
            Self::Normal(cache) => {
                let mut cache = cache.lock().unwrap();
                let mut count = 0;
                for layer in cache.0.iter_mut() {
                    layer.set_codec(codec.clone());
                    count += 1;
                }
                count
            }
            Self::Hybrid(hybrid) => {
                let mut hybrid = hybrid.lock().unwrap();
                let mut count = 0;
                for layer in hybrid.caches.iter_mut() {
                    if let Some(kv) = layer.as_kv_cache_mut() {
                        kv.set_codec(codec.clone());
                        count += 1;
                    }
                }
                count
            }
            Self::Full(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::codec::PassthroughCodec;
    use super::super::{KvCache, NormalCache};
    use super::EitherCache;

    #[test]
    fn normal_installs_codec_on_every_layer() {
        let cache = EitherCache::Normal(Arc::new(std::sync::Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 32, 16),
            KvCache::new_normal(2, 32, 16),
            KvCache::new_shared(0),
            KvCache::new_rotating(2, 16, 16),
        ]))));

        let count = cache.set_kv_cache_codec(Arc::new(PassthroughCodec));
        // 4 layers touched — Shared is a no-op internally but still counts
        // toward "layers visited" in Normal.
        assert_eq!(count, 4);

        // Verify the codec actually landed on Normal/Rotating layers.
        let guard = cache.normal();
        match &guard.0[0] {
            KvCache::Normal { k, v } => {
                assert!(k.codec.is_some());
                assert!(v.codec.is_some());
            }
            other => panic!("expected Normal, got {other:?}"),
        }
        match &guard.0[3] {
            KvCache::Rotating { k, v } => {
                assert!(k.codec.is_some());
                assert!(v.codec.is_some());
            }
            other => panic!("expected Rotating, got {other:?}"),
        }
    }
}
