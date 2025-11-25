use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

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
}
