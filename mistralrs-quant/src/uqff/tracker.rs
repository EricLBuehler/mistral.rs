use std::sync::{Arc, Mutex, MutexGuard};

use crate::PendingIsqLayer;

#[derive(Clone)]
pub struct TrackedModule {
    pub key: String,
    pub ct: Arc<PendingIsqLayer>,
    /// The ISQ type resolved at load (topology overrides included); None under capture-all.
    pub ty: Option<crate::IsqType>,
    /// The rank slice this layer's weight was loaded with (identity when replicated); None when
    /// the load applied transforms a shard cannot express (matformer narrowing), which makes the
    /// layer ineligible for from-source requantization.
    pub shard: Option<crate::Shard>,
}

#[derive(Clone)]
pub struct Tracker {
    modules: Arc<Mutex<Vec<TrackedModule>>>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            modules: Arc::new(Mutex::new(vec![])),
        }
    }

    pub fn add_module(&self, module: TrackedModule) {
        self.modules.lock().unwrap().push(module);
    }

    pub fn get(&self) -> MutexGuard<'_, Vec<TrackedModule>> {
        self.modules.lock().unwrap()
    }
}
