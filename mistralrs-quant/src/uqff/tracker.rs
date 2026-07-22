use std::sync::{Arc, Mutex, MutexGuard};

use crate::PendingIsqLayer;

#[derive(Clone)]
pub struct TrackedModule {
    pub key: String,
    pub ct: Arc<PendingIsqLayer>,
    /// The ISQ type resolved at load (topology overrides included); None under capture-all.
    pub ty: Option<crate::IsqType>,
    /// The rank slice the weight was loaded with; None when the load applied a transform a shard
    /// cannot express, making the layer ineligible for from-source requantization.
    pub shard: Option<crate::Shard>,
}

impl TrackedModule {
    pub fn resolve_type(&self, default: crate::IsqType) -> crate::IsqType {
        self.ty
            .unwrap_or_else(|| default.resolve_for_tensor(self.key.as_str()))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn module(key: &str, ty: Option<crate::IsqType>) -> TrackedModule {
        let (_tx, rx) = crate::pending_isq_channel();
        TrackedModule {
            key: key.to_string(),
            ct: Arc::new(crate::PendingIsqLayer::new(rx)),
            ty,
            shard: None,
        }
    }

    #[test]
    fn default_type_uses_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", None);
        assert_eq!(
            module.resolve_type(crate::IsqType::AFQ4),
            crate::IsqType::AFQ6
        );
    }

    #[test]
    fn explicit_type_wins_over_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", Some(crate::IsqType::AFQ2));
        assert_eq!(
            module.resolve_type(crate::IsqType::AFQ4),
            crate::IsqType::AFQ2
        );
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
