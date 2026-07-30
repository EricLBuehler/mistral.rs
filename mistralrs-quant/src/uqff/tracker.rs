use std::sync::{Arc, Mutex, MutexGuard};

use crate::PendingIsqLayer;

#[derive(Clone)]
pub struct TrackedModule {
    pub key: String,
    pub ct: Arc<PendingIsqLayer>,
    /// The ISQ type resolved at load (topology overrides included); None under capture-all.
    pub ty: Option<crate::IsqType>,
    /// Whether the model loader marked this module for default-type promotion.
    pub promote_default: bool,
    /// The rank slice the weight was loaded with; None when the load applied a transform a shard
    /// cannot express, making the layer ineligible for from-source requantization.
    pub shard: Option<crate::Shard>,
}

impl TrackedModule {
    pub fn default_type(&self, default: crate::IsqType) -> crate::IsqType {
        if self.promote_default {
            default.promote_for_sensitive_tensor()
        } else {
            default
        }
    }

    pub fn resolve_type(&self, default: crate::IsqType) -> crate::IsqType {
        self.ty.unwrap_or_else(|| self.default_type(default))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn module(key: &str, ty: Option<crate::IsqType>, promote_default: bool) -> TrackedModule {
        let (_tx, rx) = crate::pending_isq_channel();
        TrackedModule {
            key: key.to_string(),
            ct: Arc::new(crate::PendingIsqLayer::new(rx)),
            ty,
            promote_default,
            shard: None,
        }
    }

    #[test]
    fn default_type_uses_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", None, true);
        assert_eq!(
            module.resolve_type(crate::IsqType::AFQ4),
            crate::IsqType::AFQ6
        );
    }

    #[test]
    fn explicit_type_wins_over_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", Some(crate::IsqType::AFQ2), true);
        assert_eq!(
            module.resolve_type(crate::IsqType::AFQ4),
            crate::IsqType::AFQ2
        );
    }

    #[test]
    fn q_defaults_use_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", None, true);
        assert_eq!(
            module.resolve_type(crate::IsqType::Q4K),
            crate::IsqType::Q6K
        );
        assert_eq!(
            module.resolve_type(crate::IsqType::Q6K),
            crate::IsqType::Q8_0
        );
    }

    #[test]
    fn explicit_q_type_wins_over_sensitive_tensor_policy() {
        let module = module("model.embed_tokens", Some(crate::IsqType::Q4_0), true);
        assert_eq!(
            module.resolve_type(crate::IsqType::Q4K),
            crate::IsqType::Q4_0
        );
    }

    #[test]
    fn tensor_name_does_not_enable_promotion() {
        let module = module("model.embed_tokens", None, false);
        assert_eq!(
            module.resolve_type(crate::IsqType::AFQ4),
            crate::IsqType::AFQ4
        );
    }
}
