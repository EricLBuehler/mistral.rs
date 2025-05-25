use mistralrs::Model;
use std::sync::Arc;

/// Distinguish at runtime which kind of model we have loaded.
#[derive(Clone)]
pub enum LoadedModel {
    Text(Arc<Model>),
    Vision(Arc<Model>),
    /// Speech models for text-to-speech generation
    Speech(Arc<Model>),
}
