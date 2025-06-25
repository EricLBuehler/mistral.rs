use std::sync::Arc;

use crate::{lora::InstantiatedLoraAdapter, QuantMethod};

pub struct RuntimeLoraLayer {
    base: Arc<dyn QuantMethod>,
    adapters: Vec<InstantiatedLoraAdapter>,
}
