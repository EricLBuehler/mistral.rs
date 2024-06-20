use std::sync::Arc;

use crate::Pipeline;

pub struct AnyMoePipeline {
    draft: Arc<tokio::sync::Mutex<dyn Pipeline>>,
}
