use std::sync::Arc;

use crate::Pipeline;

use super::{AnyMoePipelineMixin, AnyMoeTrainerMixin};

pub struct AnyMoePipeline {
    target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
}

// TODO
impl AnyMoePipelineMixin for AnyMoePipeline {}

impl AnyMoeTrainerMixin for AnyMoePipeline {
    fn train(&mut self) -> candle_core::Result<crate::amoe::AnyMoeTrainingResult> {
        todo!()
    }
    fn trainable_params(&self) -> usize {
        todo!()
    }
}
