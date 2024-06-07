#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use image::DynamicImage;

use crate::pipeline::InputsProcessor;

use super::preprocessor_config::PreProcessorConfig;

#[allow(dead_code)]
pub(crate) struct PreprocessedImages {
    /// Without batch size, safe to concat in dim0
    pub(crate) pixel_values: Tensor,
    /// Without batch size, safe to concat in dim0
    pub(crate) pixel_attention_mask: Option<Tensor>,
    pub(crate) image_sizes: Option<(usize, usize)>,
    pub(crate) num_img_tokens: Option<Vec<usize>>,
}

/// ImagePreProcessor: process images for the model (similar to `InputsProcessor`, typically called by it)
pub trait ImagePreProcessor: InputsProcessor {
    const DEFAULT_MEAN: [f64; 3];
    const DEFAULT_STD: [f64; 3];

    /// Preprocess the images for a specific batch.
    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<PreprocessedImages>;
}
