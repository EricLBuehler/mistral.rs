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
    pub(crate) num_img_tokens: Option<usize>,
}

/// ImagePreProcessor: process images for the model (similar to `InputsProcessor`, typically called by it)
pub trait ImagePreProcessor: InputsProcessor {
    const DEFAULT_MEAN: [f64; 3];
    const DEFAULT_STD: [f64; 3];

    /// Preprocess the images.
    ///
    /// - `resize` specifies the (w,h) of the target and should be paired with `filter`.
    /// - `filter` filter type for resizing.
    /// - `rescale` multiplies by the scale.
    /// - `normalize` normalizes the image by the mean and std dev (if none, uses default mean/std).
    /// - `do_pad` pads the images to the one with the highest dimensions and will create a pixel attention mask.
    ///   Be sure to set this to `true` if the images differ in dimensions
    /// - `pad_to` pads the images to the specified dimension. This must be greater than or equal to the maximum
    ///   size of a specified image.
    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<PreprocessedImages>;
}
