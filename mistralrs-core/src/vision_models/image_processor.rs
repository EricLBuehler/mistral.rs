#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use image::DynamicImage;

use crate::pipeline::InputsProcessor;

use super::preprocessor_config::PreProcessorConfig;

#[allow(dead_code)]
pub(crate) struct PreprocessedImages {
    /// Without batch size, safe to unsqueeze & concat in dim0
    /// For QwenVL2: may be vision pixel values, depending on if image_thw or video_thw are specified
    pub(crate) pixel_values: Tensor,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) pixel_attention_mask: Option<Tensor>,
    /// (w, h)
    pub(crate) image_sizes: Option<(usize, usize)>,
    pub(crate) num_img_tokens: Option<Vec<usize>>,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) aspect_ratio_ids: Option<Tensor>,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) aspect_ratio_mask: Option<Tensor>,
    /// Without batch size
    pub(crate) num_tiles: Option<Vec<usize>>,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) image_grid_thw: Option<Tensor>,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) video_grid_thw: Option<Tensor>,
    /// Without batch size
    pub(crate) rows: Option<Vec<usize>>,
    /// Without batch size
    pub(crate) cols: Option<Vec<usize>>,
    /// Without batch size. Only images.
    pub(crate) pixel_values_list: Option<Vec<Tensor>>,
    /// Without batch size, safe to unsqueeze & concat in dim0
    pub(crate) tgt_sizes: Option<Tensor>,
    /// Without batch size. Per image. (h, w)
    pub(crate) image_sizes_all: Option<Vec<(u32, u32)>>,
    /// Without batch size
    pub(crate) num_crops: Option<Vec<usize>>,
}

/// ImagePreProcessor: process images for the model (similar to `InputsProcessor`, typically called by it)
pub trait ImagePreProcessor: InputsProcessor {
    const DEFAULT_MEAN: [f64; 3];
    const DEFAULT_STD: [f64; 3];

    /// Preprocess the images for a specific batch.
    /// `(bs, max_num_images)`, max_num_images is the max images per batches.
    /// Pixel values are in [0, 255]
    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        batch_info: (usize, usize),
    ) -> Result<PreprocessedImages>;
}
