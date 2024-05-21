use candle_core::{Result, Tensor};
use image::DynamicImage;

pub struct NormalizationMetadata {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

pub struct PreprocessedImages {
    pixel_values: Tensor,
    pixel_attention_mask: Tensor,
}

pub trait ImagePreProcessor {
    fn preprocess(
        &self,
        images: &[DynamicImage],
        do_convert_rgb: bool,
        do_resize: bool,
        rescale: Option<f32>,
        normalize: Option<NormalizationMetadata>,
        do_pad: bool,
        do_image_splitting: bool,
    ) -> Result<PreprocessedImages>;
}
