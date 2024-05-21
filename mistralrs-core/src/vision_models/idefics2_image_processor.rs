use candle_core::Result;
use image::DynamicImage;

use super::image_processor::{ImagePreProcessor, NormalizationMetadata, PreprocessedImages};

pub struct Idefics2ImageProcessor;

impl ImagePreProcessor for Idefics2ImageProcessor {
    fn preprocess(
        &self,
        images: &[DynamicImage],
        do_convert_rgb: bool,
        do_resize: bool,
        rescale: Option<f32>,
        normalize: Option<NormalizationMetadata>,
        do_pad: bool,
        do_image_splitting: bool,
    ) -> Result<PreprocessedImages> {
        todo!()
    }
}
