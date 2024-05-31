use std::collections::HashMap;

use candle_core::Result;
use image::imageops::FilterType;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct PreProcessorConfig {
    pub(crate) do_convert_rgb: Option<bool>,
    pub(crate) do_image_splitting: Option<bool>,
    pub(crate) do_normalize: Option<bool>,
    pub(crate) do_pad: Option<bool>,
    pub(crate) do_rescale: Option<bool>,
    pub(crate) do_resize: Option<bool>,
    pub(crate) image_mean: Option<[f64; 3]>,
    pub(crate) image_std: Option<[f64; 3]>,
    pub(crate) rescale_factor: Option<f64>,
    pub(crate) resampling: Option<usize>,
    pub(crate) size: Option<HashMap<String, u32>>,
    pub(crate) num_img_tokens: Option<usize>,
    pub(crate) num_crops: Option<usize>,
}

#[allow(dead_code)]
pub(crate) trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for Option<usize> {
    // https://github.com/python-pillow/Pillow/blob/4b68563e8a818fb9c528fa159ddf3f4eaefa35e6/src/PIL/Image.py#L164-L170
    // Default: https://github.com/huggingface/transformers/blob/0df888ffb72ea370555efdef45985378d3cc7b2b/src/transformers/models/idefics2/image_processing_idefics2.py#L226
    fn to_filter(self) -> Result<FilterType> {
        match self {
            Some(0) => Ok(FilterType::Nearest),
            Some(1) => Ok(FilterType::Lanczos3),
            Some(2) | None => Ok(FilterType::Triangle), // BiLinear
            Some(3) => Ok(FilterType::CatmullRom),      // BiCubic
            Some(4) => Ok(FilterType::Nearest),
            Some(x) => candle_core::bail!("Filter number {x} not supported"),
        }
    }
}
