use std::collections::HashMap;

use candle_core::Result;
use image::imageops::FilterType;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct PreProcessorConfig {
    pub do_convert_rgb: Option<bool>,
    pub do_image_splitting: Option<bool>,
    pub do_normalize: Option<bool>,
    pub do_pad: Option<bool>,
    pub do_rescale: Option<bool>,
    pub do_resize: Option<bool>,
    pub do_center_crop: Option<bool>,
    #[serde(alias = "norm_mean")]
    pub image_mean: Option<[f64; 3]>,
    #[serde(alias = "norm_std")]
    pub image_std: Option<[f64; 3]>,
    pub rescale_factor: Option<f64>,
    pub resampling: Option<usize>,
    pub max_image_size: Option<HashMap<String, u32>>,
    pub size: Option<HashMap<String, u32>>,
    pub crop_size: Option<HashMap<String, u32>>,
    pub num_img_tokens: Option<usize>,
    pub num_crops: Option<usize>,
    pub max_image_tiles: Option<usize>,
    pub min_pixels: Option<usize>,
    pub max_pixels: Option<usize>,
    pub patch_size: Option<usize>,
    pub merge_size: Option<usize>,
    pub temporal_patch_size: Option<usize>,
    pub max_slice_nums: Option<usize>,
    pub scale_resolution: Option<usize>,
    pub image_feature_size: Option<usize>,
    pub use_image_id: Option<bool>,
    pub slice_mode: Option<bool>,
    pub im_start_token: Option<String>,
    pub slice_start_token: Option<String>,
    pub unk_token: Option<String>,
    pub im_end_token: Option<String>,
    pub slice_end_token: Option<String>,
    pub im_id_start: Option<String>,
    pub im_id_end: Option<String>,
    pub dynamic_hd: Option<usize>,
}

#[allow(dead_code)]
pub trait ToFilter {
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
