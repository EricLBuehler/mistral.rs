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
    pub(crate) do_center_crop: Option<bool>,
    #[serde(alias = "norm_mean")]
    pub(crate) image_mean: Option<[f64; 3]>,
    #[serde(alias = "norm_std")]
    pub(crate) image_std: Option<[f64; 3]>,
    pub(crate) rescale_factor: Option<f64>,
    pub(crate) resampling: Option<usize>,
    pub(crate) max_image_size: Option<HashMap<String, u32>>,
    pub(crate) size: Option<HashMap<String, u32>>,
    pub(crate) crop_size: Option<HashMap<String, u32>>,
    pub(crate) num_img_tokens: Option<usize>,
    pub(crate) num_crops: Option<usize>,
    pub(crate) max_image_tiles: Option<usize>,
    pub(crate) min_pixels: Option<usize>,
    pub(crate) max_pixels: Option<usize>,
    pub(crate) patch_size: Option<usize>,
    pub(crate) merge_size: Option<usize>,
    pub(crate) temporal_patch_size: Option<usize>,
    pub(crate) max_slice_nums: Option<usize>,
    pub(crate) scale_resolution: Option<usize>,
    pub(crate) image_feature_size: Option<usize>,
    pub(crate) use_image_id: Option<bool>,
    pub(crate) slice_mode: Option<bool>,
    pub(crate) im_start_token: Option<String>,
    pub(crate) slice_start_token: Option<String>,
    pub(crate) unk_token: Option<String>,
    pub(crate) im_end_token: Option<String>,
    pub(crate) slice_end_token: Option<String>,
    pub(crate) im_id_start: Option<String>,
    pub(crate) im_id_end: Option<String>,
    pub(crate) dynamic_hd: Option<usize>,
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
