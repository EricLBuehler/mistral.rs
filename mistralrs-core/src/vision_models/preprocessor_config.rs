use std::collections::HashMap;

use candle_core::Result;
use image::imageops::FilterType;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct PreProcessorConfig {
    pub(crate) do_convert_rgb: bool,
    pub(crate) do_image_splitting: bool,
    pub(crate) do_normalize: bool,
    pub(crate) do_pad: bool,
    pub(crate) do_rescale: bool,
    pub(crate) do_resize: bool,
    pub(crate) image_mean: Option<[f64; 3]>,
    pub(crate) image_std: Option<[f64; 3]>,
    pub(crate) rescale_factor: f64,
    pub(crate) resampling: usize,
    pub(crate) size: HashMap<String, u32>,
}

pub(crate) trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for usize {
    // https://github.com/python-pillow/Pillow/blob/4b68563e8a818fb9c528fa159ddf3f4eaefa35e6/src/PIL/Image.py#L164-L170
    fn to_filter(self) -> Result<FilterType> {
        match self {
            0 => Ok(FilterType::Nearest),
            1 => Ok(FilterType::Lanczos3),
            2 => Ok(FilterType::Triangle),   // BiLinear
            3 => Ok(FilterType::CatmullRom), // BiCubic
            4 => Ok(FilterType::Nearest),
            x => candle_core::bail!("Filter number {x} not supported"),
        }
    }
}
