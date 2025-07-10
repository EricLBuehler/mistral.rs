use std::collections::HashMap;

use candle_core::Result;
use image::imageops::FilterType;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Default)]
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
    #[serde(alias = "resample")]
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
    #[serde(alias = "image_seq_length")]
    pub(crate) image_seq_len: Option<usize>,
    pub(crate) pan_and_scan_min_crop_size: Option<usize>,
    pub(crate) pan_and_scan_max_num_crops: Option<usize>,
    pub(crate) pan_and_scan_min_ratio_to_activate: Option<f64>,
    pub(crate) do_pan_and_scan: Option<bool>,
    pub(crate) default_to_square: Option<bool>,
    pub(crate) max_patches: Option<usize>,
    pub(crate) resize_to_max_canvas: Option<bool>,

    pub(crate) audio_compression_rate: Option<usize>,
    pub(crate) audio_downsample_rate: Option<usize>,
    pub(crate) audio_feat_stride: Option<usize>,

    // Audio feature extraction configuration
    pub(crate) dither: Option<f64>,
    pub(crate) feature_size: Option<usize>,
    pub(crate) fft_length: Option<usize>,
    pub(crate) fft_overdrive: Option<bool>,
    pub(crate) frame_length: Option<usize>,
    pub(crate) hop_length: Option<usize>,
    pub(crate) input_scale_factor: Option<f64>,
    pub(crate) max_frequency: Option<f64>,
    pub(crate) mel_floor: Option<f64>,
    pub(crate) min_frequency: Option<f64>,
    pub(crate) padding_side: Option<String>,
    pub(crate) padding_value: Option<f64>,
    pub(crate) per_bin_mean: Option<Vec<f64>>,
    pub(crate) per_bin_stddev: Option<Vec<f64>>,
    pub(crate) preemphasis: Option<f64>,
    pub(crate) preemphasis_htk_flavor: Option<bool>,
    pub(crate) return_attention_mask: Option<bool>,
    pub(crate) sampling_rate: Option<usize>,
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
