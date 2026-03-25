use std::collections::HashMap;

use candle_core::Result;
use image::imageops::FilterType;
use serde::Deserialize;
use serde_json::{Map, Value};

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

impl PreProcessorConfig {
    pub fn from_processor_config_json(json: &str) -> serde_json::Result<Self> {
        let value: Value = serde_json::from_str(json)?;
        let mut merged = Map::new();

        if let Some(obj) = value.as_object() {
            for (key, value) in obj {
                if key != "image_processor" && key != "feature_extractor" && !value.is_null() {
                    merged.insert(key.clone(), value.clone());
                }
            }

            if let Some(image_processor) = obj.get("image_processor").and_then(Value::as_object) {
                Self::merge_processor_section(&mut merged, image_processor);
            }
            if let Some(feature_extractor) = obj.get("feature_extractor").and_then(Value::as_object)
            {
                Self::merge_processor_section(&mut merged, feature_extractor);
            }
        }

        serde_json::from_value(Value::Object(merged))
    }

    fn merge_processor_section(target: &mut Map<String, Value>, source: &Map<String, Value>) {
        for (key, value) in source {
            if !value.is_null() {
                target.insert(key.clone(), value.clone());
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::PreProcessorConfig;

    #[test]
    fn parses_nested_processor_config_sections() {
        let json = r#"
        {
            "audio_seq_length": 750,
            "image_processor": {
                "do_convert_rgb": true,
                "do_resize": true,
                "resample": 3,
                "size": { "height": 224, "width": 224 }
            },
            "feature_extractor": {
                "feature_size": 128,
                "fft_overdrive": false,
                "frame_length": 320,
                "hop_length": 160,
                "min_frequency": 0.0,
                "max_frequency": 8000.0,
                "preemphasis": 0.0,
                "mel_floor": 0.001,
                "sampling_rate": 16000
            }
        }
        "#;

        let config = PreProcessorConfig::from_processor_config_json(json).unwrap();

        assert_eq!(config.do_convert_rgb, Some(true));
        assert_eq!(config.do_resize, Some(true));
        assert_eq!(config.resampling, Some(3));
        assert_eq!(config.feature_size, Some(128));
        assert_eq!(config.fft_overdrive, Some(false));
        assert_eq!(config.frame_length, Some(320));
        assert_eq!(config.hop_length, Some(160));
        assert_eq!(config.min_frequency, Some(0.0));
        assert_eq!(config.max_frequency, Some(8000.0));
        assert_eq!(config.preemphasis, Some(0.0));
        assert_eq!(config.mel_floor, Some(0.001));
        assert_eq!(config.sampling_rate, Some(16000));
        assert_eq!(
            config
                .size
                .as_ref()
                .and_then(|size| size.get("height"))
                .copied(),
            Some(224)
        );
        assert_eq!(
            config
                .size
                .as_ref()
                .and_then(|size| size.get("width"))
                .copied(),
            Some(224)
        );
    }
}
