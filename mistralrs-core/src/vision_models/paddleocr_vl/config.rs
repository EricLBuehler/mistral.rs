//! ERNIE-4.5-0.3B text-model hyperparameters (the LM half of PaddleOCR-VL-1.5).
//!
//! Values are hard-wired from the checkpoint config (verified against the native transformers
//! 5.13 `PaddleOCRTextConfig`): 18 layers, hidden 1024, GQA 16 heads / 2 KV heads x head_dim 128,
//! SwiGLU intermediate 3072, RMSNorm eps 1e-5, 3D chunked mrope theta=500000 sections [16,24,24].

/// Config for the ERNIE-4.5 dense decoder. `Default` = the shipped PaddleOCR-VL-1.5 values.
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// mrope_section (temporal, height, width); sum = head_dim/2 = 64.
    pub mrope_section: [usize; 3],
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 18,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            intermediate_size: 3072,
            vocab_size: 103424,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            mrope_section: [16, 24, 24],
        }
    }
}

/// Config for the SigLIP/NaViT vision tower (the checkpoint config's `vision_config`).
///
/// `head_dim` and `rope_theta` are `null` in the JSON: head_dim = hidden/num_heads = 72; the
/// SigLIP 2D axial rope uses theta=10000. `num_positions = pos_grid^2` is the
/// learned 27x27 table (image_size 384 / patch 14 = 27), bilinearly interpolated to native grids.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    /// side of the learned position table (image_size / patch_size = 27).
    pub pos_grid: usize,
    /// number of learned position rows (pos_grid^2, 729).
    pub num_positions: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub spatial_merge_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            head_dim: 72,
            intermediate_size: 4304,
            patch_size: 14,
            num_channels: 3,
            pos_grid: 27,
            num_positions: 729,
            layer_norm_eps: 1e-6,
            rope_theta: 10000.0,
            spatial_merge_size: 2,
        }
    }
}

// HF config.json deserialization (mistralrs-core loader path only, not in the port/ mirror).
// The hand-wired `TextConfig`/`VisionConfig` above are what the numerical kernels consume; `Config`
// below parses the real checkpoint config and produces those verified structs via `text_config()` /
// `vision_config()`. Loader keys the model by `architectures[0]`, not `model_type`.

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfigRaw {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub layer_norm_eps: f64,
    pub spatial_merge_size: usize,
}

/// Full HF `config.json` for PaddleOCR-VL-1.5: the ERNIE-4.5 LM fields sit flat at the top level
/// (the native custom-code `__post_init__` parses this flat layout) with a nested `vision_config`
/// for the SigLIP tower, unlike qwen3_vl, which nests both under `text_config`/`vision_config`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: MRopeScaling,
    pub max_position_embeddings: usize,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    pub vision_config: VisionConfigRaw,
}

/// theta for the SigLIP 2D axial rope: `null` in config.json, fixed by the reference.
const VISION_ROPE_THETA: f64 = 10000.0;

impl Config {
    pub fn text_config(&self) -> TextConfig {
        let ms = &self.rope_scaling.mrope_section;
        TextConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            mrope_section: [ms[0], ms[1], ms[2]],
        }
    }

    pub fn vision_config(&self) -> VisionConfig {
        let v = &self.vision_config;
        let pos_grid = v.image_size / v.patch_size;
        VisionConfig {
            hidden_size: v.hidden_size,
            num_hidden_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            head_dim: v.hidden_size / v.num_attention_heads,
            intermediate_size: v.intermediate_size,
            patch_size: v.patch_size,
            num_channels: v.num_channels,
            pos_grid,
            num_positions: pos_grid * pos_grid,
            layer_norm_eps: v.layer_norm_eps,
            rope_theta: VISION_ROPE_THETA,
            spatial_merge_size: v.spatial_merge_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The serde `Config` must read the real checkpoint config.json and reproduce exactly the
    // parity-verified constants the numerical kernels were validated against.
    #[test]
    fn config_json_matches_verified_constants() {
        let json = include_str!("reference_config.json");
        let cfg: Config = serde_json::from_str(json).unwrap();

        let (t, td) = (cfg.text_config(), TextConfig::default());
        assert_eq!(t.hidden_size, td.hidden_size);
        assert_eq!(t.num_hidden_layers, td.num_hidden_layers);
        assert_eq!(t.num_attention_heads, td.num_attention_heads);
        assert_eq!(t.num_key_value_heads, td.num_key_value_heads);
        assert_eq!(t.head_dim, td.head_dim);
        assert_eq!(t.intermediate_size, td.intermediate_size);
        assert_eq!(t.vocab_size, td.vocab_size);
        assert_eq!(t.rms_norm_eps, td.rms_norm_eps);
        assert_eq!(t.rope_theta, td.rope_theta);
        assert_eq!(t.mrope_section, td.mrope_section);

        let (v, vd) = (cfg.vision_config(), VisionConfig::default());
        assert_eq!(v.hidden_size, vd.hidden_size);
        assert_eq!(v.num_hidden_layers, vd.num_hidden_layers);
        assert_eq!(v.num_attention_heads, vd.num_attention_heads);
        assert_eq!(v.head_dim, vd.head_dim);
        assert_eq!(v.intermediate_size, vd.intermediate_size);
        assert_eq!(v.patch_size, vd.patch_size);
        assert_eq!(v.num_channels, vd.num_channels);
        assert_eq!(v.pos_grid, vd.pos_grid);
        assert_eq!(v.num_positions, vd.num_positions);
        assert_eq!(v.layer_norm_eps, vd.layer_norm_eps);
        assert_eq!(v.rope_theta, vd.rope_theta);
        assert_eq!(v.spatial_merge_size, vd.spatial_merge_size);

        assert_eq!(cfg.image_token_id, 100295);
        assert_eq!(cfg.vision_start_token_id, 101305);
        assert_eq!(cfg.vision_end_token_id, 101306);
        assert!(!cfg.tie_word_embeddings);
    }
}
