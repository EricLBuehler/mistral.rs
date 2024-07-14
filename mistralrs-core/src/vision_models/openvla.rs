use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use crate::serde_default_fn;

#[derive(Deserialize)]
enum LlmBackboneId {
    #[serde(rename = "llama")]
    Llama,
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "phi")]
    Phi2,
}

#[derive(Deserialize)]
enum ResizeStrategy {
    #[serde(rename = "resize-naive")]
    ResizeNaive,
    #[serde(rename = "resize-crop")]
    Resize,
    #[serde(rename = "letterbox")]
    Letterbox,
}

serde_default_fn!(bool, output_proj_states, false);

#[derive(Deserialize)]
pub struct OpenVLAConfig {
    // Removed as they are redundant/unused info:
    // vision_backbone_id: String,
    // llm_max_length: usize,

    // Prismatic
    llm_backbone_id: LlmBackboneId,
    arch_specifier: String,
    use_fused_vision_backbone: Option<bool>,
    image_resize_strategy: ResizeStrategy,
    text_config: Option<Value>,
    pad_to_multiple_of: usize,
    #[serde(default = "output_proj_states")]
    output_projector_states: bool,
    timm_model_ids: Vec<String>,
    timm_override_act_layers: Vec<Option<String>>,
    image_sizes: Vec<usize>,

    // OpenVLA
    n_action_bins: usize,
    #[allow(clippy::type_complexity)]
    norm_stats:
        Option<IndexMap<String, IndexMap<String, IndexMap<String, IndexMap<String, Vec<f64>>>>>>,
}
