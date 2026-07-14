use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMConfig {
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTableConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SEANetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    pub max_period: f64,
    pub dim_feedforward: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MimiConfig {
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f64,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PocketTtsConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
}

impl PocketTtsConfig {
    /// The single published pocket-tts variant `b6369a24`. Values mirror `b6369a24.yaml` in the
    /// upstream crate; the HF repo ships only weights + tokenizer, no `config.json`.
    pub fn b6369a24() -> Self {
        Self {
            flow_lm: FlowLMConfig {
                flow: FlowConfig { dim: 512, depth: 6 },
                transformer: FlowLMTransformerConfig {
                    hidden_scale: 4,
                    max_period: 10000,
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 6,
                },
                lookup_table: LookupTableConfig {
                    dim: 1024,
                    n_bins: 4000,
                },
            },
            mimi: MimiConfig {
                sample_rate: 24000,
                channels: 1,
                frame_rate: 12.5,
                seanet: SEANetConfig {
                    dimension: 512,
                    channels: 1,
                    n_filters: 64,
                    n_residual_layers: 1,
                    ratios: vec![6, 5, 4],
                    kernel_size: 7,
                    residual_kernel_size: 3,
                    last_kernel_size: 3,
                    dilation_base: 2,
                    pad_mode: "constant".to_string(),
                    compress: 2,
                },
                transformer: MimiTransformerConfig {
                    d_model: 512,
                    input_dimension: 512,
                    output_dimensions: vec![512],
                    num_heads: 8,
                    num_layers: 2,
                    layer_scale: 0.01,
                    context: 250,
                    max_period: 10000.0,
                    dim_feedforward: 2048,
                },
                quantizer: QuantizerConfig {
                    dimension: 32,
                    output_dimension: 512,
                },
            },
        }
    }
}

pub mod defaults {
    pub const TEMPERATURE: f32 = 0.7;
    pub const LSD_DECODE_STEPS: usize = 1;
    pub const EOS_THRESHOLD: f32 = -4.0;
}
