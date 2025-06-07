use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(usize, default_attention_dim, 256);
serde_default_fn!(usize, default_attention_heads, 4);
serde_default_fn!(usize, default_linear_units, 2048);
serde_default_fn!(usize, default_num_blocks, 6);
serde_default_fn!(String, default_input_layer, "nemo_conv".to_string());
serde_default_fn!(bool, default_causal, true);
serde_default_fn!(bool, default_batch_norm, false);
serde_default_fn!(usize, default_ext_pw_out_channel, 0);
serde_default_fn!(usize, default_ext_pw_kernel_size, 1);
serde_default_fn!(usize, default_depthwise_seperable_out_channel, 256);
serde_default_fn!(usize, default_depthwise_multiplier, 1);
serde_default_fn!(usize, default_chunk_se, 0);
serde_default_fn!(usize, default_kernel_size, 3);
serde_default_fn!(Activation, default_activation, Activation::Relu);
serde_default_fn!(Activation, default_conv_activation, Activation::Relu);
serde_default_fn!(Activation, default_conv_glu_type, Activation::Sigmoid);
serde_default_fn!(bool, default_bias_in_glu, true);
serde_default_fn!(bool, default_linear_glu_in_convm, false);
serde_default_fn!(String, default_attention_glu_type, "swish".to_string());
serde_default_fn!(bool, default_export, false);
serde_default_fn!(i32, default_extra_layer_output_idx, -1);
serde_default_fn!(usize, default_time_reduction, 4);
serde_default_fn!(bool, default_replication_pad_for_subsample_embedding, false);
serde_default_fn!(usize, default_attention_group_size, 1);
serde_default_fn!(String, default_subsampling, "dw_striding".to_string());
serde_default_fn!(usize, default_conv_channels, 256);
serde_default_fn!(usize, default_subsampling_conv_chunking_factor, 1);
serde_default_fn!(Activation, default_nemo_activation, Activation::Relu);
serde_default_fn!(bool, default_nemo_is_causal, false);
serde_default_fn!(usize, fake_default_sentinel, usize::MAX);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RelativeAttentionBiasArgs {
    pub t5_bias_max_distance: Option<usize>,
    pub t5_bias_symmetric: Option<bool>,
    #[serde(rename = "type")]
    pub tp: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NemoConvConfig {
    #[serde(default = "default_subsampling")]
    pub subsampling: String,
    #[serde(default = "fake_default_sentinel")]
    pub subsampling_factor: usize,
    #[serde(default = "fake_default_sentinel")]
    pub feat_in: usize,
    #[serde(default = "fake_default_sentinel")]
    pub feat_out: usize,
    #[serde(default = "default_conv_channels")]
    pub conv_channels: usize,
    #[serde(default = "default_subsampling_conv_chunking_factor")]
    pub subsampling_conv_chunking_factor: usize,
    #[serde(default = "default_nemo_activation")]
    pub activation: Activation,
    #[serde(default = "default_nemo_is_causal")]
    pub is_causal: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EncoderEmbeddingConfig {
    pub input_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConformerEncoderConfig {
    pub input_size: usize,
    pub chunk_size: i32,
    pub left_chunk: usize,
    pub num_lang: Option<usize>,
    #[serde(default = "default_attention_dim")]
    pub attention_dim: usize,
    #[serde(default = "default_attention_heads")]
    pub attention_heads: usize,
    #[serde(default = "default_linear_units")]
    pub linear_units: usize,
    #[serde(default = "default_num_blocks")]
    pub num_blocks: usize,
    #[serde(default = "default_input_layer")]
    pub input_layer: String,
    #[serde(default = "default_causal")]
    pub causal: bool,
    #[serde(default = "default_batch_norm")]
    pub batch_norm: bool,
    #[serde(default = "default_ext_pw_out_channel")]
    pub ext_pw_out_channel: usize,
    #[serde(default = "default_ext_pw_kernel_size")]
    pub ext_pw_kernel_size: usize,
    #[serde(default = "default_depthwise_seperable_out_channel")]
    pub depthwise_seperable_out_channel: usize,
    #[serde(default = "default_depthwise_multiplier")]
    pub depthwise_multiplier: usize,
    #[serde(default = "default_chunk_se")]
    pub chunk_se: usize,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    #[serde(default = "default_activation")]
    pub activation: Activation,
    #[serde(default = "default_conv_activation")]
    pub conv_activation: Activation,
    #[serde(default = "default_conv_glu_type")]
    pub conv_glu_type: Activation,
    #[serde(default = "default_bias_in_glu")]
    pub bias_in_glu: bool,
    #[serde(default = "default_linear_glu_in_convm")]
    pub linear_glu_in_convm: bool,
    #[serde(default = "default_attention_glu_type")]
    pub attention_glu_type: String,
    #[serde(default = "default_export")]
    pub export: bool,
    #[serde(default = "default_extra_layer_output_idx")]
    pub extra_layer_output_idx: i32,
    pub relative_attention_bias_args: Option<RelativeAttentionBiasArgs>,
    #[serde(default = "default_time_reduction")]
    pub time_reduction: usize,
    pub nemo_conv_settings: NemoConvConfig,
    #[serde(default = "default_replication_pad_for_subsample_embedding")]
    pub replication_pad_for_subsample_embedding: bool,
    #[serde(default = "default_attention_group_size")]
    pub attention_group_size: usize,
    pub encoder_embedding_config: Option<EncoderEmbeddingConfig>,
}

impl ConformerEncoderConfig {
    pub fn finish_nemo_config(&mut self) {
        // Override any of the defaults with the incoming, user settings
        if self.nemo_conv_settings.subsampling_factor == usize::MAX {
            self.nemo_conv_settings.subsampling_factor = self.time_reduction;
        }
        if self.nemo_conv_settings.feat_in == usize::MAX {
            self.nemo_conv_settings.feat_in = self.input_size;
        }
        if self.nemo_conv_settings.feat_out == usize::MAX {
            self.nemo_conv_settings.feat_out = self.attention_dim;
        }
    }
}
