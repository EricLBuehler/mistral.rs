use candle_core::Result;
use mistralrs_quant::ShardedVarBuilder;

use crate::vision_models::conformer::{
    nemo::NemoConvSubsampling,
    pos_embed::{AbsolutePositionalEncoding, T5RelativeAttentionLogitBias},
};

use super::config::ConformerEncoderConfig;

pub struct Encoder {
    embed: NemoConvSubsampling,
    pos_embed: AbsolutePositionalEncoding,
    relative_attention_bias_layer: T5RelativeAttentionLogitBias,
}

impl Encoder {
    pub fn new(mut cfg: ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        assert_eq!(cfg.input_layer, "nemo_conv");

        cfg.finish_nemo_config();
        let embed = NemoConvSubsampling::new(&cfg.nemo_conv_settings, vb.pp("embed"))?;

        let pos_emb = AbsolutePositionalEncoding::new(&cfg, vb.device())?;

        assert!(cfg
            .relative_attention_bias_args
            .as_ref()
            .is_some_and(|x| x.tp == "t5"));
        let relative_attention_bias_args = cfg.relative_attention_bias_args.unwrap();
        let relative_attention_bias_layer = T5RelativeAttentionLogitBias::new(
            cfg.attention_heads / cfg.attention_group_size,
            None,
            relative_attention_bias_args
                .t5_bias_max_distance
                .unwrap_or(1000),
            relative_attention_bias_args
                .t5_bias_symmetric
                .unwrap_or(false),
            vb.pp("relative_attention_bias_layer"),
        )?;

        todo!()
    }
}
