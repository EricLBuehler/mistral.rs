use candle_core::Result;
use mistralrs_quant::ShardedVarBuilder;

use crate::vision_models::conformer::{nemo::NemoConvSubsampling, pos_embed::AbsolutePositionalEncoding};

use super::config::ConformerEncoderConfig;

pub struct Encoder {
    embed: NemoConvSubsampling,
    pos_embed: AbsolutePositionalEncoding
}

impl Encoder {
    pub fn new(mut cfg: ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        assert_eq!(cfg.input_layer, "nemo_conv");

        cfg.finish_nemo_config();
        let embed = NemoConvSubsampling::new(&cfg.nemo_conv_settings, vb.pp("embed"))?;

        let pos_emb = AbsolutePositionalEncoding::new(&cfg, vb.device())?;

        todo!()
    }
}
