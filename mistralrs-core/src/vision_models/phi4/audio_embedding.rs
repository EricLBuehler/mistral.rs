use std::{collections::HashMap, sync::Arc};

use candle_core::Result;
use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    layers::{self, Activation},
    vision_models::{
        conformer::encoder::ConformerEncoder,
        phi4::config::{Phi4MMAudioConfig, Phi4MMAudioEmbedConfig},
    },
};

use super::Phi4MMConfig;

pub(super) const AUDIO_SPECIAL_TOKEN_ID: f64 = 200011.;

#[derive(Eq, Hash, PartialEq)]
pub enum AudioProjectionMode {
    /// If only speech
    Speech,
    /// If vision + speech or only vision (not sure why that is necesary though)
    Vision,
}

pub struct AudioEmbedding {
    proj: HashMap<AudioProjectionMode, Vec<Arc<dyn Module + Send + Sync>>>,
    encoder: ConformerEncoder,
}

impl AudioEmbedding {
    pub fn new(
        cfg: &Phi4MMConfig,
        audio_embd_config: &Phi4MMAudioEmbedConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_size = audio_embd_config.n_embd.unwrap_or(cfg.hidden_size);

        let conformer_config = match &cfg.audio_processor {
            Some(Phi4MMAudioConfig { config, name }) if name == "cascades" => config,
            _ => candle_core::bail!("Must have audio processor (`cascades`)"),
        };
        let encoder = ConformerEncoder::new(conformer_config.clone(), vb.pp("encoder"))?;

        // let audio_dim_in = conformer_config.input_size;
        let audio_dim_out = conformer_config.attention_dim;

        let mut proj = HashMap::new();
        {
            assert_eq!(audio_embd_config.projection_cls, "mlp");

            let dim_projection = hidden_size;
            let depth = 2;
            let linear_downsample_rate = audio_embd_config.downsample_rate;

            let embedding_cls_vb = vb.pp("audio_projection");

            let mut layers_for_speech: Vec<Arc<dyn Module + Send + Sync>> =
                vec![Arc::new(layers::linear(
                    audio_dim_out * linear_downsample_rate,
                    dim_projection,
                    embedding_cls_vb.pp("speech").pp(0),
                )?)];
            for i in 1..depth {
                layers_for_speech.push(Arc::new(Activation::Gelu));
                layers_for_speech.push(Arc::new(layers::linear(
                    dim_projection,
                    dim_projection,
                    embedding_cls_vb.pp("speech").pp(i + 1),
                )?));
            }

            let mut layers_for_vision: Vec<Arc<dyn Module + Send + Sync>> =
                vec![Arc::new(layers::linear(
                    audio_dim_out * linear_downsample_rate,
                    dim_projection,
                    embedding_cls_vb.pp("vision").pp(0),
                )?)];
            for i in 1..depth {
                layers_for_vision.push(Arc::new(Activation::Gelu));
                layers_for_vision.push(Arc::new(layers::linear(
                    dim_projection,
                    dim_projection,
                    embedding_cls_vb.pp("vision").pp(i + 1),
                )?));
            }

            proj.insert(AudioProjectionMode::Speech, layers_for_speech);
            proj.insert(AudioProjectionMode::Vision, layers_for_vision);
        }

        Ok(Self { proj, encoder })
    }
}
