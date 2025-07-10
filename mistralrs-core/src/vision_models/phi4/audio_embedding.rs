use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};

use crate::{
    layers::{self, Activation},
    vision_models::{
        conformer::encoder::ConformerEncoder,
        phi4::{
            config::{Phi4MMAudioConfig, Phi4MMAudioEmbedConfig},
            mm_embedding::InputMode,
        },
    },
};

use super::Phi4MMConfig;

pub(super) const AUDIO_SPECIAL_TOKEN_ID: f64 = 200011.;

pub struct AudioEmbedding {
    wte: candle_nn::Embedding,
    proj: HashMap<InputMode, Vec<Arc<dyn Module + Send + Sync>>>,
    encoder: ConformerEncoder,
    target_device_dtype: (Device, DType),
}

impl AudioEmbedding {
    pub fn new(
        cfg: &Phi4MMConfig,
        wte: candle_nn::Embedding,
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

            proj.insert(InputMode::Speech, layers_for_speech);
            proj.insert(InputMode::Vision, layers_for_vision);
        }

        Ok(Self {
            wte,
            proj,
            encoder,
            target_device_dtype: (vb.device().clone(), vb.dtype()),
        })
    }

    fn get_audio_features(
        &self,
        input_embeds: &Tensor,
        audio_attention_mask: Option<&Tensor>,
        input_mode: &InputMode,
    ) -> Result<Tensor> {
        // Get audio features from encoder
        let (audio_features, _masks) = self.encoder.forward(input_embeds, audio_attention_mask)?;

        // Apply projection based on mode
        let projection_layers = self.proj.get(input_mode).ok_or_else(|| {
            candle_core::Error::Msg(format!("Projection mode {input_mode:?} not found"))
        })?;

        let mut audio_set_tensor = audio_features;
        for layer in projection_layers {
            audio_set_tensor = layer.forward(&audio_set_tensor)?;
        }

        Ok(audio_set_tensor)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        audio_embed_sizes: Vec<usize>,
        audio_attention_mask: Option<&Tensor>,
        input_mode: &InputMode,
    ) -> Result<Tensor> {
        // Reshape input_ids to 2D
        let input_shape = input_ids.shape();
        let input_ids = if input_shape.rank() > 2 {
            input_ids.reshape((
                input_shape.elem_count() / input_shape.dims()[input_shape.rank() - 1],
                input_shape.dims()[input_shape.rank() - 1],
            ))?
        } else {
            input_ids.clone()
        };

        let positions = input_ids.eq(AUDIO_SPECIAL_TOKEN_ID)?.nonzero()?;

        // Get target device and dtype from projection layers
        let (target_device, target_dtype) = self.target_device_dtype.clone();

        let audio_set_tensor = if positions.dim(0)? > 0 {
            // Convert to target device/dtype if needed
            let input_embeds = if input_embeds.device().same_device(&target_device)
                || input_embeds.dtype() != target_dtype
            {
                input_embeds
                    .to_device(&target_device)?
                    .to_dtype(target_dtype)?
            } else {
                input_embeds.clone()
            };

            self.get_audio_features(&input_embeds, audio_attention_mask, input_mode)?
        } else {
            // Return early if no audio tokens and not training
            return self.wte.forward(&input_ids);
        };

        // Get initial hidden states from word embeddings
        let mut hidden_states = self.wte.forward(&input_ids)?;

        // Verify that audio_embed_sizes sum matches positions count
        let total_audio_tokens = audio_embed_sizes.iter().sum::<usize>();
        if total_audio_tokens != positions.dim(0)? {
            return Err(candle_core::Error::Msg(format!(
                "Audio embed sizes sum ({}) doesn't match positions count ({})",
                total_audio_tokens,
                positions.dim(0)?
            )));
        }

        let mut audio_sets = Vec::new();
        for (i, size) in audio_embed_sizes.into_iter().enumerate() {
            audio_sets.push(audio_set_tensor.i((i, ..size, ..))?);
        }
        let merged_audio_set_tensor = Tensor::cat(&audio_sets, 0)?;

        let original_shape = hidden_states.shape().clone();
        let (hs_b, hs_l, hs_d) = hidden_states.dims3()?;
        let mut hidden_states_flat = hidden_states.reshape(((), hs_d))?;

        // Get the equiv 0th and 1th rows of the positions_tuple
        let positions_transposed = positions.to_dtype(DType::F32)?;
        let positions_transposed_0 = positions_transposed.i((.., 0))?;
        let positions_transposed_1 = positions_transposed.i((.., 1))?;

        let mut linear_index =
            ((positions_transposed_0 * (hs_l * hs_b) as f64)? + positions_transposed_1)?;
        linear_index = linear_index.to_dtype(DType::U32)?;
        linear_index = linear_index.unsqueeze(1)?.repeat((1, hs_d))?;

        let current_vals = hidden_states_flat.gather(&linear_index, 0)?;
        let delta = merged_audio_set_tensor.broadcast_sub(&current_vals)?;

        hidden_states_flat = hidden_states_flat.scatter_add(&linear_index, &delta, 0)?;

        hidden_states = hidden_states_flat.reshape(original_shape)?;

        Ok(hidden_states)
    }
}
