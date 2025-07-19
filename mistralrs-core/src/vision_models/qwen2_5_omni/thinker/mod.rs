use std::sync::Arc;

use candle_core::{DType, IndexOp, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use text::Qwen25OThinkerTextModel;

use crate::{
    layers::CausalMasker,
    layers_masker::{masked_fill, PastKvLenCache},
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        NormalLoadingMetadata,
    },
};

use super::config::Qwen25OmniThinkerConfig;

mod text;

pub struct Qwen25OThinker {
    text: Qwen25OThinkerTextModel,
    lm_head: Arc<dyn QuantMethod>,
}

impl Qwen25OThinker {
    pub fn new(
        cfg: &Qwen25OmniThinkerConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let text = Qwen25OThinkerTextModel::new(
            &cfg.text_config,
            vb.pp("model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let lm_head = mistralrs_quant::linear_no_bias(
            cfg.text_config.hidden_size,
            cfg.text_config.vocab_size,
            &cfg.text_config.quantization_config,
            vb.pp("lm_head"),
        )?;

        Ok(Self { text, lm_head })
    }

    #[allow(clippy::too_many_arguments)]
    /// (position_ids, mrope_position_deltas)
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        attention_mask_indices: Option<&Tensor>,
        input_ids_searching: Vec<Vec<u32>>,
        image_nums: Vec<usize>,
        video_nums: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            todo!()
        } else if let Some(attention_mask) = attention_mask {
            let position_ids = (attention_mask.to_dtype(DType::F32)?.cumsum(D::Minus1)? - 1f64)?;
            let position_ids = masked_fill(&position_ids, &attention_mask.eq(0f64)?, 1i64)?;
            let position_ids = position_ids.unsqueeze(0)?.repeat((3, 1, 1))?;

            let max_position_ids = position_ids.max(0)?.max_keepdim(D::Minus1)?;
            let mrope_position_deltas =
                (max_position_ids + 1.)?.broadcast_sub(&attention_mask.sum_keepdim(D::Minus1)?)?;

            Ok((
                position_ids.to_dtype(DType::I64)?,
                mrope_position_deltas.to_dtype(DType::I64)?,
            ))
        } else {
            let position_ids = Tensor::arange(0i64, input_ids.dim(1)? as i64, input_ids.device())?
                .reshape((1, 1, ()))?
                .repeat((3, input_ids.dim(0)?, 1))?;
            let mrope_position_deltas =
                Tensor::zeros((input_ids.dim(0)?, 1), DType::I64, input_ids.device())?;

            Ok((position_ids, mrope_position_deltas))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_values_videos: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        video_grid_thw: Option<Tensor>,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        continuous_vid_pad: Vec<Vec<(usize, usize)>>,
        input_ids_searching: Vec<Vec<u32>>,
        image_nums: Vec<usize>,
        video_nums: Vec<usize>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.cfg.sliding_window,
            self.text.dtype,
            self.text.cfg.num_attn_heads,
        )?;

        let input_embeds = if pixel_values.is_some() || pixel_values_videos.is_some() {
            let mut xs = self.text.embed_tokens(input_ids)?;

            todo!();
        } else {
            self.text.embed_tokens(input_ids)?
        };

        let mut ropeidx_attn_mask_bs = Vec::new();
        let max_seqlens = *seqlens.iter().max().unwrap();
        for len in &seqlens {
            ropeidx_attn_mask_bs.push(Tensor::new(
                [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
        let mut ropeidx_attn_mask_indices_bs = Vec::new();
        for len in seqlens {
            ropeidx_attn_mask_indices_bs.push(Tensor::from_vec(
                (0..len as i64).collect(),
                (len,),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask_indices = Tensor::stack(&ropeidx_attn_mask_indices_bs, 0)?;

        let ropeidx_input_ids = if attention_mask.is_some() {
            input_ids
        } else {
            input_ids_full
        };
        let (position_ids, mut mrope_position_deltas) = self.get_rope_index(
            ropeidx_input_ids,
            image_grid_thw.as_ref(),
            video_grid_thw.as_ref(),
            Some(&ropeidx_attn_mask),
            Some(&ropeidx_attn_mask_indices),
            input_ids_searching,
            image_nums,
            video_nums,
        )?;

        let position_ids = if let Some(attention_mask) = &attention_mask {
            let delta0 = (1. - attention_mask)?.sum(D::Minus1)?.unsqueeze(1)?;
            mrope_position_deltas = mrope_position_deltas.broadcast_sub(&delta0)?;

            position_ids
        } else {
            let mut position_ids = Tensor::new(
                seqlen_offsets.iter().map(|x| *x as i64).collect::<Vec<_>>(),
                input_ids.device(),
            )?
            .reshape((1, (), 1))?
            .repeat((3, 1, 1))?;

            position_ids = position_ids.broadcast_add(&mrope_position_deltas.unsqueeze(0)?)?;

            position_ids
        };

        let out = self.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            &position_ids,
            context_lens,
            flash_params,
        )?;
        Ok(out)
    }
}
