use std::any::Any;

use candle_core::{Result, Tensor};

pub(crate) mod clip;
pub(crate) mod conformer;
pub(crate) mod idefics2;
pub(crate) use idefics2::idefics2_input_processor;
pub(crate) mod image_processor;
pub(crate) mod llava;
pub(crate) mod mllama;
pub(crate) mod phi3;
pub(crate) use phi3::phi3_inputs_processor;
pub(crate) mod preprocessor_config;
pub(crate) mod processor_config;
pub(crate) mod qwen2_5_vl;
pub(crate) mod qwen2vl;
pub(crate) use llava::llava15;
pub(crate) use llava::llava_inputs_processor;
pub(crate) use llava::llava_next;
pub(crate) use llava::llava_next_inputs_processor;
pub(crate) mod idefics3;
pub(crate) mod minicpmo;
pub(crate) mod phi4;
pub(crate) use phi4::inputs_processor;
pub(crate) mod gemma3;
pub(crate) mod gemma3n;
pub(crate) mod gemma4;
pub(crate) mod llama4;
pub(crate) mod mistral3;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_5_moe;
pub(crate) mod qwen3_vl;
pub(crate) mod qwen3_vl_moe;
pub(crate) mod siglip;
pub(crate) mod voxtral;

use crate::pipeline::{
    text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
    RecurrentBatchKind,
};

pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
}

fn mrope_position_deltas_for_broadcast(
    mrope_position_deltas: &Tensor,
    batch: usize,
) -> Result<Tensor> {
    match mrope_position_deltas.dims() {
        [b] if *b == batch => mrope_position_deltas.reshape((1, batch, 1)),
        [b, 1] if *b == batch => mrope_position_deltas.reshape((1, batch, 1)),
        [1, b, 1] if *b == batch => Ok(mrope_position_deltas.clone()),
        _ => candle_core::bail!(
            "MRoPE position deltas shape {:?} is incompatible with batch {batch}",
            mrope_position_deltas.shape()
        ),
    }
}

pub(crate) fn mrope_position_ids_for_input(
    position_ids: &Tensor,
    mrope_position_deltas: &Tensor,
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let (planes, pos_batch, full_len) = position_ids.dims3()?;
    if pos_batch != batch || seqlen_offsets.len() != batch {
        candle_core::bail!(
            "MRoPE position ids shape {:?} is incompatible with input shape {:?}",
            position_ids.shape(),
            input_ids.shape()
        );
    }

    if seqlen_offsets.iter().all(|offset| {
        offset
            .checked_add(seq_len)
            .is_some_and(|end| end <= full_len)
    }) {
        let mut indices = Vec::with_capacity(planes * batch * seq_len);
        for _ in 0..planes {
            for offset in seqlen_offsets {
                for pos in *offset..*offset + seq_len {
                    indices.push(u32::try_from(pos).map_err(candle_core::Error::wrap)?);
                }
            }
        }
        let indices = Tensor::from_vec(indices, (planes, batch, seq_len), position_ids.device())?;
        return position_ids.gather(&indices, 2);
    }

    let offsets = seqlen_offsets
        .iter()
        .map(|offset| i64::try_from(*offset).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    let offsets = Tensor::from_vec(offsets, (1, batch, 1), input_ids.device())?;
    let seq_len_i64 = i64::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let relative =
        Tensor::arange(0i64, seq_len_i64, input_ids.device())?.reshape((1, 1, seq_len))?;
    let position_ids = offsets.broadcast_add(&relative)?.repeat((planes, 1, 1))?;
    let mrope_position_deltas = mrope_position_deltas_for_broadcast(mrope_position_deltas, batch)?;
    position_ids.broadcast_add(&mrope_position_deltas)
}

pub(crate) fn text_decode_mrope_position_ids_from_context(
    input_ids: &Tensor,
    ctx: &crate::pipeline::ModelForwardContext<'_>,
) -> Result<Option<Tensor>> {
    let (batch, seq_len) = input_ids.dims2()?;
    if seq_len != 1 {
        return Ok(None);
    }
    let Some(rope_positions) = ctx.cache().rope_positions(input_ids.device()) else {
        return Ok(None);
    };
    if rope_positions.dim(0)? != batch {
        candle_core::bail!(
            "rope positions shape {:?} is incompatible with input shape {:?}",
            rope_positions.shape(),
            input_ids.shape()
        );
    }
    Ok(Some(
        rope_positions.reshape((1, batch, 1))?.repeat((3, 1, 1))?,
    ))
}
