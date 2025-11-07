#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, WithDType};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionMeta},
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
};

fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (x_i.len(),);
        padded_x.push(Tensor::from_vec(x_i, shape, device)?);
    }
    Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
}

pub struct InputMetadata {
    pub input: Tensor,
    pub flash_meta: FlashParams,
}

pub struct InnerInputProcessorOutput {
    pub inputs: InputMetadata,
    pub seq_indices: Vec<usize>,
}

// chunk_offset_toks is the number of tokens by which the tokens are offset,
// chunk_offset_toks / prompt_chunksize = number of batches
#[allow(clippy::too_many_arguments)]
pub fn make_prompt_chunk<T: WithDType + Debug>(
    chunk_offset_toks: usize,
    toks: Vec<&[T]>,
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
) -> Result<InputMetadata> {
    let max_len = toks
        .iter()
        .map(|seq| seq.len())
        .max()
        .expect("No sequences");
    let padding_tok = T::zero();
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let flash_attn = crate::using_flash_attn();
    let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
    let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
    for ctxt in toks {
        let mut ctxt = ctxt.to_vec();
        ctxt.extend(std::iter::repeat_n(
            padding_tok,
            max_len.saturating_sub(ctxt.len()),
        ));

        if flash_attn {
            seqlens_q.push(ctxt.len() as u32);
            seqlens_k.push((ctxt.len() + chunk_offset_toks) as u32);
        }

        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }

    let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
        let max_q = *seqlens_q.iter().max().unwrap();
        let max_k = *seqlens_k.iter().max().unwrap();
        let seqlens_q = Tensor::new(seqlens_q, device)?
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;
        let seqlens_k = Tensor::new(seqlens_k, device)?
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;

        let mut seqlens_q_map = HashMap::new();
        let mut seqlens_k_map = HashMap::new();

        if let Some(mapper) = &mapper {
            let devices = mapper.get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
        } else {
            seqlens_q_map.insert(device.location(), seqlens_q.to_device(device)?);
            seqlens_k_map.insert(device.location(), seqlens_k.to_device(device)?);
        }

        (max_q, max_k, seqlens_q_map, seqlens_k_map)
    } else {
        (0, 0, HashMap::new(), HashMap::new())
    };

    let input = Tensor::cat(&seqs_tensors, 0).unwrap();

    Ok(InputMetadata {
        input,
        flash_meta: FlashParams {
            max_k,
            max_q,
            cumulative_seqlens_k: seqlens_k_map,
            cumulative_seqlens_q: seqlens_q_map,
            causal: has_causal_attention,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
    toks: Vec<&[T]>,
    input_seqs: &[&mut Sequence],
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
) -> Result<InnerInputProcessorOutput> {
    let offset = input_seqs[0].token_offset();
    make_prompt_chunk(offset, toks, device, mapper, has_causal_attention).map(|inputs| {
        InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        }
    })
}

#[derive(Clone)]
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub flash_meta: FlashParams,
}

pub struct EmbeddingInputsProcessor {
    pub has_causal_attention: bool,
}

impl InputsProcessor for EmbeddingInputsProcessor {
    fn process_inputs(
        &self,
        _: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        _is_xlora: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        assert!(is_prompt);

        let metadata = get_prompt_input(
            input_seqs
                .iter()
                .map(|seq| seq.get_toks())
                .collect::<Vec<_>>(),
            input_seqs,
            device,
            mapper,
            self.has_causal_attention,
        )?;
        let InnerInputProcessorOutput {
            inputs:
                InputMetadata {
                    input: input_ids,
                    flash_meta,
                },
            seq_indices,
        } = metadata;
        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }

    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Embedding
    }
}

pub struct EmbeddingProcessor {
    pub has_causal_attention: bool,
}

impl Processor for EmbeddingProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(EmbeddingInputsProcessor {
            has_causal_attention: self.has_causal_attention,
        })
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}
