use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::sequence::Sequence;

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
}

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
    /// This should also enable matmul via f16 if prompt and the sequence length is greater than 32.
    /// Otherwise, matmul via f16 is disabled.
    ///
    /// This should return a type which can be downcasted to the proper type as used in `forward_inputs`
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Arc<Tokenizer>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
    ) -> Result<Box<dyn Any>>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, iter::repeat, sync::Arc};

    use anyhow::Result;
    use candle_core::{Device, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{layers::set_use_matmul_via_f16, sequence::Sequence};

    use super::{InputsProcessor, InputsProcessorType};

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub positions_kernel: Tensor,          // [bs, seq len]
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
    }

    pub(crate) fn get_prompt_input<T: WithDType>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
    ) -> Result<InputMetadata> {
        let max_len = input_seqs
            .iter()
            .map(|seq| seq.len())
            .max()
            .expect("No sequences");
        let padding_tok = T::zero();
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        for (seq, mut ctxt) in input_seqs.iter().zip(toks) {
            let offset = if let Some((_, offset)) = last_n_context_len {
                offset
            } else {
                0
            };
            seqlen_offsets.push(offset);

            ctxt.extend(repeat(padding_tok).take(max_len.saturating_sub(ctxt.len())));
            context_lens.push((
                seq.len() - last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                last_n_context_len.map(|(a, _)| a).unwrap_or(1),
            ));
            position_ids.push(seq.len());

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
        }

        let mut tmp = Vec::new();
        if last_n_context_len.is_some() {
            for pos in (0..seqs_tensors.len())
                .map(|i| {
                    (*seqlen_offsets.get(i).unwrap() as i64
                        ..*seqlen_offsets.get(i).unwrap() as i64 + max_len as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
            {
                tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
            }
        } else {
            for pos in (0..seqs_tensors.len())
                .map(|_| (0..max_len).map(|x| x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            {
                tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
            }
        }
        let positions_kernel = Tensor::cat(&tmp, 0)?;
        let input = Tensor::cat(&seqs_tensors, 0).unwrap();
        // Only use matmul via f16 if prompt and seqlen > 32
        if input.dim(1)? > 32 {
            set_use_matmul_via_f16(true);
        } else {
            set_use_matmul_via_f16(false);
        }
        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            positions_kernel,
            context_lens,
            position_ids,
        })
    }

    pub(crate) fn get_completion_input<T: WithDType>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
    ) -> Result<InputMetadata> {
        if no_kv_cache {
            return get_prompt_input(toks, input_seqs, device, last_n_context_len);
        }
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let start_pos = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
        }
        let mut tmp = Vec::new();
        for pos in (0..seqs_tensors.len())
            .map(|i| vec![*seqlen_offsets.get(i).unwrap() as i64])
            .collect::<Vec<_>>()
        {
            tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
        }
        let positions_kernel = Tensor::cat(&tmp, 0)?;
        set_use_matmul_via_f16(false);
        Ok(InputMetadata {
            input: Tensor::cat(&seqs_tensors, 0).unwrap(),
            positions: seqlen_offsets,
            positions_kernel,
            context_lens,
            position_ids,
        })
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub seqlen_offsets_kernel: Tensor,
        pub seqlen_offsets_kernel_full: Option<Tensor>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Arc<Tokenizer>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            _: Option<Arc<dyn Any>>,
        ) -> Result<Box<dyn Any>> {
            if is_xlora && !is_prompt {
                let InputMetadata {
                    input: input_ids_full,
                    positions: seqlen_offsets_full,
                    positions_kernel: seqlen_offsets_kernel_full,
                    context_lens: _,
                    position_ids,
                } = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks().to_vec())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                )?;
                let InputMetadata {
                    input: input_ids,
                    positions: seqlen_offsets,
                    positions_kernel: seqlen_offsets_kernel,
                    context_lens,
                    position_ids: _,
                } = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks().to_vec())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                )?;
                Ok(Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: Some(input_ids_full),
                    seqlen_offsets,
                    seqlen_offsets_full: Some(seqlen_offsets_full),
                    seqlen_offsets_kernel,
                    seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel_full),
                    context_lens,
                    position_ids,
                }))
            } else if is_xlora && is_prompt {
                let InputMetadata {
                    input: input_ids,
                    positions: seqlen_offsets,
                    positions_kernel: seqlen_offsets_kernel,
                    context_lens,
                    position_ids,
                } = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks().to_vec())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                )?;
                Ok(Box::new(ModelInputs {
                    input_ids: input_ids.clone(),
                    input_ids_full: Some(input_ids),
                    seqlen_offsets: seqlen_offsets.clone(),
                    seqlen_offsets_full: Some(seqlen_offsets),
                    seqlen_offsets_kernel: seqlen_offsets_kernel.clone(),
                    seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel),
                    context_lens,
                    position_ids,
                }))
            } else if is_prompt {
                let InputMetadata {
                    input: input_ids,
                    positions: seqlen_offsets,
                    positions_kernel: seqlen_offsets_kernel,
                    context_lens,
                    position_ids,
                } = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks().to_vec())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                )?;
                Ok(Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    seqlen_offsets_kernel,
                    seqlen_offsets_kernel_full: None,
                    context_lens,
                    position_ids,
                }))
            } else {
                let InputMetadata {
                    input: input_ids,
                    positions: seqlen_offsets,
                    positions_kernel: seqlen_offsets_kernel,
                    context_lens,
                    position_ids,
                } = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks().to_vec())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                )?;
                Ok(Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    seqlen_offsets_kernel,
                    seqlen_offsets_kernel_full: None,
                    context_lens,
                    position_ids,
                }))
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }
}
