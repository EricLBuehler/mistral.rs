#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, num::NonZeroUsize, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use text_models_inputs_processor::PagedAttentionMeta;
use tokenizers::Tokenizer;

use crate::{device_map::DeviceMapper, sequence::Sequence};

pub const DEFAULT_PROMPT_CHUNK_SIZE: usize = 512;

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
}

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
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
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_chunksize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = Result<InputProcessorOutput>>>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{
        any::Any, collections::HashMap, fmt::Debug, iter::repeat, num::NonZeroUsize, sync::Arc,
    };

    use anyhow::Result;
    use candle_core::{DType, Device, DeviceLocation, Tensor, WithDType};
    use mistralrs_quant::set_use_matmul_via_f16;
    use tokenizers::Tokenizer;

    use crate::{
        device_map::DeviceMapper,
        paged_attention::{BlockEngine, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    const VIA_F16_TOK_THRESHOLD: usize = 512;

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

    pub struct PagedAttentionMeta<'a> {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub block_engine: &'a mut BlockEngine,
    }

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    pub struct PagedAttentionInputMetadata {
        pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub slot_mappings: HashMap<DeviceLocation, Tensor>,
        pub max_context_len: Option<usize>,
        pub is_first_prompt_chunk: bool,
    }

    impl PagedAttentionInputMetadata {
        /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
        /// This is used for the case of imatrix generation.
        pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                max_context_len: None,
                slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
                is_first_prompt_chunk: true,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct FlashParams {
        pub max_q: u32,
        pub max_k: u32,
        pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
        pub cumulative_seqlens_k: HashMap<DeviceLocation, Tensor>,
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
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
        toks: Vec<Vec<T>>,
        seq_ids: &[usize],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputMetadata> {
        let max_len = toks
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
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut seqlens_q = vec![0];
        let mut seqlens_k = vec![0];
        for (seq_id, mut ctxt) in seq_ids.iter().zip(toks) {
            let prompt_len = ctxt.len();
            let offset = last_n_context_len.unwrap_or_default();
            seqlen_offsets.push(offset.1 + chunk_offset_toks);

            position_ids.push(ctxt.len() + chunk_offset_toks);
            ctxt.extend(repeat(padding_tok).take(max_len.saturating_sub(ctxt.len())));
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, ctxt.len()));
            } else {
                context_lens.push((
                    ctxt.len() - last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                    last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                ));
            }

            seqlens_q.push(ctxt.len() as u32);
            seqlens_k.push((ctxt.len() + chunk_offset_toks) as u32);

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let table = paged_attn_metadata.block_engine.block_tables.get(seq_id);

                if table.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    if prompt_len > sliding_window {
                        chunk_offset_toks.min(prompt_len - sliding_window)
                    } else {
                        chunk_offset_toks
                    }
                } else {
                    chunk_offset_toks
                };

                let mut slot_mapping = Vec::new();
                let mut ctxt_len = Vec::new();
                for i in chunk_offset_toks..prompt_len + chunk_offset_toks {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }
                    ctxt_len.push(i);

                    let block_number = if i / paged_attn_metadata.block_size >= table.len() {
                        panic!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            paged_attn_metadata.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / paged_attn_metadata.block_size).unwrap()
                    };
                    let block_offset = i % paged_attn_metadata.block_size;
                    let slot = block_number * paged_attn_metadata.block_size + block_offset;
                    slot_mapping.push(slot.try_into().unwrap());
                    block_tables.push(table.clone());
                }
                slot_mappings.push(slot_mapping);
                paged_attn_context_lens.push(ctxt_len);
            }
        }

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

        let devices = mapper.unwrap().get_unique_devices();
        for device in devices {
            seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
            seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
        }

        let input = Tensor::cat(&seqs_tensors, 0).unwrap();
        // Only use matmul via f16 if prompt and seqlen > 512
        if input.dim(1)? > VIA_F16_TOK_THRESHOLD {
            set_use_matmul_via_f16(true);
        } else {
            set_use_matmul_via_f16(false);
        }

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            let max_slot_mapping_len = slot_mappings.iter().map(|x| x.len()).max().unwrap();
            let slot_mappings =
                _make_tensor_with_pad(slot_mappings, max_slot_mapping_len, _PAD_SLOT_ID, device)?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                device,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap();

            let context_lens = _make_tensor_with_pad(
                paged_attn_context_lens
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_context_len,
                0,
                device,
            )?
            .reshape(((),))?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(max_context_len),
                is_first_prompt_chunk: chunk_offset_toks == 0,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
            },
        })
    }

    fn make_completion_chunk<T: WithDType>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut seqlens_q = vec![0];
        let mut seqlens_k = vec![0];
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let start_pos = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

            seqlens_q.push(ctxt.len() as u32);
            seqlens_k.push((ctxt.len() + start_pos) as u32);

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let table = paged_attn_metadata
                    .block_engine
                    .block_tables
                    .get(seq.id())
                    .unwrap();

                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = if start_pos / paged_attn_metadata.block_size >= table.len() {
                    panic!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", start_pos, paged_attn_metadata.block_size, table.len());
                } else {
                    table
                        .get(start_pos / paged_attn_metadata.block_size)
                        .unwrap()
                };
                let block_offset = start_pos % paged_attn_metadata.block_size;
                let slot = block_number * paged_attn_metadata.block_size + block_offset;
                let slot = slot.try_into().unwrap();
                slot_mappings.push(vec![slot]);

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let sliding_window_blocks = sliding_window / paged_attn_metadata.block_size;
                    let slide_idx = if table.len() > sliding_window_blocks {
                        table.len() - sliding_window_blocks
                    } else {
                        0
                    };
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }

                let paged_attn_context_len =
                    if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                        seq.len().min(sliding_window)
                    } else {
                        seq.len()
                    };
                paged_attn_context_lens.push(paged_attn_context_len);
            }
        }

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

        let devices = mapper.unwrap().get_unique_devices();
        for device in devices {
            seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
            seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
        }

        set_use_matmul_via_f16(false);

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            let slot_mappings = _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, device)?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();

            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                device,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens.iter().max().unwrap();

            let context_lens = Tensor::from_vec(
                paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (paged_attn_context_lens.len(),),
                device,
            )?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(*max_context_len),
                is_first_prompt_chunk: false,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input: Tensor::cat(&seqs_tensors, 0).unwrap(),
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        prompt_chunksize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = Result<InnerInputProcessorOutput>>> {
        if let (Some(prompt_chunksize), true) = (prompt_chunksize, paged_attn_metadata.is_none()) {
            let mut seq_chunks = Vec::new();
            let mut n_chunks = Vec::new();
            let prompt_chunksize: usize = prompt_chunksize.into();

            // This comes from prefix caching
            // The invariant where all token offsets are the same is handled by the scheduler
            let offset = input_seqs[0].token_offset();

            // Pad each sequence by the padding token to the max len.
            for ctxt in toks.iter() {
                let chunks = ctxt.chunks(prompt_chunksize).collect::<Vec<_>>();
                n_chunks.push(chunks.len());
                seq_chunks.push(chunks);
            }
            // Basically convert the sequences and tok chunks into chunks of seqs and the corresp toks
            let mut chunks_transposed: Vec<Vec<(Vec<T>, usize)>> = Vec::new();
            for (seq_n, seq) in seq_chunks.into_iter().enumerate() {
                for (i, chunk) in seq.into_iter().enumerate() {
                    match chunks_transposed.get_mut(i) {
                        Some(part) => part.push((chunk.to_vec(), seq_n)),
                        None => chunks_transposed.push(vec![(chunk.to_vec(), seq_n)]),
                    }
                }
            }
            let chunks = chunks_transposed
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let (toks, seq_ns): (Vec<Vec<T>>, Vec<usize>) = chunk.into_iter().unzip();
                    make_prompt_chunk(
                        i * prompt_chunksize + offset,
                        toks,
                        &seq_ns
                            .iter()
                            .map(|i| *input_seqs[*i].id())
                            .collect::<Vec<_>>(),
                        device,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_deref_mut(),
                        mapper,
                    )
                    .map(|inputs| InnerInputProcessorOutput {
                        inputs,
                        seq_indices: seq_ns,
                    })
                })
                .collect::<Vec<_>>();
            Box::new(chunks.into_iter())
        } else {
            let offset = input_seqs[0].token_offset();
            if offset != 0 && paged_attn_metadata.is_some() {
                return Box::new(std::iter::once(Err(anyhow::Error::msg(
                    "PagedAttention does not yet support sequences with an offset != 0.",
                ))));
            }
            Box::new(std::iter::once(
                make_prompt_chunk(
                    offset,
                    toks,
                    &input_seqs.iter().map(|s| *s.id()).collect::<Vec<_>>(),
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata,
                    mapper,
                )
                .map(|inputs| InnerInputProcessorOutput {
                    inputs,
                    seq_indices: (0..input_seqs.len()).collect(),
                }),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        prompt_chunksize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = Result<InnerInputProcessorOutput>>> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                prompt_chunksize,
                mapper,
            );
        }

        Box::new(std::iter::once(
            make_completion_chunk(toks, input_seqs, device, paged_attn_metadata, mapper).map(
                |inputs| InnerInputProcessorOutput {
                    inputs,
                    seq_indices: (0..input_seqs.len()).collect(),
                },
            ),
        ))
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
        pub flash_meta: FlashParams,
        pub flash_meta_full: Option<FlashParams>,
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Option<Arc<Tokenizer>>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            return_raw_logits: bool,
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
            prompt_chunksize: Option<NonZeroUsize>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Box<dyn Iterator<Item = Result<InputProcessorOutput>>> {
            if is_xlora && !is_prompt {
                Box::new(
                    get_prompt_input(
                        input_seqs
                            .iter()
                            .map(|seq| seq.get_toks().to_vec())
                            .collect::<Vec<_>>(),
                        input_seqs,
                        device,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_mut(),
                        prompt_chunksize,
                        mapper,
                    )
                    .zip(get_completion_input(
                        input_seqs
                            .iter()
                            .map(|seq| seq.get_toks().to_vec())
                            .collect::<Vec<_>>(),
                        input_seqs,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_mut(),
                        prompt_chunksize,
                        mapper,
                    ))
                    .map(|(prompt, completion)| {
                        let InnerInputProcessorOutput {
                            inputs:
                                InputMetadata {
                                    input: input_ids_full,
                                    positions: seqlen_offsets_full,
                                    context_lens: _,
                                    position_ids,
                                    paged_attn_meta: _,
                                    flash_meta: flash_meta_full,
                                },
                            seq_indices,
                        } = prompt?;
                        let InnerInputProcessorOutput {
                            inputs:
                                InputMetadata {
                                    input: input_ids,
                                    positions: seqlen_offsets,
                                    context_lens,
                                    position_ids: _,
                                    paged_attn_meta,
                                    flash_meta,
                                },
                            seq_indices: _,
                        } = completion?;
                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids,
                            input_ids_full: Some(input_ids_full),
                            seqlen_offsets,
                            seqlen_offsets_full: Some(seqlen_offsets_full),
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                            flash_meta_full: Some(flash_meta_full),
                        });
                        Ok(InputProcessorOutput {
                            inputs,
                            seq_indices,
                        })
                    }),
                )
            } else if is_xlora && is_prompt {
                Box::new(
                    get_prompt_input(
                        input_seqs
                            .iter()
                            .map(|seq| seq.get_toks().to_vec())
                            .collect::<Vec<_>>(),
                        input_seqs,
                        device,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_mut(),
                        prompt_chunksize,
                        mapper,
                    )
                    .map(|metadata| {
                        let InnerInputProcessorOutput {
                            inputs:
                                InputMetadata {
                                    input: input_ids,
                                    positions: seqlen_offsets,
                                    context_lens,
                                    position_ids,
                                    paged_attn_meta,
                                    flash_meta,
                                },
                            seq_indices,
                        } = metadata?;
                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids: input_ids.clone(),
                            input_ids_full: Some(input_ids),
                            seqlen_offsets: seqlen_offsets.clone(),
                            seqlen_offsets_full: Some(seqlen_offsets),
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta: flash_meta.clone(),
                            flash_meta_full: Some(flash_meta),
                        });
                        Ok(InputProcessorOutput {
                            inputs,
                            seq_indices,
                        })
                    }),
                )
            } else if is_prompt {
                Box::new(
                    get_prompt_input(
                        input_seqs
                            .iter()
                            .map(|seq| seq.get_toks().to_vec())
                            .collect::<Vec<_>>(),
                        input_seqs,
                        device,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_mut(),
                        prompt_chunksize,
                        mapper,
                    )
                    .map(|metadata| {
                        let InnerInputProcessorOutput {
                            inputs:
                                InputMetadata {
                                    input: input_ids,
                                    positions: seqlen_offsets,
                                    context_lens,
                                    position_ids,
                                    paged_attn_meta,
                                    flash_meta,
                                },
                            seq_indices,
                        } = metadata?;
                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids,
                            input_ids_full: None,
                            seqlen_offsets,
                            seqlen_offsets_full: None,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                            flash_meta_full: None,
                        });
                        Ok(InputProcessorOutput {
                            inputs,
                            seq_indices,
                        })
                    }),
                )
            } else {
                Box::new(
                    get_completion_input(
                        input_seqs
                            .iter()
                            .map(|seq| seq.get_toks().to_vec())
                            .collect::<Vec<_>>(),
                        input_seqs,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata.as_mut(),
                        prompt_chunksize,
                        mapper,
                    )
                    .map(|metadata| {
                        let InnerInputProcessorOutput {
                            inputs:
                                InputMetadata {
                                    input: input_ids,
                                    positions: seqlen_offsets,
                                    context_lens,
                                    position_ids,
                                    paged_attn_meta,
                                    flash_meta,
                                },
                            seq_indices,
                        } = metadata?;
                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids,
                            input_ids_full: None,
                            seqlen_offsets,
                            seqlen_offsets_full: None,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                            flash_meta_full: None,
                        });
                        Ok(InputProcessorOutput {
                            inputs,
                            seq_indices,
                        })
                    }),
                )
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }
}
