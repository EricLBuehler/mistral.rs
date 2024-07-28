use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use text_models_inputs_processor::PagedAttentionMeta;
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
        paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        token_batchsize: Option<usize>,
    ) -> Box<dyn Iterator<Item = Result<Box<dyn Any>>>>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, iter::repeat, sync::Arc};

    use anyhow::Result;
    use candle_core::{Device, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{
        layers::set_use_matmul_via_f16,
        paged_attention::{BlockEngine, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputsProcessor, InputsProcessorType};

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
        pub block_tables: Option<Tensor>,
        pub context_lens: Option<Tensor>,
        pub slot_mappings: Tensor,
        pub max_context_len: Option<usize>,
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub positions_kernel: Tensor,          // [bs, seq len]
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
    }

    // chunk_offset_toks is the number of tokens by which the tokens are offset,
    // chunk_offset_toks / token_batchsize = number of batches
    fn make_prompt_chunk<T: WithDType>(
        chunk_offset_toks: usize,
        toks: Vec<Vec<T>>,
        input_seqs: &[&Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
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
        let mut slot_mappings = Vec::new();
        dbg!(&chunk_offset_toks);
        for (seq, mut ctxt) in input_seqs.iter().zip(toks) {
            let offset = last_n_context_len.unwrap_or_default();
            seqlen_offsets.push(offset.1 + chunk_offset_toks);

            position_ids.push(ctxt.len() + chunk_offset_toks + 1);
            ctxt.extend(repeat(padding_tok).take(max_len.saturating_sub(ctxt.len())));
            context_lens.push((
                seq.len() - last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                last_n_context_len.map(|(a, _)| a).unwrap_or(1),
            ));

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let table = paged_attn_metadata.block_engine.block_tables.get(seq.id());
                let prompt_len = seq.len();

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
                        0.min(prompt_len - sliding_window)
                    } else {
                        0
                    }
                } else {
                    0
                };

                let mut slot_mapping = Vec::new();
                for i in chunk_offset_toks..prompt_len + chunk_offset_toks {
                    if i < start_idx + chunk_offset_toks {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }

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
                }
                slot_mappings.push(slot_mapping);
            }
        }
        dbg!(&seqlen_offsets);

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
        // Only use matmul via f16 if prompt and seqlen > 512
        if input.dim(1)? > VIA_F16_TOK_THRESHOLD {
            set_use_matmul_via_f16(true);
        } else {
            set_use_matmul_via_f16(false);
        }

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                *position_ids.iter().max().unwrap(),
                _PAD_SLOT_ID,
                device,
            )?;
            Some(PagedAttentionInputMetadata {
                slot_mappings,
                block_tables: None,
                context_lens: None,
                max_context_len: None,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            positions_kernel,
            context_lens,
            position_ids,
            paged_attn_meta,
        })
    }

    fn make_completion_chunk<T: WithDType>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let start_pos = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

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
        let mut tmp = Vec::new();
        for pos in (0..seqs_tensors.len())
            .map(|i| vec![*seqlen_offsets.get(i).unwrap() as i64])
            .collect::<Vec<_>>()
        {
            tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
        }
        let positions_kernel = Tensor::cat(&tmp, 0)?;
        set_use_matmul_via_f16(false);

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            let slot_mappings = _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, device)?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            #[allow(clippy::cast_possible_truncation)]
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
            #[allow(clippy::cast_possible_truncation)]
            let context_lens = Tensor::from_vec(
                paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (paged_attn_context_lens.len(),),
                device,
            )?;

            Some(PagedAttentionInputMetadata {
                slot_mappings,
                block_tables: Some(block_tables),
                context_lens: Some(context_lens),
                max_context_len: Some(*max_context_len),
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input: Tensor::cat(&seqs_tensors, 0).unwrap(),
            positions: seqlen_offsets,
            positions_kernel,
            context_lens,
            position_ids,
            paged_attn_meta,
        })
    }

    pub(crate) fn get_prompt_input<T: WithDType+std::fmt::Debug>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        token_batchsize: Option<usize>,
    ) -> Box<dyn Iterator<Item = Result<InputMetadata>>> {
        dbg!(&toks);
        let token_batchsize = Some(4);
        if let Some(token_batchsize) = token_batchsize {
            let mut seq_chunks = Vec::new();
            let mut n_chunks = Vec::new();
            // Pad each sequence by the padding token to the max len.
            for ctxt in toks.iter() {
                let chunks = ctxt.chunks(token_batchsize).collect::<Vec<_>>();
                n_chunks.push(chunks.len());
                seq_chunks.push(chunks);
            }
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
                    dbg!(&toks);
                    make_prompt_chunk(
                        i * token_batchsize,
                        toks,
                        &seq_ns
                            .into_iter()
                            .map(|i| &*input_seqs[i])
                            .collect::<Vec<_>>(),
                        device,
                        last_n_context_len,
                        paged_attn_metadata.as_mut().map(|x| &mut **x),
                    )
                })
                .collect::<Vec<_>>();
            Box::new(chunks.into_iter())
        } else {
            Box::new(std::iter::once(make_prompt_chunk(
                0,
                toks,
                &input_seqs.into_iter().map(|s| &**s).collect::<Vec<_>>(),
                device,
                last_n_context_len,
                paged_attn_metadata,
            )))
        }
    }

    pub(crate) fn get_completion_input<T: WithDType+std::fmt::Debug>(
        toks: Vec<Vec<T>>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
        token_batchsize: Option<usize>,
    ) -> Box<dyn Iterator<Item = Result<InputMetadata>>> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                paged_attn_metadata,
                token_batchsize,
            );
        }

        Box::new(std::iter::once(make_completion_chunk(
            toks,
            input_seqs,
            device,
            paged_attn_metadata,
        )))
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
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
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
            mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
            token_batchsize: Option<usize>,
        ) -> Box<dyn Iterator<Item = Result<Box<dyn Any>>>> {
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
                        paged_attn_metadata.as_mut(),
                        token_batchsize,
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
                        paged_attn_metadata.as_mut(),
                        token_batchsize,
                    ))
                    .map(|(prompt, completion)| {
                        let InputMetadata {
                            input: input_ids_full,
                            positions: seqlen_offsets_full,
                            positions_kernel: seqlen_offsets_kernel_full,
                            context_lens: _,
                            position_ids,
                            paged_attn_meta: _,
                        } = prompt?;
                        let InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            positions_kernel: seqlen_offsets_kernel,
                            context_lens,
                            position_ids: _,
                            paged_attn_meta,
                        } = completion?;
                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids,
                            input_ids_full: Some(input_ids_full),
                            seqlen_offsets,
                            seqlen_offsets_full: Some(seqlen_offsets_full),
                            seqlen_offsets_kernel,
                            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel_full),
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                        });
                        Ok(inputs)
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
                        paged_attn_metadata.as_mut(),
                        token_batchsize,
                    )
                    .map(|x| {
                        x.map(|metadata| {
                            let InputMetadata {
                                input: input_ids,
                                positions: seqlen_offsets,
                                positions_kernel: seqlen_offsets_kernel,
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            } = metadata;
                            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                                input_ids: input_ids.clone(),
                                input_ids_full: Some(input_ids),
                                seqlen_offsets: seqlen_offsets.clone(),
                                seqlen_offsets_full: Some(seqlen_offsets),
                                seqlen_offsets_kernel: seqlen_offsets_kernel.clone(),
                                seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel),
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            });
                            inputs
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
                        paged_attn_metadata.as_mut(),
                        token_batchsize,
                    )
                    .map(|x| {
                        x.map(|metadata| {
                            let InputMetadata {
                                input: input_ids,
                                positions: seqlen_offsets,
                                positions_kernel: seqlen_offsets_kernel,
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            } = metadata;
                            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                                input_ids,
                                input_ids_full: None,
                                seqlen_offsets,
                                seqlen_offsets_full: None,
                                seqlen_offsets_kernel,
                                seqlen_offsets_kernel_full: None,
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            });
                            inputs
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
                        paged_attn_metadata.as_mut(),
                        token_batchsize,
                    )
                    .map(|x| {
                        x.map(|metadata| {
                            let InputMetadata {
                                input: input_ids,
                                positions: seqlen_offsets,
                                positions_kernel: seqlen_offsets_kernel,
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            } = metadata;
                            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                                input_ids,
                                input_ids_full: None,
                                seqlen_offsets,
                                seqlen_offsets_full: None,
                                seqlen_offsets_kernel,
                                seqlen_offsets_kernel_full: None,
                                context_lens,
                                position_ids,
                                paged_attn_meta,
                            });
                            inputs
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
