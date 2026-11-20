use super::*;

/// The per-step arguments every backend arm of `Pipeline::submit_step` shares.
pub(super) struct StepCall<'a, 's> {
    pub input_seqs: &'a mut [&'s mut Sequence],
    pub is_prompt: bool,
    pub return_raw_logits: bool,
    pub prefix_cacher: &'a mut PrefixCacheManagerV2,
    pub disable_eos_stop: bool,
    pub rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    pub logger: &'a IntervalLogger,
    pub lookahead: StepLookahead,
}

pub(super) async fn submit_decision_step<P: Pipeline + ?Sized>(
    this: &mut P,
    input_seqs: &mut [&mut Sequence],
) -> Result<StepSubmission, candle_core::Error> {
    let mut duration = Duration::ZERO;
    for seq in input_seqs {
        if seq.responder().is_closed() {
            seq.set_state(crate::sequence::SequenceState::Done(
                crate::sequence::StopReason::Canceled,
            ));
            continue;
        }
        let start = Instant::now();
        let result = this.forward_decision(seq);
        duration += start.elapsed();
        if seq.responder().is_closed() {
            seq.set_state(crate::sequence::SequenceState::Done(
                crate::sequence::StopReason::Canceled,
            ));
            continue;
        }
        let response = match result {
            Ok(result) => {
                seq.set_state(crate::sequence::SequenceState::Done(
                    crate::sequence::StopReason::Length(0),
                ));
                crate::Response::Decision { result }
            }
            Err(error) => {
                seq.set_state(crate::sequence::SequenceState::Error);
                if is_inputs_processor_validation_error(&error) {
                    crate::Response::ValidationError(error.into())
                } else {
                    crate::Response::InternalError(error.into())
                }
            }
        };
        let _ = seq.responder().send(response).await;
    }
    Ok(StepSubmission::ready(duration))
}

pub(super) async fn submit_default_step<P: Pipeline + ?Sized>(
    this: &mut P,
    call: StepCall<'_, '_>,
    pre_op: CacheInstruction,
    post_op: CacheInstruction,
) -> Result<StepSubmission, candle_core::Error> {
    let StepCall {
        input_seqs,
        is_prompt,
        return_raw_logits,
        prefix_cacher,
        disable_eos_stop,
        rng,
        logger,
        lookahead: _,
    } = call;
    if !is_prompt && !return_raw_logits {
        crate::speculative::driver::clear_staged_speculative_tokens(input_seqs);
    }

    let inputs_iter = std::iter::once(this.get_processor().inputs_processor().process_inputs(
        this.tokenizer(),
        input_seqs,
        is_prompt,
        this.get_metadata().is_xlora,
        &this.device(),
        this.get_metadata().no_kv_cache,
        None,
        return_raw_logits,
        this.get_metadata().sliding_window,
        this.get_input_processor_config(),
        None,
        this.device_mapper(),
    ));

    let mut logits = vec![None; input_seqs.len()];
    let len_inputs = 1;
    let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
    let mut embedding_logits = vec![None; input_seqs.len()];

    let mut exec_duration = Duration::ZERO;
    for (i, inputs) in inputs_iter.into_iter().enumerate() {
        let InputProcessorOutput {
            inputs,
            seq_indices,
        } = inputs.map_err(candle_core::Error::msg)?;
        if i == 0 {
            match pre_op {
                CacheInstruction::In => this.clone_in_cache(input_seqs)?,
                CacheInstruction::Nothing => (),
                CacheInstruction::Reset {
                    load_preallocated_cache,
                    reset_non_granular,
                } => this.set_none_cache(
                    input_seqs,
                    reset_non_granular,
                    false,
                    load_preallocated_cache,
                )?,
                _ => unreachable!("Unreachable PRE cache op."),
            }
        }

        let preserve_causal_generation = input_seqs.len() > 1
            && !return_raw_logits
            && this.device().is_cuda()
            && ((this.supports_batched_cuda_sampling()
                && sampling::can_sample_batch_cuda(input_seqs))
                || crate::speculative::verifier::can_batch_device_verify(input_seqs));
        let start = Instant::now();
        let raw_logits = this
            .forward_inputs(inputs, return_raw_logits)?
            .into_cpu_for_batch(input_seqs.len(), preserve_causal_generation)?;
        let end = Instant::now();
        exec_duration += end.duration_since(start);

        for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
            if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                raw_out_logits[seq_idx][i] = Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
            } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                embedding_logits[seq_idx] = Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
            } else {
                logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
            }
        }
    }

    match post_op {
        CacheInstruction::Out => this.clone_out_cache(input_seqs),
        CacheInstruction::Nothing => (),
        CacheInstruction::Reset {
            load_preallocated_cache,
            reset_non_granular,
        } => this.set_none_cache(
            input_seqs,
            reset_non_granular,
            false,
            load_preallocated_cache,
        )?,
        _ => unreachable!("Unreachable POST cache op."),
    }

    let start = Instant::now();
    if send_passthrough_outputs(input_seqs, raw_out_logits, embedding_logits).await? {
        exec_duration += start.elapsed();
        return Ok(StepSubmission::ready(exec_duration));
    }

    let start = Instant::now();
    let logits = logits
        .into_iter()
        .map(|logits| logits.expect("missing forward result"))
        .collect::<Vec<_>>();

    respond_to_forward_results(
        this,
        ResponseCall {
            input_seqs,
            prefix_cacher,
            disable_eos_stop,
            rng,
            logger,
        },
        logits,
        SpeculativeSampling {
            attempt: !is_prompt && !return_raw_logits,
            batched_logits: None,
            metadata: None,
        },
    )
    .await?;
    let end = Instant::now();
    exec_duration += end.duration_since(start);

    Ok(StepSubmission::ready(exec_duration))
}

pub(super) async fn submit_paged_step<P: Pipeline + ?Sized>(
    this: &mut P,
    call: StepCall<'_, '_>,
    mut metadata: PagedAttentionMeta,
) -> Result<StepSubmission, candle_core::Error> {
    let StepCall {
        input_seqs,
        is_prompt,
        return_raw_logits,
        prefix_cacher,
        disable_eos_stop,
        rng,
        logger,
        lookahead,
    } = call;
    let block_size = metadata.block_size;
    let speculative_metadata = metadata.clone();
    let scheduled_prompt_chunks = metadata.scheduled_prompt_chunks.take();
    let scheduler_visible_prompt_step = scheduled_prompt_chunks.is_some();
    let scheduler_visible_prompt_is_final =
        scheduler_visible_prompt_step && metadata.is_final_prompt_chunk;
    let chunk_size = if !scheduler_visible_prompt_step
        && is_prompt
        && !return_raw_logits
        && !this.get_metadata().is_xlora
        && (this.device().is_cuda() || this.device().is_metal())
    {
        metadata.prompt_chunk_size
    } else {
        None
    };
    if is_prompt {
        this.get_processor()
            .inputs_processor()
            .prepare_for_paged_prompt_planning(
                this.tokenizer(),
                input_seqs,
                &this.device(),
                this.get_input_processor_config(),
                Some(&mut metadata),
            )
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        for seq in input_seqs.iter_mut() {
            seq.clip_prefix_cache_len_for_mm_features(metadata.block_size);
        }
    }
    let has_deferred_multimodal_prompt = input_seqs.iter().any(|seq| {
        (seq.has_images() || seq.has_audios() || seq.has_videos()) && seq.mm_features().is_empty()
    });
    let has_suffix_only_prefill = input_seqs
        .iter()
        .any(|seq| seq.has_suffix_only_prefill_toks());
    let hybrid_recurrent = this.cache().is_hybrid();
    let prefix_policy = this.speculative_prefix_checkpoint_policy();
    let keep_complete_packed_candidates = chunk_size.is_some_and(|chunk_size| {
        input_seqs.len() > 1
            && this.supports_packed_prefill()
            && input_seqs
                .iter()
                .all(|seq| seq.prefix_cache_len() == 0 && seq.len() <= chunk_size)
    });
    let chunk_plans = scheduled_prompt_chunks
        .map(|chunks| chunks.into_iter().map(|chunk| vec![chunk]).collect())
        .or_else(|| {
            (!has_deferred_multimodal_prompt
                && !has_suffix_only_prefill
                && !keep_complete_packed_candidates)
                .then(|| {
                    let block_align = hybrid_recurrent.then_some(block_size);
                    chunk_size.map(|chunk_size| {
                        input_seqs
                            .iter()
                            .map(|seq| {
                                build_prompt_chunk_plan(
                                    seq.get_toks().len(),
                                    seq.prefix_cache_len(),
                                    chunk_size,
                                    block_align,
                                    prefix_policy
                                        .replay_for(crate::scheduler::modality_signature(seq)),
                                    seq.mm_features(),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .flatten()
        });
    let should_chunk = scheduler_visible_prompt_step
        || chunk_plans
            .as_ref()
            .is_some_and(|plans| plans.iter().any(|plan| plan.len() > 1));
    let cuda_decode_lookahead = lookahead.is_enabled()
        && !is_prompt
        && !return_raw_logits
        && this.device().is_cuda()
        && this.supports_batched_cuda_sampling()
        && sampling::can_submit_cuda_token_batch_seqs(input_seqs)
        && sampling::can_launch_one_token_lookahead(input_seqs, this.get_metadata().max_model_len);
    #[cfg(feature = "cuda")]
    let mut batched_cuda_decode = None;
    let (mut logits, batched_causal_logits, raw_out_logits, embedding_logits, mut exec_duration) = {
        let inputs_iter = if let (Some(chunk_plans), true) = (chunk_plans, should_chunk) {
            chunked_prompt_inputs(
                this,
                input_seqs,
                ChunkedPromptCtx {
                    chunk_plans,
                    metadata: &metadata,
                    is_prompt,
                    return_raw_logits,
                    scheduler_visible_prompt_step,
                    scheduler_visible_prompt_is_final,
                    hybrid_recurrent,
                    block_size,
                    prefix_policy,
                },
            )
        } else {
            metadata.set_noncausal_mm_context(input_seqs);
            let prompt_chunk = is_prompt.then(|| SpeculativePromptChunk {
                rows: input_seqs
                    .iter()
                    .enumerate()
                    .map(|(seq_idx, seq)| SpeculativePromptRow {
                        seq_idx,
                        range: (seq.prefix_cache_len(), seq.get_toks().len()),
                        tokens: seq.get_toks().to_vec(),
                    })
                    .collect(),
                is_final_prompt_chunk: true,
            });
            vec![PagedStepInput {
                processed: this.get_processor().inputs_processor().process_inputs(
                    this.tokenizer(),
                    input_seqs,
                    is_prompt,
                    this.get_metadata().is_xlora,
                    &this.device(),
                    this.get_metadata().no_kv_cache,
                    None,
                    return_raw_logits,
                    this.get_metadata().sliding_window,
                    this.get_input_processor_config(),
                    Some(metadata),
                    this.device_mapper(),
                ),
                recurrent_boundaries: Vec::new(),
                prompt_chunk,
                computed_updates: Vec::new(),
            }]
        };

        let mut logits = vec![None; input_seqs.len()];
        let mut batched_causal_logits = None;
        let len_inputs = inputs_iter.len();
        let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
        let mut embedding_logits = vec![None; input_seqs.len()];

        let mut exec_duration = Duration::ZERO;
        for (i, step_input) in inputs_iter.into_iter().enumerate() {
            let PagedStepInput {
                processed: inputs,
                recurrent_boundaries,
                prompt_chunk,
                computed_updates,
            } = step_input;
            let InputProcessorOutput {
                inputs,
                seq_indices,
            } = inputs.map_err(candle_core::Error::msg)?;

            let preserve_causal_generation = (input_seqs.len() > 1 || cuda_decode_lookahead)
                && !return_raw_logits
                && this.device().is_cuda()
                && ((this.supports_batched_cuda_sampling()
                    && sampling::can_sample_batch_cuda(input_seqs))
                    || crate::speculative::verifier::can_batch_device_verify(input_seqs));
            if this.cache().is_hybrid() {
                install_recurrent_state_slots(this, input_seqs, &seq_indices)?;
            }
            let start = Instant::now();
            #[cfg(feature = "cuda")]
            let forward = if cuda_decode_lookahead {
                this.forward_step(inputs, return_raw_logits)?
            } else {
                ForwardStepResult::eager(this.forward_inputs(inputs, return_raw_logits)?)
            };
            #[cfg(not(feature = "cuda"))]
            let forward = ForwardStepResult::eager(this.forward_inputs(inputs, return_raw_logits)?);
            #[cfg(feature = "cuda")]
            let mut cuda_decode = forward.cuda_decode;
            let raw_logits = forward
                .output
                .into_cpu_for_batch(input_seqs.len(), preserve_causal_generation)?;
            if let Some(prompt_chunk) = prompt_chunk.as_ref() {
                this.speculative_prompt_chunk(input_seqs, prompt_chunk, &speculative_metadata)?;
            }
            for (seq_idx, end) in computed_updates {
                input_seqs[seq_idx].set_num_computed_tokens(end);
            }
            let end = Instant::now();
            exec_duration += end.duration_since(start);

            for (seq_idx, end) in recurrent_boundaries {
                this.snapshot_paged_recurrent_prefix(
                    &*input_seqs[seq_idx],
                    prefix_cacher,
                    block_size,
                    end,
                )?;
            }

            let keep_batched_causal_logits = !is_prompt
                && preserve_causal_generation
                && len_inputs == 1
                && seq_indices.len() == input_seqs.len()
                && seq_indices.iter().copied().eq(0..input_seqs.len());
            let raw_logits = if keep_batched_causal_logits {
                match raw_logits {
                    ForwardInputsResult::CausalGeneration { logits, .. } => {
                        #[cfg(feature = "cuda")]
                        {
                            batched_cuda_decode = cuda_decode.take();
                        }
                        batched_causal_logits = Some(logits);
                        continue;
                    }
                    raw_logits => raw_logits,
                }
            } else {
                raw_logits
            };

            for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                    raw_out_logits[seq_idx][i] =
                        Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
                } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                    embedding_logits[seq_idx] =
                        Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
                } else {
                    logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                }
            }
            drop(raw_logits);
            #[cfg(feature = "metal")]
            if prompt_chunk
                .as_ref()
                .is_some_and(|chunk| !chunk.is_final_prompt_chunk)
            {
                let start = Instant::now();
                synchronize_metal_devices(this)?;
                exec_duration += start.elapsed();
            }
        }
        (
            logits,
            batched_causal_logits,
            raw_out_logits,
            embedding_logits,
            exec_duration,
        )
    };

    let start = Instant::now();
    if send_passthrough_outputs(input_seqs, raw_out_logits, embedding_logits).await? {
        exec_duration += start.elapsed();
        return Ok(StepSubmission::ready(exec_duration));
    }
    if !should_sample_step(
        is_prompt,
        scheduler_visible_prompt_step,
        scheduler_visible_prompt_is_final,
    ) {
        return Ok(StepSubmission::ready(exec_duration));
    }

    let start = Instant::now();
    let mut speculative_batched_logits = None;
    if let Some(batched_causal_logits) = batched_causal_logits {
        #[cfg(feature = "cuda")]
        let mut batched_causal_logits = batched_causal_logits;
        #[cfg(feature = "cuda")]
        if cuda_decode_lookahead {
            let forward = ForwardStepResult::cuda_decode(
                ForwardInputsResult::CausalGeneration {
                    logits: batched_causal_logits,
                },
                batched_cuda_decode.take(),
            );
            match execution::submit_forward_lookahead(
                this,
                input_seqs,
                forward,
                exec_duration,
                &rng,
            )? {
                Ok(submission) => return Ok(StepSubmission::cuda(submission)),
                Err(forward) => {
                    let ForwardInputsResult::CausalGeneration { logits } = forward.output else {
                        unreachable!("CUDA lookahead changed the forward result type")
                    };
                    batched_causal_logits = logits;
                }
            }
        }
        if this
            .try_sample_causal_gen_batched(
                input_seqs,
                &batched_causal_logits,
                prefix_cacher,
                disable_eos_stop,
                rng.clone(),
            )
            .await?
        {
            if scheduler_visible_prompt_step {
                promote_prompt_and_decode(input_seqs);
            }
            exec_duration += start.elapsed();
            return Ok(StepSubmission::ready(exec_duration));
        }
        for (seq_idx, logits) in logits.iter_mut().enumerate() {
            *logits = Some(ForwardInputsResult::CausalGeneration {
                logits: batched_causal_logits.i(seq_idx)?,
            });
        }
        speculative_batched_logits = Some(batched_causal_logits);
    }
    let logits = logits
        .into_iter()
        .map(|logits| logits.expect("missing forward result"))
        .collect::<Vec<_>>();
    respond_to_forward_results(
        this,
        ResponseCall {
            input_seqs,
            prefix_cacher,
            disable_eos_stop,
            rng,
            logger,
        },
        logits,
        SpeculativeSampling {
            attempt: should_try_speculative_sampling(
                is_prompt,
                scheduler_visible_prompt_step,
                scheduler_visible_prompt_is_final,
                return_raw_logits,
                is_prompt && this.supports_speculative_prompt_bootstrap(),
            ),
            batched_logits: speculative_batched_logits.as_ref(),
            metadata: Some(speculative_metadata),
        },
    )
    .await?;
    if scheduler_visible_prompt_step {
        promote_prompt_and_decode(input_seqs);
    }
    let end = Instant::now();
    exec_duration += end.duration_since(start);

    Ok(StepSubmission::ready(exec_duration))
}

/// One forward of a paged step plus the bookkeeping to apply once it has run.
struct PagedStepInput {
    processed: anyhow::Result<InputProcessorOutput>,
    recurrent_boundaries: Vec<(usize, usize)>,
    prompt_chunk: Option<SpeculativePromptChunk>,
    computed_updates: Vec<(usize, usize)>,
}

struct ChunkedPromptCtx<'a> {
    chunk_plans: Vec<Vec<PromptChunkPlan>>,
    metadata: &'a PagedAttentionMeta,
    is_prompt: bool,
    return_raw_logits: bool,
    scheduler_visible_prompt_step: bool,
    scheduler_visible_prompt_is_final: bool,
    hybrid_recurrent: bool,
    block_size: usize,
    prefix_policy: crate::speculative::SpeculativePrefixCheckpointPolicy,
}

/// Processes each prompt chunk group in plan order, then restores the sequences' full prompts.
fn chunked_prompt_inputs<P: Pipeline + ?Sized>(
    this: &P,
    input_seqs: &mut [&mut Sequence],
    ctx: ChunkedPromptCtx<'_>,
) -> Vec<PagedStepInput> {
    let ChunkedPromptCtx {
        chunk_plans,
        metadata,
        is_prompt,
        return_raw_logits,
        scheduler_visible_prompt_step,
        scheduler_visible_prompt_is_final,
        hybrid_recurrent,
        block_size,
        prefix_policy,
    } = ctx;
    let originals = input_seqs
        .iter()
        .map(|seq| (seq.get_toks().to_vec(), seq.prefix_cache_len()))
        .collect::<Vec<_>>();
    let recurrent_checkpoint_boundaries = input_seqs
        .iter()
        .zip(&originals)
        .map(|(seq, (tokens, prefix_len))| {
            recurrent_checkpoint_boundary(
                tokens.len(),
                *prefix_len,
                hybrid_recurrent.then_some(block_size),
                prefix_policy.replay_for(crate::scheduler::modality_signature(seq)),
                seq.mm_features(),
            )
        })
        .collect::<Vec<_>>();
    let mut plan_indices = vec![0usize; chunk_plans.len()];
    let requires_uniform_prompt_batch = this.requires_uniform_prompt_batch();
    let supports_packed_prefill = this.supports_packed_prefill();
    let mut inputs = Vec::new();
    while plan_indices
        .iter()
        .zip(chunk_plans.iter())
        .any(|(plan_idx, plan)| *plan_idx < plan.len())
    {
        let (active_indices, attention_policy, planned_final_prompt_chunk) =
            next_pipeline_prompt_chunk_group(
                &plan_indices,
                &chunk_plans,
                requires_uniform_prompt_batch,
                supports_packed_prefill,
                hybrid_recurrent,
            )
            .expect("at least one chunk plan is active");
        let is_final_prompt_chunk = prompt_chunk_is_final(
            scheduler_visible_prompt_step,
            scheduler_visible_prompt_is_final,
            planned_final_prompt_chunk,
        );

        let mut recurrent_boundaries = Vec::new();
        let mut prompt_chunk = SpeculativePromptChunk {
            rows: Vec::with_capacity(active_indices.len()),
            is_final_prompt_chunk,
        };
        for &seq_idx in &active_indices {
            let chunk = chunk_plans[seq_idx][plan_indices[seq_idx]];
            let seq = &mut input_seqs[seq_idx];
            seq.set_prefix_cache_len(chunk.start);
            seq.set_prefill_toks(originals[seq_idx].0[..chunk.end].to_vec());
            if recurrent_checkpoint_boundaries[seq_idx] == Some(chunk.end) {
                recurrent_boundaries.push((seq_idx, chunk.end));
            }
            prompt_chunk.rows.push(SpeculativePromptRow {
                seq_idx,
                range: (chunk.start, chunk.end),
                tokens: originals[seq_idx].0.clone(),
            });
        }

        let mut chunk_metadata = metadata.clone();
        chunk_metadata.prompt_chunk_attention_policy = attention_policy;
        chunk_metadata.is_final_prompt_chunk = is_final_prompt_chunk;
        chunk_metadata.needs_logits = is_final_prompt_chunk || return_raw_logits;
        let mut active_input_seqs = input_seqs
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, seq)| active_indices.contains(&idx).then_some(&mut **seq))
            .collect::<Vec<_>>();
        chunk_metadata.set_noncausal_mm_context(active_input_seqs.as_slice());
        let mut processed = this.get_processor().inputs_processor().process_inputs(
            this.tokenizer(),
            active_input_seqs.as_mut_slice(),
            is_prompt,
            this.get_metadata().is_xlora,
            &this.device(),
            this.get_metadata().no_kv_cache,
            None,
            return_raw_logits,
            this.get_metadata().sliding_window,
            this.get_input_processor_config(),
            Some(chunk_metadata),
            this.device_mapper(),
        );
        drop(active_input_seqs);
        if let Ok(processed) = &mut processed {
            for seq_idx in &mut processed.seq_indices {
                *seq_idx = active_indices[*seq_idx];
            }
        }
        let computed_updates = if scheduler_visible_prompt_step {
            active_indices
                .iter()
                .map(|&seq_idx| {
                    let chunk = chunk_plans[seq_idx][plan_indices[seq_idx]];
                    (seq_idx, chunk.end)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        inputs.push(PagedStepInput {
            processed,
            recurrent_boundaries,
            prompt_chunk: Some(prompt_chunk),
            computed_updates,
        });
        for &seq_idx in &active_indices {
            plan_indices[seq_idx] += 1;
        }
    }
    for (seq, (tokens, prefix_len)) in input_seqs.iter_mut().zip(originals.iter()) {
        seq.set_prefix_cache_len(*prefix_len);
        seq.set_prefill_toks(tokens.clone());
    }
    inputs
}

fn install_recurrent_state_slots<P: Pipeline + ?Sized>(
    this: &P,
    input_seqs: &[&mut Sequence],
    seq_indices: &[usize],
) -> candle_core::Result<()> {
    let mut hybrid_cache = this.cache().hybrid();
    let sequence_slots = seq_indices
        .iter()
        .map(|&seq_idx| {
            let seq = input_seqs.get(seq_idx).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "processed sequence index {seq_idx} exceeds batch size {}",
                    input_seqs.len()
                ))
            })?;
            let slot_idx = seq.recurrent_state_idx().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "sequence {} has no recurrent state slot",
                    seq.id()
                ))
            })?;
            Ok((*seq.id(), slot_idx))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    hybrid_cache.install_sequence_state_indices(&sequence_slots)
}

// Partial chunks skip sampling readback, which otherwise releases pooled uploads.
#[cfg(feature = "metal")]
fn synchronize_metal_devices<P: Pipeline + ?Sized>(this: &P) -> candle_core::Result<()> {
    let primary = this.device();
    let mut devices = this
        .device_mapper()
        .map(|mapper| mapper.get_unique_devices())
        .unwrap_or_default();
    if !devices.iter().any(|device| device.same_device(&primary)) {
        devices.push(primary);
    }
    for device in devices.into_iter().filter(Device::is_metal) {
        device.synchronize()?;
    }
    Ok(())
}

fn promote_prompt_and_decode(input_seqs: &mut [&mut Sequence]) {
    for seq in input_seqs.iter_mut() {
        if !seq.is_finished_paged_attn()
            && matches!(seq.sequence_stepping_type(), SeqStepType::PromptAndDecode)
        {
            seq.set_state(SequenceState::RunningCompletion);
        }
    }
}

/// How the causal arm of a step hands its logits to a speculative proposer.
struct SpeculativeSampling<'a> {
    attempt: bool,
    batched_logits: Option<&'a Tensor>,
    metadata: Option<PagedAttentionMeta>,
}

struct ResponseCall<'a, 's> {
    input_seqs: &'a mut [&'s mut Sequence],
    prefix_cacher: &'a mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    logger: &'a IntervalLogger,
}

/// Sends raw logits or embeddings when the step produced them; returns whether it did.
async fn send_passthrough_outputs(
    input_seqs: &mut [&mut Sequence],
    raw_out_logits: Vec<Vec<Option<Tensor>>>,
    embedding_logits: Vec<Option<Tensor>>,
) -> Result<bool, candle_core::Error> {
    if raw_out_logits[0][0].is_some() {
        response::send_raw_responses(
            input_seqs,
            raw_out_logits
                .into_iter()
                .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                .collect(),
        )
        .await?;
        return Ok(true);
    }
    if embedding_logits[0].is_some() {
        response::send_embedding_responses(
            input_seqs,
            embedding_logits
                .into_iter()
                .map(|raw| {
                    raw.unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect(),
        )
        .await?;
        return Ok(true);
    }
    Ok(false)
}

/// Samples or sends one step's forward results; every result has the same variant.
async fn respond_to_forward_results<P: Pipeline + ?Sized>(
    this: &mut P,
    call: ResponseCall<'_, '_>,
    logits: Vec<ForwardInputsResult>,
    speculative: SpeculativeSampling<'_>,
) -> Result<(), candle_core::Error> {
    let ResponseCall {
        input_seqs,
        prefix_cacher,
        disable_eos_stop,
        rng,
        logger,
    } = call;
    match &logits[0] {
        ForwardInputsResult::RawLogits { .. } | ForwardInputsResult::Embeddings { .. } => {
            unreachable!()
        }
        ForwardInputsResult::CausalGeneration { .. } => {
            let logits = logits
                .into_iter()
                .map(|r| {
                    #[allow(irrefutable_let_patterns)]
                    let ForwardInputsResult::CausalGeneration { logits, .. } = r
                    else {
                        unreachable!("All results must have same type")
                    };
                    logits
                })
                .collect::<Vec<_>>();
            if !speculative.attempt
                || !this
                    .try_sample_speculative_causal_gen(
                        input_seqs,
                        &logits,
                        speculative.batched_logits,
                        prefix_cacher,
                        disable_eos_stop,
                        rng.clone(),
                        speculative.metadata,
                        logger,
                    )
                    .await?
            {
                this.sample_causal_gen(input_seqs, logits, prefix_cacher, disable_eos_stop, rng)
                    .await?;
            }
        }
        ForwardInputsResult::Image { .. } => {
            response::send_image_responses(
                input_seqs,
                logits
                    .into_iter()
                    .map(|r| {
                        #[allow(irrefutable_let_patterns)]
                        let ForwardInputsResult::Image { images } = r
                        else {
                            unreachable!("All results must have same type, `Image`")
                        };
                        images
                            .into_iter()
                            .next()
                            .expect("Must have at least 1 element.")
                    })
                    .collect::<Vec<_>>(),
            )
            .await?;
        }
        ForwardInputsResult::Speech { .. } => {
            let rates = logits
                .iter()
                .map(|r| {
                    #[allow(irrefutable_let_patterns)]
                    let ForwardInputsResult::Speech { rates, .. } = r
                    else {
                        unreachable!("All results must have same type, `Speech`")
                    };
                    assert_eq!(rates.len(), 1, "Each sequence must have 1 PCM output.");
                    *rates.first().unwrap()
                })
                .collect::<Vec<_>>();
            let channels = logits
                .iter()
                .map(|r| {
                    #[allow(irrefutable_let_patterns)]
                    let ForwardInputsResult::Speech { channels, .. } = r
                    else {
                        unreachable!("All results must have same type, `Speech`")
                    };
                    assert_eq!(channels.len(), 1, "Each sequence must have 1 PCM output.");
                    *channels.first().unwrap()
                })
                .collect::<Vec<_>>();
            let pcms = logits
                .into_iter()
                .map(|r| {
                    #[allow(irrefutable_let_patterns)]
                    let ForwardInputsResult::Speech { pcms, .. } = r
                    else {
                        unreachable!("All results must have same type, `Speech`")
                    };
                    assert_eq!(pcms.len(), 1, "Each sequence must have 1 PCM output.");
                    pcms.into_iter().next().unwrap()
                })
                .collect::<Vec<_>>();
            response::send_speech_responses(input_seqs, &pcms, &rates, &channels).await?;
        }
        ForwardInputsResult::BlockGeneration { .. } => {
            let mut denoise_times = Vec::with_capacity(logits.len());
            let token_blocks = logits
                .into_iter()
                .map(|r| {
                    #[allow(irrefutable_let_patterns)]
                    let ForwardInputsResult::BlockGeneration {
                        token_blocks,
                        denoise_time,
                    } = r
                    else {
                        unreachable!("All results must have same type, `BlockGeneration`")
                    };
                    denoise_times.push(denoise_time);
                    token_blocks
                        .into_iter()
                        .next()
                        .expect("Must have at least 1 element.")
                })
                .collect::<Vec<_>>();
            this.sample_block_gen(
                input_seqs,
                token_blocks,
                denoise_times,
                prefix_cacher,
                disable_eos_stop,
            )
            .await?;
        }
    }
    Ok(())
}
