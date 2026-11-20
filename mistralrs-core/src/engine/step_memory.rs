use std::sync::Arc;

use candle_core::{DType, Device};

use super::{memory, Engine, PromptMemoryRejection, RunLoopState, StepMemoryGrant};
use crate::{
    get_mut_arcmutex,
    paged_attention::{
        block_hash::MultimodalAttentionPolicy,
        plan::{self, PromptPrefillWorkspace},
        KVCacheManager, ModelConfigLike, PagedAttentionSchedulerOutput,
    },
    pipeline::text_models_inputs_processor::{uses_packed_prefill, AttentionMemoryBudget},
    MemoryGoal,
};

/// What the loaded pipeline fixes about a step's memory use, read once per admission.
struct StepMemoryFacts {
    model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>>,
    activation_dtype: DType,
    cache_dtype: DType,
    device_is_cuda: bool,
    has_metal_device: bool,
    has_sliding_window: bool,
    fa3_num_sm_by_layer: Vec<Option<usize>>,
    execution_devices: Vec<Device>,
    primary_device: Device,
    memory_plan: Option<crate::memory_plan::RuntimeMemoryPlan>,
    hnd_by_layer: Vec<bool>,
    supports_packed_prefill: bool,
}

impl StepMemoryFacts {
    fn model(&self) -> Option<&dyn ModelConfigLike> {
        self.model_metadata
            .as_deref()
            .map(|model| model as &dyn ModelConfigLike)
    }

    fn memory_goal(&self) -> MemoryGoal {
        self.memory_plan
            .map_or(MemoryGoal::Auto, |plan| plan.memory_goal)
    }

    fn has_unstructured_metal_mask(&self) -> bool {
        self.memory_plan
            .is_some_and(|plan| plan.custom_prefill_mask && !plan.metal_prefill_mask_is_structured)
    }
}

/// The token geometry of the batch as currently scheduled.
struct StepShape {
    query_lens: Vec<usize>,
    full_context_lens: Vec<usize>,
    max_pages_per_sequence: usize,
    has_noncausal_mm_context: bool,
    requires_prefix_attention: bool,
}

impl Engine {
    fn step_memory_facts(&self) -> StepMemoryFacts {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        let metadata = pipeline.get_metadata();
        let execution_devices = pipeline.execution_devices();
        let cache_engine = metadata.cache_engine.as_ref();
        StepMemoryFacts {
            model_metadata: metadata.model_metadata.clone(),
            activation_dtype: metadata.activation_dtype,
            cache_dtype: metadata
                .cache_config
                .as_ref()
                .map(|config| config.cache_type.to_dtype(metadata.activation_dtype))
                .unwrap_or(metadata.activation_dtype),
            device_is_cuda: execution_devices.iter().all(Device::is_cuda),
            has_metal_device: execution_devices.iter().any(Device::is_metal),
            has_sliding_window: metadata.sliding_window.is_some(),
            fa3_num_sm_by_layer: cache_engine
                .map(|engine| engine.fa3_prefill_num_sm_by_layer().to_vec())
                .unwrap_or_default(),
            execution_devices,
            primary_device: pipeline.device(),
            memory_plan: metadata.memory_plan,
            hnd_by_layer: cache_engine
                .map(|engine| {
                    engine
                        .get_kv_cache()
                        .iter()
                        .map(|(k, _)| k.rank() == 4)
                        .collect()
                })
                .unwrap_or_default(),
            supports_packed_prefill: pipeline.supports_packed_prefill(),
        }
    }

    fn step_shape(
        facts: &StepMemoryFacts,
        output: &PagedAttentionSchedulerOutput,
        is_prompt: bool,
        kv_cache_manager: &Arc<tokio::sync::Mutex<KVCacheManager>>,
    ) -> StepShape {
        let query_lens = output
            .scheduled
            .iter()
            .enumerate()
            .map(|(seq_idx, seq)| {
                let seq = get_mut_arcmutex!(seq);
                if !is_prompt {
                    return 1 + seq.active_staged_speculative_len();
                }
                output
                    .scheduled_prompt_chunks
                    .as_ref()
                    .and_then(|chunks| chunks.get(seq_idx))
                    .map(|chunk| chunk.end.saturating_sub(chunk.start))
                    .unwrap_or_else(|| seq.num_uncomputed_tokens())
            })
            .collect::<Vec<_>>();
        let full_context_lens = output
            .scheduled
            .iter()
            .zip(&query_lens)
            .map(|(seq, query_len)| {
                let seq = get_mut_arcmutex!(seq);
                if is_prompt {
                    seq.num_computed_tokens().saturating_add(*query_len)
                } else {
                    seq.get_toks()
                        .len()
                        .saturating_add(query_len.saturating_sub(1))
                }
            })
            .collect::<Vec<_>>();
        let max_pages_per_sequence = {
            let sequence_ids = output
                .scheduled
                .iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>();
            let manager = get_mut_arcmutex!(kv_cache_manager);
            sequence_ids
                .iter()
                .map(|seq_id| manager.num_blocks_for_request(*seq_id))
                .max()
                .unwrap_or_default()
        };
        let has_noncausal_mm_context = output
            .scheduled_prompt_chunks
            .as_ref()
            .and_then(|chunks| chunks.first())
            .is_some_and(|chunk| chunk.attention_policy == MultimodalAttentionPolicy::NonCausal)
            || output.scheduled.iter().any(|seq| {
                get_mut_arcmutex!(seq)
                    .mm_features()
                    .iter()
                    .any(|feature| feature.attention_policy == MultimodalAttentionPolicy::NonCausal)
            });
        let has_donor_cache_layers = facts
            .model_metadata
            .as_deref()
            .is_some_and(plan::model_has_donor_paged_cache_layers);
        let requires_prefix_attention = has_noncausal_mm_context
            || has_donor_cache_layers
            || query_lens
                .iter()
                .zip(&full_context_lens)
                .any(|(query, full)| full > query);
        StepShape {
            query_lens,
            full_context_lens,
            max_pages_per_sequence,
            has_noncausal_mm_context,
            requires_prefix_attention,
        }
    }

    fn step_attention_workspace(
        facts: &StepMemoryFacts,
        shape: &StepShape,
        is_prompt: bool,
        block_size: usize,
    ) -> candle_core::Result<PromptPrefillWorkspace> {
        if is_prompt {
            return plan::prompt_prefill_workspace(
                facts.model(),
                plan::PromptPrefillWorkspaceInput {
                    activation_dtype: facts.activation_dtype,
                    cache_dtype: facts.cache_dtype,
                    device_is_cuda: facts.device_is_cuda,
                    device_is_metal: facts.has_metal_device,
                    block_size,
                    query_lens: &shape.query_lens,
                    full_context_lens: &shape.full_context_lens,
                    max_pages_per_sequence: shape.max_pages_per_sequence,
                    requires_prefix_attention: shape.requires_prefix_attention,
                    is_causal: !shape.has_noncausal_mm_context,
                    causality_known: true,
                    has_custom_mask: if facts.has_metal_device {
                        facts.has_unstructured_metal_mask()
                    } else {
                        shape.has_noncausal_mm_context
                            || facts
                                .memory_plan
                                .is_some_and(|plan| plan.custom_prefill_mask)
                    },
                    has_noncausal_mm_context: shape.has_noncausal_mm_context,
                    has_sliding_window: facts.has_sliding_window,
                    fa3_num_sm_by_layer: &facts.fa3_num_sm_by_layer,
                    hnd_by_layer: &facts.hnd_by_layer,
                },
            );
        }
        #[cfg(feature = "metal")]
        let bytes = if facts.has_metal_device {
            plan::metal_decode_workspace(
                facts.model(),
                plan::MetalDecodeWorkspaceInput {
                    activation_dtype: facts.activation_dtype,
                    cache_dtype: facts.cache_dtype,
                    block_size,
                    query_lens: &shape.query_lens,
                    context_lens: &shape.full_context_lens,
                    hnd_by_layer: &facts.hnd_by_layer,
                    capacity_bound: false,
                    has_multimodal_ranges: shape.has_noncausal_mm_context,
                    has_custom_mask: facts.has_unstructured_metal_mask(),
                },
            )?
        } else {
            0
        };
        #[cfg(not(feature = "metal"))]
        let bytes = 0;
        Ok(PromptPrefillWorkspace {
            bytes,
            gather_workspace_bytes: 0,
            attention_is_complete: facts.has_metal_device,
        })
    }

    /// Returns the whole forward's transient bytes and the attention share of them.
    fn step_forward_reserve(
        &self,
        facts: &StepMemoryFacts,
        shape: &StepShape,
        output: &PagedAttentionSchedulerOutput,
        is_prompt: bool,
        workspace: PromptPrefillWorkspace,
    ) -> candle_core::Result<(memory::ForwardWorkspaceBytes, usize)> {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        let mut has_encoder_input = false;
        let mut encoder_activation_bytes = Some(0usize);
        let mut return_raw_logits = false;
        let mut cold = true;
        let mut final_chunk = true;
        for (index, seq) in output.scheduled.iter().enumerate() {
            let seq = get_mut_arcmutex!(seq);
            let seq_has_encoder_input = is_prompt
                && (seq.has_images()
                    || seq.has_audios()
                    || seq.has_videos()
                    || !seq.mm_features().is_empty());
            has_encoder_input |= seq_has_encoder_input;
            if seq_has_encoder_input {
                encoder_activation_bytes = match (
                    encoder_activation_bytes,
                    pipeline.encoder_activation_memory_bytes(&seq)?,
                ) {
                    (Some(total), Some(bytes)) => {
                        Some(total.checked_add(bytes).ok_or_else(|| {
                            candle_core::Error::msg("encoder workspace size overflow")
                        })?)
                    }
                    _ => None,
                };
            }
            return_raw_logits |= seq.return_raw_logits;
            cold &= seq.num_computed_tokens() == 0 && seq.token_offset() == 0;
            if let Some(chunk) = output
                .scheduled_prompt_chunks
                .as_ref()
                .and_then(|chunks| chunks.get(index))
            {
                cold &= chunk.start == 0;
                final_chunk &= chunk.end == seq.get_toks().len();
            }
        }
        let packed = uses_packed_prefill(
            &shape.query_lens,
            is_prompt
                && facts.supports_packed_prefill
                && facts.device_is_cuda
                && crate::using_flash_attn()
                && !return_raw_logits
                && !has_encoder_input,
            cold,
            final_chunk,
            !shape.has_noncausal_mm_context,
        );
        let query_tokens = memory::forward_query_tokens(&shape.query_lens, packed)?;
        #[cfg(feature = "metal")]
        let attention_bytes = if is_prompt && facts.has_metal_device {
            workspace
                .bytes
                .checked_add(plan::metal_cache_write_workspace(
                    facts.model(),
                    query_tokens,
                    facts.activation_dtype,
                )?)
                .ok_or_else(|| candle_core::Error::msg("prompt attention workspace overflow"))?
        } else {
            workspace.bytes
        };
        #[cfg(not(feature = "metal"))]
        let attention_bytes = workspace.bytes;
        let extra_logit_tokens = memory::extra_logit_tokens(
            query_tokens,
            shape.query_lens.len(),
            is_prompt,
            return_raw_logits,
        );
        let bytes = memory::forward_workspace_bytes(
            facts.memory_plan,
            memory::ForwardMemoryShape {
                query_tokens,
                sequences: shape.query_lens.len(),
                has_encoder_input,
                encoder_activation_bytes,
                extra_logit_tokens,
            },
            attention_bytes,
            workspace.attention_is_complete,
        )?;
        Ok((bytes, attention_bytes))
    }

    /// Sizes the step's transient workspace and trims or rejects the batch until it fits.
    pub(super) fn admit_paged_step_memory(
        &self,
        state: &mut RunLoopState,
        output: &mut PagedAttentionSchedulerOutput,
        is_prompt: bool,
        block_size: usize,
        kv_cache_manager: &Arc<tokio::sync::Mutex<KVCacheManager>>,
    ) -> Result<StepMemoryGrant, PromptMemoryRejection> {
        let facts = self.step_memory_facts();
        if !is_prompt && !facts.has_metal_device {
            return Ok(StepMemoryGrant::default());
        }
        #[cfg(not(feature = "cuda"))]
        let _ = &state;
        loop {
            let shape = Self::step_shape(&facts, output, is_prompt, kv_cache_manager);
            let workspace = Self::step_attention_workspace(&facts, &shape, is_prompt, block_size)
                .map_err(|err| {
                PromptMemoryRejection::Internal(format!(
                    "Memory preflight could not establish a safe attention plan: {err}"
                ))
            })?;
            let (workspace_bytes, attention_bytes) = self
                .step_forward_reserve(&facts, &shape, output, is_prompt, workspace)
                .map_err(|err| {
                    PromptMemoryRejection::Internal(format!(
                        "Memory preflight could not establish a forward reserve: {err}"
                    ))
                })?;
            let enforce = || {
                memory::enforce_memory_goal(
                    &facts.execution_devices,
                    &facts.primary_device,
                    facts.memory_goal(),
                    workspace_bytes,
                    facts.activation_dtype,
                )
            };
            #[cfg(feature = "cuda")]
            let mut cuda_status = if is_prompt {
                debug_assert!(state.cuda_decode_lease.is_none());
                self.maintain_cuda_prompt_memory(&mut state.cuda_memory_pool, workspace_bytes)
            } else {
                memory::PromptMemoryStatus::default()
            };
            #[allow(unused_mut)]
            let (mut memory_status, mut metal_allocation_limits) = enforce();
            #[cfg(feature = "cuda")]
            {
                while is_prompt
                    && memory_status.transient_pressure
                    && !memory_status.maintenance_failed
                    && !memory_status.insufficient_total_capacity
                {
                    let reclaimed = get_mut_arcmutex!(self.pipeline)
                        .reclaim_cuda_graph_memory(super::cuda_memory::GRAPH_RECLAIM_BATCH_SIZE);
                    if reclaimed == 0 {
                        break;
                    }
                    (memory_status, metal_allocation_limits) = enforce();
                    cuda_status = self
                        .maintain_cuda_prompt_memory(&mut state.cuda_memory_pool, workspace_bytes);
                }
                memory_status.transient_pressure |= cuda_status.transient_pressure;
                memory_status.insufficient_total_capacity |=
                    cuda_status.insufficient_total_capacity;
                memory_status.maintenance_failed |= cuda_status.maintenance_failed;
            }
            let previous = output.scheduled.len();
            match memory::prompt_batch_memory_action(
                if is_prompt { previous } else { 1 },
                memory_status.transient_pressure,
            ) {
                memory::PromptBatchMemoryAction::Proceed => {
                    return Ok(StepMemoryGrant {
                        prefix_gather_workspace_limit: is_prompt
                            .then_some(workspace.gather_workspace_bytes),
                        attention_memory: Some(AttentionMemoryBudget {
                            workspace_bytes: attention_bytes,
                            metal_allocation_limits: (!metal_allocation_limits.is_empty())
                                .then(|| Arc::new(metal_allocation_limits)),
                        }),
                    });
                }
                memory::PromptBatchMemoryAction::Retain(retained) => {
                    let first_omitted_id = output
                        .retain_prompt_prefix(retained)
                        .expect("reduced prompt batch must omit a tail");
                    get_mut_arcmutex!(self.scheduler).defer_prompt_tail(first_omitted_id);
                    memory::record_prompt_batch_reduction(previous, retained);
                }
                memory::PromptBatchMemoryAction::Reject => {
                    memory::log_forward_memory_pressure(
                        &facts.execution_devices,
                        &facts.primary_device,
                        facts.memory_goal(),
                        workspace_bytes,
                        facts.activation_dtype,
                    );
                    #[cfg(feature = "cuda")]
                    if is_prompt && !memory_status.insufficient_total_capacity {
                        state.cuda_prompt_preemption_workspace = Some(
                            state
                                .cuda_prompt_preemption_workspace
                                .map_or(workspace_bytes, |current| {
                                    current.component_max(workspace_bytes)
                                }),
                        );
                    }
                    return Err(if memory_status.insufficient_total_capacity {
                        PromptMemoryRejection::InvalidRequest(format!(
                            "Forward pass requires {workspace_bytes} of transient workspace and cannot fit after device-memory reclamation"
                        ))
                    } else if memory_status.maintenance_failed {
                        PromptMemoryRejection::Internal(
                            "Memory preflight could not verify allocator capacity".to_string(),
                        )
                    } else {
                        PromptMemoryRejection::Unavailable(format!(
                            "Memory pressure prevented forward admission requiring {workspace_bytes} of transient workspace"
                        ))
                    });
                }
            }
        }
    }
}
