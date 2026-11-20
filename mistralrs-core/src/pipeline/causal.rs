use super::isq::IsqModel;
use super::processing::BasicProcessor;
use super::{
    AnyMoePipelineMixin, CacheManager, CacheManagerMixin, DecodeGraphPrecaptureCtx, EitherCache,
    ForwardInputsResult, ForwardStepResult, GeneralMetadata, IsqPipelineMixin, MetadataMixin,
    ModelCategory, PreProcessingMixin, Processor, TokenSource,
};
use crate::amoe::AnyMoeBaseModelMixin;
#[cfg(feature = "cuda")]
use crate::cuda::gdn::GDN_PAD_SLOT;
use crate::device_map::DeviceMapper;
#[cfg(feature = "cuda")]
use crate::kv_cache::RecurrentCheckpointStateSnapshot;
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};
use crate::paged_attention::ModelConfigMetadata;
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    capture_cuda_decode_graph, cuda_decode_graph_batch_kind_supported,
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, cuda_graph_batch_bucket,
    cuda_graph_precapture_batches, cuda_graph_precapture_max_batch,
    cuda_graph_startup_capture_allowed, finish_cuda_capture_memory, hybrid_graph_slots,
    install_hybrid_graph_state_indices, prepare_cuda_capture_memory, record_cuda_graph_dispatch,
    target_cuda_graph_cache_capacity, CudaDecodeGraphCaptureCtx, CudaDecodeGraphKey,
    CudaDecodeGraphLaunch, CudaDecodeGraphReplay, CudaDecodeGraphReplayInput, CudaDecodeGraphState,
    CudaGraphCaptureMemory, CudaGraphComponent, CudaGraphDecodeStep, CudaGraphDecodeStepInputs,
    CudaGraphDispatchMode, CudaGraphDispatchReason, CudaGraphEvent, CudaGraphEventGuard,
    CudaGraphPrecaptureInputs,
};
use crate::pipeline::sampling::{sample_and_add_toks, sample_and_add_toks_batched};
use crate::pipeline::text_models_inputs_processor::{
    FlashParams, InputMetadata, PagedAttentionInputMetadata,
};
use crate::pipeline::{ChatTemplate, ModelForwardContext, RecurrentBatchKind, RecurrentMetadata};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speculative::SpeculativeTargetMixin;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
use crate::{AnyMoeExpertType, DynamicLoraRuntime, Pipeline};
use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokenizers::Tokenizer;
use tracing::info;

/// One forward step with the pipeline-specific extras split out into `args`.
pub(crate) struct CausalStepInputs<A> {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
    pub adapter_leases: Arc<[Option<crate::AdapterLease>]>,
    pub args: A,
}

/// What `CausalPipeline` needs from a model family; everything a text and a multimodal pipeline do differently lives here.
pub(crate) trait CausalModel:
    IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin + Send + Sync + 'static
{
    type Flavor: Send + Sync + 'static;
    type StepArgs;
    /// Whether verify steps of an attached proposer may run through decode graphs.
    #[cfg(feature = "cuda")]
    const SPECULATIVE_CUDA_GRAPHS: bool;

    fn as_speculative_target(&self) -> &dyn SpeculativeTargetMixin;
    fn as_speculative_target_mut(&mut self) -> &mut dyn SpeculativeTargetMixin;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn config(&self) -> &ModelConfigMetadata;
    fn supports_packed_prefill(&self) -> bool;

    fn unpack_step_inputs(inputs: Box<dyn Any>) -> CausalStepInputs<Self::StepArgs>;
    fn forward_step_args(
        &self,
        input_ids: &Tensor,
        args: Self::StepArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    #[cfg(feature = "cuda")]
    fn forward_text(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    fn speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>>;
    /// `None` asks about a text-only decode step.
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for(
        &self,
        args: Option<&Self::StepArgs>,
    ) -> candle_core::Result<bool>;
    #[cfg(feature = "cuda")]
    fn cuda_graph_step_eligible(_args: &Self::StepArgs) -> bool {
        true
    }

    fn category(pipeline: &CausalPipeline<Self>) -> ModelCategory;
    fn requires_uniform_prompt_batch(pipeline: &CausalPipeline<Self>) -> bool;
    fn reset_non_granular_state(pipeline: &CausalPipeline<Self>);
    fn input_processor_config(_flavor: &Self::Flavor) -> Option<Arc<dyn Any>> {
        None
    }
    fn processor(_flavor: &Self::Flavor) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    fn kv_cache_bypassed(_metadata: &GeneralMetadata) -> bool {
        false
    }
    /// Adjusts the batch kind and records whatever the flavor keeps per step.
    fn begin_step(
        _pipeline: &CausalPipeline<Self>,
        step: &CausalStepInputs<Self::StepArgs>,
    ) -> candle_core::Result<RecurrentBatchKind> {
        Ok(step.recurrent_batch_kind)
    }
    /// Runs the whole step outside the shared graph and eager paths.
    fn forward_override(
        _pipeline: &CausalPipeline<Self>,
        _step: &CausalStepInputs<Self::StepArgs>,
    ) -> Option<candle_core::Result<Tensor>> {
        None
    }
    fn forward_decision(
        _pipeline: &CausalPipeline<Self>,
        _sequence: &mut Sequence,
    ) -> Result<crate::decision::DecisionResponse> {
        anyhow::bail!("This pipeline does not support decision requests")
    }
    fn speculative_prompt_chunk(
        _pipeline: &mut CausalPipeline<Self>,
        _seqs: &[&mut Sequence],
        _chunk: &crate::pipeline::SpeculativePromptChunk,
        _metadata: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> candle_core::Result<()> {
        Ok(())
    }

    fn blocks_packed_prefill(&self) -> bool {
        false
    }
    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }
    fn requires_uniform_media_batch(&self) -> bool {
        false
    }
    fn supports_reasoning_budget(&self) -> bool {
        true
    }
    fn encoder_activation_memory_bytes(
        &self,
        _seq: &Sequence,
    ) -> candle_core::Result<Option<usize>> {
        Ok(None)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        None
    }
    fn reset_state_for_sequences(&self, _sequence_ids: &[usize]) {}
    fn block_generation_output(
        &self,
        _logits: &Tensor,
    ) -> candle_core::Result<Option<ForwardInputsResult>> {
        Ok(None)
    }
    fn has_custom_sampling(&self) -> bool {
        false
    }
    fn prepare_sampling(
        &self,
        _seq: &Sequence,
        logits: Tensor,
        _rng: &mut Isaac64Rng,
    ) -> candle_core::Result<(Tensor, Option<u32>)> {
        Ok((logits, None))
    }
    fn accept_sample(&self, _seq: &Sequence, _token: u32) -> candle_core::Result<()> {
        Ok(())
    }
    fn sampling_eos_tokens(&self, _seq: &Sequence, tokens: Option<&[u32]>) -> Option<Vec<u32>> {
        tokens.map(ToOwned::to_owned)
    }
    fn finalize_response(&self, _seq: &mut Sequence) -> candle_core::Result<()> {
        Ok(())
    }
    fn prepare_streaming_response(&self, _seq: &mut Sequence) -> candle_core::Result<()> {
        Ok(())
    }
    fn release_generation_sequences(&self, _sequence_ids: &[usize]) {}
    fn response_token_is_hidden(&self, _token: u32) -> bool {
        false
    }
}

/// The pipeline shared by every causal LM; `M` picks the text or multimodal flavor.
pub struct CausalPipeline<M: CausalModel + ?Sized> {
    pub(crate) model: Box<M>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) chat_template: Arc<ChatTemplate>,
    pub(crate) model_id: String,
    pub(crate) metadata: Arc<GeneralMetadata>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_decode_graph: StdMutex<CudaDecodeGraphState>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_sparse_rejection:
        StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>>,
    pub(crate) speculative_autotuner: StdMutex<crate::speculative::autotuner::SpeculativeAutotuner>,
    pub(crate) generation_defaults: Option<crate::ModelGenerationDefaults>,
    pub(crate) mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub(crate) tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    pub(crate) source_weight_files: Vec<std::path::PathBuf>,
    pub(crate) source_weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>>,
    pub(crate) dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
    pub(crate) flavor: M::Flavor,
}

#[cfg(feature = "cuda")]
type SeqRecurrentCheckpointSnapshots = Vec<(usize, RecurrentCheckpointStateSnapshot)>;
#[cfg(feature = "cuda")]
type HybridStateIndicesSnapshot = (Option<Tensor>, Option<Vec<u32>>);
#[cfg(feature = "cuda")]
struct CudaDecodeGraphCaptureInputs<'a> {
    kv_cache: &'a [(Tensor, Tensor)],
    flash_meta: &'a FlashParams,
    recurrent_batch_kind: RecurrentBatchKind,
    block_size: usize,
    speculative: bool,
}
#[cfg(feature = "cuda")]
struct CudaDecodeGraphForwardInput<'a, A> {
    input_ids: &'a Tensor,
    seqlen_offsets: &'a [usize],
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    flash_meta: &'a FlashParams,
    args: &'a A,
    recurrent_batch_kind: RecurrentBatchKind,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SpeculativeGraphTensorMetadata {
    shape: Vec<usize>,
    strides: Vec<usize>,
    contiguous: bool,
    dtype: candle_core::DType,
    device: candle_core::DeviceLocation,
}

#[cfg(feature = "cuda")]
fn speculative_graph_tensor_metadata(
    state: &dyn crate::speculative::SpeculativeGraphState,
) -> Vec<SpeculativeGraphTensorMetadata> {
    state
        .tensors()
        .into_iter()
        .map(|tensor| {
            let (_storage, layout) = tensor.storage_and_layout();
            SpeculativeGraphTensorMetadata {
                shape: tensor.dims().to_vec(),
                strides: layout.stride().to_vec(),
                contiguous: tensor.is_contiguous(),
                dtype: tensor.dtype(),
                device: tensor.device().location(),
            }
        })
        .collect()
}

#[cfg(any(feature = "cuda", test))]
fn validate_speculative_graph_tensor_metadata(
    expected: &[SpeculativeGraphTensorMetadata],
    actual: &[SpeculativeGraphTensorMetadata],
) -> candle_core::Result<()> {
    if actual.len() != expected.len() {
        candle_core::bail!(
            "speculative graph state changed tensor count between warmup and capture"
        );
    }
    if actual.iter().zip(expected).any(|(actual, expected)| {
        actual.shape != expected.shape
            || actual.strides != expected.strides
            || actual.contiguous != expected.contiguous
            || actual.dtype != expected.dtype
            || actual.device != expected.device
    }) {
        candle_core::bail!(
            "speculative graph state changed tensor metadata between warmup and capture"
        );
    }
    Ok(())
}

impl<M: CausalModel + ?Sized> PreProcessingMixin for CausalPipeline<M> {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        M::input_processor_config(&self.flavor)
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        M::processor(&self.flavor)
    }
}

impl<M: CausalModel + ?Sized> IsqPipelineMixin for CausalPipeline<M> {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if self.tracked_modules.is_empty() {
            anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
        }
        tracing::info!(
            "Re-quantizing {} layers to {dtype}.",
            self.tracked_modules.len()
        );
        self.cleanup_cuda_graphs();
        super::isq_flow::requantize_and_swap(
            &self.tracked_modules,
            dtype,
            |module| module.default_type(dtype),
            &|_| None,
        )
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.tracked_modules)?;
        #[cfg(feature = "cuda")]
        self.cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned")
            .suspend();
        Ok(())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        self.cleanup_cuda_graphs();
        let result = super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            self.source_weight_source.as_deref(),
            save_cimatrix.as_deref(),
        );
        #[cfg(feature = "cuda")]
        if result.is_ok() || !super::isq_flow::calibration_status(&self.tracked_modules).collecting
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .resume();
        }
        result
    }
}

impl<M: CausalModel + ?Sized> CacheManagerMixin for CausalPipeline<M> {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        if M::kv_cache_bypassed(&self.metadata) {
            return Ok(());
        }
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_in_cache(self, seqs, false),
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        if M::kv_cache_bypassed(&self.metadata) {
            return;
        }
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_out_cache(self, seqs, false),
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> candle_core::Result<()> {
        if !M::kv_cache_bypassed(&self.metadata) {
            match self.model.cache() {
                EitherCache::Full(_) => {
                    FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false)
                }
                EitherCache::Normal(_) => NormalCacheManager.set_none_cache(
                    self,
                    seqs,
                    modify_draft_cache,
                    load_preallocated_cache,
                ),
                EitherCache::Hybrid(_) => HybridCacheManager.set_none_cache(
                    self,
                    seqs,
                    modify_draft_cache,
                    load_preallocated_cache,
                ),
            }?;
        }
        let sequence_ids = seqs.iter().map(|seq| *seq.id()).collect::<Vec<_>>();
        self.model.reset_state_for_sequences(&sequence_ids);

        if reset_non_granular {
            self.reset_non_granular_state()
        }
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl<M: CausalModel + ?Sized> MetadataMixin for CausalPipeline<M> {
    fn supports_reasoning_budget(&self) -> bool {
        self.model.supports_reasoning_budget()
    }

    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn encoder_activation_memory_bytes(
        &self,
        seq: &Sequence,
    ) -> candle_core::Result<Option<usize>> {
        self.model.encoder_activation_memory_bytes(seq)
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        M::reset_non_granular_state(self);
    }
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .clear();
            if self.model.cache().is_hybrid() {
                if let Err(err) = self.model.cache().hybrid().release_graph_pad_slot() {
                    tracing::error!("Failed to release CUDA graph recurrent pad slot: {err}");
                }
            }
        }
    }
    fn reclaim_cuda_graph_memory(&self, max_entries: usize) -> usize {
        #[cfg(feature = "cuda")]
        {
            crate::pipeline::cuda_graph::reclaim_cuda_graph_entries(
                max_entries,
                |limit| {
                    self.cuda_decode_graph
                        .lock()
                        .expect("CUDA graph mutex poisoned")
                        .evict_lru_for_memory_pressure(limit)
                },
                |limit| self.model.evict_speculative_cuda_graphs(limit),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_entries;
            0
        }
    }
    fn precapture_cuda_decode_graphs(&self, ctx: &DecodeGraphPrecaptureCtx) {
        #[cfg(feature = "cuda")]
        {
            if !self.metadata.cuda_capture_memory_ready() {
                return;
            }
            if let Err(err) = self.precapture_cuda_decode_graphs_impl(ctx) {
                self.cuda_decode_graph
                    .lock()
                    .expect("CUDA graph mutex poisoned")
                    .clear();
                tracing::warn!(
                    "CUDA decode graph precapture failed, graphs will be captured lazily: {err}"
                );
            }
            if let Err(err) = self.model.precapture_speculative_cuda_graphs() {
                tracing::warn!("Speculative CUDA graph precapture failed, graphs will be captured lazily: {err}");
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = ctx;
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

impl<M: CausalModel + ?Sized> crate::speculative::driver::SpeculativePipelineExt
    for CausalPipeline<M>
{
    fn speculative_model(&self) -> &dyn SpeculativeTargetMixin {
        self.model.as_speculative_target()
    }

    fn speculative_model_mut(&mut self) -> &mut dyn SpeculativeTargetMixin {
        self.model.as_speculative_target_mut()
    }

    fn speculative_autotuner(
        &self,
    ) -> &StdMutex<crate::speculative::autotuner::SpeculativeAutotuner> {
        &self.speculative_autotuner
    }

    fn build_speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>> {
        self.model.speculative_verify_inputs(input_meta)
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>> {
        &self.cuda_sparse_rejection
    }
}

impl<M: CausalModel + ?Sized> CausalPipeline<M> {
    fn describe_weights(&self) -> String {
        let mut by_type: Vec<(IsqType, usize)> = Vec::new();
        for ty in self.tracked_modules.iter().filter_map(|module| module.ty) {
            match by_type.iter_mut().find(|(seen, _)| *seen == ty) {
                Some((_, count)) => *count += 1,
                None => by_type.push((ty, 1)),
            }
        }
        if by_type.is_empty() {
            return self.metadata.kind.to_string();
        }
        let isq = by_type
            .iter()
            .map(|(ty, count)| format!("{ty:?} x{count}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}, ISQ {isq}", self.metadata.kind)
    }

    fn recurrent_metadata(&self, batch_kind: RecurrentBatchKind) -> Option<RecurrentMetadata> {
        if !self.model.cache().is_hybrid() {
            return None;
        }
        let hybrid_cache = self.model.cache().hybrid();
        let state_indices_host = hybrid_cache.state_indices_host().map(ToOwned::to_owned);
        hybrid_cache.state_indices().cloned().map(|state_indices| {
            RecurrentMetadata::new(batch_kind, state_indices, state_indices_host)
        })
    }
}

#[cfg(feature = "cuda")]
impl<M: CausalModel + ?Sized> CausalPipeline<M> {
    fn uses_nonmutating_recurrent_transition_log(&self, batch_kind: RecurrentBatchKind) -> bool {
        if batch_kind != RecurrentBatchKind::SpeculativeDecode
            || !self.model.supports_recurrent_speculative_transitions()
            || self.model.speculative_verify_mutates_recurrent_state()
            || !self.model.cache().is_hybrid()
        {
            return false;
        }
        self.model.cache().hybrid().uses_recurrent_transition_log()
    }

    fn snapshot_hybrid_recurrent_checkpoints(
        &self,
        batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<Option<SeqRecurrentCheckpointSnapshots>> {
        if !self.model.cache().is_hybrid() {
            return Ok(None);
        }
        if self.uses_nonmutating_recurrent_transition_log(batch_kind) {
            return Ok(None);
        }
        self.model.flush_recurrent_state_for_current_batch()?;
        let hybrid_cache = self.model.cache().hybrid();
        let Some(mut indices) = hybrid_cache
            .logical_state_indices_host()
            .map(ToOwned::to_owned)
        else {
            return Ok(None);
        };
        indices.retain(|&idx| idx != u32::MAX);
        indices.sort_unstable();
        indices.dedup();
        let mut snapshots = Vec::with_capacity(indices.len());
        for idx in indices {
            let idx = idx as usize;
            snapshots.push((idx, hybrid_cache.snapshot_recurrent_checkpoint_state(idx)?));
        }
        Ok(Some(snapshots))
    }

    fn restore_hybrid_recurrent_checkpoints(
        &self,
        snapshots: Option<&[(usize, RecurrentCheckpointStateSnapshot)]>,
    ) -> candle_core::Result<()> {
        let Some(snapshots) = snapshots else {
            return Ok(());
        };
        let mut hybrid_cache = self.model.cache().hybrid();
        for (idx, snapshot) in snapshots {
            hybrid_cache.restore_recurrent_checkpoint_state(*idx, snapshot)?;
        }
        Ok(())
    }

    fn try_cuda_decode_graph_forward(
        &self,
        input: CudaDecodeGraphForwardInput<'_, M::StepArgs>,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        if !self.metadata.cuda_capture_memory_ready() {
            return Ok(None);
        }
        let CudaDecodeGraphForwardInput {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            args,
            recurrent_batch_kind,
        } = input;
        if !cuda_decode_graphs_enabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Disabled,
            );
            return Ok(None);
        }
        if !cuda_decode_graph_batch_kind_supported(recurrent_batch_kind) {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if !self.model.supports_cuda_decode_graphs_for(Some(args))?
            || !cuda_decode_graph_supported_for_model(self.metadata.model_metadata.as_deref())
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::ModelUnsupported,
            );
            return Ok(None);
        }
        let speculative = self.model.has_speculative_proposer();
        if speculative && !M::SPECULATIVE_CUDA_GRAPHS {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::SpeculativeConflict,
            );
            return Ok(None);
        }
        let Some((kv_cache, metadata)) = paged_attn_meta else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::PagedAttentionUnavailable,
            );
            return Ok(None);
        };
        if metadata.is_first_prompt_chunk || metadata.num_cached_tokens.is_some() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if metadata.decode_rows.is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if (q_len != 1 && !speculative)
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || position_ids.len() != batch
            || !input_ids.device().is_cuda()
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        // With a proposer attached every step is a fixed-width verify; the model must expose the
        // outputs the proposer reads so a replay can refresh them.
        if speculative && self.model.take_speculative_graph_state().is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::SpeculativeConflict,
            );
            return Ok(None);
        }
        let Some(bucket) = cuda_graph_batch_bucket(CudaGraphComponent::Target, q_len, batch)
            .filter(|bucket| {
                q_len == 1
                    || crate::speculative::SpeculativeGraphPlan::allows(
                        &self.model.speculative_graph_plans(),
                        q_len,
                        *bucket,
                    )
            })
        else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::BatchUnsupported,
            );
            return Ok(None);
        };
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::CacheConfigUnavailable,
            );
            return Ok(None);
        };
        // Captured kernels require canonical strides, but an already contiguous input needs no copy.
        let input_ids = &input_ids.contiguous()?;

        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::RuntimeDisabled,
            );
            return Ok(None);
        }
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let slots = hybrid_graph_slots(&mut self.model.cache().hybrid())?;
            if let Some(slots) = &slots {
                state.observe_recurrent_storage_generation(slots.storage_generation);
            }
            slots
        } else {
            None
        };
        let graph_pad_slot = hybrid_slots.as_ref().map(|_| GDN_PAD_SLOT);
        let Some(step) = CudaGraphDecodeStep::padded(
            CudaGraphDecodeStepInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                metadata,
                state_indices: hybrid_slots.as_ref().map(|slots| slots.real.as_slice()),
                pad_slot: graph_pad_slot,
            },
            bucket,
        )?
        else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::PaddingUnavailable,
            );
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(
            &step.input_ids,
            &step.metadata,
            cache_config.block_size,
            recurrent_batch_kind,
        )?;
        if let Some(replay) = state.replay(&key, &step, CudaDecodeGraphReplayInput::Host)? {
            if let Some(spec_state) = replay.spec_state.as_deref() {
                if let Err(err) = self.model.install_speculative_graph_state(spec_state) {
                    state.block_eager_retry();
                    return Err(err);
                }
            }
            return Ok(Some(replay));
        }

        if !crate::pipeline::cuda_graph::cuda_graph_capture_headroom(step.input_ids.device())? {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::MemoryPressure,
            );
            return Ok(None);
        }
        let replay_key = key.clone();
        let captured = self.capture_cuda_decode_graph_step(
            &mut state,
            key,
            &step,
            CudaDecodeGraphCaptureInputs {
                kv_cache: kv_cache.as_slice(),
                flash_meta,
                recurrent_batch_kind,
                block_size: cache_config.block_size,
                speculative,
            },
            true,
        )?;
        if captured.is_none() {
            return Ok(None);
        }
        super::synchronize_cuda_contexts(step.input_ids.device(), self.mapper.as_ref()).map_err(
            |err| {
                candle_core::Error::msg(format!(
                    "CUDA graph rollback synchronization failed: {err}"
                ))
            },
        )?;
        let replay = state
            .replay(&replay_key, &step, CudaDecodeGraphReplayInput::Host)?
            .ok_or_else(|| {
                candle_core::Error::msg("newly captured CUDA decode graph was not replayable")
            })?;
        if let Some(spec_state) = replay.spec_state.as_deref() {
            if let Err(err) = self.model.install_speculative_graph_state(spec_state) {
                state.block_eager_retry();
                return Err(err);
            }
        }
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Eager,
            CudaGraphDispatchReason::CachePopulation,
        );
        Ok(Some(replay))
    }

    fn precapture_cuda_decode_graphs_impl(
        &self,
        ctx: &DecodeGraphPrecaptureCtx,
    ) -> candle_core::Result<()> {
        let device = self.device();
        if !cuda_decode_graphs_enabled()
            || !device.is_cuda()
            || !self.model.supports_cuda_decode_graphs_for(None)?
            || !cuda_decode_graph_supported_for_model(self.metadata.model_metadata.as_deref())
        {
            return Ok(());
        }
        let (Some(cache_config), Some(cache_engine)) =
            (&self.metadata.cache_config, &self.metadata.cache_engine)
        else {
            return Ok(());
        };
        let speculative = self.model.has_speculative_proposer();
        if speculative && !M::SPECULATIVE_CUDA_GRAPHS {
            return Ok(());
        }
        let graph_plans = self
            .model
            .speculative_graph_plans()
            .into_iter()
            .filter(|plan| {
                M::SPECULATIVE_CUDA_GRAPHS
                    && cuda_graph_startup_capture_allowed(1 + plan.proposal_len)
            })
            .collect::<Vec<_>>();
        let mut widths = vec![(1usize, usize::MAX)];
        for plan in graph_plans {
            if plan.proposal_len > 0 {
                widths.push((
                    1 + plan.proposal_len,
                    plan.max_batch_size.unwrap_or(usize::MAX),
                ));
            }
        }
        widths.sort_unstable();
        widths.dedup();
        let kv_cache = cache_engine.get_kv_cache().clone();
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let mut cache = self.model.cache().hybrid();
            let Some(pad_slot) = cache.graph_pad_slot()? else {
                return Ok(());
            };
            let pad_slot = cache.active_physical_slot(pad_slot)?;
            let pad_slot = u32::try_from(pad_slot).map_err(|_| {
                candle_core::Error::msg(format!("recurrent graph pad slot {pad_slot} exceeds u32"))
            })?;
            Some(pad_slot)
        } else {
            None
        };
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let mut captured = 0usize;
        let runtime_max_bucket =
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, ctx.max_batch_size);
        let runtime_batch_shapes = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1)
            .filter(|bucket| *bucket <= runtime_max_bucket)
            .count();
        let startup_max_bucket = widths
            .iter()
            .map(|(q_len, max_bucket)| {
                (*max_bucket).min(cuda_graph_precapture_max_batch(
                    CudaGraphComponent::Target,
                    *q_len,
                    ctx.max_batch_size,
                ))
            })
            .max()
            .unwrap_or(runtime_max_bucket);
        let startup_shapes = widths.iter().fold(0usize, |total, (q_len, max_bucket)| {
            let width_max_bucket = cuda_graph_precapture_max_batch(
                CudaGraphComponent::Target,
                *q_len,
                ctx.max_batch_size,
            );
            total.saturating_add(
                cuda_graph_precapture_batches(CudaGraphComponent::Target, *q_len)
                    .filter(|bucket| *bucket <= (*max_bucket).min(width_max_bucket))
                    .count(),
            )
        });
        state.ensure_capacity(target_cuda_graph_cache_capacity(
            startup_shapes,
            runtime_batch_shapes,
        ));
        let mut plans = Vec::with_capacity(widths.len());
        for (q_len, max_bucket) in widths {
            let inputs = CudaGraphPrecaptureInputs::new(ctx, q_len, &device, self.device_mapper())?;
            let recurrent_batch_kind = if q_len == 1 {
                RecurrentBatchKind::Decode
            } else if speculative {
                RecurrentBatchKind::SpeculativeDecode
            } else {
                RecurrentBatchKind::Prefill
            };
            let width_max_bucket = cuda_graph_precapture_max_batch(
                CudaGraphComponent::Target,
                q_len,
                ctx.max_batch_size,
            );
            let buckets = cuda_graph_precapture_batches(CudaGraphComponent::Target, q_len)
                .filter(|bucket| *bucket <= max_bucket.min(width_max_bucket))
                .collect::<Vec<_>>();
            plans.push((inputs, recurrent_batch_kind, buckets));
        }
        // Every graph pins host and device memory, so small batches of every width go first and capture
        // stops once unified memory runs short; the rest are captured lazily under the same guard
        let mut order = plans
            .iter()
            .enumerate()
            .flat_map(|(plan_idx, (_, _, buckets))| {
                buckets.iter().map(move |bucket| (*bucket, plan_idx))
            })
            .collect::<Vec<_>>();
        order.sort_unstable();
        let live = hybrid_slots.map(|pad_slot| vec![pad_slot]);
        let graph_pad_slot = hybrid_slots.map(|_| GDN_PAD_SLOT);
        let mut deferred = 0usize;
        for (bucket, plan_idx) in order {
            let (inputs, recurrent_batch_kind, _) = &plans[plan_idx];
            let recurrent_batch_kind = *recurrent_batch_kind;
            let Some(step) = CudaGraphDecodeStep::padded(
                inputs.step_inputs(live.as_deref(), graph_pad_slot),
                bucket,
            )?
            else {
                continue;
            };
            let key = CudaDecodeGraphKey::new(
                &step.input_ids,
                &step.metadata,
                cache_config.block_size,
                recurrent_batch_kind,
            )?;
            if state.contains(&key) {
                continue;
            }
            if deferred > 0 || !crate::pipeline::cuda_graph::cuda_graph_capture_headroom(&device)? {
                deferred += 1;
                continue;
            }
            // The live step's slot table is whatever the pad slot is; the model only sees it
            // through the installed graph buffers
            if self
                .capture_cuda_decode_graph_step(
                    &mut state,
                    key,
                    &step,
                    CudaDecodeGraphCaptureInputs {
                        kv_cache: kv_cache.as_slice(),
                        flash_meta: &inputs.flash_meta,
                        recurrent_batch_kind,
                        block_size: cache_config.block_size,
                        speculative,
                    },
                    false,
                )?
                .is_some()
            {
                captured += 1;
            }
        }
        if deferred > 0 {
            info!(
                "Deferred {deferred} CUDA decode graph shapes to lazy capture for memory headroom; reduce --kv-cache to precapture more."
            );
        }
        if speculative {
            let _ = self.model.take_speculative_graph_state();
        }
        if captured > 0 {
            info!(
                "Captured {captured} CUDA decode graphs through batch bucket {} in {:.2?}",
                startup_max_bucket,
                start.elapsed()
            );
        }
        Ok(())
    }

    /// Captures after one eager warmup; live calls roll it back so the first replay is canonical.
    fn capture_cuda_decode_graph_step(
        &self,
        state: &mut CudaDecodeGraphState,
        key: CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        inputs: CudaDecodeGraphCaptureInputs<'_>,
        rollback_live_state: bool,
    ) -> candle_core::Result<Option<Tensor>> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Capture);
        let CudaDecodeGraphCaptureInputs {
            kv_cache,
            flash_meta,
            recurrent_batch_kind,
            block_size,
            speculative,
        } = inputs;
        if speculative {
            state.prepare_spec_state_admission_for_key(&key);
        }
        let Device::Cuda(cuda_device) = step.input_ids.device() else {
            candle_core::bail!("CUDA graph decode expected CUDA input ids");
        };
        let execution_devices = self.execution_devices();
        let capture_memory = CudaGraphCaptureMemory {
            plan: self.metadata.memory_plan,
            reservation: self.metadata.runtime_memory_reservation.as_deref(),
            devices: &execution_devices,
            primary: step.input_ids.device(),
            activation_dtype: self.metadata.activation_dtype,
            block_size,
            step,
            kv_cache,
            model: self.metadata.model_metadata.as_deref(),
        };
        if !prepare_cuda_capture_memory(state, &capture_memory)? {
            return Ok(None);
        }
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let metadata = step
            .metadata
            .materialize_decode_tensors()
            .map_err(candle_core::Error::msg)?;

        let uses_recurrent_transition_log = self.model.cache().is_hybrid()
            && self.model.cache().hybrid().uses_recurrent_transition_log();
        if rollback_live_state
            && recurrent_batch_kind == RecurrentBatchKind::Decode
            && self.model.supports_recurrent_speculative_transitions()
            && uses_recurrent_transition_log
            && !self
                .model
                .apply_recurrent_speculative_transitions_for_current_batch()?
        {
            candle_core::bail!(
                "CUDA graph capture could not materialize pending recurrent transitions"
            );
        }

        let nonmutating_transition_capture =
            self.uses_nonmutating_recurrent_transition_log(recurrent_batch_kind);
        // Precapture runs before any sequence owns a slot, so there is no live state to restore
        let recurrent_snapshots = if rollback_live_state {
            self.snapshot_hybrid_recurrent_checkpoints(recurrent_batch_kind)?
        } else {
            None
        };
        let live_state_indices = self.snapshot_hybrid_state_indices();
        let mut warm_spec_state = None;
        let mut warm_spec_metadata = None;
        let mut warm_live_spec_state = None;
        let mut graph_spec_state = None;
        let capture_attempt: candle_core::Result<_> = (|| {
            let state_index_buffers = match &step.state_indices {
                Some(host) => Some(install_hybrid_graph_state_indices(
                    &mut self.model.cache().hybrid(),
                    host,
                )?),
                None => None,
            };
            let mut ctx = ModelForwardContext::new(
                &step.seqlen_offsets,
                &step.context_lens,
                &step.position_ids,
                Some((kv_cache, &metadata)),
                flash_meta,
            )
            .with_recurrent_batch_kind(recurrent_batch_kind)
            .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
            let warmup_logits = self.model.forward_text(&step.input_ids, &mut ctx)?;
            warm_spec_state = speculative
                .then(|| self.model.take_speculative_graph_state())
                .flatten();
            warm_spec_metadata = warm_spec_state
                .as_deref()
                .map(speculative_graph_tensor_metadata);
            if !nonmutating_transition_capture {
                warm_live_spec_state = warm_spec_state
                    .as_deref()
                    .map(|state| state.for_real_batch(step.real_batch))
                    .transpose()?;
            }
            step.input_ids.device().synchronize()?;
            let live_logits = step.narrow_rows(&warmup_logits)?;
            if !prepare_cuda_capture_memory(state, &capture_memory)? {
                return Ok((live_logits, None));
            }

            let spec_state_usage = warm_spec_state
                .as_deref()
                .map(|warm| state.prepare_spec_state_admission(warm))
                .transpose()?;
            if nonmutating_transition_capture {
                warm_spec_state = None;
            }
            // CUDA stream capture records recurrent writes without executing them.
            let entry = capture_cuda_decode_graph(
                CudaDecodeGraphCaptureCtx {
                    key,
                    input_ids: &step.input_ids,
                    seqlen_offsets: &step.seqlen_offsets,
                    position_ids: &step.position_ids,
                    block_size,
                    kv_cache,
                    metadata: &metadata,
                    model_metadata: self.metadata.model_metadata.as_deref(),
                    activation_dtype: self.metadata.activation_dtype,
                    warmup_logits: &warmup_logits,
                    state_indices: state_index_buffers,
                    real_batch: step.real_batch,
                },
                |graph_input_ids, graph_metadata| {
                    let mut ctx = ModelForwardContext::new(
                        &step.seqlen_offsets,
                        &step.context_lens,
                        &step.position_ids,
                        Some((kv_cache, graph_metadata)),
                        flash_meta,
                    )
                    .with_recurrent_batch_kind(recurrent_batch_kind)
                    .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
                    let logits = self.model.forward_text(graph_input_ids, &mut ctx)?;
                    if let Some(expected) = warm_spec_metadata.as_deref() {
                        let captured =
                            self.model.take_speculative_graph_state().ok_or_else(|| {
                                candle_core::Error::msg(
                                    "captured forward left no speculative state",
                                )
                            })?;
                        let actual = speculative_graph_tensor_metadata(&*captured);
                        validate_speculative_graph_tensor_metadata(expected, &actual)?;
                        graph_spec_state = Some(captured);
                    }
                    Ok(logits)
                },
            )?;
            let entry = entry.with_spec_state(graph_spec_state.take(), spec_state_usage);
            let entry = finish_cuda_capture_memory(state, &capture_memory, entry)?;
            Ok((live_logits, entry))
        })();
        self.restore_hybrid_state_indices(live_state_indices.as_ref());
        let capture_attempt = if let Some(warm) = warm_live_spec_state.as_deref() {
            match self.model.install_speculative_graph_state(warm) {
                Ok(()) => capture_attempt,
                Err(install_err) => {
                    state.block_eager_retry();
                    match capture_attempt {
                        Ok(_) => Err(install_err),
                        Err(capture_err) => Err(candle_core::Error::msg(format!(
                            "CUDA graph capture failed: {capture_err}; warm speculative state restoration failed: {install_err}"
                        ))),
                    }
                }
            }
        } else {
            capture_attempt
        };
        let (logits, entry) = self.finish_cuda_graph_capture_attempt(
            state,
            capture_attempt,
            recurrent_snapshots.as_deref(),
            live_state_indices.as_ref(),
            rollback_live_state,
        )?;
        if let Some(entry) = entry {
            state.insert(entry);
            graph_event.success();
            Ok(Some(logits))
        } else {
            drop(logits);
            drop(metadata);
            drop(recurrent_snapshots);
            drop(live_state_indices);
            drop(warm_spec_state);
            drop(warm_live_spec_state);
            drop(graph_spec_state);
            super::cuda_graph::admit_cuda_eager_after_capture(state, &capture_memory)?;
            Ok(None)
        }
    }

    fn snapshot_hybrid_state_indices(&self) -> Option<HybridStateIndicesSnapshot> {
        self.model.cache().is_hybrid().then(|| {
            let cache = self.model.cache().hybrid();
            (
                cache.state_indices().cloned(),
                cache.state_indices_host().map(ToOwned::to_owned),
            )
        })
    }

    fn restore_hybrid_state_indices(&self, snapshot: Option<&HybridStateIndicesSnapshot>) {
        if let Some((tensor, host)) = snapshot {
            self.model
                .cache()
                .hybrid()
                .set_physical_state_indices_with_host(tensor.clone(), host.clone());
        }
    }

    fn finish_cuda_graph_capture_attempt<T>(
        &self,
        state: &mut CudaDecodeGraphState,
        attempt: candle_core::Result<T>,
        recurrent_snapshots: Option<&[(usize, RecurrentCheckpointStateSnapshot)]>,
        live_state_indices: Option<&HybridStateIndicesSnapshot>,
        rollback_live_state: bool,
    ) -> candle_core::Result<T> {
        self.restore_hybrid_state_indices(live_state_indices);
        match attempt {
            Ok(value) if !rollback_live_state => Ok(value),
            Ok(value) => {
                self.restore_hybrid_recurrent_checkpoints(recurrent_snapshots)
                    .map_err(|restore_err| {
                        state.block_eager_retry();
                        candle_core::Error::msg(format!(
                            "CUDA graph captured, but recurrent checkpoint rollback failed: {restore_err}"
                        ))
                    })?;
                Ok(value)
            }
            Err(capture_err) => {
                if let Err(restore_err) =
                    self.restore_hybrid_recurrent_checkpoints(recurrent_snapshots)
                {
                    state.block_eager_retry();
                    return Err(candle_core::Error::msg(format!(
                        "CUDA graph capture failed: {capture_err}; recurrent checkpoint rollback failed: {restore_err}"
                    )));
                }
                Err(capture_err)
            }
        }
    }

    fn disable_cuda_decode_graph(&self, err: &candle_core::Error) -> bool {
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        let eager_retry_allowed = state.take_eager_retry_allowed();
        if !state.disabled() {
            tracing::warn!("CUDA decode graphs disabled after capture/replay error: {err:?}");
        }
        state.disable();
        drop(state);
        if self.model.cache().is_hybrid() {
            if let Err(release_err) = self.model.cache().hybrid().release_graph_pad_slot() {
                tracing::error!(
                    "Failed to release recurrent graph pad after graph disable: {release_err}"
                );
            }
        }
        eager_retry_allowed
    }
}

#[async_trait::async_trait]
impl<M: CausalModel + ?Sized> Pipeline for CausalPipeline<M> {
    fn requires_uniform_prompt_batch(&self) -> bool {
        M::requires_uniform_prompt_batch(self)
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        self.model.requires_uniform_completion_batch()
    }

    fn requires_uniform_media_batch(&self) -> bool {
        self.model.requires_uniform_media_batch()
    }

    fn supports_packed_prefill(&self) -> bool {
        self.model.supports_packed_prefill()
            && self.metadata.cache_engine.is_some()
            && !self.model.blocks_packed_prefill()
            && (!self.model.has_speculative_proposer()
                || self.model.supports_speculative_packed_prefill())
            && self.model.device().is_cuda()
            && self.mapper.get_unique_devices().iter().all(Device::is_cuda)
            && crate::using_flash_attn()
            && crate::attention::flash_backend_supports_sdpa(
                self.model.config().k_head_dim,
                false,
                self.metadata.sliding_window.is_some(),
            )
            && crate::attention::flash_dtype_supported(self.metadata.activation_dtype)
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        !self.model.has_speculative_proposer()
    }

    fn log_runtime_plan(&self) {
        if self.metadata.loaded_for_uqff_write {
            return;
        }
        let device = self.model.device();
        #[cfg(feature = "cuda")]
        let decode_graphs = device.is_cuda().then(|| {
            cuda_decode_graphs_enabled()
                && self
                    .model
                    .supports_cuda_decode_graphs_for(None)
                    .unwrap_or(false)
                && cuda_decode_graph_supported_for_model(self.metadata.model_metadata.as_deref())
        });
        #[cfg(not(feature = "cuda"))]
        let decode_graphs = None;
        let plan = super::plan_log::RuntimePlan::resolve(&super::plan_log::RuntimePlanInput {
            device,
            activation_dtype: self.metadata.activation_dtype,
            weights: self.describe_weights(),
            model: self
                .metadata
                .model_metadata
                .as_deref()
                .map(|model| model as &dyn crate::paged_attention::ModelConfigLike),
            cache: self.metadata.cache_config.as_ref(),
            head_dim: self.model.config().k_head_dim,
            has_sliding_window: self.metadata.sliding_window.is_some(),
            packed_prefill: self.supports_packed_prefill(),
            batched_sampling: self.supports_batched_cuda_sampling(),
            has_proposer: self.model.has_speculative_proposer(),
            decode_graphs,
        });
        info!("{}", plan.render());
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.model.supports_speculative_prompt_bootstrap()
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        self.model.speculative_prefix_replay()
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        self.model.supports_paged_auxiliary_prefix_state()
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> candle_core::Result<Option<Arc<dyn crate::prefix_cacher::PagedAuxiliaryPrefixState>>> {
        self.model
            .capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn crate::prefix_cacher::PagedAuxiliaryPrefixState,
    ) -> candle_core::Result<()> {
        self.model
            .restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
    }

    fn adapter_runtime(&self) -> Option<Arc<DynamicLoraRuntime>> {
        self.dynamic_lora.clone()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        Ok(self.forward_step(inputs, return_raw_logits)?.output)
    }

    fn forward_decision(
        &self,
        sequence: &mut Sequence,
    ) -> Result<crate::decision::DecisionResponse> {
        M::forward_decision(self, sequence)
    }

    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> candle_core::Result<ForwardStepResult> {
        let step = M::unpack_step_inputs(inputs);
        let lora_execution = super::resolve_lora_execution(
            self.dynamic_lora.as_deref(),
            &step.input_ids,
            step.paged_attn_meta.as_ref(),
            &step.flash_meta,
            &step.adapter_leases,
        )?;
        match (&self.metadata.cache_engine, &step.paged_attn_meta) {
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            _ => {}
        }
        if let Some(logits) = M::forward_override(self, &step) {
            let logits = logits?;
            return Ok(ForwardStepResult::eager(if return_raw_logits {
                ForwardInputsResult::RawLogits { logits }
            } else {
                ForwardInputsResult::CausalGeneration { logits }
            }));
        }
        let recurrent_batch_kind = M::begin_step(self, &step)?;
        let CausalStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            args,
            ..
        } = step;
        let paged_attn_meta = match (&self.metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let mut cuda_graph_eager_fallback = None;
        #[cfg(feature = "cuda")]
        if lora_execution.is_none() && !return_raw_logits && M::cuda_graph_step_eligible(&args) {
            match self.try_cuda_decode_graph_forward(CudaDecodeGraphForwardInput {
                input_ids: &input_ids,
                seqlen_offsets: &seqlen_offsets,
                context_lens: &context_lens,
                position_ids: &position_ids,
                paged_attn_meta: paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), *b)),
                flash_meta: &flash_meta,
                args: &args,
                recurrent_batch_kind,
            }) {
                Ok(Some(replay)) => {
                    return Ok(ForwardStepResult::cuda_decode(
                        ForwardInputsResult::CausalGeneration {
                            logits: replay.logits,
                        },
                        replay.launch,
                    ))
                }
                Ok(None) => {}
                Err(err) => {
                    if !self.disable_cuda_decode_graph(&err) {
                        return Err(err);
                    }
                    cuda_graph_eager_fallback = Some(CudaGraphEventGuard::new(
                        CudaGraphComponent::Target,
                        CudaGraphEvent::EagerFallback,
                    ));
                }
            }
        }
        let paged_attn_meta = paged_attn_meta
            .map(|(kv_cache, metadata)| {
                metadata
                    .materialize_decode_tensors()
                    .map(|metadata| (kv_cache, metadata))
            })
            .transpose()
            .map_err(candle_core::Error::msg)?;
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            paged_attn_meta
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache.as_slice(), meta)),
            &flash_meta,
        )
        .with_recurrent_batch_kind(recurrent_batch_kind)
        .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
        let eager_result = mistralrs_quant::with_lora_execution(lora_execution, || {
            self.model.forward_step_args(&input_ids, args, &mut ctx)
        });
        #[cfg(feature = "cuda")]
        if eager_result.is_ok() {
            if let Some(graph_event) = cuda_graph_eager_fallback.take() {
                graph_event.success();
            }
        }
        let logits = eager_result?;
        let output = if return_raw_logits {
            ForwardInputsResult::RawLogits { logits }
        } else if let Some(output) = self.model.block_generation_output(&logits)? {
            output
        } else {
            ForwardInputsResult::CausalGeneration { logits }
        };
        Ok(ForwardStepResult::eager(output))
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_decode_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> candle_core::Result<Option<ForwardStepResult>> {
        let replay = {
            let mut state = self
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned");
            if state.disabled() {
                return Ok(None);
            }
            state.replay_one_token(launch)
        };
        match replay {
            Ok(Some(replay)) => {
                if let Some(spec_state) = replay.spec_state.as_deref() {
                    if let Err(err) = self.model.install_speculative_graph_state(spec_state) {
                        let _ = self.disable_cuda_decode_graph(&err);
                        return Err(err);
                    }
                }
                Ok(Some(ForwardStepResult::cuda_decode(
                    ForwardInputsResult::CausalGeneration {
                        logits: replay.logits,
                    },
                    replay.launch,
                )))
            }
            Ok(None) => Ok(None),
            Err(err) => {
                let _ = self.disable_cuda_decode_graph(&err);
                Err(err)
            }
        }
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> candle_core::Result<()> {
        self.attach_speculative_with_runtime(
            config,
            crate::speculative::MtpRuntimeConfig::default(),
        )
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> candle_core::Result<()> {
        if self.dynamic_lora.is_some() {
            candle_core::bail!("dynamic LoRA does not support speculative decoding");
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_))
            && self.get_metadata().cache_engine.is_none()
        {
            candle_core::bail!(
                "MTP speculative decoding currently requires PagedAttention for this pipeline."
            );
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_)) {
            self.cleanup_cuda_graphs();
            self.model.disable_recurrent_decode_deferred_storage()?;
        }
        if let Some(info) = self
            .model
            .attach_speculative_with_runtime(config, runtime)?
        {
            self.model.log_speculative_attach(&info);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> candle_core::Result<()> {
        self.model.release_speculative_sequences(seq_ids)
    }

    fn flush_recurrent_speculative_transitions(
        &self,
        seq_ids: &[usize],
    ) -> candle_core::Result<()> {
        self.model.flush_recurrent_speculative_transitions(seq_ids)
    }

    fn speculative_prompt_chunk(
        &mut self,
        seqs: &[&mut Sequence],
        chunk: &crate::pipeline::SpeculativePromptChunk,
        metadata: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> candle_core::Result<()> {
        M::speculative_prompt_chunk(self, seqs, chunk, metadata)
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        seqs: &mut [&mut Sequence],
        logits: &[Tensor],
        batched_logits: Option<&Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::pipeline::text_models_inputs_processor::PagedAttentionMeta>,
        logger: &crate::IntervalLogger,
    ) -> candle_core::Result<bool> {
        if !self.model.has_speculative_proposer() {
            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            return Ok(false);
        }

        let general_metadata = self.get_metadata();
        if let Some(cache_engine) = general_metadata.cache_engine.as_ref() {
            let Some(metadata) = metadata else {
                crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                return Ok(false);
            };
            let cache = crate::speculative::cache::PagedSpeculativeCacheAccess::new(
                &metadata,
                cache_engine,
            );
            return crate::speculative::driver::try_sample_speculative_causal_gen(
                self,
                seqs,
                logits,
                batched_logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                &cache,
                logger,
            )
            .await;
        }

        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Ok(false)
    }

    async fn try_sample_causal_gen_batched(
        &self,
        seqs: &mut [&mut Sequence],
        logits: &Tensor,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<bool, candle_core::Error> {
        if self.model.has_speculative_proposer() || self.model.has_custom_sampling() {
            return Ok(false);
        }
        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        sample_and_add_toks_batched(
            self,
            seqs,
            logits.clone(),
            prefix_cacher,
            disable_eos_stop,
            rng,
        )
        .await?;
        Ok(true)
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        if !self.model.has_custom_sampling() {
            return sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
                .await;
        }
        for (seq, logits) in seqs.iter_mut().zip(logits) {
            let sampling_rng = seq.sampling_rng(&rng);
            let (logits, forced) = self.model.prepare_sampling(
                seq,
                logits,
                &mut sampling_rng.lock().expect("sampling RNG poisoned"),
            )?;
            let metadata = self.get_metadata();
            let eos = self.model.sampling_eos_tokens(
                seq,
                seq.effective_eos_tokens(&metadata.eos_tok, disable_eos_stop),
            );
            let next = if let Some(token) = forced {
                crate::sampler::Logprobs {
                    token,
                    logprob: 0.0,
                    bytes: None,
                    top_logprobs: None,
                }
            } else {
                let return_logprobs = seq.return_logprobs();
                let sampled = super::sampling::sample_sequence(
                    logits.to_device(&Device::Cpu)?,
                    seq,
                    return_logprobs,
                    eos.as_deref(),
                    metadata.llg_factory.clone(),
                    metadata.max_model_len,
                    rng.clone(),
                    false,
                    false,
                    false,
                )
                .await;
                crate::handle_seq_error_stateaware_ok!(sampled, seq)
            };
            self.model.accept_sample(seq, next.token)?;
            let eos = self.model.sampling_eos_tokens(
                seq,
                seq.effective_eos_tokens(&metadata.eos_tok, disable_eos_stop),
            );
            let cache_prefix = seq.audio_output().is_none();
            super::sampling::finish_or_add_toks_to_seq(
                self,
                prefix_cacher,
                seq,
                next,
                eos.as_deref(),
                cache_prefix,
            )
            .await?;
        }
        Ok(())
    }

    fn finalize_response(&self, seq: &mut crate::sequence::Sequence) -> candle_core::Result<()> {
        self.model.finalize_response(seq)
    }
    fn prepare_streaming_response(
        &self,
        seq: &mut crate::sequence::Sequence,
    ) -> candle_core::Result<()> {
        self.model.prepare_streaming_response(seq)
    }
    fn release_generation_sequences(&self, sequence_ids: &[usize]) {
        self.model.release_generation_sequences(sequence_ids);
    }
    fn response_token_is_hidden(&self, token: u32) -> bool {
        self.model.response_token_is_hidden(token)
    }

    async fn sample_block_gen(
        &self,
        input_seqs: &mut [&mut Sequence],
        token_blocks: Vec<Vec<u32>>,
        denoise_times: Vec<std::time::Duration>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
    ) -> Result<(), candle_core::Error> {
        crate::pipeline::sampling::finalize_block_gen(
            self,
            input_seqs,
            token_blocks,
            denoise_times,
            prefix_cacher,
            disable_eos_stop,
        )
        .await
    }
    fn category(&self) -> ModelCategory {
        M::category(self)
    }

    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    )> {
        self.model.encoder_cache_counters()
    }
}

impl<M: CausalModel + ?Sized> AnyMoePipelineMixin for CausalPipeline<M> {
    fn amoe_finish_training(&mut self, gate_model_id: Option<String>) -> candle_core::Result<()> {
        self.model.finish_training(gate_model_id)
    }
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        self.model.get_vars()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        self.model.trainable_params()
    }
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.model.take_cached_gating_outputs()
    }
    fn amoe_create_layers(
        &mut self,
        model_ids: Vec<String>,
        token: &TokenSource,
        revision: Option<String>,
        match_regex: &str,
        config: crate::amoe::AnyMoeConfig,
        dtype: candle_core::DType,
        dev: &Device,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        silent: bool,
        gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        let mut vbs = Vec::new();
        // Precompile regex here
        let regex = Regex::new(match_regex).map_err(candle_core::Error::msg)?;
        for model_id in model_ids {
            let filenames = super::amoe::download_expert_files(
                &self.model_id,
                &model_id,
                token,
                revision.as_deref().unwrap_or("main"),
                silent,
            )
            .map_err(candle_core::Error::msg)?;

            let regex = regex.clone();
            let match_regex_clone = match_regex.to_string();
            let layers_clone = layers.clone();
            let vb = from_mmaped_safetensors(
                filenames,
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                move |key| {
                    if regex.is_match(&key) {
                        // Idx of the last char of the layer id, +1
                        // Assumes N.MLP
                        let last_layer_idx = key.find(&match_regex_clone).unwrap() - 1;
                        let first_layer_idx = key[..last_layer_idx].rfind('.').unwrap();
                        let layer_n = key[first_layer_idx + 1..last_layer_idx]
                            .parse::<usize>()
                            .unwrap();
                        layers_clone.contains(&layer_n) || layers_clone.is_empty()
                    } else {
                        false
                    }
                },
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            vbs.push(vb);
        }

        let gate_vb = if let Some(gate_model_id) = gate_model_id {
            let gate_filenames = super::amoe::download_expert_files(
                &self.model_id,
                &gate_model_id,
                token,
                revision.as_deref().unwrap_or("main"),
                silent,
            )
            .map_err(candle_core::Error::msg)?;
            assert_eq!(
                gate_filenames.len(),
                1,
                "Gate model ID must contain only one .safetensors file"
            );

            let vb = from_mmaped_safetensors(
                gate_filenames.clone(),
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            info!(
                "Loaded gating layers from `{}`",
                gate_filenames[0].display()
            );
            Some(vb)
        } else {
            None
        };

        self.model
            .create_anymoe_layers(vbs, config, (prefix, mlp), layers, expert_type, gate_vb)
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe_supported()
    }
}

#[cfg(test)]
#[path = "causal_tests.rs"]
mod tests;
