use crate::{
    distributed,
    paged_attention::block_hash::{adapter_generation_key, compute_block_hashes, BlockHash},
    pipeline::{
        execution::StepSubmissionKind,
        llg::{constraint_from_llg_grammar, llg_grammar_from_constraint},
        text_models_inputs_processor::PagedAttentionMeta,
        CacheBackendMetadata, CacheInstruction, DecodeGraphPrecaptureCtx, StepLookahead,
        StepSubmission, RECURRENT_GRAPH_PAD_SLOTS,
    },
    prefix_cacher::PrefixCacheManagerV2,
    scheduler::{DefaultSchedulerMethod, PagedPrefixCacheValidator, Scheduler, SchedulerOutput},
    search::{self, rag::SearchPipeline},
    sequence::{SeqStepType, StopReason},
    tools, SchedulerConfig, DEBUG,
};
use interprocess::local_socket::{traits::Listener, ListenerOptions};
use llguidance::ParserFactory;
pub use logger::IntervalLogger;
use mistralrs_quant::RingConfig;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    io::{BufWriter, Write},
    net::TcpListener,
    ops::Deref,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock,
    },
    time::{Duration, Instant},
};
use tokio::{
    select,
    sync::{
        mpsc::{error::TryRecvError, Receiver, Sender},
        Mutex, Notify,
    },
    task::JoinHandle,
};

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error,
    pipeline::{ModelCategory, Pipeline},
    request::Request,
    response::{
        ChatCompletionResponse, Choice, CompletionChoice, CompletionResponse, ResponseMessage,
    },
    sequence::{SequenceRecognizer, SequenceState},
    Constraint,
};

mod add_request;
mod admission;
pub(crate) mod agentic_loop;
#[cfg(any(feature = "cuda", test))]
mod cuda_decode;
pub use agentic_loop::DEFAULT_MAX_TOOL_ROUNDS;
pub(crate) mod agentic_session;
mod file_tools;
mod logger;
mod tool_dispatch;

#[cfg(feature = "cuda")]
use self::cuda_decode::CudaDecodeCompletionWorker;
#[cfg(feature = "cuda")]
use crate::pipeline::execution::{
    submit_decode_tail, CudaDecodeTail, CudaStepCompletion, CudaStepSubmission, CudaTailSubmission,
};
#[cfg(feature = "cuda")]
use crate::sequence::Sequence;

pub enum EngineInstruction {
    Terminate,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Embedding model used for ranking web search results internally.
pub enum SearchEmbeddingModel {
    #[default]
    #[serde(rename = "embedding_gemma")]
    EmbeddingGemma300M,
}

impl SearchEmbeddingModel {
    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma300M => "google/embeddinggemma-300m",
        }
    }

    pub fn variants() -> &'static [&'static str] {
        &["embedding_gemma"]
    }
}

impl fmt::Display for SearchEmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddingGemma300M => f.write_str("embedding_gemma"),
        }
    }
}

impl FromStr for SearchEmbeddingModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "embedding_gemma" => Ok(Self::EmbeddingGemma300M),
            other => Err(format!(
                "Unknown search embedding model `{other}`. Supported values: {}",
                Self::variants().join(", ")
            )),
        }
    }
}

const SEED: u64 = 0;
/// Terminate all sequences on the next scheduling step. Be sure to reset this.
/// This is a global flag for terminating all engines at once (e.g., Ctrl+C).
pub static TERMINATE_ALL_NEXT_STEP: AtomicBool = AtomicBool::new(false);

/// Engine-specific termination flags, per Engine thread ID.
static ENGINE_TERMINATE_FLAGS: LazyLock<
    std::sync::Mutex<HashMap<std::thread::ThreadId, Arc<AtomicBool>>>,
> = LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Get or create a termination flag for the current engine thread.
pub fn get_engine_terminate_flag() -> Arc<AtomicBool> {
    let thread_id = std::thread::current().id();
    let mut flags = ENGINE_TERMINATE_FLAGS.lock().unwrap();
    flags
        .entry(thread_id)
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

/// Check if the current engine should terminate sequences.
pub fn should_terminate_engine_sequences() -> bool {
    // Check global flag first
    if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
        return true;
    }
    // Then check engine-specific flag
    let thread_id = std::thread::current().id();
    if let Ok(flags) = ENGINE_TERMINATE_FLAGS.lock() {
        if let Some(flag) = flags.get(&thread_id) {
            return flag.load(Ordering::SeqCst);
        }
    }
    false
}

/// Reset termination flags for the current engine.
pub fn reset_engine_terminate_flag() {
    let thread_id = std::thread::current().id();
    if let Ok(flags) = ENGINE_TERMINATE_FLAGS.lock() {
        if let Some(flag) = flags.get(&thread_id) {
            flag.store(false, Ordering::SeqCst);
        }
    }
}

/// Engine instructions, per Engine (MistralRs) ID.
pub static ENGINE_INSTRUCTIONS: LazyLock<
    std::sync::Mutex<HashMap<usize, Option<EngineInstruction>>>,
> = LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

pub struct Engine {
    tx: Sender<Request>,
    rx: Arc<Mutex<Receiver<Request>>>,
    pipeline: Arc<Mutex<dyn Pipeline>>,
    search_pipeline: Arc<Mutex<Option<SearchPipeline>>>,
    search_callback: Option<Arc<search::SearchCallback>>,
    tool_callbacks: tools::ToolCallbacksWithTools,
    scheduler: Arc<Mutex<dyn Scheduler>>,
    max_active_sequences: usize,
    id: Arc<Mutex<usize>>,
    no_kv_cache: bool,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
    is_debug: bool,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
    logger: Arc<IntervalLogger>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pending_notify: Arc<Notify>,
    pub(crate) session_store: Arc<std::sync::Mutex<agentic_session::AgenticSessionStore>>,
    pub(crate) file_store: crate::files::FileStore,
    // Re-runs the decode graph precapture after the recurrent pool grows and drops every graph
    pub(crate) graph_precapture_ctx: Option<DecodeGraphPrecaptureCtx>,
    #[cfg(feature = "cuda")]
    cuda_decode_enabled: bool,
}

#[cfg(feature = "cuda")]
struct CudaDecodeBatchLease {
    rows: Vec<Arc<std::sync::Mutex<Sequence>>>,
    sequence_ids: Box<[usize]>,
    tail: CudaDecodeTail,
    started: Instant,
}

#[cfg(feature = "cuda")]
impl CudaDecodeBatchLease {
    fn new(
        rows: Vec<Arc<std::sync::Mutex<Sequence>>>,
        tail: CudaDecodeTail,
    ) -> candle_core::Result<Self> {
        if tail.batch_size()? != rows.len() {
            candle_core::bail!(
                "CUDA decode tail has {} rows for a leased batch of {}",
                tail.batch_size()?,
                rows.len()
            );
        }
        let sequence_ids = rows
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            rows,
            sequence_ids,
            tail,
            started: Instant::now(),
        })
    }
}

struct HybridPagedPrefixValidator {
    pipeline: Arc<Mutex<dyn Pipeline>>,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
}

impl HybridPagedPrefixValidator {
    fn reset_recurrent_state(&self, slot_idx: usize) {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        if !pipeline.cache().is_hybrid() {
            return;
        }
        let mut hybrid_cache = pipeline.cache().hybrid();
        if let Err(e) = hybrid_cache.reset_seq(slot_idx) {
            tracing::warn!("Failed to reset recurrent state slot {slot_idx}: {e}");
        }
    }
}

impl PagedPrefixCacheValidator for HybridPagedPrefixValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        seq: &mut crate::sequence::Sequence,
        block_hashes: &[BlockHash],
        cached_tokens: usize,
        block_size: usize,
    ) -> usize {
        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return 0;
        };

        if cached_tokens == 0 || !cached_tokens.is_multiple_of(block_size) {
            self.reset_recurrent_state(slot_idx);
            return 0;
        }

        let max_blocks = cached_tokens / block_size;
        let Some((n_blocks, snapshots)) = get_mut_arcmutex!(self.prefix_cacher)
            .get_longest_paged_recurrent_prefix(block_hashes, max_blocks)
        else {
            self.reset_recurrent_state(slot_idx);
            return 0;
        };

        let pipeline = get_mut_arcmutex!(self.pipeline);
        if !pipeline.cache().is_hybrid() {
            return cached_tokens;
        }
        let mut hybrid_cache = pipeline.cache().hybrid();
        if hybrid_cache
            .restore_recurrent_state(slot_idx, &snapshots)
            .is_ok()
        {
            n_blocks * block_size
        } else {
            if let Err(e) = hybrid_cache.reset_seq(slot_idx) {
                tracing::warn!("Failed to reset recurrent state slot {slot_idx}: {e}");
            }
            0
        }
    }

    fn release_recurrent_state(&mut self, slot_idx: usize) -> bool {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        if !pipeline.cache().is_hybrid() {
            return false;
        }
        pipeline.cache().hybrid().free_seq(slot_idx);
        true
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        for handle in &*get_mut_arcmutex!(self.handles) {
            handle.abort();
        }
    }
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tx: Sender<Request>,
        rx: Receiver<Request>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        config: SchedulerConfig,
        mut no_kv_cache: bool,
        mut no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
        throughput_logging_enabled: bool,
        search_embedding_model: Option<SearchEmbeddingModel>,
        search_callback: Option<Arc<search::SearchCallback>>,
        tool_callbacks: tools::ToolCallbacksWithTools,
        logger: Arc<IntervalLogger>,
        session_store: Arc<std::sync::Mutex<agentic_session::AgenticSessionStore>>,
        file_store: crate::files::FileStore,
    ) -> anyhow::Result<Self> {
        no_kv_cache |= get_mut_arcmutex!(pipeline).get_metadata().no_kv_cache;

        no_prefix_cache = no_prefix_cache
            || no_kv_cache
            || get_mut_arcmutex!(pipeline).get_metadata().no_prefix_cache
            || prefix_cache_n == 0;

        let search_pipeline = match search_embedding_model {
            Some(search_embedding_model) => Some(SearchPipeline::new(
                search_embedding_model,
                &get_mut_arcmutex!(pipeline).device(),
            )?),
            None => None,
        };

        let config = if no_kv_cache {
            match config {
                SchedulerConfig::PagedAttentionMeta { max_num_seqs, .. } => {
                    SchedulerConfig::DefaultScheduler {
                        method: DefaultSchedulerMethod::Fixed(max_num_seqs.try_into().unwrap()),
                    }
                }
                config => config,
            }
        } else {
            config
        };
        let max_active_sequences = match &config {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(max_num_seqs),
            } => max_num_seqs.get(),
            SchedulerConfig::PagedAttentionMeta { max_num_seqs, .. } => *max_num_seqs,
        };
        let capacity_metric = u32::try_from(max_active_sequences).unwrap_or(u32::MAX);
        metrics::gauge!("mistralrs_sequences_capacity").set(f64::from(capacity_metric));

        let (
            requires_uniform_prompt_batch,
            requires_uniform_completion_batch,
            requires_uniform_media_batch,
            supports_packed_prefill,
            scheduler_visible_prompt_chunks,
            prompt_chunks_require_block_alignment,
        ) = {
            let pipeline = get_mut_arcmutex!(pipeline);
            let pipeline_metadata = pipeline.get_metadata();
            (
                pipeline.requires_uniform_prompt_batch(),
                pipeline.requires_uniform_completion_batch(),
                pipeline.requires_uniform_media_batch(),
                pipeline.supports_packed_prefill(),
                pipeline.device().is_cuda() && !pipeline_metadata.is_xlora,
                pipeline.cache().is_hybrid(),
            )
        };
        let recurrent_capacity = match &config {
            SchedulerConfig::PagedAttentionMeta { max_num_seqs, .. } => Some(
                max_num_seqs
                    .checked_add(RECURRENT_GRAPH_PAD_SLOTS)
                    .ok_or_else(|| anyhow::anyhow!("maximum sequence count overflow"))?,
            ),
            SchedulerConfig::DefaultScheduler { .. } => None,
        };
        if let Some(recurrent_capacity) = recurrent_capacity {
            let pipeline = get_mut_arcmutex!(pipeline);
            if pipeline.cache().is_hybrid() {
                pipeline
                    .cache()
                    .hybrid()
                    .reserve_recurrent_capacity(recurrent_capacity)?;
            }
        }
        let scheduler = config.into_scheduler();
        get_mut_arcmutex!(scheduler)
            .set_requires_uniform_prompt_batch(requires_uniform_prompt_batch);
        get_mut_arcmutex!(scheduler)
            .set_requires_uniform_completion_batch(requires_uniform_completion_batch);
        get_mut_arcmutex!(scheduler).set_requires_uniform_media_batch(requires_uniform_media_batch);
        get_mut_arcmutex!(scheduler).set_supports_packed_prefill(supports_packed_prefill);
        get_mut_arcmutex!(scheduler).set_scheduler_visible_prompt_chunks(
            scheduler_visible_prompt_chunks,
            prompt_chunks_require_block_alignment,
        );

        // Configure prefix caching on the scheduler based on the global no_prefix_cache flag
        // This ensures PagedAttention prefix caching respects the same setting
        get_mut_arcmutex!(scheduler).set_prefix_caching_enabled(!no_prefix_cache);

        let has_paged_attention = get_mut_arcmutex!(scheduler).kv_cache_manager().is_some();
        #[cfg(feature = "cuda")]
        let cuda_decode_enabled =
            has_paged_attention && get_mut_arcmutex!(pipeline).device().is_cuda();
        let graph_precapture_ctx = if no_kv_cache {
            None
        } else {
            let ctx = {
                let scheduler = get_mut_arcmutex!(scheduler);
                scheduler
                    .kv_cache_manager()
                    .zip(scheduler.block_size())
                    .map(|(kv_cache_manager, block_size)| {
                        let pipeline = get_mut_arcmutex!(pipeline);
                        let pipeline_metadata = pipeline.get_metadata();
                        let max_paged_context_len = {
                            let kv_mgr = get_mut_arcmutex!(kv_cache_manager);
                            kv_mgr.num_gpu_blocks().saturating_sub(1).max(1) * block_size
                        };
                        DecodeGraphPrecaptureCtx {
                            block_size,
                            max_paged_context_len,
                            attention_backend: pipeline_metadata
                                .model_metadata
                                .as_ref()
                                .map(|metadata| metadata.attention_backend_kind())
                                .unwrap_or(crate::paged_attention::AttentionBackendKind::Standard),
                            sliding_window: pipeline_metadata.sliding_window,
                            num_kv_heads: pipeline_metadata
                                .model_metadata
                                .as_ref()
                                .map(|metadata| metadata.num_kv_heads())
                                .unwrap_or(1)
                                .max(1),
                        }
                    })
            };
            if let Some(ctx) = &ctx {
                get_mut_arcmutex!(pipeline).precapture_cuda_decode_graphs(ctx);
            }
            ctx
        };

        Ok(Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            pipeline,
            search_pipeline: Arc::new(Mutex::new(search_pipeline)),
            search_callback,
            tool_callbacks,
            scheduler: scheduler.clone(),
            max_active_sequences,
            id: Arc::new(Mutex::new(0)),
            no_kv_cache,
            prefix_cacher: Arc::new(Mutex::new(PrefixCacheManagerV2::new(
                prefix_cache_n,
                no_prefix_cache,
                has_paged_attention,
            ))),
            is_debug: DEBUG.load(Ordering::Relaxed),
            disable_eos_stop,
            throughput_logging_enabled,
            logger,
            handles: Arc::new(Mutex::new(Vec::new())),
            pending_notify: Arc::new(Notify::new()),
            session_store,
            file_store,
            graph_precapture_ctx,
            #[cfg(feature = "cuda")]
            cuda_decode_enabled,
        })
    }

    /// Returns the maximum supported sequence length for the underlying model, if applicable.
    #[allow(dead_code)]
    pub fn max_sequence_length(&self) -> Option<usize> {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        let category = pipeline.category();

        if matches!(category, ModelCategory::Diffusion | ModelCategory::Speech) {
            None
        } else {
            Some(pipeline.get_metadata().max_seq_len)
        }
    }

    fn free_finished_scheduler_sequences(&self, scheduler: &mut dyn Scheduler) {
        let finished_sequence_ids = scheduler.get_finished_sequence_ids();
        let recurrent_indices = scheduler.get_finished_recurrent_indices();
        if !finished_sequence_ids.is_empty() || !recurrent_indices.is_empty() {
            let mut pipeline = get_mut_arcmutex!(self.pipeline);
            pipeline.release_speculative_sequences(&finished_sequence_ids);
            if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                let mut hybrid_cache = pipeline.cache().hybrid();
                for idx in recurrent_indices {
                    hybrid_cache.free_seq(idx);
                }
            }
        }
        scheduler.free_finished_sequence_groups();
    }

    fn resolve_adapter_generation(&self, request: &mut Request) -> Result<(), String> {
        let Request::Normal(request) = request else {
            return Ok(());
        };
        let Some(selection) = request.adapter.as_mut() else {
            return Ok(());
        };
        if selection.is_pinned() {
            return Ok(());
        }

        let runtime = get_mut_arcmutex!(self.pipeline)
            .adapter_runtime()
            .ok_or_else(|| {
                "request selected an adapter, but this pipeline has no dynamic LoRA runtime"
                    .to_string()
            })?;
        selection.pin(&runtime).map_err(|err| err.to_string())
    }

    async fn prepare_request_for_dispatch(&self, mut request: Request) -> Option<Request> {
        if let Err(error) = self.resolve_adapter_generation(&mut request) {
            if let Request::Normal(request) = request {
                request
                    .response
                    .send(crate::Response::ValidationError(error.into()))
                    .await
                    .unwrap_or_else(|_| tracing::warn!("Receiver disconnected"));
            }
            return None;
        }
        Some(request)
    }

    fn admission_class(request: &Request) -> admission::AdmissionClass {
        match request {
            Request::Normal(request) => admission::AdmissionClass::Workload {
                sequences: request.sampling_params.n_choices.max(1),
            },
            Request::Tokenize(_) | Request::Detokenize(_) | Request::TerminateAllSeqsNextStep => {
                admission::AdmissionClass::BypassControl
            }
            Request::Terminate => admission::AdmissionClass::Shutdown,
            Request::ReIsq(_) | Request::Calibration(_) => {
                admission::AdmissionClass::OrderedControl
            }
        }
    }

    async fn collect_pending_requests(
        &self,
        pending: &mut admission::AdmissionQueue<Request>,
    ) -> bool {
        let (requests, disconnected) = {
            let mut rx = self.rx.lock().await;
            let to_receive = rx.len().min(pending.remaining_capacity());
            let mut requests = Vec::with_capacity(to_receive);
            let mut disconnected = false;
            for _ in 0..to_receive {
                match rx.try_recv() {
                    Ok(request) => requests.push(request),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            disconnected |= rx.is_closed() && rx.is_empty();
            (requests, disconnected)
        };

        for request in requests {
            let Some(request) = self.prepare_request_for_dispatch(request).await else {
                continue;
            };
            let class = Self::admission_class(&request);
            pending
                .push(request, class)
                .expect("pending request capacity changed while collecting ingress");
        }
        disconnected
    }

    async fn dispatch_prepared_request(self: &Arc<Self>, request: Request) -> bool {
        self.replicate_request_to_daemons(&request);
        if matches!(request, Request::Terminate) {
            return false;
        }
        self.clone().handle_request(request).await;
        true
    }

    #[cfg(feature = "cuda")]
    async fn complete_cuda_step(
        &self,
        worker: &CudaDecodeCompletionWorker,
        submission: CudaStepSubmission,
    ) -> candle_core::Result<CudaStepCompletion> {
        let (current, pending) = submission.into_parts();
        let completion = worker.submit(current).await?;
        pending.finish(completion.await?)
    }

    #[cfg(feature = "cuda")]
    fn account_cuda_decode_rows(&self, rows: &[Arc<std::sync::Mutex<Sequence>>]) {
        for row in rows {
            get_mut_arcmutex!(row).advance_num_computed_tokens(1);
        }
        self.logger.add_tokens_processed(rows.len());
    }

    #[cfg(feature = "cuda")]
    fn drain_cuda_decode_batch(&self, lease: CudaDecodeBatchLease) -> candle_core::Result<()> {
        let CudaDecodeBatchLease { rows, tail, .. } = lease;
        tail.drain()?;
        self.account_cuda_decode_rows(&rows);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    async fn continue_cuda_decode_batch(
        &self,
        lease: CudaDecodeBatchLease,
        worker: &CudaDecodeCompletionWorker,
        allow_lookahead: bool,
    ) -> candle_core::Result<Option<CudaDecodeBatchLease>> {
        let CudaDecodeBatchLease {
            rows,
            sequence_ids,
            tail,
            started,
        } = lease;
        let commit_rows = rows
            .iter()
            .map(|seq| !get_mut_arcmutex!(seq).is_finished_paged_attn())
            .collect::<Vec<_>>();
        if !commit_rows.iter().any(|commit| *commit) {
            tail.drain()?;
            self.account_cuda_decode_rows(&rows);
            return Ok(None);
        }

        for (row, commit) in rows.iter().zip(&commit_rows) {
            if *commit {
                get_mut_arcmutex!(row).start_completion_timing();
            }
        }

        let submission = {
            let mut guards = rows
                .iter()
                .map(|seq| seq.lock().unwrap())
                .collect::<Vec<_>>();
            let guards_mut = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
            let lookahead = if allow_lookahead {
                StepLookahead::OneToken
            } else {
                StepLookahead::Disabled
            };
            let mut pipeline = get_mut_arcmutex!(self.pipeline);
            match submit_decode_tail(&mut *pipeline, &guards_mut, tail, Duration::ZERO, lookahead)?
            {
                CudaTailSubmission::Submitted(submission) => submission,
                CudaTailSubmission::Unsupported(mut unsupported) => {
                    unsupported.synchronize()?;
                    candle_core::bail!(
                        "leased CUDA decode batch no longer supports resident completion"
                    );
                }
            }
        };

        let has_next_tail = submission.has_tail();
        if has_next_tail {
            let mut scheduler = get_mut_arcmutex!(self.scheduler);
            scheduler.record_decode_continuation();
        }
        let mut completion = self.complete_cuda_step(worker, submission).await?;

        let step_duration = started.elapsed();
        let completion = {
            let mut guards = rows
                .iter()
                .map(|seq| seq.lock().unwrap())
                .collect::<Vec<_>>();
            let mut guards_mut = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
            for seq in &mut guards_mut {
                seq.advance_num_computed_tokens(1);
            }
            self.logger.add_tokens_processed(guards_mut.len());

            let pipeline = get_mut_arcmutex!(self.pipeline);
            if crate::pipeline::sampling::greedy_batch_will_finish(
                &*pipeline,
                &guards_mut,
                completion.token_ids(),
                &commit_rows,
                self.disable_eos_stop,
            )? {
                completion.synchronize_tail()?;
            }
            let completion = completion
                .finish(
                    &*pipeline,
                    &mut guards_mut,
                    &commit_rows,
                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                    self.disable_eos_stop,
                )
                .await?;
            for (seq, commit) in guards_mut.iter_mut().zip(&commit_rows) {
                if *commit {
                    seq.finish_completion_timing(step_duration);
                }
            }
            completion
        };

        let Some(mut next_tail) = completion.into_cuda_tail() else {
            return Ok(None);
        };
        let any_live = rows
            .iter()
            .any(|seq| !get_mut_arcmutex!(seq).is_finished_paged_attn());
        if !any_live {
            next_tail.synchronize()?;
            self.account_cuda_decode_rows(&rows);
            return Ok(None);
        }

        debug_assert!(has_next_tail);
        debug_assert_eq!(
            sequence_ids.as_ref(),
            rows.iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>()
                .as_slice()
        );
        CudaDecodeBatchLease::new(rows, next_tail).map(Some)
    }

    pub async fn run(self: Arc<Self>) {
        if self.throughput_logging_enabled {
            self.logger.enable_logging();
        }

        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        let max_pending_requests = {
            let rx = self.rx.lock().await;
            rx.max_capacity()
        };
        let policy =
            admission::AdmissionPolicy::new(self.max_active_sequences, max_pending_requests);
        let mut pending = admission::AdmissionQueue::new(policy);
        #[cfg(feature = "cuda")]
        let cuda_completion_worker = if self.cuda_decode_enabled {
            match CudaDecodeCompletionWorker::new() {
                Ok(worker) => Some(worker),
                Err(err) => {
                    tracing::warn!("Failed to start the CUDA decode completion worker: {err}");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let mut cuda_decode_lease: Option<CudaDecodeBatchLease> = None;
        'lp: loop {
            let should_terminate = || {
                matches!(
                    ENGINE_INSTRUCTIONS
                        .lock()
                        .expect("`ENGINE_INSTRUCTIONS` was poisoned")
                        .get(get_mut_arcmutex!(self.id).deref()),
                    Some(Some(EngineInstruction::Terminate))
                )
            };

            if should_terminate() {
                #[cfg(feature = "cuda")]
                if let Some(lease) = cuda_decode_lease.take() {
                    if let Err(err) = self.drain_cuda_decode_batch(lease) {
                        tracing::warn!("Failed to drain the CUDA decode tail: {err}");
                    }
                }
                self.replicate_request_to_daemons(&Request::Terminate);
                break 'lp;
            }

            let channel_disconnected = self.collect_pending_requests(&mut pending).await;
            #[cfg(feature = "cuda")]
            let decode_batch_leased = cuda_decode_lease.is_some();
            #[cfg(not(feature = "cuda"))]
            let decode_batch_leased = false;
            if let Some(request) = pending.take_shutdown() {
                #[cfg(feature = "cuda")]
                if let Some(lease) = cuda_decode_lease.take() {
                    if let Err(err) = self.drain_cuda_decode_batch(lease) {
                        tracing::warn!("Failed to drain the CUDA decode tail: {err}");
                    }
                }
                self.replicate_request_to_daemons(&request);
                break 'lp;
            }

            let mut dispatches = 0;
            while dispatches < pending.max_dispatches_per_step() {
                let Some(request) = pending.take_bypass_control() else {
                    break;
                };
                if !self.dispatch_prepared_request(request).await {
                    break 'lp;
                }
                dispatches += 1;
            }

            while dispatches < pending.max_dispatches_per_step() {
                let active_sequences = {
                    let scheduler = get_mut_arcmutex!(self.scheduler);
                    scheduler.waiting_len() + scheduler.running_len()
                };
                let request = if decode_batch_leased {
                    pending.pop_admissible_workload(active_sequences)
                } else {
                    pending.pop_admissible(active_sequences)
                };
                let Some(request) = request else {
                    break;
                };
                if !self.dispatch_prepared_request(request).await {
                    break 'lp;
                }
                dispatches += 1;
            }
            let pending_metric = u32::try_from(pending.len()).unwrap_or(u32::MAX);
            metrics::gauge!("mistralrs_requests_pending_admission").set(f64::from(pending_metric));

            let (waiting_len, running_len) = {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                (scheduler.waiting_len(), scheduler.running_len())
            };
            let scheduler_idle = waiting_len == 0 && running_len == 0;

            if scheduler_idle {
                if !pending.is_empty() {
                    continue;
                }
                if channel_disconnected {
                    break 'lp;
                }
                if should_terminate() {
                    self.replicate_request_to_daemons(&Request::Terminate);
                    break 'lp;
                }
                enum WaitEvent {
                    Request(Option<Request>),
                    Wake,
                }
                let wait_for_request = async {
                    let mut rx = self.rx.lock().await;
                    rx.recv().await
                };
                tokio::pin!(wait_for_request);
                let wait_for_wake = self.pending_notify.notified();
                tokio::pin!(wait_for_wake);

                let event = select! {
                    res = &mut wait_for_request => WaitEvent::Request(res),
                    _ = &mut wait_for_wake => WaitEvent::Wake,
                };

                match event {
                    WaitEvent::Request(Some(request)) => {
                        let Some(request) = self.prepare_request_for_dispatch(request).await else {
                            continue;
                        };
                        let class = Self::admission_class(&request);
                        pending
                            .push(request, class)
                            .expect("idle admission queue must have capacity");
                        // Give a concurrently submitted request wave one turn to reach the ingress channel.
                        tokio::task::yield_now().await;
                        continue;
                    }
                    WaitEvent::Request(None) => break 'lp,
                    WaitEvent::Wake => {
                        continue;
                    }
                }
            }

            if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
                self.replicate_request_to_daemons(&Request::TerminateAllSeqsNextStep);
                #[cfg(feature = "cuda")]
                if let Some(lease) = cuda_decode_lease.take() {
                    let leased_rows = lease.rows.clone();
                    let result = self.drain_cuda_decode_batch(lease);
                    let mut guards = leased_rows
                        .iter()
                        .map(|seq| seq.lock().unwrap())
                        .collect::<Vec<_>>();
                    let mut guards_mut =
                        guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
                    handle_pipeline_forward_error!(
                        "CUDA decode cancellation drain",
                        result,
                        &mut guards_mut,
                        self.pipeline,
                        'lp,
                        self.prefix_cacher
                    );
                }
            }

            #[cfg(feature = "cuda")]
            if let Some(lease) = cuda_decode_lease.take() {
                let leased_rows = lease.rows.clone();
                let allow_lookahead = !pending.blocks_decode_continuation() && {
                    let scheduler = get_mut_arcmutex!(self.scheduler);
                    scheduler.can_continue_decode_batch(&lease.sequence_ids)
                };
                let result = self
                    .continue_cuda_decode_batch(
                        lease,
                        cuda_completion_worker
                            .as_ref()
                            .expect("CUDA decode lease requires a completion worker"),
                        allow_lookahead,
                    )
                    .await;
                let mut guards = leased_rows
                    .iter()
                    .map(|seq| seq.lock().unwrap())
                    .collect::<Vec<_>>();
                let mut guards_mut = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
                cuda_decode_lease = handle_pipeline_forward_error!(
                    "resident CUDA decode step",
                    result,
                    &mut guards_mut,
                    self.pipeline,
                    'lp,
                    self.prefix_cacher
                );
                drop(guards_mut);
                drop(guards);
                if cuda_decode_lease.is_none() {
                    let mut scheduler = get_mut_arcmutex!(self.scheduler);
                    self.free_finished_scheduler_sequences(&mut *scheduler);
                }
                continue 'lp;
            }

            let run_start = Instant::now();
            let use_hybrid_prefix_validator = {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                !self.no_kv_cache && pipeline.cache().is_hybrid()
            };
            let mut hybrid_prefix_validator =
                use_hybrid_prefix_validator.then(|| HybridPagedPrefixValidator {
                    pipeline: self.pipeline.clone(),
                    prefix_cacher: self.prefix_cacher.clone(),
                });
            let prefix_validator = hybrid_prefix_validator
                .as_mut()
                .map(|v| v as &mut dyn PagedPrefixCacheValidator);
            let mut scheduler = get_mut_arcmutex!(self.scheduler);
            self.free_finished_scheduler_sequences(&mut *scheduler);
            let scheduled = scheduler.schedule(&self.logger, prefix_validator);

            match scheduled {
                SchedulerOutput::DefaultScheduler {
                    output: mut scheduled,
                } => {
                    if !scheduled.completion.is_empty() {
                        let current_completion_ids: Vec<usize> =
                            scheduled.completion.iter().map(|seq| *seq.id()).collect();
                        for seq in scheduled.completion.iter_mut() {
                            seq.start_completion_timing();
                        }
                        let res = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);
                            let pre_op = if !self.no_kv_cache
                                && last_completion_ids != current_completion_ids
                            {
                                CacheInstruction::In
                            } else {
                                CacheInstruction::Nothing
                            };
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                }
                            };

                            let return_raw_logits = scheduled.completion[0].return_raw_logits;
                            assert!(
                                scheduled
                                    .completion
                                    .iter()
                                    .all(|seq| seq.return_raw_logits == return_raw_logits),
                                "All sequences must either return raw logits, or not."
                            );

                            pipeline
                                .step(
                                    &mut scheduled.completion,
                                    false,
                                    return_raw_logits,
                                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                                    self.disable_eos_stop,
                                    rng.clone(),
                                    CacheBackendMetadata::DefaultInstructions { pre_op, post_op },
                                    self.logger.as_ref(),
                                )
                                .await
                        };

                        let completion_exec_time = handle_pipeline_forward_error!(
                            "completion step",
                            res,
                            &mut scheduled.completion,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );
                        for seq in scheduled.completion.iter_mut() {
                            seq.finish_completion_timing(completion_exec_time);
                        }

                        self.logger.add_tokens_processed(scheduled.completion.len());

                        last_completion_ids = current_completion_ids;
                    }

                    if !scheduled.prompt.is_empty() {
                        for seq in scheduled.prompt.iter_mut() {
                            seq.start_prompt_timing();
                        }

                        let prompt_exec_time = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);

                            // Run the prompt seqs
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                }
                            };

                            let return_raw_logits = scheduled.prompt[0].return_raw_logits;
                            assert!(
                                scheduled
                                    .prompt
                                    .iter()
                                    .all(|seq| seq.return_raw_logits == return_raw_logits),
                                "All sequences must either return raw logits, or not."
                            );

                            // This comes from prefix caching
                            // The invariant where all token offsets are the same is handled by the scheduler
                            let pre_op = if scheduled.prompt[0].token_offset() != 0 {
                                CacheInstruction::In
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: true,
                                    reset_non_granular: false,
                                }
                            };

                            pipeline
                                .step(
                                    &mut scheduled.prompt,
                                    true,
                                    return_raw_logits,
                                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                                    self.disable_eos_stop,
                                    rng.clone(),
                                    CacheBackendMetadata::DefaultInstructions { pre_op, post_op },
                                    self.logger.as_ref(),
                                )
                                .await
                        };

                        let prompt_exec_time = handle_pipeline_forward_error!(
                            "prompt step",
                            prompt_exec_time,
                            &mut scheduled.prompt,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        let total_processed_tokens: usize = scheduled
                            .prompt
                            .iter()
                            .map(|seq| seq.get_toks().len())
                            .sum();
                        self.logger.add_tokens_processed(total_processed_tokens);

                        for seq in scheduled.prompt.iter_mut() {
                            if !seq.is_finished_paged_attn() {
                                match seq.sequence_stepping_type() {
                                    SeqStepType::OneShot => seq
                                        .set_state(SequenceState::Done(StopReason::GeneratedImage)),
                                    SeqStepType::PromptAndDecode => {
                                        seq.set_state(SequenceState::RunningCompletion)
                                    }
                                }
                            }
                            seq.finish_prompt_timing(prompt_exec_time);
                        }
                        last_completion_ids = vec![];
                    }

                    if self.is_debug {
                        let ms_from_last_run = run_start.elapsed().as_secs_f64();
                        let total_len = scheduled.prompt.len() + scheduled.completion.len();
                        if total_len > 0 {
                            let prompt_lengths = scheduled
                                .prompt
                                .iter()
                                .map(|seq| seq.len().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");

                            let completion_lengths = scheduled
                                .completion
                                .iter()
                                .map(|seq| seq.len().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");

                            tracing::info!(
                                "Prompt[{}] Completion[{}] - {}ms",
                                prompt_lengths,
                                completion_lengths,
                                ms_from_last_run * 1000.,
                            );
                        }
                    }
                }
                SchedulerOutput::PagedAttention {
                    mut output,
                    preempted_sequence_ids,
                } => {
                    let block_size = scheduler.block_size().unwrap();
                    let kv_cache_manager = scheduler.kv_cache_manager().unwrap();
                    let is_prompt = output
                        .scheduled
                        .first()
                        .is_some_and(|seq| get_mut_arcmutex!(seq).is_prompt());
                    #[cfg(feature = "cuda")]
                    let step_lookahead = if !is_prompt
                        && cuda_completion_worker.is_some()
                        && !pending.blocks_decode_continuation()
                        && scheduler.can_continue_decode_batch(
                            &output
                                .scheduled
                                .iter()
                                .map(|seq| *get_mut_arcmutex!(seq).id())
                                .collect::<Vec<_>>(),
                        ) {
                        StepLookahead::OneToken
                    } else {
                        StepLookahead::Disabled
                    };
                    #[cfg(not(feature = "cuda"))]
                    let step_lookahead = StepLookahead::Disabled;
                    drop(scheduler);
                    if !preempted_sequence_ids.is_empty() {
                        get_mut_arcmutex!(self.pipeline)
                            .release_speculative_sequences(&preempted_sequence_ids);
                    }
                    if !output.scheduled.is_empty() {
                        for seq in output.scheduled.iter() {
                            let mut seq_guard = get_mut_arcmutex!(seq);
                            if is_prompt {
                                seq_guard.start_prompt_timing();
                            } else {
                                seq_guard.start_completion_timing();
                            }
                        }

                        let mut guards = output
                            .scheduled
                            .iter_mut()
                            .map(|seq| seq.lock().unwrap())
                            .collect::<Vec<_>>();

                        let mut guards_mut =
                            guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();

                        let staged_width =
                            crate::speculative::staging::staged_batch_width(&guards_mut);
                        let scheduler_visible_prompt_step =
                            output.scheduled_prompt_chunks.is_some();
                        let num_computed_before_step = guards_mut
                            .iter()
                            .map(|seq| seq.num_computed_tokens())
                            .collect::<Vec<_>>();
                        let scheduled_token_counts = guards_mut
                            .iter()
                            .enumerate()
                            .map(|(seq_idx, seq)| {
                                if is_prompt {
                                    if let Some(chunk) = output
                                        .scheduled_prompt_chunks
                                        .as_ref()
                                        .and_then(|chunks| chunks.get(seq_idx))
                                    {
                                        return chunk.end - chunk.start;
                                    }
                                }
                                let staged = staged_width
                                    .map(|_| seq.active_staged_speculative_len())
                                    .unwrap_or_default();
                                seq.num_uncomputed_tokens().saturating_add(staged)
                            })
                            .collect::<Vec<_>>();

                        let res = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);

                            if guards_mut.is_empty() {
                                Ok(StepSubmission::ready(Duration::ZERO))
                            } else {
                                let pipeline_metadata = pipeline.get_metadata();
                                let model_metadata = pipeline_metadata.model_metadata.as_ref();
                                let max_paged_context_len = {
                                    let kv_mgr = get_mut_arcmutex!(kv_cache_manager);
                                    kv_mgr.num_gpu_blocks().saturating_sub(1).max(1) * block_size
                                };
                                let scheduled_prompt_chunks = output.scheduled_prompt_chunks.take();
                                let prompt_chunk_attention_policy = scheduled_prompt_chunks
                                    .as_ref()
                                    .and_then(|chunks| chunks.first())
                                    .map(|chunk| chunk.attention_policy)
                                    .unwrap_or(
                                        crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
                                    );
                                let is_final_prompt_chunk =
                                    scheduled_prompt_chunks.as_ref().is_none_or(|chunks| {
                                        chunks
                                            .iter()
                                            .zip(guards_mut.iter())
                                            .all(|(chunk, seq)| chunk.end == seq.get_toks().len())
                                    });
                                let metadata = PagedAttentionMeta {
                                    block_size,
                                    max_paged_context_len,
                                    sliding_window: pipeline_metadata.sliding_window,
                                    attention_backend: model_metadata
                                        .map(|metadata| metadata.attention_backend_kind())
                                        .unwrap_or(
                                            crate::paged_attention::AttentionBackendKind::Standard,
                                        ),
                                    has_flashinfer_decode_layers: model_metadata
                                        .is_some_and(|metadata| {
                                            (0..metadata.num_layers()).any(|layer_idx| {
                                                metadata.attention_backend_kind_for_layer(layer_idx)
                                                    == crate::paged_attention::AttentionBackendKind::FlashInfer
                                            })
                                        }),
                                    prefill_attention_heads: model_metadata
                                        .map(|metadata| metadata.num_attn_heads())
                                        .unwrap_or(1)
                                        .max(1),
                                    prefill_key_value_heads: model_metadata
                                        .map(|metadata| metadata.num_kv_heads())
                                        .unwrap_or(1)
                                        .max(1),
                                    prefill_head_dim: model_metadata
                                        .map(|metadata| metadata.k_head_dim())
                                        .unwrap_or(1)
                                        .max(1),
                                    kv_cache_manager: kv_cache_manager.clone(),
                                    prompt_chunk_size: output.prompt_chunk_size,
                                    scheduled_prompt_chunks,
                                    prompt_chunk_attention_policy,
                                    has_noncausal_mm_context: false,
                                    mm_prefix_ranges_by_seq_id: HashMap::new(),
                                    full_mm_prefix_ranges_by_seq_id: HashMap::new(),
                                    enable_packed_prefill: pipeline.supports_packed_prefill(),
                                    is_final_prompt_chunk,
                                };

                                let return_raw_logits = guards_mut[0].return_raw_logits;
                                assert!(
                                    guards_mut
                                        .iter()
                                        .all(|seq| seq.return_raw_logits == return_raw_logits),
                                    "All sequences must either return raw logits, or not."
                                );

                                pipeline
                                    .submit_step(
                                        &mut guards_mut,
                                        is_prompt,
                                        return_raw_logits,
                                        &mut *get_mut_arcmutex!(self.prefix_cacher),
                                        self.disable_eos_stop,
                                        rng.clone(),
                                        CacheBackendMetadata::PagedAttention { metadata },
                                        self.logger.as_ref(),
                                        step_lookahead,
                                    )
                                    .await
                            }
                        };

                        let submission = handle_pipeline_forward_error!(
                            "step",
                            res,
                            &mut guards_mut,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );
                        drop(guards_mut);
                        drop(guards);

                        #[cfg(feature = "cuda")]
                        if submission.cuda_has_tail() {
                            let mut scheduler = get_mut_arcmutex!(self.scheduler);
                            scheduler.record_decode_continuation();
                        }

                        let step_exec_time = match submission.into_inner() {
                            StepSubmissionKind::Ready(completion) => completion.duration(),
                            #[cfg(feature = "cuda")]
                            StepSubmissionKind::Cuda(submission) => {
                                let completion_result = self
                                    .complete_cuda_step(
                                        cuda_completion_worker.as_ref().expect(
                                            "CUDA decode submission requires a completion worker",
                                        ),
                                        submission,
                                    )
                                    .await;
                                let mut completion_guards = output
                                    .scheduled
                                    .iter()
                                    .map(|seq| seq.lock().unwrap())
                                    .collect::<Vec<_>>();
                                let mut completion_guards_mut = completion_guards
                                    .iter_mut()
                                    .map(|seq| &mut **seq)
                                    .collect::<Vec<_>>();
                                let mut completion = handle_pipeline_forward_error!(
                                    "CUDA decode completion",
                                    completion_result,
                                    &mut completion_guards_mut,
                                    self.pipeline,
                                    'lp,
                                    self.prefix_cacher
                                );

                                for ((seq, before), scheduled) in completion_guards_mut
                                    .iter_mut()
                                    .zip(num_computed_before_step.iter().copied())
                                    .zip(scheduled_token_counts.iter().copied())
                                {
                                    if seq.num_computed_tokens() == before {
                                        seq.advance_num_computed_tokens(scheduled);
                                    }
                                }
                                let commit_rows = completion_guards_mut
                                    .iter()
                                    .map(|seq| !seq.is_finished_paged_attn())
                                    .collect::<Vec<_>>();
                                let finish_result: candle_core::Result<_> = async {
                                    let pipeline = get_mut_arcmutex!(self.pipeline);
                                    if crate::pipeline::sampling::greedy_batch_will_finish(
                                        &*pipeline,
                                        &completion_guards_mut,
                                        completion.token_ids(),
                                        &commit_rows,
                                        self.disable_eos_stop,
                                    )? {
                                        completion.synchronize_tail()?;
                                    }
                                    completion
                                        .finish(
                                            &*pipeline,
                                            &mut completion_guards_mut,
                                            &commit_rows,
                                            &mut *get_mut_arcmutex!(self.prefix_cacher),
                                            self.disable_eos_stop,
                                        )
                                        .await
                                }
                                .await;
                                let completion = handle_pipeline_forward_error!(
                                    "CUDA decode finalize",
                                    finish_result,
                                    &mut completion_guards_mut,
                                    self.pipeline,
                                    'lp,
                                    self.prefix_cacher
                                );
                                let duration = run_start.elapsed();
                                let tail = completion.into_cuda_tail();
                                let any_live = completion_guards_mut
                                    .iter()
                                    .any(|seq| !seq.is_finished_paged_attn());
                                drop(completion_guards_mut);
                                drop(completion_guards);

                                if let Some(tail) = tail {
                                    if any_live {
                                        let lease_result = CudaDecodeBatchLease::new(
                                            output.scheduled.clone(),
                                            tail,
                                        );
                                        let mut error_guards = output
                                            .scheduled
                                            .iter()
                                            .map(|seq| seq.lock().unwrap())
                                            .collect::<Vec<_>>();
                                        let mut error_guards_mut = error_guards
                                            .iter_mut()
                                            .map(|seq| &mut **seq)
                                            .collect::<Vec<_>>();
                                        cuda_decode_lease = Some(handle_pipeline_forward_error!(
                                            "CUDA decode lease",
                                            lease_result,
                                            &mut error_guards_mut,
                                            self.pipeline,
                                            'lp,
                                            self.prefix_cacher
                                        ));
                                    } else {
                                        let drain_result = tail.drain();
                                        let mut error_guards = output
                                            .scheduled
                                            .iter()
                                            .map(|seq| seq.lock().unwrap())
                                            .collect::<Vec<_>>();
                                        let mut error_guards_mut = error_guards
                                            .iter_mut()
                                            .map(|seq| &mut **seq)
                                            .collect::<Vec<_>>();
                                        handle_pipeline_forward_error!(
                                            "CUDA decode tail drain",
                                            drain_result,
                                            &mut error_guards_mut,
                                            self.pipeline,
                                            'lp,
                                            self.prefix_cacher
                                        );
                                        drop(error_guards_mut);
                                        drop(error_guards);
                                        self.account_cuda_decode_rows(&output.scheduled);
                                    }
                                }
                                duration
                            }
                        };

                        let mut guards = output
                            .scheduled
                            .iter()
                            .map(|seq| seq.lock().unwrap())
                            .collect::<Vec<_>>();
                        let mut guards_mut =
                            guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
                        for ((seq, before), scheduled) in guards_mut
                            .iter_mut()
                            .zip(num_computed_before_step.iter().copied())
                            .zip(scheduled_token_counts.iter().copied())
                        {
                            if seq.num_computed_tokens() == before {
                                seq.advance_num_computed_tokens(scheduled);
                            }
                        }
                        if is_prompt && !scheduler_visible_prompt_step {
                            for seq in guards_mut.iter_mut() {
                                if !seq.is_finished_paged_attn()
                                    && matches!(
                                        seq.sequence_stepping_type(),
                                        SeqStepType::PromptAndDecode
                                    )
                                {
                                    seq.set_state(SequenceState::RunningCompletion);
                                }
                            }
                        }
                        for seq in guards_mut.iter_mut() {
                            if is_prompt {
                                seq.finish_prompt_timing(step_exec_time);
                            } else {
                                seq.finish_completion_timing(step_exec_time);
                            }
                        }

                        let total_processed_tokens: usize = scheduled_token_counts.iter().sum();
                        self.logger.add_tokens_processed(total_processed_tokens);

                        // Capture recurrent states at full-block boundaries so hybrid models can
                        // reuse recurrent prefix state when paged prefix caching hits. Prompt steps
                        // only: a chat template re-renders a finished assistant turn differently
                        // than it was generated, so no later lookup can ever reach a key past the
                        // end of a prompt. Chunked prefill snapshots inline instead, in
                        // `snapshot_paged_recurrent_prefix`.
                        {
                            let pipeline = get_mut_arcmutex!(self.pipeline);
                            let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
                            if is_prompt
                                && !scheduler_visible_prompt_step
                                && pipeline.cache().is_hybrid()
                                && prefix_cacher.accepts_paged_recurrent_prefix()
                            {
                                let hybrid_cache = pipeline.cache().hybrid();

                                for seq in guards_mut.iter() {
                                    let encoded_len = seq.num_computed_tokens();
                                    if encoded_len == 0 || encoded_len % block_size != 0 {
                                        continue;
                                    }

                                    let Some(slot_idx) = seq.recurrent_state_idx() else {
                                        continue;
                                    };

                                    let snapshots = match hybrid_cache
                                        .snapshot_recurrent_state(slot_idx)
                                    {
                                        Ok(snapshots) => snapshots,
                                        Err(e) => {
                                            tracing::warn!(
                                                    "Failed snapshotting recurrent state for sequence {}: {e}",
                                                    seq.id()
                                                );
                                            continue;
                                        }
                                    };
                                    if snapshots.is_empty() {
                                        continue;
                                    }

                                    let num_blocks = encoded_len / block_size;
                                    let adapter_key =
                                        adapter_generation_key(seq.adapter_generation());
                                    let block_hashes = compute_block_hashes(
                                        seq.get_toks(),
                                        block_size,
                                        seq.mm_features(),
                                        adapter_key.as_slice(),
                                    );
                                    if block_hashes.len() < num_blocks {
                                        continue;
                                    }
                                    prefix_cacher.add_paged_recurrent_prefix(
                                        block_hashes[..num_blocks].to_vec(),
                                        snapshots,
                                    );
                                }
                            }
                        }

                        if self.is_debug {
                            let ms_from_last_run = run_start.elapsed().as_secs_f64();
                            let total_len = guards.len();
                            if total_len > 0 {
                                let lengths = guards
                                    .iter()
                                    .map(|seq| seq.len().to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");

                                let (prompt_lengths, completion_lengths) = if is_prompt {
                                    (lengths, "".to_string())
                                } else {
                                    ("".to_string(), lengths)
                                };

                                tracing::info!(
                                    "Prompt[{}] Completion[{}] - {}ms",
                                    prompt_lengths,
                                    completion_lengths,
                                    ms_from_last_run * 1000.,
                                );
                            }
                        }
                    }
                    scheduler = get_mut_arcmutex!(self.scheduler);
                }
            }

            #[cfg(feature = "cuda")]
            if cuda_decode_lease.is_none() {
                self.free_finished_scheduler_sequences(&mut *scheduler);
            }
            #[cfg(not(feature = "cuda"))]
            self.free_finished_scheduler_sequences(&mut *scheduler);
        }
    }

    fn build_sequence_recognizer(
        factory: &Option<Arc<ParserFactory>>,
        constraint: &Constraint,
    ) -> anyhow::Result<SequenceRecognizer> {
        if let Some(grm) = llg_grammar_from_constraint(constraint)? {
            let factory = factory
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No token environment (llg_factory) found."))?;
            let llg = constraint_from_llg_grammar(factory, grm)?;
            Ok(SequenceRecognizer::Llguidance(Box::new(llg)))
        } else {
            Ok(SequenceRecognizer::None)
        }
    }

    fn replicate_request_to_daemons(&self, request: &Request) {
        if !distributed::is_daemon() && mistralrs_quant::distributed::use_nccl() {
            let name = distributed::ipc_name().unwrap();
            let num_workers =
                mistralrs_quant::distributed::get_global_tp_size_from_devices().unwrap() - 1;
            let listener = ListenerOptions::new().name(name).create_sync().unwrap();

            for _ in 0..num_workers {
                let stream = listener.accept().unwrap();
                let mut writer = BufWriter::new(stream);
                let req = format!("{}\n", serde_json::to_string(&request).unwrap());
                writer.write_all(req.as_bytes()).unwrap();
            }
        } else if !distributed::is_daemon() && cfg!(feature = "ring") {
            let num_workers =
                mistralrs_quant::distributed::get_global_tp_size_from_devices().unwrap() - 1;
            let master_port = RingConfig::load().master_port;
            let listener =
                TcpListener::bind(format!("0.0.0.0:{master_port}")).expect("bind replicator");

            for _ in 0..num_workers {
                let (stream, _) = listener.accept().unwrap();
                let mut writer = BufWriter::new(stream);
                let req = format!("{}\n", serde_json::to_string(&request).unwrap());
                writer.write_all(req.as_bytes()).unwrap();
            }
        }
    }
}
