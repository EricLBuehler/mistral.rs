use crate::{
    distributed,
    paged_attention::block_hash::{compute_block_hashes, BlockHash},
    pipeline::{
        llg::{constraint_from_llg_grammar, llg_grammar_from_constraint},
        text_models_inputs_processor::PagedAttentionMeta,
        CacheBackendMetadata, CacheInstruction,
    },
    prefix_cacher::PrefixCacheManagerV2,
    response::CompletionChoice,
    scheduler::{PagedPrefixCacheValidator, Scheduler, SchedulerOutput},
    search::{self, rag::SearchPipeline},
    sequence::{SeqStepType, StopReason},
    tools, CompletionResponse, SchedulerConfig, DEBUG,
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
    response::{ChatCompletionResponse, Choice, ResponseMessage},
    sequence::{SequenceRecognizer, SequenceState},
    Constraint,
};

mod add_request;
pub(crate) mod agentic_loop;
pub use agentic_loop::DEFAULT_MAX_TOOL_ROUNDS;
pub(crate) mod agentic_session;
mod file_tools;
mod logger;
mod tool_dispatch;

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
}

struct HybridPagedPrefixValidator {
    pipeline: Arc<Mutex<dyn Pipeline>>,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
}

impl PagedPrefixCacheValidator for HybridPagedPrefixValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        seq: &mut crate::sequence::Sequence,
        block_hashes: &[BlockHash],
        cached_tokens: usize,
        block_size: usize,
    ) -> usize {
        if cached_tokens == 0 || !cached_tokens.is_multiple_of(block_size) {
            return 0;
        }

        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return 0;
        };
        let max_blocks = cached_tokens / block_size;
        let Some((n_blocks, snapshots)) = get_mut_arcmutex!(self.prefix_cacher)
            .get_longest_paged_recurrent_prefix(block_hashes, max_blocks)
        else {
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
            0
        }
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

        let scheduler = config.into_scheduler();

        // Configure prefix caching on the scheduler based on the global no_prefix_cache flag
        // This ensures PagedAttention prefix caching respects the same setting
        get_mut_arcmutex!(scheduler).set_prefix_caching_enabled(!no_prefix_cache);

        let has_paged_attention = get_mut_arcmutex!(scheduler).kv_cache_manager().is_some();

        Ok(Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            pipeline,
            search_pipeline: Arc::new(Mutex::new(search_pipeline)),
            search_callback,
            tool_callbacks,
            scheduler: scheduler.clone(),
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

    pub async fn run(self: Arc<Self>) {
        if self.throughput_logging_enabled {
            self.logger.enable_logging();
        }

        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
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
                self.replicate_request_to_daemons(&Request::Terminate);
                break 'lp;
            }

            let mut channel_disconnected = false;
            loop {
                let next_request = {
                    let mut rx = self.rx.lock().await;
                    rx.try_recv()
                };

                match next_request {
                    Ok(request) => {
                        self.replicate_request_to_daemons(&request);
                        if matches!(request, Request::Terminate) {
                            break 'lp;
                        }
                        self.clone().handle_request(request).await;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        channel_disconnected = true;
                        break;
                    }
                }
            }

            if channel_disconnected {
                break 'lp;
            }

            let (waiting_len, running_len) = {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                (scheduler.waiting_len(), scheduler.running_len())
            };
            let scheduler_idle = waiting_len == 0 && running_len == 0;

            if scheduler_idle {
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
                        self.replicate_request_to_daemons(&request);
                        if matches!(request, Request::Terminate) {
                            break 'lp;
                        }
                        self.clone().handle_request(request).await;
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
                            match seq.sequence_stepping_type() {
                                SeqStepType::OneShot => {
                                    seq.set_state(SequenceState::Done(StopReason::GeneratedImage))
                                }
                                SeqStepType::PromptAndDecode => {
                                    seq.set_state(SequenceState::RunningCompletion)
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
                SchedulerOutput::PagedAttention { mut output } => {
                    if !output.scheduled.is_empty() {
                        let is_prompt = get_mut_arcmutex!(output.scheduled[0]).is_prompt();

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

                        let res = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);

                            let block_size = scheduler.block_size().unwrap();

                            if guards_mut.is_empty() {
                                Ok(Duration::ZERO)
                            } else {
                                let pipeline_metadata = pipeline.get_metadata();
                                let model_metadata = pipeline_metadata.model_metadata.as_ref();
                                let kv_cache_manager = scheduler.kv_cache_manager().unwrap();
                                let max_paged_context_len = {
                                    let kv_mgr = get_mut_arcmutex!(kv_cache_manager);
                                    kv_mgr.num_gpu_blocks().saturating_sub(1).max(1) * block_size
                                };
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
                                    kv_cache_manager,
                                    prompt_chunk_attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
                                    has_noncausal_mm_context: false,
                                };

                                let return_raw_logits = guards_mut[0].return_raw_logits;
                                assert!(
                                    guards_mut
                                        .iter()
                                        .all(|seq| seq.return_raw_logits == return_raw_logits),
                                    "All sequences must either return raw logits, or not."
                                );

                                pipeline
                                    .step(
                                        &mut guards_mut,
                                        is_prompt,
                                        return_raw_logits,
                                        &mut *get_mut_arcmutex!(self.prefix_cacher),
                                        self.disable_eos_stop,
                                        rng.clone(),
                                        CacheBackendMetadata::PagedAttention { metadata },
                                    )
                                    .await
                            }
                        };

                        let step_exec_time = handle_pipeline_forward_error!(
                            "step",
                            res,
                            &mut guards_mut,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );
                        for seq in guards_mut.iter_mut() {
                            if is_prompt {
                                seq.finish_prompt_timing(step_exec_time);
                            } else {
                                seq.finish_completion_timing(step_exec_time);
                            }
                        }

                        let total_processed_tokens: usize = guards_mut
                            .iter()
                            .map(|seq| {
                                if seq.is_prompt() {
                                    seq.get_toks().len()
                                } else {
                                    1
                                }
                            })
                            .sum();
                        self.logger.add_tokens_processed(total_processed_tokens);

                        // Capture recurrent states at full-block boundaries so hybrid models can
                        // reuse recurrent prefix state when paged prefix caching hits.
                        {
                            let pipeline = get_mut_arcmutex!(self.pipeline);
                            if pipeline.cache().is_hybrid() {
                                let block_size = scheduler.block_size().unwrap();
                                let hybrid_cache = pipeline.cache().hybrid();
                                let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);

                                for seq in guards_mut.iter() {
                                    let seq_len = seq.get_toks().len();
                                    if seq_len == 0 || seq_len % block_size != 0 {
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

                                    let num_blocks = seq_len / block_size;
                                    let block_hashes = compute_block_hashes(
                                        seq.get_toks(),
                                        block_size,
                                        seq.mm_features(),
                                        &[],
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
                }
            }

            // Free recurrent state pool slots for finished sequences (hybrid models)
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                    let recurrent_indices = scheduler.get_finished_recurrent_indices();
                    if !recurrent_indices.is_empty() {
                        let mut hybrid_cache = pipeline.cache().hybrid();
                        for idx in recurrent_indices {
                            hybrid_cache.free_seq(idx);
                        }
                    }
                }
            }
            scheduler.free_finished_sequence_groups();
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
