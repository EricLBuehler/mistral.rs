use crate::{
    distributed,
    embedding::bert::BertPipeline,
    pipeline::{
        llg::{constraint_from_llg_grammar, llg_grammar_from_constraint},
        text_models_inputs_processor::PagedAttentionMeta,
        CacheBackendMetadata, CacheInstruction,
    },
    prefix_cacher::PrefixCacheManagerV2,
    response::CompletionChoice,
    scheduler::{Scheduler, SchedulerOutput},
    search,
    sequence::{SeqStepType, StopReason},
    tools, CompletionResponse, SchedulerConfig, DEBUG,
};
use interprocess::local_socket::{traits::Listener, ListenerOptions};
use llguidance::ParserFactory;
pub use logger::IntervalLogger;
use mistralrs_quant::RingConfig;
use once_cell::sync::Lazy;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    net::TcpListener,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc::Receiver, Mutex},
    task::JoinHandle,
};

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error,
    pipeline::Pipeline,
    request::Request,
    response::{ChatCompletionResponse, Choice, ResponseMessage},
    sequence::{SequenceRecognizer, SequenceState},
    Constraint,
};

mod add_request;
mod logger;
mod search_request;

pub enum EngineInstruction {
    Terminate,
}

#[derive(Debug, Default, Clone)]
/// Embedding model used for ranking web search results internally.
pub enum BertEmbeddingModel {
    #[default]
    SnowflakeArcticEmbedL,
    Custom(String),
}

const SEED: u64 = 0;
/// Terminate all sequences on the next scheduling step. Be sure to reset this.
/// This is a global flag for terminating all engines at once (e.g., Ctrl+C).
pub static TERMINATE_ALL_NEXT_STEP: AtomicBool = AtomicBool::new(false);

/// Engine-specific termination flags, per Engine thread ID.
static ENGINE_TERMINATE_FLAGS: Lazy<
    std::sync::Mutex<HashMap<std::thread::ThreadId, Arc<AtomicBool>>>,
> = Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

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
pub static ENGINE_INSTRUCTIONS: Lazy<std::sync::Mutex<HashMap<usize, Option<EngineInstruction>>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

pub struct Engine {
    rx: Arc<Mutex<Receiver<Request>>>,
    pipeline: Arc<Mutex<dyn Pipeline>>,
    bert_pipeline: Arc<Mutex<Option<BertPipeline>>>,
    search_callback: Option<Arc<search::SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    scheduler: Arc<Mutex<dyn Scheduler>>,
    id: Arc<Mutex<usize>>,
    truncate_sequence: bool,
    no_kv_cache: bool,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
    is_debug: bool,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
    logger: IntervalLogger,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
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
        rx: Receiver<Request>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        config: SchedulerConfig,
        truncate_sequence: bool,
        mut no_kv_cache: bool,
        mut no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
        throughput_logging_enabled: bool,
        search_embedding_model: Option<BertEmbeddingModel>,
        search_callback: Option<Arc<search::SearchCallback>>,
        tool_callbacks: tools::ToolCallbacks,
        tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    ) -> anyhow::Result<Self> {
        no_kv_cache |= get_mut_arcmutex!(pipeline).get_metadata().no_kv_cache;

        no_prefix_cache = no_prefix_cache
            || no_kv_cache
            || get_mut_arcmutex!(pipeline).get_metadata().no_prefix_cache
            || prefix_cache_n == 0;

        let bert_pipeline = match search_embedding_model {
            Some(search_embedding_model) => Some(BertPipeline::new(
                search_embedding_model,
                &get_mut_arcmutex!(pipeline).device(),
            )?),
            None => None,
        };

        let scheduler = config.into_scheduler();
        let block_engine = get_mut_arcmutex!(scheduler).block_engine();

        Ok(Self {
            rx: Arc::new(Mutex::new(rx)),
            pipeline,
            bert_pipeline: Arc::new(Mutex::new(bert_pipeline)),
            search_callback,
            tool_callbacks,
            tool_callbacks_with_tools,
            scheduler: scheduler.clone(),
            id: Arc::new(Mutex::new(0)),
            truncate_sequence,
            no_kv_cache,
            prefix_cacher: Arc::new(Mutex::new(PrefixCacheManagerV2::new(
                prefix_cache_n,
                no_prefix_cache,
                block_engine,
            ))),
            is_debug: DEBUG.load(Ordering::Relaxed),
            disable_eos_stop,
            throughput_logging_enabled,
            logger: IntervalLogger::new(Duration::from_secs(5)),
            handles: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub async fn run(self: Arc<Self>) {
        if self.throughput_logging_enabled {
            self.logger.enable_logging();
        }

        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        'lp: loop {
            if matches!(
                ENGINE_INSTRUCTIONS
                    .lock()
                    .expect("`ENGINE_INSTRUCTIONS` was poisoned")
                    .get(get_mut_arcmutex!(self.id).deref()),
                Some(Some(EngineInstruction::Terminate))
            ) {
                self.replicate_request_to_daemons(&Request::Terminate);
                break 'lp;
            }

            while let Ok(request) = get_mut_arcmutex!(self.rx).try_recv() {
                self.replicate_request_to_daemons(&request);
                if matches!(request, Request::Terminate) {
                    break 'lp;
                }
                self.clone().handle_request(request).await;
            }

            if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
                self.replicate_request_to_daemons(&Request::TerminateAllSeqsNextStep);
            }

            let run_start = Instant::now();
            let mut scheduler = get_mut_arcmutex!(self.scheduler);
            let scheduled = scheduler.schedule(&self.logger);

            match scheduled {
                SchedulerOutput::DefaultScheduler {
                    output: mut scheduled,
                } => {
                    if !scheduled.completion.is_empty() {
                        let current_completion_ids: Vec<usize> =
                            scheduled.completion.iter().map(|seq| *seq.id()).collect();
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

                        handle_pipeline_forward_error!(
                            "completion step",
                            res,
                            &mut scheduled.completion,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        self.logger.add_tokens_processed(scheduled.completion.len());

                        last_completion_ids = current_completion_ids;
                    }

                    if !scheduled.prompt.is_empty() {
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
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time travel has occurred!")
                                .as_millis();
                            #[allow(clippy::cast_precision_loss)]
                            let prompt_tok_per_sec =
                                seq.len() as f32 / prompt_exec_time.as_secs_f32();
                            seq.prompt_tok_per_sec = prompt_tok_per_sec;
                            seq.prompt_timestamp = Some(now);
                            seq.total_prompt_time = Some(prompt_exec_time.as_millis());
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

                            let metadata = PagedAttentionMeta {
                                block_size,
                                sliding_window: pipeline.get_metadata().sliding_window,
                                block_engine: scheduler.block_engine().unwrap(),
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
                                    CacheBackendMetadata::PagedAttention {
                                        metadata,
                                        blocks_to_copy: output.blocks_to_copy,
                                        blocks_to_swap_in: output.blocks_to_swap_in,
                                        blocks_to_swap_out: output.blocks_to_swap_out,
                                    },
                                )
                                .await
                        };

                        handle_pipeline_forward_error!(
                            "step",
                            res,
                            &mut guards_mut,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        let total_processed_tokens: usize = guards
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

                        if is_prompt {
                            for mut seq in guards {
                                let now = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Time travel has occurred!")
                                    .as_millis();
                                #[allow(clippy::cast_precision_loss)]
                                let prompt_tok_per_sec =
                                    seq.len() as f32 / (now - seq.timestamp()) as f32;
                                seq.prompt_tok_per_sec = prompt_tok_per_sec * 1000.;
                                seq.prompt_timestamp = Some(now);
                                seq.total_prompt_time = Some(now - seq.timestamp());
                            }
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
