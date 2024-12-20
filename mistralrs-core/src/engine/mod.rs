use candle_core::Tensor;
use either::Either;
use llguidance::toktrie::TokEnv;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc::Receiver, Mutex};

use crate::{
    pipeline::{
        llg::{constraint_from_llg_grammar, llg_grammar_from_constraint},
        text_models_inputs_processor::PagedAttentionMeta,
        AdapterInstruction, CacheBackendMetadata, CacheInstruction, EitherCache, NormalCache,
    },
    prefix_cacher_v2::PrefixCacheManagerV2,
    request::{DetokenizationRequest, NormalRequest, TokenizationRequest},
    response::CompletionChoice,
    scheduler::{Scheduler, SchedulerOutput},
    sequence::{SeqStepType, StopReason},
    tools::{ToolCallingMatcher, ToolChoice},
    CompletionResponse, RequestMessage, Response, SchedulerConfig, DEBUG,
};
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use tracing::{info, warn};

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error, handle_seq_error,
    pipeline::Pipeline,
    request::Request,
    response::{ChatCompletionResponse, Choice, ResponseMessage},
    sampler::Sampler,
    sequence::{Sequence, SequenceGroup, SequenceRecognizer, SequenceState},
    Constraint, StopTokens,
};

pub enum EngineInstruction {
    Terminate,
}

const SEED: u64 = 0;
/// Terminate all sequences on the next scheduling step. Be sure to reset this.
pub static TERMINATE_ALL_NEXT_STEP: AtomicBool = AtomicBool::new(false);

/// Engine instructions, per Engine (MistralRs) ID.
pub static ENGINE_INSTRUCTIONS: Lazy<std::sync::Mutex<HashMap<usize, Option<EngineInstruction>>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Arc<Mutex<dyn Pipeline>>,
    scheduler: Box<dyn Scheduler>,
    id: usize,
    truncate_sequence: bool,
    no_kv_cache: bool,
    prefix_cacher: PrefixCacheManagerV2,
    is_debug: bool,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: Receiver<Request>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        config: SchedulerConfig,
        truncate_sequence: bool,
        no_kv_cache: bool,
        no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
        throughput_logging_enabled: bool,
    ) -> Self {
        let device = get_mut_arcmutex!(pipeline).device().clone();
        let has_no_kv_cache = get_mut_arcmutex!(pipeline).get_metadata().has_no_kv_cache;
        if no_kv_cache {
            // Diffusion models...
            assert_eq!(has_no_kv_cache, no_kv_cache);
        }
        // Prefix caching is always disabled if using PagedAttention for now.
        // TODO
        let no_prefix_cache = matches!(config, SchedulerConfig::PagedAttentionMeta { .. })
            || no_prefix_cache
            || has_no_kv_cache;
        Self {
            rx,
            pipeline,
            scheduler: config.into_scheduler(),
            id: 0,
            truncate_sequence,
            no_kv_cache: no_kv_cache & !has_no_kv_cache,
            prefix_cacher: PrefixCacheManagerV2::new(device, prefix_cache_n, no_prefix_cache),
            is_debug: DEBUG.load(Ordering::Relaxed),
            disable_eos_stop,
            throughput_logging_enabled,
        }
    }

    pub async fn run(&mut self) {
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        'lp: loop {
            if matches!(
                ENGINE_INSTRUCTIONS
                    .lock()
                    .expect("`ENGINE_INSTRUCTIONS` was poisioned")
                    .get(&self.id),
                Some(Some(EngineInstruction::Terminate))
            ) {
                break 'lp;
            }

            while let Ok(request) = self.rx.try_recv() {
                if matches!(request, Request::Terminate) {
                    break 'lp;
                }
                self.handle_request(request).await;
            }
            let run_start = Instant::now();
            let scheduled = self.scheduler.schedule();

            match scheduled {
                SchedulerOutput::DefaultScheduler {
                    output: mut scheduled,
                } => {
                    let mut prompt_ts = None;
                    let mut completion_ts = None;
                    if scheduled.completion.len() > 0 {
                        let throughput_start = Instant::now();
                        let current_completion_ids: Vec<usize> =
                            scheduled.completion.iter().map(|seq| *seq.id()).collect();
                        let res = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);
                            let pre_op = if !self.no_kv_cache
                                && last_completion_ids != current_completion_ids
                            {
                                CacheInstruction::In(
                                    scheduled.completion[0]
                                        .get_adapters()
                                        .map(AdapterInstruction::Activate)
                                        .unwrap_or(AdapterInstruction::None),
                                )
                            } else {
                                CacheInstruction::Nothing(
                                    scheduled.completion[0]
                                        .get_adapters()
                                        .map(AdapterInstruction::Activate)
                                        .unwrap_or(AdapterInstruction::None),
                                )
                            };
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                    adapter_inst: AdapterInstruction::None,
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
                                    &mut self.prefix_cacher,
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

                        let throughput_end = Instant::now();
                        #[allow(clippy::cast_precision_loss)]
                        if self.throughput_logging_enabled {
                            completion_ts = Some(
                                scheduled.completion.len() as f64
                                    / throughput_end
                                        .duration_since(throughput_start)
                                        .as_secs_f64(),
                            );
                        }

                        last_completion_ids = current_completion_ids;
                    }

                    if scheduled.prompt.len() > 0 {
                        let prompt_exec_time = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);

                            // Run the prompt seqs
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                    adapter_inst: AdapterInstruction::None,
                                }
                            };
                            let adapter_inst = scheduled.prompt[0]
                                .get_adapters()
                                .map(AdapterInstruction::Activate)
                                .unwrap_or(AdapterInstruction::None);

                            let return_raw_logits = scheduled.prompt[0].return_raw_logits;
                            assert!(
                                scheduled
                                    .prompt
                                    .iter()
                                    .all(|seq| seq.return_raw_logits == return_raw_logits),
                                "All sequences must either return raw logits, or not."
                            );

                            // Reset non granular state because the old sequence must be dead.
                            // Technically we don't need to do this but it is better to be safe.
                            pipeline
                                .step(
                                    &mut scheduled.prompt,
                                    true,
                                    return_raw_logits,
                                    &mut self.prefix_cacher,
                                    self.disable_eos_stop,
                                    rng.clone(),
                                    CacheBackendMetadata::DefaultInstructions {
                                        pre_op: CacheInstruction::Reset {
                                            load_preallocated_cache: true,
                                            reset_non_granular: false,
                                            adapter_inst,
                                        },
                                        post_op,
                                    },
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

                        #[allow(clippy::cast_precision_loss)]
                        if self.throughput_logging_enabled {
                            prompt_ts = Some(
                                scheduled
                                    .prompt
                                    .iter()
                                    .map(|seq| seq.get_toks().len())
                                    .sum::<usize>() as f64
                                    / prompt_exec_time.as_secs_f64(),
                            );
                        }

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

                    if self.throughput_logging_enabled {
                        match (prompt_ts, completion_ts) {
                            (Some(prompt), Some(completion)) => {
                                info!("Throughput (scheduler V1): Prompt: {prompt} T/s Completion {completion} T/s");
                            }
                            (None, Some(completion)) => {
                                info!("Throughput (scheduler V1): Completion {completion} T/s");
                            }
                            (Some(prompt), None) => {
                                info!("Throughput (scheduler V1): Prompt: {prompt} T/s");
                            }
                            (None, None) => (),
                        }
                    }

                    if scheduled.prompt.len() == 0
                        && scheduled.completion.len() == 0
                        && self.scheduler.waiting_len() == 0
                    {
                        // If there is nothing to do, sleep until a request comes in
                        if let Some(request) = self.rx.recv().await {
                            if matches!(request, Request::Terminate) {
                                break 'lp;
                            }
                            self.handle_request(request).await;
                        }
                    }
                }
                SchedulerOutput::PagedAttention { mut output } => {
                    if !output.scheduled.is_empty() {
                        let throughput_start = Instant::now();

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

                            let block_size = self.scheduler.block_size().unwrap();

                            let metadata = PagedAttentionMeta {
                                block_size,
                                sliding_window: pipeline.get_metadata().sliding_window,
                                block_engine: self.scheduler.block_engine().unwrap(),
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
                                    &mut self.prefix_cacher,
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

                        let throughput_end = Instant::now();
                        #[allow(clippy::cast_precision_loss)]
                        if self.throughput_logging_enabled {
                            let n_toks = if is_prompt {
                                guards.iter().map(|seq| seq.get_toks().len()).sum::<usize>()
                            } else {
                                guards.len()
                            };
                            let ts = n_toks as f64
                                / throughput_end
                                    .duration_since(throughput_start)
                                    .as_secs_f64();
                            info!("Throughput (scheduler V2): {ts} T/s");
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
                            }
                        }
                    }
                }
            }

            self.scheduler.free_finished_sequence_groups();
        }
    }

    fn build_sequence_recognizer(
        tok_env: &Option<TokEnv>,
        constraint: &Constraint,
    ) -> anyhow::Result<SequenceRecognizer> {
        if let Some(grm) = llg_grammar_from_constraint(constraint)? {
            let tok_env = tok_env
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No token environment found."))?;
            let llg = constraint_from_llg_grammar(tok_env.clone(), grm)?;
            Ok(SequenceRecognizer::Llguidance(Box::new(llg)))
        } else {
            Ok(SequenceRecognizer::None)
        }
    }

    async fn handle_request(&mut self, request: Request) {
        match request {
            Request::ActivateAdapters(adapters) => {
                match get_mut_arcmutex!(self.pipeline).activate_adapters(adapters) {
                    Ok(n) => info!("Swapped adapters in {n} LoRA layers."),
                    Err(e) => warn!("Adapter activation failed: {e:?}"),
                }
            }
            Request::Normal(request) => self.add_request(request).await,
            Request::ReIsq(level) => {
                if let Err(e) = get_mut_arcmutex!(self.pipeline).re_isq_model(level) {
                    warn!("ISQ requantization failed: {e:?}");
                }
            }
            Request::Tokenize(req) => self.tokenize_text(req).await,
            Request::Detokenize(req) => self.detokenize_text(req).await,
            Request::Terminate => panic!("This is unreachable in `handle_request`. Termination is handled in the `run` loop."),
        }
    }

    async fn add_request(&mut self, request: NormalRequest) {
        let is_chat = matches!(
            request.messages,
            RequestMessage::Chat(_) | RequestMessage::VisionChat { .. }
        );
        let echo_prompt = matches!(
            request.messages,
            RequestMessage::Completion {
                echo_prompt: true,
                ..
            }
        );

        let best_of = match request.messages {
            RequestMessage::Completion { best_of, .. } => best_of,
            RequestMessage::Chat(_)
            | RequestMessage::CompletionTokens(_)
            | RequestMessage::VisionChat { .. }
            | RequestMessage::ImageGeneration { .. } => None,
        };
        if is_chat
            && !get_mut_arcmutex!(self.pipeline)
                .get_chat_template()
                .as_ref()
                .is_some_and(|ch_t| ch_t.has_chat_template())
        {
            request
                    .response
                    .send(Response::ValidationError(
                        "Received messages for a model which does not have a chat template. Either use a different model or pass a single string as the prompt".into(),
                    )).await.expect("Expected receiver.");
            return;
        }

        let images = match request.messages {
            RequestMessage::VisionChat {
                ref images,
                messages: _,
            } => Some(images.clone()),
            _ => None,
        };

        let matcher = if request.tools.is_some() {
            Some(Arc::new(handle_seq_error!(
                ToolCallingMatcher::new(request.tool_choice.unwrap_or(ToolChoice::Auto),),
                request.response
            )))
        } else {
            None
        };

        let image_generation_format = match &request.messages {
            RequestMessage::ImageGeneration { format, .. } => Some(*format),
            _ => None,
        };

        let seq_step_type = match &request.messages {
            RequestMessage::ImageGeneration { .. } => SeqStepType::OneShot,
            _ => SeqStepType::PromptAndDecode,
        };

        let diffusion_params = match &request.messages {
            RequestMessage::ImageGeneration {
                generation_params, ..
            } => Some(generation_params.clone()),
            _ => None,
        };

        let (mut prompt_tokens, prompt_text) = match request.messages {
            RequestMessage::Chat(messages)
            | RequestMessage::VisionChat {
                images: _,
                messages,
            } => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    true,
                    true,
                    request.tools.unwrap_or_default(),
                );
                handle_seq_error!(template, request.response)
            }
            RequestMessage::Completion { text, .. } => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    request
                        .response
                        .send(Response::ValidationError(
                            "Completion requests require the pipeline to have a tokenizer".into(),
                        ))
                        .await
                        .expect("Expected receiver.");
                    return;
                };
                let prompt = tokenizer
                    .encode(text.clone(), true)
                    .map_err(anyhow::Error::msg);
                (
                    handle_seq_error!(prompt, request.response)
                        .get_ids()
                        .to_vec(),
                    text,
                )
            }
            RequestMessage::ImageGeneration { prompt, .. } => (vec![u32::MAX], prompt),
            RequestMessage::CompletionTokens(it) => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    request
                        .response
                        .send(Response::ValidationError(
                            "Completion requests w/ raw tokens require the pipeline to have a tokenizer".into(),
                        ))
                        .await
                        .expect("Expected receiver.");
                    return;
                };
                let prompt = tokenizer
                    .decode(&it, false)
                    .map_err(|e| anyhow::Error::msg(e.to_string()));
                (it, handle_seq_error!(prompt, request.response))
            }
        };
        if prompt_tokens.is_empty() {
            request
                .response
                .send(Response::ValidationError(
                    "Received an empty prompt.".into(),
                ))
                .await
                .expect("Expected receiver.");
            return;
        }

        if prompt_tokens.len() > get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len {
            if !self.truncate_sequence {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len).into(),
                    )).await.expect("Expected receiver.");
                return;
            } else {
                let prompt_len = prompt_tokens.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len;
                let currently_over = prompt_len - max_len;
                let sampling_max = if let Some(sampling_max) = request.sampling_params.max_len {
                    if currently_over + sampling_max >= prompt_len {
                        10
                    } else {
                        sampling_max
                    }
                } else {
                    10
                };
                prompt_tokens = prompt_tokens[(currently_over + sampling_max)..].to_vec();
                warn!("Prompt for request {} was {} tokens over the model maximum length. The last {} tokens were truncated to make space for generation.", request.id, currently_over, prompt_len - prompt_tokens.len());
            }
        }
        let prefill_cache = handle_seq_error!(
            self.prefix_cacher.search_for_matching_cache(&prompt_tokens),
            request.response
        );

        let topk = request
            .sampling_params
            .top_k
            .map(|x| x as i64)
            .unwrap_or(-1);
        let topp = request.sampling_params.top_p.unwrap_or(1.0);
        let minp = request.sampling_params.min_p.unwrap_or(0.0);
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline)
            .get_metadata()
            .num_hidden_layers;

        let (stop_toks, stop_strings) = match request.sampling_params.stop_toks {
            None => (vec![], vec![]),
            Some(StopTokens::Ids(ref i)) => {
                let tok_env = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    pipeline.get_metadata().tok_env.clone()
                };
                for id in i {
                    // We can't use ` ` (space) as a stop token because other tokens like ` moon` start with a space.
                    if let Some(tok_env) = tok_env.as_ref() {
                        let tok_trie = tok_env.tok_trie();
                        if tok_trie.has_extensions(tok_trie.token(*id)) {
                            request
                                .response
                                .send(Response::ValidationError(
                                    format!("Stop token {:?} is also a prefix of other tokens and cannot be used as a stop token.", tok_trie.token_str(*id)).into(),
                                ))
                                .await .expect("Expected receiver.");
                            return;
                        }
                    }
                }

                (i.clone(), vec![])
            }
            Some(StopTokens::Seqs(ref s)) => {
                let mut stop_toks = Vec::new();
                let mut stop_strings: Vec<String> = Vec::new();

                let (tok_env, tokenizer) = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    let tok_env = pipeline.get_metadata().tok_env.clone();
                    let tokenizer = pipeline.tokenizer();
                    (tok_env, tokenizer)
                };

                for stop_txt in s {
                    let Some(tokenizer) = &tokenizer else {
                        request
                            .response
                            .send(Response::ValidationError(
                                "Completion requests require the pipeline to have a tokenizer"
                                    .into(),
                            ))
                            .await
                            .expect("Expected receiver.");
                        return;
                    };
                    let encoded = tokenizer.encode(stop_txt.to_string(), true);
                    let toks = handle_seq_error!(encoded, request.response)
                        .get_ids()
                        .to_vec();

                    if toks.len() == 1 {
                        if tok_env.as_ref().is_some_and(|tok_env| {
                            let tok_trie = tok_env.tok_trie();
                            tok_trie.has_extensions(tok_trie.token(toks[0]))
                        }) {
                            stop_strings.push(stop_txt.clone());
                        } else {
                            stop_toks.push(toks[0]);
                        }
                    } else {
                        stop_strings.push(stop_txt.clone());
                    }
                }

                (stop_toks, stop_strings)
            }
        };

        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            request.sampling_params.n_choices,
            request.is_streaming,
            is_chat,
            best_of,
        )));

        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let sampler = Sampler::new(
            Some(request.sampling_params.temperature.unwrap_or(1.0)),
            request.sampling_params.top_n_logprobs,
            tokenizer,
            request.sampling_params.frequency_penalty,
            request.sampling_params.presence_penalty,
            request.sampling_params.dry_params,
            topk,
            topp,
            minp,
            request.logits_processors.unwrap_or_default(),
        );
        let sampler = handle_seq_error!(sampler, request.response);

        if request.sampling_params.n_choices == 0 {
            request
                .response
                .send(Response::ValidationError(
                    "Number of choices must be greater than 0.".into(),
                ))
                .await
                .expect("Expected receiver.");
            return;
        }

        // Add sequences
        for response_index in 0..request.sampling_params.n_choices {
            let trie = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .tok_env
                .clone();
            let recognizer = match Self::build_sequence_recognizer(&trie, &request.constraint) {
                Ok(recognizer) => recognizer,
                Err(err) => {
                    request
                        .response
                        .send(Response::ValidationError(
                            format!("Invalid grammar. {}", err).into(),
                        ))
                        .await
                        .expect("Expected receiver.");
                    return;
                }
            };

            let block_size = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .cache_config
                .clone()
                .map(|conf| conf.block_size);

            let cache = get_mut_arcmutex!(self.pipeline).cache().clone();
            let seq_preallocated_cache = if let EitherCache::Normal(_cache) = cache {
                let metadata = get_mut_arcmutex!(self.pipeline).get_metadata();
                let model_metadata = metadata
                    .model_metadata
                    .as_ref()
                    .expect("If a model has a NormalCache it must have a model metadata");
                let n_tokens = prompt_tokens.len();
                let required_blocks = n_tokens.div_ceil(NormalCache::CACHE_GROW_SIZE);
                let max_seq_len = required_blocks * NormalCache::CACHE_GROW_SIZE;
                let kv_shape = (
                    1usize,
                    model_metadata.num_kv_heads(),
                    max_seq_len,
                    model_metadata.head_dim(),
                );
                let dtype = get_mut_arcmutex!(self.pipeline)
                    .get_metadata()
                    .activation_dtype;
                let seq_cache =
                    Tensor::zeros(kv_shape, dtype, &get_mut_arcmutex!(self.pipeline).device());
                let seq_cache = match seq_cache {
                    Ok(x) => x,
                    Err(_) => {
                        request
                            .response
                            .send(Response::InternalError(
                                "Failed to allocate preallocated KV cache."
                                    .to_string()
                                    .into(),
                            ))
                            .await
                            .expect("Expected receiver.");
                        return;
                    }
                };
                Some(seq_cache)
            } else {
                None
            };

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!");
            let seq = Sequence::new_waiting(
                prompt_tokens.clone(),
                prompt_text.clone(),
                self.id,
                now.as_millis(),
                num_hidden_layers,
                request.response.clone(),
                sampler.clone(),
                stop_toks.clone(),
                stop_strings.clone(),
                request.sampling_params.max_len,
                request.return_logprobs,
                get_mut_arcmutex!(self.pipeline).get_metadata().is_xlora,
                group.clone(),
                response_index,
                now.as_secs(),
                recognizer,
                request.suffix.clone(),
                if echo_prompt {
                    Some(prompt_text.clone())
                } else {
                    None
                },
                request.adapters.clone(),
                images.clone(),
                block_size,
                matcher.clone(),
                image_generation_format,
                seq_step_type,
                diffusion_params.clone(),
                seq_preallocated_cache,
                request.return_raw_logits,
            );
            let seq = if let Some(prefill_cache) = prefill_cache.clone() {
                seq.prefill_v2(
                    prefill_cache.normal,
                    prefill_cache.toks,
                    prefill_cache.offset,
                )
            } else {
                seq
            };
            self.id += 1;
            self.scheduler.add_seq(seq);
        }
    }

    async fn tokenize_text(&self, request: TokenizationRequest) {
        match request.text {
            Either::Left(messages) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    request.add_generation_prompt,
                    request.add_special_tokens,
                    request.tools.unwrap_or_default(),
                );
                let toks = match template {
                    Ok((toks, _)) => toks,
                    Err(e) => {
                        request
                            .response
                            .send(Err(e))
                            .await
                            .expect("Expected receiver.");
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks))
                    .await
                    .expect("Sender disconnected unexpectedly!");
            }
            Either::Right(text) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tokenizer = pipeline.tokenizer();
                let tokenizer = match tokenizer {
                    Some(tokenizer) => tokenizer,
                    None => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(
                                "Pipeline does not include a toksnizer.",
                            )))
                            .await
                            .expect("Expected receiver.");
                        return;
                    }
                };
                let toks = tokenizer.encode(text, request.add_special_tokens);
                let toks = match toks {
                    Ok(tokenizer) => tokenizer,
                    Err(e) => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(e)))
                            .await
                            .expect("Expected receiver.");
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks.get_ids().to_vec()))
                    .await
                    .expect("Sender disconnected unexpectedly!");
            }
        };
    }

    async fn detokenize_text(&self, request: DetokenizationRequest) {
        let pipeline = &*get_mut_arcmutex!(self.pipeline);
        let tokenizer = pipeline.tokenizer();
        let tokenizer = match tokenizer {
            Some(tokenizer) => tokenizer,
            None => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(
                        "Pipeline does not include a toksnizer.",
                    )))
                    .await
                    .expect("Expected receiver.");
                return;
            }
        };
        let txt = tokenizer.decode(&request.tokens, request.skip_special_tokens);
        let txt = match txt {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(e)))
                    .await
                    .expect("Expected receiver.");
                return;
            }
        };
        request
            .response
            .send(Ok(txt))
            .await
            .expect("Sender disconnected unexpectedly!");
    }
}
