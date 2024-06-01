use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc::Receiver, Mutex};

use crate::{
    aici::{cfg::CfgParser, recognizer::StackRecognizer, rx::RecRx},
    pipeline::{AdapterInstruction, CacheInstruction},
    request::NormalRequest,
    response::CompletionChoice,
    CompletionResponse, RequestMessage, Response, DEBUG,
};
use candle_core::{Result, Tensor};
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use tracing::{info, warn};

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error, handle_seq_error,
    pipeline::Pipeline,
    prefix_cacher::PrefixCacheManager,
    request::Request,
    response::{ChatCompletionResponse, Choice, ResponseMessage},
    sampler::Sampler,
    scheduler::{Scheduler, SchedulerMethod},
    sequence::{Sequence, SequenceGroup, SequenceRecognizer, SequenceState},
    Constraint, StopTokens,
};

const SEED: u64 = 0;
/// Terminate all sequences on the next scheduling step. Be sure to reset this.
pub static TERMINATE_ALL_NEXT_STEP: AtomicBool = AtomicBool::new(false);

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Arc<Mutex<dyn Pipeline>>,
    scheduler: Scheduler<VecDeque<Sequence>>,
    id: usize,
    truncate_sequence: bool,
    no_kv_cache: bool,
    prefix_cacher: PrefixCacheManager,
    is_debug: bool,
    disable_eos_stop: bool,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: Receiver<Request>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        truncate_sequence: bool,
        no_kv_cache: bool,
        no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
    ) -> Self {
        let device = get_mut_arcmutex!(pipeline).device().clone();
        let is_xlora = get_mut_arcmutex!(pipeline).get_metadata().is_xlora;
        Self {
            rx,
            pipeline,
            scheduler: Scheduler::new(method),
            id: 0,
            truncate_sequence,
            no_kv_cache,
            prefix_cacher: PrefixCacheManager::new(
                device,
                prefix_cache_n,
                is_xlora,
                no_prefix_cache,
            ),
            is_debug: DEBUG.load(Ordering::Relaxed),
            disable_eos_stop,
        }
    }

    pub async fn run(&mut self) {
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        'lp: loop {
            while let Ok(request) = self.rx.try_recv() {
                self.handle_request(request).await;
            }
            let run_start = Instant::now();
            let mut scheduled = self.scheduler.schedule();

            if scheduled.completion.len() > 0 {
                let current_completion_ids: Vec<usize> =
                    scheduled.completion.iter().map(|seq| *seq.id()).collect();
                let res = {
                    let mut pipeline = get_mut_arcmutex!(self.pipeline);
                    let pre_op =
                        if !self.no_kv_cache && last_completion_ids != current_completion_ids {
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
                            reset_non_granular: false,
                            adapter_inst: AdapterInstruction::None,
                        }
                    };

                    pipeline
                        .step(
                            &mut scheduled.completion,
                            false,
                            &mut self.prefix_cacher,
                            self.disable_eos_stop,
                            rng.clone(),
                            pre_op,
                            post_op,
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

                last_completion_ids = current_completion_ids;
            }

            if scheduled.prompt.len() > 0 {
                let logits = {
                    let mut pipeline = get_mut_arcmutex!(self.pipeline);

                    // Run the prompt seqs
                    let post_op = if !self.no_kv_cache {
                        CacheInstruction::Out
                    } else {
                        CacheInstruction::Reset {
                            reset_non_granular: false,
                            adapter_inst: AdapterInstruction::None,
                        }
                    };
                    let adapter_inst = scheduled.prompt[0]
                        .get_adapters()
                        .map(AdapterInstruction::Activate)
                        .unwrap_or(AdapterInstruction::None);

                    // Reset non granular state because the old sequence must be dead.
                    // Technically we don't need to do this but it is better to be safe.
                    pipeline
                        .step(
                            &mut scheduled.prompt,
                            true,
                            &mut self.prefix_cacher,
                            self.disable_eos_stop,
                            rng.clone(),
                            CacheInstruction::Reset {
                                reset_non_granular: false,
                                adapter_inst,
                            },
                            post_op,
                        )
                        .await
                };

                handle_pipeline_forward_error!(
                    "prompt step",
                    logits,
                    &mut scheduled.prompt,
                    self.pipeline,
                    'lp,
                    self.prefix_cacher
                );

                for seq in scheduled.prompt.iter_mut() {
                    seq.set_state(SequenceState::RunningCompletion);
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time travel has occurred!")
                        .as_millis();
                    #[allow(clippy::cast_precision_loss)]
                    let prompt_tok_per_sec = seq.len() as f32 / (now - seq.timestamp()) as f32;
                    seq.prompt_tok_per_sec = prompt_tok_per_sec * 1000.;
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
            if scheduled.prompt.len() == 0
                && scheduled.completion.len() == 0
                && self.scheduler.waiting_len() == 0
            {
                // If there is nothing to do, sleep until a request comes in
                if let Some(request) = self.rx.recv().await {
                    self.handle_request(request).await;
                }
            }
        }
    }

    fn build_sequence_recognizer(constraint: &Constraint) -> anyhow::Result<SequenceRecognizer> {
        let recognizer = match constraint {
            Constraint::Regex(rx) => {
                SequenceRecognizer::Regex(StackRecognizer::from(RecRx::from_rx(rx)?)?.into())
            }
            Constraint::Yacc(cfg) => SequenceRecognizer::Cfg(CfgParser::from_yacc(cfg)?.into()),
            Constraint::None => SequenceRecognizer::None,
        };
        Ok(recognizer)
    }

    fn alloc_logits_bias(&self, logits_bias: Option<HashMap<u32, f32>>) -> Result<Option<Tensor>> {
        let device = get_mut_arcmutex!(self.pipeline).device().clone();
        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();
        let vocab_size = tokenizer.get_vocab_size(true);

        match logits_bias {
            Some(bias) => {
                let mut logits_bias = vec![0.0; vocab_size];
                for (k, v) in bias {
                    logits_bias[k as usize] = v;
                }
                Ok(Some(Tensor::from_vec(logits_bias, vocab_size, &device)?))
            }
            None => Ok(None),
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
            | RequestMessage::VisionChat { .. } => 1,
        };
        if is_chat
            && !get_mut_arcmutex!(self.pipeline)
                .get_chat_template()
                .has_chat_template()
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

        let mut prompt = match request.messages {
            RequestMessage::Chat(messages)
            | RequestMessage::VisionChat {
                images: _,
                messages,
            } => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let template = pipeline.get_processor().process(pipeline, messages, true);
                handle_seq_error!(template, request.response)
            }
            RequestMessage::Completion { text, .. } => {
                let prompt = get_mut_arcmutex!(self.pipeline)
                    .tokenizer()
                    .encode(text, false)
                    .map_err(|e| anyhow::Error::msg(e.to_string()));
                handle_seq_error!(prompt, request.response)
                    .get_ids()
                    .to_vec()
            }
            RequestMessage::CompletionTokens(it) => it,
        };
        if prompt.is_empty() {
            request
                .response
                .send(Response::ValidationError(
                    "Received an empty prompt.".into(),
                ))
                .await
                .expect("Expected receiver.");
            return;
        }

        if prompt.len() > get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len {
            if !self.truncate_sequence {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len).into(),
                    )).await.expect("Expected receiver.");
                return;
            } else {
                let prompt_len = prompt.len();
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
                prompt = prompt[(currently_over + sampling_max)..].to_vec();
                warn!("Prompt for request {} was {} tokens over the model maximum length. The last {} tokens were truncated to make space for generation.", request.id, currently_over, prompt_len - prompt.len());
            }
        }
        let prefill_cache = handle_seq_error!(
            self.prefix_cacher.search_for_matching_cache(&prompt),
            request.response
        );

        let topk = request
            .sampling_params
            .top_k
            .map(|x| x as i64)
            .unwrap_or(-1);
        let topp = request.sampling_params.top_p.unwrap_or(1.0);
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline)
            .get_metadata()
            .num_hidden_layers;

        let (stop_toks, stop_strings) = match request.sampling_params.stop_toks {
            None => (vec![], vec![]),
            Some(StopTokens::Ids(ref i)) => {
                let tok_trie = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    pipeline.get_metadata().tok_trie.clone()
                };
                for id in i {
                    // We can't use ` ` (space) as a stop token because other tokens like ` moon` start with a space.
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

                (i.clone(), vec![])
            }
            Some(StopTokens::Seqs(ref s)) => {
                let mut stop_toks = Vec::new();
                let mut stop_strings: Vec<String> = Vec::new();

                let (tok_trie, tokenizer) = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    let tok_trie = pipeline.get_metadata().tok_trie.clone();
                    let tokenizer = pipeline.tokenizer();
                    (tok_trie, tokenizer)
                };

                for stop_txt in s {
                    let encoded = tokenizer.encode(stop_txt.to_string(), false);
                    let toks = handle_seq_error!(encoded, request.response)
                        .get_ids()
                        .to_vec();

                    if toks.len() == 1 {
                        if tok_trie.has_extensions(tok_trie.token(toks[0])) {
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!");

        let logits_bias = match self.alloc_logits_bias(request.sampling_params.logits_bias) {
            Ok(logits_bias) => logits_bias,
            Err(err) => {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Failed creation of logits bias. {}", err).into(),
                    ))
                    .await
                    .expect("Expected receiver.");
                return;
            }
        };
        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let sampler = Sampler::new(
            Some(request.sampling_params.temperature.unwrap_or(1.0)),
            request.sampling_params.top_n_logprobs,
            tokenizer,
            request.sampling_params.frequency_penalty,
            request.sampling_params.presence_penalty,
            logits_bias,
            topk,
            topp,
        );

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
            let recognizer = match Self::build_sequence_recognizer(&request.constraint) {
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

            let seq = Sequence::new_waiting(
                prompt.clone(),
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
                    Some(
                        get_mut_arcmutex!(self.pipeline)
                            .tokenizer()
                            .decode(&prompt, false)
                            .expect("cannot decode completion tokens"),
                    )
                } else {
                    None
                },
                request.adapters.clone(),
                images.clone(),
            );
            let seq = if let Some(prefill_cache) = prefill_cache.clone() {
                seq.prefill(
                    prefill_cache.normal,
                    prefill_cache.xlora,
                    prefill_cache.toks,
                )
            } else {
                seq
            };
            self.id += 1;
            self.scheduler.add_seq(seq);
        }
    }
}
