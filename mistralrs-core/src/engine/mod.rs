use std::{
    collections::{HashMap, VecDeque},
    iter::zip,
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::Receiver;

use crate::{
    aici::{cfg::CfgParser, recognizer::StackRecognizer, rx::RecRx},
    handle_seq_error_ok, handle_seq_error_stateaware_ok,
    pipeline::sample_sequence,
    response::CompletionChoice,
    CompletionResponse, RequestMessage,
};
use candle_core::{quantized::GgmlDType, Device, Result, Tensor};
use futures::future;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use tracing::info;

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error, handle_seq_error,
    pipeline::Pipeline,
    prefix_cacher::PrefixCacheManager,
    request::Request,
    response::{
        ChatCompletionResponse, Choice, ChunkChoice, Delta, Logprobs, Response, ResponseLogprob,
        ResponseMessage, SYSTEM_FINGERPRINT,
    },
    sampler::Sampler,
    scheduler::{Scheduler, SchedulerMethod},
    sequence::{Sequence, SequenceGroup, SequenceRecognizer, SequenceState, StopReason},
    Constraint, StopTokens,
};

const SEED: u64 = 0;

pub struct Engine {
    rx: Receiver<Request>,
    isq_rx: Receiver<GgmlDType>,
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
        isq_rx: Receiver<GgmlDType>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        truncate_sequence: bool,
        no_kv_cache: bool,
        no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
    ) -> Self {
        let device = get_mut_arcmutex!(pipeline).device().clone();
        let is_xlora = get_mut_arcmutex!(pipeline).is_xlora();
        Self {
            rx,
            isq_rx,
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
            is_debug: std::env::var("RUST_LOG")
                .unwrap_or_default()
                .contains("debug"),
            disable_eos_stop,
        }
    }

    pub async fn run(&mut self) {
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        'lp: loop {
            while let Ok(request) = self.rx.try_recv() {
                self.add_request(request).await;
            }
            let run_start = Instant::now();
            let mut scheduled = self.scheduler.schedule();
            if let Ok(dtype) = self.isq_rx.try_recv() {
                if let Err(e) = get_mut_arcmutex!(self.pipeline).re_isq_model(dtype) {
                    info!("⚠️ WARNING: ISQ requantization failed: {e:?}");
                }
            }

            if scheduled.completion.len() > 0 {
                let current_completion_ids: Vec<usize> =
                    scheduled.completion.iter().map(|seq| *seq.id()).collect();
                let logits = {
                    let logits = {
                        let mut pipeline = get_mut_arcmutex!(self.pipeline);
                        // Run the completion seqs
                        // Run the completion seqs
                        if !self.no_kv_cache && last_completion_ids != current_completion_ids {
                            Self::clone_in_cache(&mut *pipeline, &mut scheduled.completion);
                        }
                        pipeline.forward(&mut scheduled.completion, false)
                    };
                    let logits = handle_pipeline_forward_error!(
                        "completion",
                        logits,
                        &mut scheduled.completion,
                        self.pipeline,
                        'lp,
                        self.prefix_cacher
                    );

                    {
                        let mut pipeline = get_mut_arcmutex!(self.pipeline);
                        if !self.no_kv_cache {
                            Self::clone_out_cache(&mut *pipeline, &mut scheduled.completion);
                        } else {
                            Self::set_none_cache(&mut *pipeline);
                        }
                    }
                    logits
                };

                let sampled_result = Self::sample_seqs(
                    self.pipeline.clone(),
                    &mut scheduled.completion,
                    logits,
                    &mut self.prefix_cacher,
                    self.disable_eos_stop,
                    rng.clone(),
                )
                .await;
                handle_pipeline_forward_error!(
                    "sampling",
                    sampled_result,
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
                    Self::set_none_cache(&mut *pipeline);
                    pipeline.forward(&mut scheduled.prompt, true)
                };
                let logits = handle_pipeline_forward_error!(
                    "prompt",
                    logits,
                    &mut scheduled.prompt,
                    self.pipeline,
                    'lp,
                    self.prefix_cacher
                );

                {
                    let mut pipeline = get_mut_arcmutex!(self.pipeline);
                    if !self.no_kv_cache {
                        Self::clone_out_cache(&mut *pipeline, &mut scheduled.prompt);
                    } else {
                        Self::set_none_cache(&mut *pipeline);
                    }
                }

                let sampled_result = Self::sample_seqs(
                    self.pipeline.clone(),
                    &mut scheduled.prompt,
                    logits,
                    &mut self.prefix_cacher,
                    self.disable_eos_stop,
                    rng.clone(),
                )
                .await;
                handle_pipeline_forward_error!(
                    "sampling",
                    sampled_result,
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
                let ms_from_last_run = run_start.elapsed().as_millis();
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
                        ms_from_last_run
                    );
                }
            }
            if scheduled.prompt.len() == 0
                && scheduled.completion.len() == 0
                && self.scheduler.waiting_len() == 0
            {
                // If there is nothing to do, sleep until a request comes in
                if let Some(request) = self.rx.recv().await {
                    self.add_request(request).await;
                }
            }
        }
    }

    async fn sample_seqs(
        pipeline: Arc<Mutex<dyn Pipeline>>,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<()> {
        let seqs_len = seqs.len();
        let logits_seq = logits.to_device(&Device::Cpu)?.chunk(seqs_len, 0)?;
        debug_assert_eq!(logits_seq.len(), seqs_len);

        let (tok_trie, repeat_last_n, eos_tok, max_seq_len, pipeline_name) = {
            let pipeline = get_mut_arcmutex!(pipeline);
            let repeat_last_n = pipeline.get_repeat_last_n();
            let tok_trie = pipeline.tok_trie();
            let eos_tok = pipeline.eos_tok().to_vec();
            let max_seq_len = pipeline.get_max_seq_len();
            let pipeline_name = pipeline.name();
            (tok_trie, repeat_last_n, eos_tok, max_seq_len, pipeline_name)
        };

        let use_async_pool = seqs_len > 1;

        let sampling_futures: Vec<_> = zip(logits_seq, seqs.iter_mut())
            .map(|(logits_per_seq, seq)| {
                let return_logprobs = seq.return_logprobs();
                sample_sequence(
                    logits_per_seq,
                    seq,
                    return_logprobs,
                    repeat_last_n,
                    tok_trie.clone(),
                    rng.clone(),
                    use_async_pool,
                )
            })
            .collect();
        let sampled_vec = future::join_all(sampling_futures).await;

        for (sampled, seq) in zip(sampled_vec, seqs.iter_mut()) {
            let next_token = handle_seq_error_stateaware_ok!(sampled, seq);
            let next_token_id = next_token.token;

            let eos_tok = if disable_eos_stop {
                None
            } else {
                Some(eos_tok.as_ref())
            };
            let is_done = seq.is_done(next_token_id, eos_tok, max_seq_len);
            seq.add_token(
                next_token.clone(),
                tok_trie.decode(&[next_token_id]),
                &is_done,
            );
            // Handle streaming requests
            if seq.get_mut_group().is_streaming && seq.get_mut_group().is_chat {
                let token_index = seq.get_toks().len();
                let rate_limit_allowed = is_done.is_some() || token_index % 3 == 0;

                if rate_limit_allowed {
                    if let Some(delta) = handle_seq_error_ok!(seq.get_delta(), seq.responder()) {
                        seq.add_streaming_chunk_choice_to_group(ChunkChoice {
                            delta: Delta {
                                content: delta.clone(),
                                role: "assistant".to_string(),
                            },
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|x| x.to_string()),
                            logprobs: if seq.return_logprobs() {
                                Some(ResponseLogprob {
                                    token: delta,
                                    bytes: next_token.bytes.clone().into_bytes(),
                                    logprob: next_token.logprob,
                                    top_logprobs: next_token.top_logprobs.unwrap().clone(),
                                })
                            } else {
                                None
                            },
                        });

                        if let Some(reason) = is_done {
                            prefix_cacher.add_sequence(seq);
                            prefix_cacher.evict_to_cpu()?;
                            seq.set_state(SequenceState::Done(reason));
                            get_mut_arcmutex!(pipeline).reset_non_granular_state();
                        }

                        if seq
                            .get_mut_group()
                            .maybe_send_streaming_response(seq, pipeline_name.clone())
                            .await
                            .is_err()
                        {
                            // If we can't send the response, cancel the sequence
                            seq.set_state(SequenceState::Done(StopReason::Canceled));
                            get_mut_arcmutex!(pipeline).reset_non_granular_state();
                        }
                    }
                }
            } else if let Some(reason) = is_done {
                Self::finish_seq(pipeline.clone(), seq, reason, prefix_cacher).await?;
                get_mut_arcmutex!(pipeline).reset_non_granular_state();
            }
        }

        Ok(())
    }

    async fn finish_seq(
        pipeline: Arc<Mutex<dyn Pipeline>>,
        seq: &mut Sequence,
        reason: StopReason,
        prefix_cacher: &mut PrefixCacheManager,
    ) -> Result<()> {
        seq.set_state(SequenceState::Done(reason));
        let (tokenizer, pipeline_name) = {
            let pipeline = get_mut_arcmutex!(pipeline);
            let pipeline_name = pipeline.name();
            let tokenizer = pipeline.tokenizer();
            (tokenizer, pipeline_name)
        };

        let logprobs = if seq.return_logprobs() {
            let mut logprobs = Vec::new();
            for logprob in seq.logprobs() {
                let resp_logprob = ResponseLogprob {
                    token: handle_seq_error_ok!(
                        tokenizer.decode(&[logprob.token], false),
                        seq.responder()
                    ),
                    bytes: logprob.bytes.clone().into_bytes(),
                    logprob: logprob.logprob,
                    top_logprobs: logprob.top_logprobs.clone().unwrap(),
                };
                logprobs.push(resp_logprob);
            }
            Some(logprobs)
        } else {
            None
        };

        let text = match reason {
            StopReason::Length(_)
            | StopReason::ModelLength(_)
            | StopReason::Eos
            | StopReason::StopTok(_)
            | StopReason::Canceled => String::from_utf8_lossy(seq.completion_bytes())
                .trim_start()
                .to_string(),
            StopReason::StopString {
                completion_bytes_pos,
                ..
            } => {
                let txt = String::from_utf8_lossy(seq.completion_bytes());
                txt[..completion_bytes_pos].trim_start().to_string()
            }
        };

        if seq.get_mut_group().is_chat {
            let choice = Choice {
                finish_reason: reason.to_string(),
                index: seq.get_response_index(),
                message: ResponseMessage {
                    content: text,
                    role: "assistant".to_string(),
                },
                logprobs: logprobs.map(|l| Logprobs { content: Some(l) }),
            };
            seq.add_choice_to_group(choice);
        } else {
            let choice = CompletionChoice {
                finish_reason: reason.to_string(),
                index: seq.get_response_index(),
                text,
                logprobs: None,
            };
            seq.add_completion_choice_to_group(choice);
        }

        prefix_cacher.add_sequence(seq);
        prefix_cacher.evict_to_cpu()?;

        let group = seq.get_mut_group();
        if group.is_chat {
            group
                .maybe_send_done_response(
                    ChatCompletionResponse {
                        id: seq.id().to_string(),
                        choices: group.get_choices().to_vec(),
                        created: seq.creation_time(),
                        model: pipeline_name,
                        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                        object: "chat.completion".to_string(),
                        usage: group.get_usage(),
                    },
                    seq.responder(),
                )
                .await
                .map_err(candle_core::Error::msg)?;
        } else {
            group
                .maybe_send_completion_done_response(
                    CompletionResponse {
                        id: seq.id().to_string(),
                        choices: group.get_completion_choices().to_vec(),
                        created: seq.creation_time(),
                        model: pipeline_name,
                        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                        object: "text_completion".to_string(),
                        usage: group.get_usage(),
                    },
                    seq.responder(),
                )
                .await
                .map_err(candle_core::Error::msg)?;
        }

        Ok(())
    }

    /// Clone the cache FROM the sequences' cache TO the model cache. Only used for completion seqs.
    fn clone_in_cache(pipeline: &mut dyn Pipeline, seqs: &mut [&mut Sequence]) {
        let mut new_cache = Vec::new();
        for layer in 0..pipeline.num_hidden_layers() {
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();
            for seq in &mut *seqs {
                let seq_cache = &*seq.cache();
                let cache = seq_cache.get(layer).unwrap();
                let cache = cache
                    .as_ref()
                    .expect("Not handling completions in `clone_in_cache`.");
                k_vec.push(cache.0.clone());
                v_vec.push(cache.1.clone());
            }
            new_cache.push(Some((
                if k_vec.len() > 1 {
                    Tensor::cat(&k_vec, 0).unwrap()
                } else {
                    k_vec[0].clone()
                },
                if v_vec.len() > 1 {
                    Tensor::cat(&v_vec, 0).unwrap()
                } else {
                    v_vec[0].clone()
                },
            )));
        }
        if pipeline.is_xlora() && !pipeline.has_no_kv_cache() {
            let mut new_cache = Vec::new();
            for layer in 0..pipeline.num_hidden_layers() {
                let mut k_vec = Vec::new();
                let mut v_vec = Vec::new();
                for seq in &mut *seqs {
                    let seq_cache = &*seq.xlora_cache();
                    let cache = seq_cache.get(layer).unwrap();
                    let cache = cache
                        .as_ref()
                        .expect("Not handling completions in `clone_in_cache`.");
                    k_vec.push(cache.0.clone());
                    v_vec.push(cache.1.clone());
                }
                new_cache.push(Some((
                    if k_vec.len() > 1 {
                        Tensor::cat(&k_vec, 0).unwrap()
                    } else {
                        k_vec[0].clone()
                    },
                    if v_vec.len() > 1 {
                        Tensor::cat(&v_vec, 0).unwrap()
                    } else {
                        v_vec[0].clone()
                    },
                )));
            }
            *pipeline.cache().xlora_lock() = new_cache;
        }
        if pipeline.is_xlora() {
            *pipeline.cache().get_scalings_cache() = seqs[0].scaling_cache().clone();
        }
        *pipeline.cache().lock() = new_cache;
    }

    /// Set the model cache to all None. Only used for prompt seqs.
    fn set_none_cache(pipeline: &mut dyn Pipeline) {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.num_hidden_layers() {
            new_cache.push(None);
        }
        *pipeline.cache().lock() = new_cache.clone();
        if pipeline.cache().is_xlora() {
            *pipeline.cache().xlora_lock() = new_cache;
        }
    }

    /// Clone the cache FROM the model cache TO the sequences. Used for prompt, completion seqs.
    fn clone_out_cache(pipeline: &mut dyn Pipeline, seqs: &mut [&mut Sequence]) {
        let num_hidden_layers = pipeline.num_hidden_layers();
        for layer in 0..num_hidden_layers {
            let cache = pipeline.cache().lock();
            let cache = cache.get(layer).unwrap();
            let k_cache = cache.as_ref().unwrap().0.clone();
            let v_cache = cache.as_ref().unwrap().1.clone();

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let seq_cache = seq.cache();
                let seq_cache = &mut seq_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();
                *seq_cache = Some((k, v));
            }
            if pipeline.is_xlora() && !pipeline.has_no_kv_cache() {
                let cache = pipeline.cache().xlora_lock();
                let cache = cache.get(layer).unwrap();
                let k_cache = cache.as_ref().unwrap().0.clone();
                let v_cache = cache.as_ref().unwrap().1.clone();

                let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(k_caches.len(), seqs.len());
                let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(v_caches.len(), seqs.len());

                for (seq_i, seq) in seqs.iter_mut().enumerate() {
                    let seq_cache = seq.xlora_cache();
                    let seq_cache = &mut seq_cache[layer];
                    let k = k_caches.get(seq_i).unwrap().clone();
                    let v = v_caches.get(seq_i).unwrap().clone();
                    *seq_cache = Some((k, v));
                }
            }
            if pipeline.is_xlora() {
                *seqs[0].scaling_cache() = pipeline.cache().get_scalings_cache().clone();
            }
        }
    }

    fn build_sequence_recognizer(constraint: &Constraint) -> anyhow::Result<SequenceRecognizer> {
        let recognizer = match constraint {
            Constraint::Regex(rx) => {
                SequenceRecognizer::Regex(StackRecognizer::from(RecRx::from_rx(rx)?).into())
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

    async fn add_request(&mut self, request: Request) {
        let is_chat = matches!(request.messages, RequestMessage::Chat(_));
        let echo_prompt = matches!(
            request.messages,
            RequestMessage::Completion {
                echo_prompt: true,
                ..
            }
        );

        let best_of = match request.messages {
            RequestMessage::Completion { best_of, .. } => best_of,
            RequestMessage::Chat(_) | RequestMessage::CompletionTokens(_) => 1,
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

        let mut force_tokens = None;
        let formatted_prompt = match request.messages {
            RequestMessage::Chat(messages) => {
                let template = get_mut_arcmutex!(self.pipeline).apply_chat_template(messages, true);
                handle_seq_error!(template, request.response)
            }
            RequestMessage::Completion { text, .. } => text,
            RequestMessage::CompletionTokens(it) => {
                let res = get_mut_arcmutex!(self.pipeline)
                    .tokenizer()
                    .decode(&it, false)
                    .expect("cannot decode completion tokens");
                force_tokens = Some(it);
                res
            }
        };
        if formatted_prompt.is_empty() {
            request
                .response
                .send(Response::ValidationError(
                    "Received an empty prompt.".into(),
                ))
                .await
                .expect("Expected receiver.");
            return;
        }
        let mut prompt = match force_tokens {
            Some(tks) => tks,
            None => {
                let prompt = get_mut_arcmutex!(self.pipeline).tokenize_prompt(&formatted_prompt);
                handle_seq_error!(prompt, request.response)
            }
        };

        if prompt.len() > get_mut_arcmutex!(self.pipeline).get_max_seq_len() {
            if !self.truncate_sequence {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_max_seq_len()).into(),
                    )).await.expect("Expected receiver.");
                return;
            } else {
                let prompt_len = prompt.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_max_seq_len();
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
                info!("⚠️ WARNING: Prompt for request {} was {} tokens over the model maximum length. The last {} tokens were truncated to make space for generation.", request.id, currently_over, prompt_len - prompt.len());
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
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline).num_hidden_layers();

        let (stop_toks, stop_strings) = match request.sampling_params.stop_toks {
            None => (vec![], vec![]),
            Some(StopTokens::Ids(ref i)) => {
                let tok_trie = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    pipeline.tok_trie()
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
                    let tok_trie = pipeline.tok_trie();
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
                get_mut_arcmutex!(self.pipeline).is_xlora(),
                group.clone(),
                response_index,
                now.as_secs(),
                recognizer.clone(),
                request.suffix.clone(),
                if echo_prompt {
                    Some(formatted_prompt.clone())
                } else {
                    None
                },
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
