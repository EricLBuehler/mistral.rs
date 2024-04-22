use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
    sync::{mpsc::Receiver, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    aici::{cfg::CfgParser, recognizer::StackRecognizer, rx::RecRx},
    handle_seq_error_ok, handle_seq_error_stateaware_ok,
    response::CompletionChoice,
    CompletionResponse, RequestMessage,
};
use candle_core::{Result, Tensor};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use tracing::warn;

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
    pipeline: Box<Mutex<dyn Pipeline>>,
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
        pipeline: Box<Mutex<dyn Pipeline>>,
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

    pub fn run(&mut self) {
        let mut last_run = Instant::now();
        'lp: loop {
            while let Ok(request) = self.rx.try_recv() {
                self.add_request(request);
            }
            let mut scheduled = self.scheduler.schedule();
            let mut pipeline = get_mut_arcmutex!(self.pipeline);

            if scheduled.completion.len() > 0 {
                // Run the completion seqs
                if !self.no_kv_cache {
                    Self::clone_in_cache(&mut *pipeline, &mut scheduled.completion);
                }
                let logits = pipeline.forward(&scheduled.completion, false);
                let logits = handle_pipeline_forward_error!(
                    "completion",
                    logits,
                    &mut scheduled.completion,
                    pipeline,
                    'lp,
                    self.prefix_cacher
                );

                if !self.no_kv_cache {
                    Self::clone_out_cache(&mut *pipeline, &mut scheduled.completion);
                } else {
                    Self::set_none_cache(&mut *pipeline);
                }

                handle_pipeline_forward_error!(
                    "sampling",
                    Self::sample_seqs(&*pipeline, &mut scheduled.completion, logits, self.disable_eos_stop),
                    &mut scheduled.completion,
                    pipeline,
                    'lp,
                    self.prefix_cacher
                );

                for seq in scheduled.completion.iter_mut() {
                    self.prefix_cacher.add_sequence(seq);
                }
                handle_pipeline_forward_error!(
                    "evict",
                    self.prefix_cacher.evict_to_cpu(),
                    &mut scheduled.completion,
                    pipeline,
                    'lp,
                    self.prefix_cacher
                );
            }

            if scheduled.prompt.len() > 0 {
                // Run the prompt seqs
                Self::set_none_cache(&mut *pipeline);
                let logits = pipeline.forward(&scheduled.prompt, true);
                let logits = handle_pipeline_forward_error!(
                    "prompt",
                    logits,
                    &mut scheduled.prompt,
                    pipeline,
                    'lp,
                    self.prefix_cacher
                );

                if !self.no_kv_cache {
                    Self::clone_out_cache(&mut *pipeline, &mut scheduled.prompt);
                } else {
                    Self::set_none_cache(&mut *pipeline);
                }

                handle_pipeline_forward_error!(
                    "sampling",
                    Self::sample_seqs(&*pipeline, &mut scheduled.prompt, logits, self.disable_eos_stop),
                    &mut scheduled.prompt,
                    pipeline,
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

                for seq in scheduled.prompt.iter_mut() {
                    self.prefix_cacher.add_sequence(seq);
                }
                handle_pipeline_forward_error!(
                    "evict",
                    self.prefix_cacher.evict_to_cpu(),
                    &mut scheduled.prompt,
                    pipeline,
                    'lp,
                    self.prefix_cacher
                );
            }

            if self.is_debug {
                let ms_from_last_run = last_run.elapsed().as_millis();
                last_run = Instant::now();
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
            drop(pipeline);
            if scheduled.prompt.len() == 0
                && scheduled.completion.len() == 0
                && self.scheduler.waiting_len() == 0
            {
                // If there is nothing to do, sleep until a request comes in
                if let Ok(request) = self.rx.recv() {
                    self.add_request(request);
                }
            }
        }
    }

    fn sample_seqs<'p>(
        pipeline: &dyn Pipeline,
        seqs: &'p mut [&mut Sequence],
        logits: Tensor,
        disable_eos_stop: bool,
    ) -> Result<()> {
        let seqs_len = seqs.len();
        let logits_seq = logits.chunk(seqs_len, 0).unwrap();
        debug_assert_eq!(logits_seq.len(), seqs_len);
        let eos_tok = pipeline.eos_tok().to_vec();
        struct Item<'a> {
            logits: Tensor,
            seq: &'a mut Sequence,
        }
        let mut pairs: Vec<Item<'p>> = logits_seq
            .iter()
            .zip(seqs)
            .map(|(l, s)| Item {
                logits: l.clone(),
                seq: s,
            })
            .collect::<Vec<_>>();
        pairs.par_iter_mut().try_for_each(
            |Item {
                 logits: logits_per_seq,
                 seq,
             }| {
                // Sample and extract next token
                let return_logprobs = seq.return_logprobs();
                let sampled = pipeline.sample(logits_per_seq.clone(), seq, return_logprobs);
                let next_token = handle_seq_error_stateaware_ok!(sampled, seq);
                let next_token_id = next_token.token;

                let eos_tok = if disable_eos_stop {
                    None
                } else {
                    Some(eos_tok.as_ref())
                };
                let is_done = seq.is_done(next_token_id, eos_tok, pipeline.get_max_seq_len());
                seq.add_token(
                    next_token.clone(),
                    pipeline.tok_trie().decode(&[next_token_id]),
                    &is_done,
                );
                // Handle streaming requests
                if seq.get_mut_group().is_streaming && seq.get_mut_group().is_chat {
                    let token_index = seq.get_toks().len();
                    let rate_limit_allowed = is_done.is_some() || token_index % 3 == 0;

                    if rate_limit_allowed {
                        if let Some(delta) = handle_seq_error_ok!(seq.get_delta(), seq.responder())
                        {
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
                                seq.set_state(SequenceState::Done(reason));
                                pipeline.reset_non_granular_state();
                            }

                            seq.get_mut_group()
                                .maybe_send_streaming_response(seq, pipeline.name());
                        }
                    }
                } else if let Some(reason) = is_done {
                    Self::finish_seq(pipeline, seq, reason)?;
                    pipeline.reset_non_granular_state();
                }
                Result::<()>::Ok(())
            },
        )?;
        Ok(())
    }

    fn finish_seq(pipeline: &dyn Pipeline, seq: &mut Sequence, reason: StopReason) -> Result<()> {
        seq.set_state(SequenceState::Done(reason));

        let logprobs = if seq.return_logprobs() {
            let tokenizer = pipeline.tokenizer().clone();
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
            | StopReason::StopTok(_) => String::from_utf8_lossy(seq.completion_bytes())
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

        let group = seq.get_mut_group();
        if group.is_chat {
            group.maybe_send_done_response(
                ChatCompletionResponse {
                    id: seq.id().to_string(),
                    choices: group.get_choices().to_vec(),
                    created: seq.creation_time(),
                    model: pipeline.name(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "chat.completion".to_string(),
                    usage: group.get_usage(),
                },
                seq.responder(),
            );
        } else {
            group.maybe_send_completion_done_response(
                CompletionResponse {
                    id: seq.id().to_string(),
                    choices: group.get_completion_choices().to_vec(),
                    created: seq.creation_time(),
                    model: pipeline.name(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "text_completion".to_string(),
                    usage: group.get_usage(),
                },
                seq.responder(),
            );
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

    fn add_request(&mut self, request: Request) {
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
                    )).expect("Expected receiver.");
            return;
        }

        let mut force_tokens = None;
        let formatted_prompt = match request.messages {
            RequestMessage::Chat(messages) => {
                handle_seq_error!(
                    get_mut_arcmutex!(self.pipeline).apply_chat_template(messages, true),
                    request.response
                )
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
                .expect("Expected receiver.");
            return;
        }
        let mut prompt = match force_tokens {
            Some(tks) => tks,
            None => handle_seq_error!(
                get_mut_arcmutex!(self.pipeline).tokenize_prompt(&formatted_prompt),
                request.response
            ),
        };

        if prompt.len() > get_mut_arcmutex!(self.pipeline).get_max_seq_len() {
            if !self.truncate_sequence {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_max_seq_len()).into(),
                    )).expect("Expected receiver.");
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
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline).num_hidden_layers();

        let (stop_toks, stop_strings) = match request.sampling_params.stop_toks {
            None => (vec![], vec![]),
            Some(StopTokens::Ids(ref i)) => {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                let tok_trie = pipeline.tok_trie();
                for id in i {
                    // We can't use ` ` (space) as a stop token because other tokens like ` moon` start with a space.
                    if tok_trie.has_extensions(tok_trie.token(*id)) {
                        request
                            .response
                            .send(Response::ValidationError(
                                format!("Stop token {:?} is also a prefix of other tokens and cannot be used as a stop token.", tok_trie.token_str(*id)).into(),
                            ))
                            .expect("Expected receiver.");
                        return;
                    }
                }

                (i.clone(), vec![])
            }
            Some(StopTokens::Seqs(ref s)) => {
                let mut stop_toks = Vec::new();
                let mut stop_strings: Vec<String> = Vec::new();

                let pipeline = get_mut_arcmutex!(self.pipeline);
                let tok_trie = pipeline.tok_trie();
                let tokenizer = pipeline.tokenizer();

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

        let group = Rc::new(RefCell::new(SequenceGroup::new(
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
                    .expect("Expected receiver.");
                return;
            }
        };
        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let sampler = Sampler::new(
            SEED,
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
                    .expect("Expected receiver.");
                return;
            }
        };
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
