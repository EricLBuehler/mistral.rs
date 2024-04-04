use std::{
    cell::RefCell,
    collections::VecDeque,
    iter::zip,
    rc::Rc,
    sync::{mpsc::Receiver, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use candle_core::{Device, Tensor};
use tracing::warn;

use crate::{
    deref_mut_refcell, deref_refcell, get_mut_arcmutex, handle_seq_error,
    handle_seq_error_stateaware,
    pipeline::Pipeline,
    request::Request,
    response::{
        ChatCompletionResponse, Choice, ChunkChoice, Delta, Logprobs, Response, ResponseLogprob,
        ResponseMessage, SYSTEM_FINGERPRINT,
    },
    sampler::Sampler,
    scheduler::{Scheduler, SchedulerMethod},
    sequence::{Sequence, SequenceGroup, SequenceState, StopReason},
    StopTokens,
};

const SEED: u64 = 0;

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Box<Mutex<dyn Pipeline>>,
    scheduler: Scheduler<VecDeque<Rc<RefCell<Sequence>>>>,
    id: usize,
    truncate_sequence: bool,
    no_kv_cache: bool,
}

impl Engine {
    pub fn new(
        rx: Receiver<Request>,
        pipeline: Box<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        truncate_sequence: bool,
        no_kv_cache: bool,
    ) -> Self {
        Self {
            rx,
            pipeline,
            scheduler: Scheduler::new(method),
            id: 0,
            truncate_sequence,
            no_kv_cache,
        }
    }

    pub fn run(&mut self) {
        loop {
            if let Ok(request) = self.rx.try_recv() {
                self.add_request(request);
            }
            let scheduled = self.scheduler.schedule();

            if scheduled.completion.len() > 0 {
                // Run the completion seqs
                if !self.no_kv_cache {
                    self.clone_in_cache(&scheduled.completion);
                }
                let logits =
                    get_mut_arcmutex!(self.pipeline).forward(scheduled.completion.clone(), false);
                self.synchronize(get_mut_arcmutex!(self.pipeline).device());

                let before_sample = Instant::now();
                self.sample_seqs(&scheduled.completion, logits);
                let sampling_time = before_sample.elapsed().as_millis();
                for seq in scheduled.completion.iter() {
                    deref_mut_refcell!(seq).total_sampling_time += sampling_time;
                }

                if !self.no_kv_cache {
                    self.clone_out_cache(&scheduled.completion);
                } else {
                    self.set_none_cache();
                }
            }

            if scheduled.prompt.len() > 0 {
                // Run the prompt seqs
                self.set_none_cache();
                let logits =
                    get_mut_arcmutex!(self.pipeline).forward(scheduled.prompt.clone(), true);
                self.synchronize(get_mut_arcmutex!(self.pipeline).device());
                for seq in scheduled.prompt.iter() {
                    deref_mut_refcell!(seq).set_state(SequenceState::RunningCompletion);
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time travel has occurred!")
                        .as_millis();
                    #[allow(clippy::cast_precision_loss)]
                    let prompt_tok_per_sec = deref_refcell!(seq).len() as f32
                        / (now - deref_refcell!(seq).timestamp()) as f32;
                    deref_mut_refcell!(seq).prompt_tok_per_sec = prompt_tok_per_sec * 1000.;
                    deref_mut_refcell!(seq).prompt_timestamp = Some(now);
                }

                let before_sample = Instant::now();
                self.sample_seqs(&scheduled.prompt, logits);
                let sampling_time = before_sample.elapsed().as_millis();
                for seq in scheduled.prompt.iter() {
                    deref_mut_refcell!(seq).total_sampling_time += sampling_time;
                }

                if !self.no_kv_cache {
                    self.clone_out_cache(&scheduled.prompt);
                } else {
                    self.set_none_cache();
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn synchronize(&self, dev: &Device) {
        if let candle_core::Device::Cuda(dev) = dev {
            dev.synchronize().unwrap();
        }
    }
    #[cfg(not(feature = "cuda"))]
    fn synchronize(&self, _dev: &Device) {}

    fn sample_seqs(&self, seqs: &[Rc<RefCell<Sequence>>], logits: Tensor) {
        let seqs_len = seqs.len();
        let logits_seq = logits.chunk(seqs_len, 0).unwrap();
        debug_assert_eq!(logits_seq.len(), seqs_len);
        let eos_tok = get_mut_arcmutex!(self.pipeline).eos_tok();
        for (logits_per_seq, seq) in zip(logits_seq, seqs.iter()) {
            // Sample and extract next token
            let sampled = get_mut_arcmutex!(self.pipeline).sample(
                logits_per_seq,
                seq.clone(),
                deref_refcell!(seq).return_logprobs(),
            );
            let next_token = handle_seq_error_stateaware!(sampled, seq);
            let next_token_id = next_token.token;
            deref_mut_refcell!(seq).add_token(next_token.clone());
            let is_done = deref_refcell!(seq).is_done(
                next_token_id,
                eos_tok,
                get_mut_arcmutex!(self.pipeline).get_max_seq_len(),
            );
            // Handle streaming requests
            if deref_refcell!(seq).get_group().is_streaming {
                let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer().clone();
                let mut seq = deref_mut_refcell!(seq);
                if let Some(delta) = handle_seq_error!(seq.get_delta(&tokenizer), seq.responder()) {
                    seq.add_streaming_chunk_choice_to_group(ChunkChoice {
                        delta: Delta {
                            content: delta.clone(),
                            role: "assistant".to_string(),
                        },
                        index: seq.get_response_index(),
                        stopreason: is_done.map(|x| x.to_string()),
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
                    }

                    seq.get_mut_group().maybe_send_streaming_response(
                        &seq,
                        get_mut_arcmutex!(self.pipeline).name(),
                    );
                }
            } else if let Some(reason) = is_done {
                self.finish_seq(seq, reason);
                get_mut_arcmutex!(self.pipeline).reset_non_granular_state();
            }
        }
    }

    fn finish_seq(&self, seq: &Rc<RefCell<Sequence>>, reason: StopReason) {
        deref_mut_refcell!(seq).set_state(SequenceState::Done(reason));

        let logprobs = if deref_refcell!(seq).return_logprobs() {
            let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer().clone();
            let mut logprobs = Vec::new();
            for logprob in deref_refcell!(seq).logprobs() {
                let resp_logprob = ResponseLogprob {
                    token: handle_seq_error!(
                        tokenizer.decode(&[logprob.token], false),
                        deref_refcell!(seq).responder()
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

        let res = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).tokenizer().decode(
                &deref_refcell!(seq).get_toks()[deref_refcell!(seq).prompt_tokens()..],
                false
            ),
            deref_refcell!(seq).responder()
        );

        let choice = Choice {
            stopreason: reason.to_string(),
            index: deref_refcell!(seq).get_response_index(),
            message: ResponseMessage {
                content: res,
                role: "assistant".to_string(),
            },
            logprobs: if let Some(logprobs) = logprobs {
                Some(Logprobs {
                    content: Some(logprobs),
                })
            } else {
                None
            },
        };
        deref_mut_refcell!(seq).add_choice_to_group(choice);

        deref_refcell!(seq).get_group().maybe_send_done_response(
            ChatCompletionResponse {
                id: deref_refcell!(seq).id().to_string(),
                choices: deref_refcell!(seq).get_group().get_choices().to_vec(),
                created: deref_refcell!(seq).creation_time(),
                model: get_mut_arcmutex!(self.pipeline).name(),
                system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                object: "chat.completion".to_string(),
                usage: deref_refcell!(seq).get_group().get_usage(),
            },
            deref_refcell!(seq).responder(),
        );
    }

    /// Clone the cache FROM the sequences' cache TO the model cache. Only used for completion seqs.
    fn clone_in_cache(&self, seqs: &[Rc<RefCell<Sequence>>]) {
        let mut new_cache = Vec::new();
        for layer in 0..get_mut_arcmutex!(self.pipeline).num_hidden_layers() {
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();
            for seq in seqs.iter() {
                let mut seq = deref_mut_refcell!(seq);
                let seq_cache = &*seq.cache();
                let cache = seq_cache.get(layer).unwrap();
                // Note(EricLBuehler): Unwrap reasoning: We are handling completions seqs so unwrap is OK.
                let cache = cache.as_ref().unwrap();
                k_vec.push(cache.0.clone());
                v_vec.push(cache.1.clone());
            }
            // NOTE(EricLBuehler): Unwrap reasoning: We have the correct dims
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
        if get_mut_arcmutex!(self.pipeline).is_xlora()
            && !get_mut_arcmutex!(self.pipeline).has_no_kv_cache()
        {
            let mut new_cache = Vec::new();
            for layer in 0..get_mut_arcmutex!(self.pipeline).num_hidden_layers() {
                let mut k_vec = Vec::new();
                let mut v_vec = Vec::new();
                for seq in seqs.iter() {
                    let mut seq = deref_mut_refcell!(seq);
                    let seq_cache = &*seq.xlora_cache();
                    let cache = seq_cache.get(layer).unwrap();
                    // Note(EricLBuehler): Unwrap reasoning: We are handling completions seqs so unwrap is OK.
                    let cache = cache.as_ref().unwrap();
                    k_vec.push(cache.0.clone());
                    v_vec.push(cache.1.clone());
                }
                // NOTE(EricLBuehler): Unwrap reasoning: We have the correct dims
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
            *get_mut_arcmutex!(self.pipeline).cache().xlora_lock() = new_cache;
        }
        if get_mut_arcmutex!(self.pipeline).is_xlora() {
            *get_mut_arcmutex!(self.pipeline)
                .cache()
                .get_scalings_cache() = deref_mut_refcell!(seqs[0]).scaling_cache().clone();
        }
        *get_mut_arcmutex!(self.pipeline).cache().lock() = new_cache;
    }

    /// Set the model cache to all None. Only used for prompt seqs.
    fn set_none_cache(&self) {
        let mut new_cache = Vec::new();
        for _ in 0..get_mut_arcmutex!(self.pipeline).num_hidden_layers() {
            new_cache.push(None);
        }
        *get_mut_arcmutex!(self.pipeline).cache().lock() = new_cache.clone();
        if get_mut_arcmutex!(self.pipeline).cache().is_xlora() {
            *get_mut_arcmutex!(self.pipeline).cache().xlora_lock() = new_cache;
        }
    }

    /// Clone the cache FROM the model cache TO the sequences. Used for prompt, completion seqs.
    fn clone_out_cache(&self, seqs: &[Rc<RefCell<Sequence>>]) {
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline).num_hidden_layers();
        for layer in 0..num_hidden_layers {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            let cache = pipeline.cache().lock();
            let cache = cache.get(layer).unwrap();
            let k_cache = cache.as_ref().unwrap().0.clone();
            let v_cache = cache.as_ref().unwrap().1.clone();

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter().enumerate() {
                let mut seq = deref_mut_refcell!(seq);
                let seq_cache = seq.cache();
                let seq_cache = seq_cache.get_mut(layer).unwrap();
                *seq_cache = Some((
                    k_caches.get(seq_i).unwrap().clone(),
                    v_caches.get(seq_i).unwrap().clone(),
                ));
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

                for (seq_i, seq) in seqs.iter().enumerate() {
                    let mut seq = deref_mut_refcell!(seq);
                    let seq_cache = seq.xlora_cache();
                    let seq_cache = seq_cache.get_mut(layer).unwrap();
                    *seq_cache = Some((
                        k_caches.get(seq_i).unwrap().clone(),
                        v_caches.get(seq_i).unwrap().clone(),
                    ));
                }
            }
            if pipeline.is_xlora() {
                *deref_mut_refcell!(seqs[0]).scaling_cache() =
                    pipeline.cache().get_scalings_cache().clone();
            }
        }
    }

    fn add_request(&mut self, request: Request) {
        let formatted_prompt = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).apply_chat_template(request.messages.clone(), true),
            request.response
        );
        let mut prompt = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).tokenize_prompt(&formatted_prompt),
            request.response
        );
        if prompt.len() > get_mut_arcmutex!(self.pipeline).get_max_seq_len() {
            if !self.truncate_sequence {
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                request
                    .response
                    .send(Response::Error(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_max_seq_len()).into(),
                    ))
                    .unwrap();
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

        let topk = request
            .sampling_params
            .top_k
            .map(|x| x as i64)
            .unwrap_or(-1);
        let topp = request.sampling_params.top_p.unwrap_or(1.0);
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline).num_hidden_layers();
        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let stop_toks = match request.sampling_params.stop_toks {
            None => vec![],
            Some(StopTokens::Ids(ref i)) => i.clone(),
            Some(StopTokens::Seqs(ref s)) => {
                let mut stop_toks = Vec::new();
                let encoded = tokenizer.encode(s.clone(), false);
                let toks = handle_seq_error!(encoded, request.response)
                    .get_ids()
                    .to_vec();
                if toks.len() > 1 {
                    // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                    request
                        .response
                        .send(Response::Error(
                            format!("Stop sequence '{s:?}' encodes to multiple tokens when it should only encode to 1.").into(),
                        ))
                        .unwrap();
                }
                stop_toks.push(toks[0]);
                stop_toks
            }
        };

        let group = Rc::new(RefCell::new(SequenceGroup::new(
            request.sampling_params.n_choices,
            request.is_streaming,
        )));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!");
        let sampler = Sampler::new(
            SEED,
            Some(request.sampling_params.temperature.unwrap_or(1.0)),
            request.sampling_params.top_n_logprobs,
            tokenizer.clone(),
            request.sampling_params.repeat_penalty,
            request.sampling_params.presence_penalty,
            request.sampling_params.logits_bias.clone(),
            topk,
            topp,
        );
        // Add sequences
        for response_index in 0..request.sampling_params.n_choices {
            let seq = Sequence::new_waiting(
                prompt.clone(),
                formatted_prompt.clone(),
                self.id,
                now.as_millis(),
                num_hidden_layers,
                request.response.clone(),
                sampler.clone(),
                stop_toks.clone(),
                request.sampling_params.max_len,
                request.return_logprobs,
                get_mut_arcmutex!(self.pipeline).is_xlora(),
                group.clone(),
                response_index,
                now.as_secs(),
            );
            self.id += 1;
            self.scheduler.add_seq(seq);
        }
    }
}
