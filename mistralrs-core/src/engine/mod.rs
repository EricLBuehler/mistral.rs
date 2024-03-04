use std::{
    cell::RefCell,
    collections::VecDeque,
    iter::zip,
    rc::Rc,
    sync::{mpsc::Receiver, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::Tensor;
use candle_sampling::logits_processor::{LogitsProcessor, SamplingMethod};

use crate::{
    deref_mut_refcell, deref_refcell, get_mut_arcmutex, handle_seq_error,
    handle_seq_error_stateaware,
    pipeline::Pipeline,
    request::Request,
    response::{
        ChatCompletionResponse, ChatCompletionUsage, Choice, Logprobs, Response, ResponseLogprob,
        ResponseMessage,
    },
    scheduler::{Scheduler, SchedulerMethod},
    sequence::{Sequence, SequenceState, StopReason},
    StopTokens,
};

const SEED: u64 = 0;

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Box<Mutex<dyn Pipeline>>,
    requests: VecDeque<Request>,
    scheduler: Scheduler<VecDeque<Rc<RefCell<Sequence>>>>,
    id: usize,
    truncate_sequence: bool,
}

impl Engine {
    pub fn new(
        rx: Receiver<Request>,
        pipeline: Box<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        truncate_sequence: bool,
    ) -> Self {
        Self {
            rx,
            pipeline,
            requests: VecDeque::new(),
            scheduler: Scheduler::new(method),
            id: 0,
            truncate_sequence,
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
                self.clone_in_cache(&scheduled.completion);
                let logits =
                    get_mut_arcmutex!(self.pipeline).forward(scheduled.completion.clone(), false);
                self.sample_seqs(&scheduled.completion, logits);
                self.clone_out_cache(&scheduled.completion);
            }

            if scheduled.prompt.len() > 0 {
                // Run the prompt seqs
                self.set_none_cache();
                let logits =
                    get_mut_arcmutex!(self.pipeline).forward(scheduled.prompt.clone(), true);
                for seq in scheduled.prompt.iter() {
                    deref_mut_refcell!(seq).set_state(SequenceState::RunningCompletion);
                }
                self.sample_seqs(&scheduled.prompt, logits);
                self.clone_out_cache(&scheduled.prompt);
            }
        }
    }

    fn sample_seqs(&self, seqs: &[Rc<RefCell<Sequence>>], logits: Tensor) {
        let seqs_len = seqs.len();
        let logits_seq = logits.chunk(seqs_len, 0).unwrap();
        debug_assert_eq!(logits_seq.len(), seqs_len);
        let eos_tok = get_mut_arcmutex!(self.pipeline).eos_tok();
        for (logits_per_seq, seq) in zip(logits_seq, seqs.iter()) {
            let sampled = get_mut_arcmutex!(self.pipeline).sample(logits_per_seq, seq.clone());
            let next_token = handle_seq_error_stateaware!(sampled, seq);
            let next_token_id = next_token.token;
            deref_mut_refcell!(seq).add_token(next_token);
            let is_done = deref_refcell!(seq).is_done(
                next_token_id,
                eos_tok,
                get_mut_arcmutex!(self.pipeline).get_max_seq_len(),
            );
            if let Some(reason) = is_done {
                deref_mut_refcell!(seq).set_state(SequenceState::Done(reason));

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
                        top_logprobs: logprob.top_logprobs.clone(),
                    };
                    logprobs.push(resp_logprob);
                }

                let res = handle_seq_error!(
                    get_mut_arcmutex!(self.pipeline)
                        .tokenizer()
                        .decode(deref_refcell!(seq).get_toks(), false),
                    deref_refcell!(seq).responder()
                );

                let choice = Choice {
                    stopreason: match reason {
                        StopReason::Eos => "stop".to_string(),
                        StopReason::Length(_) | StopReason::ModelLength(_) => "length".to_string(),
                        StopReason::StopTok(_) => "stop".to_string(),
                    },
                    index: 0,
                    message: ResponseMessage {
                        content: res,
                        role: "assistant".to_string(),
                    },
                    logprobs: if deref_refcell!(seq).return_logprobs() {
                        Some(Logprobs {
                            content: Some(logprobs),
                        })
                    } else {
                        None
                    },
                };

                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                deref_refcell!(seq)
                    .responder()
                    .send(Response::Done(ChatCompletionResponse {
                        id: deref_refcell!(seq).id().to_string(),
                        choices: vec![choice],
                        created: deref_refcell!(seq).timestamp(),
                        model: get_mut_arcmutex!(self.pipeline).name(),
                        system_fingerprint: "local".to_string(),
                        object: "chat.completion".to_string(),
                        usage: ChatCompletionUsage {
                            completion_tokens: deref_refcell!(seq).logprobs().len(),
                            prompt_tokens: deref_refcell!(seq).prompt_tokens(),
                            total_tokens: deref_refcell!(seq).len(),
                        },
                    }))
                    .unwrap();
            }
        }
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
        *get_mut_arcmutex!(self.pipeline).cache().lock() = new_cache;
    }

    /// Set the model cache to all None. Only used for prompt seqs.
    fn set_none_cache(&self) {
        let mut new_cache = Vec::new();
        for _ in 0..get_mut_arcmutex!(self.pipeline).num_hidden_layers() {
            new_cache.push(None);
        }
        *get_mut_arcmutex!(self.pipeline).cache().lock() = new_cache;
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
        }
    }

    fn add_request(&mut self, request: Request) {
        let mut prompt = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).tokenize_prompt(&request.prompt),
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
                    sampling_max
                } else {
                    10
                };
                prompt = prompt[(currently_over + sampling_max)..].to_vec();
            }
        }

        let sampling_method = match (
            request.sampling_params.top_k,
            request.sampling_params.top_p,
            request.sampling_params.temperature,
        ) {
            (Some(topk), None, _) => SamplingMethod::TopK(topk),
            (None, Some(topp), _) => SamplingMethod::TopP(topp),
            (None, None, Some(_)) | (Some(_), Some(_), Some(_)) | (Some(_), Some(_), None) => {
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                request
                    .response
                    .send(Response::Error(
                        "Please specify either topk or topp, or neither if temperature is not specified.".into(),
                    ))
                    .unwrap();
                return;
            }
            (None, None, None) => SamplingMethod::Multinomial,
        };
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

        let seq = Sequence::new_waiting(
            prompt,
            self.id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            num_hidden_layers,
            request.response.clone(),
            LogitsProcessor::new(
                SEED,
                request.sampling_params.temperature,
                sampling_method,
                request.sampling_params.top_n_logprobs,
                tokenizer,
                request.sampling_params.repeat_penalty,
                request.sampling_params.presence_penalty,
            ),
            stop_toks,
            request.sampling_params.max_len,
            request.return_logprobs,
        );
        self.id += 1;

        self.requests.push_back(request);
        self.scheduler.add_seq(seq);
    }
}
