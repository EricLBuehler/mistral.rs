use std::{
    cell::RefCell,
    iter::zip,
    rc::Rc,
    sync::{mpsc::Receiver, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use candle_core::{DType, Device, Result, Tensor};
use candle_sampling::logits_processor::{LogitsProcessor, SamplingMethod};

use crate::{
    deref_mut_refcell, deref_refcell, get_mut_arcmutex, handle_seq_error,
    handle_seq_error_stateaware,
    pa::{
        cache_engine::{CacheConfig, CacheEngine},
        InputMetadata, PreparedInputs, _PAD_SLOT_ID,
    },
    pipeline::{Pipeline, _make_tensor_with_pad},
    request::Request,
    response::{
        ChatCompletionResponse, Choice, Logprobs, Response, ResponseLogprob, ResponseMessage,
    },
    scheduler::{Scheduler, SchedulerOutput},
    sequence::{Sequence, SequenceGroup, SequenceState, StopReason},
    StopTokens,
};

const SEED: u64 = 0;

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Box<Mutex<dyn Pipeline>>,
    scheduler: Scheduler,
    id: usize,
    truncate_sequence: bool,
    no_kv_cache: bool,
    cache_config: CacheConfig,
    cache_engine: CacheEngine,
}

const BLOCK_SIZE: usize = 16;
const CPU_SWAP_SPACE: usize = 1024 * 1024 * 1024; // 1 gb
const GPU_UTILIZATION: usize = 1024 * 1024 * 1024 * 10; // 10gb

impl Engine {
    pub fn new(
        rx: Receiver<Request>,
        pipeline: Box<Mutex<dyn Pipeline>>,
        max_num_seqs: usize,
        truncate_sequence: bool,
        no_kv_cache: bool,
    ) -> Self {
        let (cache_engine, conf, num_cpu, num_gpu) = {
            let pipeline = get_mut_arcmutex!(pipeline);
            let key_cache_block = BLOCK_SIZE
                * pipeline.config().get_num_attention_heads()
                * pipeline.config().get_head_size();
            let value_cache_block = key_cache_block;
            let total =
                pipeline.config().get_num_hidden_layers() * (key_cache_block + value_cache_block);
            let cache_block_size = DType::BF16.size_in_bytes() * total;
            let num_cpu_blocks = CPU_SWAP_SPACE / cache_block_size;
            let num_gpu_blocks = GPU_UTILIZATION / cache_block_size;
            let conf = CacheConfig {
                block_size: BLOCK_SIZE,
                num_cpu_blocks: Some(num_cpu_blocks),
                num_gpu_blocks: Some(num_gpu_blocks),
                fully_init: true,
            };
            dbg!(pipeline.device());
            dbg!(CPU_SWAP_SPACE);
            dbg!(cache_block_size);
            (
                CacheEngine::new(&*pipeline.config(), conf, DType::BF16, pipeline.device())
                    .unwrap(),
                conf,
                num_cpu_blocks,
                num_gpu_blocks,
            )
        };
        Self {
            rx,
            pipeline,
            scheduler: Scheduler::new(max_num_seqs, BLOCK_SIZE, num_gpu, num_cpu),
            id: 0,
            truncate_sequence,
            no_kv_cache,
            cache_config: conf,
            cache_engine,
        }
    }

    pub fn run(&mut self) {
        loop {
            if let Ok(request) = self.rx.try_recv() {
                self.add_request(request);
            }
            let scheduler_outputs = self.scheduler.schedule();

            let scheduled = &*scheduler_outputs.scheduled;

            let seqs = scheduled
                .iter()
                .flat_map(|group| {
                    deref_refcell!(group)
                        .get_seqs()
                        .values()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            if scheduled.len() > 0 {
                self.execute_scheduler_ops(&scheduler_outputs).unwrap();

                // NOTE(EricLBuehler): assume all are prompts or completions
                let firstseq = deref_refcell!(scheduled.first().unwrap())
                    .get_seqs()
                    .values()
                    .nth(0)
                    .unwrap()
                    .clone();
                let PreparedInputs {
                    tokens,
                    positions,
                    metadata,
                } = if deref_refcell!(firstseq).is_prompt() {
                    self.prepare_prompt(scheduled).unwrap()
                } else {
                    // Because of the KV cache, we only need to take
                    // the last token.
                    self.prepare_completion(scheduled).unwrap()
                };

                // Run the completion seqs
                let logits = get_mut_arcmutex!(self.pipeline).forward(
                    tokens,
                    positions,
                    Some(&*self.cache_engine.get_kv_cache()),
                    metadata,
                );
                self.synchronize(get_mut_arcmutex!(self.pipeline).device());

                let before_sample = Instant::now();
                self.sample_seqs(&seqs, logits);
                let sampling_time = before_sample.elapsed().as_millis();
                for seq in scheduled.iter() {
                    deref_mut_refcell!(seq).total_sampling_time += sampling_time;
                }

                self.scheduler.free_finished_sequence_groups();
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
            let sampled = get_mut_arcmutex!(self.pipeline).sample(logits_per_seq, seq.clone());
            let next_token = handle_seq_error_stateaware!(sampled, seq);
            let next_token_id = next_token.token;
            if deref_refcell!(seq).is_prompt() {
                deref_mut_refcell!(seq).prompt_timestamp = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time travel has occurred!")
                        .as_millis(),
                );
            }
            deref_mut_refcell!(seq).add_token(next_token);
            let is_done = deref_refcell!(seq).is_done(
                next_token_id,
                eos_tok,
                get_mut_arcmutex!(self.pipeline).get_max_seq_len(),
            );
            if let Some(reason) = is_done {
                self.finish_seq(seq, reason);
            }
        }
    }

    fn finish_seq(&self, seq: &Rc<RefCell<Sequence>>, reason: StopReason) {
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
            get_mut_arcmutex!(self.pipeline).tokenizer().decode(
                &deref_refcell!(seq).get_toks()[deref_refcell!(seq).prompt_tokens()..],
                false
            ),
            deref_refcell!(seq).responder()
        );

        let choice = Choice {
            stopreason: match reason {
                StopReason::Eos => "stop".to_string(),
                StopReason::Length(_) | StopReason::ModelLength(_) => "length".to_string(),
                StopReason::StopTok(_) => "stop".to_string(),
            },
            index: deref_refcell!(seq).get_next_choice_index(),
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
        deref_mut_refcell!(seq).add_choice_to_group(choice);

        let group = deref_refcell!(seq).get_group();
        // Is the group done?
        if !deref_refcell!(group).is_done() {
            return;
        }

        // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
        deref_refcell!(seq)
            .responder()
            .send(Response::Done(ChatCompletionResponse {
                id: deref_refcell!(seq).id().to_string(),
                choices: deref_refcell!(group).get_choices().to_vec(),
                created: deref_refcell!(seq).timestamp(),
                model: get_mut_arcmutex!(self.pipeline).name(),
                system_fingerprint: "local".to_string(),
                object: "chat.completion".to_string(),
                usage: deref_refcell!(group).get_usage(),
            }))
            .unwrap();
    }

    fn execute_scheduler_ops(&mut self, scheduler_output: &SchedulerOutput) -> Result<()> {
        self.cache_engine
            .swap_in(scheduler_output.blocks_to_swap_in.clone())?;
        self.cache_engine
            .swap_out(scheduler_output.blocks_to_swap_out.clone())?;
        self.cache_engine
            .copy(scheduler_output.blocks_to_copy.clone())?;
        Ok(())
    }

    fn prepare_prompt(&self, groups: &[Rc<RefCell<SequenceGroup>>]) -> Result<PreparedInputs> {
        let mut prompt_lens = Vec::new();
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut slot_mappings = Vec::new();
        for group in groups {
            for seq in deref_refcell!(group).get_seqs().values().cloned() {
                let seq = deref_refcell!(seq);
                let prompt_ids = seq.get_toks().to_vec();

                let prompt_len = prompt_ids.len();
                prompt_lens.push(prompt_len);

                input_tokens.push(prompt_ids);
                input_positions.push((0..prompt_len).collect::<Vec<_>>());
                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.id())
                    .unwrap();
                if table.is_empty() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = get_mut_arcmutex!(self.pipeline)
                    .config()
                    .get_sliding_window()
                {
                    0.max(prompt_len as i64 - sliding_window as i64)
                } else {
                    0
                } as usize;

                let mut slot_mapping = Vec::new();
                for i in 0..prompt_len {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }

                    let block_number = table[i / self.cache_config.block_size];
                    let block_offset = i % self.cache_config.block_size;
                    let slot = block_number * self.cache_config.block_size + block_offset;
                    slot_mapping.push(slot.try_into().unwrap());
                }
                slot_mappings.push(slot_mapping);
            }
        }

        let max_prompt_len = prompt_lens.iter().max().unwrap();
        let dev = get_mut_arcmutex!(self.pipeline).device().clone();
        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
            &dev,
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
            &dev,
        )?;
        dbg!(&slot_mappings);
        let slot_mapping =
            _make_tensor_with_pad(slot_mappings, *max_prompt_len, _PAD_SLOT_ID, &dev)?;

        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens,
                slot_mapping,
                max_context_len: None,
                context_lens: None,
                block_tables: None,
                is_prompt: true,
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
            },
        })
    }

    fn prepare_completion(&self, groups: &[Rc<RefCell<SequenceGroup>>]) -> Result<PreparedInputs> {
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut context_lens = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in deref_refcell!(group).get_seqs().values().cloned() {
                let last_token_id = *deref_refcell!(seq).get_toks().last().unwrap();
                input_tokens.push(vec![last_token_id]);

                let position = deref_refcell!(seq).len() - 1;
                input_positions.push(vec![position]);

                let context_len = if let Some(sliding_window) = get_mut_arcmutex!(self.pipeline)
                    .config()
                    .get_sliding_window()
                {
                    deref_refcell!(seq).len().min(sliding_window)
                } else {
                    deref_refcell!(seq).len()
                };
                context_lens.push(context_len);

                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&deref_refcell!(seq).id())
                    .unwrap();
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = table[position / self.cache_config.block_size];
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot = slot.try_into().unwrap();
                slot_mappings.push(vec![slot]);

                if let Some(sliding_window) = get_mut_arcmutex!(self.pipeline)
                    .config()
                    .get_sliding_window()
                {
                    let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                    if sliding_window_blocks > table.len() {
                        block_tables.push(table);
                    } else {
                        block_tables.push(
                            table
                                .get(table.len() - sliding_window_blocks..)
                                .unwrap()
                                .to_vec(),
                        );
                    }
                } else {
                    block_tables.push(table);
                }
            }
        }

        let dev = get_mut_arcmutex!(self.pipeline).device().clone();
        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
            &dev,
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
            &dev,
        )?;
        dbg!(&slot_mappings);
        let slot_mapping = _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, &dev)?;

        let max_context_len = context_lens.iter().max().unwrap();
        let context_lens = Tensor::from_vec(
            context_lens.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            (context_lens.len(),),
            &dev,
        )?;
        dbg!(&context_lens.to_vec1::<i64>());

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = _make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
            &dev,
        )?;

        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens: vec![],
                slot_mapping,
                max_context_len: Some(*max_context_len),
                context_lens: Some(context_lens),
                block_tables: Some(block_tables),
                is_prompt: false,
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
            },
        })
    }

    fn add_request(&mut self, request: Request) {
        let prompt = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).apply_chat_template(request.messages.clone(), true),
            request.response
        );
        let mut prompt = handle_seq_error!(
            get_mut_arcmutex!(self.pipeline).tokenize_prompt(&prompt),
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
            (Some(topk), None, Some(_)) => SamplingMethod::TopK(topk),
            (None, Some(topp), Some(_)) => SamplingMethod::TopP(topp),
            (Some(topk), Some(topp), Some(_)) => SamplingMethod::TopKP((topk, topp)),
            (None, None, None) => SamplingMethod::Multinomial,
            (Some(_), Some(_), None) | (None, Some(_), None) | (Some(_), None, None) => {
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                request
                    .response
                    .send(Response::Error(
                        "If topp or topk are specified and temperature is not specified then argmax sampling will be used. Consider using a temperature of 1.".into(),
                    ))
                    .unwrap();
                return;
            }
            (None, None, Some(_)) => {
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                request
                    .response
                    .send(Response::Error(
                        "If topp and topk are not specified but temperature is set then argmax sampling will be used.".into(),
                    ))
                    .unwrap();
                return;
            }
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

        let group = Rc::new(RefCell::new(SequenceGroup::new(
            request.sampling_params.n_choices,
            self.id,
        )));
        self.id += 1;
        // Add sequences
        for _ in 0..request.sampling_params.n_choices {
            let seq = Sequence::new_waiting(
                prompt.clone(),
                self.id,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time travel has occurred!")
                    .as_millis(),
                num_hidden_layers,
                request.response.clone(),
                LogitsProcessor::new(
                    SEED,
                    request.sampling_params.temperature,
                    sampling_method.clone(),
                    request.sampling_params.top_n_logprobs,
                    tokenizer.clone(),
                    request.sampling_params.repeat_penalty,
                    request.sampling_params.presence_penalty,
                    request.sampling_params.logits_bias.clone(),
                ),
                stop_toks.clone(),
                request.sampling_params.max_len,
                request.return_logprobs,
                get_mut_arcmutex!(self.pipeline).is_xlora(),
                Rc::downgrade(&group),
                BLOCK_SIZE,
            );
            self.id += 1;
            let seq = Rc::new(RefCell::new(seq));
            deref_mut_refcell!(group).add_seq(seq);
        }
        self.scheduler.add_seq(group);
    }
}
