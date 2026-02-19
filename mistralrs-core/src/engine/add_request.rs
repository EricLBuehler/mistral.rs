use crate::{
    pipeline::NormalCache,
    prefix_cacher::MatchingCache,
    request::{DetokenizationRequest, NormalRequest, TokenizationRequest},
    sequence::SeqStepType,
    tools::{ToolCallingMatcher, ToolChoice},
    ModelCategory, RequestMessage, Response,
};
use candle_core::Tensor;
use either::Either;
use std::{
    ops::Deref,
    sync::{atomic::Ordering, Arc},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

use crate::{
    get_mut_arcmutex, handle_seq_error,
    request::Request,
    sampler::Sampler,
    sequence::{Sequence, SequenceGroup},
    StopTokens,
};

use super::{search_request, Engine, TERMINATE_ALL_NEXT_STEP};

impl Engine {
    pub async fn handle_request(self: Arc<Self>, request: Request) {
        match request {
            Request::Normal(request) => {
                let is_chat = matches!(
                    &request.messages,
                    RequestMessage::Chat { .. } | RequestMessage::VisionChat { .. }
                );
                let has_tooling =
                    !self.tool_callbacks.is_empty() || !self.tool_callbacks_with_tools.is_empty();
                let has_search = request.web_search_options.is_some();

                if is_chat && (has_search || has_tooling) {
                    search_request::search_request(self.clone(), *request).await;
                } else {
                    self.add_request(*request).await;
                }
            }
            Request::ReIsq(level) => {
                if let Err(e) = get_mut_arcmutex!(self.pipeline).re_isq_model(level) {
                    warn!("ISQ requantization failed: {e:?}");
                }
            }
            Request::Tokenize(req) => self.tokenize_text(req).await,
            Request::Detokenize(req) => self.detokenize_text(req).await,
            Request::Terminate => (),
            Request::TerminateAllSeqsNextStep => {
                TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst)
            }
        }
    }

    pub(super) async fn add_request(&self, request: NormalRequest) {
        let is_chat = matches!(
            request.messages,
            RequestMessage::Chat { .. } | RequestMessage::VisionChat { .. }
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
            RequestMessage::Chat { .. }
            | RequestMessage::CompletionTokens(_)
            | RequestMessage::VisionChat { .. }
            | RequestMessage::ImageGeneration { .. }
            | RequestMessage::SpeechGeneration { .. }
            | RequestMessage::Embedding { .. }
            | RequestMessage::EmbeddingTokens { .. } => None,
        };
        let truncate_sequence = request.truncate_sequence;
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
                    ))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
            return;
        }

        // Verify the model's category matches the messages received.
        match (
            get_mut_arcmutex!(self.pipeline).category(),
            &request.messages,
        ) {
            (
                ModelCategory::Text | ModelCategory::Vision { .. },
                RequestMessage::Chat { .. }
                | RequestMessage::VisionChat { .. }
                | RequestMessage::Completion { .. }
                | RequestMessage::CompletionTokens(_),
            ) => (),
            (ModelCategory::Diffusion, RequestMessage::ImageGeneration { .. }) => (),
            (ModelCategory::Speech, RequestMessage::SpeechGeneration { .. }) => (),
            (
                ModelCategory::Embedding,
                RequestMessage::Embedding { .. } | RequestMessage::EmbeddingTokens { .. },
            ) => (),
            _ => {
                request
                    .response
                    .send(Response::ValidationError(
                        "Received a request incompatible for this model's category.".into(),
                    ))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            }
        }

        let images = match request.messages {
            RequestMessage::VisionChat { ref images, .. } => Some(images.clone()),
            _ => None,
        };

        let audios = match request.messages {
            RequestMessage::VisionChat { ref audios, .. } => Some(audios.clone()),
            _ => None,
        };
        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
        let matcher = Arc::new(handle_seq_error!(
            ToolCallingMatcher::new(request.tool_choice.unwrap_or(ToolChoice::Auto),),
            request.response
        ));

        let image_generation_format = match &request.messages {
            RequestMessage::ImageGeneration { format, .. } => Some(*format),
            _ => None,
        };

        let seq_step_type = match &request.messages {
            RequestMessage::ImageGeneration { .. }
            | RequestMessage::SpeechGeneration { .. }
            | RequestMessage::Embedding { .. }
            | RequestMessage::EmbeddingTokens { .. } => SeqStepType::OneShot,
            _ => SeqStepType::PromptAndDecode,
        };

        let diffusion_params = match &request.messages {
            RequestMessage::ImageGeneration {
                generation_params, ..
            } => Some(generation_params.clone()),
            _ => None,
        };

        let image_gen_save_file = match &request.messages {
            RequestMessage::ImageGeneration { save_file, .. } => save_file.clone(),
            _ => None,
        };
        let mut added_seq = false;

        let (mut prompt_tokens, prompt_text) = match request.messages {
            RequestMessage::Chat {
                messages,
                enable_thinking,
                reasoning_effort,
            }
            | RequestMessage::VisionChat {
                images: _,
                audios: _,
                messages,
                enable_thinking,
                reasoning_effort,
            } => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tools = request.tools.unwrap_or_default();
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    true,
                    true,
                    enable_thinking,
                    reasoning_effort,
                    tools,
                );
                handle_seq_error!(template, request.response)
            }
            RequestMessage::Completion { text, .. }
            | RequestMessage::Embedding { prompt: text } => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    request
                        .response
                        .send(Response::ValidationError(
                            "Completion requests require the pipeline to have a tokenizer".into(),
                        ))
                        .await
                        .unwrap_or_else(|_| warn!("Receiver disconnected"));
                    return;
                };
                let prompt = tokenizer
                    .encode_fast(text.clone(), true)
                    .map_err(anyhow::Error::msg);
                (
                    handle_seq_error!(prompt, request.response)
                        .get_ids()
                        .to_vec(),
                    text,
                )
            }
            RequestMessage::ImageGeneration { prompt, .. }
            | RequestMessage::SpeechGeneration { prompt } => (vec![u32::MAX], prompt),
            RequestMessage::CompletionTokens(it)
            | RequestMessage::EmbeddingTokens { prompt: it } => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    request
                        .response
                        .send(Response::ValidationError(
                            "Completion requests w/ raw tokens require the pipeline to have a tokenizer".into(),
                        ))
                        .await
                        .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
                .unwrap_or_else(|_| warn!("Receiver disconnected"));
            return;
        }

        if matches!(
            get_mut_arcmutex!(self.pipeline).category(),
            ModelCategory::Text | ModelCategory::Vision { .. } | ModelCategory::Embedding
        ) && prompt_tokens.len() > get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len
        {
            // text/vision => truncate from start
            // embedding => truncate from end
            let category = get_mut_arcmutex!(self.pipeline).category();
            if !truncate_sequence {
                request
                    .response
                    .send(Response::ValidationError(
                        format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len).into(),
                    ))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            } else if matches!(category, ModelCategory::Text | ModelCategory::Vision { .. }) {
                let prompt_len = prompt_tokens.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len;
                let currently_over = prompt_len - max_len;

                // Reserve space for generation tokens
                // If user specified max_len (generation length), reserve that many tokens (capped to max_len)
                // Otherwise, reserve just 1 token minimum to allow at least some generation
                let sampling_max = if let Some(sampling_max) = request.sampling_params.max_len {
                    sampling_max.min(max_len)
                } else {
                    1
                };

                // Calculate how many prompt tokens to keep: max_len - sampling_max
                // This ensures we have room for generation
                let tokens_to_keep = max_len.saturating_sub(sampling_max);

                // Safely calculate slice start position - keep the end of the prompt
                let slice_start = prompt_len.saturating_sub(tokens_to_keep);

                prompt_tokens = prompt_tokens[slice_start..].to_vec();
                warn!("Prompt for request {} was {currently_over} tokens over the model maximum length. The first {slice_start} tokens were truncated to make space for generation.", request.id);
            } else {
                let prompt_len = prompt_tokens.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len;
                let currently_over = prompt_len - max_len;

                prompt_tokens = prompt_tokens[..max_len].to_vec();
                warn!("Prompt for request {} was {currently_over} tokens over the model maximum length. The last {currently_over} tokens were truncated to make space for generation.", request.id);
            }
        }

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
                    pipeline.get_metadata().tok_env()
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
                                .await
                                .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
                    let tok_env = pipeline.get_metadata().tok_env();
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
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    };
                    let encoded = tokenizer.encode_fast(stop_txt.to_string(), true);
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
            request.sampling_params.repetition_penalty,
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
                .unwrap_or_else(|_| warn!("Receiver disconnected"));
            return;
        }

        // Add sequences
        for response_index in 0..request.sampling_params.n_choices {
            let factory = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .llg_factory
                .clone();
            let recognizer = match Self::build_sequence_recognizer(&factory, &request.constraint) {
                Ok(recognizer) => recognizer,
                Err(err) => {
                    request
                        .response
                        .send(Response::ValidationError(
                            format!("Invalid grammar. {err}").into(),
                        ))
                        .await
                        .unwrap_or_else(|_| warn!("Receiver disconnected"));
                    return;
                }
            };

            let block_size = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .cache_config
                .clone()
                .map(|conf| conf.block_size);

            let eos_toks = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .eos_tok
                .clone();

            let seq_preallocated_cache = if matches!(
                get_mut_arcmutex!(self.pipeline).category(),
                ModelCategory::Text | ModelCategory::Vision { .. }
            ) && get_mut_arcmutex!(self.pipeline)
                .do_preallocated_cache()
            {
                let metadata = get_mut_arcmutex!(self.pipeline).get_metadata();
                let model_metadata = metadata
                    .model_metadata
                    .as_ref()
                    .expect("If a model has a NormalCache it must have a model metadata");
                let n_tokens = prompt_tokens.len();
                let required_blocks = n_tokens.div_ceil(NormalCache::CACHE_GROW_SIZE);
                let max_seq_len = required_blocks * NormalCache::CACHE_GROW_SIZE;
                let k_shape = (
                    1usize,
                    model_metadata.num_kv_heads(),
                    max_seq_len,
                    model_metadata.k_head_dim(),
                );
                let v_shape = (
                    1usize,
                    model_metadata.num_kv_heads(),
                    max_seq_len,
                    model_metadata.v_head_dim(),
                );
                let dtype = get_mut_arcmutex!(self.pipeline)
                    .get_metadata()
                    .activation_dtype;

                let k_seq_cache = {
                    let k_seq_cache =
                        Tensor::zeros(k_shape, dtype, &get_mut_arcmutex!(self.pipeline).device());
                    match k_seq_cache {
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
                                .unwrap_or_else(|_| warn!("Receiver disconnected"));
                            return;
                        }
                    }
                };
                let v_seq_cache = if k_shape == v_shape {
                    k_seq_cache.clone()
                } else {
                    let v_seq_cache =
                        Tensor::zeros(v_shape, dtype, &get_mut_arcmutex!(self.pipeline).device());
                    match v_seq_cache {
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
                                .unwrap_or_else(|_| warn!("Receiver disconnected"));
                            return;
                        }
                    }
                };
                Some((k_seq_cache, v_seq_cache))
            } else {
                None
            };

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!");
            let mut seq = Sequence::new_waiting(
                prompt_tokens.clone(),
                prompt_text.clone(),
                *get_mut_arcmutex!(self.id).deref(),
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
                images.clone(),
                audios.clone(),
                block_size,
                if has_tools {
                    Some(matcher.clone())
                } else {
                    None
                },
                image_generation_format,
                seq_step_type,
                diffusion_params.clone(),
                image_gen_save_file.clone(),
                seq_preallocated_cache,
                request.return_raw_logits,
                eos_toks,
            );

            // Only "track" a new sequence if it is a traditional one
            if matches!(seq_step_type, SeqStepType::PromptAndDecode) {
                self.logger.add_new_sequence();
            }

            // Enable Harmony mode if the chat template uses Harmony format
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if let Some(chat_template) = pipeline.get_chat_template() {
                    if chat_template.is_harmony_format() {
                        // Pre-warm the Harmony encoding if not already done.
                        // This must be done in a blocking context because openai-harmony
                        // uses reqwest::blocking which creates its own tokio runtime.
                        if !crate::harmony::is_harmony_encoding_ready() {
                            if let Err(e) = tokio::task::block_in_place(|| {
                                crate::harmony::prewarm_harmony_encoding();
                                Ok::<(), anyhow::Error>(())
                            }) {
                                warn!("Failed to initialize Harmony encoding: {e}");
                            }
                        }
                        if let Err(e) = seq.enable_harmony_mode() {
                            warn!("Failed to enable Harmony mode: {e}");
                        }
                    } else if chat_template.uses_think_tags() {
                        // Enable think tag mode if the chat template uses <think> tags
                        seq.enable_think_tag_mode();
                    }
                }
            }

            // Allocate Mamba state pool slot for hybrid models
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                    let mut hybrid_cache = pipeline.cache().hybrid();
                    if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                        seq.set_mamba_state_idx(Some(slot_idx));
                    }
                }
            }

            // Run the inputs processor to update the prompt for multimodal models.
            if images.is_some() || audios.is_some() {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                let _ = pipeline.get_processor().inputs_processor().process_inputs(
                    pipeline.tokenizer(),
                    &mut [&mut seq],
                    true,
                    pipeline.get_metadata().is_xlora,
                    &pipeline.device(),
                    pipeline.get_metadata().no_kv_cache,
                    None,
                    false,
                    pipeline.get_input_processor_config(),
                    None,
                    pipeline.device_mapper(),
                );
            }

            let prefill_cache = handle_seq_error!(
                get_mut_arcmutex!(self.prefix_cacher).search_for_matching_cache(
                    seq.get_toks(),
                    seq.image_hashes(),
                    seq.audio_hashes(),
                ),
                request.response
            );

            seq = match prefill_cache.clone() {
                Some(MatchingCache::Normal {
                    normal,
                    images_to_keep,
                    audios_to_keep,
                    toks,
                    offset,
                }) => {
                    self.logger.add_prefix_cache_hit();

                    seq.keep_num_images(images_to_keep);
                    seq.keep_num_audios(audios_to_keep);
                    seq.prefill_v2_normal(normal, toks, offset)
                }
                None => seq,
            };

            *get_mut_arcmutex!(self.id) += 1;
            get_mut_arcmutex!(self.scheduler).add_seq(seq);
            added_seq = true;
        }
        if added_seq {
            self.pending_notify.notify_one();
        }
    }

    async fn tokenize_text(&self, request: TokenizationRequest) {
        match request.text {
            Either::Left(messages) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tools = request.tools.unwrap_or_default();
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    request.add_generation_prompt,
                    request.add_special_tokens,
                    request.enable_thinking,
                    request.reasoning_effort,
                    tools,
                );
                let toks = match template {
                    Ok((toks, _)) => toks,
                    Err(e) => {
                        request
                            .response
                            .send(Err(e))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                let toks = tokenizer.encode_fast(text, request.add_special_tokens);
                let toks = match toks {
                    Ok(tokenizer) => tokenizer,
                    Err(e) => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(e)))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
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
