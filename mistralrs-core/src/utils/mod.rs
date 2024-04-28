pub(crate) mod tokens;
pub(crate) mod varbuilder_utils;

#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
        }
    };
}

#[macro_export]
macro_rules! handle_seq_error {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                $response
                    .blocking_send(Response::InternalError(e.into()))
                    .expect("Expected receiver.");
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! handle_seq_error_ok {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                $response
                    .blocking_send(Response::InternalError(e.into()))
                    .expect("Expected receiver.");
                return Ok(());
            }
        }
    };
}

#[macro_export]
macro_rules! handle_seq_error_stateaware {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                $seq.responder()
                    .send(Response::InternalError(e.into()))
                    .expect("Expected receiver.");
                $seq.set_state(SequenceState::Error);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! handle_seq_error_stateaware_ok {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                $seq.responder()
                    .blocking_send(Response::InternalError(e.into()))
                    .expect("Expected receiver.");
                $seq.set_state(SequenceState::Error);
                return Ok(());
            }
        }
    };
}

#[macro_export]
macro_rules! handle_pipeline_forward_error {
    ($stage: tt, $fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt, $prefix_cacher:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                let mut pipeline = $pipeline;
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::Engine;
                use $crate::response::SYSTEM_FINGERPRINT;
                use tracing::error;
                error!("{} - Model failed with error: {:?}", $stage, &e);
                for seq in $seq_slice.iter_mut() {
                    // Step 1: Add all choices to groups
                    let res = match pipeline
                        .tokenizer()
                        .decode(&seq.get_toks()[seq.prompt_tokens()..], false)
                    {
                        Ok(v) => v,
                        Err(_) => "".to_string(),
                    };

                    if seq.get_mut_group().is_chat {
                        let choice = Choice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            message: ResponseMessage {
                                content: res,
                                role: "assistant".to_string(),
                            },
                            logprobs: None,
                        };
                        seq.add_choice_to_group(choice);
                    } else {
                        let choice = CompletionChoice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            text: res,
                            logprobs: None,
                        };
                        seq.add_completion_choice_to_group(choice);
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 2: Respond with all groups
                    let group = seq.get_mut_group();

                    if group.is_chat {
                        let partial_completion_response = ChatCompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline.name(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "chat.completion".to_string(),
                            usage: group.get_usage(),
                        };

                        seq.responder()
                            .blocking_send(Response::ModelError(
                                e.to_string(),
                                partial_completion_response
                            ))
                            .unwrap();
                    } else {
                        let partial_completion_response = CompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_completion_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline.name(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "text_completion".to_string(),
                            usage: group.get_usage(),
                        };

                        seq.responder()
                            .blocking_send(Response::CompletionModelError(
                                e.to_string(),
                                partial_completion_response
                            ))
                            .unwrap();
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
                    seq.set_state(SequenceState::Error);
                }

                Engine::set_none_cache(&mut *pipeline);
                $prefix_cacher.evict_all_to_cpu().unwrap();

                continue $label;
            }
        }
    };
}

#[macro_export]
macro_rules! get_mut_group {
    ($this:expr) => {
        loop {
            if let Ok(inner) = $this.group.try_borrow_mut() {
                break inner;
            }
        }
    };
}

#[macro_export]
macro_rules! get_bias_if_not_allowed {
    ($tok_trie:expr, $rx:expr, $next_token_id:expr) => {
        if $tok_trie.token_allowed($rx, $next_token_id) {
            None
        } else {
            let mut token_set = $tok_trie.alloc_token_set();
            $tok_trie.compute_bias($rx, &mut token_set);
            Some(token_set)
        }
    };
}

#[macro_export]
macro_rules! sample_async {
    (
        $use_async_pool: expr,
        $sampler: expr,
        $logits: expr,
        $ctx: expr,
        $return_logprobs: expr,
        $rng: expr
     ) => {
        if $use_async_pool {
            tokio_rayon::spawn(move || {
                $sampler.sample($logits, Some(&$ctx), $return_logprobs, $rng)
            })
            .await?
        } else {
            $sampler.sample($logits, Some(&$ctx), $return_logprobs, $rng)?
        }
    };
}

#[macro_export]
macro_rules! impl_mask {
    () => {
        fn mask(&mut self, t: usize, u: usize, device: &Device) -> Result<Tensor> {
            if let Some(mask) = self.masks.get(&(t, u)) {
                Ok(mask.clone())
            } else {
                let mask: Vec<_> = (0..t)
                    .flat_map(|i| (0..u).map(move |j| u8::from(j + t > i + u)))
                    .collect();
                let mask = Tensor::from_slice(&mask, (t, u), device)?;
                self.masks.insert((t, u), mask.clone());
                Ok(mask)
            }
        }
    };
}
