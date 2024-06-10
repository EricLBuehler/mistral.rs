pub(crate) mod debug;
pub(crate) mod gguf_metadata;
pub(crate) mod model_config;
pub(crate) mod normal;
pub(crate) mod progress;
pub(crate) mod tokenizer;
pub(crate) mod tokens;
pub(crate) mod varbuilder_utils;

#[doc(hidden)]
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

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                $response
                    .send(Response::InternalError(e.into()))
                    .await
                    .expect("Expected receiver.");
                return;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_ok {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                $response
                    .send(Response::InternalError(e.into()))
                    .await
                    .expect("Expected receiver.");
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_stateaware_ok {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                $seq.responder()
                    .send(Response::InternalError(e.into()))
                    .await
                    .expect("Expected receiver.");
                $seq.set_state(SequenceState::Error);
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_pipeline_forward_error {
    ($stage: tt, $fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt, $prefix_cacher:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                let (tokenizer, pipeline_name) = {
                    let pipeline = get_mut_arcmutex!($pipeline);
                    let pipeline_name = pipeline.name();
                    let tokenizer = pipeline.tokenizer();
                    (tokenizer, pipeline_name)
                };
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::response::SYSTEM_FINGERPRINT;
                use tracing::error;
                error!("{} - Model failed with error: {:?}", $stage, &e);
                for seq in $seq_slice.iter_mut() {
                    // Step 1: Add all choices to groups
                    let res = match tokenizer
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
                            model: pipeline_name.clone(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "chat.completion".to_string(),
                            usage: group.get_usage(),
                        };

                        seq.responder()
                            .send(Response::ModelError(
                                e.to_string(),
                                partial_completion_response
                            ))
                            .await
                            .unwrap();
                    } else {
                        let partial_completion_response = CompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_completion_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline_name.clone(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "text_completion".to_string(),
                            usage: group.get_usage(),
                        };

                        seq.responder()
                            .send(Response::CompletionModelError(
                                e.to_string(),
                                partial_completion_response
                            ))
                            .await
                            .unwrap();
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
                    seq.set_state(SequenceState::Error);
                }

                let mut p = get_mut_arcmutex!($pipeline);
                // Also reset non granular state because:
                // - The sequence is gone
                // - We should reset the state then, including draft.
                p.set_none_cache(true, true);
                $prefix_cacher.evict_all_to_cpu().unwrap();

                continue $label;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_group {
    ($this:expr) => {
        loop {
            if let Ok(inner) = $this.group.try_lock() {
                break inner;
            }
        }
    };
}

#[doc(hidden)]
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

#[doc(hidden)]
#[macro_export]
macro_rules! sample_async {
    (
        $use_async_pool: expr,
        $sampler: expr,
        $logits: expr,
        $ctx: expr,
        $return_logprobs: expr,
        $rng: expr,
        $sample_speculative: expr
     ) => {
        if $use_async_pool {
            tokio_rayon::spawn(move || {
                $sampler.sample(
                    $logits,
                    Some(&$ctx),
                    $return_logprobs,
                    $rng,
                    $sample_speculative,
                )
            })
            .await?
        } else {
            $sampler.sample(
                $logits,
                Some(&$ctx),
                $return_logprobs,
                $rng,
                $sample_speculative,
            )?
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}
