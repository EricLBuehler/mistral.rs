pub(crate) mod debug;
pub(crate) mod gguf_metadata;
pub(crate) mod memory_usage;
pub(crate) mod model_config;
pub(crate) mod normal;
pub(crate) mod progress;
pub(crate) mod tiktoken;
pub(crate) mod tokenizer;
pub(crate) mod tokens;
pub(crate) mod unvarbuilder;
pub(crate) mod varbuilder_utils;

// Re-export loading progress types for external use
pub use progress::{
    set_loading_progress_callback, LoadingProgress, LoadingProgressCallback, LoadingProgressGuard,
};

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            // This prevents deadlock when a spawned async task busy-loops while
            // another task holds the lock across an await point.
            std::thread::yield_now();
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
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
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
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
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
                if let Err(_) = $seq
                    .responder()
                    .send(Response::InternalError(e.into()))
                    .await
                {
                    tracing::warn!("Receiver disconnected");
                }
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
                    let start = seq.prompt_tokens().min(seq.get_toks().len());
                    let res = match &tokenizer {
                        Some(tok) => match tok.decode(&seq.get_toks()[start..], false) {
                            Ok(t) => t,
                            Err(_) => "".to_string(),
                        },
                        None => "".to_string(),
                    };

                    if seq.get_mut_group().is_chat {
                        let choice = Choice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            message: ResponseMessage {
                                content: Some(res),
                                role: "assistant".to_string(),
                                tool_calls: None,
                                reasoning_content: None,
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

                let p = get_mut_arcmutex!($pipeline);
                // Also reset non granular state because:
                // - The sequence is gone
                // - We should reset the state then, including draft.
                p.set_none_cache($seq_slice, true, true, false);
                get_mut_arcmutex!($prefix_cacher).evict_all_caches().unwrap();

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
            // Yield to allow other threads to make progress and release the lock.
            std::thread::yield_now();
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

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const fn paged_attn_supported() -> bool {
    true
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub const fn paged_attn_supported() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub const fn using_flash_attn() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
pub const fn using_flash_attn() -> bool {
    true
}
