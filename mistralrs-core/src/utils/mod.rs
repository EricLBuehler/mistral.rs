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
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                $response.send(Response::InternalError(e.into())).unwrap();
                return;
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
                // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
                $seq.responder().send(Response::InternalError(e.into())).unwrap();
                $seq.set_state(SequenceState::Error);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! handle_pipeline_forward_error {
    ($stage: tt, $fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::Engine;
                use $crate::response::SYSTEM_FINGERPRINT;
                println!("{} - Model failed with error: {:?}", $stage, &e);
                for seq in $seq_slice.iter_mut() {
                    // Step 1: Add all choices to groups
                    let res = match $pipeline
                        .tokenizer()
                        .decode(&seq.get_toks()[seq.prompt_tokens()..], false)
                    {
                        Ok(v) => v,
                        Err(_) => "".to_string(),
                    };
                    let choice = Choice {
                        stopreason: "error".to_string(),
                        index: seq.get_response_index(),
                        message: ResponseMessage {
                            content: res,
                            role: "assistant".to_string(),
                        },
                        logprobs: None,
                    };
                    seq.add_choice_to_group(choice);
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 2: Respond with all groups
                    let group = seq.get_mut_group();

                    let partial_completion_response = ChatCompletionResponse {
                        id: seq.id().to_string(),
                        choices: group.get_choices().to_vec(),
                        created: seq.creation_time(),
                        model: $pipeline.name(),
                        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                        object: "chat.completion".to_string(),
                        usage: group.get_usage(),
                    };

                    seq.responder()
                        .send(Response::ModelError(
                            e.to_string(),
                            partial_completion_response
                        ))
                        .unwrap();
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
                    seq.set_state(SequenceState::Error);
                }

                Engine::set_none_cache(&mut *$pipeline);

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
