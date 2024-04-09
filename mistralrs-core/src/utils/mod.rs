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
    ($fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::Engine;
                println!("Model failed with error: {:?}", &e);
                for seq in $seq_slice.iter_mut() {
                    seq.responder()
                        .send(Response::InternalError(e.to_string().into()))
                        .unwrap();
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
