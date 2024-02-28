pub(crate) mod dtype;
pub(crate) mod tokens;
pub(crate) mod varbuilder_utils;

#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.lock() {
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
                use crate::response::Response;
                // NOTE(EricLBuehler): Unwrap reasoning: The reciever should really be there, otherwise it is their fault.
                $response.send(Response::Error(e.into())).unwrap();
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! deref_refcell {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_borrow() {
                break inner;
            }
        }
    };
}

#[macro_export]
macro_rules! deref_mut_refcell {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_borrow_mut() {
                break inner;
            }
        }
    };
}
