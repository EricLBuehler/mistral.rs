use crate::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::empty_state;

#[test]
fn missing_default_sender_is_model_not_found() {
    assert!(matches!(
        empty_state().get_sender(None),
        Err(MistralRsError::ModelNotFound(model)) if model == "default"
    ));
    assert!(matches!(
        empty_state().get_sender(Some("wrong-model")),
        Err(MistralRsError::ModelNotFound(model)) if model == "wrong-model"
    ));
}

#[test]
fn reloading_sender_preserves_model_state_error() {
    let state = empty_state();
    state
        .reloading_models
        .write()
        .unwrap()
        .insert("model".to_string());
    assert!(matches!(
        state.get_sender(Some("model")),
        Err(MistralRsError::ModelReloading(model)) if model == "model"
    ));
}

#[test]
fn fallible_file_helpers_preserve_poisoned_engine_error() {
    let state = empty_state();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = state.engines.write().unwrap();
        panic!("poison engines lock");
    }));
    assert!(result.is_err());

    assert!(matches!(
        state.try_find_file("file-id"),
        Err(MistralRsError::EnginePoisoned)
    ));
    assert!(matches!(
        state.try_list_files(),
        Err(MistralRsError::EnginePoisoned)
    ));
    assert!(matches!(
        state.try_remove_file("file-id"),
        Err(MistralRsError::EnginePoisoned)
    ));
}
