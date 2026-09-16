mod registry;
mod shutdown;

fn empty_state() -> crate::MistralRs {
    use crate::*;

    MistralRs {
        shutdown: crate::shutdown::Shutdown::default(),
        engines: RwLock::new(HashMap::new()),
        unloaded_models: RwLock::new(HashMap::new()),
        reloading_models: RwLock::new(HashSet::new()),
        default_engine_id: RwLock::new(None),
        model_aliases: RwLock::new(HashMap::new()),
        log: None,
        id: "test".to_string(),
        creation_time: 0,
        next_request_id: Mutex::new(RefCell::new(1)),
    }
}
