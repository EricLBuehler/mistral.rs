use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct Tracker {
    tensors: Arc<Mutex<Vec<String>>>,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            tensors: Arc::new(Mutex::new(vec![])),
        }
    }

    pub fn add_tensor(&self, path: String) {
        self.tensors.lock().unwrap().push(path);
    }

    pub fn get(&self) -> MutexGuard<'_, Vec<String>> {
        self.tensors.lock().unwrap()
    }
}
