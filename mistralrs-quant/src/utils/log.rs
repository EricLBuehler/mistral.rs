use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::{LazyLock, Mutex},
};

use tracing::{info, warn};

static CACHED_INFO: LazyLock<Mutex<Vec<u64>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static CACHED_WARN: LazyLock<Mutex<Vec<u64>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub fn once_log_info<M: AsRef<str>>(msg: M) {
    let msg = msg.as_ref();
    let mut hasher = DefaultHasher::new();
    msg.hash(&mut hasher);
    let hash = hasher.finish();

    let mut log = CACHED_INFO.lock().expect("Poisoned Lock");
    if !log.contains(&hash) {
        info!("{msg}");
        log.push(hasher.finish());
    } else {
        log.push(hasher.finish());
    }
}

pub fn once_log_warn<M: AsRef<str>>(msg: M) {
    let msg = msg.as_ref();
    let mut hasher = DefaultHasher::new();
    msg.hash(&mut hasher);
    let hash = hasher.finish();

    let mut log = CACHED_WARN.lock().expect("Poisoned Lock");
    if !log.contains(&hash) {
        warn!("{msg}");
        log.push(hasher.finish());
    } else {
        log.push(hasher.finish());
    }
}
