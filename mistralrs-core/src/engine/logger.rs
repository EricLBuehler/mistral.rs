#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tracing::info;

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_hits: Arc<AtomicUsize>,
    tokens_processed: Arc<AtomicUsize>,
    total_new_seqs: Arc<AtomicUsize>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(interval: Duration) -> Self {
        let prefix_cache_hits = Arc::new(AtomicUsize::new(0));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let total_new_seqs = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));

        let t_prefix_cache_hits = prefix_cache_hits.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_total_new_seqs = total_new_seqs.clone();
        let t_enable_logging = enable_logging.clone();
        thread::spawn(move || {
            // Wait
            while !t_enable_logging.load(Ordering::Relaxed) {}

            // Start the actual logging
            loop {
                thread::sleep(interval);

                let total_new_seqs = t_total_new_seqs.load(Ordering::Relaxed);
                let prefix_cache_hits = t_prefix_cache_hits.load(Ordering::Relaxed);
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);

                if total_new_seqs != 0 && tokens_processed != 0 {
                    info!(
                        "Throughput (T/s) {:.2}, Prefix cache hitrate {:.2}%",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );
                }
            }
        });

        Self {
            prefix_cache_hits,
            tokens_processed,
            total_new_seqs,
            enable_logging,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    pub fn add_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
    }

    pub fn add_new_sequence(&self) {
        self.total_new_seqs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_prefix_cache_hit(&self) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
}
