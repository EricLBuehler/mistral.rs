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
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_hits = Arc::new(AtomicUsize::new(0));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let total_new_seqs = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));

        let t_prefix_cache_hits = prefix_cache_hits.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_total_new_seqs = total_new_seqs.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let (encoder_cache_hits, encoder_cache_misses) = match encoder_cache_counters {
            Some((h, m)) => (Some(h), Some(m)),
            None => (None, None),
        };
        let t_enc_hits = encoder_cache_hits.clone();
        let t_enc_misses = encoder_cache_misses.clone();
        thread::spawn(move || {
            // Start the actual logging
            loop {
                thread::sleep(interval);
                if !t_enable_logging.load(Ordering::Relaxed) {
                    continue;
                }

                let total_new_seqs = t_total_new_seqs.load(Ordering::Relaxed);
                let prefix_cache_hits = t_prefix_cache_hits.load(Ordering::Relaxed);
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);

                if total_new_seqs != 0 && tokens_processed != 0 {
                    let enc_cache_info =
                        if let (Some(ref hits), Some(ref misses)) = (&t_enc_hits, &t_enc_misses) {
                            let h = hits.load(Ordering::Relaxed);
                            let m = misses.load(Ordering::Relaxed);
                            let total = h + m;
                            if total > 0 {
                                format!(
                                    ", Encoder cache hitrate {:.2}%",
                                    100. * h as f64 / total as f64
                                )
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                    // Throughput = tokens processed during this interval / interval duration.
                    // Combines both prefill and decode tokens. The counter is atomically
                    // swapped to 0 each interval, so the metric reflects only the current
                    // window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2}, Prefix cache hitrate {:.2}%{enc_cache_info}, {num_running} running, {num_waiting} waiting",
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
            num_running,
            num_waiting,
            encoder_cache_hits,
            encoder_cache_misses,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    pub fn reset(&self) {
        self.prefix_cache_hits.store(0, Ordering::Relaxed);
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.total_new_seqs.store(0, Ordering::Relaxed);
        self.num_running.store(0, Ordering::Relaxed);
        self.num_waiting.store(0, Ordering::Relaxed);
        if let Some(ref hits) = self.encoder_cache_hits {
            hits.store(0, Ordering::Relaxed);
        }
        if let Some(ref misses) = self.encoder_cache_misses {
            misses.store(0, Ordering::Relaxed);
        }
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

    pub fn set_num_running(&self, running: usize) {
        self.num_running.store(running, Ordering::Relaxed);
    }

    pub fn set_num_waiting(&self, waiting: usize) {
        self.num_waiting.store(waiting, Ordering::Relaxed);
    }

    /// Return cumulative prefix cache (hits, total_sequences).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        (
            self.prefix_cache_hits.load(Ordering::Relaxed),
            self.total_new_seqs.load(Ordering::Relaxed),
        )
    }

    /// Return cumulative encoder cache (hits, misses), or `None` if no encoder cache exists.
    pub fn encoder_cache_stats(&self) -> Option<(usize, usize)> {
        match (&self.encoder_cache_hits, &self.encoder_cache_misses) {
            (Some(h), Some(m)) => Some((h.load(Ordering::Relaxed), m.load(Ordering::Relaxed))),
            _ => None,
        }
    }
}
