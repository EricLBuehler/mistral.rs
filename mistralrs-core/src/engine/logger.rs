#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tracing::info;

#[derive(Default)]
struct PrefixCacheStats {
    hits: usize,
    total_sequences: usize,
}

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_stats: Arc<Mutex<PrefixCacheStats>>,
    tokens_processed: Arc<AtomicUsize>,
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
    spec_drafts: Arc<AtomicUsize>,
    spec_draft_tokens: Arc<AtomicUsize>,
    spec_accepted_tokens: Arc<AtomicUsize>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_stats = Arc::new(Mutex::new(PrefixCacheStats::default()));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));
        let spec_drafts = Arc::new(AtomicUsize::new(0));
        let spec_draft_tokens = Arc::new(AtomicUsize::new(0));
        let spec_accepted_tokens = Arc::new(AtomicUsize::new(0));

        let t_prefix_cache_stats = prefix_cache_stats.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let t_spec_drafts = spec_drafts.clone();
        let t_spec_draft_tokens = spec_draft_tokens.clone();
        let t_spec_accepted_tokens = spec_accepted_tokens.clone();
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

                let (prefix_cache_hits, total_new_seqs) = {
                    let stats = t_prefix_cache_stats.lock().unwrap();
                    (stats.hits, stats.total_sequences)
                };
                if let (Some(hits), Some(misses)) = (&t_enc_hits, &t_enc_misses) {
                    metrics::counter!("mistralrs_encoder_cache_hits_total")
                        .absolute(hits.load(Ordering::Relaxed) as u64);
                    metrics::counter!("mistralrs_encoder_cache_misses_total")
                        .absolute(misses.load(Ordering::Relaxed) as u64);
                }
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);
                let spec_drafts = t_spec_drafts.swap(0, Ordering::Relaxed);
                let spec_draft_tokens = t_spec_draft_tokens.swap(0, Ordering::Relaxed);
                let spec_accepted_tokens = t_spec_accepted_tokens.swap(0, Ordering::Relaxed);

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
                    let spec_info = if spec_draft_tokens > 0 {
                        // vLLM-style rates: accept rate over proposed draft tokens,
                        // mean acceptance length includes the bonus token.
                        let accept_rate =
                            100. * spec_accepted_tokens as f64 / spec_draft_tokens as f64;
                        let mean_len = 1. + spec_accepted_tokens as f64 / spec_drafts.max(1) as f64;
                        format!(", MTP accept {accept_rate:.1}% (len {mean_len:.2})")
                    } else {
                        String::new()
                    };

                    // Throughput = tokens processed during this interval / interval duration.
                    // Combines both prefill and decode tokens. The counter is atomically
                    // swapped to 0 each interval, so the metric reflects only the current
                    // window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2}, Prefix cache hitrate {:.2}%{enc_cache_info}{spec_info}, {num_running} running, {num_waiting} waiting",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );
                }
            }
        });

        Self {
            prefix_cache_stats,
            tokens_processed,
            enable_logging,
            num_running,
            num_waiting,
            encoder_cache_hits,
            encoder_cache_misses,
            spec_drafts,
            spec_draft_tokens,
            spec_accepted_tokens,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    pub fn reset(&self) {
        *self.prefix_cache_stats.lock().unwrap() = PrefixCacheStats::default();
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.num_running.store(0, Ordering::Relaxed);
        self.num_waiting.store(0, Ordering::Relaxed);
        if let Some(ref hits) = self.encoder_cache_hits {
            hits.store(0, Ordering::Relaxed);
        }
        if let Some(ref misses) = self.encoder_cache_misses {
            misses.store(0, Ordering::Relaxed);
        }
        self.spec_drafts.store(0, Ordering::Relaxed);
        self.spec_draft_tokens.store(0, Ordering::Relaxed);
        self.spec_accepted_tokens.store(0, Ordering::Relaxed);
    }

    pub fn add_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        metrics::counter!("mistralrs_tokens_processed_total").increment(num_tokens as u64);
    }

    /// Record one speculative verification batch (across all its sequences).
    pub fn add_speculative_stats(
        &self,
        num_drafts: usize,
        num_draft_tokens: usize,
        num_accepted_tokens: usize,
        accepted_per_pos: &[usize],
    ) {
        if num_drafts == 0 {
            return;
        }
        self.spec_drafts.fetch_add(num_drafts, Ordering::Relaxed);
        self.spec_draft_tokens
            .fetch_add(num_draft_tokens, Ordering::Relaxed);
        self.spec_accepted_tokens
            .fetch_add(num_accepted_tokens, Ordering::Relaxed);
        metrics::counter!("mistralrs_speculative_drafts_total").increment(num_drafts as u64);
        metrics::counter!("mistralrs_speculative_draft_tokens_proposed_total")
            .increment(num_draft_tokens as u64);
        metrics::counter!("mistralrs_speculative_draft_tokens_accepted_total")
            .increment(num_accepted_tokens as u64);
        for (position, count) in accepted_per_pos.iter().enumerate() {
            if *count > 0 {
                metrics::counter!(
                    "mistralrs_speculative_draft_tokens_accepted_per_pos_total",
                    "position" => position.to_string()
                )
                .increment(*count as u64);
            }
        }
    }

    pub fn add_new_sequence(&self) {
        self.prefix_cache_stats.lock().unwrap().total_sequences += 1;
        metrics::counter!("mistralrs_prefix_cache_lookups_total").increment(1);
    }

    pub fn add_prefix_cache_hit(&self) {
        self.prefix_cache_stats.lock().unwrap().hits += 1;
        metrics::counter!("mistralrs_prefix_cache_hits_total").increment(1);
    }

    pub fn set_num_running(&self, running: usize) {
        self.num_running.store(running, Ordering::Relaxed);
        metrics::gauge!("mistralrs_sequences_running").set(running as f64);
    }

    pub fn set_num_waiting(&self, waiting: usize) {
        self.num_waiting.store(waiting, Ordering::Relaxed);
        metrics::gauge!("mistralrs_sequences_waiting").set(waiting as f64);
    }

    /// Return cumulative prefix cache (hits, total_sequences).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        let stats = self.prefix_cache_stats.lock().unwrap();
        (stats.hits, stats.total_sequences)
    }

    /// Return cumulative encoder cache (hits, misses), or `None` if no encoder cache exists.
    pub fn encoder_cache_stats(&self) -> Option<(usize, usize)> {
        match (&self.encoder_cache_hits, &self.encoder_cache_misses) {
            (Some(h), Some(m)) => Some((h.load(Ordering::Relaxed), m.load(Ordering::Relaxed))),
            _ => None,
        }
    }
}
