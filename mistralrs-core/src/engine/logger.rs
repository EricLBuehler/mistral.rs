#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::info;

use crate::sequence::Sequence;

pub(super) struct DecodeTokenSnapshot(usize);

impl DecodeTokenSnapshot {
    pub(super) fn capture<'a>(sequences: impl Iterator<Item = &'a Sequence>) -> Self {
        Self(sequences.map(Sequence::generated_len).sum())
    }

    pub(super) fn record<'a>(
        self,
        logger: &IntervalLogger,
        sequences: impl Iterator<Item = &'a Sequence>,
    ) {
        let generated_tokens = sequences.map(Sequence::generated_len).sum::<usize>();
        logger.add_decode_tokens_processed(generated_tokens.saturating_sub(self.0));
    }
}

#[derive(Default)]
struct PrefixCacheStats {
    hits: usize,
    total_sequences: usize,
}

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_stats: Arc<Mutex<PrefixCacheStats>>,
    tokens_processed: Arc<AtomicUsize>,
    prefill_tokens_processed: Arc<AtomicUsize>,
    decode_tokens_processed: Arc<AtomicUsize>,
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    sequence_capacity: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
    spec_drafts: Arc<AtomicUsize>,
    spec_draft_tokens: Arc<AtomicUsize>,
    spec_accepted_tokens: Arc<AtomicUsize>,
    shutdown_tx: Sender<()>,
    worker: Option<JoinHandle<()>>,
    #[cfg(test)]
    worker_exited: Arc<AtomicBool>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_stats = Arc::new(Mutex::new(PrefixCacheStats::default()));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let prefill_tokens_processed = Arc::new(AtomicUsize::new(0));
        let decode_tokens_processed = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));
        let sequence_capacity = Arc::new(AtomicUsize::new(0));
        let spec_drafts = Arc::new(AtomicUsize::new(0));
        let spec_draft_tokens = Arc::new(AtomicUsize::new(0));
        let spec_accepted_tokens = Arc::new(AtomicUsize::new(0));

        let t_prefix_cache_stats = prefix_cache_stats.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_prefill_tokens_processed = prefill_tokens_processed.clone();
        let t_decode_tokens_processed = decode_tokens_processed.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let t_sequence_capacity = sequence_capacity.clone();
        let t_spec_drafts = spec_drafts.clone();
        let t_spec_draft_tokens = spec_draft_tokens.clone();
        let t_spec_accepted_tokens = spec_accepted_tokens.clone();
        let (encoder_cache_hits, encoder_cache_misses) = match encoder_cache_counters {
            Some((h, m)) => (Some(h), Some(m)),
            None => (None, None),
        };
        let t_enc_hits = encoder_cache_hits.clone();
        let t_enc_misses = encoder_cache_misses.clone();
        #[cfg(test)]
        let worker_exited = Arc::new(AtomicBool::new(false));
        #[cfg(test)]
        let t_worker_exited = worker_exited.clone();
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            // Start the actual logging
            while let Err(RecvTimeoutError::Timeout) = shutdown_rx.recv_timeout(interval) {
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);
                metrics::gauge!("mistralrs_sequences_running").set(num_running as f64);
                metrics::gauge!("mistralrs_sequences_waiting").set(num_waiting as f64);
                metrics::gauge!("mistralrs_sequences_capacity")
                    .set(t_sequence_capacity.load(Ordering::Relaxed) as f64);

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
                let prefill_tokens_processed =
                    t_prefill_tokens_processed.swap(0, Ordering::Relaxed);
                let decode_tokens_processed = t_decode_tokens_processed.swap(0, Ordering::Relaxed);
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
                        let mean_depth = spec_draft_tokens as f64 / spec_drafts.max(1) as f64;
                        format!(
                            ", MTP accept {accept_rate:.1}% (len {mean_len:.2}, depth {mean_depth:.1})"
                        )
                    } else {
                        String::new()
                    };

                    // Throughput = tokens processed during this interval / interval duration.
                    // The counter is atomically swapped to 0 each interval, so the metric
                    // reflects only the current window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2} (prefill {:.2}, decode {:.2}), Prefix cache hitrate {:.2}%{enc_cache_info}{spec_info}, {num_running} running, {num_waiting} waiting",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        prefill_tokens_processed as f64 / interval.as_secs_f64(),
                        decode_tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );
                }
            }
            #[cfg(test)]
            t_worker_exited.store(true, Ordering::Release);
        });

        Self {
            prefix_cache_stats,
            tokens_processed,
            prefill_tokens_processed,
            decode_tokens_processed,
            enable_logging,
            num_running,
            num_waiting,
            sequence_capacity,
            encoder_cache_hits,
            encoder_cache_misses,
            spec_drafts,
            spec_draft_tokens,
            spec_accepted_tokens,
            shutdown_tx,
            worker: Some(worker),
            #[cfg(test)]
            worker_exited,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    pub fn reset(&self) {
        *self.prefix_cache_stats.lock().unwrap() = PrefixCacheStats::default();
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.prefill_tokens_processed.store(0, Ordering::Relaxed);
        self.decode_tokens_processed.store(0, Ordering::Relaxed);
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

    /// Count prompt (prefill) tokens through the pipeline. Also advances the
    /// combined `mistralrs_tokens_processed_total` counter, which always equals
    /// prefill plus decode tokens.
    pub fn add_prefill_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        self.prefill_tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        metrics::counter!("mistralrs_tokens_processed_total").increment(num_tokens as u64);
        metrics::counter!("mistralrs_prefill_tokens_processed_total").increment(num_tokens as u64);
    }

    /// Count generated (decode) tokens through the pipeline. With speculative
    /// decoding only verified tokens are counted. Also advances the combined
    /// `mistralrs_tokens_processed_total` counter.
    pub fn add_decode_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        self.decode_tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        metrics::counter!("mistralrs_tokens_processed_total").increment(num_tokens as u64);
        metrics::counter!("mistralrs_decode_tokens_processed_total").increment(num_tokens as u64);
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

    pub fn set_sequence_capacity(&self, capacity: usize) {
        self.sequence_capacity.store(capacity, Ordering::Relaxed);
        metrics::gauge!("mistralrs_sequences_capacity").set(capacity as f64);
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

impl Drop for IntervalLogger {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sampler::{Logprobs, Sampler},
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer, SequenceState, StopReason},
    };
    use std::collections::HashMap;

    const INACTIVE_LOGGER_INTERVAL: Duration = Duration::from_secs(3600);
    const TEST_SEQUENCE_CAPACITY: usize = 16;
    const TEST_CONTEXT_LENGTH: usize = 128;
    const TEST_EOS_TOKEN: u32 = 99;
    const TEST_STOP_TOKEN: u32 = 42;

    fn test_sequence() -> Sequence {
        let (sender, _receiver) = tokio::sync::mpsc::channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            1, false, true, None,
        )));
        Sequence::new_waiting(
            vec![1, 2, 3, 4],
            "prompt".to_string(),
            0,
            0,
            1,
            sender,
            sampler,
            vec![TEST_STOP_TOKEN],
            vec!["STOP".to_string()],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            None,
        )
    }

    fn commit_token(seq: &mut Sequence, token: u32, bytes: &[u8]) -> Option<StopReason> {
        let reason = seq.is_done(token, Some(&[TEST_EOS_TOKEN]), TEST_CONTEXT_LENGTH);
        let reason = seq.add_token(
            Logprobs {
                token,
                logprob: 0.0,
                bytes: None,
                top_logprobs: None,
            },
            bytes.to_vec(),
            reason,
        );
        if let Some(reason) = reason {
            seq.set_state(SequenceState::Done(reason));
        }
        reason
    }

    #[test]
    fn decode_counts_committed_tokens_instead_of_speculative_width() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
        let mut seq = test_sequence();
        commit_token(&mut seq, 10, b"first");
        seq.set_num_computed_tokens(seq.get_toks().len() - 1);
        seq.set_staged_speculative(vec![11, 12, 13, 14], None);
        let snapshot = DecodeTokenSnapshot::capture([&seq].into_iter());

        seq.take_staged_speculative_tokens();
        commit_token(&mut seq, 11, b" accepted");
        commit_token(&mut seq, 20, b" replacement");
        snapshot.record(&logger, [&seq].into_iter());

        assert_eq!(logger.decode_tokens_processed.load(Ordering::Relaxed), 2);
        assert_eq!(logger.tokens_processed.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn decode_counts_terminal_tokens_with_uncommitted_drafts() {
        for (token, bytes) in [
            (TEST_EOS_TOKEN, b"<eos>".as_slice()),
            (TEST_STOP_TOKEN, b"<stop>".as_slice()),
            (10, b"prefix STOP suffix".as_slice()),
        ] {
            let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
            let mut seq = test_sequence();
            seq.set_staged_speculative(vec![token, 11, 12, 13], None);
            let snapshot = DecodeTokenSnapshot::capture([&seq].into_iter());

            assert!(commit_token(&mut seq, token, bytes).is_some());
            snapshot.record(&logger, [&seq].into_iter());

            assert_eq!(logger.decode_tokens_processed.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn decode_does_not_count_computed_lookahead_without_committed_tokens() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
        let mut seq = test_sequence();
        commit_token(&mut seq, TEST_EOS_TOKEN, b"<eos>");
        seq.set_num_computed_tokens(seq.get_toks().len() - 1);
        let snapshot = DecodeTokenSnapshot::capture([&seq].into_iter());

        seq.advance_num_computed_tokens(1);
        snapshot.record(&logger, [&seq].into_iter());

        assert_eq!(logger.decode_tokens_processed.load(Ordering::Relaxed), 0);
        assert_eq!(logger.tokens_processed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sequence_capacity_is_retained_for_recurring_publication() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);

        logger.set_sequence_capacity(TEST_SEQUENCE_CAPACITY);

        assert_eq!(
            logger.sequence_capacity.load(Ordering::Relaxed),
            TEST_SEQUENCE_CAPACITY
        );
    }

    #[test]
    fn drop_wakes_and_joins_worker() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
        let worker_exited = logger.worker_exited.clone();

        drop(logger);

        assert!(worker_exited.load(Ordering::Acquire));
    }
}
