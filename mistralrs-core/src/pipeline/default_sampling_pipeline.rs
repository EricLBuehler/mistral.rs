use std::{
    iter::zip,
    sync::{Arc, Mutex},
};

use candle_core::Device;
use futures::future;
use rand_isaac::Isaac64Rng;

use crate::{aici::toktree::TokTrie, get_mut_arcmutex, handle_seq_error_ok, handle_seq_error_stateaware_ok, pipeline::sample_sequence, sequence::SequenceState, ChunkChoice, Delta, Pipeline, ResponseLogprob};

use super::SamplingPipeline;

pub struct DefaultSamplingPipeline {
    repeat_last_n: usize,
    tok_trie: Arc<TokTrie>,
    rng: Arc<Mutex<Isaac64Rng>>,
    max_seq_len: usize,
}

impl SamplingPipeline for DefaultSamplingPipeline {
    async fn sample(
        &self,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        seqs: &mut [&mut crate::sequence::Sequence],
        logits: candle_core::Tensor,
        prefix_cacher: &mut crate::prefix_cacher::PrefixCacheManager,
        disable_eos_stop: bool,
    ) -> anyhow::Result<()> {
        let seqs_len = seqs.len();
        let logits_seq = logits.to_device(&Device::Cpu)?.chunk(seqs_len, 0)?;
        debug_assert_eq!(logits_seq.len(), seqs_len);

        let use_async_pool = seqs_len > 1;

        let sampling_futures: Vec<_> = zip(logits_seq, seqs.iter_mut())
            .map(|(logits_per_seq, seq)| {
                let return_logprobs = seq.return_logprobs();
                sample_sequence(
                    logits_per_seq,
                    seq,
                    return_logprobs,
                    self.repeat_last_n,
                    self.tok_trie.clone(),
                    self.rng.clone(),
                    use_async_pool,
                )
            })
            .collect();
        let sampled_vec = future::join_all(sampling_futures).await;

        for (sampled, seq) in zip(sampled_vec, seqs.iter_mut()) {
            let next_token = handle_seq_error_stateaware_ok!(sampled, seq);
            let next_token_id = next_token.token;

            let eos_tok = if disable_eos_stop {
                None
            } else {
                Some(eos_tok.as_ref())
            };
            let is_done = seq.is_done(next_token_id, eos_tok, self.max_seq_len);
            seq.add_token(
                next_token.clone(),
                self.tok_trie.decode(&[next_token_id]),
                &is_done,
            );
            // Handle streaming requests
            if seq.get_mut_group().is_streaming && seq.get_mut_group().is_chat {
                let token_index = seq.get_toks().len();
                let rate_limit_allowed = is_done.is_some() || token_index % 3 == 0;

                if rate_limit_allowed {
                    if let Some(delta) = handle_seq_error_ok!(seq.get_delta(), seq.responder()) {
                        seq.add_streaming_chunk_choice_to_group(ChunkChoice {
                            delta: Delta {
                                content: delta.clone(),
                                role: "assistant".to_string(),
                            },
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|x| x.to_string()),
                            logprobs: if seq.return_logprobs() {
                                Some(ResponseLogprob {
                                    token: delta,
                                    bytes: next_token.bytes.clone().into_bytes(),
                                    logprob: next_token.logprob,
                                    top_logprobs: next_token.top_logprobs.unwrap().clone(),
                                })
                            } else {
                                None
                            },
                        });

                        if let Some(reason) = is_done {
                            prefix_cacher.add_sequence(seq);
                            prefix_cacher.evict_to_cpu()?;
                            seq.set_state(SequenceState::Done(reason));
                            get_mut_arcmutex!(pipeline).reset_non_granular_state();
                        }

                        if seq
                            .get_mut_group()
                            .maybe_send_streaming_response(seq, pipeline_name.clone())
                            .await
                            .is_err()
                        {
                            // If we can't send the response, cancel the sequence
                            seq.set_state(SequenceState::Done(StopReason::Canceled));
                            get_mut_arcmutex!(pipeline).reset_non_granular_state();
                        }
                    }
                }
            } else if let Some(reason) = is_done {
                Self::finish_seq(pipeline.clone(), seq, reason, prefix_cacher).await?;
                get_mut_arcmutex!(pipeline).reset_non_granular_state();
            }
        }

        todo!()
    }
}
