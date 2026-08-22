mod default_scheduler;

use std::sync::Arc;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};
use tokio::sync::Mutex;

use crate::{
    engine::IntervalLogger,
    paged_attention::{
        block_hash::{BlockHash, MultimodalKind},
        CacheConfig, KVCacheManager, PagedAttentionScheduler, PagedAttentionSchedulerConfig,
        PagedAttentionSchedulerOutput,
    },
    sequence::Sequence,
};

pub(crate) const IMAGE_MODALITY: u8 = 1 << 0;
pub(crate) const AUDIO_MODALITY: u8 = 1 << 1;
pub(crate) const VIDEO_MODALITY: u8 = 1 << 2;
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 4096;
pub const DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL: usize = 8;

pub(crate) fn modality_signature(sequence: &Sequence) -> u8 {
    let mut signature = 0;
    if sequence.has_images()
        || sequence
            .mm_features()
            .iter()
            .any(|feature| feature.kind == MultimodalKind::Image)
    {
        signature |= IMAGE_MODALITY;
    }
    if sequence.has_audios()
        || sequence
            .mm_features()
            .iter()
            .any(|feature| feature.kind == MultimodalKind::Audio)
    {
        signature |= AUDIO_MODALITY;
    }
    if sequence.has_videos()
        || sequence
            .mm_features()
            .iter()
            .any(|feature| feature.kind == MultimodalKind::Video)
    {
        signature |= VIDEO_MODALITY;
    }
    signature
}

#[derive(Clone)]
pub enum SchedulerConfig {
    DefaultScheduler {
        method: DefaultSchedulerMethod,
    },
    PagedAttentionMeta {
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        max_decode_steps_before_prefill: usize,
        config: CacheConfig,
    },
}

impl SchedulerConfig {
    pub(crate) fn refresh_paged_cache_config(
        &mut self,
        realized_cache_config: Option<CacheConfig>,
    ) -> anyhow::Result<()> {
        match (self, realized_cache_config) {
            (Self::PagedAttentionMeta { config, .. }, Some(realized_cache_config)) => {
                *config = realized_cache_config;
                Ok(())
            }
            (Self::DefaultScheduler { .. }, None) => Ok(()),
            _ => anyhow::bail!(
                "reloaded pipeline PagedAttention mode does not match its scheduler configuration"
            ),
        }
    }

    pub fn into_scheduler(self) -> Arc<Mutex<dyn Scheduler>> {
        match self {
            Self::DefaultScheduler { method } => {
                Arc::new(Mutex::new(DefaultScheduler::new(method)))
            }
            Self::PagedAttentionMeta {
                max_num_seqs,
                max_num_batched_tokens,
                max_decode_steps_before_prefill,
                config,
            } => Arc::new(Mutex::new(PagedAttentionScheduler::new(
                PagedAttentionSchedulerConfig {
                    max_num_seqs,
                    max_num_batched_tokens,
                    max_decode_steps_before_prefill,
                },
                config,
            ))),
        }
    }
}

pub enum SchedulerOutput<'a> {
    DefaultScheduler {
        output: DefaultSchedulerOutput<'a>,
    },
    PagedAttention {
        output: PagedAttentionSchedulerOutput,
        preempted_sequence_ids: Vec<usize>,
    },
}

pub trait PagedPrefixCacheValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        seq: &mut Sequence,
        block_hashes: &[BlockHash],
        cached_tokens: usize,
        block_size: usize,
    ) -> usize;

    fn release_recurrent_state(&mut self, _slot_idx: usize) -> bool {
        false
    }
}

pub trait Scheduler: Send + Sync {
    fn schedule(
        &mut self,
        logger: &IntervalLogger,
        prefix_validator: Option<&mut dyn PagedPrefixCacheValidator>,
    ) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn running_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    /// This may do nothing. It depends on the implementation
    fn free_finished_sequence_groups(&mut self);
    /// Get recurrent state pool indices of finished sequences for freeing.
    /// Called before free_finished_sequence_groups to allow cleanup of hybrid cache slots.
    fn get_finished_recurrent_indices(&self) -> Vec<usize>;
    /// Get IDs of finished sequences before free_finished_sequence_groups removes them.
    fn get_finished_sequence_ids(&self) -> Vec<usize>;

    // PagedAttention metadata
    fn block_size(&self) -> Option<usize>;
    fn kv_cache_manager(&self) -> Option<Arc<Mutex<KVCacheManager>>>;

    /// Set whether prefix caching is enabled. Called by Engine after creation
    /// to synchronize with the global no_prefix_cache setting.
    fn set_prefix_caching_enabled(&mut self, enabled: bool);

    fn set_requires_uniform_prompt_batch(&mut self, _required: bool) {}

    fn set_requires_uniform_completion_batch(&mut self, _required: bool) {}

    fn set_requires_uniform_media_batch(&mut self, _required: bool) {}

    fn set_supports_packed_prefill(&mut self, _supported: bool) {}

    fn set_scheduler_visible_prompt_chunks(
        &mut self,
        _enabled: bool,
        _require_block_alignment: bool,
    ) {
    }

    fn can_continue_decode_batch(&self, _sequence_ids: &[usize]) -> bool {
        false
    }

    fn record_decode_continuation(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PagedCacheType;
    use std::num::NonZeroUsize;

    fn cache_config(num_gpu_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: 32,
            num_gpu_blocks,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0, 1],
        }
    }

    #[test]
    fn refresh_paged_cache_config_preserves_scheduler_limits() -> anyhow::Result<()> {
        let mut scheduler = SchedulerConfig::PagedAttentionMeta {
            max_num_seqs: 16,
            max_num_batched_tokens: 4096,
            max_decode_steps_before_prefill: 8,
            config: cache_config(128),
        };

        scheduler.refresh_paged_cache_config(Some(cache_config(256)))?;

        let SchedulerConfig::PagedAttentionMeta {
            max_num_seqs,
            max_num_batched_tokens,
            max_decode_steps_before_prefill,
            config,
        } = scheduler
        else {
            panic!("expected PagedAttention scheduler")
        };
        assert_eq!(max_num_seqs, 16);
        assert_eq!(max_num_batched_tokens, 4096);
        assert_eq!(max_decode_steps_before_prefill, 8);
        assert_eq!(config.num_gpu_blocks, 256);
        assert_eq!(config.kv_cache_group_ids, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn refresh_paged_cache_config_rejects_mode_mismatch() {
        let mut scheduler = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()),
        };
        assert!(scheduler
            .refresh_paged_cache_config(Some(cache_config(128)))
            .is_err());

        let mut scheduler = SchedulerConfig::PagedAttentionMeta {
            max_num_seqs: 16,
            max_num_batched_tokens: 4096,
            max_decode_steps_before_prefill: 8,
            config: cache_config(128),
        };
        assert!(scheduler.refresh_paged_cache_config(None).is_err());
    }
}
