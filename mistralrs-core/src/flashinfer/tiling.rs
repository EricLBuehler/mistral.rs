// Matches the FlashInfer FA2 prefill GQA group specializations we dispatch to.
const PREFILL_MAX_GROUP_SIZE: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferPrefillTiling {
    group_size: usize,
    tile_q: usize,
}

impl FlashInferPrefillTiling {
    pub fn new(
        query_lens: &[usize],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Option<Self> {
        let group_size = Self::group_size_for(q_heads, kv_heads)?;
        // FlashInfer dispatches one CTA_TILE_Q for the whole batch from average packed QO length.
        let avg_packed_qo_len = if query_lens.is_empty() {
            0
        } else {
            Self::avg_packed_qo_len(query_lens, group_size)?
        };
        Some(Self {
            group_size,
            tile_q: Self::tile_q_for(avg_packed_qo_len, head_dim),
        })
    }

    pub fn group_size(self) -> usize {
        self.group_size
    }

    pub fn tile_q(self) -> usize {
        self.tile_q
    }

    pub fn supports_group_size(q_heads: usize, kv_heads: usize) -> bool {
        Self::group_size_for(q_heads, kv_heads).is_some()
    }

    pub fn tile_q_for(avg_packed_qo_len: usize, head_dim: usize) -> usize {
        // Mirrors FA2DetermineCtaTileQ for the Ampere+ FlashInfer prefill path.
        if avg_packed_qo_len > 64 && head_dim < 256 {
            128
        } else if avg_packed_qo_len > 16 {
            64
        } else {
            16
        }
    }

    fn group_size_for(q_heads: usize, kv_heads: usize) -> Option<usize> {
        (kv_heads != 0 && q_heads.is_multiple_of(kv_heads))
            .then_some(q_heads / kv_heads)
            .filter(|group_size| *group_size <= PREFILL_MAX_GROUP_SIZE)
    }

    fn avg_packed_qo_len(query_lens: &[usize], group_size: usize) -> Option<usize> {
        let sum_packed_qo_len = query_lens.iter().try_fold(0usize, |acc, len| {
            acc.checked_add(len.checked_mul(group_size)?)
        })?;
        Some(sum_packed_qo_len / query_lens.len())
    }
}
