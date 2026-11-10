use candle_core::{Result, Tensor};

use super::{AttentionMask, Sdpa, SdpaParams};
use crate::kv_cache::KvCache;
use crate::paged_attention::PagedAttention;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};

/// One layer's paged KV blocks plus the step metadata that addresses them.
pub(crate) type PagedLayer<'a> = ((Tensor, Tensor), &'a PagedAttentionInputMetadata);

/// Where a layer's keys and values live between steps.
pub(crate) enum KvStore<'a> {
    Contiguous(&'a mut KvCache),
    Paged {
        attn: &'a PagedAttention,
        layer: Option<PagedLayer<'a>>,
    },
}

impl<'a> KvStore<'a> {
    pub(crate) fn new(
        paged_attn: Option<&'a PagedAttention>,
        layer: Option<PagedLayer<'a>>,
        kv_cache: &'a mut KvCache,
    ) -> Self {
        match paged_attn {
            Some(attn) => Self::Paged { attn, layer },
            None => Self::Contiguous(kv_cache),
        }
    }
}

/// A self-attention call over `[batch, heads, seq, head_dim]` projections.
pub(crate) struct LayerAttention<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub mask: &'a AttentionMask,
    pub sdpa_params: &'a SdpaParams,
    pub flash_params: &'a FlashParams,
}

impl LayerAttention<'_> {
    /// Attends through `store` and returns `[batch, seq, heads * head_dim]`.
    pub(crate) fn run(&self, store: KvStore<'_>) -> Result<Tensor> {
        let y = match store {
            KvStore::Paged { attn, layer } => self.run_paged(attn, layer)?,
            KvStore::Contiguous(kv_cache) => {
                let (k, v) = kv_cache.append(self.k, self.v)?;
                Sdpa.run_attention(
                    self.q,
                    &k,
                    &v,
                    self.mask,
                    Some(self.flash_params),
                    self.sdpa_params,
                )?
            }
        };
        self.merge_heads(y)
    }

    /// Output keeps the backend's layout; see `merge_heads`.
    pub(crate) fn run_paged(
        &self,
        attn: &PagedAttention,
        layer: Option<PagedLayer<'_>>,
    ) -> Result<Tensor> {
        match layer {
            Some(((key_cache, value_cache), input_metadata)) => attn.forward(
                self.q,
                self.k,
                self.v,
                self.mask,
                Some(key_cache),
                Some(value_cache),
                input_metadata,
                self.sdpa_params,
                Some(self.flash_params),
            ),
            None => {
                // No cache blocks means prompt-only processing (imatrix collection), so nothing is stored.
                let input_metadata = PagedAttentionInputMetadata::dummy(self.q.device())?;
                assert!(!matches!(self.mask, AttentionMask::None));
                attn.forward(
                    self.q,
                    self.k,
                    self.v,
                    self.mask,
                    None,
                    None,
                    &input_metadata,
                    self.sdpa_params,
                    Some(self.flash_params),
                )
            }
        }
    }

    /// Masked steps come back as `[batch, heads, seq, head_dim]`, unmasked decode steps already head-merged.
    pub(crate) fn merge_heads(&self, y: Tensor) -> Result<Tensor> {
        let (batch, _, seq_len, _) = self.q.dims4()?;
        if matches!(self.mask, AttentionMask::None) {
            y.reshape((batch, seq_len, ()))
        } else {
            y.transpose(1, 2)?.reshape((batch, seq_len, ()))
        }
    }
}
