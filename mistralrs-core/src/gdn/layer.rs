use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, Shard, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::kv_cache::{GdnDeferredStateSpec, GdnPendingTransitionSpec};
use crate::kv_cache::{RecurrentStateLayout, RecurrentStatePool};
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
#[cfg(feature = "cuda")]
use super::config::GdnVHeadLayout;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::packed::PackedGdnLayout;
use super::projection::{GdnCoreProjection, GdnInputProjection, GdnProjection};
use super::weights::{GdnInputProjectionKind, GdnWeightLoadCtx, GdnWeights};

/// Pre-convolution projected inputs of one forward, kept so a speculative rollback can re-advance
/// the recurrent state over an accepted prefix without re-reading any projection weights.
#[derive(Clone)]
pub struct GdnForwardStash {
    pub mixed_qkv: Tensor,
    pub convolved_qkv: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

#[derive(Clone)]
pub struct GdnTransitionStash;

#[derive(Clone)]
pub enum GdnSpeculativeStash {
    Replay(GdnForwardStash),
    Transition(GdnTransitionStash),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnTransitionCommitConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_width: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
}

pub(crate) struct GdnForwardContext<'a> {
    pub batch_kind: RecurrentBatchKind,
    pub checkpoint_lanes: usize,
    pub transition_checkpoints: bool,
    pub stash_out: Option<&'a mut Option<GdnSpeculativeStash>>,
}

enum GdnCoreOutput {
    Recurrent(Tensor),
    #[cfg(feature = "cuda")]
    Normalized(Tensor),
    #[cfg(feature = "cuda")]
    Quantized(mistralrs_quant::QuantizedActivation),
}

#[derive(Clone, Copy)]
struct GdnCoreContext {
    batch_size: usize,
    seq_len: usize,
    dtype: DType,
    batch_kind: RecurrentBatchKind,
}

fn exact_ragged_conv_state(
    padded_input: &Tensor,
    initial_state: &Tensor,
    layout: &PackedGdnLayout,
    expected_width: usize,
) -> Result<Tensor> {
    let query_lens = layout.query_lens();
    let (batch_size, padded_len, conv_dim) = padded_input.dims3()?;
    let (state_batch, state_conv_dim, state_width) = initial_state.dims3()?;
    if query_lens.len() != batch_size
        || query_lens.contains(&0)
        || query_lens.iter().any(|&len| len > padded_len)
        || state_batch != batch_size
        || state_conv_dim != conv_dim
        || state_width != expected_width
        || state_width == 0
    {
        candle_core::bail!("padded GDN convolution state has incompatible dimensions");
    }

    if let Some(cu_seqlens) = layout.cu_seqlens(padded_input.device())? {
        if let Some(state) = crate::cuda::gdn::try_gdn_extract_ragged_conv_state_cuda(
            crate::cuda::gdn::GdnRaggedConvState {
                padded_input,
                initial_state,
                cu_seqlens,
                batch_size,
            },
        )? {
            return Ok(state);
        }
    }

    let mut rows = Vec::with_capacity(batch_size);
    for (batch_index, &query_len) in query_lens.iter().enumerate() {
        let input = padded_input
            .narrow(0, batch_index, 1)?
            .narrow(1, 0, query_len)?
            .transpose(1, 2)?;
        let row = if query_len >= state_width {
            input.narrow(2, query_len - state_width, state_width)?
        } else {
            let retained = initial_state.narrow(0, batch_index, 1)?.narrow(
                2,
                query_len,
                state_width - query_len,
            )?;
            Tensor::cat(&[&retained, &input], 2)?
        };
        rows.push(row);
    }
    Tensor::cat(&rows, 0)
}

fn index_select_rows(source: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if source.is_contiguous() {
        source.index_select(indices, 0)
    } else {
        source.contiguous()?.index_select(indices, 0)
    }
}

pub(crate) fn speculative_checkpoint_dims_supported(dims: &GdnDims) -> bool {
    let Some(key_dim) = dims.num_k_heads.checked_mul(dims.head_k_dim) else {
        return false;
    };
    let Some(value_dim) = dims.num_v_heads.checked_mul(dims.head_v_dim) else {
        return false;
    };
    let Some(conv_dim) = key_dim
        .checked_mul(2)
        .and_then(|key_dim| key_dim.checked_add(value_dim))
    else {
        return false;
    };
    dims.num_k_heads > 0
        && dims.num_v_heads > 0
        && dims.num_v_heads.is_multiple_of(dims.num_k_heads)
        && dims.head_k_dim > 0
        && dims.head_k_dim <= crate::cuda::gdn::GDN_SPEC_CHECKPOINT_MAX_K
        && dims.head_v_dim > 0
        && dims.conv_kernel_size > 0
        && dims.conv_kernel_size <= crate::cuda::gdn::GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH
        && dims.conv_dim == conv_dim
}

pub(crate) fn speculative_state_commit_dims_supported(dims: &GdnDims) -> bool {
    let Some(key_dim) = dims.num_k_heads.checked_mul(dims.head_k_dim) else {
        return false;
    };
    let Some(value_dim) = dims.num_v_heads.checked_mul(dims.head_v_dim) else {
        return false;
    };
    let Some(conv_dim) = key_dim
        .checked_mul(2)
        .and_then(|key_dim| key_dim.checked_add(value_dim))
    else {
        return false;
    };
    dims.num_k_heads > 0
        && dims.num_v_heads > 0
        && dims.num_v_heads.is_multiple_of(dims.num_k_heads)
        && dims.head_k_dim > 0
        && dims.head_k_dim <= crate::cuda::gdn::GDN_SPEC_COMMIT_MAX_K
        && dims.head_k_dim.is_multiple_of(32)
        && dims.head_v_dim > 0
        && dims.head_v_dim.is_multiple_of(4)
        && dims.conv_kernel_size > 0
        && dims.conv_dim == conv_dim
}

pub struct GatedDeltaNet {
    pub input_proj: GdnInputProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    out_proj_input_shard: Option<Shard>,
    dims: GdnDims,
}

impl GatedDeltaNet {
    pub(crate) fn is_dynamic_lora_active(&self) -> bool {
        self.input_proj.is_dynamic_lora_active() || self.out_proj.is_dynamic_lora_active()
    }

    pub(crate) fn speculative_checkpoints_supported(
        &self,
        pool: &RecurrentStatePool,
        activation_dtype: DType,
    ) -> bool {
        if !cfg!(feature = "cuda")
            || !pool.device().is_cuda()
            || !matches!(activation_dtype, DType::F16 | DType::BF16)
            || pool.conv_dtype() != activation_dtype
            || !crate::cuda::gdn::recurrent_state_dtype_supported(pool.recurrent_dtype())
            || !speculative_checkpoint_dims_supported(&self.dims)
        {
            return false;
        }
        let state_dims = match pool.state_layout() {
            RecurrentStateLayout::GdnKeyMajor => (self.dims.head_k_dim, self.dims.head_v_dim),
            RecurrentStateLayout::GdnValueMajor => (self.dims.head_v_dim, self.dims.head_k_dim),
            RecurrentStateLayout::Opaque => return false,
        };
        let physical_capacity = pool.physical_capacity();
        if physical_capacity == 0
            || !physical_capacity.is_multiple_of(pool.checkpoint_lanes())
            || self.conv1d_weight.dims() != [self.dims.conv_dim, 1, self.dims.conv_kernel_size]
            || self.conv1d_weight.dtype() != activation_dtype
            || self.a_log.dims() != [self.dims.num_v_heads]
            || self.dt_bias.dims() != [self.dims.num_v_heads]
            || pool.conv_state.dims()
                != [
                    physical_capacity,
                    self.dims.conv_dim,
                    self.dims.conv_kernel_size,
                ]
            || pool.recurrent_state.dims()
                != [
                    physical_capacity,
                    self.dims.num_v_heads,
                    state_dims.0,
                    state_dims.1,
                ]
            || pool.conv_state.dtype() != activation_dtype
            || pool.recurrent_state.dtype() != pool.recurrent_dtype()
            || !pool.conv_state.is_contiguous()
            || !pool.recurrent_state.is_contiguous()
        {
            return false;
        }
        [
            self.conv1d_weight.device(),
            self.a_log.device(),
            self.dt_bias.device(),
        ]
        .into_iter()
        .all(|device| device.same_device(pool.device()))
    }

    pub(crate) fn speculative_transitions_supported(
        &self,
        pool: &RecurrentStatePool,
        activation_dtype: DType,
    ) -> bool {
        self.speculative_checkpoints_supported(pool, activation_dtype)
            && pool.checkpoint_lanes() == 1
            && pool.state_layout() == RecurrentStateLayout::GdnValueMajor
            && self.dims.head_k_dim == crate::cuda::gdn::GDN_DECODE_K_DIM
            && self.dims.head_v_dim == crate::cuda::gdn::GDN_DECODE_V_DIM
            && self.dims.conv_kernel_size <= crate::cuda::gdn::GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH
            && self.norm.weight.dtype() == activation_dtype
            && self.norm.weight.device().same_device(pool.device())
    }

    pub(crate) fn deferred_decode_supported(
        &self,
        pool: &RecurrentStatePool,
        activation_dtype: DType,
    ) -> bool {
        self.speculative_checkpoints_supported(pool, activation_dtype)
            && pool.checkpoint_lanes() == 1
            && pool.state_layout() == RecurrentStateLayout::GdnValueMajor
            && pool.recurrent_dtype() == DType::F32
            && self.dims.head_k_dim == crate::cuda::gdn::GDN_DECODE_K_DIM
            && self.dims.head_v_dim == crate::cuda::gdn::GDN_DECODE_V_DIM
            && self.norm.weight.dtype() == activation_dtype
            && self.norm.weight.device().same_device(pool.device())
    }

    pub(crate) fn transition_commit_config(
        &self,
        pool: &RecurrentStatePool,
    ) -> GdnTransitionCommitConfig {
        GdnTransitionCommitConfig {
            num_k_heads: self.dims.num_k_heads,
            num_v_heads: self.dims.num_v_heads,
            head_k_dim: self.dims.head_k_dim,
            head_v_dim: self.dims.head_v_dim,
            conv_dim: self.dims.conv_dim,
            conv_width: self.dims.conv_kernel_size,
            tiled_v_heads: self.dims.v_head_layout == super::config::GdnVHeadLayout::Tiled,
            state_layout: pool.state_layout(),
        }
    }

    pub(crate) fn pending_transition_spec(&self, max_rows: usize) -> GdnPendingTransitionSpec {
        GdnPendingTransitionSpec {
            num_k_heads: self.dims.num_k_heads,
            max_rows,
        }
    }

    pub(crate) fn deferred_state_spec(&self) -> GdnDeferredStateSpec {
        GdnDeferredStateSpec {
            num_k_heads: self.dims.num_k_heads,
            depth: crate::cuda::gdn::GDN_DEFERRED_STATE_DEPTH,
        }
    }

    pub(crate) fn speculative_state_commit_supported(
        &self,
        stash: &GdnForwardStash,
        initial_conv_state: &Tensor,
        initial_recurrent_state: &Tensor,
        pool: &RecurrentStatePool,
    ) -> bool {
        let Ok((batch_size, seq_len, conv_dim)) = stash.mixed_qkv.dims3() else {
            return false;
        };
        if !cfg!(feature = "cuda")
            || batch_size == 0
            || seq_len == 0
            || conv_dim != self.dims.conv_dim
            || !pool.device().is_cuda()
            || !matches!(stash.mixed_qkv.dtype(), DType::F16 | DType::BF16)
            || !speculative_state_commit_dims_supported(&self.dims)
        {
            return false;
        }
        let state_dims = match pool.state_layout() {
            RecurrentStateLayout::GdnKeyMajor => (self.dims.head_k_dim, self.dims.head_v_dim),
            RecurrentStateLayout::GdnValueMajor => (self.dims.head_v_dim, self.dims.head_k_dim),
            RecurrentStateLayout::Opaque => return false,
        };
        let physical_capacity = pool.physical_capacity();
        if stash.convolved_qkv.dims() != [batch_size, seq_len, conv_dim]
            || stash.b.dims() != [batch_size, seq_len, self.dims.num_v_heads]
            || stash.a.dims() != [batch_size, seq_len, self.dims.num_v_heads]
            || initial_conv_state.dims() != [batch_size, conv_dim, self.dims.conv_kernel_size]
            || initial_recurrent_state.dims()
                != [
                    batch_size,
                    self.dims.num_v_heads,
                    state_dims.0,
                    state_dims.1,
                ]
            || self.a_log.dims() != [self.dims.num_v_heads]
            || self.dt_bias.dims() != [self.dims.num_v_heads]
            || pool.conv_state.dims() != [physical_capacity, conv_dim, self.dims.conv_kernel_size]
            || pool.recurrent_state.dims()
                != [
                    physical_capacity,
                    self.dims.num_v_heads,
                    state_dims.0,
                    state_dims.1,
                ]
        {
            return false;
        }
        let activation_dtype = stash.mixed_qkv.dtype();
        if [
            &stash.convolved_qkv,
            &stash.b,
            &stash.a,
            initial_conv_state,
            &pool.conv_state,
        ]
        .into_iter()
        .any(|tensor| tensor.dtype() != activation_dtype)
            || !crate::cuda::gdn::recurrent_state_dtype_supported(initial_recurrent_state.dtype())
            || initial_recurrent_state.dtype() != pool.recurrent_state.dtype()
            || !pool.conv_state.is_contiguous()
            || !pool.recurrent_state.is_contiguous()
        {
            return false;
        }
        [
            &stash.mixed_qkv,
            &stash.convolved_qkv,
            &stash.b,
            &stash.a,
            initial_conv_state,
            initial_recurrent_state,
            &self.a_log,
            &self.dt_bias,
            &pool.conv_state,
            &pool.recurrent_state,
        ]
        .into_iter()
        .all(|tensor| tensor.device().same_device(pool.device()))
    }

    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<Comm>,
        input_projection_kind: GdnInputProjectionKind,
    ) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let weights = GdnWeights::load(
            vb,
            GdnWeightLoadCtx {
                cfg,
                dims: &dims,
                mapper,
                layer_idx,
                loading_isq,
                comm,
                input_projection_kind,
            },
        )?;
        let out_proj_input_shard = (comm.world_size() > 1).then_some(Shard::Simple {
            dim: 2,
            rank: comm.rank(),
            world_size: comm.world_size(),
        });
        Ok(Self {
            input_proj: weights.input_proj,
            conv1d_weight: weights.conv1d_weight,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            norm: weights.norm,
            out_proj: weights.out_proj,
            out_proj_input_shard,
            dims,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        self.forward_with_stash(x, cache, batch_kind, 1, false, None)
    }

    pub fn forward_with_stash(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        checkpoint_lanes: usize,
        transition_checkpoints: bool,
        stash_out: Option<&mut Option<GdnSpeculativeStash>>,
    ) -> Result<Tensor> {
        self.forward_with_context(
            x,
            cache,
            GdnForwardContext {
                batch_kind,
                checkpoint_lanes,
                transition_checkpoints,
                stash_out,
            },
        )
    }

    pub(crate) fn forward_with_context(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        context: GdnForwardContext<'_>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let projected = self.project(x, batch_size, seq_len)?;
        self.forward_projected_with_context(x, projected, cache, context)
    }

    pub(crate) fn input_activation_quantization_scheme_for(
        &self,
        x: &Tensor,
    ) -> Option<mistralrs_quant::ActivationQuantizationScheme> {
        self.input_proj.activation_quantization_scheme_for(x)
    }

    pub(crate) fn preferred_input_activation_scale_layout_for(
        &self,
        x: &Tensor,
    ) -> Option<mistralrs_quant::ActivationScaleLayout> {
        self.input_proj.preferred_activation_scale_layout_for(x)
    }

    pub(crate) fn forward_quantized_with_context(
        &self,
        x: &Tensor,
        activation: &mistralrs_quant::QuantizedActivation,
        cache: &mut GdnLayerCache,
        context: GdnForwardContext<'_>,
    ) -> Result<Option<Tensor>> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let Some(projected) = self
            .input_proj
            .try_forward_quantized(x, activation, &self.dims, batch_size, seq_len)?
        else {
            return Ok(None);
        };
        self.forward_projected_with_context(x, projected, cache, context)
            .map(Some)
    }

    fn forward_projected_with_context(
        &self,
        x: &Tensor,
        projected: GdnProjection,
        cache: &mut GdnLayerCache,
        context: GdnForwardContext<'_>,
    ) -> Result<Tensor> {
        let GdnForwardContext {
            batch_kind,
            checkpoint_lanes,
            transition_checkpoints,
            stash_out,
        } = context;
        #[cfg(not(feature = "cuda"))]
        let _ = (checkpoint_lanes, transition_checkpoints);
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        #[cfg(feature = "cuda")]
        let accelerated = if checkpoint_lanes > 1
            && batch_kind == RecurrentBatchKind::SpeculativeDecode
            && mixed_qkv.device().is_cuda()
            && cache.slots.is_some()
        {
            Some(self.forward_speculative_checkpoints(
                &mixed_qkv,
                &projected,
                cache,
                checkpoint_lanes,
                transition_checkpoints,
            )?)
        } else if crate::cuda::gdn::deferred_decode_batch_supported(batch_size)
            && batch_kind == RecurrentBatchKind::Decode
            && seq_len == 1
            && mixed_qkv.dtype() == DType::BF16
            && cache.slots.is_some()
            && cache.deferred_state.is_some()
            && cache.state_layout == RecurrentStateLayout::GdnValueMajor
            && cache.recurrent_state.dtype() == DType::F32
            && self.dims.head_k_dim == crate::cuda::gdn::GDN_DECODE_K_DIM
            && self.dims.head_v_dim == crate::cuda::gdn::GDN_DECODE_V_DIM
        {
            Some(self.forward_deferred_decode(&mixed_qkv, &projected, cache)?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let accelerated: Option<(Tensor, GdnCoreOutput, Option<GdnTransitionStash>)> = None;
        let (convolved_qkv, output, transitions) = match accelerated {
            Some(accelerated) => accelerated,
            None => {
                let core_projection = projected.core_projection();
                let (convolved_qkv, y) = self.forward_recurrent_core(
                    &mixed_qkv,
                    &core_projection,
                    cache,
                    GdnCoreContext {
                        batch_size,
                        seq_len,
                        dtype,
                        batch_kind,
                    },
                )?;
                (convolved_qkv, GdnCoreOutput::Recurrent(y), None)
            }
        };
        if let Some(stash_out) = stash_out {
            *stash_out = Some(match transitions {
                Some(transitions) => GdnSpeculativeStash::Transition(transitions),
                None => GdnSpeculativeStash::Replay(GdnForwardStash {
                    mixed_qkv,
                    convolved_qkv: convolved_qkv.clone(),
                    b: projected.b.clone(),
                    a: projected.a.clone(),
                }),
            });
        }

        self.finish_forward(output, projected.z, batch_size, seq_len, dtype)
    }

    fn forward_recurrent_core(
        &self,
        mixed_qkv: &Tensor,
        projected: &GdnCoreProjection,
        cache: &mut GdnLayerCache,
        ctx: GdnCoreContext,
    ) -> Result<(Tensor, Tensor)> {
        let convolved_qkv = backend::causal_conv1d(
            mixed_qkv,
            &self.conv1d_weight,
            &self.dims,
            cache,
            ctx.batch_kind,
        )?;
        let y = backend::apply_recurrence_from_convolved(
            &convolved_qkv,
            &projected.b,
            &projected.a,
            &self.a_log,
            &self.dt_bias,
            &self.dims,
            ctx.batch_size,
            ctx.seq_len,
            cache,
            ctx.dtype,
        )?;
        Ok((convolved_qkv, y))
    }

    pub(crate) fn forward_projected_prefill_core(
        &self,
        projected: &GdnCoreProjection,
        cache: &mut GdnLayerCache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = projected.b.dims3()?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        let (_, y) = self.forward_recurrent_core(
            &mixed_qkv,
            projected,
            cache,
            GdnCoreContext {
                batch_size,
                seq_len,
                dtype: projected.b.dtype(),
                batch_kind: RecurrentBatchKind::Prefill,
            },
        )?;
        Ok(y)
    }

    pub(crate) fn forward_projected_padded_prefill_core(
        &self,
        projected: &GdnCoreProjection,
        cache: &mut GdnLayerCache,
        layout: &PackedGdnLayout,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = projected.b.dims3()?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        let exact_conv_state = exact_ragged_conv_state(
            &mixed_qkv,
            &cache.conv_state,
            layout,
            self.dims.conv_kernel_size,
        )?;
        let (_, y) = self.forward_recurrent_core(
            &mixed_qkv,
            projected,
            cache,
            GdnCoreContext {
                batch_size,
                seq_len,
                dtype: projected.b.dtype(),
                batch_kind: RecurrentBatchKind::Prefill,
            },
        )?;
        cache.conv_state = exact_conv_state;
        Ok(y)
    }

    #[cfg(feature = "cuda")]
    fn deferred_recurrence_context<'a>(
        &'a self,
        mixed_qkv: &'a Tensor,
        projected: &'a GdnProjection,
        cache: &'a GdnLayerCache,
    ) -> Result<crate::cuda::gdn::GdnDeferredRecurrence<'a>> {
        let active_slots = cache
            .slots
            .as_ref()
            .expect("deferred decode requires pooled state");
        let deferred = cache
            .deferred_state
            .as_ref()
            .expect("deferred decode requires transition storage");
        Ok(crate::cuda::gdn::GdnDeferredRecurrence {
            mixed_qkv,
            b: &projected.b,
            a: &projected.a,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            state_pool: &cache.recurrent_state,
            active_slots,
            deferred_key: &deferred.key,
            deferred_delta: &deferred.delta,
            deferred_decay: &deferred.decay,
            deferred_cursor: &deferred.pending_rows,
            gate: &projected.z,
            norm_weight: &self.norm.weight,
            norm_eps: self.norm.eps(),
            num_k_heads: self.dims.num_k_heads,
            num_v_heads: self.dims.num_v_heads,
            head_k_dim: self.dims.head_k_dim,
            head_v_dim: self.dims.head_v_dim,
            tiled_v_heads: self.dims.v_head_layout == GdnVHeadLayout::Tiled,
            state_layout: cache.state_layout,
            quantization: self.fp8_output_spec(&projected.z, mixed_qkv.dim(0)?, 1),
        })
    }

    #[cfg(feature = "cuda")]
    fn deferred_output(output: crate::cuda::gdn::GdnPostOpOutput) -> GdnCoreOutput {
        match output {
            crate::cuda::gdn::GdnPostOpOutput::Tensor(output) => GdnCoreOutput::Normalized(output),
            crate::cuda::gdn::GdnPostOpOutput::Quantized(output) => {
                GdnCoreOutput::Quantized(output)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_deferred_decode(
        &self,
        mixed_qkv: &Tensor,
        projected: &GdnProjection,
        cache: &mut GdnLayerCache,
    ) -> Result<(Tensor, GdnCoreOutput, Option<GdnTransitionStash>)> {
        let convolved_qkv = backend::causal_conv1d(
            mixed_qkv,
            &self.conv1d_weight,
            &self.dims,
            cache,
            RecurrentBatchKind::Decode,
        )?;
        let output = crate::cuda::gdn::deferred_recurrence_rmsnorm_gate_cuda(
            self.deferred_recurrence_context(&convolved_qkv, projected, cache)?,
        )?;
        let output = Self::deferred_output(output);
        Ok((convolved_qkv, output, None))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn flush_deferred_state(
        &self,
        pool: &RecurrentStatePool,
        active_slots: &Tensor,
        activation_dtype: DType,
    ) -> Result<bool> {
        if !self.deferred_decode_supported(pool, activation_dtype) {
            return Ok(false);
        }
        let Some(deferred) = pool.deferred_state() else {
            return Ok(false);
        };
        crate::cuda::gdn::flush_deferred_state_cuda(crate::cuda::gdn::GdnDeferredStateFlush {
            state_pool: &pool.recurrent_state,
            active_slots,
            deferred_key: &deferred.key,
            deferred_delta: &deferred.delta,
            deferred_decay: &deferred.decay,
            deferred_cursor: &deferred.pending_rows,
            num_k_heads: self.dims.num_k_heads,
            num_v_heads: self.dims.num_v_heads,
            head_k_dim: self.dims.head_k_dim,
            head_v_dim: self.dims.head_v_dim,
            tiled_v_heads: self.dims.v_head_layout == GdnVHeadLayout::Tiled,
            state_layout: pool.state_layout(),
        })?;
        Ok(true)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn flush_deferred_state(
        &self,
        _pool: &RecurrentStatePool,
        _active_slots: &Tensor,
        _activation_dtype: DType,
    ) -> Result<bool> {
        Ok(false)
    }

    #[cfg(feature = "cuda")]
    fn forward_speculative_checkpoints(
        &self,
        mixed_qkv: &Tensor,
        projected: &GdnProjection,
        cache: &GdnLayerCache,
        checkpoint_lanes: usize,
        transition_checkpoints: bool,
    ) -> Result<(Tensor, GdnCoreOutput, Option<GdnTransitionStash>)> {
        let (batch_size, seq_len, _) = mixed_qkv.dims3()?;
        let active_slots = cache
            .slots
            .as_ref()
            .expect("checkpoint path requires pooled state");
        let weight = self
            .conv1d_weight
            .squeeze(1)?
            .to_dtype(mixed_qkv.dtype())?
            .contiguous()?;
        let conv_input = mixed_qkv.clone();
        let pending_conv = cache.pending_transitions.as_ref().map(|pending| {
            crate::cuda::gdn::GdnPendingSpeculativeConv {
                conv_input: &pending.conv_input,
                keep_rows: &pending.keep_rows,
                pending_epochs: &pending.pending_epochs,
                applied_epochs: &pending.conv_applied_epochs,
            }
        });
        let convolved_qkv = crate::cuda::gdn::speculative_conv_checkpoints_cuda(
            crate::cuda::gdn::GdnSpeculativeConvCheckpoints {
                x: &conv_input,
                weight: &weight,
                state_pool: &cache.conv_state,
                active_slots,
                checkpoint_lanes: if transition_checkpoints {
                    1
                } else {
                    checkpoint_lanes
                },
                write_checkpoints: !transition_checkpoints,
                pending: pending_conv,
            },
        )?;
        let fused_norm = cache.state_layout == RecurrentStateLayout::GdnValueMajor
            && self.dims.head_k_dim == crate::cuda::gdn::GDN_DECODE_K_DIM
            && self.dims.head_v_dim == crate::cuda::gdn::GDN_DECODE_V_DIM
            && seq_len <= crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS
            && projected.z.dtype() == mixed_qkv.dtype()
            && self.norm.weight.dtype() == mixed_qkv.dtype();
        let post_op = fused_norm.then_some(crate::cuda::gdn::GdnSpeculativeRmsNormGate {
            gate: &projected.z,
            weight: &self.norm.weight,
            eps: self.norm.eps(),
            quantization: self.fp8_output_spec(&projected.z, batch_size, seq_len),
        });
        let pending_recurrence = cache.pending_transitions.as_ref().map(|pending| {
            crate::cuda::gdn::GdnPendingSpeculativeRecurrence {
                key_banks: &pending.key_banks,
                key_bank: &pending.key_bank,
                delta: &pending.delta,
                decay: &pending.decay,
                keep_rows: &pending.keep_rows,
                pending_epochs: &pending.pending_epochs,
                applied_epochs: &pending.recurrent_applied_epochs,
            }
        });
        let recurrence = crate::cuda::gdn::speculative_recurrence_checkpoints_cuda(
            crate::cuda::gdn::GdnSpeculativeRecurrenceCheckpoints {
                mixed_qkv: &convolved_qkv,
                b: &projected.b,
                a: &projected.a,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                state_pool: &cache.recurrent_state,
                active_slots,
                checkpoint_lanes: if transition_checkpoints {
                    1
                } else {
                    checkpoint_lanes
                },
                num_k_heads: self.dims.num_k_heads,
                num_v_heads: self.dims.num_v_heads,
                head_k_dim: self.dims.head_k_dim,
                head_v_dim: self.dims.head_v_dim,
                tiled_v_heads: self.dims.v_head_layout == GdnVHeadLayout::Tiled,
                state_layout: cache.state_layout,
                post_op,
                record_transitions: transition_checkpoints,
                pending: pending_recurrence,
            },
        )?;
        let transitions = transition_checkpoints.then_some(GdnTransitionStash);
        let output = match recurrence.output {
            crate::cuda::gdn::GdnPostOpOutput::Tensor(output) if fused_norm => {
                GdnCoreOutput::Normalized(output)
            }
            crate::cuda::gdn::GdnPostOpOutput::Tensor(output) => {
                let output = output
                    .reshape((
                        batch_size,
                        self.dims.num_v_heads,
                        seq_len,
                        self.dims.head_v_dim,
                    ))?
                    .transpose(1, 2)?;
                GdnCoreOutput::Recurrent(output)
            }
            crate::cuda::gdn::GdnPostOpOutput::Quantized(output) => {
                GdnCoreOutput::Quantized(output)
            }
        };
        Ok((convolved_qkv, output, transitions))
    }

    /// Advance a gathered cache over the selected rows of a stashed forward without rerunning projections.
    pub fn advance_state_batch_from_stash(
        &self,
        stash: &GdnForwardStash,
        batch_indices: &Tensor,
        rows: usize,
        cache: &mut GdnLayerCache,
    ) -> Result<()> {
        let batch_size = batch_indices.dim(0)?;
        let mixed_qkv = index_select_rows(&stash.mixed_qkv.narrow(1, 0, rows)?, batch_indices)?;
        let b = index_select_rows(&stash.b.narrow(1, 0, rows)?, batch_indices)?;
        let a = index_select_rows(&stash.a.narrow(1, 0, rows)?, batch_indices)?;
        let mixed_qkv = backend::causal_conv1d(
            &mixed_qkv,
            &self.conv1d_weight,
            &self.dims,
            cache,
            RecurrentBatchKind::Prefill,
        )?;
        backend::apply_recurrence_from_convolved(
            &mixed_qkv,
            &b,
            &a,
            &self.a_log,
            &self.dt_bias,
            &self.dims,
            batch_size,
            rows,
            cache,
            stash.mixed_qkv.dtype(),
        )?;
        Ok(())
    }

    pub fn commit_state_batch_from_stash_cuda(
        &self,
        stash: &GdnForwardStash,
        initial_conv_state: &Tensor,
        initial_recurrent_state: &Tensor,
        keep_rows: &Tensor,
        slots: &Tensor,
        pool: &RecurrentStatePool,
    ) -> Result<bool> {
        #[cfg(feature = "cuda")]
        if stash.mixed_qkv.device().is_cuda() {
            crate::cuda::gdn::speculative_state_commit_cuda(
                crate::cuda::gdn::GdnSpeculativeStateCommit {
                    mixed_qkv: &stash.mixed_qkv,
                    convolved_qkv: &stash.convolved_qkv,
                    b: &stash.b,
                    a: &stash.a,
                    initial_conv_state,
                    initial_recurrent_state,
                    a_log: &self.a_log,
                    dt_bias: &self.dt_bias,
                    conv_state_pool: &pool.conv_state,
                    recurrent_state_pool: &pool.recurrent_state,
                    keep_rows,
                    slot_indices: slots,
                    num_k_heads: self.dims.num_k_heads,
                    num_v_heads: self.dims.num_v_heads,
                    head_k_dim: self.dims.head_k_dim,
                    head_v_dim: self.dims.head_v_dim,
                    tiled_v_heads: self.dims.v_head_layout == super::config::GdnVHeadLayout::Tiled,
                    state_layout: pool.state_layout(),
                },
            )?;
            return Ok(true);
        }

        let _ = (
            stash,
            initial_conv_state,
            initial_recurrent_state,
            keep_rows,
            slots,
            pool,
        );
        Ok(false)
    }

    pub(crate) fn project(
        &self,
        x: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GdnProjection> {
        self.input_proj.forward(x, &self.dims, batch_size, seq_len)
    }

    #[cfg(all(feature = "cuda", has_gdn_fp8_producer))]
    fn fp8_output_spec(
        &self,
        gate: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Option<crate::cuda::gdn::GdnFp8OutputSpec> {
        if self.out_proj_input_shard.is_some()
            || gate.dtype() != DType::BF16
            || !gate.device().is_cuda()
            || self.norm.weight.dtype() != DType::BF16
            || !self.norm.weight.device().same_device(gate.device())
            || self.out_proj.is_dynamic_lora_active()
            || self.out_proj.stats_snapshot().is_some()
        {
            return None;
        }
        let source_shape = [batch_size, seq_len, self.dims.value_dim];
        let reshaped;
        let source = if gate.dims() == source_shape {
            gate
        } else {
            reshaped = gate.reshape(&source_shape).ok()?;
            &reshaped
        };
        let scheme = self.out_proj.activation_quantization_scheme_for(source)?;
        let scale_layout = self
            .out_proj
            .preferred_activation_scale_layout_for(source)?;
        crate::cuda::gdn::GdnFp8OutputSpec::new(
            [batch_size, seq_len, self.dims.value_dim],
            scheme,
            scale_layout,
            self.dims.num_v_heads,
            self.dims.head_v_dim,
        )
    }

    #[cfg(all(feature = "cuda", not(has_gdn_fp8_producer)))]
    fn fp8_output_spec(
        &self,
        _gate: &Tensor,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Option<crate::cuda::gdn::GdnFp8OutputSpec> {
        None
    }

    pub(crate) fn finish_projected_recurrent(&self, output: Tensor, z: Tensor) -> Result<Tensor> {
        let batch_size = z.dim(0)?;
        let seq_len = z.dim(1)?;
        let dtype = z.dtype();
        self.finish_forward(
            GdnCoreOutput::Recurrent(output),
            z,
            batch_size,
            seq_len,
            dtype,
        )
    }

    fn finish_forward(
        &self,
        output: GdnCoreOutput,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        _dtype: DType,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        let y = match output {
            GdnCoreOutput::Recurrent(y) => {
                if let Some(spec) = self.fp8_output_spec(&z, batch_size, seq_len) {
                    let activation = self.norm.forward_quantized(
                        &y,
                        &z,
                        &spec,
                        self.dims.num_v_heads,
                        self.dims.head_v_dim,
                    )?;
                    return self.out_proj.forward_quantized(&activation);
                }
                self.norm.forward(&y, &z)?
            }
            GdnCoreOutput::Normalized(y) => y,
            GdnCoreOutput::Quantized(activation) => {
                return self.out_proj.forward_quantized(&activation)
            }
        };
        #[cfg(not(feature = "cuda"))]
        let y = match output {
            GdnCoreOutput::Recurrent(y) => self.norm.forward(&y, &z)?,
        };
        let y = y.reshape((batch_size, seq_len, self.dims.value_dim))?;
        let y = shard_out_proj_input(y, self.out_proj_input_shard)?;
        self.out_proj.forward(&y)
    }
}

fn shard_out_proj_input(y: Tensor, shard: Option<Shard>) -> Result<Tensor> {
    match shard {
        Some(shard) => shard.apply_to(&y)?.contiguous(),
        None => Ok(y),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethod, QuantMethodConfig, Shard, UnquantLinear};

    use super::super::config::GdnVHeadLayout;
    use super::super::norm::RmsNormGated;
    use super::super::projection::GdnInputProjection;
    use super::super::{try_forward_grouped_packed_gdn, GdnLayerCache, PackedGdnLayout};
    use super::{
        index_select_rows, shard_out_proj_input, speculative_checkpoint_dims_supported,
        speculative_state_commit_dims_supported, GatedDeltaNet, GdnDims,
    };
    use crate::kv_cache::RecurrentStateLayout;
    use crate::pipeline::RecurrentBatchKind;

    const PACKED_ASSERT_EPS: f32 = 1e-6;

    fn checkpoint_dims() -> GdnDims {
        GdnDims {
            hidden_size: 64,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 8,
            head_v_dim: 16,
            conv_kernel_size: 4,
            key_dim: 16,
            value_dim: 64,
            conv_dim: 96,
            v_per_group: 2,
            v_head_layout: GdnVHeadLayout::Grouped,
        }
    }

    struct PackedTestGdn {
        gdn: GatedDeltaNet,
        projections: Vec<Arc<dyn QuantMethod>>,
    }

    fn packed_dims() -> GdnDims {
        GdnDims {
            hidden_size: 4,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_kernel_size: 2,
            key_dim: 2,
            value_dim: 2,
            conv_dim: 6,
            v_per_group: 1,
            v_head_layout: GdnVHeadLayout::Grouped,
        }
    }

    fn test_weight(rows: usize, cols: usize, phase: usize) -> candle_core::Result<Tensor> {
        let values = (0..rows * cols)
            .map(|index| (((index + phase) % 11) as f32 - 5.0) * 0.03)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu)
    }

    fn test_linear(
        rows: usize,
        cols: usize,
        phase: usize,
    ) -> candle_core::Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(test_weight(rows, cols, phase)?, None)),
        )?))
    }

    fn packed_test_gdn() -> candle_core::Result<PackedTestGdn> {
        let dims = packed_dims();
        let qkv = test_linear(dims.conv_dim, dims.hidden_size, 0)?;
        let z = test_linear(dims.value_dim, dims.hidden_size, 2)?;
        let b = test_linear(dims.num_v_heads, dims.hidden_size, 4)?;
        let a = test_linear(dims.num_v_heads, dims.hidden_size, 6)?;
        let out = test_linear(dims.hidden_size, dims.value_dim, 8)?;
        let projections = vec![qkv.clone(), z.clone(), b.clone(), a.clone(), out.clone()];
        let conv1d_weight = Tensor::from_vec(
            (0..dims.conv_dim * dims.conv_kernel_size)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.04)
                .collect::<Vec<_>>(),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &Device::Cpu,
        )?;
        let gdn = GatedDeltaNet {
            input_proj: GdnInputProjection::Split {
                in_proj_qkv: qkv,
                in_proj_z: z,
                in_proj_b: b,
                in_proj_a: a,
                merged_qkv_z: None,
                merged_b_a: None,
            },
            conv1d_weight,
            dt_bias: Tensor::new(&[0.1f32], &Device::Cpu)?,
            a_log: Tensor::new(&[-0.2f32], &Device::Cpu)?,
            norm: RmsNormGated::from_parts(
                Tensor::ones(dims.value_dim, DType::F32, &Device::Cpu)?,
                1e-6,
            ),
            out_proj: out,
            out_proj_input_shard: None,
            dims,
        };
        Ok(PackedTestGdn { gdn, projections })
    }

    fn packed_test_cache(logical_batch: usize) -> candle_core::Result<GdnLayerCache> {
        let dims = packed_dims();
        let conv_state = Tensor::from_vec(
            (0..logical_batch * dims.conv_dim * dims.conv_kernel_size)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.01)
                .collect::<Vec<_>>(),
            (logical_batch, dims.conv_dim, dims.conv_kernel_size),
            &Device::Cpu,
        )?;
        let recurrent_state = Tensor::from_vec(
            (0..logical_batch * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim)
                .map(|index| ((index % 9) as f32 - 4.0) * 0.02)
                .collect::<Vec<_>>(),
            (
                logical_batch,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            &Device::Cpu,
        )?;
        Ok(GdnLayerCache::gathered(
            conv_state,
            recurrent_state,
            RecurrentStateLayout::GdnKeyMajor,
        ))
    }

    fn reference_ragged_forward(
        gdn: &GatedDeltaNet,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        query_lens: &[usize],
    ) -> candle_core::Result<Tensor> {
        let mut offset = 0;
        let mut outputs = Vec::with_capacity(query_lens.len());
        let mut conv_states = Vec::with_capacity(query_lens.len());
        let mut recurrent_states = Vec::with_capacity(query_lens.len());
        for (state_index, &query_len) in query_lens.iter().enumerate() {
            let mut row_cache = GdnLayerCache::gathered(
                cache.conv_state.narrow(0, state_index, 1)?,
                cache.recurrent_state.narrow(0, state_index, 1)?,
                cache.state_layout,
            );
            outputs.push(gdn.forward(
                &x.narrow(1, offset, query_len)?,
                &mut row_cache,
                RecurrentBatchKind::Prefill,
            )?);
            conv_states.push(row_cache.conv_state);
            recurrent_states.push(row_cache.recurrent_state);
            offset += query_len;
        }
        cache.conv_state = Tensor::cat(&conv_states, 0)?;
        cache.recurrent_state = Tensor::cat(&recurrent_states, 0)?;
        Tensor::cat(&outputs, 1)
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
    ) -> candle_core::Result<()> {
        assert_eq!(actual.dims(), expected.dims(), "{label} shape");
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{label} actual"
        );
        assert!(
            expected.iter().all(|value| value.is_finite()),
            "{label} expected"
        );
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= PACKED_ASSERT_EPS,
            "{label} max error {max_error}"
        );
        Ok(())
    }

    fn assert_tensor_exact(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
    ) -> candle_core::Result<()> {
        assert_eq!(actual.dims(), expected.dims(), "{label} shape");
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{label} actual"
        );
        assert!(
            expected.iter().all(|value| value.is_finite()),
            "{label} expected"
        );
        assert_eq!(actual, expected, "{label}");
        Ok(())
    }

    fn assert_packed_ragged_equivalence(query_lens: &[usize]) -> candle_core::Result<()> {
        let PackedTestGdn { gdn, projections } = packed_test_gdn()?;
        let token_count = query_lens.iter().sum::<usize>();
        let x = Tensor::from_vec(
            (0..token_count * packed_dims().hidden_size)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.025)
                .collect::<Vec<_>>(),
            (1, token_count, packed_dims().hidden_size),
            &Device::Cpu,
        )?;
        let initial_cache = packed_test_cache(query_lens.len())?;
        let mut reference_cache = initial_cache.clone();
        let expected = reference_ragged_forward(&gdn, &x, &mut reference_cache, query_lens)?;

        for projection in &projections {
            projection.begin_track_stats()?;
        }
        let mut packed_cache = initial_cache;
        let layout = PackedGdnLayout::new(query_lens.to_vec(), HashMap::new())?;
        let actual = try_forward_grouped_packed_gdn(&gdn, &x, &mut packed_cache, &layout)?
            .expect("ragged packed GDN should use the projected path");

        for projection in projections {
            assert_eq!(projection.stats_snapshot(), Some((1, token_count)));
        }
        assert_tensor_close("output", &actual, &expected)?;
        assert_tensor_exact(
            "convolution state",
            &packed_cache.conv_state,
            &reference_cache.conv_state,
        )?;
        assert_tensor_close(
            "recurrent state",
            &packed_cache.recurrent_state,
            &reference_cache.recurrent_state,
        )
    }

    #[test]
    fn packed_ragged_projects_once_and_matches_per_sequence_forward() -> candle_core::Result<()> {
        assert_packed_ragged_equivalence(&[2, 3, 2])?;
        assert_packed_ragged_equivalence(&[1, 2, 3])?;
        assert_packed_ragged_equivalence(&[7, 8, 9])
    }

    #[test]
    fn padded_core_preserves_states_across_conv_width_boundaries() -> candle_core::Result<()> {
        let query_lens = [1, 2, 3];
        let PackedTestGdn { gdn, .. } = packed_test_gdn()?;
        let token_count = query_lens.iter().sum::<usize>();
        let x = Tensor::from_vec(
            (0..token_count * packed_dims().hidden_size)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.025)
                .collect::<Vec<_>>(),
            (1, token_count, packed_dims().hidden_size),
            &Device::Cpu,
        )?;
        let initial_cache = packed_test_cache(query_lens.len())?;
        let mut reference_cache = initial_cache.clone();
        let expected = reference_ragged_forward(&gdn, &x, &mut reference_cache, &query_lens)?;

        let projected = gdn.project(&x, 1, token_count)?;
        let layout = PackedGdnLayout::new(query_lens.to_vec(), HashMap::new())?;
        let padded_projection = projected.pad_core_packed(&layout, layout.max_seq_len())?;
        let mut padded_cache = initial_cache;
        let padded_output = gdn.forward_projected_padded_prefill_core(
            &padded_projection,
            &mut padded_cache,
            &layout,
        )?;
        let output_rows = query_lens
            .iter()
            .enumerate()
            .map(|(batch_index, &query_len)| {
                padded_output
                    .narrow(0, batch_index, 1)?
                    .narrow(1, 0, query_len)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let output = Tensor::cat(&output_rows, 1)?;
        let actual = gdn.finish_projected_recurrent(output, projected.z)?;

        assert_tensor_close("padded output", &actual, &expected)?;
        assert_tensor_exact(
            "padded convolution state",
            &padded_cache.conv_state,
            &reference_cache.conv_state,
        )?;
        assert_tensor_close(
            "padded recurrent state",
            &padded_cache.recurrent_state,
            &reference_cache.recurrent_state,
        )
    }

    #[test]
    fn speculative_checkpoint_dims_reject_unsupported_shapes() {
        assert!(speculative_checkpoint_dims_supported(&checkpoint_dims()));

        let mut dims = checkpoint_dims();
        dims.conv_kernel_size = 17;
        assert!(!speculative_checkpoint_dims_supported(&dims));

        let mut dims = checkpoint_dims();
        dims.head_k_dim = 257;
        assert!(!speculative_checkpoint_dims_supported(&dims));

        let mut dims = checkpoint_dims();
        dims.num_v_heads = 3;
        assert!(!speculative_checkpoint_dims_supported(&dims));

        let mut dims = checkpoint_dims();
        dims.conv_dim += 1;
        assert!(!speculative_checkpoint_dims_supported(&dims));
    }

    #[test]
    fn speculative_state_commit_dims_reject_unsupported_shapes() {
        let mut dims = checkpoint_dims();
        dims.head_k_dim = 64;
        dims.key_dim = dims.num_k_heads * dims.head_k_dim;
        dims.conv_dim = 2 * dims.key_dim + dims.value_dim;
        assert!(speculative_state_commit_dims_supported(&dims));

        let mut unsupported = dims;
        unsupported.head_k_dim = 48;
        assert!(!speculative_state_commit_dims_supported(&unsupported));

        let mut unsupported = dims;
        unsupported.head_v_dim = 14;
        assert!(!speculative_state_commit_dims_supported(&unsupported));

        let mut unsupported = dims;
        unsupported.conv_kernel_size = 0;
        assert!(!speculative_state_commit_dims_supported(&unsupported));
    }

    #[test]
    fn index_select_rows_accepts_a_non_contiguous_prefix() -> candle_core::Result<()> {
        let source = Tensor::from_vec(
            (0..24).map(|value| value as f32).collect::<Vec<_>>(),
            (3, 4, 2),
            &Device::Cpu,
        )?;
        let prefix = source.narrow(1, 0, 2)?;
        assert!(!prefix.is_contiguous());
        let indices = Tensor::from_vec(vec![2u32, 0], (2,), &Device::Cpu)?;
        let selected = index_select_rows(&prefix, &indices)?;
        assert_eq!(selected.dims(), &[2, 2, 2]);
        assert_eq!(
            selected.flatten_all()?.to_vec1::<f32>()?,
            vec![16., 17., 18., 19., 0., 1., 2., 3.]
        );
        Ok(())
    }

    #[test]
    fn gdn_row_parallel_projection_receives_its_activation_shard() -> candle_core::Result<()> {
        let input = Tensor::new(&[[[0f32, 1., 2., 3., 4., 5., 6., 7.]]], &Device::Cpu)?;
        let shard = Shard::Simple {
            dim: 2,
            rank: 1,
            world_size: 2,
        };
        let output = shard_out_proj_input(input, Some(shard))?;
        assert_eq!(output.dims(), &[1, 1, 4]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            vec![4., 5., 6., 7.]
        );
        Ok(())
    }

    #[test]
    fn gdn_tp_partials_match_the_full_output_projection() -> candle_core::Result<()> {
        let input = Tensor::new(&[[[0f32, 1., 2., 3., 4., 5., 6., 7.]]], &Device::Cpu)?;
        let weight = Tensor::new(
            &[
                [1f32, 2., 3., 4., 5., 6., 7., 8.],
                [8f32, 7., 6., 5., 4., 3., 2., 1.],
            ],
            &Device::Cpu,
        )?;
        let expected = input.reshape((1, 8))?.matmul(&weight.t()?)?;
        let mut partials = Vec::new();
        for rank in 0..2 {
            let input_shard = shard_out_proj_input(
                input.clone(),
                Some(Shard::Simple {
                    dim: 2,
                    rank,
                    world_size: 2,
                }),
            )?;
            let weight_shard = Shard::Simple {
                dim: 1,
                rank,
                world_size: 2,
            }
            .apply_to(&weight)?;
            partials.push(input_shard.reshape((1, 4))?.matmul(&weight_shard.t()?)?);
        }
        let actual = (&partials[0] + &partials[1])?;
        assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }
}
