use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, Shard, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::kv_cache::{RecurrentStateLayout, RecurrentStatePool};
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
#[cfg(feature = "cuda")]
use super::config::GdnVHeadLayout;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::{GdnInputProjection, GdnProjection};
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
    pub(crate) fn speculative_checkpoints_supported(
        &self,
        pool: &RecurrentStatePool,
        activation_dtype: DType,
    ) -> bool {
        if !cfg!(feature = "cuda")
            || !pool.device().is_cuda()
            || !matches!(activation_dtype, DType::F16 | DType::BF16)
            || pool.conv_dtype() != activation_dtype
            || pool.recurrent_dtype() != DType::F32
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
            || pool.recurrent_state.dtype() != DType::F32
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
            || initial_recurrent_state.dtype() != DType::F32
            || pool.recurrent_state.dtype() != DType::F32
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
        self.forward_with_stash(x, cache, batch_kind, 1, None)
    }

    pub fn forward_with_stash(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        checkpoint_lanes: usize,
        stash_out: Option<&mut Option<GdnForwardStash>>,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "cuda"))]
        let _ = checkpoint_lanes;
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let projected = self.project(x, batch_size, seq_len)?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        #[cfg(feature = "cuda")]
        let checkpointed = if checkpoint_lanes > 1
            && batch_kind == RecurrentBatchKind::SpeculativeDecode
            && mixed_qkv.device().is_cuda()
            && cache.slots.is_some()
        {
            Some(self.forward_speculative_checkpoints(
                &mixed_qkv,
                &projected,
                cache,
                checkpoint_lanes,
            )?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let checkpointed: Option<(Tensor, Tensor)> = None;
        let (convolved_qkv, y) = match checkpointed {
            Some(checkpointed) => checkpointed,
            None => {
                let convolved_qkv = backend::causal_conv1d(
                    &mixed_qkv,
                    &self.conv1d_weight,
                    &self.dims,
                    cache,
                    batch_kind,
                )?;
                let y = backend::apply_recurrence_from_convolved(
                    &convolved_qkv,
                    &projected.b,
                    &projected.a,
                    &self.a_log,
                    &self.dt_bias,
                    &self.dims,
                    batch_size,
                    seq_len,
                    cache,
                    dtype,
                )?;
                (convolved_qkv, y)
            }
        };
        if let Some(stash_out) = stash_out {
            *stash_out = Some(GdnForwardStash {
                mixed_qkv,
                convolved_qkv: convolved_qkv.clone(),
                b: projected.b.clone(),
                a: projected.a.clone(),
            });
        }

        self.finish_forward(y, projected.z, batch_size, seq_len, dtype)
    }

    #[cfg(feature = "cuda")]
    fn forward_speculative_checkpoints(
        &self,
        mixed_qkv: &Tensor,
        projected: &GdnProjection,
        cache: &GdnLayerCache,
        checkpoint_lanes: usize,
    ) -> Result<(Tensor, Tensor)> {
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
        let convolved_qkv = crate::cuda::gdn::speculative_conv_checkpoints_cuda(
            crate::cuda::gdn::GdnSpeculativeConvCheckpoints {
                x: mixed_qkv,
                weight: &weight,
                state_pool: &cache.conv_state,
                active_slots,
                checkpoint_lanes,
            },
        )?;
        let output = crate::cuda::gdn::speculative_recurrence_checkpoints_cuda(
            crate::cuda::gdn::GdnSpeculativeRecurrenceCheckpoints {
                mixed_qkv: &convolved_qkv,
                b: &projected.b,
                a: &projected.a,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                state_pool: &cache.recurrent_state,
                active_slots,
                checkpoint_lanes,
                num_k_heads: self.dims.num_k_heads,
                num_v_heads: self.dims.num_v_heads,
                head_k_dim: self.dims.head_k_dim,
                head_v_dim: self.dims.head_v_dim,
                tiled_v_heads: self.dims.v_head_layout == GdnVHeadLayout::Tiled,
                state_layout: cache.state_layout,
            },
        )?;
        let output = output
            .reshape((
                batch_size,
                self.dims.num_v_heads,
                seq_len,
                self.dims.head_v_dim,
            ))?
            .transpose(1, 2)?;
        Ok((convolved_qkv, output))
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

    fn project(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<GdnProjection> {
        self.input_proj.forward(x, &self.dims, batch_size, seq_len)
    }

    fn finish_forward(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        _dtype: DType,
    ) -> Result<Tensor> {
        let y = self.norm.forward(&y, &z)?;
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
    use candle_core::{Device, Tensor};
    use mistralrs_quant::Shard;

    use super::super::config::GdnVHeadLayout;
    use super::{
        index_select_rows, shard_out_proj_input, speculative_checkpoint_dims_supported,
        speculative_state_commit_dims_supported, GdnDims,
    };

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
