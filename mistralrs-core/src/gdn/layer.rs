use candle_core::{DType, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{Comm, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

use super::backend;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::GdnProjection;
use super::weights::{GdnWeightMode, GdnWeights};

pub struct GatedDeltaNet {
    pub in_proj_qkvz: Linear,
    pub in_proj_ba: Linear,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    dims: GdnDims,
}

impl GatedDeltaNet {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<Comm>,
        weight_mode: GdnWeightMode,
    ) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let weights = GdnWeights::load(
            vb,
            cfg,
            &dims,
            mapper,
            layer_idx,
            loading_isq,
            comm,
            weight_mode,
        )?;
        Ok(Self {
            in_proj_qkvz: weights.in_proj_qkvz,
            in_proj_ba: weights.in_proj_ba,
            conv1d_weight: weights.conv1d_weight,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            norm: weights.norm,
            out_proj: weights.out_proj,
            dims,
        })
    }

    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let projected = self.project(x, batch_size, seq_len)?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        let mixed_qkv = backend::causal_conv1d(&mixed_qkv, &self.conv1d_weight, &self.dims, cache)?;
        let recurrent_input =
            projected.with_convolved_qkv(mixed_qkv, &self.dims, batch_size, seq_len)?;
        let (beta, g) = backend::compute_beta_g(
            &recurrent_input.b,
            &recurrent_input.a,
            &self.a_log,
            &self.dt_bias,
            dtype,
        )?;
        let (q, k) = recurrent_input.normalized_qk(&self.dims, batch_size, seq_len)?;
        let y = backend::apply_recurrence(
            &q,
            &k,
            &recurrent_input.v,
            &g,
            &beta,
            &self.dims,
            batch_size,
            seq_len,
            cache,
            dtype,
        )?;

        cache.seqlen_offset += seq_len;
        self.finish_forward(y, recurrent_input.z, batch_size, seq_len, dtype)
    }

    fn project(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<GdnProjection> {
        let mixed_qkvz = self.in_proj_qkvz.forward(x)?;
        let mixed_ba = self.in_proj_ba.forward(x)?;
        GdnProjection::new(mixed_qkvz, mixed_ba, &self.dims, batch_size, seq_len)
    }

    fn finish_forward(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        _dtype: DType,
    ) -> Result<Tensor> {
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.dims.head_v_dim))?;
        let z = z.reshape(((), self.dims.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.dims.value_dim))?;
        self.out_proj.forward(&y)
    }
}
