use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::GdnProjection;
use super::weights::{GdnWeightLoadCtx, GdnWeightMode, GdnWeights};

pub struct GatedDeltaNet {
    pub in_proj: Arc<dyn QuantMethod>,
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
            GdnWeightLoadCtx {
                cfg,
                dims: &dims,
                mapper,
                layer_idx,
                loading_isq,
                comm,
                weight_mode,
            },
        )?;
        Ok(Self {
            in_proj: weights.in_proj,
            conv1d_weight: weights.conv1d_weight,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            norm: weights.norm,
            out_proj: weights.out_proj,
            dims,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let projected = self.project(x, batch_size, seq_len)?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        let mixed_qkv = backend::causal_conv1d(
            &mixed_qkv,
            &self.conv1d_weight,
            &self.dims,
            cache,
            batch_kind,
        )?;
        let y = backend::apply_recurrence_from_convolved(
            &mixed_qkv,
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

        self.finish_forward(y, projected.z, batch_size, seq_len, dtype)
    }

    fn project(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<GdnProjection> {
        let mixed = self.in_proj.forward(x)?;
        GdnProjection::from_packed(mixed, &self.dims, batch_size, seq_len)
    }

    pub fn residual_input_projection_tensors(&self) -> (Tensor, Tensor) {
        let weight = self
            .in_proj
            .dequantize_w()
            .expect("failed to dequantize GDN input projection");
        let qkvz = weight
            .narrow(0, 0, self.dims.qkvz_out_dim())
            .expect("failed to split GDN qkvz projection");
        let ba = weight
            .narrow(0, self.dims.qkvz_out_dim(), self.dims.ba_out_dim())
            .expect("failed to split GDN ba projection");
        (qkvz, ba)
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
