use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, Shard, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::{GdnInputProjection, GdnProjection};
use super::weights::{GdnInputProjectionKind, GdnWeightLoadCtx, GdnWeights};

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
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.dims.head_v_dim))?;
        let z = z.reshape(((), self.dims.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
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

    use super::shard_out_proj_input;

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
