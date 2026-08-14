use std::sync::{Arc, OnceLock};

use candle_core::{quantized::GgmlDType, DType, Result, Tensor};
use mistralrs_quant::QuantMethod;

use crate::ops::SplitOp;

type SplitProjections<'a> = (&'a Arc<dyn QuantMethod>, &'a Arc<dyn QuantMethod>);

#[derive(Debug)]
pub enum MlaKvBProjection {
    Fused(Arc<dyn QuantMethod>),
    Split {
        key: Arc<dyn QuantMethod>,
        value: Arc<dyn QuantMethod>,
        expanded_weights: OnceLock<(Tensor, Tensor)>,
    },
}

impl MlaKvBProjection {
    pub fn fused(projection: Arc<dyn QuantMethod>) -> Self {
        Self::Fused(projection)
    }

    pub fn split(key: Arc<dyn QuantMethod>, value: Arc<dyn QuantMethod>) -> Self {
        Self::Split {
            key,
            value,
            expanded_weights: OnceLock::new(),
        }
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn fused_projection(&self) -> Option<&dyn QuantMethod> {
        match self {
            Self::Fused(projection) => Some(projection.as_ref()),
            Self::Split { .. } => None,
        }
    }

    pub fn fused_layer(&self) -> Option<&Arc<dyn QuantMethod>> {
        match self {
            Self::Fused(projection) => Some(projection),
            Self::Split { .. } => None,
        }
    }

    pub fn split_projections(&self) -> Option<SplitProjections<'_>> {
        match self {
            Self::Fused(_) => None,
            Self::Split { key, value, .. } => Some((key, value)),
        }
    }

    pub fn is_split(&self) -> bool {
        matches!(self, Self::Split { .. })
    }

    #[cfg(any(all(feature = "cuda", target_family = "unix"), test))]
    pub fn is_dynamic_lora_active(&self) -> bool {
        match self {
            Self::Fused(projection) => projection.is_dynamic_lora_active(),
            Self::Split { key, value, .. } => {
                key.is_dynamic_lora_active() || value.is_dynamic_lora_active()
            }
        }
    }

    pub fn project_query(&self, query: &Tensor) -> Result<Tensor> {
        let Self::Split {
            key,
            value,
            expanded_weights,
        } = self
        else {
            candle_core::bail!("split MLA key projection is not available");
        };
        ensure_split_lora_inactive(key, value)?;
        if supports_indexed_cuda_projection(key.as_ref(), query) {
            indexed_head_forward(key.as_ref(), query)
        } else {
            let (key_weight, _) = expanded_split_weights(key, value, expanded_weights)?;
            dense_head_forward(query, &key_weight, false)
        }
    }

    pub fn project_value(&self, value_states: &Tensor) -> Result<Tensor> {
        let Self::Split {
            key,
            value,
            expanded_weights,
        } = self
        else {
            candle_core::bail!("split MLA value projection is not available");
        };
        ensure_split_lora_inactive(key, value)?;
        if supports_indexed_cuda_projection(value.as_ref(), value_states) {
            indexed_head_forward(value.as_ref(), value_states)
        } else {
            let (_, value_weight) = expanded_split_weights(key, value, expanded_weights)?;
            dense_head_forward(value_states, &value_weight, true)
        }
    }

    pub fn expanded_kv(
        &self,
        compressed_kv: &Tensor,
        num_attention_heads: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, kv_lora_rank) = compressed_kv.dims3()?;
        match self {
            Self::Fused(projection) => {
                let kv = projection
                    .forward(compressed_kv)?
                    .reshape((
                        batch,
                        seq_len,
                        num_attention_heads,
                        qk_nope_head_dim + v_head_dim,
                    ))?
                    .transpose(1, 2)?;
                let kv = kv.split(&[qk_nope_head_dim, v_head_dim], candle_core::D::Minus1)?;
                Ok((kv[0].clone(), kv[1].clone()))
            }
            Self::Split {
                key,
                value,
                expanded_weights,
            } => {
                ensure_split_lora_inactive(key, value)?;
                let (key_weight, value_weight) =
                    expanded_split_weights(key, value, expanded_weights)?;
                let input_dtype = compressed_kv.dtype();
                let input = compressed_kv
                    .reshape((batch * seq_len, kv_lora_rank))?
                    .to_dtype(DType::F32)?;
                let key = input.matmul(&key_weight.t()?)?.reshape((
                    batch,
                    seq_len,
                    num_attention_heads,
                    qk_nope_head_dim,
                ))?;
                let value = input.matmul(&value_weight.t()?)?.reshape((
                    batch,
                    seq_len,
                    num_attention_heads,
                    v_head_dim,
                ))?;
                Ok((
                    key.to_dtype(input_dtype)?.transpose(1, 2)?,
                    value.to_dtype(input_dtype)?.transpose(1, 2)?,
                ))
            }
        }
    }
}

fn ensure_split_lora_inactive(
    key: &Arc<dyn QuantMethod>,
    value: &Arc<dyn QuantMethod>,
) -> Result<()> {
    if key.is_dynamic_lora_active() || value.is_dynamic_lora_active() {
        candle_core::bail!("split MLA K/V projections do not support active dynamic LoRA adapters");
    }
    Ok(())
}

fn indexed_head_forward(projection: &dyn QuantMethod, input: &Tensor) -> Result<Tensor> {
    let (batch, num_heads, seq_len, in_dim) = input.dims4()?;
    let input =
        input
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * seq_len, num_heads, in_dim))?;
    let num_heads_u32 = u32::try_from(num_heads)
        .map_err(|_| candle_core::Error::msg("MLA head count exceeds u32"))?;
    let head_ids = Tensor::arange(0u32, num_heads_u32, input.device())?
        .unsqueeze(0)?
        .repeat((batch * seq_len, 1))?;
    let output = projection.gather_forward(&input, &head_ids)?;
    let out_dim = output.dim(2)?;
    output
        .reshape((batch, seq_len, num_heads, out_dim))?
        .transpose(1, 2)
}

fn supports_indexed_cuda_projection(projection: &dyn QuantMethod, input: &Tensor) -> bool {
    if !input.device().is_cuda() {
        return false;
    }
    projection.get_qtensor().is_none_or(|weight| {
        matches!(
            weight.dtype(),
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q8_1
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        )
    })
}

fn dense_head_forward(input: &Tensor, weight: &Tensor, transpose_weight: bool) -> Result<Tensor> {
    let (batch, num_heads, seq_len, in_dim) = input.dims4()?;
    let (rows, cols) = weight.dims2()?;
    let (weight, out_dim) = if transpose_weight {
        if cols != in_dim || !rows.is_multiple_of(num_heads) {
            candle_core::bail!("split MLA value projection has incompatible dimensions");
        }
        let out_dim = rows / num_heads;
        (
            weight
                .reshape((num_heads, out_dim, in_dim))?
                .transpose(1, 2)?,
            out_dim,
        )
    } else {
        if rows != num_heads * in_dim {
            candle_core::bail!("split MLA key projection has incompatible dimensions");
        }
        (weight.reshape((num_heads, in_dim, cols))?, cols)
    };
    input
        .to_dtype(DType::F32)?
        .broadcast_matmul(&weight.unsqueeze(0)?)?
        .to_dtype(input.dtype())?
        .reshape((batch, num_heads, seq_len, out_dim))
}

fn expanded_split_weights(
    key: &Arc<dyn QuantMethod>,
    value: &Arc<dyn QuantMethod>,
    cache: &OnceLock<(Tensor, Tensor)>,
) -> Result<(Tensor, Tensor)> {
    if let Some(weights) = cache.get() {
        return Ok(weights.clone());
    }
    let key = key.dequantize_w()?;
    let value = value.dequantize_w()?;
    let (num_heads, kv_lora_rank, qk_nope_head_dim) = key.dims3()?;
    let (value_heads, v_head_dim, value_kv_lora_rank) = value.dims3()?;
    if value_heads != num_heads || value_kv_lora_rank != kv_lora_rank {
        candle_core::bail!(
            "split MLA K/V dimensions are incompatible: K {:?}, V {:?}",
            key.dims(),
            value.dims()
        );
    }
    let key = key
        .transpose(1, 2)?
        .contiguous()?
        .reshape((num_heads * qk_nope_head_dim, kv_lora_rank))?
        .to_dtype(DType::F32)?;
    let value = value
        .reshape((num_heads * v_head_dim, kv_lora_rank))?
        .to_dtype(DType::F32)?;
    let _ = cache.set((key, value));
    Ok(cache
        .get()
        .expect("expanded MLA weights initialized")
        .clone())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use mistralrs_quant::{
        maybe_wrap_dynamic_lora, with_lora_execution, LoraExecution, LoraLayerRegistry,
        LoraLinearSpec, LoraWeights, QuantMethod, QuantMethodConfig, ShardedSafeTensors,
        UnquantLinear,
    };

    use super::{dense_head_forward, supports_indexed_cuda_projection, MlaKvBProjection};

    fn layer(weight: Tensor) -> candle_core::Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?))
    }

    #[test]
    fn split_projection_matches_per_head_algebra() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let key = Tensor::new(&[[[1f32, 0.], [0., 1.]], [[1f32, 1.], [1., -1.]]], &device)?;
        let value = Tensor::new(&[[[1f32, 2.], [3., 4.]], [[2f32, 0.], [0., 2.]]], &device)?;
        let projection = MlaKvBProjection::split(layer(key)?, layer(value)?);

        let query = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;
        let query = projection.project_query(&query)?;
        assert_eq!(query.flatten_all()?.to_vec1::<f32>()?, vec![2., 3., 5., 3.]);

        let latent = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;
        let value = projection.project_value(&latent)?;
        assert_eq!(
            value.flatten_all()?.to_vec1::<f32>()?,
            vec![8., 18., 8., 2.]
        );
        Ok(())
    }

    #[test]
    fn split_projection_expands_compressed_kv() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let key = Tensor::new(&[[[1f32, 0.], [0., 1.]], [[1f32, 1.], [1., -1.]]], &device)?;
        let value = Tensor::new(&[[[1f32, 2.], [3., 4.]], [[2f32, 0.], [0., 2.]]], &device)?;
        let projection = MlaKvBProjection::split(layer(key)?, layer(value)?);
        let compressed = Tensor::new(&[[[2f32, 3.]]], &device)?;
        let (key, value) = projection.expanded_kv(&compressed, 2, 2, 2)?;

        assert_eq!(key.dims(), &[1, 2, 1, 2]);
        assert_eq!(value.dims(), &[1, 2, 1, 2]);
        assert_eq!(key.flatten_all()?.to_vec1::<f32>()?, vec![2., 3., 5., -1.]);
        assert_eq!(
            value.flatten_all()?.to_vec1::<f32>()?,
            vec![8., 18., 4., 6.]
        );
        Ok(())
    }

    #[test]
    fn dense_split_projection_fallback_matches_head_algebra() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let query = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;
        let key = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.], [1., -1.]], &device)?;
        let query = dense_head_forward(&query, &key, false)?;
        assert_eq!(query.flatten_all()?.to_vec1::<f32>()?, vec![2., 3., 5., 3.]);

        let latent = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;
        let value = Tensor::new(&[[1f32, 2.], [3., 4.], [2., 0.], [0., 2.]], &device)?;
        let value = dense_head_forward(&latent, &value, true)?;
        assert_eq!(
            value.flatten_all()?.to_vec1::<f32>()?,
            vec![8., 18., 8., 2.]
        );
        Ok(())
    }

    #[test]
    fn cpu_split_projection_populates_expanded_weight_cache() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let key = layer(Tensor::new(
            &[[[1f32, 0.], [0., 1.]], [[1f32, 1.], [1., -1.]]],
            &device,
        )?)?;
        let value = layer(Tensor::new(
            &[[[1f32, 2.], [3., 4.]], [[2f32, 0.], [0., 2.]]],
            &device,
        )?)?;
        let query = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;

        assert!(!supports_indexed_cuda_projection(key.as_ref(), &query));
        let projection = MlaKvBProjection::split(key, value);
        let MlaKvBProjection::Split {
            expanded_weights, ..
        } = &projection
        else {
            unreachable!()
        };
        assert!(expanded_weights.get().is_none());
        projection.project_query(&query)?;
        assert!(expanded_weights.get().is_some());
        projection.project_value(&query)?;
        assert!(expanded_weights.get().is_some());
        Ok(())
    }

    #[test]
    fn split_projection_rejects_active_dynamic_lora() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let registry = Arc::new(LoraLayerRegistry::new());
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, device.clone())
                .with_lora_registry(registry.clone())
                .pp("k_b_proj");
        let key = Tensor::new(&[[[1f32, 0.], [0., 1.]], [[1f32, 1.], [1., -1.]]], &device)?;
        let key = maybe_wrap_dynamic_lora(&vb, layer(key)?, LoraLinearSpec::replicated(2, 4))?;
        registry.finalize()?;
        let site = registry
            .sites()
            .into_iter()
            .next()
            .expect("registered split key projection");
        let value = Tensor::new(&[[[1f32, 2.], [3., 4.]], [[2f32, 0.], [0., 2.]]], &device)?;
        let projection = MlaKvBProjection::split(key, layer(value)?);
        let query = Tensor::new(&[[[[2f32, 3.]], [[4., 1.]]]], &device)?;
        let compressed = Tensor::new(&[[[2f32, 3.]]], &device)?;

        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.]], &device)?,
                Tensor::new(&[[1f32], [0.], [0.], [0.]], &device)?,
                1.0,
            )?,
        )?;
        let errors = with_lora_execution(Some(Arc::new(execution)), || {
            (
                projection
                    .project_query(&query)
                    .expect_err("active split LoRA must be rejected")
                    .to_string(),
                projection
                    .expanded_kv(&compressed, 2, 2, 2)
                    .expect_err("active split LoRA fallback must be rejected")
                    .to_string(),
            )
        });
        assert!(errors
            .0
            .contains("split MLA K/V projections do not support active dynamic LoRA adapters"));
        assert!(errors
            .1
            .contains("split MLA K/V projections do not support active dynamic LoRA adapters"));
        Ok(())
    }
}
