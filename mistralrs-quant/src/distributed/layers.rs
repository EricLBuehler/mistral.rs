use std::sync::Arc;

#[cfg(all(feature = "cuda", has_marlin_kernels))]
use candle_core::DType;
use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;

use crate::{
    blockwise_fp8::{blockwise_fp8_linear_b, blockwise_fp8_moe},
    distributed,
    gptq::gptq_linear,
    lora::maybe_wrap_dynamic_lora_with_key,
    make_dummy_or_error, maybe_wrap_dynamic_lora,
    pertensor_fp8::pertensor_fp8_linear_b,
    should_apply_immediate_isq,
    utils::isq::apply_immediate_isq_sharded,
    AfqLayer, BnbLinear, DistributedKind, LoraLinearSpec, LoraSiteKey, MXFP4Layer, QuantMethod,
    QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde, Shard,
    ShardedVarBuilder, UnquantLinear,
};

use super::Comm;

fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}

fn load_weight_source_linear(vb: &ShardedVarBuilder) -> Result<Option<Arc<dyn QuantMethod>>> {
    load_weight_source_linear_shard(Shard::default(), vb)
}

fn load_weight_source_linear_shard(
    shard: Shard,
    vb: &ShardedVarBuilder,
) -> Result<Option<Arc<dyn QuantMethod>>> {
    let Some(source) = vb.weight_source() else {
        return Ok(None);
    };

    source.load_linear(&vb.prefix(), &crate::weight_source_load_device(vb), shard)
}

fn load_weight_source_dense(
    vb: &ShardedVarBuilder,
    bias: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let Some(layer) = load_weight_source_linear(vb)? else {
        return Ok(None);
    };
    let weight = layer.dequantize_w()?.to_dtype(vb.dtype())?.contiguous()?;
    let bias = if bias {
        let source = vb.weight_source().expect("weight source present");
        let name = crate::safetensors::full_tensor_name(vb, "bias");
        source
            .load_optional_tensor(&name, &crate::weight_source_load_device(vb))?
            .map(|bias| bias.to_dtype(vb.dtype()))
            .transpose()?
    } else {
        None
    };
    Ok(Some((weight, bias)))
}

fn matformer_narrow(
    tensor: Tensor,
    dim: usize,
    original: usize,
    selected: usize,
    context: &str,
) -> Result<Tensor> {
    if selected > original {
        candle_core::bail!(
            "{context} selected dimension {selected} exceeds original dimension {original}"
        );
    }
    match tensor.dim(dim)? {
        size if size == original => tensor.narrow(dim, 0, selected)?.contiguous(),
        size if size == selected => Ok(tensor),
        size => candle_core::bail!(
            "{context} source dimension {dim} has size {size}, expected {original} or {selected}"
        ),
    }
}

/// This layer has a weight that is parallelized along the input dimension,
/// returning the "full" output dimension.
#[derive(Debug)]
pub struct RowParallelLayer {
    weight: Arc<dyn QuantMethod>,
    bias: Option<Tensor>,
    all_reduce: distributed::SumAllReduce,
}

impl RowParallelLayer {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(1, rank, world_size);
        Self::new_with_lora_spec(
            LoraLinearSpec::row(in_dim, out_dim, shard),
            config,
            bias,
            comm,
            vb,
        )
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_with_lora_spec(
        lora_spec: LoraLinearSpec,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(1, rank, world_size);
        if lora_spec.row_input_shard() != Some(shard) {
            candle_core::bail!(
                "row-parallel layer LoRA spec must use input shard {shard:?}, got {:?}",
                lora_spec.parallelism()
            );
        }
        let in_dim = lora_spec.in_features();
        let out_dim = lora_spec.out_features();

        let base_vb = vb.clone();
        if let Some(weight) = load_weight_source_linear_shard(shard, &base_vb)? {
            let weight = maybe_wrap_dynamic_lora(&base_vb, weight, lora_spec)?;
            let layer = if world_size == 1 {
                // Bias is embedded in the layer when the input dim is not actually sharded.
                weight
            } else {
                // Row-sharded deserializes skip the bias; it must be applied once, post-reduce.
                let bias = if bias {
                    let load_device = crate::weight_source_load_device(&base_vb);
                    base_vb
                        .weight_source()
                        .expect("weight source present")
                        .load_optional_tensor(&format!("{}.bias", base_vb.prefix()), &load_device)?
                } else {
                    None
                };
                Arc::new(Self {
                    weight,
                    bias,
                    all_reduce: distributed::SumAllReduce::new(comm),
                })
            };
            return apply_immediate_isq_sharded(layer, base_vb, Some(shard));
        }

        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::GptqAwq { .. }
                    | QuantizedConfig::Bitsandbytes { .. }
                    | QuantizedConfig::Afq { .. }
            ) && comm.world_size() != 1
            {
                candle_core::bail!(
                    "GPTQ and BNB and AFQ quantization types to not support tensor parallelism, but got a world size of {}",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            if !vb.contains_tensor("weight") {
                make_dummy_or_error("row_parallel_linear", &vb, &["weight"])?
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };
        let weight = maybe_wrap_dynamic_lora(&base_vb, weight, lora_spec)?;

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self {
            weight,
            bias,
            all_reduce: distributed::SumAllReduce::new(comm),
        });
        let this: Arc<dyn QuantMethod> =
            apply_immediate_isq_sharded(this_unquant, base_vb, Some(shard))?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_matformer(
        in_dim: usize,
        out_dim: usize,
        orig_intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(1, rank, world_size);

        let base_vb = vb.clone();
        if let Some((weight, bias)) = load_weight_source_dense(&base_vb, bias)? {
            if weight.rank() != 2 || weight.dim(0)? != out_dim {
                candle_core::bail!(
                    "row-parallel MatFormer source at `{}` has shape {:?}, expected [{out_dim}, {orig_intermediate_size}] or [{out_dim}, {in_dim}]",
                    base_vb.prefix(),
                    weight.dims()
                );
            }
            let weight = matformer_narrow(
                weight,
                1,
                orig_intermediate_size,
                in_dim,
                "row-parallel MatFormer",
            )?;
            let weight = shard.apply_to(&weight)?.contiguous()?;
            let weight = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?) as Arc<dyn QuantMethod>;
            let weight = maybe_wrap_dynamic_lora(
                &base_vb,
                weight,
                LoraLinearSpec::row(in_dim, out_dim, shard),
            )?;
            let layer = Arc::new(Self {
                weight,
                bias,
                all_reduce: distributed::SumAllReduce::new(comm),
            });
            return apply_immediate_isq_sharded(layer, base_vb, None);
        }
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        if config.is_some() {
            candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
        }

        let weight = if !vb.contains_tensor("weight") {
            make_dummy_or_error("row_parallel_matformer_linear", &vb, &["weight"])?
        } else {
            let weight = vb
                .get_with_hints(
                    (out_dim, orig_intermediate_size),
                    "weight",
                    Default::default(),
                )?
                .i((.., ..in_dim))?
                .contiguous()?;

            let weight = shard.apply_to(&weight)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };
        let weight = maybe_wrap_dynamic_lora(
            &base_vb,
            weight,
            LoraLinearSpec::row(in_dim, out_dim, shard),
        )?;

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self {
            weight,
            bias,
            all_reduce: distributed::SumAllReduce::new(comm),
        });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq_sharded(this_unquant, base_vb, None)?;
        Ok(this)
    }
}

impl QuantMethod for RowParallelLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("RowParallelLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        let mut xs = self.weight.forward_raw(a)?;
        if !self.all_reduce.is_noop() {
            let xs_contiguous = xs.contiguous()?;
            xs = self.all_reduce.sum_all_reduce(&xs_contiguous)?;
        }
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.weight.add_delta_w(delta)?;
        Ok(Arc::new(Self {
            weight,
            bias: self.bias.clone(),
            all_reduce: self.all_reduce.clone(),
        }))
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.weight.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.weight.dtype_and_device()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        self.weight.plan_isq(request)
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.weight.begin_track_stats()
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.weight.stats_snapshot()
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.weight.process_routed_stats(x, ids)
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.weight.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.weight.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        if self.all_reduce.is_noop() {
            self.weight.unquant_weight_bias()
        } else {
            None
        }
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.weight.is_dynamic_lora_active()
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        self.weight.preserve_dynamic_lora(replacement)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some() || self.weight.has_bias()
    }

    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        if self.all_reduce.is_noop() {
            self.weight.get_qtensor()
        } else {
            None
        }
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn prepare_gguf_affine_raw(
        &self,
        flat_batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<bool> {
        if self.all_reduce.is_noop() {
            self.weight
                .prepare_gguf_affine_raw(flat_batch, dtype, device)
        } else {
            Ok(false)
        }
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn try_gguf_affine_forward_raw(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if self.all_reduce.is_noop() {
            self.weight.try_gguf_affine_forward_raw(a)
        } else {
            Ok(None)
        }
    }

    fn afq_inner(&self) -> Option<crate::AfqInner> {
        if self.all_reduce.is_noop() {
            self.weight.afq_inner()
        } else {
            None
        }
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight =
            self.weight
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        let bias = match &self.bias {
            Some(b) => {
                let (dtype, device) = weight.dtype_and_device();
                Some(b.to_device(&device)?.to_dtype(dtype)?)
            }
            None => None,
        };
        Ok(Arc::new(Self {
            weight,
            bias,
            all_reduce: self.all_reduce.clone(),
        }))
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::RowParallel)
    }
}

impl QuantizedSerde for RowParallelLayer {
    fn isq_serde_supported(&self) -> bool {
        self.weight.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.weight.name()
    }
    fn uqff_type(&self) -> Option<crate::IsqType> {
        self.weight.uqff_type()
    }
    fn serialize_uqff(&self, prefix: &str, ty: crate::IsqType) -> Result<Vec<crate::UqffTensor>> {
        let mut tensors = self.weight.serialize_uqff(prefix, ty)?;
        if let Some(bias) = &self.bias {
            let bias_key = format!("{prefix}.bias");
            tensors.retain(|tensor| tensor.name() != bias_key);
            tensors.push(crate::UqffTensor::from_tensor(bias_key, bias)?);
        }
        Ok(tensors)
    }
}

#[derive(Debug)]
/// This layer has a weight that is parallelized along the output dimension,
/// taking the "full" input dimension.
pub struct ColumnParallelLayer {
    weight: Arc<dyn QuantMethod>,
    bias: Option<Tensor>,
}

impl ColumnParallelLayer {
    #[allow(clippy::new_ret_no_self)]
    pub fn new_with_shard(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        shard: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let site_key = LoraSiteKey::new(vb.prefix());
        Self::new_with_shard_and_key(in_dim, out_dim, config, bias, comm, shard, vb, site_key)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_shard_and_key(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        shard: Shard,
        vb: ShardedVarBuilder,
        site_key: LoraSiteKey,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base_vb = vb.clone();
        if let Some(layer) = load_weight_source_linear_shard(shard, &base_vb)? {
            let layer = maybe_wrap_dynamic_lora_with_key(
                &base_vb,
                layer,
                site_key,
                LoraLinearSpec::column(in_dim, out_dim, shard),
            )?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(shard));
        }

        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::GptqAwq { .. }
                    | QuantizedConfig::Bitsandbytes { .. }
                    | QuantizedConfig::Afq { .. }
            ) && comm.world_size() != 1
            {
                candle_core::bail!(
                    "GPTQ/AWQ and BNB and AFQ quantization types to not support tensor parallelism, but got a world size of {}",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            if !vb.contains_tensor("weight") {
                make_dummy_or_error("column_parallel_linear", &vb, &["weight"])?
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };
        let weight = maybe_wrap_dynamic_lora_with_key(
            &base_vb,
            weight,
            site_key,
            LoraLinearSpec::column(in_dim, out_dim, shard),
        )?;

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> =
            apply_immediate_isq_sharded(this_unquant, base_vb, Some(shard))?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(0, rank, world_size);

        Self::new_with_shard(in_dim, out_dim, config, bias, comm, shard, vb)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_matformer(
        in_dim: usize,
        out_dim: usize,
        orig_intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(0, rank, world_size);

        let base_vb = vb.clone();
        if let Some((weight, bias)) = load_weight_source_dense(&base_vb, bias)? {
            if weight.rank() != 2 || weight.dim(1)? != in_dim {
                candle_core::bail!(
                    "column-parallel MatFormer source at `{}` has shape {:?}, expected [{orig_intermediate_size}, {in_dim}] or [{out_dim}, {in_dim}]",
                    base_vb.prefix(),
                    weight.dims()
                );
            }
            let weight = matformer_narrow(
                weight,
                0,
                orig_intermediate_size,
                out_dim,
                "column-parallel MatFormer",
            )?;
            let weight = shard.apply_to(&weight)?.contiguous()?;
            let bias = bias
                .map(|bias| {
                    matformer_narrow(
                        bias,
                        0,
                        orig_intermediate_size,
                        out_dim,
                        "column-parallel MatFormer bias",
                    )
                })
                .transpose()?
                .map(|bias| shard.apply_to(&bias).and_then(|bias| bias.contiguous()))
                .transpose()?;
            let weight = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?) as Arc<dyn QuantMethod>;
            let weight = maybe_wrap_dynamic_lora(
                &base_vb,
                weight,
                LoraLinearSpec::column(in_dim, out_dim, shard),
            )?;
            let layer = Arc::new(Self { weight, bias });
            return apply_immediate_isq_sharded(layer, base_vb, None);
        }
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        if config.is_some() {
            candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
        }

        let weight = if !vb.contains_tensor("weight") {
            make_dummy_or_error("column_parallel_matformer_linear", &vb, &["weight"])?
        } else {
            let weight = vb
                .get_with_hints(
                    (orig_intermediate_size, in_dim),
                    "weight",
                    Default::default(),
                )?
                .i((..out_dim, ..))?
                .contiguous()?;

            let weight = shard.apply_to(&weight)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };
        let weight = maybe_wrap_dynamic_lora(
            &base_vb,
            weight,
            LoraLinearSpec::column(in_dim, out_dim, shard),
        )?;

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq_sharded(this_unquant, base_vb, None)?;
        Ok(this)
    }

    pub fn new_merged(
        in_dim: usize,
        out_dim: usize,
        chunks: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Vec<Arc<dyn QuantMethod>>> {
        let mut vec_layers = Vec::<Arc<dyn QuantMethod>>::new();
        for chunk_idx in 0..chunks {
            let site_key = LoraSiteKey::with_slice(vb.prefix(), chunk_idx, chunks)?;
            let layer = ColumnParallelLayer::new_with_shard_and_key(
                in_dim,
                out_dim,
                config,
                bias,
                comm,
                shard(
                    0,
                    chunk_idx * comm.world_size() + comm.rank(),
                    chunks * comm.world_size(),
                ),
                vb.clone(),
                site_key,
            )?;
            vec_layers.push(layer);
        }
        Ok(vec_layers)
    }
}

impl QuantMethod for ColumnParallelLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("ColumnParallelLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        let mut xs = self.weight.forward_raw(a)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let mut xs = self.weight.gather_forward_raw(a, indices)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.weight.add_delta_w(delta)?;
        Ok(Arc::new(Self {
            weight,
            bias: self.bias.clone(),
        }))
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.weight.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.weight.dtype_and_device()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        self.weight.plan_isq(request)
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.weight.begin_track_stats()
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.weight.stats_snapshot()
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.weight.process_routed_stats(x, ids)
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.weight.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.weight.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.weight.unquant_weight_bias()
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.weight.is_dynamic_lora_active()
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        self.weight.preserve_dynamic_lora(replacement)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some() || self.weight.has_bias()
    }

    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        self.weight.get_qtensor()
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn prepare_gguf_affine_raw(
        &self,
        flat_batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<bool> {
        self.weight
            .prepare_gguf_affine_raw(flat_batch, dtype, device)
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn try_gguf_affine_forward_raw(&self, a: &Tensor) -> Result<Option<Tensor>> {
        self.weight.try_gguf_affine_forward_raw(a)
    }

    fn afq_inner(&self) -> Option<crate::AfqInner> {
        self.weight.afq_inner()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight =
            self.weight
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        let bias = match &self.bias {
            Some(b) => {
                let (dtype, device) = weight.dtype_and_device();
                Some(b.to_device(&device)?.to_dtype(dtype)?)
            }
            None => None,
        };
        Ok(Arc::new(Self { weight, bias }))
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::ColumnParallel)
    }
}

impl QuantizedSerde for ColumnParallelLayer {
    fn isq_serde_supported(&self) -> bool {
        self.weight.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.weight.name()
    }
    fn uqff_type(&self) -> Option<crate::IsqType> {
        self.weight.uqff_type()
    }
    fn serialize_uqff(&self, prefix: &str, ty: crate::IsqType) -> Result<Vec<crate::UqffTensor>> {
        let mut tensors = self.weight.serialize_uqff(prefix, ty)?;
        if let Some(bias) = &self.bias {
            let bias_key = format!("{prefix}.bias");
            tensors.retain(|tensor| tensor.name() != bias_key);
            tensors.push(crate::UqffTensor::from_tensor(bias_key, bias)?);
        }
        Ok(tensors)
    }
}

#[derive(Debug)]
/// This layer has no parallelization
pub struct ReplicatedLayer(Arc<dyn QuantMethod>);

impl ReplicatedLayer {
    pub fn from_linear(lin: Linear, vb: ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
        let (out_dim, in_dim) = lin.weight().dims2()?;
        let spec = LoraLinearSpec::replicated(in_dim, out_dim);
        if let Some(layer) = load_weight_source_linear(&vb)? {
            let layer = maybe_wrap_dynamic_lora(&vb, layer, spec)?;
            return apply_immediate_isq_sharded(layer, vb, Some(Shard::default()));
        }

        let dev = lin.weight().device().clone();
        let vb = vb.set_device(dev.clone());
        let lin = if should_apply_immediate_isq(&vb) && !dev.is_cpu() {
            Linear::new(
                lin.weight().to_device(&Device::Cpu)?,
                lin.bias()
                    .map(|bias| bias.to_device(&Device::Cpu))
                    .transpose()?,
            )
        } else {
            lin
        };
        let layer: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lin))?);
        let layer = maybe_wrap_dynamic_lora(&vb, layer, spec)?;
        apply_immediate_isq_sharded(layer, vb, None)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::new_with_lora_spec(
            LoraLinearSpec::replicated(in_dim, out_dim),
            config,
            bias,
            vb,
        )
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_with_lora_spec(
        lora_spec: LoraLinearSpec,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !lora_spec.is_replicated() {
            candle_core::bail!(
                "replicated layer LoRA spec must use replicated parallelism, got {:?}",
                lora_spec.parallelism()
            );
        }
        let in_dim = lora_spec.in_features();
        let out_dim = lora_spec.out_features();
        let base_vb = vb.clone();
        if let Some(layer) = load_weight_source_linear(&base_vb)? {
            let layer = maybe_wrap_dynamic_lora(&base_vb, layer, lora_spec)?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()));
        }

        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let layer = if let Some(quant_conf) = &config {
            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            if !vb.contains_tensor("weight") {
                make_dummy_or_error("replicated_linear", &vb, &["weight"])?
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;

                let bias = if bias {
                    Some(vb.get_with_hints((out_dim,), "bias", Default::default())?)
                } else {
                    None
                };
                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, bias),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };
        let layer = maybe_wrap_dynamic_lora(&base_vb, layer, lora_spec)?;

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> =
            apply_immediate_isq_sharded(this_unquant, base_vb, Some(crate::Shard::default()))?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_layers_matformer_indices(
        in_dim: usize,
        out_dim: usize,
        kept_layers_indices: Option<&Tensor>,
        orig_num_hidden_layers: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base_vb = vb.clone();
        if let Some((mut weight, mut bias_tensor)) = load_weight_source_dense(&base_vb, bias)? {
            if weight.rank() != 2 || weight.dim(1)? != in_dim {
                candle_core::bail!(
                    "replicated MatFormer source at `{}` has shape {:?}, expected [{out_dim}, {in_dim}]",
                    base_vb.prefix(),
                    weight.dims()
                );
            }
            if let Some(kept_layers_indices) = kept_layers_indices {
                if !out_dim.is_multiple_of(orig_num_hidden_layers) {
                    candle_core::bail!(
                        "replicated MatFormer output dimension {out_dim} is not divisible by {orig_num_hidden_layers} layers"
                    );
                }
                let per_layer = out_dim / orig_num_hidden_layers;
                let selected_out = kept_layers_indices.elem_count() * per_layer;
                weight = match weight.dim(0)? {
                    size if size == out_dim => weight
                        .reshape((orig_num_hidden_layers, per_layer, in_dim))?
                        .index_select(&kept_layers_indices.to_device(weight.device())?, 0)?
                        .reshape((selected_out, in_dim))?
                        .contiguous()?,
                    size if size == selected_out => weight,
                    size => candle_core::bail!(
                        "replicated MatFormer source output has size {size}, expected {out_dim} or {selected_out}"
                    ),
                };
                bias_tensor = bias_tensor
                    .map(|bias| -> Result<Tensor> {
                        match bias.dim(0)? {
                            size if size == out_dim => bias
                                .reshape((orig_num_hidden_layers, per_layer))?
                                .index_select(&kept_layers_indices.to_device(bias.device())?, 0)?
                                .reshape(selected_out)?
                                .contiguous(),
                            size if size == selected_out => Ok(bias),
                            size => candle_core::bail!(
                                "replicated MatFormer bias has size {size}, expected {out_dim} or {selected_out}"
                            ),
                        }
                    })
                    .transpose()?;
            } else if weight.dim(0)? != out_dim {
                candle_core::bail!(
                    "replicated MatFormer source output has size {}, expected {out_dim}",
                    weight.dim(0)?
                );
            }
            let runtime_out = weight.dim(0)?;
            let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, bias_tensor),
            ))?) as Arc<dyn QuantMethod>;
            let layer = maybe_wrap_dynamic_lora(
                &base_vb,
                layer,
                LoraLinearSpec::replicated(in_dim, runtime_out),
            )?;
            let layer = Arc::new(Self(layer));
            return apply_immediate_isq_sharded(layer, base_vb, None);
        }
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let layer = if let Some(quant_conf) = &config {
            if kept_layers_indices.is_some() {
                candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
            }

            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            if !vb.contains_tensor("weight") {
                make_dummy_or_error("replicated_matformer_linear", &vb, &["weight"])?
            } else {
                let mut weight =
                    vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;

                if let Some(kept_layers_indices) = &kept_layers_indices {
                    let weight_reshaped = weight.reshape((
                        orig_num_hidden_layers,
                        weight.dim(0)? / orig_num_hidden_layers,
                        weight.dim(1)?,
                    ))?;

                    weight = weight_reshaped
                        .index_select(&kept_layers_indices.to_device(weight.device())?, 0)?
                        .reshape(((), weight_reshaped.dim(D::Minus1)?))?
                        .contiguous()?;
                }

                let bias = if bias {
                    Some(vb.get_with_hints((out_dim,), "bias", Default::default())?)
                } else {
                    None
                };
                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, bias),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };
        let layer =
            maybe_wrap_dynamic_lora(&base_vb, layer, LoraLinearSpec::replicated(in_dim, out_dim))?;

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> = apply_immediate_isq_sharded(this_unquant, base_vb, None)?;
        Ok(this)
    }
}

impl QuantMethod for ReplicatedLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("ReplicatedLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward_raw(a)
    }

    fn embedding_forward(&self, ids: &Tensor, output_dtype: candle_core::DType) -> Result<Tensor> {
        self.0.embedding_forward(ids, output_dtype)
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        self.0.embedding_forward_raw(ids)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        self.0.add_delta_w(delta)
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.0.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.0.dtype_and_device()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        self.0.plan_isq(request)
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.0.begin_track_stats()
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.0.stats_snapshot()
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.0.process_routed_stats(x, ids)
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.0.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.0.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.0.unquant_weight_bias()
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.0.is_dynamic_lora_active()
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        self.0.preserve_dynamic_lora(replacement)
    }

    fn has_bias(&self) -> bool {
        self.0.has_bias()
    }

    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        self.0.get_qtensor()
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn prepare_gguf_affine_raw(
        &self,
        flat_batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<bool> {
        self.0.prepare_gguf_affine_raw(flat_batch, dtype, device)
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn try_gguf_affine_forward_raw(&self, a: &Tensor) -> Result<Option<Tensor>> {
        self.0.try_gguf_affine_forward_raw(a)
    }

    fn afq_inner(&self) -> Option<crate::AfqInner> {
        self.0.afq_inner()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        self.0
            .clone()
            .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::Replicated)
    }
}

impl QuantizedSerde for ReplicatedLayer {
    fn isq_serde_supported(&self) -> bool {
        self.0.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn uqff_type(&self) -> Option<crate::IsqType> {
        self.0.uqff_type()
    }
    fn serialize_uqff(&self, prefix: &str, ty: crate::IsqType) -> Result<Vec<crate::UqffTensor>> {
        self.0.serialize_uqff(prefix, ty)
    }
}

#[derive(Debug)]
pub struct PreQuantizedExperts {
    pub fused_gate_proj: Arc<dyn QuantMethod>,
    pub fused_up_proj: Arc<dyn QuantMethod>,
    pub fused_down_proj: Arc<dyn QuantMethod>,
}

impl PreQuantizedExperts {
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        quantization_config: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Detect if weights are in stacked format (e.g., Qwen3 VL MoE):
        // - experts.gate_up_proj: (num_experts, hidden_size, intermediate_size * 2)
        // - experts.down_proj: (num_experts, intermediate_size, hidden_size)
        // Or per-expert format (e.g., Qwen3 MoE):
        // - experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, experts.{i}.down_proj.weight
        let experts_vb = vb.pp("experts");
        let is_stacked_format = experts_vb.contains_tensor("gate_up_proj");

        let (fused_gate_proj, fused_up_proj, fused_down_proj) = if matches!(
            &quantization_config,
            Some(QuantizedConfig::Afq { .. })
        ) {
            let quantization_config = quantization_config.as_ref().unwrap();

            let fused_gate_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.gate_proj"),
            )?;
            let fused_up_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.up_proj"),
            )?;
            let fused_down_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                moe_intermediate_size,
                hidden_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.down_proj"),
            )?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if is_stacked_format
            && matches!(&quantization_config, Some(QuantizedConfig::Fp8 { .. }))
        {
            // Stacked format with FP8 quantization
            // Keep weights as FP8 using BlockwiseFP8 to leverage native FP8 GEMM in gather_forward
            let has_fp8_scales = experts_vb.contains_tensor("gate_up_proj.weight_scale_inv");

            if has_fp8_scales {
                let weight_block_size = match quantization_config {
                    Some(QuantizedConfig::Fp8 { weight_block_size }) => weight_block_size.clone(),
                    _ => unreachable!(),
                };

                let Some(weight_block_size) = weight_block_size else {
                    candle_core::bail!(
                        "Blockwise FP8 for stacked experts requires weight_block_size to be set."
                    )
                };
                if weight_block_size.len() != 2 {
                    candle_core::bail!(
                        "Expected weight_block_size to have length 2, got {weight_block_size:?}"
                    );
                }

                // Load gate_up_proj FP8 tensor and scale
                // Shape: [num_experts, hidden_size, intermediate_size * 2]
                let gate_up_fp8 = experts_vb.get_with_hints_dtype(
                    (num_experts, hidden_size, moe_intermediate_size * 2),
                    "gate_up_proj",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let gate_up_scale = experts_vb.get_with_hints_dtype(
                    (
                        num_experts,
                        hidden_size.div_ceil(weight_block_size[0]),
                        (moe_intermediate_size * 2).div_ceil(weight_block_size[1]),
                    ),
                    "gate_up_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                // Load down_proj FP8 tensor and scale
                // Shape: [num_experts, intermediate_size, hidden_size]
                let down_fp8 = experts_vb.get_with_hints_dtype(
                    (num_experts, moe_intermediate_size, hidden_size),
                    "down_proj",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let down_scale = experts_vb.get_with_hints_dtype(
                    (
                        num_experts,
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "down_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                // Split gate_up into gate and up
                let gate_fp8 = gate_up_fp8.narrow(2, 0, moe_intermediate_size)?;
                let up_fp8 = gate_up_fp8.narrow(2, moe_intermediate_size, moe_intermediate_size)?;

                // Split scales similarly
                let gate_scale = gate_up_scale.narrow(
                    2,
                    0,
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                )?;
                let up_scale = gate_up_scale.narrow(
                    2,
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                )?;

                // Transpose to match expected format: [num_experts, N, K]
                // gate/up: [num_experts, hidden_size, intermediate_size] -> [num_experts, intermediate_size, hidden_size]
                let gate_fp8 = gate_fp8.transpose(1, 2)?.contiguous()?;
                let up_fp8 = up_fp8.transpose(1, 2)?.contiguous()?;
                // down: [num_experts, intermediate_size, hidden_size] -> [num_experts, hidden_size, intermediate_size]
                let down_fp8 = down_fp8.transpose(1, 2)?.contiguous()?;

                // Transpose scales to match weight layout
                let gate_scale = gate_scale.transpose(1, 2)?.contiguous()?;
                let up_scale = up_scale.transpose(1, 2)?.contiguous()?;
                let down_scale = down_scale.transpose(1, 2)?.contiguous()?;

                // Create BlockwiseFP8Linear for each projection
                let fused_gate_proj =
                    blockwise_fp8_moe(gate_fp8, gate_scale, weight_block_size.clone(), vb.dtype())?;
                let fused_up_proj =
                    blockwise_fp8_moe(up_fp8, up_scale, weight_block_size.clone(), vb.dtype())?;
                let fused_down_proj =
                    blockwise_fp8_moe(down_fp8, down_scale, weight_block_size, vb.dtype())?;

                (fused_gate_proj, fused_up_proj, fused_down_proj)
            } else {
                candle_core::bail!(
                    "FP8 quantization config without scale tensors; load via the unquantized expert path."
                );
            }
        } else if is_stacked_format
            && matches!(&quantization_config, Some(QuantizedConfig::MXFP4 {}))
        {
            // Stacked format with MXFP4 quantization
            // For MXFP4, weights are stored as packed FP4 (2 values per byte)
            // with E8M0 scales
            let quantization_config = quantization_config.as_ref().unwrap();

            // Load MXFP4 packed experts using MXFP4Layer::packed_linear_b
            // The tensors are expected at:
            //   gate_proj.blocks: [num_experts, intermediate_size, hidden_size/2]
            //   gate_proj.scales: [num_experts, intermediate_size, hidden_size/32]
            let fused_gate_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                experts_vb.pp("gate_proj"),
            )?;
            let fused_up_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                experts_vb.pp("up_proj"),
            )?;
            let fused_down_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                moe_intermediate_size,
                hidden_size,
                quantization_config,
                false,
                experts_vb.pp("down_proj"),
            )?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if matches!(&quantization_config, Some(QuantizedConfig::Fp8 { .. })) {
            // Per-expert format with FP8 quantization
            // Keep weights as FP8 using BlockwiseFP8 to leverage native FP8 GEMM in gather_forward
            let weight_block_size = match quantization_config {
                Some(QuantizedConfig::Fp8 { weight_block_size }) => weight_block_size.clone(),
                _ => unreachable!(),
            };

            let Some(weight_block_size) = weight_block_size else {
                candle_core::bail!(
                    "Blockwise FP8 for per-expert format requires weight_block_size to be set."
                )
            };
            if weight_block_size.len() != 2 {
                candle_core::bail!(
                    "Expected weight_block_size to have length 2, got {weight_block_size:?}"
                );
            }

            let mut gate_fp8_vec = Vec::new();
            let mut gate_scale_vec = Vec::new();
            let mut up_fp8_vec = Vec::new();
            let mut up_scale_vec = Vec::new();
            let mut down_fp8_vec = Vec::new();
            let mut down_scale_vec = Vec::new();

            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(i);

                // Load FP8 weights and scales for each projection
                let gate_fp8 = expert_vb.get_with_hints_dtype(
                    (moe_intermediate_size, hidden_size),
                    "gate_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let gate_scale = expert_vb.get_with_hints_dtype(
                    (
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "gate_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                let up_fp8 = expert_vb.get_with_hints_dtype(
                    (moe_intermediate_size, hidden_size),
                    "up_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let up_scale = expert_vb.get_with_hints_dtype(
                    (
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "up_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                let down_fp8 = expert_vb.get_with_hints_dtype(
                    (hidden_size, moe_intermediate_size),
                    "down_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let down_scale = expert_vb.get_with_hints_dtype(
                    (
                        hidden_size.div_ceil(weight_block_size[0]),
                        moe_intermediate_size.div_ceil(weight_block_size[1]),
                    ),
                    "down_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                gate_fp8_vec.push(gate_fp8);
                gate_scale_vec.push(gate_scale);
                up_fp8_vec.push(up_fp8);
                up_scale_vec.push(up_scale);
                down_fp8_vec.push(down_fp8);
                down_scale_vec.push(down_scale);
            }

            // Stack into [num_experts, N, K]
            let gate_fp8 = Tensor::stack(&gate_fp8_vec, 0)?;
            let gate_scale = Tensor::stack(&gate_scale_vec, 0)?;
            let up_fp8 = Tensor::stack(&up_fp8_vec, 0)?;
            let up_scale = Tensor::stack(&up_scale_vec, 0)?;
            let down_fp8 = Tensor::stack(&down_fp8_vec, 0)?;
            let down_scale = Tensor::stack(&down_scale_vec, 0)?;

            // Create BlockwiseFP8Linear for each projection
            let fused_gate_proj =
                blockwise_fp8_moe(gate_fp8, gate_scale, weight_block_size.clone(), vb.dtype())?;
            let fused_up_proj =
                blockwise_fp8_moe(up_fp8, up_scale, weight_block_size.clone(), vb.dtype())?;
            let fused_down_proj =
                blockwise_fp8_moe(down_fp8, down_scale, weight_block_size, vb.dtype())?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else {
            candle_core::bail!(
                "PreQuantizedExperts loads pre-quantized expert formats only (AFQ, blockwise FP8, MXFP4)."
            );
        };

        Ok(Self {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }
}

fn validate_tp_kv_heads(total_num_kv_heads: usize, tensor_parallel_size: usize) -> Result<()> {
    if total_num_kv_heads == 0 {
        candle_core::bail!("Total number of KV heads must be greater than 0.");
    }
    if tensor_parallel_size <= total_num_kv_heads {
        if !total_num_kv_heads.is_multiple_of(tensor_parallel_size) {
            candle_core::bail!(
                "Total number of KV heads ({total_num_kv_heads}) must be divisible by tensor parallel size ({tensor_parallel_size}) when KV heads are partitioned."
            );
        }
    } else if !tensor_parallel_size.is_multiple_of(total_num_kv_heads) {
        candle_core::bail!(
            "Tensor parallel size ({tensor_parallel_size}) must be divisible by total number of KV heads ({total_num_kv_heads}) when KV heads are replicated."
        );
    }
    Ok(())
}

pub fn validate_tp_head_layout(
    total_num_attention_heads: usize,
    total_num_kv_heads: usize,
    tensor_parallel_size: usize,
) -> Result<()> {
    if total_num_attention_heads == 0 {
        candle_core::bail!("Total number of attention heads must be greater than 0.");
    }
    if !total_num_attention_heads.is_multiple_of(tensor_parallel_size) {
        candle_core::bail!(
            "Total number of attention heads ({total_num_attention_heads}) must be divisible by tensor parallel size ({tensor_parallel_size})."
        );
    }
    validate_tp_kv_heads(total_num_kv_heads, tensor_parallel_size)
}

/// Compute the appropriate KV shard. This handles KV head replication. Be sure to use `compute_n_kv_groups` in tandem.
pub fn compute_kv_shard(total_num_kv_heads: usize, head_dim: usize, comm: &Comm) -> Result<Shard> {
    if comm.world_size() == 1 {
        return Ok(Shard::default());
    }

    validate_tp_kv_heads(total_num_kv_heads, comm.world_size())?;
    let kv_replicate = if comm.world_size() > total_num_kv_heads {
        comm.world_size() / total_num_kv_heads
    } else {
        return Ok(Shard::Simple {
            dim: 0,
            rank: comm.rank(),
            world_size: comm.world_size(),
        });
    };

    let num_kv_heads = (total_num_kv_heads / comm.world_size()).max(1);
    let kv_shard_id = (comm.rank() / kv_replicate) * num_kv_heads;
    Ok(Shard::Offset {
        dim: 0,
        offset: kv_shard_id * head_dim,
        len: head_dim,
    })
}

/// Compute the number of KV groups, taking into account KV head replication.
pub fn compute_n_kv_groups(
    total_num_kv_heads: usize,
    num_attention_heads: usize,
    comm: &Comm,
) -> Result<usize> {
    validate_tp_head_layout(num_attention_heads, total_num_kv_heads, comm.world_size())?;
    let kv_replicate = if comm.world_size() > total_num_kv_heads {
        comm.world_size() / total_num_kv_heads
    } else {
        1
    };
    Ok((num_attention_heads / total_num_kv_heads)
        .checked_div(kv_replicate)
        .unwrap_or(num_attention_heads / total_num_kv_heads))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use regex::Regex;

    use super::{validate_tp_head_layout, ColumnParallelLayer, ReplicatedLayer, RowParallelLayer};
    use crate::{
        create_isq_executor, set_immediate_isq_config, Comm, Id, ImmediateIsqConfig,
        ImmediateIsqOverride, IsqCaptureMode, IsqExecutorConfig, IsqType, LoraLayerRegistry,
        LoraLinearSpec, QuantMethod, QuantMethodConfig, QuantizedConfig, QuantizedWeightSource,
        Shard, ShardedSafeTensors, UnquantLinear,
    };

    struct DenseWeightSource;

    impl QuantizedWeightSource for DenseWeightSource {
        fn contains(&self, name: &str) -> bool {
            name == "model.embed_tokens.weight"
        }

        fn load_linear(
            &self,
            key: &str,
            device: &Device,
            _shard: Shard,
        ) -> candle_core::Result<Option<std::sync::Arc<dyn QuantMethod>>> {
            if key != "model.embed_tokens" {
                return Ok(None);
            }
            let weight = Tensor::zeros((64, 32), DType::F32, device)?;
            Ok(Some(std::sync::Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, None)),
            )?)))
        }

        fn load_optional_tensor(
            &self,
            _name: &str,
            _device: &Device,
        ) -> candle_core::Result<Option<Tensor>> {
            Ok(None)
        }

        fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    struct MatformerWeightSource;

    impl QuantizedWeightSource for MatformerWeightSource {
        fn contains(&self, name: &str) -> bool {
            matches!(name, "row.weight" | "column.weight" | "replicated.weight")
        }

        fn load_linear(
            &self,
            key: &str,
            device: &Device,
            _shard: Shard,
        ) -> candle_core::Result<Option<std::sync::Arc<dyn QuantMethod>>> {
            let shape = match key {
                "row" => (5, 8),
                "column" => (8, 4),
                "replicated" => (6, 4),
                _ => return Ok(None),
            };
            let weight = Tensor::zeros(shape, DType::F32, device)?;
            Ok(Some(std::sync::Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, None)),
            )?)))
        }

        fn load_optional_tensor(
            &self,
            name: &str,
            device: &Device,
        ) -> candle_core::Result<Option<Tensor>> {
            let size = match name {
                "row.bias" => 5,
                "column.bias" => 8,
                "replicated.bias" => 6,
                _ => return Ok(None),
            };
            Tensor::zeros(size, DType::F32, device).map(Some)
        }

        fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    fn install_immediate(ty: IsqType, overrides: Vec<ImmediateIsqOverride>) {
        let ty = Some(ty);
        let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
        let promoted = Regex::new(r"^model\.embed_tokens\.weight$").unwrap();
        set_immediate_isq_config(
            ImmediateIsqConfig::new(ty, vec![promoted.clone()], IsqCaptureMode::Immediate)
                .with_promoted_predicates(vec![promoted])
                .with_overrides(overrides),
            executor,
        );
    }

    fn tracked_from_linear() -> crate::TrackedModule {
        let vb = ShardedSafeTensors::wrap(
            HashMap::<String, candle_core::Tensor>::new(),
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )
        .pp("model")
        .pp("embed_tokens");
        let tracker = vb.tracker().clone();
        let linear = candle_nn::Linear::new(
            candle_core::Tensor::zeros(
                (64, 256),
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap(),
            None,
        );
        let _ = ReplicatedLayer::from_linear(linear, vb).unwrap();
        let tracked = tracker.get()[0].clone();
        crate::clear_immediate_isq();
        tracked
    }

    #[test]
    fn replicated_from_linear_uses_sensitive_default() {
        install_immediate(IsqType::AFQ4, Vec::new());
        assert_eq!(tracked_from_linear().ty, Some(IsqType::AFQ6));
    }

    #[test]
    fn replicated_from_linear_preserves_explicit_type() {
        install_immediate(
            IsqType::AFQ4,
            vec![ImmediateIsqOverride {
                predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
                layer_range: None,
                ty: Some(IsqType::AFQ2),
                device: None,
            }],
        );
        assert_eq!(tracked_from_linear().ty, Some(IsqType::AFQ2));
    }

    #[test]
    fn replicated_from_linear_promotes_q4k_sensitive_default() {
        install_immediate(IsqType::Q4K, Vec::new());
        assert_eq!(tracked_from_linear().ty, Some(IsqType::Q6K));
    }

    #[test]
    fn replicated_from_linear_promotes_q6k_sensitive_default() {
        install_immediate(IsqType::Q6K, Vec::new());
        assert_eq!(tracked_from_linear().ty, Some(IsqType::Q8_0));
    }

    #[test]
    fn replicated_from_linear_preserves_explicit_q_type() {
        install_immediate(
            IsqType::Q4K,
            vec![ImmediateIsqOverride {
                predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
                layer_range: None,
                ty: Some(IsqType::Q4_0),
                device: None,
            }],
        );
        assert_eq!(tracked_from_linear().ty, Some(IsqType::Q4_0));
    }

    #[test]
    fn replicated_weight_source_is_tracked_for_immediate_isq() {
        install_immediate(IsqType::Q8_0, Vec::new());
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                .with_weight_source(std::sync::Arc::new(DenseWeightSource))
                .pp("model")
                .pp("embed_tokens");
        let tracker = vb.tracker().clone();
        let layer = ReplicatedLayer::new(32, 64, &None, false, vb).unwrap();
        crate::clear_immediate_isq();

        layer
            .forward_raw(&Tensor::zeros((1, 32), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        assert_eq!(tracker.get().len(), 1);
        assert_eq!(tracker.get()[0].key, "model.embed_tokens");
    }

    #[test]
    fn weight_source_precedes_checkpoint_quantization_for_parallel_linears() {
        let config = Some(QuantizedConfig::GptqAwq {
            bits: 4,
            group_size: 128,
            checkpoint_format: None,
            is_awq: true,
        });
        let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).unwrap());
        let vb = || {
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                .with_weight_source(std::sync::Arc::new(DenseWeightSource))
                .pp("model")
                .pp("embed_tokens")
        };

        let layers = [
            ReplicatedLayer::new(32, 64, &config, false, vb()).unwrap(),
            ColumnParallelLayer::new(32, 64, &config, false, &comm, vb()).unwrap(),
            RowParallelLayer::new(32, 64, &config, false, &comm, vb()).unwrap(),
        ];
        for layer in layers {
            assert_eq!(layer.name(), "unquant-linear");
            assert_eq!(
                layer
                    .forward_raw(&Tensor::zeros((1, 32), DType::F32, &Device::Cpu).unwrap())
                    .unwrap()
                    .dims(),
                &[1, 64]
            );
        }
    }

    #[test]
    fn matformer_factories_transform_weight_source_layers() -> candle_core::Result<()> {
        let ty = Some(IsqType::Q8_0);
        let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
        set_immediate_isq_config(
            ImmediateIsqConfig::new(
                ty,
                vec![Regex::new(r"^(row|column|replicated)\.weight$").unwrap()],
                IsqCaptureMode::CaptureMatches,
            ),
            executor,
        );
        let config = Some(QuantizedConfig::GptqAwq {
            bits: 4,
            group_size: 128,
            checkpoint_format: None,
            is_awq: true,
        });
        let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
        let vb = |prefix| {
            let vb =
                ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                    .with_weight_source(std::sync::Arc::new(MatformerWeightSource))
                    .pp(prefix);
            let tracker = vb.tracker().clone();
            (vb, tracker)
        };

        let (row_vb, row_tracker) = vb("row");
        let row = RowParallelLayer::new_matformer(4, 5, 8, &config, true, &comm, row_vb)?;
        let (column_vb, column_tracker) = vb("column");
        let column = ColumnParallelLayer::new_matformer(4, 4, 8, &config, true, &comm, column_vb)?;
        let kept = Tensor::new(&[0u32, 2], &Device::Cpu)?;
        let (replicated_vb, replicated_tracker) = vb("replicated");
        let replicated = ReplicatedLayer::new_layers_matformer_indices(
            4,
            6,
            Some(&kept),
            3,
            &config,
            true,
            replicated_vb,
        )?;
        crate::clear_immediate_isq();

        for tracker in [row_tracker, column_tracker, replicated_tracker] {
            let tracked = tracker.get();
            assert_eq!(tracked.len(), 1);
            assert!(tracked[0].shard.is_none());
        }

        for (layer, input, output) in [(row, 4, 5), (column, 4, 4), (replicated, 4, 4)] {
            assert_eq!(
                layer
                    .forward_raw(&Tensor::zeros((1, input), DType::F32, &Device::Cpu)?)?
                    .dims(),
                &[1, output]
            );
        }
        Ok(())
    }

    #[test]
    fn spec_aware_layers_register_exact_feature_maps() -> candle_core::Result<()> {
        let replicated_registry = std::sync::Arc::new(LoraLayerRegistry::new());
        let replicated_vb = ShardedSafeTensors::wrap(
            HashMap::from([(
                "replicated.weight".to_string(),
                Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
            )]),
            DType::F32,
            Device::Cpu,
        )
        .with_lora_registry(replicated_registry.clone())
        .pp("replicated");
        let replicated_spec =
            LoraLinearSpec::replicated(4, 3).with_output_runtime_to_canonical(vec![2, 0, 1])?;
        ReplicatedLayer::new_with_lora_spec(replicated_spec.clone(), &None, false, replicated_vb)?;
        assert_eq!(replicated_registry.sites()[0].spec(), &replicated_spec);

        let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
        let row_registry = std::sync::Arc::new(LoraLayerRegistry::new());
        let row_vb = ShardedSafeTensors::wrap(
            HashMap::from([(
                "row.weight".to_string(),
                Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
            )]),
            DType::F32,
            Device::Cpu,
        )
        .with_lora_registry(row_registry.clone())
        .pp("row");
        let row_spec = LoraLinearSpec::row(
            4,
            3,
            Shard::Simple {
                dim: 1,
                rank: 0,
                world_size: 1,
            },
        )
        .with_input_runtime_to_canonical(vec![0, 2, 1, 3])?;
        RowParallelLayer::new_with_lora_spec(row_spec.clone(), &None, false, &comm, row_vb)?;
        assert_eq!(row_registry.sites()[0].spec(), &row_spec);
        Ok(())
    }

    #[test]
    fn spec_aware_layers_reject_wrong_parallelism() -> candle_core::Result<()> {
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu);
        let row_spec = LoraLinearSpec::row(
            4,
            3,
            Shard::Simple {
                dim: 1,
                rank: 0,
                world_size: 1,
            },
        );
        assert!(ReplicatedLayer::new_with_lora_spec(row_spec, &None, false, vb.clone()).is_err());

        let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
        assert!(RowParallelLayer::new_with_lora_spec(
            LoraLinearSpec::replicated(4, 3),
            &None,
            false,
            &comm,
            vb,
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn tp_head_layout_accepts_partitioned_kv_heads() {
        validate_tp_head_layout(48, 12, 3).unwrap();
    }

    #[test]
    fn tp_head_layout_accepts_replicated_kv_heads() {
        validate_tp_head_layout(32, 2, 4).unwrap();
    }

    #[test]
    fn tp_head_layout_rejects_attention_head_remainder() {
        let err = validate_tp_head_layout(40, 8, 6).unwrap_err();
        assert!(err.to_string().contains("attention heads (40)"));
    }

    #[test]
    fn tp_head_layout_rejects_partitioned_kv_remainder() {
        let err = validate_tp_head_layout(30, 8, 3).unwrap_err();
        assert!(err.to_string().contains("KV heads (8)"));
    }

    #[test]
    fn tp_head_layout_rejects_replicated_kv_remainder() {
        let err = validate_tp_head_layout(24, 2, 3).unwrap_err();
        assert!(err.to_string().contains("Tensor parallel size (3)"));
    }
}
