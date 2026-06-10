use std::sync::Arc;

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;

use crate::{
    blockwise_fp8::{blockwise_fp8_linear_b, blockwise_fp8_moe},
    distributed,
    gptq::gptq_linear,
    lora::merge_lora_weights,
    make_dummy_or_error,
    pertensor_fp8::pertensor_fp8_linear_b,
    should_apply_immediate_isq,
    utils::isq::{apply_immediate_isq, spawn_pending_isq},
    AfqLayer, BnbLinear, DistributedKind, MXFP4Layer, QuantMethod, QuantMethodConfig,
    QuantizeOntoGuard, QuantizedConfig, QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
};

use super::Comm;

fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}

fn load_uqff_linear(
    config: &Option<QuantizedConfig>,
    vb: &ShardedVarBuilder,
) -> Result<Option<Arc<dyn QuantMethod>>> {
    load_uqff_linear_shard(config, Shard::default(), vb)
}

fn load_uqff_linear_shard(
    config: &Option<QuantizedConfig>,
    shard: Shard,
    vb: &ShardedVarBuilder,
) -> Result<Option<Arc<dyn QuantMethod>>> {
    if config.is_some() {
        return Ok(None);
    }

    let Some(reader) = vb.uqff_reader() else {
        return Ok(None);
    };

    reader.load_linear(&vb.prefix(), vb.device(), shard)
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

        let base_vb = vb.clone();
        if let Some(weight) = load_uqff_linear_shard(config, shard, &base_vb)? {
            if world_size == 1 {
                // Bias is embedded in the layer when the input dim is not actually sharded.
                return Ok(weight);
            }
            // Row-sharded deserializes skip the bias; it must be applied once, post-reduce.
            let bias = if bias {
                base_vb
                    .uqff_reader()
                    .expect("reader present")
                    .load_optional_tensor(&format!("{}.bias", base_vb.prefix()), base_vb.device())?
            } else {
                None
            };
            return Ok(Arc::new(Self {
                weight,
                bias,
                all_reduce: distributed::SumAllReduce::new(comm),
            }));
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
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };

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
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };

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
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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

    fn begin_track_stats(&self) -> Result<()> {
        self.weight.begin_track_stats()
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

    fn has_bias(&self) -> bool {
        self.bias.is_some() || self.weight.has_bias()
    }

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        self.weight.get_qtensor()
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
    fn serialize_directly(
        &self,
        prefix: &str,
        ty: crate::IsqType,
    ) -> Result<Vec<crate::UqffTensor>> {
        let mut tensors = self.weight.serialize_directly(prefix, ty)?;
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
        let base_vb = vb.clone();
        if let Some(layer) = load_uqff_linear_shard(config, shard, &base_vb)? {
            return Ok(layer);
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
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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
            let layer = ColumnParallelLayer::new_with_shard(
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

    fn begin_track_stats(&self) -> Result<()> {
        self.weight.begin_track_stats()
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

    fn has_bias(&self) -> bool {
        self.bias.is_some() || self.weight.has_bias()
    }

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        self.weight.get_qtensor()
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
    fn serialize_directly(
        &self,
        prefix: &str,
        ty: crate::IsqType,
    ) -> Result<Vec<crate::UqffTensor>> {
        let mut tensors = self.weight.serialize_directly(prefix, ty)?;
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
        if let Some(layer) = load_uqff_linear(&None, &vb)? {
            return Ok(layer);
        }

        let dev = lin.weight().device().clone();
        if let Some(params) = crate::get_immediate_isq() {
            if let Some(immediate_isq) = params.ty {
                let lin = if !dev.is_cpu() {
                    Linear::new(lin.weight().to_device(&Device::Cpu)?, lin.bias().cloned())
                } else {
                    lin
                };
                let layer: Arc<dyn QuantMethod> =
                    Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lin))?);
                let layer = spawn_pending_isq(layer, Some(immediate_isq), dev, &params);
                vb.tracker().add_module(crate::TrackedModule {
                    key: vb.prefix(),
                    ct: layer.clone(),
                    ty: Some(immediate_isq),
                });
                return Ok(layer);
            }
            if params.capture != crate::IsqCaptureMode::Immediate {
                let layer: Arc<dyn QuantMethod> =
                    Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lin))?);
                let layer = spawn_pending_isq(layer, None, dev, &params);
                vb.tracker().add_module(crate::TrackedModule {
                    key: vb.prefix(),
                    ct: layer.clone(),
                    ty: params.ty,
                });
                return Ok(layer);
            }
        }

        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(lin),
        )?))
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base_vb = vb.clone();
        if let Some(layer) = load_uqff_linear(config, &base_vb)? {
            return Ok(layer);
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
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

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

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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

                weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

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

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
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

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        self.0.add_delta_w(delta)
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.0.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.0.dtype_and_device()
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.0.begin_track_stats()
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

    fn has_bias(&self) -> bool {
        self.0.has_bias()
    }

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        self.0.get_qtensor()
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
    fn serialize_directly(
        &self,
        prefix: &str,
        ty: crate::IsqType,
    ) -> Result<Vec<crate::UqffTensor>> {
        self.0.serialize_directly(prefix, ty)
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
    use super::validate_tp_head_layout;

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
