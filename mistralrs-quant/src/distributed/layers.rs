use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;

use crate::{
    blockwise_fp8::blockwise_fp8_moe, distributed, gptq::gptq_linear,
    lora::maybe_wrap_dynamic_lora_with_key, make_dummy_or_error, maybe_wrap_dynamic_lora,
    should_apply_immediate_isq, utils::isq::apply_immediate_isq_sharded, AfqLayer, BnbLinear,
    DistributedKind, LoraLinearSpec, LoraSiteKey, MXFP4Layer, QuantMethod, QuantMethodConfig,
    QuantizeOntoGuard, QuantizedActivation, QuantizedConfig, QuantizedSerde, Shard,
    ShardedVarBuilder, UnquantLinear,
};

use super::{
    packing::{load_packed_weights, PackRequest},
    Comm,
};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedOutputLayout {
    runtime_to_canonical: Arc<[usize]>,
}

impl PackedOutputLayout {
    pub fn identity(rows: usize) -> Self {
        Self {
            runtime_to_canonical: (0..rows).collect::<Vec<_>>().into(),
        }
    }

    pub fn rank_local_interleaved_to_grouped(
        groups: usize,
        segment_sizes: &[usize],
        world_size: usize,
    ) -> Result<Self> {
        if groups == 0 || world_size == 0 || segment_sizes.is_empty() {
            candle_core::bail!(
                "packed output layout requires nonzero groups, world size, and segments"
            );
        }
        if !groups.is_multiple_of(world_size) || segment_sizes.contains(&0) {
            candle_core::bail!(
                "packed output groups {groups} and segments {segment_sizes:?} are incompatible with world size {world_size}"
            );
        }
        let group_width = segment_sizes.iter().try_fold(0usize, |width, &segment| {
            width
                .checked_add(segment)
                .ok_or_else(|| candle_core::Error::msg("packed output group width overflow"))
        })?;
        let local_groups = groups / world_size;
        let local_rows = local_groups
            .checked_mul(group_width)
            .ok_or_else(|| candle_core::Error::msg("packed output local row count overflow"))?;
        let total_rows = local_rows
            .checked_mul(world_size)
            .ok_or_else(|| candle_core::Error::msg("packed output row count overflow"))?;
        let mut runtime_to_canonical = Vec::with_capacity(total_rows);
        for rank in 0..world_size {
            let rank_start = rank * local_rows;
            let mut segment_start = 0;
            for &segment_size in segment_sizes {
                for group in 0..local_groups {
                    let canonical_start = rank_start + group * group_width + segment_start;
                    runtime_to_canonical.extend(canonical_start..canonical_start + segment_size);
                }
                segment_start += segment_size;
            }
        }
        Self::from_runtime_to_canonical(runtime_to_canonical)
    }

    pub fn runtime_to_canonical(&self) -> &[usize] {
        &self.runtime_to_canonical
    }

    pub fn from_runtime_to_canonical(runtime_to_canonical: Vec<usize>) -> Result<Self> {
        let mut seen = vec![false; runtime_to_canonical.len()];
        for &canonical in &runtime_to_canonical {
            let Some(slot) = seen.get_mut(canonical) else {
                candle_core::bail!("packed output row permutation is out of bounds");
            };
            if std::mem::replace(slot, true) {
                candle_core::bail!("packed output row permutation contains duplicates");
            }
        }
        Ok(Self {
            runtime_to_canonical: runtime_to_canonical.into(),
        })
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.runtime_to_canonical
            .iter()
            .enumerate()
            .all(|(runtime, &canonical)| runtime == canonical)
    }

    pub(crate) fn local_runtime_to_canonical(
        &self,
        out_dim: usize,
        shard: Shard,
    ) -> Result<Option<Arc<[usize]>>> {
        if self.runtime_to_canonical.len() != out_dim {
            candle_core::bail!(
                "packed output layout has {} rows, expected {out_dim}",
                self.runtime_to_canonical.len()
            );
        }
        let (start, len) = match shard {
            Shard::Simple {
                dim: 0,
                rank,
                world_size,
            } => {
                if world_size == 0 || rank >= world_size || !out_dim.is_multiple_of(world_size) {
                    candle_core::bail!("invalid packed output shard");
                }
                let len = out_dim / world_size;
                (rank * len, len)
            }
            Shard::Offset {
                dim: 0,
                offset,
                len,
            } if offset.checked_add(len).is_some_and(|end| end <= out_dim) => (offset, len),
            _ => candle_core::bail!("packed output layouts require an output-dimension shard"),
        };
        let end = start + len;
        let local = self.runtime_to_canonical[start..end]
            .iter()
            .map(|&canonical| {
                if !(start..end).contains(&canonical) {
                    candle_core::bail!(
                        "packed output layout moves rows across tensor-parallel shard boundaries"
                    );
                }
                Ok(canonical - start)
            })
            .collect::<Result<Vec<_>>>()?;
        if local
            .iter()
            .enumerate()
            .all(|(runtime, &canonical)| runtime == canonical)
        {
            Ok(None)
        } else {
            Ok(Some(local.into()))
        }
    }
}

pub(crate) fn select_rows(tensor: &Tensor, rows: &[usize]) -> Result<Tensor> {
    // CUDA index_select has no FP8 kernel, so gather FP8 rows as contiguous runs instead.
    if tensor.dtype() == DType::F8E4M3 {
        let mut runs = Vec::new();
        let mut start = 0;
        for end in 1..=rows.len() {
            if end == rows.len() || rows[end] != rows[end - 1] + 1 {
                runs.push(tensor.narrow(0, rows[start], end - start)?);
                start = end;
            }
        }
        return Tensor::cat(&runs, 0);
    }
    let indices = rows
        .iter()
        .map(|&row| u32::try_from(row).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    let len = indices.len();
    tensor.index_select(&Tensor::from_vec(indices, len, tensor.device())?, 0)
}

#[derive(Debug)]
pub(crate) struct RuntimeOutputLinear {
    inner: Arc<dyn QuantMethod>,
    runtime_to_canonical: Arc<[usize]>,
    canonical_to_runtime: Arc<[usize]>,
}

impl RuntimeOutputLinear {
    pub(crate) fn wrap(
        inner: Arc<dyn QuantMethod>,
        runtime_to_canonical: Option<Arc<[usize]>>,
    ) -> Arc<dyn QuantMethod> {
        let Some(runtime_to_canonical) = runtime_to_canonical else {
            return inner;
        };
        let mut canonical_to_runtime = vec![0; runtime_to_canonical.len()];
        for (runtime, &canonical) in runtime_to_canonical.iter().enumerate() {
            canonical_to_runtime[canonical] = runtime;
        }
        Arc::new(Self {
            inner,
            runtime_to_canonical,
            canonical_to_runtime: canonical_to_runtime.into(),
        })
    }

    fn canonical_weight(&self) -> Result<Tensor> {
        select_rows(&self.inner.dequantize_w()?, &self.canonical_to_runtime)
    }

    fn runtime_weight(&self, canonical: &Tensor) -> Result<Tensor> {
        select_rows(canonical, &self.runtime_to_canonical)
    }
}

impl RuntimeOutputLinear {
    fn inner_layer(&self) -> Option<&dyn QuantMethod> {
        Some(&*self.inner)
    }
}

impl QuantMethod for RuntimeOutputLinear {
    crate::forward_quant_hooks!(meta, activation, output, weights, group: Self::inner_layer);

    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("RuntimeOutputLinear requires an existing projection")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.canonical_weight()
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.inner.forward_raw(a)
    }

    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.inner.gather_forward_raw(a, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        self.inner.quantized_act_type()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        self.inner.dtype_and_device()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        self.inner.plan_isq(request)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let inner = self.inner.add_delta_w(&self.runtime_weight(delta)?)?;
        Ok(Self::wrap(inner, Some(self.runtime_to_canonical.clone())))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if guard.consumer() == Some(crate::IsqConsumer::UqffWrite) {
            let canonical = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(self.canonical_weight()?, None),
            ))?) as Arc<dyn QuantMethod>;
            return canonical.apply_isq(dtype, device, n_quantized, imatrix_weight, guard);
        }
        let inner =
            self.inner
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        Ok(Self::wrap(inner, Some(self.runtime_to_canonical.clone())))
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        let (weight, bias) = self.inner.unquant_weight_bias()?;
        let weight = select_rows(&weight, &self.canonical_to_runtime).ok()?;
        let bias = bias
            .map(|bias| select_rows(&bias, &self.canonical_to_runtime))
            .transpose()
            .ok()?;
        Some((weight, bias))
    }

    fn has_bias(&self) -> bool {
        self.inner.has_bias()
    }
}

impl QuantizedSerde for RuntimeOutputLinear {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn isq_serde_supported(&self) -> bool {
        false
    }
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
                QuantizedConfig::Fp8 { .. }
                | QuantizedConfig::CompressedTensors { .. }
                | QuantizedConfig::ModelOpt { .. } => crate::fp8_config::checkpoint_linear_b(
                    in_dim,
                    out_dim,
                    quant_conf,
                    false,
                    shard,
                    vb.clone(),
                )?,
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

impl RowParallelLayer {
    fn weight_layer(&self) -> Option<&dyn QuantMethod> {
        Some(&*self.weight)
    }

    fn transparent_layer(&self) -> Option<&dyn QuantMethod> {
        (self.all_reduce.is_noop() && self.bias.is_none()).then_some(&*self.weight)
    }

    fn reduce_output(&self, mut xs: Tensor) -> Result<Tensor> {
        if !self.all_reduce.is_noop() {
            xs = self.all_reduce.sum_all_reduce(&xs.contiguous()?)?;
        }
        match &self.bias {
            Some(bias) => xs.broadcast_add(bias),
            None => Ok(xs),
        }
    }
}

impl QuantMethod for RowParallelLayer {
    crate::forward_quant_hooks!(meta, activation: Self::weight_layer);
    crate::forward_quant_hooks!(weights, group: Self::transparent_layer);

    fn forward_quantized(&self, a: &QuantizedActivation) -> Result<Tensor> {
        self.reduce_output(self.weight.forward_quantized(a)?)
    }

    fn forward_glu(
        &self,
        input: crate::GluInput<'_>,
        activation: crate::GluActivationType,
    ) -> Result<Option<Tensor>> {
        self.weight
            .forward_glu(input, activation)?
            .map(|output| self.reduce_output(output))
            .transpose()
    }

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
                QuantizedConfig::Fp8 { .. }
                | QuantizedConfig::CompressedTensors { .. }
                | QuantizedConfig::ModelOpt { .. } => crate::fp8_config::checkpoint_linear_b(
                    in_dim,
                    out_dim,
                    quant_conf,
                    false,
                    shard,
                    vb.clone(),
                )?,
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

    #[allow(clippy::too_many_arguments)]
    pub fn new_packed(
        in_dim: usize,
        out_dims: &[usize],
        names: &[&str],
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        shards: Option<&[Shard]>,
        vb: ShardedVarBuilder,
    ) -> Result<Option<PackedColumnParallel>> {
        let output_layouts = out_dims
            .iter()
            .map(|&rows| PackedOutputLayout::identity(rows))
            .collect::<Vec<_>>();
        Self::new_packed_with_output_layouts(
            in_dim,
            out_dims,
            names,
            &output_layouts,
            config,
            bias,
            comm,
            shards,
            vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_packed_with_output_layouts(
        in_dim: usize,
        out_dims: &[usize],
        names: &[&str],
        output_layouts: &[PackedOutputLayout],
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        shards: Option<&[Shard]>,
        vb: ShardedVarBuilder,
    ) -> Result<Option<PackedColumnParallel>> {
        if bias {
            return Ok(None);
        }
        if output_layouts.len() != names.len() {
            candle_core::bail!(
                "packed projection output layout count does not match projection count"
            );
        }
        let default_shard = shard(0, comm.rank(), comm.world_size());
        if shards.is_some_and(|shards| shards.len() != names.len()) {
            candle_core::bail!("packed projection shard count does not match projection count");
        }
        let shards = (0..names.len())
            .map(|i| shards.map_or(default_shard, |shards| shards[i]))
            .collect::<Vec<_>>();
        let Some(loaded) = load_packed_weights(&PackRequest {
            in_dim,
            out_dims,
            names,
            shards: &shards,
            output_layouts,
            config,
            vb: &vb,
        })?
        else {
            return Ok(None);
        };

        let mut constituents = Vec::with_capacity(loaded.constituents.len());
        for ((((name, &out_dim), &shard), layout), weight) in names
            .iter()
            .zip(out_dims)
            .zip(&shards)
            .zip(output_layouts)
            .zip(loaded.constituents)
        {
            let vb_n = vb.pp(name);
            let mut lora_spec = LoraLinearSpec::column(in_dim, out_dim, shard);
            if !layout.is_identity() {
                lora_spec = lora_spec
                    .with_output_runtime_to_canonical(layout.runtime_to_canonical.clone())?;
            }
            let wrapped = maybe_wrap_dynamic_lora_with_key(
                &vb_n,
                weight,
                LoraSiteKey::new(vb_n.prefix()),
                lora_spec,
            )?;
            constituents.push(Arc::new(Self {
                weight: wrapped,
                bias: None,
            }) as Arc<dyn QuantMethod>);
        }
        Ok(Some(PackedLinear {
            packed: loaded.packed,
            constituents,
            rows_per_rank: loaded.rows_per_rank,
        }))
    }

    /// Like `new_packed` for a checkpoint that already stores the projections fused in one
    /// tensor of `chunks` equal chunks (e.g. `gate_up_proj`): the fused tensor is loaded once and
    /// becomes the sole owner, constituents are chunk views. Single-rank only; tensor-parallel
    /// runs keep the sharded `new_merged` path.
    pub fn new_packed_from_fused(
        in_dim: usize,
        out_dim: usize,
        chunks: usize,
        config: &Option<QuantizedConfig>,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Option<PackedColumnParallel>> {
        if config.is_some() || crate::get_immediate_isq().is_some() || comm.world_size() != 1 {
            return Ok(None);
        }
        if should_apply_immediate_isq(&vb)
            || load_weight_source_linear_shard(Shard::default(), &vb)?.is_some()
            || !vb.contains_tensor("weight")
        {
            return Ok(None);
        }
        let packed_weight = vb.get_with_hints((out_dim, in_dim), "weight", Shard::default())?;
        let packed = Arc::new(<UnquantLinear as QuantMethod>::new(
            QuantMethodConfig::Unquantized(Linear::new(packed_weight.clone(), None)),
        )?) as Arc<dyn QuantMethod>;

        let rows = out_dim / chunks;
        let mut constituents = Vec::with_capacity(chunks);
        for chunk_idx in 0..chunks {
            let view = packed_weight.narrow(0, chunk_idx * rows, rows)?;
            let view_linear = Arc::new(<UnquantLinear as QuantMethod>::new(
                QuantMethodConfig::Unquantized(Linear::new(view, None)),
            )?) as Arc<dyn QuantMethod>;
            let wrapped = maybe_wrap_dynamic_lora_with_key(
                &vb,
                view_linear,
                LoraSiteKey::with_slice(vb.prefix(), chunk_idx, chunks)?,
                LoraLinearSpec::column(in_dim, out_dim, shard(0, chunk_idx, chunks)),
            )?;
            constituents.push(Arc::new(Self {
                weight: wrapped,
                bias: None,
            }) as Arc<dyn QuantMethod>);
        }
        Ok(Some(PackedLinear {
            packed,
            constituents,
            rows_per_rank: vec![rows; chunks],
        }))
    }
}

pub struct PackedLinear {
    pub packed: Arc<dyn QuantMethod>,
    pub constituents: Vec<Arc<dyn QuantMethod>>,
    pub rows_per_rank: Vec<usize>,
}

pub type PackedColumnParallel = PackedLinear;

impl ColumnParallelLayer {
    fn weight_layer(&self) -> Option<&dyn QuantMethod> {
        Some(&*self.weight)
    }

    fn transparent_layer(&self) -> Option<&dyn QuantMethod> {
        self.bias.is_none().then_some(&*self.weight)
    }

    fn add_bias(&self, xs: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => xs.broadcast_add(bias),
            None => Ok(xs),
        }
    }
}

impl QuantMethod for ColumnParallelLayer {
    crate::forward_quant_hooks!(meta, activation: Self::weight_layer);
    crate::forward_quant_hooks!(weights, group: Self::transparent_layer);

    fn forward_quantized(&self, a: &QuantizedActivation) -> Result<Tensor> {
        self.add_bias(self.weight.forward_quantized(a)?)
    }

    fn forward_glu(
        &self,
        input: crate::GluInput<'_>,
        activation: crate::GluActivationType,
    ) -> Result<Option<Tensor>> {
        self.weight
            .forward_glu(input, activation)?
            .map(|output| self.add_bias(output))
            .transpose()
    }

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
                QuantizedConfig::Fp8 { .. }
                | QuantizedConfig::CompressedTensors { .. }
                | QuantizedConfig::ModelOpt { .. } => crate::fp8_config::checkpoint_linear_b(
                    in_dim,
                    out_dim,
                    quant_conf,
                    bias,
                    Default::default(),
                    vb.clone(),
                )?,
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

    pub fn new_packed(
        lora_specs: &[LoraLinearSpec],
        names: &[&str],
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Option<PackedLinear>> {
        let output_layouts = lora_specs
            .iter()
            .map(|spec| PackedOutputLayout::identity(spec.out_features()))
            .collect::<Vec<_>>();
        Self::new_packed_with_output_layouts(lora_specs, names, &output_layouts, config, bias, vb)
    }

    pub fn new_packed_with_output_layouts(
        lora_specs: &[LoraLinearSpec],
        names: &[&str],
        output_layouts: &[PackedOutputLayout],
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Option<PackedLinear>> {
        if bias || lora_specs.is_empty() || lora_specs.len() != names.len() {
            return Ok(None);
        }
        if output_layouts.len() != names.len() {
            candle_core::bail!(
                "packed projection output layout count does not match projection count"
            );
        }
        let in_dim = lora_specs[0].in_features();
        if lora_specs
            .iter()
            .any(|spec| !spec.is_replicated() || spec.in_features() != in_dim)
        {
            candle_core::bail!(
                "packed replicated projections must share an input dimension and replicated layout"
            );
        }
        let out_dims = lora_specs
            .iter()
            .map(LoraLinearSpec::out_features)
            .collect::<Vec<_>>();
        let shards = vec![Shard::default(); names.len()];
        let Some(loaded) = load_packed_weights(&PackRequest {
            in_dim,
            out_dims: &out_dims,
            names,
            shards: &shards,
            output_layouts,
            config,
            vb: &vb,
        })?
        else {
            return Ok(None);
        };

        let mut constituents = Vec::with_capacity(loaded.constituents.len());
        for (((name, spec), layout), weight) in names
            .iter()
            .zip(lora_specs)
            .zip(output_layouts)
            .zip(loaded.constituents)
        {
            let vb_n = vb.pp(name);
            let mut spec = spec.clone();
            if !layout.is_identity() {
                spec =
                    spec.with_output_runtime_to_canonical(layout.runtime_to_canonical.clone())?;
            }
            let wrapped = maybe_wrap_dynamic_lora_with_key(
                &vb_n,
                weight,
                LoraSiteKey::new(vb_n.prefix()),
                spec,
            )?;
            constituents.push(Arc::new(Self(wrapped)) as Arc<dyn QuantMethod>);
        }
        Ok(Some(PackedLinear {
            packed: loaded.packed,
            constituents,
            rows_per_rank: loaded.rows_per_rank,
        }))
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
                QuantizedConfig::Fp8 { .. }
                | QuantizedConfig::CompressedTensors { .. }
                | QuantizedConfig::ModelOpt { .. } => crate::fp8_config::checkpoint_linear_b(
                    in_dim,
                    out_dim,
                    quant_conf,
                    bias,
                    Default::default(),
                    vb.clone(),
                )?,
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

impl ReplicatedLayer {
    fn inner_layer(&self) -> Option<&dyn QuantMethod> {
        Some(&*self.0)
    }
}

impl QuantMethod for ReplicatedLayer {
    crate::forward_quant_hooks!(meta, activation, output, weights, group: Self::inner_layer);

    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.0.gather_forward_raw(a, indices)
    }

    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("ReplicatedLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward_raw(a)
    }

    fn embedding_forward(&self, ids: &Tensor, out: crate::EmbeddingOutput) -> Result<Tensor> {
        self.0.embedding_forward(ids, out)
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

struct CheckpointExpertLoad<'a> {
    config: &'a QuantizedConfig,
    experts_vb: &'a ShardedVarBuilder,
    num_experts: usize,
}

impl CheckpointExpertLoad<'_> {
    fn projection(
        &self,
        in_dim: usize,
        out_dim: usize,
        name: &str,
    ) -> Result<Arc<dyn QuantMethod>> {
        use crate::fp8_config::{CheckpointLinearSpec, Fp8ActivationMode, Fp8WeightScaleLayout};

        let builders = (0..self.num_experts)
            .map(|expert| self.experts_vb.pp(expert).pp(name))
            .collect::<Vec<_>>();
        let specs = builders
            .iter()
            .map(|vb| self.config.resolve_checkpoint(&vb.prefix()))
            .collect::<Result<Vec<_>>>()?;
        let Some(first) = specs.first() else {
            candle_core::bail!("Checkpoint expert loading requires at least one expert");
        };
        if specs.iter().any(|spec| spec != first) {
            candle_core::bail!(
                "Checkpoint experts at `{}.*.{name}` require the same quantization format and activation scheme within each projection",
                self.experts_vb.prefix()
            );
        }

        match first {
            Some(CheckpointLinearSpec::Nvfp4(spec)) => {
                let layers = builders
                    .into_iter()
                    .map(|vb| {
                        crate::nvfp4::Nvfp4Layer::load(
                            in_dim,
                            out_dim,
                            *spec,
                            false,
                            Shard::default(),
                            vb,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(crate::nvfp4::Nvfp4Layer::stack(layers)?))
            }
            Some(CheckpointLinearSpec::Mxfp8(_)) => {
                let layers = builders
                    .into_iter()
                    .map(|vb| {
                        crate::mxfp8::Mxfp8Linear::load(
                            in_dim,
                            out_dim,
                            false,
                            Shard::default(),
                            vb,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(crate::mxfp8::Mxfp8Linear::stack(layers)?))
            }
            Some(CheckpointLinearSpec::Fp8(spec)) => {
                // Tensor/channel scales become one block per output row, which the blockwise gather serves exactly.
                let block_size = match spec.weight_scale {
                    Fp8WeightScaleLayout::Block(block_size) => block_size,
                    Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => [1, in_dim],
                };
                // Static/per-token activation scales are dropped: experts run W8A16 like the dense dequant fallback.
                let activation_scheme = match spec.activation {
                    Fp8ActivationMode::DynamicBlock(_) => Some(crate::Fp8ActivationScheme::Dynamic),
                    Fp8ActivationMode::None
                    | Fp8ActivationMode::StaticTensor
                    | Fp8ActivationMode::DynamicToken => None,
                };
                let mut weights = Vec::with_capacity(self.num_experts);
                let mut scales = Vec::with_capacity(self.num_experts);
                for vb in builders {
                    let scale_name = spec.scale_names.weight_name(&vb)?;
                    let scale_shape = vb
                        .tensor_shape(scale_name)
                        .map(<[usize]>::to_vec)
                        .unwrap_or(spec.weight_scale.logical_shape([out_dim, in_dim])?);
                    weights.push(vb.get_with_hints_dtype(
                        (out_dim, in_dim),
                        "weight",
                        Shard::default(),
                        DType::F8E4M3,
                    )?);
                    let scale = spec.weight_scale.normalize(
                        vb.get_with_hints_dtype(
                            scale_shape,
                            scale_name,
                            Shard::default(),
                            DType::F32,
                        )?,
                        [out_dim, in_dim],
                    )?;
                    scales.push(match spec.weight_scale {
                        Fp8WeightScaleLayout::Block(_) => scale,
                        Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => scale
                            .reshape(((), 1))?
                            .broadcast_as((out_dim, 1))?
                            .contiguous()?,
                    });
                }
                blockwise_fp8_moe(
                    Tensor::stack(&weights, 0)?,
                    Tensor::stack(&scales, 0)?,
                    block_size.to_vec(),
                    activation_scheme,
                    self.experts_vb.dtype(),
                )
            }
            None => {
                let weights = builders
                    .iter()
                    .map(|vb| vb.get((out_dim, in_dim), "weight"))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&weights, 0)?, None)),
                )?))
            }
        }
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
        } else if matches!(
            quantization_config,
            Some(QuantizedConfig::ModelOpt { .. } | QuantizedConfig::CompressedTensors { .. })
        ) {
            let loader = CheckpointExpertLoad {
                config: quantization_config.as_ref().unwrap(),
                experts_vb: &experts_vb,
                num_experts,
            };
            (
                loader.projection(hidden_size, moe_intermediate_size, "gate_proj")?,
                loader.projection(hidden_size, moe_intermediate_size, "up_proj")?,
                loader.projection(moe_intermediate_size, hidden_size, "down_proj")?,
            )
        } else if is_stacked_format
            && matches!(&quantization_config, Some(QuantizedConfig::Fp8 { .. }))
        {
            // Stacked format with FP8 quantization
            // Keep weights as FP8 using BlockwiseFP8 to leverage native FP8 GEMM in gather_forward
            let has_fp8_scales = experts_vb.contains_tensor("gate_up_proj.weight_scale_inv");

            if has_fp8_scales {
                let (weight_block_size, activation_scheme) = match quantization_config {
                    Some(QuantizedConfig::Fp8 {
                        weight_block_size,
                        activation_scheme,
                        ..
                    }) => (weight_block_size.clone(), *activation_scheme),
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
                let fused_gate_proj = blockwise_fp8_moe(
                    gate_fp8,
                    gate_scale,
                    weight_block_size.clone(),
                    activation_scheme,
                    vb.dtype(),
                )?;
                let fused_up_proj = blockwise_fp8_moe(
                    up_fp8,
                    up_scale,
                    weight_block_size.clone(),
                    activation_scheme,
                    vb.dtype(),
                )?;
                let fused_down_proj = blockwise_fp8_moe(
                    down_fp8,
                    down_scale,
                    weight_block_size,
                    activation_scheme,
                    vb.dtype(),
                )?;

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
            let (weight_block_size, activation_scheme) = match quantization_config {
                Some(QuantizedConfig::Fp8 {
                    weight_block_size,
                    activation_scheme,
                    ..
                }) => (weight_block_size.clone(), *activation_scheme),
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
            let fused_gate_proj = blockwise_fp8_moe(
                gate_fp8,
                gate_scale,
                weight_block_size.clone(),
                activation_scheme,
                vb.dtype(),
            )?;
            let fused_up_proj = blockwise_fp8_moe(
                up_fp8,
                up_scale,
                weight_block_size.clone(),
                activation_scheme,
                vb.dtype(),
            )?;
            let fused_down_proj = blockwise_fp8_moe(
                down_fp8,
                down_scale,
                weight_block_size,
                activation_scheme,
                vb.dtype(),
            )?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else {
            candle_core::bail!(
                "PreQuantizedExperts loads pre-quantized expert formats only (AFQ, blockwise FP8, NVFP4, MXFP4)."
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
#[path = "layers_tests.rs"]
mod tests;
