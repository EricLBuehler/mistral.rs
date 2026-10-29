use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;

use super::layers::{select_rows, PackedOutputLayout, RuntimeOutputLinear};
use crate::{
    fp8_config::{
        checkpoint_fp8_dequant_dtype, checkpoint_fp8_weight_scale_name,
        load_checkpoint_fp8_tensors, CheckpointFp8Tensors,
    },
    mxfp8::Mxfp8Linear,
    pertensor_fp8::PerTensorFp8Parts,
    should_apply_immediate_isq, BlockwiseFP8Linear, CheckpointLinearSpec, Fp8ActivationMode,
    Fp8LinearSpec, Fp8WeightScaleLayout, Nvfp4Layer, Nvfp4LinearSpec, PackedLinear,
    PerTensorFP8Linear, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
};

const DENSE_PACKED_ROW_ALIGNMENT: usize = 16;

/// Output-row alignment of a packed dense weight; half-precision CUDA GEMMs want 16-row strides.
pub fn dense_pack_row_alignment(dtype: DType, cuda: bool) -> usize {
    if cuda && matches!(dtype, DType::F16 | DType::BF16) {
        DENSE_PACKED_ROW_ALIGNMENT
    } else {
        1
    }
}

/// Resident bytes of a dense packed projection with `rows` logical output rows.
pub fn packed_dense_bytes(rows: usize, cols: usize, dtype: DType, cuda: bool) -> usize {
    rows.next_multiple_of(dense_pack_row_alignment(dtype, cuda)) * cols * dtype.size_in_bytes()
}

pub(super) struct PackRequest<'a> {
    pub in_dim: usize,
    pub out_dims: &'a [usize],
    pub names: &'a [&'a str],
    pub shards: &'a [Shard],
    pub output_layouts: &'a [PackedOutputLayout],
    pub config: &'a Option<QuantizedConfig>,
    pub vb: &'a ShardedVarBuilder,
}

struct PackSite<'a> {
    vb: &'a ShardedVarBuilder,
    in_dim: usize,
    out_dim: usize,
    shard: Shard,
}

impl PackSite<'_> {
    fn local_rows(&self) -> usize {
        match self.shard {
            Shard::Simple { world_size, .. } => self.out_dim / world_size,
            Shard::Offset { len, .. } => len,
        }
    }
}

/// A projection format whose output rows can be loaded per part, permuted, and row-concatenated.
trait PackableLinear: Sized {
    type Spec;

    /// `None` leaves the group unpacked so the per-projection loader handles (or reports) it.
    fn load_shard(spec: Self::Spec, site: &PackSite<'_>) -> Result<Option<Self>>;

    /// `runtime_to_canonical[i]` is the local checkpoint row stored at runtime row `i`.
    fn permute_output_rows(self, runtime_to_canonical: &[usize]) -> Result<Option<Self>>;

    fn merge(parts: Vec<Self>) -> Result<Option<PackedLinear>>;
}

enum PackFormat {
    Dense(DenseSource),
    Mxfp8,
    Nvfp4(Vec<Nvfp4LinearSpec>),
    Fp8(Vec<Fp8LinearSpec>),
}

pub(super) fn load_packed_weights(req: &PackRequest<'_>) -> Result<Option<PackedLinear>> {
    let count = req.names.len();
    if count == 0
        || req.out_dims.len() != count
        || req.shards.len() != count
        || req.output_layouts.len() != count
    {
        candle_core::bail!(
            "packed projection requires matching nonempty output dimensions, names, and shards"
        );
    }
    if crate::get_immediate_isq().is_some() {
        return Ok(None);
    }
    let builders = req
        .names
        .iter()
        .map(|name| req.vb.pp(name))
        .collect::<Vec<_>>();
    if builders.iter().any(should_apply_immediate_isq) {
        return Ok(None);
    }
    match detect_format(req.config, &builders)? {
        None => Ok(None),
        Some(PackFormat::Dense(source)) => pack::<DensePart>(req, &builders, vec![source; count]),
        Some(PackFormat::Mxfp8) => pack::<Mxfp8Linear>(req, &builders, vec![(); count]),
        Some(PackFormat::Nvfp4(specs)) => pack::<Nvfp4Layer>(req, &builders, specs),
        Some(PackFormat::Fp8(specs)) => pack::<Fp8Part>(req, &builders, specs),
    }
}

fn detect_format(
    config: &Option<QuantizedConfig>,
    builders: &[ShardedVarBuilder],
) -> Result<Option<PackFormat>> {
    if builders.iter().any(|vb| vb.weight_source().is_some()) {
        for vb in builders {
            let Some(source) = vb.weight_source() else {
                return Ok(None);
            };
            if !source.supports_dense_packing(&vb.prefix())? {
                return Ok(None);
            }
        }
        return Ok(Some(PackFormat::Dense(DenseSource::WeightSource)));
    }
    let checkpoint_dense = || {
        builders
            .iter()
            .all(|vb| vb.contains_tensor("weight"))
            .then_some(PackFormat::Dense(DenseSource::Checkpoint))
    };
    let config = match config {
        None => return Ok(checkpoint_dense()),
        Some(
            config @ (QuantizedConfig::Fp8 { .. }
            | QuantizedConfig::ModelOpt { .. }
            | QuantizedConfig::CompressedTensors { .. }),
        ) => config,
        Some(_) => return Ok(None),
    };
    let specs = builders
        .iter()
        .map(|vb| config.resolve_checkpoint(&vb.prefix()))
        .collect::<Result<Vec<_>>>()?;
    if specs.iter().all(Option::is_none) {
        return Ok(checkpoint_dense());
    }
    let mut mxfp8 = 0;
    let mut nvfp4 = Vec::new();
    let mut fp8 = Vec::new();
    for (vb, spec) in builders.iter().zip(specs) {
        match spec {
            Some(CheckpointLinearSpec::Mxfp8(_)) if vb.contains_tensor("weight_scale") => {
                mxfp8 += 1
            }
            Some(CheckpointLinearSpec::Nvfp4(spec))
                if vb.contains_tensor(spec.scale_names.weight) =>
            {
                nvfp4.push(spec)
            }
            Some(CheckpointLinearSpec::Fp8(spec)) => fp8.push(spec),
            _ => return Ok(None),
        }
    }
    let count = builders.len();
    Ok(if mxfp8 == count {
        Some(PackFormat::Mxfp8)
    } else if nvfp4.len() == count {
        Some(PackFormat::Nvfp4(nvfp4))
    } else if fp8.len() == count {
        Some(PackFormat::Fp8(fp8))
    } else {
        None
    })
}

fn pack<P: PackableLinear>(
    req: &PackRequest<'_>,
    builders: &[ShardedVarBuilder],
    specs: Vec<P::Spec>,
) -> Result<Option<PackedLinear>> {
    let mut parts = Vec::with_capacity(builders.len());
    let mut output_maps = Vec::with_capacity(builders.len());
    for ((((vb, &out_dim), &shard), layout), spec) in builders
        .iter()
        .zip(req.out_dims)
        .zip(req.shards)
        .zip(req.output_layouts)
        .zip(specs)
    {
        let site = PackSite {
            vb,
            in_dim: req.in_dim,
            out_dim,
            shard,
        };
        let Some(part) = P::load_shard(spec, &site)? else {
            return Ok(None);
        };
        let output_map = layout.local_runtime_to_canonical(out_dim, shard)?;
        let part = match &output_map {
            Some(rows) => match part.permute_output_rows(rows)? {
                Some(part) => part,
                None => return Ok(None),
            },
            None => part,
        };
        parts.push(part);
        output_maps.push(output_map);
    }
    let Some(group) = P::merge(parts)? else {
        return Ok(None);
    };
    Ok(Some(PackedLinear {
        packed: group.packed,
        constituents: group
            .constituents
            .into_iter()
            .zip(output_maps)
            .map(|(layer, output_map)| RuntimeOutputLinear::wrap(layer, output_map))
            .collect(),
        rows_per_rank: group.rows_per_rank,
    }))
}

fn same_input_and_device(weights: &[&Tensor]) -> Result<bool> {
    let Some(first) = weights.first() else {
        return Ok(false);
    };
    let k = first.dim(D::Minus1)?;
    for weight in weights {
        if weight.rank() != 2 || weight.dim(1)? != k || !weight.device().same_device(first.device())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
enum DenseSource {
    WeightSource,
    Checkpoint,
}

struct DensePart(Tensor);

impl PackableLinear for DensePart {
    type Spec = DenseSource;

    fn load_shard(spec: DenseSource, site: &PackSite<'_>) -> Result<Option<Self>> {
        let weight = match spec {
            DenseSource::Checkpoint => {
                site.vb
                    .get_with_hints((site.out_dim, site.in_dim), "weight", site.shard)?
            }
            DenseSource::WeightSource => {
                let Some(source) = site.vb.weight_source() else {
                    return Ok(None);
                };
                let device = crate::weight_source_load_device(site.vb);
                let Some(layer) = source.load_linear(&site.vb.prefix(), &device, site.shard)?
                else {
                    return Ok(None);
                };
                let Some((weight, None)) = layer.unquant_weight_bias() else {
                    return Ok(None);
                };
                weight
            }
        };
        if weight.dims() != [site.local_rows(), site.in_dim]
            || !matches!(weight.dtype(), DType::BF16 | DType::F16 | DType::F32)
        {
            return Ok(None);
        }
        Ok(Some(Self(weight)))
    }

    fn permute_output_rows(self, runtime_to_canonical: &[usize]) -> Result<Option<Self>> {
        select_rows(&self.0, runtime_to_canonical).map(|weight| Some(Self(weight)))
    }

    fn merge(parts: Vec<Self>) -> Result<Option<PackedLinear>> {
        let weights = parts.iter().map(|part| &part.0).collect::<Vec<_>>();
        if !same_input_and_device(&weights)?
            || weights
                .iter()
                .any(|weight| weight.dtype() != weights[0].dtype())
        {
            return Ok(None);
        }
        pack_dense_weights(parts.into_iter().map(|part| part.0).collect()).map(Some)
    }
}

fn unquant(weight: Tensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(Linear::new(weight, None)),
    )?))
}

fn pack_dense_weights(weights: Vec<Tensor>) -> Result<PackedLinear> {
    let alignment = dense_pack_row_alignment(weights[0].dtype(), weights[0].device().is_cuda());
    pack_dense_weights_aligned(weights, alignment)
}

fn pack_dense_weights_aligned(mut weights: Vec<Tensor>, alignment: usize) -> Result<PackedLinear> {
    let rows_per_rank = weights
        .iter()
        .map(|weight| weight.dim(0))
        .collect::<Result<Vec<_>>>()?;
    let total_rows = rows_per_rank.iter().sum::<usize>();
    let (dtype, device, cols) = (
        weights[0].dtype(),
        weights[0].device().clone(),
        weights[0].dim(1)?,
    );
    let padding = total_rows.next_multiple_of(alignment) - total_rows;
    if padding != 0 {
        weights.push(Tensor::zeros((padding, cols), dtype, &device)?);
    }
    let packed_weight = Tensor::cat(&weights, 0)?;
    let mut constituents = Vec::with_capacity(rows_per_rank.len());
    let mut offset = 0;
    for &rows in &rows_per_rank {
        constituents.push(unquant(packed_weight.narrow(0, offset, rows)?)?);
        offset += rows;
    }
    let packed = unquant(packed_weight)?;
    let packed = if padding == 0 {
        packed
    } else {
        Arc::new(PaddedOutputLinear {
            inner: packed,
            rows: total_rows,
        })
    };
    Ok(PackedLinear {
        packed,
        constituents,
        rows_per_rank,
    })
}

/// Keeps padded storage rows for GEMM alignment while exposing only the logical outputs.
#[derive(Debug)]
struct PaddedOutputLinear {
    inner: Arc<dyn QuantMethod>,
    rows: usize,
}

impl QuantMethod for PaddedOutputLinear {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("PaddedOutputLinear requires an existing projection")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.inner.dequantize_w()?.narrow(0, 0, self.rows)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.inner.forward_raw(a)?.narrow(D::Minus1, 0, self.rows)
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

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("packed projections apply weight deltas through their constituents")
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let inner =
            self.inner
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        Ok(Arc::new(Self {
            inner,
            rows: self.rows,
        }))
    }
}

impl QuantizedSerde for PaddedOutputLinear {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn isq_serde_supported(&self) -> bool {
        false
    }
}

impl PackableLinear for Mxfp8Linear {
    type Spec = ();

    fn load_shard(_spec: (), site: &PackSite<'_>) -> Result<Option<Self>> {
        Self::load(
            site.in_dim,
            site.out_dim,
            false,
            site.shard,
            site.vb.clone(),
        )
        .map(Some)
    }

    fn permute_output_rows(self, runtime_to_canonical: &[usize]) -> Result<Option<Self>> {
        Mxfp8Linear::permute_output_rows(self, runtime_to_canonical).map(Some)
    }

    fn merge(parts: Vec<Self>) -> Result<Option<PackedLinear>> {
        Mxfp8Linear::merge(parts)
    }
}

impl PackableLinear for Nvfp4Layer {
    type Spec = Nvfp4LinearSpec;

    fn load_shard(spec: Nvfp4LinearSpec, site: &PackSite<'_>) -> Result<Option<Self>> {
        Self::load(
            site.in_dim,
            site.out_dim,
            spec,
            false,
            site.shard,
            site.vb.clone(),
        )
        .map(Some)
    }

    fn permute_output_rows(self, runtime_to_canonical: &[usize]) -> Result<Option<Self>> {
        Nvfp4Layer::permute_output_rows(self, runtime_to_canonical).map(Some)
    }

    fn merge(parts: Vec<Self>) -> Result<Option<PackedLinear>> {
        Nvfp4Layer::merge(parts)
    }
}

/// One FP8 projection with checkpoint scales, normalized to `()`, `[rows]`, or `[row_blocks, col_blocks]`.
struct Fp8Part {
    weight: Tensor,
    scale: Tensor,
    layout: Fp8WeightScaleLayout,
    activation: Fp8ActivationMode,
    activation_scale: Option<Tensor>,
    dequant_dtype: DType,
}

/// Block indices for a row permutation that moves whole scale blocks, or `None` if it splits one.
fn block_permutation(runtime_to_canonical: &[usize], block_rows: usize) -> Option<Vec<usize>> {
    if !runtime_to_canonical.len().is_multiple_of(block_rows) {
        return None;
    }
    runtime_to_canonical
        .chunks(block_rows)
        .map(|block| {
            let start = block[0];
            (start.is_multiple_of(block_rows)
                && block
                    .iter()
                    .enumerate()
                    .all(|(offset, &canonical)| canonical == start + offset))
            .then_some(start / block_rows)
        })
        .collect()
}

fn gather_blocks(tensor: &Tensor, blocks: &[usize], block_rows: usize) -> Result<Tensor> {
    let rows = blocks
        .iter()
        .flat_map(|&block| block * block_rows..(block + 1) * block_rows)
        .collect::<Vec<_>>();
    select_rows(tensor, &rows)
}

impl PackableLinear for Fp8Part {
    type Spec = Fp8LinearSpec;

    fn load_shard(spec: Fp8LinearSpec, site: &PackSite<'_>) -> Result<Option<Self>> {
        if !site.vb.contains_tensor("weight") {
            return Ok(None);
        }
        let Some(scale_name) = checkpoint_fp8_weight_scale_name(site.vb, spec)? else {
            return Ok(None);
        };
        if let Fp8ActivationMode::DynamicBlock(_) = spec.activation {
            if site.vb.device().is_metal()
                || !matches!(spec.weight_scale, Fp8WeightScaleLayout::Block(_))
            {
                return Ok(None);
            }
        }
        let CheckpointFp8Tensors {
            weight,
            weight_scale,
            activation_scale,
        } = load_checkpoint_fp8_tensors(
            [site.out_dim, site.in_dim],
            spec,
            scale_name,
            site.shard,
            site.vb,
        )?;
        if let Fp8WeightScaleLayout::Block([block_rows, _]) = spec.weight_scale {
            let rows = weight.dim(0)?;
            if !rows.is_multiple_of(block_rows) {
                tracing::debug!(
                    projection = site.vb.prefix(),
                    rows,
                    block_rows,
                    "Skipping FP8 projection packing because an output boundary is not block aligned"
                );
                return Ok(None);
            }
        }
        Ok(Some(Self {
            weight,
            scale: weight_scale,
            layout: spec.weight_scale,
            activation: spec.activation,
            activation_scale,
            dequant_dtype: checkpoint_fp8_dequant_dtype(site.vb.dtype()),
        }))
    }

    fn permute_output_rows(self, runtime_to_canonical: &[usize]) -> Result<Option<Self>> {
        let (weight, scale) = match self.layout {
            Fp8WeightScaleLayout::Tensor => {
                (select_rows(&self.weight, runtime_to_canonical)?, self.scale)
            }
            Fp8WeightScaleLayout::Channel => (
                select_rows(&self.weight, runtime_to_canonical)?,
                select_rows(&self.scale, runtime_to_canonical)?,
            ),
            Fp8WeightScaleLayout::Block([block_rows, _]) => {
                let Some(blocks) = block_permutation(runtime_to_canonical, block_rows) else {
                    tracing::debug!(
                        block_rows,
                        "Skipping FP8 projection packing because its output layout splits scale blocks"
                    );
                    return Ok(None);
                };
                (
                    gather_blocks(&self.weight, &blocks, block_rows)?,
                    gather_blocks(&self.scale, &blocks, 1)?,
                )
            }
        };
        Ok(Some(Self {
            weight,
            scale,
            ..self
        }))
    }

    fn merge(parts: Vec<Self>) -> Result<Option<PackedLinear>> {
        let Some(first) = parts.first() else {
            return Ok(None);
        };
        let weights = parts.iter().map(|part| &part.weight).collect::<Vec<_>>();
        if !same_input_and_device(&weights)?
            || parts.iter().any(|part| {
                part.activation != first.activation || part.dequant_dtype != first.dequant_dtype
            })
        {
            return Ok(None);
        }
        let activation_scale = match &first.activation_scale {
            Some(scale) => {
                let value = scale.to_scalar::<f32>()?.to_bits();
                for part in &parts {
                    let Some(other) = &part.activation_scale else {
                        return Ok(None);
                    };
                    // One static input scale quantizes the shared activation; differing scales cannot merge exactly.
                    if other.to_scalar::<f32>()?.to_bits() != value {
                        return Ok(None);
                    }
                }
                Some(scale.clone())
            }
            None => None,
        };
        let Some((layout, scale)) = merged_fp8_scales(&parts)? else {
            return Ok(None);
        };
        let rows_per_rank = parts
            .iter()
            .map(|part| part.weight.dim(0))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?;
        let (activation, dequant_dtype) = (first.activation, first.dequant_dtype);
        if let (Fp8ActivationMode::DynamicBlock(_), Fp8WeightScaleLayout::Block(block)) =
            (activation, layout)
        {
            return merge_blockwise_fp8(weight, scale, block, dequant_dtype, rows_per_rank)
                .map(Some);
        }
        PerTensorFP8Linear::merge_packed(
            PerTensorFp8Parts {
                weight,
                weight_scale_inv: scale,
                weight_scale_layout: layout,
                activation_mode: activation,
                activation_scale,
                bias: None,
                dequant_dtype,
            },
            rows_per_rank,
        )
        .map(Some)
    }
}

/// Concatenated scales in a layout every part can express exactly; tensor scales widen to per-row.
fn merged_fp8_scales(parts: &[Fp8Part]) -> Result<Option<(Fp8WeightScaleLayout, Tensor)>> {
    let first = &parts[0];
    if let Fp8WeightScaleLayout::Block(_) = first.layout {
        if parts.iter().any(|part| part.layout != first.layout) {
            return Ok(None);
        }
        let scales = parts.iter().map(|part| &part.scale).collect::<Vec<_>>();
        return Ok(Some((first.layout, Tensor::cat(&scales, 0)?)));
    }
    if parts
        .iter()
        .any(|part| matches!(part.layout, Fp8WeightScaleLayout::Block(_)))
    {
        return Ok(None);
    }
    if parts
        .iter()
        .all(|part| part.layout == Fp8WeightScaleLayout::Tensor)
    {
        let value = first.scale.to_scalar::<f32>()?.to_bits();
        let mut uniform = true;
        for part in parts {
            uniform &= part.scale.to_scalar::<f32>()?.to_bits() == value;
        }
        if uniform {
            return Ok(Some((Fp8WeightScaleLayout::Tensor, first.scale.clone())));
        }
    }
    let scales = parts
        .iter()
        .map(|part| match part.layout {
            Fp8WeightScaleLayout::Tensor => part
                .scale
                .reshape(1)?
                .broadcast_as(part.weight.dim(0)?)?
                .contiguous(),
            _ => Ok(part.scale.clone()),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some((
        Fp8WeightScaleLayout::Channel,
        Tensor::cat(&scales, 0)?,
    )))
}

fn merge_blockwise_fp8(
    weight: Tensor,
    scale: Tensor,
    block: [usize; 2],
    dequant_dtype: DType,
    rows_per_rank: Vec<usize>,
) -> Result<PackedLinear> {
    let make_layer = |weight: Tensor, weight_scale_inv: Tensor| -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(BlockwiseFP8Linear::new(
            QuantMethodConfig::BlockwiseFP8 {
                weight,
                weight_scale_inv,
                bias: None,
                dequant_dtype,
                weight_block_size: block.to_vec(),
                activation_scheme: Some(crate::Fp8ActivationScheme::Dynamic),
            },
        )?))
    };
    let mut constituents = Vec::with_capacity(rows_per_rank.len());
    let mut offset = 0;
    for &rows in &rows_per_rank {
        constituents.push(make_layer(
            weight.narrow(0, offset, rows)?,
            scale.narrow(0, offset / block[0], rows / block[0])?,
        )?);
        offset += rows;
    }
    Ok(PackedLinear {
        packed: make_layer(weight, scale)?,
        constituents,
        rows_per_rank,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{DType, Device, Tensor};

    use super::{pack_dense_weights_aligned, select_rows, RuntimeOutputLinear};
    use crate::{
        IsqType, LoraLinearSpec, PackedOutputLayout, QuantizedConfig, ReplicatedLayer, Shard,
        ShardedSafeTensors,
    };

    const IN_DIM: usize = 32;
    const TOKENS: usize = 3;
    const ALIGNMENT: usize = 16;
    const CPU_TOLERANCE: f32 = 1e-5;
    #[cfg(feature = "cuda")]
    const CUDA_TOLERANCE: f32 = 2e-2;

    struct PackCase<'a> {
        config: Option<QuantizedConfig>,
        tensors: HashMap<String, Tensor>,
        names: &'a [&'a str],
        out_dims: &'a [usize],
        layouts: &'a [PackedOutputLayout],
        in_dim: usize,
        dtype: DType,
        device: &'a Device,
    }

    fn config(value: serde_json::Value) -> candle_core::Result<Option<QuantizedConfig>> {
        serde_json::from_value(value)
            .map(Some)
            .map_err(candle_core::Error::msg)
    }

    fn values(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) * 0.25)
            .collect()
    }

    fn fp8(rows: usize, seed: usize) -> candle_core::Result<Tensor> {
        Tensor::from_vec(values(rows, IN_DIM, seed), (rows, IN_DIM), &Device::Cpu)?
            .to_dtype(DType::F8E4M3)
    }

    fn input(device: &Device, dtype: DType, in_dim: usize) -> candle_core::Result<Tensor> {
        Tensor::from_vec(
            (0..TOKENS * in_dim)
                .map(|i| ((i as f32) * 0.37).sin())
                .collect::<Vec<_>>(),
            (TOKENS, in_dim),
            device,
        )?
        .to_dtype(dtype)
    }

    fn max_rel_err(actual: &Tensor, expected: &Tensor) -> candle_core::Result<f32> {
        let actual = actual.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let expected = expected.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let err = (actual - &expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = expected.abs()?.max_all()?.to_scalar::<f32>()?.max(1.0);
        Ok(err / scale)
    }

    fn load_group(case: &PackCase<'_>) -> candle_core::Result<Option<crate::PackedLinear>> {
        let vb = ShardedSafeTensors::wrap(case.tensors.clone(), case.dtype, case.device.clone())
            .pp("model");
        let specs = case
            .out_dims
            .iter()
            .map(|&rows| LoraLinearSpec::replicated(case.in_dim, rows))
            .collect::<Vec<_>>();
        ReplicatedLayer::new_packed_with_output_layouts(
            &specs,
            case.names,
            case.layouts,
            &case.config,
            false,
            vb,
        )
    }

    // Packed output and each constituent must equal the unpacked loader's output in runtime row order.
    fn assert_packed_matches_unpacked(
        case: PackCase<'_>,
        tolerance: f32,
    ) -> candle_core::Result<()> {
        let group = load_group(&case)?.expect("projections should pack");
        let vb = ShardedSafeTensors::wrap(case.tensors.clone(), case.dtype, case.device.clone())
            .pp("model");
        let x = input(case.device, case.dtype, case.in_dim)?;
        let packed = group.packed.forward(&x)?;
        assert_eq!(
            packed.dims(),
            &[TOKENS, case.out_dims.iter().sum::<usize>()]
        );
        let mut offset = 0;
        for (index, (&name, &rows)) in case.names.iter().zip(case.out_dims).enumerate() {
            let unpacked =
                ReplicatedLayer::new(case.in_dim, rows, &case.config, false, vb.pp(name))?
                    .forward(&x)?;
            let runtime = select_rows(
                &unpacked.t()?.contiguous()?,
                case.layouts[index].runtime_to_canonical(),
            )?
            .t()?;
            let err = max_rel_err(&packed.narrow(1, offset, rows)?, &runtime)?;
            assert!(err <= tolerance, "{name}: packed rel err {err}");
            let err = max_rel_err(&group.constituents[index].forward(&x)?, &runtime)?;
            assert!(err <= tolerance, "{name}: constituent rel err {err}");
            offset += rows;
        }
        Ok(())
    }

    fn swapped_halves(rows: usize) -> PackedOutputLayout {
        let half = rows / 2;
        PackedOutputLayout::from_runtime_to_canonical((half..rows).chain(0..half).collect())
            .unwrap()
    }

    fn reversed(rows: usize) -> PackedOutputLayout {
        PackedOutputLayout::from_runtime_to_canonical((0..rows).rev().collect()).unwrap()
    }

    fn insert_part(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        parts: impl IntoIterator<Item = (&'static str, Tensor)>,
    ) {
        for (suffix, tensor) in parts {
            tensors.insert(format!("model.{name}.{suffix}"), tensor);
        }
    }

    fn per_tensor_fp8_tensors(
        out_dims: &[usize],
        names: &[&str],
        input_scales: &[f32],
    ) -> candle_core::Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for (index, (&name, &rows)) in names.iter().zip(out_dims).enumerate() {
            insert_part(
                &mut tensors,
                name,
                [
                    ("weight", fp8(rows, index)?),
                    (
                        "weight_scale",
                        Tensor::new(0.01f32 * (index + 1) as f32, &Device::Cpu)?,
                    ),
                    (
                        "input_scale",
                        Tensor::new(input_scales[index], &Device::Cpu)?,
                    ),
                ],
            );
        }
        Ok(tensors)
    }

    fn check_modelopt_per_tensor_fp8(
        device: &Device,
        dtype: DType,
        tol: f32,
    ) -> candle_core::Result<()> {
        let names = ["q_proj", "k_proj", "v_proj"];
        let out_dims = [16, 8, 8];
        let layouts = out_dims.map(PackedOutputLayout::identity);
        let modelopt = serde_json::json!({"quant_method": "modelopt", "quant_algo": "FP8"});
        assert_packed_matches_unpacked(
            PackCase {
                config: config(modelopt.clone())?,
                tensors: per_tensor_fp8_tensors(&out_dims, &names, &[0.05; 3])?,
                names: &names,
                out_dims: &out_dims,
                layouts: &layouts,
                in_dim: IN_DIM,
                dtype,
                device,
            },
            tol,
        )?;
        let layouts = [reversed(16), layouts[1].clone(), layouts[2].clone()];
        assert_packed_matches_unpacked(
            PackCase {
                config: config(modelopt.clone())?,
                tensors: per_tensor_fp8_tensors(&out_dims, &names, &[0.05; 3])?,
                names: &names,
                out_dims: &out_dims,
                layouts: &layouts,
                in_dim: IN_DIM,
                dtype,
                device,
            },
            tol,
        )?;
        assert!(load_group(&PackCase {
            config: config(modelopt)?,
            tensors: per_tensor_fp8_tensors(&out_dims, &names, &[0.05, 0.06, 0.05])?,
            names: &names,
            out_dims: &out_dims,
            layouts: &layouts,
            in_dim: IN_DIM,
            dtype,
            device,
        })?
        .is_none());
        Ok(())
    }

    fn check_compressed_tensors_channel_fp8(
        device: &Device,
        dtype: DType,
        tol: f32,
    ) -> candle_core::Result<()> {
        let names = ["gate_proj", "up_proj"];
        let out_dims = [16, 16];
        let mut tensors = HashMap::new();
        for (index, (&name, &rows)) in names.iter().zip(&out_dims).enumerate() {
            insert_part(
                &mut tensors,
                name,
                [
                    ("weight", fp8(rows, index + 5)?),
                    (
                        "weight_scale",
                        Tensor::from_vec(
                            (0..rows)
                                .map(|row| 0.001 * (row + index + 1) as f32)
                                .collect::<Vec<_>>(),
                            (rows, 1),
                            &Device::Cpu,
                        )?,
                    ),
                ],
            );
        }
        assert_packed_matches_unpacked(
            PackCase {
                config: config(serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "float-quantized",
                    "config_groups": {"group_0": {
                        "targets": ["Linear"],
                        "weights": {"dynamic": false, "num_bits": 8, "strategy": "channel",
                            "symmetric": true, "type": "float"},
                        "input_activations": {"dynamic": true, "num_bits": 8,
                            "strategy": "token", "symmetric": true, "type": "float"}
                    }}
                }))?,
                tensors,
                names: &names,
                out_dims: &out_dims,
                layouts: &[reversed(16), PackedOutputLayout::identity(16)],
                in_dim: IN_DIM,
                dtype,
                device,
            },
            tol,
        )
    }

    const BLOCK: [usize; 2] = [4, 16];

    fn blockwise_fp8_tensors(
        out_dims: &[usize],
        names: &[&str],
    ) -> candle_core::Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for (index, (&name, &rows)) in names.iter().zip(out_dims).enumerate() {
            let scale_shape = (rows / BLOCK[0], IN_DIM / BLOCK[1]);
            insert_part(
                &mut tensors,
                name,
                [
                    ("weight", fp8(rows, index + 9)?),
                    (
                        "weight_scale_inv",
                        Tensor::from_vec(
                            (0..scale_shape.0 * scale_shape.1)
                                .map(|i| 0.002 * (i + index + 1) as f32)
                                .collect::<Vec<_>>(),
                            scale_shape,
                            &Device::Cpu,
                        )?,
                    ),
                ],
            );
        }
        Ok(tensors)
    }

    fn check_blockwise_fp8(device: &Device, dtype: DType, tol: f32) -> candle_core::Result<()> {
        let names = ["q_proj", "k_proj"];
        let out_dims = [16, 8];
        let native = serde_json::json!({
            "quant_method": "fp8", "weight_block_size": BLOCK, "activation_scheme": "dynamic"
        });
        for layouts in [
            [
                PackedOutputLayout::identity(16),
                PackedOutputLayout::identity(8),
            ],
            [swapped_halves(16), swapped_halves(8)],
        ] {
            assert_packed_matches_unpacked(
                PackCase {
                    config: config(native.clone())?,
                    tensors: blockwise_fp8_tensors(&out_dims, &names)?,
                    names: &names,
                    out_dims: &out_dims,
                    layouts: &layouts,
                    in_dim: IN_DIM,
                    dtype,
                    device,
                },
                tol,
            )?;
        }
        assert!(load_group(&PackCase {
            config: config(native)?,
            tensors: blockwise_fp8_tensors(&out_dims, &names)?,
            names: &names,
            out_dims: &out_dims,
            layouts: &[reversed(16), PackedOutputLayout::identity(8)],
            in_dim: IN_DIM,
            dtype,
            device,
        })?
        .is_none());
        Ok(())
    }

    fn check_mxfp8(device: &Device, dtype: DType, tol: f32) -> candle_core::Result<()> {
        let names = ["q_proj", "k_proj"];
        let out_dims = [16, 8];
        let mut tensors = HashMap::new();
        let mut layers = serde_json::Map::new();
        for (index, (&name, &rows)) in names.iter().zip(&out_dims).enumerate() {
            insert_part(
                &mut tensors,
                name,
                [
                    ("weight", fp8(rows, index + 3)?),
                    (
                        "weight_scale",
                        Tensor::from_vec(
                            (0..rows)
                                .map(|i| 125 + ((i + index) % 4) as u8)
                                .collect::<Vec<_>>(),
                            (rows, 1),
                            &Device::Cpu,
                        )?,
                    ),
                ],
            );
            layers.insert(
                format!("model.{name}"),
                serde_json::json!({"quant_algo": "MXFP8", "group_size": 32}),
            );
        }
        assert_packed_matches_unpacked(
            PackCase {
                config: config(serde_json::json!({
                    "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": layers
                }))?,
                tensors,
                names: &names,
                out_dims: &out_dims,
                layouts: &[reversed(16), PackedOutputLayout::identity(8)],
                in_dim: IN_DIM,
                dtype,
                device,
            },
            tol,
        )
    }

    #[test]
    fn packed_fp8_and_mxfp8_match_unpacked_cpu() -> candle_core::Result<()> {
        let device = Device::Cpu;
        check_modelopt_per_tensor_fp8(&device, DType::F32, CPU_TOLERANCE)?;
        check_compressed_tensors_channel_fp8(&device, DType::F32, CPU_TOLERANCE)?;
        check_blockwise_fp8(&device, DType::F32, CPU_TOLERANCE)?;
        check_mxfp8(&device, DType::F32, CPU_TOLERANCE)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn packed_fp8_and_mxfp8_match_unpacked_cuda() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        check_modelopt_per_tensor_fp8(&device, DType::BF16, CUDA_TOLERANCE)?;
        check_compressed_tensors_channel_fp8(&device, DType::BF16, CUDA_TOLERANCE)?;
        check_blockwise_fp8(&device, DType::BF16, CUDA_TOLERANCE)?;
        check_mxfp8(&device, DType::BF16, CUDA_TOLERANCE)
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    #[test]
    fn packed_nvfp4_permuted_layout_matches_unpacked_cuda() -> candle_core::Result<()> {
        const K: usize = 256;
        const NVFP4_BLOCK: usize = 16;
        let device = Device::new_cuda(0)?;
        if let Device::Cuda(cuda) = &device {
            if !crate::cutile::nvfp4_supported(cuda) {
                return Ok(());
            }
        }
        let names = ["q_proj", "k_proj"];
        let out_dims = [128, 64];
        let mut tensors = HashMap::new();
        for (index, (&name, &rows)) in names.iter().zip(&out_dims).enumerate() {
            let weight = (0..rows * K / 2)
                .map(|i| ((i * 37 + index * 11) % 256) as u8)
                .collect::<Vec<_>>();
            let scales = (0..rows * K / NVFP4_BLOCK)
                .map(|i| 0.25 * (1 + (i + index) % 4) as f32)
                .collect::<Vec<_>>();
            insert_part(
                &mut tensors,
                name,
                [
                    (
                        "weight",
                        Tensor::from_vec(weight, (rows, K / 2), &Device::Cpu)?,
                    ),
                    (
                        "weight_scale",
                        Tensor::from_vec(scales, (rows, K / NVFP4_BLOCK), &Device::Cpu)?
                            .to_dtype(DType::F8E4M3)?,
                    ),
                    (
                        "weight_scale_2",
                        Tensor::new(0.5f32 * (index + 1) as f32, &Device::Cpu)?,
                    ),
                ],
            );
        }
        let case = PackCase {
            config: config(serde_json::json!({
                "quant_method": "modelopt", "quant_algo": "W4A16_NVFP4", "group_size": NVFP4_BLOCK
            }))?,
            tensors,
            names: &names,
            out_dims: &out_dims,
            layouts: &[swapped_halves(128), reversed(64)],
            in_dim: K,
            dtype: DType::BF16,
            device: &device,
        };
        assert_packed_matches_unpacked(case, CUDA_TOLERANCE)
    }

    #[test]
    fn packed_dense_padding_exposes_logical_rows_and_layouts() -> candle_core::Result<()> {
        const COLUMNS: usize = 4;
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32, 2., -1., 0.5], [2., -1., 0.5, 1.]], &device)?;
        for dimensions in [[5, 3], [8, 8]] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for alignment in [1, ALIGNMENT] {
                    let first = Tensor::from_vec(
                        values(dimensions[0], COLUMNS, 1),
                        (dimensions[0], COLUMNS),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let second = Tensor::full(0.25f32, (dimensions[1], COLUMNS), &device)?
                        .to_dtype(dtype)?;
                    let permutation: Arc<[usize]> =
                        (0..dimensions[0]).rev().collect::<Vec<_>>().into();
                    let runtime_first = select_rows(&first, &permutation)?;
                    let mut group = pack_dense_weights_aligned(
                        vec![runtime_first.clone(), second.clone()],
                        alignment,
                    )?;
                    group.constituents[0] =
                        RuntimeOutputLinear::wrap(group.constituents[0].clone(), Some(permutation));
                    let rows = dimensions.iter().sum::<usize>();
                    let padded_rows = rows.next_multiple_of(alignment);
                    assert_eq!(group.rows_per_rank, dimensions);
                    let expected = Tensor::cat(&[&runtime_first, &second], 0)?;
                    assert_eq!(
                        group
                            .packed
                            .dequantize_w()?
                            .to_dtype(DType::F32)?
                            .to_vec2::<f32>()?,
                        expected.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    );
                    for (projection, canonical) in group.constituents.iter().zip([&first, &second])
                    {
                        assert_eq!(
                            projection
                                .dequantize_w()?
                                .to_dtype(DType::F32)?
                                .to_vec2::<f32>()?,
                            canonical.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                        );
                    }
                    if dtype == DType::F32 {
                        let output = group.packed.forward(&x)?;
                        assert_eq!(output.dims(), &[2, rows]);
                        assert_eq!(output.stride(), &[padded_rows, 1]);
                        let mut offset = 0;
                        for (projection, &width) in
                            group.constituents.iter().zip(&group.rows_per_rank)
                        {
                            assert_eq!(
                                output.narrow(1, offset, width)?.to_vec2::<f32>()?,
                                projection.forward(&x)?.to_vec2::<f32>()?
                            );
                            offset += width;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn padded_dense_constituents_roundtrip_uqff_without_padding() -> candle_core::Result<()> {
        const COLUMNS: usize = 4;
        const ROWS: [usize; 2] = [320, 4];
        const PREFIXES: [&str; 2] = ["hc.input_mix_weight_down", "hc.block_inject_weight"];
        let device = Device::Cpu;
        let weights = ROWS
            .iter()
            .enumerate()
            .map(|(projection, &rows)| {
                Tensor::from_vec(values(rows, COLUMNS, projection), (rows, COLUMNS), &device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let group = pack_dense_weights_aligned(weights.clone(), ALIGNMENT)?;
        let packed_weight = group.packed.dequantize_w()?;
        assert_eq!(packed_weight.dims(), &[ROWS.iter().sum::<usize>(), COLUMNS]);
        let mut tensors = crate::uqff_version_tensors();
        for (projection, prefix) in group.constituents.iter().zip(PREFIXES) {
            let weight = projection.dequantize_w()?;
            {
                let (packed_storage, _) = packed_weight.storage_and_layout();
                let (storage, _) = weight.storage_and_layout();
                assert!(std::ptr::eq(&*packed_storage, &*storage));
            }
            tensors.extend(projection.serialize_uqff(prefix, IsqType::Q4K)?);
        }
        let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let path = dir.path().join("padded-constituents.uqff");
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .map_err(candle_core::Error::wrap)?;
        let reader = crate::UqffReader::open(&[path])?;
        for (prefix, expected) in PREFIXES.into_iter().zip(&weights) {
            assert_eq!(
                reader.tensor_dims(&format!("{prefix}.weight"))?,
                expected.dims()
            );
            let projection = reader
                .load_linear(prefix, &device, Shard::default())?
                .expect("serialized constituent");
            assert_eq!(
                projection.dequantize_w()?.to_vec2::<f32>()?,
                expected.to_vec2::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn packed_dense_bytes_matches_padded_storage() {
        assert_eq!(
            super::packed_dense_bytes(20, 8, DType::BF16, true),
            32 * 8 * 2
        );
        assert_eq!(
            super::packed_dense_bytes(20, 8, DType::BF16, false),
            20 * 8 * 2
        );
        assert_eq!(
            super::packed_dense_bytes(20, 8, DType::F32, true),
            20 * 8 * 4
        );
    }
}
