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
