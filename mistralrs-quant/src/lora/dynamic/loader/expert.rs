use std::collections::BTreeSet;

use candle_core::{DType, Device, Result, Tensor};

use crate::{LoraConfig, Shard, ShardedVarBuilder};

use super::AdapterTensorIndex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExpertProjection {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GateUpOrder {
    Concatenated,
    Interleaved,
}

#[derive(Clone, Copy)]
pub(super) struct ExpertSiteMeta<'a> {
    pub path: &'a str,
    pub num_experts: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_name: &'a str,
    pub up_name: &'a str,
    pub down_name: &'a str,
    pub gate_up_order: GateUpOrder,
    pub gate_up_output_shard: Shard,
    pub down_input_shard: Shard,
    pub activation_dtype: DType,
    pub device: &'a Device,
}

#[derive(Debug)]
pub(super) struct LoadedExpertProjection {
    pub a: Tensor,
    pub b: Tensor,
    pub scales: Tensor,
}

#[derive(Debug)]
pub(super) struct LoadedExpertSite {
    pub gate: Option<LoadedExpertProjection>,
    pub up: Option<LoadedExpertProjection>,
    pub down: Option<LoadedExpertProjection>,
}

#[derive(Clone)]
struct TensorPairPlan {
    a_name: String,
    b_name: String,
    rank: usize,
    scale: f32,
}

#[derive(Clone, Copy)]
enum FusedTensorLayout {
    Flat,
    Packed,
}

enum ExpertSiteSourcePlan {
    PerExpert {
        gate: Option<PerExpertProjectionPlan>,
        up: Option<PerExpertProjectionPlan>,
        down: Option<PerExpertProjectionPlan>,
    },
    Fused {
        gate_up: Option<(TensorPairPlan, FusedTensorLayout)>,
        down: Option<(TensorPairPlan, FusedTensorLayout)>,
    },
}

struct PerExpertProjectionPlan {
    projection: ExpertProjection,
    rank: usize,
    pairs: Vec<Option<TensorPairPlan>>,
}

pub(super) struct ExpertSiteLoadPlan {
    source: ExpertSiteSourcePlan,
    consumed: BTreeSet<String>,
    bytes: u64,
}

impl ExpertSiteLoadPlan {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn consumed(&self) -> &BTreeSet<String> {
        &self.consumed
    }
}

fn pair_for_paths(
    tensors: &AdapterTensorIndex,
    paths: &[String],
) -> Result<Option<(String, String, String)>> {
    let mut found = None;
    for path in paths {
        let Some((a_name, b_name)) = tensors.pair_for_site(path)? else {
            continue;
        };
        if found.is_some() {
            candle_core::bail!(
                "multiple routed MoE LoRA tensor pairs match `{}`",
                paths.join("` or `")
            );
        }
        found = Some((path.clone(), a_name.to_string(), b_name.to_string()));
    }
    Ok(found)
}

fn source_shape<'a>(weights: &'a ShardedVarBuilder, name: &str) -> Result<&'a [usize]> {
    weights.tensor_shape(name).ok_or_else(|| {
        candle_core::Error::msg(format!(
            "routed MoE LoRA tensor `{name}` has no shape metadata"
        ))
    })
}

fn check_shape(shape: &[usize], expected: &[usize], name: &str) -> Result<()> {
    if shape != expected {
        candle_core::bail!(
            "routed MoE LoRA tensor `{name}` has shape {shape:?}, expected {expected:?}"
        );
    }
    Ok(())
}

fn scale_f32(config: &LoraConfig, path: &str) -> Result<f32> {
    let scale = config.scale_for(path)? as f32;
    if !scale.is_finite() {
        candle_core::bail!("LoRA scale for `{path}` cannot be represented as f32");
    }
    Ok(scale)
}

fn projection_name<'a>(meta: ExpertSiteMeta<'a>, projection: ExpertProjection) -> &'a str {
    match projection {
        ExpertProjection::Gate => meta.gate_name,
        ExpertProjection::Up => meta.up_name,
        ExpertProjection::Down => meta.down_name,
    }
}

fn projection_dims(meta: ExpertSiteMeta<'_>, projection: ExpertProjection) -> (usize, usize) {
    match projection {
        ExpertProjection::Gate | ExpertProjection::Up => (meta.hidden_size, meta.intermediate_size),
        ExpertProjection::Down => (meta.intermediate_size, meta.hidden_size),
    }
}

fn per_expert_projection_plan(
    meta: ExpertSiteMeta<'_>,
    projection: ExpertProjection,
    config: &LoraConfig,
    tensors: &AdapterTensorIndex,
    weights: &ShardedVarBuilder,
) -> Result<Option<PerExpertProjectionPlan>> {
    let name = projection_name(meta, projection);
    let (input, output) = projection_dims(meta, projection);
    let mut pairs = Vec::with_capacity(meta.num_experts);
    let mut max_rank = 0;
    for expert in 0..meta.num_experts {
        let path = format!("{}.{expert}.{name}", meta.path);
        let Some((a_name, b_name)) = tensors.pair_for_site(&path)? else {
            pairs.push(None);
            continue;
        };
        if !config.try_targets_path(&path)? {
            candle_core::bail!(
                "routed MoE LoRA tensors for site `{path}` are not declared by target_modules"
            );
        }
        if config.try_excludes_path(&path)? {
            candle_core::bail!(
                "routed MoE LoRA tensors for site `{path}` are excluded by exclude_modules"
            );
        }
        let rank = config.try_rank_for(&path)?;
        if rank == 0 {
            candle_core::bail!("LoRA rank for `{path}` must be nonzero");
        }
        check_shape(source_shape(weights, a_name)?, &[rank, input], a_name)?;
        check_shape(source_shape(weights, b_name)?, &[output, rank], b_name)?;
        max_rank = max_rank.max(rank);
        pairs.push(Some(TensorPairPlan {
            a_name: a_name.to_string(),
            b_name: b_name.to_string(),
            rank,
            scale: scale_f32(config, &path)?,
        }));
    }
    Ok((max_rank != 0).then_some(PerExpertProjectionPlan {
        projection,
        rank: max_rank,
        pairs,
    }))
}

struct FusedPairSpec<'a> {
    projection_path: &'a str,
    tensor_paths: &'a [String],
    input: usize,
    output: usize,
}

fn fused_pair_plan(
    meta: ExpertSiteMeta<'_>,
    config: &LoraConfig,
    tensors: &AdapterTensorIndex,
    weights: &ShardedVarBuilder,
    spec: FusedPairSpec<'_>,
) -> Result<Option<(TensorPairPlan, FusedTensorLayout)>> {
    let Some((_tensor_path, a_name, b_name)) = pair_for_paths(tensors, spec.tensor_paths)? else {
        return Ok(None);
    };
    if !config.targets_parameter(spec.projection_path) {
        candle_core::bail!(
            "routed MoE LoRA tensors for parameter `{}` are not declared by target_parameters",
            spec.projection_path
        );
    }
    if config.try_excludes_path(spec.projection_path)? {
        candle_core::bail!(
            "routed MoE LoRA tensors for parameter `{}` are excluded by exclude_modules",
            spec.projection_path
        );
    }
    let rank = config.try_rank_for(spec.projection_path)?;
    if rank == 0 {
        candle_core::bail!("LoRA rank for `{}` must be nonzero", spec.projection_path);
    }
    let expert_rank = meta
        .num_experts
        .checked_mul(rank)
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA shape overflow"))?;
    let a_shape = source_shape(weights, &a_name)?;
    let b_shape = source_shape(weights, &b_name)?;
    let layout = match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            check_shape(a_shape, &[expert_rank, spec.input], &a_name)?;
            check_shape(b_shape, &[spec.output, expert_rank], &b_name)?;
            FusedTensorLayout::Flat
        }
        (3, 3) => {
            check_shape(a_shape, &[meta.num_experts, rank, spec.input], &a_name)?;
            check_shape(b_shape, &[meta.num_experts, spec.output, rank], &b_name)?;
            FusedTensorLayout::Packed
        }
        _ => candle_core::bail!(
            "routed MoE LoRA tensor pair `{}` must both be flat rank-2 or packed rank-3 tensors",
            spec.projection_path
        ),
    };
    Ok(Some((
        TensorPairPlan {
            a_name,
            b_name,
            rank,
            scale: scale_f32(config, spec.projection_path)?,
        },
        layout,
    )))
}

fn shard_len(size: usize, shard: Shard, name: &str) -> Result<usize> {
    Ok(shard_bounds(size, shard, name)?.1)
}

fn shard_bounds(size: usize, shard: Shard, name: &str) -> Result<(usize, usize)> {
    match shard {
        Shard::Simple { world_size: 1, .. } => Ok((0, size)),
        Shard::Simple {
            rank, world_size, ..
        } => {
            if rank >= world_size || !size.is_multiple_of(world_size) {
                candle_core::bail!("invalid routed MoE LoRA shard for `{name}`");
            }
            let len = size / world_size;
            Ok((rank * len, len))
        }
        Shard::Offset { offset, len, .. } => {
            if offset.checked_add(len).is_none_or(|end| end > size) {
                candle_core::bail!("invalid routed MoE LoRA shard for `{name}`");
            }
            Ok((offset, len))
        }
    }
}

fn checked_elements(shape: &[usize]) -> Result<u64> {
    shape.iter().try_fold(1u64, |elements, dim| {
        elements
            .checked_mul(u64::try_from(*dim).map_err(candle_core::Error::wrap)?)
            .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))
    })
}

fn packed_projection_bytes(
    meta: ExpertSiteMeta<'_>,
    projection: ExpertProjection,
    rank: usize,
) -> Result<u64> {
    let (input, output) = projection_dims(meta, projection);
    let (input, output) = match projection {
        ExpertProjection::Gate | ExpertProjection::Up => (
            input,
            shard_len(output, meta.gate_up_output_shard, meta.path)?,
        ),
        ExpertProjection::Down => (shard_len(input, meta.down_input_shard, meta.path)?, output),
    };
    let weight_elements = checked_elements(&[meta.num_experts, rank, input])?
        .checked_add(checked_elements(&[meta.num_experts, output, rank])?)
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?;
    let weight_bytes = weight_elements
        .checked_mul(
            u64::try_from(meta.activation_dtype.size_in_bytes())
                .map_err(candle_core::Error::wrap)?,
        )
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?;
    weight_bytes
        .checked_add(
            u64::try_from(meta.num_experts)
                .map_err(candle_core::Error::wrap)?
                .checked_mul(
                    u64::try_from(DType::F32.size_in_bytes()).map_err(candle_core::Error::wrap)?,
                )
                .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?,
        )
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))
}

fn packed_gate_up_bytes(meta: ExpertSiteMeta<'_>, rank: usize) -> Result<u64> {
    let local_intermediate =
        shard_len(meta.intermediate_size, meta.gate_up_output_shard, meta.path)?;
    let weight_elements = checked_elements(&[meta.num_experts, rank, meta.hidden_size])?
        .checked_add(
            checked_elements(&[meta.num_experts, local_intermediate, rank])?
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?,
        )
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?;
    let weight_bytes = weight_elements
        .checked_mul(
            u64::try_from(meta.activation_dtype.size_in_bytes())
                .map_err(candle_core::Error::wrap)?,
        )
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?;
    weight_bytes
        .checked_add(
            u64::try_from(meta.num_experts)
                .map_err(candle_core::Error::wrap)?
                .checked_mul(
                    u64::try_from(DType::F32.size_in_bytes()).map_err(candle_core::Error::wrap)?,
                )
                .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))?,
        )
        .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA tensor size overflow"))
}

pub(super) fn plan_expert_site(
    meta: ExpertSiteMeta<'_>,
    config: &LoraConfig,
    tensors: &AdapterTensorIndex,
    weights: &ShardedVarBuilder,
) -> Result<Option<ExpertSiteLoadPlan>> {
    let gate_up_path = format!("{}.gate_up_proj", meta.path);
    let down_path = format!("{}.down_proj", meta.path);
    let gate_up = fused_pair_plan(
        meta,
        config,
        tensors,
        weights,
        FusedPairSpec {
            projection_path: &gate_up_path,
            tensor_paths: &[format!("{}.base_layer", meta.path), gate_up_path.clone()],
            input: meta.hidden_size,
            output: meta
                .intermediate_size
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::msg("routed MoE LoRA shape overflow"))?,
        },
    )?;
    let fused_down = fused_pair_plan(
        meta,
        config,
        tensors,
        weights,
        FusedPairSpec {
            projection_path: &down_path,
            tensor_paths: &[meta.path.to_string(), down_path.clone()],
            input: meta.intermediate_size,
            output: meta.hidden_size,
        },
    )?;
    let has_fused = gate_up.is_some() || fused_down.is_some();
    let source = if has_fused {
        ExpertSiteSourcePlan::Fused {
            gate_up,
            down: fused_down,
        }
    } else {
        let gate =
            per_expert_projection_plan(meta, ExpertProjection::Gate, config, tensors, weights)?;
        let up = per_expert_projection_plan(meta, ExpertProjection::Up, config, tensors, weights)?;
        let down =
            per_expert_projection_plan(meta, ExpertProjection::Down, config, tensors, weights)?;
        if gate.is_none() && up.is_none() && down.is_none() {
            return Ok(None);
        }
        ExpertSiteSourcePlan::PerExpert { gate, up, down }
    };
    let mut consumed = BTreeSet::new();
    let mut bytes = 0u64;
    match &source {
        ExpertSiteSourcePlan::PerExpert { gate, up, down } => {
            for plan in [gate, up, down].into_iter().flatten() {
                bytes = bytes
                    .checked_add(packed_projection_bytes(meta, plan.projection, plan.rank)?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("routed MoE LoRA tensor size overflow")
                    })?;
                for pair in plan.pairs.iter().flatten() {
                    consumed.insert(pair.a_name.clone());
                    consumed.insert(pair.b_name.clone());
                }
            }
        }
        ExpertSiteSourcePlan::Fused { gate_up, down } => {
            if let Some((pair, _)) = gate_up {
                bytes = bytes
                    .checked_add(packed_gate_up_bytes(meta, pair.rank)?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("routed MoE LoRA tensor size overflow")
                    })?;
                consumed.insert(pair.a_name.clone());
                consumed.insert(pair.b_name.clone());
            }
            if let Some((pair, _)) = down {
                bytes = bytes
                    .checked_add(packed_projection_bytes(
                        meta,
                        ExpertProjection::Down,
                        pair.rank,
                    )?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("routed MoE LoRA tensor size overflow")
                    })?;
                consumed.insert(pair.a_name.clone());
                consumed.insert(pair.b_name.clone());
            }
        }
    }
    Ok(Some(ExpertSiteLoadPlan {
        source,
        consumed,
        bytes,
    }))
}

fn shard_on(shard: Shard, dim: usize) -> Shard {
    match shard {
        Shard::Simple {
            rank, world_size, ..
        } => Shard::Simple {
            dim,
            rank,
            world_size,
        },
        Shard::Offset { offset, len, .. } => Shard::Offset { dim, offset, len },
    }
}

fn pad_pair_rank(a: Tensor, b: Tensor, rank: usize) -> Result<(Tensor, Tensor)> {
    let source_rank = a.dim(0)?;
    if source_rank == rank {
        return Ok((a, b));
    }
    let (_, input) = a.dims2()?;
    let (output, b_rank) = b.dims2()?;
    if source_rank != b_rank || source_rank > rank {
        candle_core::bail!("invalid routed MoE LoRA rank while packing expert tensors");
    }
    let padding = rank - source_rank;
    let a_padding = Tensor::zeros((padding, input), a.dtype(), a.device())?;
    let b_padding = Tensor::zeros((output, padding), b.dtype(), b.device())?;
    Ok((
        Tensor::cat(&[&a, &a_padding], 0)?,
        Tensor::cat(&[&b, &b_padding], 1)?,
    ))
}

fn load_per_expert_projection(
    meta: ExpertSiteMeta<'_>,
    plan: PerExpertProjectionPlan,
    weights: &ShardedVarBuilder,
) -> Result<LoadedExpertProjection> {
    let (input, output) = projection_dims(meta, plan.projection);
    let (local_input, local_output, a_shard, b_shard) = match plan.projection {
        ExpertProjection::Gate | ExpertProjection::Up => (
            input,
            shard_len(output, meta.gate_up_output_shard, meta.path)?,
            Shard::default(),
            shard_on(meta.gate_up_output_shard, 0),
        ),
        ExpertProjection::Down => (
            shard_len(input, meta.down_input_shard, meta.path)?,
            output,
            shard_on(meta.down_input_shard, 1),
            Shard::default(),
        ),
    };
    let staging_device = if meta.device.is_cpu() {
        meta.device.clone()
    } else {
        Device::Cpu
    };
    let site_weights = weights
        .root()
        .set_device(staging_device.clone())
        .set_dtype(meta.activation_dtype);
    let mut a_experts = Vec::with_capacity(meta.num_experts);
    let mut b_experts = Vec::with_capacity(meta.num_experts);
    let mut scales = Vec::with_capacity(meta.num_experts);
    for pair in plan.pairs {
        let Some(pair) = pair else {
            a_experts.push(Tensor::zeros(
                (plan.rank, local_input),
                meta.activation_dtype,
                &staging_device,
            )?);
            b_experts.push(Tensor::zeros(
                (local_output, plan.rank),
                meta.activation_dtype,
                &staging_device,
            )?);
            scales.push(0.0);
            continue;
        };
        let a = site_weights.get_with_hints((pair.rank, input), &pair.a_name, a_shard)?;
        let b = site_weights.get_with_hints((output, pair.rank), &pair.b_name, b_shard)?;
        let (a, b) = pad_pair_rank(a, b, plan.rank)?;
        a_experts.push(a);
        b_experts.push(b);
        scales.push(pair.scale);
    }
    Ok(LoadedExpertProjection {
        a: Tensor::stack(&a_experts, 0)?.to_device(meta.device)?,
        b: Tensor::stack(&b_experts, 0)?.to_device(meta.device)?,
        scales: Tensor::from_vec(scales, meta.num_experts, meta.device)?,
    })
}

fn load_fused_a(
    meta: ExpertSiteMeta<'_>,
    pair: &TensorPairPlan,
    layout: FusedTensorLayout,
    full_input: usize,
    local_input: usize,
    shard: Shard,
    weights: &ShardedVarBuilder,
) -> Result<Tensor> {
    let site_weights = weights
        .root()
        .set_device(meta.device.clone())
        .set_dtype(meta.activation_dtype);
    match layout {
        FusedTensorLayout::Flat => site_weights
            .get_with_hints(
                (meta.num_experts * pair.rank, full_input),
                &pair.a_name,
                shard,
            )?
            .reshape((meta.num_experts, pair.rank, local_input)),
        FusedTensorLayout::Packed => site_weights.get_with_hints(
            (meta.num_experts, pair.rank, full_input),
            &pair.a_name,
            shard,
        ),
    }
}

fn load_fused_b(
    meta: ExpertSiteMeta<'_>,
    pair: &TensorPairPlan,
    layout: FusedTensorLayout,
    full_output: usize,
    local_output: usize,
    shard: Shard,
    weights: &ShardedVarBuilder,
) -> Result<Tensor> {
    match layout {
        FusedTensorLayout::Flat => {
            let staging_device = if meta.device.is_cpu() {
                meta.device.clone()
            } else {
                Device::Cpu
            };
            weights
                .root()
                .set_device(staging_device)
                .set_dtype(meta.activation_dtype)
                .get_with_hints(
                    (full_output, meta.num_experts * pair.rank),
                    &pair.b_name,
                    shard,
                )?
                .reshape((local_output, meta.num_experts, pair.rank))?
                .permute((1, 0, 2))?
                .contiguous()?
                .to_device(meta.device)
        }
        FusedTensorLayout::Packed => weights
            .root()
            .set_device(meta.device.clone())
            .set_dtype(meta.activation_dtype)
            .get_with_hints(
                (meta.num_experts, full_output, pair.rank),
                &pair.b_name,
                shard,
            ),
    }
}

fn projection_scales(meta: ExpertSiteMeta<'_>, scale: f32) -> Result<Tensor> {
    Tensor::from_vec(vec![scale; meta.num_experts], meta.num_experts, meta.device)
}

fn fused_a_input_dim(layout: FusedTensorLayout) -> usize {
    match layout {
        FusedTensorLayout::Flat => 1,
        FusedTensorLayout::Packed => 2,
    }
}

fn fused_b_output_dim(layout: FusedTensorLayout) -> usize {
    match layout {
        FusedTensorLayout::Flat => 0,
        FusedTensorLayout::Packed => 1,
    }
}

fn range_shard(dim: usize, offset: usize, len: usize, full: usize) -> Shard {
    if offset == 0 && len == full {
        Shard::default()
    } else {
        Shard::Offset { dim, offset, len }
    }
}

fn split_gate_up_b(
    meta: ExpertSiteMeta<'_>,
    b: &Tensor,
    intermediate_size: usize,
) -> Result<(Tensor, Tensor)> {
    match meta.gate_up_order {
        GateUpOrder::Concatenated => Ok((
            b.narrow(1, 0, intermediate_size)?,
            b.narrow(1, intermediate_size, intermediate_size)?,
        )),
        GateUpOrder::Interleaved => {
            let gate_indices = (0..intermediate_size)
                .map(|index| u32::try_from(index * 2).map_err(candle_core::Error::wrap))
                .collect::<Result<Vec<_>>>()?;
            let up_indices = (0..intermediate_size)
                .map(|index| u32::try_from(index * 2 + 1).map_err(candle_core::Error::wrap))
                .collect::<Result<Vec<_>>>()?;
            let gate_indices = Tensor::from_vec(gate_indices, intermediate_size, meta.device)?;
            let up_indices = Tensor::from_vec(up_indices, intermediate_size, meta.device)?;
            Ok((
                b.index_select(&gate_indices, 1)?,
                b.index_select(&up_indices, 1)?,
            ))
        }
    }
}

pub(super) fn load_expert_site(
    meta: ExpertSiteMeta<'_>,
    plan: ExpertSiteLoadPlan,
    weights: &ShardedVarBuilder,
) -> Result<LoadedExpertSite> {
    match plan.source {
        ExpertSiteSourcePlan::PerExpert { gate, up, down } => Ok(LoadedExpertSite {
            gate: gate
                .map(|plan| load_per_expert_projection(meta, plan, weights))
                .transpose()?,
            up: up
                .map(|plan| load_per_expert_projection(meta, plan, weights))
                .transpose()?,
            down: down
                .map(|plan| load_per_expert_projection(meta, plan, weights))
                .transpose()?,
        }),
        ExpertSiteSourcePlan::Fused { gate_up, down } => {
            let (gate, up) = if let Some((pair, layout)) = gate_up {
                let a = load_fused_a(
                    meta,
                    &pair,
                    layout,
                    meta.hidden_size,
                    meta.hidden_size,
                    Shard::default(),
                    weights,
                )?;
                let (offset, local_intermediate) =
                    shard_bounds(meta.intermediate_size, meta.gate_up_output_shard, meta.path)?;
                let b_dim = fused_b_output_dim(layout);
                let (gate_b, up_b) = match meta.gate_up_order {
                    GateUpOrder::Concatenated => (
                        load_fused_b(
                            meta,
                            &pair,
                            layout,
                            meta.intermediate_size * 2,
                            local_intermediate,
                            range_shard(
                                b_dim,
                                offset,
                                local_intermediate,
                                meta.intermediate_size * 2,
                            ),
                            weights,
                        )?,
                        load_fused_b(
                            meta,
                            &pair,
                            layout,
                            meta.intermediate_size * 2,
                            local_intermediate,
                            range_shard(
                                b_dim,
                                meta.intermediate_size + offset,
                                local_intermediate,
                                meta.intermediate_size * 2,
                            ),
                            weights,
                        )?,
                    ),
                    GateUpOrder::Interleaved => {
                        let b = load_fused_b(
                            meta,
                            &pair,
                            layout,
                            meta.intermediate_size * 2,
                            local_intermediate * 2,
                            range_shard(
                                b_dim,
                                offset * 2,
                                local_intermediate * 2,
                                meta.intermediate_size * 2,
                            ),
                            weights,
                        )?;
                        split_gate_up_b(meta, &b, local_intermediate)?
                    }
                };
                let scales = projection_scales(meta, pair.scale)?;
                (
                    Some(LoadedExpertProjection {
                        a: a.clone(),
                        b: gate_b.contiguous()?,
                        scales: scales.clone(),
                    }),
                    Some(LoadedExpertProjection {
                        a,
                        b: up_b.contiguous()?,
                        scales,
                    }),
                )
            } else {
                (None, None)
            };
            let down = if let Some((pair, layout)) = down {
                let (offset, local_intermediate) =
                    shard_bounds(meta.intermediate_size, meta.down_input_shard, meta.path)?;
                let a = load_fused_a(
                    meta,
                    &pair,
                    layout,
                    meta.intermediate_size,
                    local_intermediate,
                    range_shard(
                        fused_a_input_dim(layout),
                        offset,
                        local_intermediate,
                        meta.intermediate_size,
                    ),
                    weights,
                )?;
                let b = load_fused_b(
                    meta,
                    &pair,
                    layout,
                    meta.hidden_size,
                    meta.hidden_size,
                    Shard::default(),
                    weights,
                )?;
                Some(LoadedExpertProjection {
                    a: a.contiguous()?,
                    b,
                    scales: projection_scales(meta, pair.scale)?,
                })
            } else {
                None
            };
            Ok(LoadedExpertSite { gate, up, down })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Device;

    use super::*;
    use crate::ShardedSafeTensors;

    fn config(value: serde_json::Value) -> LoraConfig {
        serde_json::from_value(value).unwrap()
    }

    fn meta<'a>(device: &'a Device, gate_up_order: GateUpOrder) -> ExpertSiteMeta<'a> {
        ExpertSiteMeta {
            path: "model.layers.0.mlp.experts",
            num_experts: 2,
            hidden_size: 2,
            intermediate_size: 4,
            gate_name: "gate_proj",
            up_name: "up_proj",
            down_name: "down_proj",
            gate_up_order,
            gate_up_output_shard: Shard::default(),
            down_input_shard: Shard::default(),
            activation_dtype: DType::F32,
            device,
        }
    }

    fn tensor(values: Vec<f32>, shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    #[test]
    fn loads_flat_fused_b_with_expert_major_rank_columns_after_tp_slice() -> Result<()> {
        let device = Device::Cpu;
        let name = "model.layers.0.mlp.experts.lora_B.weight";
        let backend = HashMap::from([(
            name.to_string(),
            tensor((0..24).map(|value| value as f32).collect(), (4, 6))?,
        )]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let pair = TensorPairPlan {
            a_name: String::new(),
            b_name: name.to_string(),
            rank: 3,
            scale: 1.0,
        };
        let loaded = load_fused_b(
            meta(&device, GateUpOrder::Concatenated),
            &pair,
            FusedTensorLayout::Flat,
            4,
            2,
            Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            },
            &weights,
        )?;
        assert_eq!(
            loaded.to_vec3::<f32>()?,
            vec![
                vec![vec![12., 13., 14.], vec![18., 19., 20.]],
                vec![vec![15., 16., 17.], vec![21., 22., 23.]],
            ]
        );
        Ok(())
    }

    #[test]
    fn packs_per_expert_peft_tensors_with_rank_padding() -> Result<()> {
        let device = Device::Cpu;
        let root = "base_model.model.model.layers.0.mlp.experts";
        let backend = HashMap::from([
            (
                format!("{root}.0.gate_proj.lora_A.weight"),
                tensor(vec![1., 2.], (1, 2))?,
            ),
            (
                format!("{root}.0.gate_proj.lora_B.weight"),
                tensor(vec![10., 11., 12., 13.], (4, 1))?,
            ),
            (
                format!("{root}.1.gate_proj.lora_A.default.weight"),
                tensor(vec![3., 4., 5., 6.], (2, 2))?,
            ),
            (
                format!("{root}.1.gate_proj.lora_B.default.weight"),
                tensor(vec![20., 21., 22., 23., 24., 25., 26., 27.], (4, 2))?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let tensors = AdapterTensorIndex::new(weights.tensor_names().unwrap());
        let config = config(serde_json::json!({
            "r": 1,
            "lora_alpha": 2,
            "target_modules": ["gate_proj"],
            "rank_pattern": {"model.layers.0.mlp.experts.1.gate_proj": 2}
        }));
        let plan = plan_expert_site(
            meta(&device, GateUpOrder::Concatenated),
            &config,
            &tensors,
            &weights,
        )?
        .expect("expert plan");
        assert_eq!(plan.bytes(), 104);
        assert_eq!(plan.consumed().len(), 4);

        let loaded = load_expert_site(meta(&device, GateUpOrder::Concatenated), plan, &weights)?;
        let gate = loaded.gate.expect("gate projection");
        assert_eq!(gate.a.dims(), &[2, 2, 2]);
        assert_eq!(gate.b.dims(), &[2, 4, 2]);
        assert_eq!(
            gate.a.to_vec3::<f32>()?,
            vec![
                vec![vec![1., 2.], vec![0., 0.]],
                vec![vec![3., 4.], vec![5., 6.]]
            ]
        );
        assert_eq!(gate.scales.to_vec1::<f32>()?, vec![2., 1.]);
        assert!(loaded.up.is_none());
        assert!(loaded.down.is_none());
        Ok(())
    }

    #[test]
    fn packs_mixtral_w1_w3_w2_names() -> Result<()> {
        let device = Device::Cpu;
        let root = "model.layers.0.mlp.experts";
        let mut backend = HashMap::new();
        for expert in 0..2 {
            for (name, input, output) in [("w1", 2, 4), ("w3", 2, 4), ("w2", 4, 2)] {
                let path = format!("{root}.{expert}.{name}");
                backend.insert(
                    format!("{path}.lora_A.weight"),
                    Tensor::zeros((1, input), DType::F32, &device)?,
                );
                backend.insert(
                    format!("{path}.lora_B.weight"),
                    Tensor::zeros((output, 1), DType::F32, &device)?,
                );
            }
        }
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let tensors = AdapterTensorIndex::new(weights.tensor_names().unwrap());
        let config = config(serde_json::json!({
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["w1", "w3", "w2"]
        }));
        let mut site = meta(&device, GateUpOrder::Concatenated);
        site.gate_name = "w1";
        site.up_name = "w3";
        site.down_name = "w2";
        let plan = plan_expert_site(site, &config, &tensors, &weights)?.expect("expert plan");
        assert_eq!(plan.bytes(), 168);
        assert_eq!(plan.consumed().len(), 12);
        let loaded = load_expert_site(site, plan, &weights)?;
        assert!(loaded.gate.is_some());
        assert!(loaded.up.is_some());
        assert!(loaded.down.is_some());
        Ok(())
    }

    #[test]
    fn loads_flat_target_parameters_and_applies_tp_shards() -> Result<()> {
        let device = Device::Cpu;
        let root = "model.layers.0.mlp.experts";
        let backend = HashMap::from([
            (
                format!("{root}.base_layer.lora_A.weight"),
                tensor(vec![1., 2., 3., 4.], (2, 2))?,
            ),
            (
                format!("{root}.base_layer.lora_B.weight"),
                tensor((0..16).map(|value| value as f32).collect(), (8, 2))?,
            ),
            (
                format!("{root}.lora_A.weight"),
                tensor((0..8).map(|value| value as f32).collect(), (2, 4))?,
            ),
            (
                format!("{root}.lora_B.weight"),
                tensor(vec![30., 31., 40., 41.], (2, 2))?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let tensors = AdapterTensorIndex::new(weights.tensor_names().unwrap());
        let config = config(serde_json::json!({
            "r": 1,
            "lora_alpha": 2,
            "target_parameters": [
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj"
            ]
        }));
        let mut site = meta(&device, GateUpOrder::Concatenated);
        site.gate_up_output_shard = Shard::Simple {
            dim: 1,
            rank: 1,
            world_size: 2,
        };
        site.down_input_shard = Shard::Simple {
            dim: 2,
            rank: 1,
            world_size: 2,
        };
        let plan = plan_expert_site(site, &config, &tensors, &weights)?.expect("expert plan");
        assert_eq!(plan.bytes(), 96);
        let loaded = load_expert_site(site, plan, &weights)?;
        let gate = loaded.gate.expect("gate projection");
        let up = loaded.up.expect("up projection");
        let down = loaded.down.expect("down projection");
        assert_eq!(gate.a.dims(), &[2, 1, 2]);
        assert_eq!(gate.b.dims(), &[2, 2, 1]);
        assert_eq!(up.b.dims(), &[2, 2, 1]);
        assert_eq!(down.a.dims(), &[2, 1, 2]);
        assert_eq!(down.b.dims(), &[2, 2, 1]);
        assert_eq!(
            gate.b.to_vec3::<f32>()?,
            vec![vec![vec![4.], vec![6.]], vec![vec![5.], vec![7.]]]
        );
        assert_eq!(
            up.b.to_vec3::<f32>()?,
            vec![vec![vec![12.], vec![14.]], vec![vec![13.], vec![15.]]]
        );
        assert_eq!(
            down.a.to_vec3::<f32>()?,
            vec![vec![vec![2., 3.]], vec![vec![6., 7.]]]
        );

        let registry = crate::LoraLayerRegistry::new();
        let handle = registry.register_expert(
            crate::LoraSiteKey::new(site.path),
            crate::LoraExpertSiteSpec::new(
                site.num_experts,
                site.hidden_size,
                site.intermediate_size,
                crate::LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                site.gate_up_output_shard,
                site.down_input_shard,
            )?,
            DType::F32,
            device.clone(),
        )?;
        registry.finalize()?;
        let expert_weights = crate::LoraExpertWeights::new(
            &handle,
            Some(crate::LoraExpertProjectionWeights::new_loaded(
                gate.a,
                gate.b,
                gate.scales,
            )?),
            None,
            Some(crate::LoraExpertProjectionWeights::new_loaded(
                down.a,
                down.b,
                down.scales,
            )?),
        )?;
        let mut execution = crate::LoraExecution::new(registry.runtime_id(), vec![Some(1)]);
        execution.insert_expert(&handle, 1, expert_weights)?;
        let topk = Tensor::new(&[[0u32, 1]], &device)?;
        let gate_delta = crate::add_expert_delta_reference(
            &execution,
            &handle,
            crate::LoraExpertDelta::new(
                crate::LoraExpertProjection::Gate,
                &Tensor::new(&[[1f32, 2.]], &device)?,
                Tensor::zeros((1, 2, 2), DType::F32, &device)?,
                &topk,
                crate::LoraExpertInputMode::TokenRows,
            ),
        )?;
        assert_eq!(
            gate_delta.to_vec3::<f32>()?,
            vec![vec![vec![40., 60.], vec![110., 154.]]]
        );
        let down_delta = crate::add_expert_delta_reference(
            &execution,
            &handle,
            crate::LoraExpertDelta::new(
                crate::LoraExpertProjection::Down,
                &Tensor::new(&[[[1f32, 2.], [3., 4.]]], &device)?,
                Tensor::zeros((1, 2, 2), DType::F32, &device)?,
                &topk,
                crate::LoraExpertInputMode::RoutedRows,
            ),
        )?;
        assert_eq!(
            down_delta.to_vec3::<f32>()?,
            vec![vec![vec![480., 640.], vec![2852., 3772.]]]
        );
        Ok(())
    }

    #[test]
    fn loads_packed_interleaved_gate_up_tensors() -> Result<()> {
        let device = Device::Cpu;
        let root = "model.layers.0.mlp.experts";
        let backend = HashMap::from([
            (
                format!("{root}.gate_up_proj.lora_A.weight"),
                tensor(vec![1., 2., 3., 4.], (2, 1, 2))?,
            ),
            (
                format!("{root}.gate_up_proj.lora_B.weight"),
                tensor((0..16).map(|value| value as f32).collect(), (2, 8, 1))?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let tensors = AdapterTensorIndex::new(weights.tensor_names().unwrap());
        let config = config(serde_json::json!({
            "r": 1,
            "lora_alpha": 1,
            "target_parameters": ["mlp.experts.gate_up_proj"]
        }));
        let site = meta(&device, GateUpOrder::Interleaved);
        let plan = plan_expert_site(site, &config, &tensors, &weights)?.expect("expert plan");
        let loaded = load_expert_site(site, plan, &weights)?;
        let gate = loaded.gate.expect("gate projection");
        let up = loaded.up.expect("up projection");
        assert_eq!(
            gate.b.to_vec3::<f32>()?,
            vec![
                vec![vec![0.], vec![2.], vec![4.], vec![6.]],
                vec![vec![8.], vec![10.], vec![12.], vec![14.]]
            ]
        );
        assert_eq!(
            up.b.to_vec3::<f32>()?,
            vec![
                vec![vec![1.], vec![3.], vec![5.], vec![7.]],
                vec![vec![9.], vec![11.], vec![13.], vec![15.]]
            ]
        );
        Ok(())
    }
}
