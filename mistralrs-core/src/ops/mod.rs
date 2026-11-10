use std::sync::Arc;

use candle_core::{shape::Dim, DType, Result, Tensor, D};

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if input.device().is_cpu() && matches!(input.dtype(), DType::F16 | DType::BF16) {
        mistralrs_quant::cpu_rms_norm(input, weight, eps)
    } else {
        candle_nn::ops::rms_norm(&input.contiguous()?, weight, eps)
    }
}

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
use crate::layers::Activation;
#[cfg(feature = "cuda")]
use candle_core::Shape;

mod moe_router;
mod norm_rope;
mod sampling;
mod sort;

pub(crate) use moe_router::*;
pub(crate) use norm_rope::*;
pub(crate) use sampling::*;
pub(crate) use sort::*;

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Use optimized parallel topk kernel on CUDA
        // Single kernel call, no post-processing overhead
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_topk(self, topk);
        }

        // Fallback: full sort (CPU or non-CUDA)
        let (values, sorted_indices) = self.sort_last_dim(false)?;

        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let TopKOutput { values, indices } = self.topk(topk)?;
        // Reorder the indices ascending
        let reorder_indices = if indices.device().is_cuda() {
            indices.arg_sort(true)?
        } else {
            indices.arg_sort_last_dim(true)?
        };
        let topk_indices_unsorted = indices
            .to_dtype(DType::F32)?
            .gather(&reorder_indices, D::Minus1)?
            .to_dtype(DType::U32)?;
        let topk_values_unsorted = values.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait RepeatInterleaveOp {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor>;
    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor>;
}

impl RepeatInterleaveOp for Tensor {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor> {
        let dim = dim.to_index(self.shape(), "repeat_interleave")?;
        let dim_elements = self.dim(dim)?;
        // For metal
        assert!(self.dtype().is_float());
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..dim_elements)
                .flat_map(|i| vec![i as u32; repeats])
                .collect::<Vec<_>>(),
            self.device(),
        )?;
        self.index_select(&indices, dim)
    }

    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor> {
        let xs = self.flatten_all()?;
        if repeats.len() != xs.dim(0)? {
            candle_core::bail!(
                "repeats ({}) must match flattened self length ({})",
                repeats.len(),
                xs.dim(0)?
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..xs.dim(0)?)
                .flat_map(|i| vec![i as u32; repeats[i] as usize])
                .collect::<Vec<_>>(),
            xs.device(),
        )?;
        xs.index_select(&indices, 0)
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }
}

// https://github.com/mokeyish/candle-ext/blob/ca4547c803469bd51c00ce5eda2f18dd249c8f10/src/triangular.rs#L21
pub fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

/// Elementwise multiply and activation. The following activations are supported:
/// - `gelu`
/// - `silu`
/// - `relu`
///
/// This is equivalent to:
/// `act(a) * b`
///
/// With supported dtypes (F16, BF16, F32) and fused activations,
/// this uses a fused kernel for better performance by eliminating intermediate
/// memory allocation. Optimized implementations are available for:
/// - CUDA: Custom CUDA kernel with vec4 optimization
/// - Metal: Native Metal kernel
/// - CPU: Rayon-parallelized implementation
pub(crate) fn glu_activation_type(act: Activation) -> Option<mistralrs_quant::GluActivationType> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::GluActivationType::Silu),
        Activation::NewGelu | Activation::GeluPytorchTanh => {
            Some(mistralrs_quant::GluActivationType::Gelu)
        }
        Activation::Gelu => Some(mistralrs_quant::GluActivationType::GeluErf),
        Activation::Relu => Some(mistralrs_quant::GluActivationType::Relu),
        Activation::Sigmoid => Some(mistralrs_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

fn candle_glu_activation_type(
    act: candle_nn::Activation,
) -> Option<mistralrs_quant::GluActivationType> {
    match act {
        candle_nn::Activation::Silu | candle_nn::Activation::Swish => {
            Some(mistralrs_quant::GluActivationType::Silu)
        }
        candle_nn::Activation::NewGelu | candle_nn::Activation::GeluPytorchTanh => {
            Some(mistralrs_quant::GluActivationType::Gelu)
        }
        candle_nn::Activation::Gelu => Some(mistralrs_quant::GluActivationType::GeluErf),
        candle_nn::Activation::Relu => Some(mistralrs_quant::GluActivationType::Relu),
        candle_nn::Activation::Sigmoid => Some(mistralrs_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedActivationOrder {
    GateUp,
    UpGate,
}

pub fn mul_and_act(a: &Tensor, b: &Tensor, act: Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = glu_activation_type(act) {
            return mistralrs_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub(crate) fn try_fused_gated_projection(
    gate: &Tensor,
    value: &Tensor,
    act: Activation,
    projection: &dyn mistralrs_quant::QuantMethod,
) -> Result<Option<Tensor>> {
    let Some(activation) = glu_activation_type(act) else {
        return Ok(None);
    };
    mistralrs_quant::forward_fused_glu(
        projection,
        mistralrs_quant::GluInput::Split { gate, value },
        activation,
    )
}

pub fn mul_and_candle_act(a: &Tensor, b: &Tensor, act: candle_nn::Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = candle_glu_activation_type(act) {
            return mistralrs_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub fn split_mul_and_act(xs: &Tensor, split_size: usize, act: Activation) -> Result<Tensor> {
    split_mul_and_act_order(xs, split_size, act, GatedActivationOrder::GateUp)
}

pub(crate) fn try_fused_split_glu_quantized_forward(
    xs: &Tensor,
    split_size: usize,
    act: Activation,
    projection: &dyn mistralrs_quant::QuantMethod,
) -> Result<Option<Tensor>> {
    let Some(activation) = glu_activation_type(act) else {
        return Ok(None);
    };
    mistralrs_quant::forward_fused_glu(
        projection,
        mistralrs_quant::GluInput::Packed {
            input: xs,
            split_size,
        },
        activation,
    )
}

pub fn split_mul_and_act_order(
    xs: &Tensor,
    split_size: usize,
    act: Activation,
    order: GatedActivationOrder,
) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let Some(expected_last_dim) = split_size.checked_mul(2) else {
        candle_core::bail!("split_mul_and_act split size overflow: {split_size}");
    };
    if last_dim != expected_last_dim {
        candle_core::bail!(
            "split_mul_and_act expected last dim {expected_last_dim}, got {last_dim}"
        );
    }
    if order == GatedActivationOrder::GateUp
        && matches!(xs.dtype(), DType::F16 | DType::BF16 | DType::F32)
    {
        if let Some(activation_type) = glu_activation_type(act) {
            return mistralrs_quant::fused_split_glu(xs, split_size, activation_type);
        }
    }

    let first = xs.narrow(D::Minus1, 0, split_size)?;
    let second = xs.narrow(D::Minus1, split_size, split_size)?;
    match order {
        GatedActivationOrder::GateUp => mul_and_act(&first, &second, act),
        GatedActivationOrder::UpGate => mul_and_act(&second, &first, act),
    }
}

#[derive(Clone)]
pub(crate) struct MergedDenseProjection {
    proj: Arc<dyn mistralrs_quant::QuantMethod>,
    originals: Vec<Arc<dyn mistralrs_quant::QuantMethod>>,
    output_dims: Vec<usize>,
}

impl MergedDenseProjection {
    /// Wrap a packed projection group: `packed` owns the fused weight, `constituents` are its
    /// view-backed layers used for the dynamic-LoRA fallback path.
    pub(crate) fn from_packed(group: &mistralrs_quant::PackedColumnParallel) -> Self {
        Self {
            proj: group.packed.clone(),
            originals: group.constituents.clone(),
            output_dims: group.rows_per_rank.clone(),
        }
    }

    fn originals_require_forward(&self) -> bool {
        self.originals.iter().any(|projection| {
            projection.is_dynamic_lora_active() || projection.stats_snapshot().is_some()
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let Some(ys) = self.forward_packed(xs)? else {
            return self.originals.iter().map(|proj| proj.forward(xs)).collect();
        };
        let mut parts = Vec::with_capacity(self.output_dims.len());
        let mut offset = 0;
        for &dim in &self.output_dims {
            parts.push(ys.narrow(D::Minus1, offset, dim)?);
            offset += dim;
        }
        Ok(parts)
    }

    pub(crate) fn forward_packed(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        if self.originals_require_forward() {
            Ok(None)
        } else {
            self.proj.forward(xs).map(Some)
        }
    }

    pub(crate) fn activation_quantization_scheme_for(
        &self,
        xs: &Tensor,
    ) -> Option<mistralrs_quant::ActivationQuantizationScheme> {
        if self.originals_require_forward() {
            None
        } else {
            self.proj.activation_quantization_scheme_for(xs)
        }
    }

    pub(crate) fn preferred_activation_scale_layout_for(
        &self,
        xs: &Tensor,
    ) -> Option<mistralrs_quant::ActivationScaleLayout> {
        if self.originals_require_forward() {
            None
        } else {
            self.proj.preferred_activation_scale_layout_for(xs)
        }
    }

    pub(crate) fn forward_quantized_packed(
        &self,
        activation: &mistralrs_quant::QuantizedActivation,
    ) -> Result<Tensor> {
        if self.originals_require_forward() {
            candle_core::bail!(
                "packed projection cannot use prequantized activation with active adapters or stats collection"
            )
        }
        self.proj.forward_quantized(activation)
    }

    pub(crate) fn forward_quantized(
        &self,
        activation: &mistralrs_quant::QuantizedActivation,
    ) -> Result<Vec<Tensor>> {
        let ys = self.forward_quantized_packed(activation)?;
        let mut parts = Vec::with_capacity(self.output_dims.len());
        let mut offset = 0;
        for &dim in &self.output_dims {
            parts.push(ys.narrow(D::Minus1, offset, dim)?);
            offset += dim;
        }
        Ok(parts)
    }
}

/// Feed-forward path for quantized gate/up/down projections.
pub(crate) fn quantized_ffn(
    xs: &Tensor,
    gate: &dyn mistralrs_quant::QuantMethod,
    up: &dyn mistralrs_quant::QuantMethod,
    down: &dyn mistralrs_quant::QuantMethod,
    act: Activation,
) -> Result<Tensor> {
    if let Some(activation_type) = glu_activation_type(act) {
        if let Some(out) = mistralrs_quant::try_fused_ffn(xs, gate, up, down, activation_type)? {
            return Ok(out);
        }
        if let Some(inter) = mistralrs_quant::try_fused_gate_up(xs, gate, up, activation_type)? {
            return down.forward(&inter);
        }
    }

    if xs.device().is_cpu() {
        if let Some(mut out) =
            mistralrs_quant::forward_group(xs, &[gate, up], mistralrs_quant::GroupKind::Dense)?
        {
            let rhs = out.pop().unwrap();
            let lhs = out.pop().unwrap();
            let inter = mul_and_act(&lhs, &rhs, act)?;
            return down.forward(&inter);
        }
    }

    let lhs = gate.forward(xs)?;
    let rhs = up.forward(xs)?;
    if let Some(output) = try_fused_gated_projection(&lhs, &rhs, act, down)? {
        return Ok(output);
    }
    let inter = mul_and_act(&lhs, &rhs, act)?;
    down.forward(&inter)
}

pub(crate) fn qkv_projections(
    xs: &Tensor,
    q_proj: &dyn mistralrs_quant::QuantMethod,
    k_proj: &dyn mistralrs_quant::QuantMethod,
    v_proj: &dyn mistralrs_quant::QuantMethod,
) -> Result<(Tensor, Tensor, Tensor)> {
    if let Some(qkv) = mistralrs_quant::try_fused_qkv(xs, q_proj, k_proj, v_proj)? {
        return Ok(qkv);
    }

    Ok((
        q_proj.forward(xs)?,
        k_proj.forward(xs)?,
        v_proj.forward(xs)?,
    ))
}

#[cfg(test)]
mod tests;
