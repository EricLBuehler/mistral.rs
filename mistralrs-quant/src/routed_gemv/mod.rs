//! Shared MoE decode GEMV: formats supply a codec, one kernel template supplies fused GLU, top-k reduce and prefetch.

use candle_core::{Result, Tensor};

use crate::{GluActivationType, QuantMethod};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(test, feature = "cuda"))]
mod tests;

/// How a routed expert stack is packed; each variant has one CUDA codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedCodec {
    /// E2M1 `[E, N, K/2]`, E4M3 per-16 scales, f32 `[E, N]` global multipliers.
    Nvfp4,
    /// E2M1 `[E, N, K/2]`, E8M0 per-32 scales.
    Mxfp4,
    /// E4M3 `[E, N, K]`, E8M0 per-32 scales.
    Mxfp8,
    /// E4M3 `[E, N, K]` with f32 scales per `[block_rows, block_cols]` (`block_cols >= K` is one scale per row).
    Fp8 {
        block_rows: usize,
        block_cols: usize,
    },
    /// Affine `bits`-bit bitstream with activation-dtype scales and biases per `group` columns.
    Afq { bits: usize, group: usize },
}

/// A packed expert stack as one routed GEMV codec sees it; all tensors stay on the weights' device.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct RoutedWeights {
    pub(crate) codec: RoutedCodec,
    pub(crate) data: Tensor,
    pub(crate) scales: Tensor,
    pub(crate) extra: Option<Tensor>,
    /// `[E, N]` per-expert or `[N]` shared, in the activation dtype.
    pub(crate) bias: Option<Tensor>,
    pub(crate) experts: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl RoutedWeights {
    pub(crate) fn new(
        codec: RoutedCodec,
        data: Tensor,
        scales: Tensor,
        extra: Option<Tensor>,
        bias: Option<Tensor>,
        [experts, n, k]: [usize; 3],
    ) -> Self {
        Self {
            codec,
            data,
            scales,
            extra,
            bias,
            experts,
            n,
            k,
        }
    }

    fn shape(&self) -> [usize; 3] {
        [self.experts, self.n, self.k]
    }
}

/// `x [T, 1|topk, K]` against expert stacks routed by `indices [T, topk]`; `None` when no routed codec fits.
pub fn try_routed_gather(
    x: &Tensor,
    indices: &Tensor,
    weights: &RoutedWeights,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && cuda::enabled() {
        return cuda::gather(x, indices, weights);
    }
    let _ = (x, indices, weights);
    Ok(None)
}

/// `act(gate(x)) * up(x)` per route, `[T, topk, N]`, from one fused pass over both expert stacks.
pub fn try_routed_glu(
    x: &Tensor,
    indices: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    let (Some(gate), Some(up)) = (gate.routed_weights(), up.routed_weights()) else {
        return Ok(None);
    };
    if gate.codec != up.codec || gate.shape() != up.shape() {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && cuda::enabled() {
        return cuda::glu(x, indices, &gate, &up, activation);
    }
    let _ = (x, indices, activation);
    Ok(None)
}

/// Routed MoE FFN `sum_k routing[t, k] * down(act(gate(x)) * up(x))`, `[T, hidden]`, for `x [T, hidden]`.
pub fn try_routed_ffn(
    x: &Tensor,
    indices: &Tensor,
    routing: &Tensor,
    [gate, up, down]: [&dyn QuantMethod; 3],
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    if !crate::group_allowed(x, &[gate, up, down]) {
        return Ok(None);
    }
    let Some(down) = down.routed_weights() else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && cuda::enabled() {
        let Some(intermediate) = try_routed_glu(x, indices, gate, up, activation)? else {
            return Ok(None);
        };
        return cuda::reduce(&intermediate, indices, routing, &down);
    }
    let _ = (indices, routing, activation, down);
    Ok(None)
}

/// Reference gather over dense `[E, N, K]` weights for formats without a routed codec at this shape.
pub(crate) fn dense_gather(
    weight: &Tensor,
    bias: Option<&Tensor>,
    x: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    let (tokens, topk) = indices.dims2()?;
    let (_, n, k) = weight.dims3()?;
    let ids = indices.flatten_all()?;
    let selected = weight.to_dtype(x.dtype())?.index_select(&ids, 0)?;
    let x = x
        .reshape((tokens, (), k))?
        .broadcast_as((tokens, topk, k))?
        .contiguous()?
        .reshape((tokens * topk, 1, k))?;
    let output = x
        .matmul(&selected.transpose(1, 2)?)?
        .reshape((tokens, topk, n))?;
    add_routed_bias(output, bias, indices)
}

pub(crate) fn add_routed_bias(
    output: Tensor,
    bias: Option<&Tensor>,
    indices: &Tensor,
) -> Result<Tensor> {
    let Some(bias) = bias else {
        return Ok(output);
    };
    let bias = bias.to_dtype(output.dtype())?;
    let bias = if bias.rank() == 2 {
        bias.index_select(&indices.flatten_all()?, 0)?
            .reshape(output.shape())?
    } else {
        bias
    };
    output.broadcast_add(&bias)
}
