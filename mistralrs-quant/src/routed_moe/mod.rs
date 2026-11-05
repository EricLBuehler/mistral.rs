use candle_core::{DType, Module, Result, Tensor, D};

use crate::{
    GluActivationType, LoraExpertExecution, LoraExpertInputMode, LoraExpertProjection, QuantMethod,
};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
pub const GROUPED_PREFILL_MIN_TOKENS: usize = 32;

/// Logical expert projections; storage layout and kernel selection belong to the quantization backend.
pub struct RoutedExperts<'a> {
    pub gate: &'a dyn QuantMethod,
    pub up: &'a dyn QuantMethod,
    pub down: &'a dyn QuantMethod,
    pub num_experts: usize,
}

/// Token-major input `[tokens, hidden]`, expert indices and routing weights `[tokens, topk]`.
#[derive(Clone, Copy)]
pub struct RoutedInput<'a> {
    pub xs: &'a Tensor,
    pub indices: &'a Tensor,
    pub routing: &'a Tensor,
    pub output_dtype: DType,
    pub lora: Option<&'a LoraExpertExecution>,
}

#[derive(Clone, Copy)]
pub struct RoutedActivation<'a> {
    pub fused: Option<GluActivationType>,
    pub fallback: &'a dyn Module,
}

impl RoutedActivation<'_> {
    fn apply(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if matches!(gate.dtype(), DType::BF16 | DType::F16 | DType::F32)
            && gate.dtype() == up.dtype()
        {
            if let Some(activation) = self.fused {
                return crate::fused_glu(gate, up, activation);
            }
        }
        self.fallback.forward(gate)? * up
    }

    #[cfg(feature = "cuda")]
    fn split(&self, packed: &Tensor, split: usize) -> Result<Tensor> {
        if matches!(packed.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            if let Some(activation) = self.fused {
                return crate::fused_split_glu(packed, split, activation);
            }
        }
        self.apply(
            &packed.narrow(D::Minus1, 0, split)?,
            &packed.narrow(D::Minus1, split, split)?,
        )
    }
}

struct Shape {
    num_tokens: usize,
    hidden_dim: usize,
}

struct Execution<'a> {
    xs_flat: &'a Tensor,
    topk_ids: &'a Tensor,
    topk_weights: &'a Tensor,
    original_dtype: DType,
    shape: Shape,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    topk: usize,
    lora: Option<&'a LoraExpertExecution>,
    activation: RoutedActivation<'a>,
}

impl<'a> Execution<'a> {
    fn new(input: RoutedInput<'a>, activation: RoutedActivation<'a>) -> Result<Self> {
        let (num_tokens, hidden_dim) = input.xs.dims2()?;
        let (index_tokens, topk) = input.indices.dims2()?;
        if index_tokens != num_tokens || input.routing.dims() != [num_tokens, topk] {
            candle_core::bail!("routed FFN inputs must share the token and expert dimensions");
        }
        Ok(Self {
            xs_flat: input.xs,
            topk_ids: input.indices,
            topk_weights: input.routing,
            original_dtype: input.output_dtype,
            shape: Shape {
                num_tokens,
                hidden_dim,
            },
            #[cfg(any(feature = "cuda", feature = "metal"))]
            topk,
            lora: input.lora,
            activation,
        })
    }
}

impl RoutedExperts<'_> {
    fn members(&self) -> [&dyn QuantMethod; 3] {
        [self.gate, self.up, self.down]
    }

    pub fn forward(
        &self,
        input: RoutedInput<'_>,
        activation: RoutedActivation<'_>,
    ) -> Result<Tensor> {
        let forward = Execution::new(input, activation)?;
        if crate::group_allowed(forward.xs_flat, &self.members()) {
            if forward.lora.is_some() {
                #[cfg(feature = "cuda")]
                match self.select_cuda_fast_path(&forward) {
                    Some(cuda::CudaRoute::Grouped) if self.grouped_lora_preserves_route_order() => {
                        if let Some(result) = self.forward_grouped(&forward)? {
                            return Ok(result);
                        }
                    }
                    Some(_) => {
                        if let Some(result) = self.forward_decode_lora(&forward)? {
                            return Ok(result);
                        }
                    }
                    None => {}
                }
            } else {
                #[cfg(feature = "cuda")]
                if let Some(result) = self.forward_cuda(&forward)? {
                    return Ok(result);
                }
                #[cfg(feature = "metal")]
                if let Some(result) = self.forward_metal_sorted(&forward)? {
                    return Ok(result);
                }
            }
        }
        self.gather_impl(&forward)
    }

    /// Execute the same operation through individual projections, including stats and adapter updates.
    pub fn gather(
        &self,
        input: RoutedInput<'_>,
        activation: RoutedActivation<'_>,
    ) -> Result<Tensor> {
        self.gather_impl(&Execution::new(input, activation)?)
    }

    fn gather_impl(&self, forward: &Execution<'_>) -> Result<Tensor> {
        let ids = forward.topk_ids;
        let tokens = forward.shape.num_tokens;
        let hidden = forward.shape.hidden_dim;
        self.gate.process_routed_stats(forward.xs_flat, ids)?;
        self.up.process_routed_stats(forward.xs_flat, ids)?;
        let xs = forward.xs_flat.reshape((tokens, 1, hidden))?;
        let grouped = if forward.lora.is_none() {
            crate::forward_group(
                &xs,
                &[self.gate, self.up],
                crate::GroupKind::Gather { indices: ids },
            )?
        } else {
            None
        };
        let (gate, up) = if let Some(outputs) = grouped {
            let mut outputs = outputs.into_iter();
            (outputs.next().unwrap(), outputs.next().unwrap())
        } else {
            (
                self.gate.gather_forward(&xs, ids)?,
                self.up.gather_forward(&xs, ids)?,
            )
        };
        let (gate, up) = if let Some(lora) = forward.lora {
            let dtype = forward.xs_flat.dtype();
            let gate_up = Tensor::cat(&[gate.to_dtype(dtype)?, up.to_dtype(dtype)?], D::Minus1)?;
            lora.add_gate_up_delta_owned(forward.xs_flat, gate_up, ids)?
        } else {
            (gate, up)
        };
        let down_input = forward.activation.apply(&gate, &up)?;
        self.down.process_routed_stats(&down_input, ids)?;
        let down = self.down.gather_forward(&down_input, ids)?;
        let down = if let Some(lora) = forward.lora {
            lora.add_delta_owned(
                LoraExpertProjection::Down,
                &down_input,
                down.to_dtype(down_input.dtype())?,
                ids,
                None,
                LoraExpertInputMode::RoutedRows,
            )?
        } else {
            down
        };
        topk_weight_reduce(&down, forward.topk_weights, forward.original_dtype)
    }
}

fn topk_weight_reduce(ys: &Tensor, topk_weights: &Tensor, dtype: DType) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if ys.device().is_cuda()
        && matches!(ys.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
    {
        return crate::moe::moe_weighted_reduce(ys, topk_weights, dtype);
    }
    #[cfg(feature = "metal")]
    if ys.device().is_metal() && matches!(ys.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        let (tokens, topk, hidden) = ys.dims3()?;
        return crate::metal_moe_weighted_reduce_flat(
            &ys.reshape((tokens * topk, hidden))?,
            topk_weights,
            tokens,
            topk,
        )?
        .to_dtype(dtype);
    }
    ys.to_dtype(DType::F32)?
        .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
        .sum(D::Minus2)?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests;
