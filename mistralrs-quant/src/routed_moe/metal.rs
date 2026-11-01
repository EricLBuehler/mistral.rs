use candle_core::{DType, Result, Tensor};

use super::{topk_weight_reduce, Execution, RoutedExperts};
use crate::GluActivationType;

const METAL_SORTED_MIN_ROUTED_ROWS: usize = 64;

impl RoutedExperts<'_> {
    pub(super) fn forward_metal_sorted(&self, forward: &Execution) -> Result<Option<Tensor>> {
        use crate::{
            afq_gather_qmm_rhs_sorted, afq_gather_qmm_rhs_sorted_gate_up,
            afq_sorted_gather_supported, metal_arg_sort_u32_1d,
        };

        if self.members().iter().any(|member| member.has_bias()) {
            return Ok(None);
        }
        let top_k = forward.topk;
        let rows = forward.shape.num_tokens * top_k;
        if !forward.xs_flat.device().is_metal() || rows < METAL_SORTED_MIN_ROUTED_ROWS {
            return Ok(None);
        }
        let (Some(gate), Some(up), Some(down)) = (
            self.gate.afq_inner(),
            self.up.afq_inner(),
            self.down.afq_inner(),
        ) else {
            return Ok(None);
        };
        if [&gate, &up, &down]
            .iter()
            .any(|layer| !afq_sorted_gather_supported(layer.group_size, layer.bits))
            || gate.bits != up.bits
            || gate.group_size != up.group_size
        {
            return Ok(None);
        }
        // Matches fused_glu_activation in quantized.metal.
        let act_idx = match forward.activation.fused {
            Some(GluActivationType::Silu) => 0,
            Some(GluActivationType::Gelu) => 1,
            Some(GluActivationType::GeluErf) => 2,
            Some(GluActivationType::Relu) => 3,
            _ => return Ok(None),
        };

        let ids = forward
            .topk_ids
            .reshape((rows,))?
            .to_dtype(DType::U32)?
            .contiguous()?;
        let perm = metal_arg_sort_u32_1d(&ids)?;
        let sorted_expert_ids = ids.gather(&perm, 0)?;
        let sorted_token_idx = perm
            .to_dtype(DType::F32)?
            .affine(
                1.0 / f64::from(u32::try_from(top_k).map_err(candle_core::Error::wrap)?),
                0.0,
            )?
            .floor()?
            .to_dtype(DType::U32)?;
        let x_sorted = forward.xs_flat.index_select(&sorted_token_idx, 0)?;
        let mid = afq_gather_qmm_rhs_sorted_gate_up(
            &x_sorted,
            &gate.w_q,
            &gate.scales,
            &gate.biases,
            &up.w_q,
            &up.scales,
            &up.biases,
            &sorted_expert_ids,
            gate.group_size,
            gate.bits,
            act_idx,
        )?;
        let down_sorted = afq_gather_qmm_rhs_sorted(
            &mid,
            &down.w_q,
            &down.scales,
            &down.biases,
            &sorted_expert_ids,
            down.group_size,
            down.bits,
        )?;
        let inv_perm = metal_arg_sort_u32_1d(&perm)?;
        let ys = down_sorted.index_select(&inv_perm, 0)?.reshape((
            forward.shape.num_tokens,
            top_k,
            forward.shape.hidden_dim,
        ))?;
        topk_weight_reduce(&ys, forward.topk_weights, forward.original_dtype).map(Some)
    }
}
